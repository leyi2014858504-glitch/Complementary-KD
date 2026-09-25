"""
Extended tabular pipeline for the PRLetters major revision.

Reuses the exact experimental protocol of `distill_ablation.py` (same seeded
splits, same alpha/beta construction, same teacher/student settings) and adds
the metrics requested by the reviewers:

  - test accuracy / test loss (NLL) at the best-val checkpoint and final epoch
  - calibration: ECE (15 bins)
  - teacher metrics: test acc, ECE, confidence, predictive entropy,
    soft-label entropy on D_S, per-class test accuracy
  - teacher-student agreement on the overlap vs non-overlap regions of D_S
  - class-wise test accuracy for all student groups

Protocol fix (documented in the revision response): the student's output
dimension is now the GLOBAL number of classes of the dataset, and teacher
probabilities are mapped onto global class indices via `teacher.classes_`
(the original pipeline relied on per-seed class unions and could silently
misalign class columns when a rare class was missing from D_S).

Outputs (results_ext/):
  alpha_results.csv, beta_results.csv, alpha_curves.pkl, beta_curves.pkl,
  progress.log

Usage:
  python tabular_ext.py --mode alpha
  python tabular_ext.py --mode beta
  python tabular_ext.py                # both
  python tabular_ext.py --smoke        # 1 seed, alphas {0,1}, pipeline check
Resumable: completed keys are skipped; files flushed after every run.
"""
import argparse
import copy
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

from distill_ablation import (
    CONFIG, MLP, _train_test_split_deterministic, _val_pool_split,
    build_D_T, train_teacher, _load_uci,
)
from image_distill import ece_score, entropy_stats, classwise_accuracy

OUT_DIR = 'results_ext'
BASE_COLS = ('e_star', 'best_val_acc', 'auc_val', 'final_gen_gap',
             'train_acc_final', 'val_acc_final', 'n_epochs',
             'test_acc_best', 'test_acc_final', 'test_loss_best',
             'test_loss_final', 'ece_best', 'ece_final',
             'entropy_mean_best', 'conf_mean_best')


# ======================== TEACHER PROBABILITY ALIGNMENT ========================
def align_teacher_probs(teacher, probs, n_classes):
    """Map teacher probabilities onto global class indices [0, n_classes)."""
    aligned = np.full((probs.shape[0], n_classes), 1e-8, dtype=np.float64)
    for j, c in enumerate(np.asarray(teacher.classes_).astype(int)):
        if 0 <= c < n_classes:
            aligned[:, c] = probs[:, j]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


# ======================== METRICS ========================
def probs_test_metrics(probs, y_test, n_classes):
    acc = float((probs.argmax(1) == y_test).mean())
    loss = float(-np.log(np.clip(probs[np.arange(len(y_test)), y_test],
                                 1e-12, 1.0)).mean())
    ent, conf = entropy_stats(probs)
    return {
        'test_acc': acc, 'test_loss': loss, 'ece': ece_score(probs, y_test),
        'entropy_mean': ent, 'conf_mean': conf,
        'classwise_acc': classwise_accuracy(probs, y_test, n_classes),
    }


def _model_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        return torch.softmax(logits, dim=1).numpy()


# ======================== STUDENT (extended copy of train_student) ========================
def train_student_ext(X_train, y_train, X_val, y_val, X_test, y_test, n_classes,
                      soft_labels=None, lambda_mix=None, seed=0,
                      label_smoothing=0.0, temperature=1.0,
                      return_train_probs=False):
    """
    Identical training logic to distill_ablation.train_student (same seeding,
    optimizer, early stopping, e*/checkpoint semantics), with test-set and
    calibration evaluation added.  `soft_labels` must already be aligned to
    `n_classes` global class indices.

    temperature: standard KD temperature.  Teacher probabilities q are
    re-tempered as q^(1/T) (renormalized), student logits are divided by T,
    and the KL term is scaled by T^2.  T=1 reproduces the main pipeline.
    """
    if lambda_mix is None:
        lambda_mix = CONFIG['lambda_mix']

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    use_distill = soft_labels is not None
    n_features = X_train.shape[1]

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    if use_distill:
        soft = np.asarray(soft_labels, dtype=np.float64)
        if temperature != 1.0:
            soft = np.power(np.clip(soft, 1e-12, 1.0), 1.0 / temperature)
            soft = soft / soft.sum(axis=1, keepdims=True)
        soft_t = torch.FloatTensor(soft)
        train_ds = TensorDataset(X_train_t, y_train_t, soft_t)
    else:
        train_ds = TensorDataset(X_train_t, y_train_t,
                                 torch.zeros(len(X_train_t), n_classes))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)

    model = MLP(n_features, CONFIG['student_hidden'][0],
                CONFIG['student_hidden'][1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(CONFIG['max_epochs']):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch_X, batch_y = batch[0].to(device), batch[1].to(device)
            batch_soft = batch[2].to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            if use_distill:
                hard_loss = F.cross_entropy(logits, batch_y)
                soft_loss = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                                     batch_soft, reduction='batchmean') \
                    * (temperature ** 2)
                loss = lambda_mix * soft_loss + (1 - lambda_mix) * hard_loss
            else:
                loss = F.cross_entropy(logits, batch_y,
                                       label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_X.size(0)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = F.cross_entropy(val_logits, y_val_t.to(device)).item()
            val_acc = (val_logits.argmax(dim=1) == y_val_t.to(device)).float().mean().item()

        history['train_loss'].append(total_loss / total)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(correct / total)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            break

    history['gen_gap_curve'] = [history['train_acc'][i] - history['val_acc'][i]
                                for i in range(len(history['train_acc']))]

    # test metrics with the FINAL-epoch model (before loading the best state)
    m_final = probs_test_metrics(_model_probs(model, X_test), y_test, n_classes)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_val_acc = (model(X_val_t.to(device)).argmax(dim=1)
                        == y_val_t.to(device)).float().mean().item()
    probs_best = _model_probs(model, X_test)
    m_best = probs_test_metrics(probs_best, y_test, n_classes)

    out = {
        'gen_gap_curve': history['gen_gap_curve'],
        'val_acc': history['val_acc'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'e_star': int(np.argmin(history['val_loss'])),
        'best_val_acc': float(best_val_acc),
        'auc_val': float(np.trapezoid(history['val_acc'])),
        'final_gen_gap': history['gen_gap_curve'][-1],
        'train_acc_final': history['train_acc'][-1],
        'val_acc_final': history['val_acc'][-1],
        'n_epochs': len(history['val_acc']),
        'test_acc_best': m_best['test_acc'],
        'test_acc_final': m_final['test_acc'],
        'test_loss_best': m_best['test_loss'],
        'test_loss_final': m_final['test_loss'],
        'ece_best': m_best['ece'],
        'ece_final': m_final['ece'],
        'entropy_mean_best': m_best['entropy_mean'],
        'conf_mean_best': m_best['conf_mean'],
        'classwise_acc_best': m_best['classwise_acc'],
    }
    if return_train_probs:
        out['train_probs_best'] = _model_probs(model, X_train)
    return out


# ======================== SHARED DRIVER HELPERS ========================
def _row(ds, sweep, sweep_val, group, seed, rec):
    row = {'dataset': ds, sweep: sweep_val, 'group': group, 'seed': seed}
    for k in BASE_COLS:
        row[k] = rec.get(k)
    t = rec.get('teacher', {})
    row.update({
        'teacher_test_acc': t.get('test_acc'),
        'teacher_ece': t.get('ece'),
        'teacher_conf_mean': t.get('conf_mean'),
        'teacher_entropy_mean': t.get('entropy_mean'),
        'teacher_soft_entropy': t.get('soft_entropy'),
        'teacher_soft_conf': t.get('soft_conf'),
        'agreement_overlap': rec.get('agreement_overlap'),
        'agreement_nonoverlap': rec.get('agreement_nonoverlap'),
    })
    return row


def _save(out_dir, store, pkl_name, csv_name, sweep, sweep_values, is_alpha):
    with open(os.path.join(out_dir, pkl_name), 'wb') as f:
        pickle.dump(store, f)
    rows = []
    for key, rec in store.items():
        if is_alpha:
            if len(key) == 3:            # (ds, group, seed) — full-alpha groups
                ds, group, seed = key
                for alpha in sweep_values:
                    rows.append(_row(ds, 'alpha', alpha, group, seed, rec))
            else:                        # (ds, 'Distill-S', alpha, seed)
                ds, group, alpha, seed = key
                rows.append(_row(ds, 'alpha', alpha, group, seed, rec))
        else:
            ds, group = key[0], key[2]
            if len(key) == 4:            # (ds, 'beta', group, seed)
                seed = key[3]
                for beta in sweep_values:
                    rows.append(_row(ds, 'beta', beta, group, seed, rec))
            else:                        # (ds, 'beta', group, beta, seed)
                beta, seed = key[3], key[4]
                rows.append(_row(ds, 'beta', beta, group, seed, rec))
    df = pd.DataFrame(rows).sort_values(
        ['dataset', 'group', 'seed', sweep]).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, csv_name), index=False)


def _log(out_dir, msg):
    with open(os.path.join(out_dir, 'progress.log'), 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def _run_teacher_ext(teacher, X_S, X_test, y_test, n_classes):
    soft_S = align_teacher_probs(teacher, teacher.predict_proba(X_S), n_classes)
    t_probs = align_teacher_probs(teacher, teacher.predict_proba(X_test), n_classes)
    tm = probs_test_metrics(t_probs, y_test, n_classes)
    soft_ent, soft_conf = entropy_stats(soft_S)
    tm['soft_entropy'] = soft_ent
    tm['soft_conf'] = soft_conf
    return soft_S, tm


def _agreement(rec, soft_S, overlap_size):
    teacher_pred = soft_S.argmax(1)
    student_pred = rec['train_probs_best'].argmax(1)
    agree = teacher_pred == student_pred
    ov = float(agree[:overlap_size].mean()) if overlap_size > 0 else float('nan')
    nov = float(agree[overlap_size:].mean()) if overlap_size < len(agree) else float('nan')
    rec.pop('train_probs_best', None)
    rec['agreement_overlap'] = ov
    rec['agreement_nonoverlap'] = nov
    rec['overlap_size'] = int(overlap_size)
    return ov, nov


# ======================== ALPHA SWEEP ========================
def run_alpha_sweep_ext(ds_name, X, y, out_dir, seeds, alphas):
    pkl_path = os.path.join(out_dir, 'alpha_curves.pkl')
    store = pickle.load(open(pkl_path, 'rb')) if os.path.exists(pkl_path) else {}
    n_classes = len(np.unique(y))

    for seed in seeds:
        X_train_all_raw, y_train_all_raw, X_test_raw, y_test_raw = \
            _train_test_split_deterministic(X, y, seed)
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_all_raw)
        X_test = scaler.transform(X_test_raw)
        X_val, y_val, X_pool, y_pool = _val_pool_split(X_train_all, y_train_all_raw, seed)

        n_pool = len(X_pool)
        half = n_pool // 2
        X_S = X_pool[half:].copy()
        y_S = y_pool[half:].copy()

        base_specs = [
            ('Baseline-Half', X_S, y_S, {}),
            ('Baseline-LS', X_S, y_S, {'label_smoothing': CONFIG['ls_smoothing']}),
            ('Baseline-Full', X_pool, y_pool, {}),
        ]
        for group, Xg, yg, extra in base_specs:
            key = (ds_name, group, seed)
            if key in store and store[key].get('complete'):
                continue
            t0 = time.time()
            rec = train_student_ext(Xg, yg, X_val, y_val, X_test, y_test_raw,
                                    n_classes, seed=seed, **extra)
            rec['complete'] = True
            store[key] = rec
            _save(out_dir, store, 'alpha_curves.pkl', 'alpha_results.csv',
                  'alpha', alphas, is_alpha=True)
            msg = (f"[{ds_name}] seed={seed} {group}: val={rec['best_val_acc']:.4f} "
                   f"test={rec['test_acc_best']:.4f} gap={rec['final_gen_gap']:.4f} "
                   f"({time.time() - t0:.0f}s)")
            print(msg, flush=True)
            _log(out_dir, msg)

        for alpha in alphas:
            key = (ds_name, 'Distill-S', alpha, seed)
            if key in store and store[key].get('complete'):
                continue
            t0 = time.time()
            X_T, y_T = build_D_T(X_pool, y_pool, X_S, y_S, alpha, half, seed)
            teacher = train_teacher(X_T, y_T, random_state=seed)
            soft_S, tm = _run_teacher_ext(teacher, X_S, X_test, y_test_raw, n_classes)

            rec = train_student_ext(X_S, y_S, X_val, y_val, X_test, y_test_raw,
                                    n_classes, soft_labels=soft_S, seed=seed,
                                    return_train_probs=True)
            ov, nov = _agreement(rec, soft_S, int(alpha * half))
            rec['teacher'] = tm
            rec['complete'] = True
            store[key] = rec
            _save(out_dir, store, 'alpha_curves.pkl', 'alpha_results.csv',
                  'alpha', alphas, is_alpha=True)
            msg = (f"[{ds_name}] seed={seed} Distill-S a={alpha}: "
                   f"val={rec['best_val_acc']:.4f} test={rec['test_acc_best']:.4f} "
                   f"gap={rec['final_gen_gap']:.4f} | teacher={tm['test_acc']:.4f} "
                   f"agree_ov={ov:.3f} agree_nov={nov:.3f} ({time.time() - t0:.0f}s)")
            print(msg, flush=True)
            _log(out_dir, msg)


# ======================== BETA SWEEP ========================
def run_beta_sweep_ext(ds_name, X, y, out_dir, seeds, betas):
    pkl_path = os.path.join(out_dir, 'beta_curves.pkl')
    store = pickle.load(open(pkl_path, 'rb')) if os.path.exists(pkl_path) else {}
    n_classes = len(np.unique(y))

    for seed in seeds:
        X_train_all_raw, y_train_all_raw, X_test_raw, y_test_raw = \
            _train_test_split_deterministic(X, y, seed)
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_all_raw)
        X_test = scaler.transform(X_test_raw)
        X_val, y_val, X_pool, y_pool = _val_pool_split(X_train_all, y_train_all_raw, seed)
        n_pool = len(X_pool)

        key = (ds_name, 'beta', 'Baseline-Full', seed)
        if key not in store:
            t0 = time.time()
            rec = train_student_ext(X_pool, y_pool, X_val, y_val, X_test,
                                    y_test_raw, n_classes, seed=seed)
            rec['complete'] = True
            store[key] = rec
            _save(out_dir, store, 'beta_curves.pkl', 'beta_results.csv',
                  'beta', betas, is_alpha=False)
            print(f"[{ds_name}] seed={seed} Baseline-Full: "
                  f"test={rec['test_acc_best']:.4f} ({time.time() - t0:.0f}s)", flush=True)

        for beta in betas:
            n_T = max(1, int(beta * n_pool))
            X_T_, y_T_ = X_pool[:n_T].copy(), y_pool[:n_T].copy()
            X_S, y_S = X_pool[n_T:].copy(), y_pool[n_T:].copy()

            key = (ds_name, 'beta', 'Baseline-Student', beta, seed)
            if key not in store:
                rec = train_student_ext(X_S, y_S, X_val, y_val, X_test,
                                        y_test_raw, n_classes, seed=seed)
                rec['complete'] = True
                store[key] = rec
                _save(out_dir, store, 'beta_curves.pkl', 'beta_results.csv',
                      'beta', betas, is_alpha=False)

            key = (ds_name, 'beta', 'Distill-S', beta, seed)
            if key in store and store[key].get('complete'):
                continue
            t0 = time.time()
            teacher = train_teacher(X_T_, y_T_, random_state=seed)
            soft_S, tm = _run_teacher_ext(teacher, X_S, X_test, y_test_raw, n_classes)
            rec = train_student_ext(X_S, y_S, X_val, y_val, X_test, y_test_raw,
                                    n_classes, soft_labels=soft_S, seed=seed,
                                    return_train_probs=True)
            ov, nov = _agreement(rec, soft_S, 0)     # alpha=0, no overlap region
            rec['teacher'] = tm
            rec['complete'] = True
            store[key] = rec
            _save(out_dir, store, 'beta_curves.pkl', 'beta_results.csv',
                  'beta', betas, is_alpha=False)
            msg = (f"[{ds_name}] seed={seed} beta={beta} Distill-S: "
                   f"test={rec['test_acc_best']:.4f} gap={rec['final_gen_gap']:.4f} "
                   f"| teacher={tm['test_acc']:.4f} ({time.time() - t0:.0f}s)")
            print(msg, flush=True)
            _log(out_dir, msg)


# ======================== MAIN ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='both', choices=['alpha', 'beta', 'both'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n_seeds', type=int, default=None)
    ap.add_argument('--datasets', default='wine,breastcancer,digits,glass')
    ap.add_argument('--output', default=OUT_DIR)
    args = ap.parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()
    glass_X, glass_y = _load_uci('glass')
    loaders = {
        'wine': (wine.data, wine.target),
        'breastcancer': (cancer.data, cancer.target),
        'digits': (digits.data, digits.target),
        'glass': (glass_X, glass_y),
    }
    if args.smoke:
        seeds, alphas, betas = [0], [0.0, 1.0], [0.2, 0.8]
    else:
        seeds = CONFIG['seeds'][:args.n_seeds] if args.n_seeds else CONFIG['seeds']
        alphas, betas = CONFIG['alpha_values'], CONFIG['beta_values']

    for ds in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        X, y = loaders[ds]
        print(f"\n===== {ds} ({X.shape[0]} samples) =====", flush=True)
        if args.mode in ('alpha', 'both'):
            run_alpha_sweep_ext(ds, X, y, out_dir, seeds, alphas)
        if args.mode in ('beta', 'both'):
            run_beta_sweep_ext(ds, X, y, out_dir, seeds, betas)

    print('\nAll tabular runs complete.', flush=True)


if __name__ == '__main__':
    main()