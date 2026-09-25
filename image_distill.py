"""
Image-benchmark extension for the alpha-overlap distillation study
(Pattern Recognition Letters, Major Revision experiments).

Mirrors the tabular pipeline in `distill_ablation.py`:
  - validation = 20% of the training split (seed+1000 permutation),
    pool = remainder; D_S = second half of pool (size N/2, beta=0.5)
  - D_T(alpha) = first int(alpha*half) samples of D_S (overlap portion)
    + (half - overlap) samples drawn from the pool head, RNG seeded with
    seed*100 + int(alpha*100)  (non-nested design across alpha)
  - Groups: Baseline-Half, Baseline-LS (eps=0.1), Baseline-Full,
    Distill-S (lambda=0.5 mixed loss, T=1)
  - Early stopping: patience=20 on validation CE; e* = argmin val-loss epoch;
    best checkpoint = state at min val loss

Deviations from the tabular pipeline (documented in the revision response):
  - Official image train/test splits are used (test fixed across seeds);
    the tabular datasets have no official split and reuse the 80/20 protocol.
  - Fixed-budget teacher (no early stopping), analogous to the tabular
    "fixed 300 iterations, no early stopping".

Reviewer-requested additions:
  - test accuracy / test loss (NLL) at the best-val checkpoint AND final epoch
  - calibration: ECE (15 bins) at best/final
  - teacher metrics: test acc, ECE, confidence, predictive entropy,
    soft-label entropy on D_S, per-class test accuracy
  - teacher-student agreement on the overlap vs non-overlap regions of D_S
  - class-wise test accuracy for all student groups

Usage:
  python image_distill.py --smoke                        # quick pipeline check
  python image_distill.py --datasets fashion_mnist
  python image_distill.py --datasets cifar10 --n_seeds 20

Resumable: completed (dataset, group, seed[, alpha]) keys are skipped, and the
CSV + pickle are flushed after every finished run.
"""
import argparse
import copy
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

torch.backends.cudnn.benchmark = True

# ======================== CONFIG ========================
CONFIG = {
    'alpha_values': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'seeds': list(range(20)),
    'val_ratio': 0.2,
    'n_classes': 10,
    'batch_size': 128,
    'eval_batch_size': 512,
    'lr': 1e-3,
    'max_epochs': 120,
    'patience': 20,
    'teacher_epochs': 60,
    'lambda_mix': 0.5,
    'ls_smoothing': 0.1,
    'data_root': 'data',
    'output_dir': 'results_images',
    'smoke': {
        'n_train': 12000,
        'seeds': [0],
        'alpha_values': [0.0, 1.0],
        'max_epochs': 3,
        'patience': 2,
        'teacher_epochs': 2,
        'output_dir': 'results_images_smoke',
    },
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ENABLED = DEVICE.type == 'cuda'
# bf16 on Ampere+ (e.g. RTX 40xx); fp16 + GradScaler on Turing (e.g. Kaggle T4)
_AMP_DTYPE = (torch.bfloat16 if (AMP_ENABLED and torch.cuda.is_bf16_supported())
              else torch.float16)
# Optional override for testing the Kaggle/T4 path locally:
#   IMAGE_AMP_DTYPE=fp16|bf16|off
_amp_override = os.environ.get('IMAGE_AMP_DTYPE', '').strip().lower()
if _amp_override == 'fp16':
    _AMP_DTYPE = torch.float16
elif _amp_override == 'bf16':
    _AMP_DTYPE = torch.bfloat16
elif _amp_override in ('off', 'none', '0'):
    AMP_ENABLED = False
_USE_SCALER = AMP_ENABLED and _AMP_DTYPE == torch.float16
if DEVICE.type == 'cuda':
    # Shared-GPU guard: cap this process (~2.4 GB of 8 GB) so a co-running
    # job cannot be starved or OOM'd by this pipeline.  This pipeline's real
    # footprint is ~1 GB; the cap only matters when VRAM is almost full.
    torch.cuda.set_per_process_memory_fraction(0.30)


def _autocast():
    return torch.autocast('cuda', dtype=_AMP_DTYPE, enabled=AMP_ENABLED)


def _make_scaler():
    return torch.amp.GradScaler('cuda', enabled=_USE_SCALER)


# ======================== DATA ========================
def load_image_dataset(name):
    """
    Returns (X_train, y_train, X_test, y_test) as normalized float32 tensors
    of shape (N, C, 32, 32).  Uses the official train/test splits; Fashion-
    MNIST is zero-padded from 28x28 to 32x32 so both datasets share the model.
    """
    if name == 'fashion_mnist':
        tr = datasets.FashionMNIST(CONFIG['data_root'], train=True, download=True)
        te = datasets.FashionMNIST(CONFIG['data_root'], train=False, download=True)
        X_tr = tr.data.float().unsqueeze(1) / 255.0          # N,1,28,28
        X_te = te.data.float().unsqueeze(1) / 255.0
        X_tr = F.pad(X_tr, (2, 2, 2, 2))                     # -> 32x32
        X_te = F.pad(X_te, (2, 2, 2, 2))
        X_tr = (X_tr - 0.2860) / 0.3530
        X_te = (X_te - 0.2860) / 0.3530
        y_tr, y_te = tr.targets.clone(), te.targets.clone()
        in_channels = 1
    elif name == 'cifar10':
        tr = datasets.CIFAR10(CONFIG['data_root'], train=True, download=True)
        te = datasets.CIFAR10(CONFIG['data_root'], train=False, download=True)
        X_tr = torch.from_numpy(tr.data).float().permute(0, 3, 1, 2) / 255.0
        X_te = torch.from_numpy(te.data).float().permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std
        y_tr = torch.tensor(tr.targets, dtype=torch.long)
        y_te = torch.tensor(te.targets, dtype=torch.long)
        in_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X_tr, y_tr, X_te, y_te, in_channels


def build_D_T(S_idx, head_idx, half, alpha, seed):
    """
    Index-based mirror of build_D_T() in distill_ablation.py.
    Returns index array into the full training split.
    """
    overlap_size = int(alpha * half)
    n_non_overlap = half - overlap_size
    sample_rng = np.random.RandomState(seed * 100 + int(alpha * 100))
    head_pos = sample_rng.choice(half, size=n_non_overlap, replace=False)
    return np.concatenate([S_idx[:overlap_size], head_idx[head_pos]]), overlap_size


# ======================== MODEL ========================
class SmallCNN(nn.Module):
    """3-block VGG-style CNN: 32->32 pool, 64->64 pool, 128 pool, FC 256."""

    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ======================== METRIC HELPERS ========================
@torch.no_grad()
def predict_probs(model, X, bs=None):
    """Softmax probabilities for all rows of X (CPU tensor -> numpy)."""
    bs = bs or CONFIG['eval_batch_size']
    model.eval()
    outs = []
    for b in range(0, len(X), bs):
        with _autocast():
            logits = model(X[b:b + bs].to(DEVICE))
        outs.append(F.softmax(logits.float(), dim=1).cpu())
    return torch.cat(outs).numpy()


@torch.no_grad()
def eval_loss_acc(model, X, y, bs=None):
    bs = bs or CONFIG['eval_batch_size']
    model.eval()
    total_loss, correct = 0.0, 0
    for b in range(0, len(X), bs):
        xb, yb = X[b:b + bs].to(DEVICE), y[b:b + bs].to(DEVICE)
        with _autocast():
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction='sum')
        total_loss += loss.item()
        correct += (logits.float().argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def ece_score(probs, labels, n_bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = pred == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / len(labels)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def entropy_stats(probs):
    """Mean predictive entropy (nats) and mean max-probability."""
    p = np.clip(probs, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    return float(ent.mean()), float(p.max(axis=1).mean())


def classwise_accuracy(probs, labels, n_classes=10):
    pred = probs.argmax(axis=1)
    out = []
    for c in range(n_classes):
        m = labels == c
        out.append(float((pred[m] == labels[m]).mean()) if m.sum() > 0 else float('nan'))
    return out


def test_metrics(model, X_test, y_test):
    """Test acc/loss/ECE/entropy/classwise for one model state."""
    probs = predict_probs(model, X_test)
    acc = float((probs.argmax(1) == y_test.numpy()).mean())
    loss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-12)),
                            y_test).item())
    ent, conf = entropy_stats(probs)
    return {
        'test_acc': acc,
        'test_loss': loss,
        'ece': ece_score(probs, y_test.numpy()),
        'entropy_mean': ent,
        'conf_mean': conf,
        'classwise_acc': classwise_accuracy(probs, y_test.numpy()),
    }, probs


# ======================== TRAINING ========================
def train_cnn(X_train, y_train, X_val, y_val, X_test, y_test, in_channels,
              soft_labels=None, lambda_mix=None, seed=0, label_smoothing=0.0,
              max_epochs=None, patience=None, return_train_probs=False):
    """
    Mirror of train_student() in distill_ablation.py, with added test-set and
    calibration evaluation.  Returns a history dict:
      curves (train/val acc & loss, gen_gap), e_star, best_val_acc, auc_val,
      final_gen_gap, train_acc_final, val_acc_final, n_epochs,
      test_{acc,loss,ece,entropy_mean,conf_mean,classwise_acc} for the final
      model and the best-val checkpoint, and (optional) best-checkpoint
      probabilities on the training set (for agreement analysis).
    """
    max_epochs = max_epochs or CONFIG['max_epochs']
    patience = patience or CONFIG['patience']
    if lambda_mix is None:
        lambda_mix = CONFIG['lambda_mix']

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = CONFIG['n_classes']
    use_distill = soft_labels is not None
    n = len(X_train)

    if use_distill:
        soft_t = torch.from_numpy(np.asarray(soft_labels, dtype=np.float32))
    else:
        soft_t = torch.zeros(n, n_classes)

    model = SmallCNN(in_channels, n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scaler = _make_scaler()
    y_train_cpu = y_train.cpu()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    bs = CONFIG['batch_size']

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            xb = X_train[idx].to(DEVICE)
            yb = y_train_cpu[idx].to(DEVICE)
            optimizer.zero_grad()
            with _autocast():
                logits = model(xb)
                if use_distill:
                    hard_loss = F.cross_entropy(logits, yb)
                    soft_loss = F.kl_div(F.log_softmax(logits, dim=1),
                                         soft_t[idx].to(DEVICE),
                                         reduction='batchmean')
                    loss = lambda_mix * soft_loss + (1 - lambda_mix) * hard_loss
                else:
                    loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(idx)
            correct += (logits.float().argmax(1) == yb).sum().item()

        train_loss, train_acc = total_loss / n, correct / n
        val_loss, val_acc = eval_loss_acc(model, X_val, y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    history['gen_gap_curve'] = [history['train_acc'][i] - history['val_acc'][i]
                                for i in range(len(history['train_acc']))]

    # --- test metrics with the FINAL-epoch model (before loading best state)
    m_final, _ = test_metrics(model, X_test, y_test)

    # --- load best checkpoint and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
    _, best_val_acc = eval_loss_acc(model, X_val, y_val)
    m_best, _ = test_metrics(model, X_test, y_test)

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
        out['train_probs_best'] = predict_probs(model, X_train)
    return out


def train_teacher_cnn(X_T, y_T, in_channels, seed, epochs=None):
    """Fixed-budget teacher (no early stopping), analogous to the tabular one."""
    epochs = epochs or CONFIG['teacher_epochs']
    torch.manual_seed(seed + 5000)
    n = len(X_T)
    model = SmallCNN(in_channels, CONFIG['n_classes']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scaler = _make_scaler()
    y_cpu = y_T.cpu()
    bs = CONFIG['batch_size']
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            optimizer.zero_grad()
            with _autocast():
                logits = model(X_T[idx].to(DEVICE))
                loss = F.cross_entropy(logits, y_cpu[idx].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    model.eval()
    return model


# ======================== EXPERIMENT DRIVER ========================
def _save(out_dir, pkl_path, csv_path, store, alphas):
    with open(pkl_path, 'wb') as f:
        pickle.dump(store, f)
    rows = []
    for key, rec in store.items():
        if len(key) == 3:
            ds, group, seed = key
            for alpha in alphas:
                rows.append(_row(ds, alpha, group, seed, rec))
        else:
            ds, group, alpha, seed = key
            rows.append(_row(ds, alpha, group, seed, rec))
    df = pd.DataFrame(rows)
    df = df.sort_values(['dataset', 'group', 'seed', 'alpha']).reset_index(drop=True)
    df.to_csv(csv_path, index=False)


def _row(ds, alpha, group, seed, rec):
    row = {'dataset': ds, 'alpha': alpha, 'group': group, 'seed': seed}
    for k in ('e_star', 'best_val_acc', 'auc_val', 'final_gen_gap',
              'train_acc_final', 'val_acc_final', 'n_epochs',
              'test_acc_best', 'test_acc_final', 'test_loss_best',
              'test_loss_final', 'ece_best', 'ece_final',
              'entropy_mean_best', 'conf_mean_best'):
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


def run_dataset(ds_name, out_dir, seeds, alphas, max_epochs, patience,
                teacher_epochs, n_train_subset=None):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'image_curves.pkl')
    csv_path = os.path.join(out_dir, 'image_results.csv')
    log_path = os.path.join(out_dir, 'progress.log')

    store = pickle.load(open(pkl_path, 'rb')) if os.path.exists(pkl_path) else {}
    print(f"Device: {DEVICE}; dataset={ds_name}; existing cached runs: {len(store)}",
          flush=True)

    X_tr, y_tr, X_te, y_te, in_ch = load_image_dataset(ds_name)
    if n_train_subset is not None:
        X_tr, y_tr = X_tr[:n_train_subset], y_tr[:n_train_subset]
    print(f"Loaded {ds_name}: train={tuple(X_tr.shape)}, test={tuple(X_te.shape)}",
          flush=True)

    for seed in seeds:
        pool_rng = np.random.RandomState(seed + 1000)
        n = len(X_tr)
        n_val = int(CONFIG['val_ratio'] * n)
        perm = pool_rng.permutation(n)
        val_idx, pool_idx = perm[:n_val], perm[n_val:]
        half = len(pool_idx) // 2
        head_idx, S_idx = pool_idx[:half], pool_idx[half:]
        X_val, y_val = X_tr[val_idx], y_tr[val_idx]

        base_specs = [
            ('Baseline-Half', S_idx, {}),
            ('Baseline-LS', S_idx, {'label_smoothing': CONFIG['ls_smoothing']}),
            ('Baseline-Full', pool_idx, {}),
        ]
        for group, idx, extra in base_specs:
            key = (ds_name, group, seed)
            if key in store and store[key].get('complete'):
                continue
            t0 = time.time()
            rec = train_cnn(X_tr[idx], y_tr[idx], X_val, y_val, X_te, y_te,
                            in_ch, seed=seed, max_epochs=max_epochs,
                            patience=patience, **extra)
            rec['complete'] = True
            store[key] = rec
            _save(out_dir, pkl_path, csv_path, store, alphas)
            print(f"[{ds_name}] seed={seed} {group}: val_acc={rec['best_val_acc']:.4f} "
                  f"test_acc={rec['test_acc_best']:.4f} gen_gap={rec['final_gen_gap']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%H:%M:%S')} {ds_name} {group} seed={seed} "
                        f"test_acc={rec['test_acc_best']:.4f}\n")

        for alpha in alphas:
            key = (ds_name, 'Distill-S', alpha, seed)
            if key in store and store[key].get('complete'):
                continue
            t0 = time.time()
            T_idx, overlap_size = build_D_T(S_idx, head_idx, half, alpha, seed)
            teacher = train_teacher_cnn(X_tr[T_idx], y_tr[T_idx], in_ch, seed,
                                        epochs=teacher_epochs)
            soft_S = predict_probs(teacher, X_tr[S_idx])
            t_metrics, teacher_test_probs = test_metrics(teacher, X_te, y_te)
            soft_ent, soft_conf = entropy_stats(soft_S)

            rec = train_cnn(X_tr[S_idx], y_tr[S_idx], X_val, y_val, X_te, y_te,
                            in_ch, soft_labels=soft_S, seed=seed,
                            max_epochs=max_epochs, patience=patience,
                            return_train_probs=True)

            teacher_pred_S = soft_S.argmax(1)
            student_pred_S = rec['train_probs_best'].argmax(1)
            agree = teacher_pred_S == student_pred_S
            ov = agree[:overlap_size].mean() if overlap_size > 0 else float('nan')
            nov = agree[overlap_size:].mean() if overlap_size < len(agree) else float('nan')

            rec['teacher'] = {
                'test_acc': t_metrics['test_acc'],
                'ece': t_metrics['ece'],
                'conf_mean': t_metrics['conf_mean'],
                'entropy_mean': t_metrics['entropy_mean'],
                'soft_entropy': soft_ent,
                'soft_conf': soft_conf,
                'classwise_acc': t_metrics['classwise_acc'],
            }
            rec['agreement_overlap'] = float(ov)
            rec['agreement_nonoverlap'] = float(nov)
            rec['overlap_size'] = int(overlap_size)
            rec.pop('train_probs_best', None)
            rec['complete'] = True
            store[key] = rec
            _save(out_dir, pkl_path, csv_path, store, alphas)
            print(f"[{ds_name}] seed={seed} Distill-S alpha={alpha}: "
                  f"val_acc={rec['best_val_acc']:.4f} test_acc={rec['test_acc_best']:.4f} "
                  f"gen_gap={rec['final_gen_gap']:.4f} | teacher_test={t_metrics['test_acc']:.4f} "
                  f"agree_ov={ov:.3f} agree_nonov={nov:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%H:%M:%S')} {ds_name} Distill-S "
                        f"alpha={alpha} seed={seed} test_acc={rec['test_acc_best']:.4f}\n")

    _save(out_dir, pkl_path, csv_path, store, alphas)
    print(f"\n[{ds_name}] done. CSV: {csv_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='fashion_mnist',
                    help='comma-separated: fashion_mnist,cifar10')
    ap.add_argument('--smoke', action='store_true', help='tiny config for pipeline check')
    ap.add_argument('--n_seeds', type=int, default=None)
    ap.add_argument('--seed_start', type=int, default=0,
                    help='first seed index into 0..19 (for multi-GPU sharding)')
    ap.add_argument('--output', default=None)
    ap.add_argument('--data_root', default=None,
                    help="dataset download/extract directory (default: ./data)")
    args = ap.parse_args()

    if args.data_root:
        CONFIG['data_root'] = args.data_root
    torch.manual_seed(0)

    ds_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    if args.smoke:
        s = CONFIG['smoke']
        for ds in ds_list:
            run_dataset(ds, args.output or s['output_dir'], s['seeds'],
                        s['alpha_values'], s['max_epochs'], s['patience'],
                        s['teacher_epochs'], n_train_subset=s['n_train'])
    else:
        all_seeds = CONFIG['seeds'][args.seed_start:]
        seeds = all_seeds[:args.n_seeds] if args.n_seeds else all_seeds
        for ds in ds_list:
            run_dataset(ds, args.output or CONFIG['output_dir'], seeds,
                        CONFIG['alpha_values'], CONFIG['max_epochs'],
                        CONFIG['patience'], CONFIG['teacher_epochs'])


if __name__ == '__main__':
    main()