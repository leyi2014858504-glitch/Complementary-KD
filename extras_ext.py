"""
Revision extras for the PRLetters major revision.

  mode sensitivity (Reviewer 1, point 5):
      LS smoothing strength eps in {0.05, 0.1, 0.2, 0.3}
      KD temperature T in {1, 2, 4} x mixing weight lambda in {0.3, 0.5, 0.7}
      at alpha=0 (fully complementary), beta=0.5, on all four tabular datasets.
      Both families share the same validation-based selection protocol
      (selection is performed in the analysis stage from these raw runs).

  mode capacity (Reviewer 1, point 4):
      teacher capacity variants (32,16) / (64,32) / (128,64) with the student
      fixed at (64,32), at alpha=0 on Glass and BreastCancer.

The teacher at alpha=0 is trained once per (dataset, seed) and reused across
all sensitivity configs.  Resumable; CSV + pickle flushed after every run.

Usage:
  python extras_ext.py --mode sensitivity --output results_ext
  python extras_ext.py --mode capacity --output results_ext
  python extras_ext.py --smoke
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from distill_ablation import (CONFIG, _train_test_split_deterministic,
                              _val_pool_split, build_D_T, train_teacher, _load_uci)
from tabular_ext import train_student_ext, align_teacher_probs, BASE_COLS

SENS_EPS = [0.05, 0.1, 0.2, 0.3]
SENS_T = [1.0, 2.0, 4.0]
SENS_LAM = [0.3, 0.5, 0.7]
CAP_HIDDEN = [(32, 16), (64, 32), (128, 64)]
CAP_DATASETS = ['glass', 'breastcancer']


def _loaders():
    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()
    glass_X, glass_y = _load_uci('glass')
    return {
        'wine': (wine.data, wine.target),
        'breastcancer': (cancer.data, cancer.target),
        'digits': (digits.data, digits.target),
        'glass': (glass_X, glass_y),
    }


def _splits(X, y, seed):
    """Mirror the main pipeline's seeded splits; returns D_S / val / test / pool."""
    X_train_all_raw, y_train_all_raw, X_test_raw, y_test_raw = \
        _train_test_split_deterministic(X, y, seed)
    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all_raw)
    X_test = scaler.transform(X_test_raw)
    X_val, y_val, X_pool, y_pool = _val_pool_split(X_train_all, y_train_all_raw, seed)
    half = len(X_pool) // 2
    return (X_pool[half:].copy(), y_pool[half:].copy(), X_val, y_val,
            X_test, y_test_raw, X_pool, y_pool, half)


def _row(rec, key):
    row = {'key': str(key)}
    for k in BASE_COLS:
        row[k] = rec.get(k)
    row['teacher_test_acc'] = rec.get('teacher', {}).get('test_acc')
    return row


def _save(out_dir, store, pkl_name, csv_name):
    with open(os.path.join(out_dir, pkl_name), 'wb') as f:
        pickle.dump(store, f)
    pd.DataFrame([_row(r, k) for k, r in store.items()]).to_csv(
        os.path.join(out_dir, csv_name), index=False)


def run_sensitivity(out_dir, datasets, seeds):
    pkl_path = os.path.join(out_dir, 'extras_sensitivity.pkl')
    store = pickle.load(open(pkl_path, 'rb')) if os.path.exists(pkl_path) else {}

    for ds_name, (X, y) in datasets.items():
        n_classes = len(np.unique(y))
        for seed in seeds:
            X_S, y_S, X_val, y_val, X_test, y_test, X_pool, y_pool, half = \
                _splits(X, y, seed)
            X_T, y_T = build_D_T(X_pool, y_pool, X_S, y_S, 0.0, half, seed)
            teacher = train_teacher(X_T, y_T, random_state=seed)
            soft = align_teacher_probs(teacher, teacher.predict_proba(X_S), n_classes)
            t_acc = float((teacher.predict(X_test) == y_test).mean())

            for eps in SENS_EPS:
                key = (ds_name, 'sens-LS', eps, seed)
                if key in store and store[key].get('complete'):
                    continue
                t0 = time.time()
                rec = train_student_ext(X_S, y_S, X_val, y_val, X_test, y_test,
                                        n_classes, seed=seed, label_smoothing=eps)
                rec['complete'] = True
                store[key] = rec
                _save(out_dir, store, 'extras_sensitivity.pkl', 'extras_sensitivity.csv')
                print(f"[{ds_name}] seed={seed} LS eps={eps}: test={rec['test_acc_best']:.4f} "
                      f"val={rec['best_val_acc']:.4f} gap={rec['final_gen_gap']:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

            for T in SENS_T:
                for lam in SENS_LAM:
                    key = (ds_name, 'sens-KD', T, lam, seed)
                    if key in store and store[key].get('complete'):
                        continue
                    t0 = time.time()
                    rec = train_student_ext(X_S, y_S, X_val, y_val, X_test, y_test,
                                            n_classes, soft_labels=soft, seed=seed,
                                            lambda_mix=lam, temperature=T)
                    rec['teacher'] = {'test_acc': t_acc}
                    rec['complete'] = True
                    store[key] = rec
                    _save(out_dir, store, 'extras_sensitivity.pkl', 'extras_sensitivity.csv')
                    print(f"[{ds_name}] seed={seed} KD T={T} lam={lam}: "
                          f"test={rec['test_acc_best']:.4f} val={rec['best_val_acc']:.4f} "
                          f"gap={rec['final_gen_gap']:.4f} ({time.time() - t0:.0f}s)", flush=True)


def run_capacity(out_dir, datasets, seeds):
    pkl_path = os.path.join(out_dir, 'extras_capacity.pkl')
    store = pickle.load(open(pkl_path, 'rb')) if os.path.exists(pkl_path) else {}

    for ds_name in CAP_DATASETS:
        X, y = datasets[ds_name]
        n_classes = len(np.unique(y))
        for seed in seeds:
            X_S, y_S, X_val, y_val, X_test, y_test, X_pool, y_pool, half = \
                _splits(X, y, seed)
            X_T, y_T = build_D_T(X_pool, y_pool, X_S, y_S, 0.0, half, seed)
            for hidden in CAP_HIDDEN:
                key = (ds_name, 'cap', hidden, seed)
                if key in store and store[key].get('complete'):
                    continue
                t0 = time.time()
                teacher = MLPClassifier(hidden_layer_sizes=hidden,
                                        max_iter=CONFIG['teacher_max_iter'],
                                        random_state=seed, early_stopping=False)
                teacher.fit(X_T, y_T)
                soft = align_teacher_probs(teacher, teacher.predict_proba(X_S), n_classes)
                t_acc = float((teacher.predict(X_test) == y_test).mean())
                rec = train_student_ext(X_S, y_S, X_val, y_val, X_test, y_test,
                                        n_classes, soft_labels=soft, seed=seed)
                rec['teacher'] = {'test_acc': t_acc, 'hidden': hidden}
                rec['complete'] = True
                store[key] = rec
                _save(out_dir, store, 'extras_capacity.pkl', 'extras_capacity.csv')
                print(f"[{ds_name}] seed={seed} teacher={hidden}: "
                      f"teacher_test={t_acc:.4f} student_test={rec['test_acc_best']:.4f} "
                      f"gap={rec['final_gen_gap']:.4f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='both',
                    choices=['sensitivity', 'capacity', 'both'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n_seeds', type=int, default=None)
    ap.add_argument('--output', default='results_ext')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    datasets = _loaders()
    seeds = [0] if args.smoke else (CONFIG['seeds'][:args.n_seeds]
                                    if args.n_seeds else CONFIG['seeds'])

    if args.mode in ('sensitivity', 'both'):
        run_sensitivity(args.output, datasets, seeds)
    if args.mode in ('capacity', 'both'):
        run_capacity(args.output, datasets, seeds)
    print('\nExtras complete.', flush=True)


if __name__ == '__main__':
    main()