"""
Complete Label Smoothing Baseline experiment.
Run once in your terminal:
    python run_ls_baseline.py

This script:
1. Trains Baseline-LS for all 4 datasets (20 seeds each)
2. Merges LS rows into results.csv (idempotent)
3. Saves LS curves to all_curves.pkl
4. Regenerates all 5 figures
"""
import os, pickle, pandas as pd, numpy as np
from distill_ablation import (
    train_student, run_all_experiments, run_beta_experiment,
    make_plots, make_alpha_gap_plot, make_beta_plots,
    wilcoxon_test, _sig_stars, CONFIG, _load_uci,
    _train_test_split_deterministic, _val_pool_split
)
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

output_dir = CONFIG['output_dir']
os.makedirs(output_dir, exist_ok=True)

print("Loading datasets...")
wine = load_wine()
cancer = load_breast_cancer()
digits = load_digits()
glass_X, glass_y = _load_uci('glass')
print("Done.\n")

datasets = {
    'Wine':         (wine.data, wine.target),
    'BreastCancer': (cancer.data, cancer.target),
    'Digits':       (digits.data, digits.target),
    'Glass':        (glass_X, glass_y),
}

csv_path = os.path.join(output_dir, 'results.csv')
pkl_path = os.path.join(output_dir, 'all_curves.pkl')

# Load existing
df = pd.read_csv(csv_path)
curves = pickle.load(open(pkl_path, 'rb'))

all_new_rows = []

for ds_name, (X, y) in datasets.items():
    # Skip if LS rows already present and complete
    existing_ls = df[(df['dataset'] == ds_name) & (df['group'] == 'Baseline-LS')]
    if len(existing_ls) == len(CONFIG['seeds']) * len(CONFIG['alpha_values']):
        print(f"[SKIP] {ds_name} — LS rows already complete ({len(existing_ls)} rows)")
        continue

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    ds_new = []
    for seed in CONFIG['seeds']:
        key = (ds_name, 'Baseline-LS', seed)
        if key in curves:
            cached = curves[key]
            if 'e_star' in cached:
                print(f"  s{seed} cached, skip")
                h_curve = cached
            else:
                # Legacy cache: only curve data; recompute scalars
                print(f"  s{seed} cached (rebuilding metrics)...")
                gen_gap = np.array(cached.get('gen_gap_curve', [0]))
                val_acc = np.array(cached.get('val_acc', [0]))
                best_val_acc = float(np.max(val_acc)) if len(val_acc) > 0 else 0.0
                final_gen_gap = float(gen_gap[-1]) if len(gen_gap) > 0 else 0.0
                # e_star from val_loss — but we don't have val_loss in legacy cache.
                # Use best_val_acc's epoch as proxy for e_star.
                e_star = int(np.argmax(val_acc)) if len(val_acc) > 0 else 0
                h_curve = {
                    'gen_gap_curve': cached.get('gen_gap_curve', []),
                    'val_acc': cached.get('val_acc', []),
                    'e_star': e_star,
                    'best_val_acc': best_val_acc,
                    'final_gen_gap': final_gen_gap,
                    'auc_val': 0.0,
                }
                # Also update the cache with full metrics
                curves[key] = h_curve
        else:
            print(f"  s{seed} training      ", end='')
            Xt, yt, _, _ = _train_test_split_deterministic(X, y, seed)
            Xt = StandardScaler().fit_transform(Xt)
            Xv, yv, Xp, yp = _val_pool_split(Xt, yt, seed)
            half = len(Xp) // 2
            XS, yS = Xp[half:].copy(), yp[half:].copy()

            h = train_student(XS, yS, Xv, yv, seed=seed,
                              label_smoothing=CONFIG['ls_smoothing'])
            curves[key] = {'gen_gap_curve': h['gen_gap_curve'],
                                       'val_acc': h['val_acc'],
                                       'e_star': h['e_star'],
                                       'best_val_acc': h['best_val_acc'],
                                       'final_gen_gap': h['final_gen_gap']}
            h_curve = h
            print("")

        for alpha in CONFIG['alpha_values']:
            ds_new.append({
                'dataset': ds_name, 'alpha': alpha,
                'group': 'Baseline-LS', 'seed': seed,
                'e_star': h_curve['e_star'],
                'best_val_acc': h_curve['best_val_acc'],
                'auc_val': h_curve.get('auc_val', 0),
                'final_gen_gap': h_curve['final_gen_gap'],
            })

    all_new_rows.extend(ds_new)
    print(f"  -> {len(ds_new)} LS rows for {ds_name}")

# Save curves
pickle.dump(curves, open(pkl_path, 'wb'))
print(f"\n[Saved] {pkl_path} ({len(curves)} curves)")

# Merge CSV: only remove LS rows for datasets we processed (replacing old with new)
if all_new_rows:
    processed_datasets = set(r['dataset'] for r in all_new_rows)
    df = df[~((df['group'] == 'Baseline-LS') & (df['dataset'].isin(processed_datasets)))]
    df_new = pd.DataFrame(all_new_rows)
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path} ({len(df)} rows, {df.seed.nunique()} seeds, groups: {sorted(df['group'].unique())})")
else:
    print("No new LS rows to add.")

# Summary
print(f"\n=== Aggregated Summary (mean across {len(CONFIG['seeds'])} seeds) ===")
summary = df.groupby(['dataset', 'alpha', 'group']).agg(
    final_gen_gap_mean=('final_gen_gap', 'mean'),
    best_val_acc_mean=('best_val_acc', 'mean'),
).round(4)
for ds_name in sorted(df['dataset'].unique()):
    print(f"\n{ds_name}:")
    ds_s = summary.loc[ds_name]
    for grp in ['Baseline-Half', 'Baseline-LS', 'Distill-S']:
        try:
            a0 = ds_s.loc[(slice(None), grp), 'final_gen_gap_mean']
            a0_vals = a0.loc[0.0] if 0.0 in a0.index.get_level_values('alpha') else a0.mean()
            print(f"  {grp:20s} gen_gap = {a0_vals:.4f}")
        except:
            print(f"  {grp:20s} not available")

# Wilcoxon
print(f"\n=== Wilcoxon: Distill-S vs Baseline-LS @ alpha=0.0 ===")
for ds_name in datasets:
    _, p_gap = wilcoxon_test(df, ds_name, 0.0, 'Distill-S', 'Baseline-LS', 'final_gen_gap')
    stars = _sig_stars(p_gap)
    print(f"  {ds_name:15s} final_gen_gap p={p_gap:.4f} {stars}")

# Regenerate all plots
print(f"\n{'='*60}")
print("Generating all plots...")
make_plots(df, curves, output_dir)
make_alpha_gap_plot(df, output_dir)

beta_csv = os.path.join(output_dir, 'results_beta.csv')
if os.path.exists(beta_csv):
    df_beta = pd.read_csv(beta_csv)
    make_beta_plots(df_beta, output_dir)

print(f"\n{'='*60}")
print("ALL DONE — LS baseline added and figures regenerated.")
print(f"{'='*60}")