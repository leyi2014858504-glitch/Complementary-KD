"""
Paired statistical analysis for the PRLetters revision (Reviewer 1, point 6):
  - bootstrap 95% CI for the paired mean difference (10k resamples)
  - Wilcoxon signed-rank test (two-sided, paired by seed)
  - rank-biserial effect size r
  - Holm correction within each (contrast x metric) family over dataset x sweep

Works on any of the results CSVs produced by this project (same schema):
  results_ext/alpha_results.csv, results_ext/beta_results.csv,
  results_images/image_results.csv, results.csv (legacy, subset of metrics)

Usage:
  python stats_ext.py --results results_ext/alpha_results.csv --out results_ext/stats_alpha.csv
  python stats_ext.py --results results.csv --sweep alpha --out results_ext/stats_legacy.csv
  python stats_ext.py --results results_ext/beta_results.csv --sweep beta \
      --control Baseline-Student --out results_ext/stats_beta.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

METRICS = ['final_gen_gap', 'best_val_acc', 'test_acc_best', 'test_acc_final',
           'test_loss_best', 'test_loss_final', 'ece_best', 'ece_final', 'e_star']
N_BOOT = 10000


def paired_stats(a, b, seed=0):
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diffs = a - b
    n = len(diffs)
    if n < 3:
        return None
    rng = np.random.RandomState(seed)
    boots = rng.choice(diffs, size=(N_BOOT, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    try:
        _, p = wilcoxon(a, b)          # two-sided by default
    except ValueError:
        p = np.nan
    nz = diffs[diffs != 0]
    if len(nz) > 0:
        ranks = rankdata(np.abs(nz))
        w_plus = ranks[nz > 0].sum()
        w_minus = ranks[nz < 0].sum()
        r_rb = (w_plus - w_minus) / (w_plus + w_minus)
    else:
        r_rb = 0.0
    return {'n_seeds': n, 'mean_diff': diffs.mean(), 'ci_low': ci_low,
            'ci_high': ci_high, 'p_wilcoxon': p, 'r_rank_biserial': r_rb}


def holm_correction(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pv[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    out[valid] = adj
    return out


def stars(p):
    if np.isnan(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results_ext/alpha_results.csv')
    ap.add_argument('--sweep', default='alpha', choices=['alpha', 'beta'])
    ap.add_argument('--treat', default='Distill-S')
    ap.add_argument('--control', default='Baseline-Half')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    out_path = args.out or args.results.replace('.csv', '_stats.csv')

    rows = []
    metrics = [m for m in METRICS if m in df.columns]
    for ds in sorted(df['dataset'].unique()):
        for sweep_val in sorted(df[args.sweep].unique()):
            sub = df[(df['dataset'] == ds) & (df[args.sweep] == sweep_val)]
            treat = sub[sub['group'] == args.treat].set_index('seed')
            ctrl = sub[sub['group'] == args.control].set_index('seed')
            common = treat.index.intersection(ctrl.index)
            if len(common) < 3:
                continue
            for metric in metrics:
                a = treat.loc[common, metric].to_numpy(dtype=float)
                b = ctrl.loc[common, metric].to_numpy(dtype=float)
                res = paired_stats(a, b)
                if res is None:
                    continue
                res.update({'dataset': ds, args.sweep: sweep_val, 'metric': metric,
                            'contrast': f'{args.treat} vs {args.control}'})
                rows.append(res)

    out = pd.DataFrame(rows)
    # Holm within each (contrast x metric) family over dataset x sweep
    out['p_holm'] = np.nan
    for (_, metric), grp in out.groupby(['contrast', 'metric']):
        out.loc[grp.index, 'p_holm'] = holm_correction(grp['p_wilcoxon'].to_numpy())
    out['sig'] = [stars(p) or 'ns' for p in out['p_holm']]
    out = out[['dataset', args.sweep, 'metric', 'contrast', 'n_seeds', 'mean_diff',
               'ci_low', 'ci_high', 'p_wilcoxon', 'p_holm', 'r_rank_biserial', 'sig']]
    out = out.sort_values(['metric', 'dataset', args.sweep]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}  ({len(out)} tests)")

    show = out[out['metric'].isin(['final_gen_gap', 'test_acc_best'])]
    with pd.option_context('display.float_format', lambda v: f'{v:.4f}'):
        print(show.to_string(index=False))


if __name__ == '__main__':
    main()