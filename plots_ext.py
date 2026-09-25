"""
Revision figures from the extended tabular results (results_ext/) and, when
available, the image-benchmark results (results_images*/).

Generates (300 dpi, results_ext/figs/):
  fig_test_acc_alpha.png     alpha vs test accuracy (best-val checkpoint)
  fig_ece_alpha.png          alpha vs ECE (best-val checkpoint)
  fig_teacher_quality.png    teacher test acc vs distillation benefit (gen gap delta)
  fig_agreement.png          teacher-student agreement: overlap vs non-overlap region
  fig_sensitivity.png        LS eps sweep | KD T x lambda grid (Glass + BreastCancer)
  fig_capacity.png           teacher capacity vs student outcome
  fig_image_summary.png      image benchmarks: gen gap & test acc vs alpha (if data exists)

Usage:  python plots_ext.py
"""
import ast
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = os.path.join('results_ext', 'figs')
COLORS = {'Baseline-Half': '#2ca02c', 'Baseline-Full': '#1f77b4',
          'Baseline-LS': '#ff7f0e', 'Baseline-Student': '#9467bd',
          'Distill-S': '#d62728'}
MARKERS = {'Baseline-Half': 's', 'Baseline-Full': '^', 'Baseline-LS': 'D',
           'Baseline-Student': 'v', 'Distill-S': 'o'}
LS = {'Baseline-Half': '--', 'Baseline-Full': ':', 'Baseline-LS': '-.',
      'Baseline-Student': (0, (3, 1, 1, 1)), 'Distill-S': '-'}


def _agg(df, metric):
    return (df.groupby(['dataset', 'alpha', 'group'])[metric]
              .agg(['mean', 'std', 'count']).reset_index())


def _style(ax, title, xlabel='alpha (overlap)'):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)


def _legend(fig, groups, y=0.99):
    handles = [plt.Line2D([0], [0], color=COLORS[g], marker=MARKERS[g],
                          linestyle=LS[g], linewidth=2, label=g) for g in groups]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y),
               ncol=min(len(groups), 5), fontsize=10, frameon=False)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {path}")


def _metric_vs_alpha(df, metric, ylabel, fname, groups, sharey=None):
    datasets = sorted(df['dataset'].unique())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.6), sharey=sharey)
    axes = np.atleast_1d(axes)
    agg = _agg(df, metric)
    for ax, ds in zip(axes, datasets):
        sub = agg[agg['dataset'] == ds]
        for g in groups:
            gd = sub[sub['group'] == g].sort_values('alpha')
            if len(gd) == 0:
                continue
            ax.errorbar(gd['alpha'], gd['mean'], yerr=gd['std'], color=COLORS[g],
                        marker=MARKERS[g], linestyle=LS[g], capsize=3,
                        markersize=5, linewidth=2, label=g)
        _style(ax, ds)
        ax.set_ylabel(ylabel, fontsize=10)
    _legend(fig, groups, y=1.02)
    fig.tight_layout()
    _save(fig, fname)


def fig_test_and_ece(df):
    groups = ['Baseline-Half', 'Baseline-LS', 'Baseline-Full', 'Distill-S']
    _metric_vs_alpha(df, 'test_acc_best', 'Test accuracy (best-val ckpt)',
                     'fig_test_acc_alpha.png', groups, sharey=False)
    _metric_vs_alpha(df, 'ece_best', 'ECE (15 bins, best-val ckpt)',
                     'fig_ece_alpha.png', groups, sharey=False)


def fig_teacher_quality(df):
    """Teacher test acc (x) vs gen-gap delta Distill-S - Baseline-Half (y)."""
    rows = []
    for (ds, alpha), sub in df.groupby(['dataset', 'alpha']):
        d = sub[sub['group'] == 'Distill-S'].set_index('seed')
        b = sub[sub['group'] == 'Baseline-Half'].set_index('seed')
        common = d.index.intersection(b.index)
        if len(common) == 0:
            continue
        rows.append({
            'dataset': ds, 'alpha': alpha,
            'teacher_acc': d.loc[common, 'teacher_test_acc'].mean(),
            'gap_delta': (d.loc[common, 'final_gen_gap']
                          - b.loc[common, 'final_gen_gap']).mean(),
        })
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for ds, sub in t.groupby('dataset'):
        ax.scatter(sub['teacher_acc'], sub['gap_delta'], s=28 + 90 * sub['alpha'],
                   alpha=0.75, label=ds, edgecolor='k', linewidth=0.4)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('Teacher test accuracy', fontsize=10)
    ax.set_ylabel('Gen-gap delta (Distill-S − Baseline-Half)', fontsize=10)
    ax.set_title('Teacher quality vs distillation regularization benefit',
                 fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    r = t[['teacher_acc', 'gap_delta']].corr().iloc[0, 1]
    ax.text(0.02, 0.02, f'Pearson r = {r:.2f} (pooled)', transform=ax.transAxes,
            fontsize=9, va='bottom')
    fig.tight_layout()
    _save(fig, 'fig_teacher_quality.png')


def fig_agreement(df):
    d = df[df['group'] == 'Distill-S']
    datasets = sorted(d['dataset'].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6),
                             sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        sub = d[d['dataset'] == ds].groupby('alpha')[
            ['agreement_overlap', 'agreement_nonoverlap']].mean().reset_index()
        ax.plot(sub['alpha'], sub['agreement_overlap'], 'o-', color='#d62728',
                linewidth=2, label='overlap region')
        ax.plot(sub['alpha'], sub['agreement_nonoverlap'], 's--', color='#7f7f7f',
                linewidth=2, label='non-overlap region')
        _style(ax, ds)
        ax.set_ylabel('Teacher-student agreement', fontsize=10)
        ax.set_ylim(0.5, 1.02)
    handles = [plt.Line2D([0], [0], color='#d62728', marker='o', lw=2,
                          label='overlap region'),
               plt.Line2D([0], [0], color='#7f7f7f', marker='s', linestyle='--',
                          lw=2, label='non-overlap region')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.04),
               ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    _save(fig, 'fig_agreement.png')


def _parse_extras(path):
    s = pd.read_csv(path)
    keys = s['key'].apply(ast.literal_eval)
    s['dataset'] = keys.apply(lambda k: k[0])
    s['kind'] = keys.apply(lambda k: k[1])
    s['p1'] = keys.apply(lambda k: k[2] if len(k) > 3 else np.nan)
    s['p2'] = keys.apply(lambda k: k[3] if len(k) > 4 else np.nan)
    for col in ('final_gen_gap', 'test_acc_best', 'teacher_test_acc'):
        s[col] = pd.to_numeric(s[col], errors='coerce')
    return s


def fig_sensitivity():
    s = _parse_extras('results_ext/extras_sensitivity.csv')
    datasets = ['glass', 'breastcancer']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, ds in enumerate(datasets):
        ls = s[(s.dataset == ds) & (s.kind == 'sens-LS')]
        ax = axes[0, col]
        g = ls.groupby('p1')[['final_gen_gap', 'test_acc_best']].mean()
        ax2 = ax.twinx()
        ax.plot(g.index, g['final_gen_gap'], 'o-', color='#d62728', lw=2,
                label='gen gap')
        ax2.plot(g.index, g['test_acc_best'], 's--', color='#1f77b4', lw=2,
                label='test acc')
        ax.set_xlabel('LS smoothing strength ε', fontsize=10)
        ax.set_title(f'{ds}: label smoothing sweep', fontsize=11)
        ax.set_ylabel('final gen gap', color='#d62728', fontsize=10)
        ax2.set_ylabel('test acc', color='#1f77b4', fontsize=10)
        ax.grid(True, alpha=0.3)

        kd = s[(s.dataset == ds) & (s.kind == 'sens-KD')]
        ax = axes[1, col]
        for T, sub in kd.groupby('p1'):
            g = sub.groupby('p2')['final_gen_gap'].mean()
            ax.plot(g.index, g.values, 'o-', lw=2, label=f'T={T:g}')
        ax.set_xlabel('mixing weight λ', fontsize=10)
        ax.set_title(f'{ds}: KD temperature × λ (gen gap)', fontsize=11)
        ax.set_ylabel('final gen gap', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig_sensitivity.png')


def fig_capacity():
    c = _parse_extras('results_ext/extras_capacity.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, ds in zip(axes, sorted(c['dataset'].unique())):
        sub = c[c.dataset == ds].groupby('p1')[
            ['teacher_test_acc', 'final_gen_gap', 'test_acc_best']].mean()
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub['teacher_test_acc'], 0.4, label='teacher test acc',
               color='#7f7f7f')
        ax.bar(x + 0.2, sub['test_acc_best'], 0.4, label='student test acc',
               color='#d62728')
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in sub.index], fontsize=9)
        ax.set_title(f'{ds}: teacher capacity (alpha=0)', fontsize=11)
        ax.set_xlabel('teacher hidden layers', fontsize=10)
        ax.set_ylabel('accuracy', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, 'fig_capacity.png')


def fig_image_summary():
    candidates = []
    for pat in ('results_images/image_results.csv',
                'results_images_cifar_all/image_results.csv'):
        files = glob.glob(pat)
        candidates += files
    if not candidates:
        print('  [Skip] fig_image_summary.png — no image results found yet')
        return
    df = pd.concat([pd.read_csv(p) for p in candidates], ignore_index=True)
    groups = ['Baseline-Half', 'Baseline-LS', 'Baseline-Full', 'Distill-S']
    datasets = sorted(df['dataset'].unique())
    fig, axes = plt.subplots(1, 2 * len(datasets),
                             figsize=(4.0 * 2 * len(datasets), 3.6))
    for i, ds in enumerate(datasets):
        sub = df[df.dataset == ds]
        for j, (metric, ylab) in enumerate(
                [('final_gen_gap', 'final gen gap'),
                 ('test_acc_best', 'test acc (best-val ckpt)')]):
            ax = axes[2 * i + j]
            agg = _agg(sub, metric)
            for g in groups:
                gd = agg[agg.group == g].sort_values('alpha')
                if len(gd) == 0:
                    continue
                ax.errorbar(gd['alpha'], gd['mean'], yerr=gd['std'],
                            color=COLORS[g], marker=MARKERS[g], linestyle=LS[g],
                            capsize=3, markersize=5, linewidth=2)
            _style(ax, f'{ds}')
            ax.set_ylabel(ylab, fontsize=10)
    _legend(fig, groups, y=1.04)
    fig.tight_layout()
    _save(fig, 'fig_image_summary.png')


def fig_beta_sweep():
    """Final generalization gap vs beta, all four datasets (post-fix data)."""
    df = pd.read_csv('results_ext/beta_results.csv')
    datasets = ['wine', 'breastcancer', 'digits', 'glass']
    groups = ['Baseline-Student', 'Distill-S', 'Baseline-Full']

    agg = (df.groupby(['dataset', 'beta', 'group'])['final_gen_gap']
             .agg(['mean', 'std']).reset_index())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, ds in zip(axes.flat, datasets):
        for grp in groups:
            gd = agg[(agg['dataset'] == ds) & (agg['group'] == grp)] \
                .sort_values('beta')
            ax.errorbar(gd['beta'], gd['mean'], yerr=gd['std'], label=grp,
                        color=COLORS[grp], marker=MARKERS[grp],
                        linestyle=LS[grp], capsize=3, markersize=5,
                        linewidth=1.6)
        ds_g = agg[(agg['dataset'] == ds) &
                   (agg['group'] == 'Distill-S')].sort_values('beta')
        bs_g = agg[(agg['dataset'] == ds) &
                   (agg['group'] == 'Baseline-Student')].sort_values('beta')
        for (_, r), (_, b) in zip(ds_g.iterrows(), bs_g.iterrows()):
            delta = r['mean'] - b['mean']
            ax.annotate(f"{'+' if delta >= 0 else ''}{delta:.3f}",
                        xy=(r['beta'], r['mean'] + r['std'] + 0.006),
                        fontsize=8, color=COLORS['Distill-S'], ha='center')
        _style(ax, f'{ds}  |  Gen Gap', xlabel=r'$\beta = |D_T|/|pool|$')
    _legend(fig, groups, y=0.99)
    fig.suptitle(r'$\beta$-sweep ($\alpha=0$, corrected pipeline, $n=20$ seeds)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'fig_beta_sweep.png')


def fig_alpha_gap(df):
    """figure_1 replacement: final gen gap vs alpha, 1x4, delta+Holm stars."""
    import pickle
    st = pd.read_csv('results_ext/stats_alpha.csv')
    st = st[(st['metric'] == 'final_gen_gap') &
            (st['contrast'] == 'Distill-S vs Baseline-Half')]
    pmap = {(r.dataset, r.alpha): r.p_holm for r in st.itertuples()}

    groups = ['Baseline-Half', 'Baseline-Full', 'Baseline-LS', 'Distill-S']
    datasets = ['wine', 'breastcancer', 'digits', 'glass']
    agg = _agg(df, 'final_gen_gap')
    alphas = sorted(df['alpha'].unique())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, ds in zip(axes, datasets):
        for grp in groups:
            gd = agg[(agg['dataset'] == ds) & (agg['group'] == grp)] \
                .sort_values('alpha')
            ax.errorbar(gd['alpha'], gd['mean'], yerr=gd['std'], label=grp,
                        color=COLORS[grp], marker=MARKERS[grp], linestyle=LS[grp],
                        capsize=3, markersize=5, linewidth=2.0)
        dsd = agg[(agg['dataset'] == ds) &
                  (agg['group'] == 'Distill-S')].sort_values('alpha')
        bh = agg[(agg['dataset'] == ds) &
                 (agg['group'] == 'Baseline-Half')].sort_values('alpha')
        for i, a in enumerate(alphas):
            delta = dsd.iloc[i]['mean'] - bh.iloc[i]['mean']
            ph = pmap.get((ds, a), 1.0)
            star = '' if ph >= 0.05 else ('*' if ph < 0.05 else '')
            if ph < 0.001:
                star = '***'
            elif ph < 0.01:
                star = '**'
            elif ph < 0.05:
                star = '*'
            ax.annotate(f"{delta:+.3f}{star}",
                        xy=(a, dsd.iloc[i]['mean'] + dsd.iloc[i]['std'] + 0.004),
                        fontsize=8, color=COLORS['Distill-S'], ha='center')
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel(r'$\alpha$ (overlap)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Final gen gap')
    handles = [plt.Line2D([0], [0], color=COLORS[g], marker=MARKERS[g],
                          linestyle=LS[g], linewidth=2, label=g)
               for g in groups]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=4, fontsize=10, frameon=False)
    fig.tight_layout()
    _save(fig, 'fig_alpha_gap.png')


def fig_gap_curves():
    """figure_2 replacement: mean gen-gap curves (alpha=0 vs 1 vs BH)."""
    import pickle
    store = pickle.load(open('results_ext/alpha_curves.pkl', 'rb'))
    datasets = ['wine', 'breastcancer', 'digits', 'glass']

    def curves(ds, grp, alpha=None):
        arrs = []
        for k, v in store.items():
            if k[0] != ds or k[1] != grp:
                continue
            if grp == 'Distill-S' and (len(k) < 4 or k[2] != alpha):
                continue
            c = np.asarray(v['gen_gap_curve'], dtype=float)
            arrs.append(c)
        m = min(len(a) for a in arrs)
        M = np.stack([a[:m] for a in arrs])
        return np.nanmean(M, axis=0), np.nanstd(M, axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, ds in zip(axes, datasets):
        for label, fn, color, style in [
                ('Distill-S $\\alpha{=}0$',
                 lambda: curves(ds, 'Distill-S', 0.0), '#d62728', '-'),
                ('Distill-S $\\alpha{=}1$',
                 lambda: curves(ds, 'Distill-S', 1.0), '#ff7f0e', '-'),
                ('Baseline-Half',
                 lambda: curves(ds, 'Baseline-Half'), '#2ca02c', '--')]:
            mean, std = fn()
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=color, linestyle=style, label=label,
                    linewidth=2.0)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Gen gap (train - val)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=10, frameon=False)
    fig.tight_layout()
    _save(fig, 'fig_gap_curves.png')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv('results_ext/alpha_results.csv')
    print('Tabular figures:')
    fig_alpha_gap(df)
    fig_gap_curves()
    fig_test_and_ece(df)
    fig_teacher_quality(df)
    fig_agreement(df)
    fig_sensitivity()
    fig_capacity()
    fig_beta_sweep()
    print('Image figures:')
    fig_image_summary()


if __name__ == '__main__':
    main()