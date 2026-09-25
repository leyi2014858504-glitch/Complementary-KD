"""Alpha-sweep driver for the Yeast (hard tabular) dataset.

Reuses the exact tabular pipeline (splits, groups, metrics) from tabular_ext.
"""
import argparse
import os

from distill_ablation import CONFIG
from tabular_ext import run_alpha_sweep_ext
from yeast_data import load_yeast

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--n_seeds', type=int, default=None)
args = ap.parse_args()

X, y = load_yeast()
os.makedirs('results_yeast', exist_ok=True)

if args.smoke:
    seeds, alphas = [0], [0.0, 1.0]
else:
    seeds = CONFIG['seeds'][:args.n_seeds] if args.n_seeds else CONFIG['seeds']
    alphas = CONFIG['alpha_values']

run_alpha_sweep_ext('yeast', X, y, 'results_yeast', seeds, alphas)
print('yeast sweep done.')
