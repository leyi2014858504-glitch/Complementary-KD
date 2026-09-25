# -*- coding: utf-8 -*-
"""核验 Glass 各 alpha 的 gap 均值与配对统计（Distill-S vs Baseline-Half）。"""
import pandas as pd
import numpy as np
from scipy import stats

# --- 1. Glass: 各 alpha 的 gap 配对统计 ---
df = pd.read_csv(r"d:\project ML\results_ext\alpha_results.csv")
g = df[df["dataset"].str.lower() == "glass"]

piv = g.pivot_table(index=["alpha", "seed"], columns="group",
                    values="final_gen_gap")
for alpha in sorted(piv.index.get_level_values(0).unique()):
    sub = piv.loc[alpha]
    ds = sub["Distill-S"].dropna()
    bh = sub["Baseline-Half"].dropna()
    common = ds.index.intersection(bh.index)
    d = ds.loc[common] - bh.loc[common]
    w = stats.wilcoxon(d)
    print(f"Glass alpha={alpha:.1f}  DS={ds.mean():.4f}  BH={bh.mean():.4f}  "
          f"delta={d.mean():+.4f}  wilcoxon_p={w.pvalue:.2e}  n={len(d)}")

# --- 2. Digits: alpha=0 的 test_acc_best 各组均值 ---
print()
d2 = df[df["dataset"].str.lower() == "digits"]
piv2 = d2.pivot_table(index=["alpha", "seed"], columns="group",
                      values="test_acc_best")
sub2 = piv2.loc[0.0]
print("Digits alpha=0 test_acc_best mean by group:")
print(sub2.mean(numeric_only=True).round(4).to_string())
