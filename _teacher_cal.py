# -*- coding: utf-8 -*-
"""汇总各数据集 alpha=0 时 teacher 的 ECE / 软标签熵 / 置信度（用于 §4.4 补句）。"""
import pandas as pd

SOURCES = {
    "tabular": r"d:\project ML\results_ext\alpha_results.csv",
    "fmnist": r"d:\project ML\results_images\image_results.csv",
    "cifar": r"d:\project ML\results_images_cifar_all\image_results.csv",
    "yeast": r"d:\project ML\results_yeast\alpha_results.csv",
}
COLS = ["teacher_test_acc", "teacher_ece", "teacher_soft_entropy",
        "teacher_soft_conf"]

rows = []
for name, path in SOURCES.items():
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("skip", name)
        continue
    d0 = df[(df["alpha"] == 0.0) & df["teacher_ece"].notna()]
    if d0.empty:
        print("no teacher cols in", name)
        continue
    # teacher 指标与 group 无关，按 seed 去重后取均值
    d0 = d0.drop_duplicates(subset=["dataset", "seed"])
    for ds, g in d0.groupby("dataset"):
        rows.append({"dataset": ds, "n_seed": len(g),
                     **{c: round(g[c].mean(), 3) for c in COLS if c in g}})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
