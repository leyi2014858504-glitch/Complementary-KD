# Knowledge Distillation in Data-Constrained Regimes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Experiments investigating the effect of **teacher-student data overlap** and **data allocation ratio** on knowledge distillation, with a **Label Smoothing baseline** to disentangle soft-label regularization from complementary teacher knowledge.

## Research Questions

1. How does teacher-student data overlap ratio α (0 = disjoint, 1 = identical) affect distillation's benefits?
2. What is the optimal data allocation ratio β between student direct training data and teacher distillation data?
3. Does distillation's generalization gap reduction come from (a) soft label smoothing, or (b) complementary knowledge from the teacher trained on disjoint data?
4. How does dataset difficulty (linear separability) correlate with distillation effectiveness?

## Key Findings

- **Disjoint distillation (α = 0) consistently reduces generalization gap** across all four datasets compared to Baseline-Half.
- **The benefit decays as α increases** — more overlap means less complementary knowledge.
- **Label Smoothing baseline disentangles two mechanisms:**
  - On simpler datasets (Digits, LDA 5-fold CV = 90.7%), LS *outperforms* Distill-S (p = 0.0027) — teacher complementary knowledge can be *harmful* on easy tasks.
  - On harder datasets (Glass, LDA 5-fold CV = 60.3%), Distill-S *outperforms* LS (p = 0.0007) — complementary knowledge is most valuable when data is difficult.
- **β sweep** (BreastCancer + Glass) reveals a pareto-optimal region around β ∈ [0.4, 0.6] balancing direct training data and distillation signals.

## Datasets

| Dataset | Samples | Features | Classes | LDA 5-fold CV Accuracy |
|---------|---------|----------|---------|------------------------|
| Wine | 178 | 13 | 3 | 97.2% |
| BreastCancer | 569 | 30 | 2 | 96.0% |
| Digits | 1,797 | 64 | 10 | 90.7% |
| Glass | 214 | 9 | 6 | 60.3% |

## Experimental Setup

**Models:** MLP with two hidden layers (64, 32) with ReLU activations.

**Training protocol:**
- 80/20 train/test split; from train: 20% validation, 80% pool
- Teacher: scikit-learn `MLPClassifier` trained on D_T (non-overlapping portion of pool)
- Student: PyTorch MLP trained on D_S + soft labels from teacher
- Mixed loss: λ · KL(soft) + (1 − λ) · CE(hard), with λ = 0.5
- Temperature T = 1
- Early stopping with patience = 20 epochs (max 300)
- Batch size 32, learning rate 1e-3

**Control groups:**
| Group | Description |
|-------|-------------|
| Baseline-Half | Student trained on D_S only (same data budget as Distill-S) |
| Baseline-Full | Student trained on full pool (upper bound) |
| Baseline-LS | Baseline-Half with Label Smoothing (ε = 0.1) — isolates soft-target regularization |
| Distill-S | Student on D_S + teacher soft labels from D_T |

**α-sweep:** α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} controls overlap between D_S and D_T (nested design, β = 0.5 fixed).

**β-sweep:** β ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} controls fraction of pool assigned to D_S vs D_T (α = 0 fixed, BreastCancer + Glass only).

**Statistical testing:** Two-sided Wilcoxon signed-rank test (paired by seed), N = 20 seeds.

## Repository Structure

```
.
├── distill_ablation.py    # Main experiment: training, plotting, CLI
├── run_ls_baseline.py     # Label Smoothing baseline runner
├── fisher_result.txt      # LDA separability analysis results
├── .gitignore
├── README.md
└── results/
    ├── fig1_alpha_vs_best_val_acc.png    # α vs best validation accuracy
    ├── fig2_alpha_vs_estar.png           # α vs e* (overfitting onset epoch)
    ├── fig3_gen_gap_curves.png           # Gen gap curves over epochs (α=0 vs α=1)
    ├── fig_alpha_vs_gen_gap.png          # α vs final generalization gap (with significance)
    ├── fig4_beta_sweep.png               # β sweep: val_acc and gen_gap
    ├── results.csv                       # Raw results: 1920+ rows, all seeds × datasets
    ├── results_aggregated.csv            # Mean ± std per (dataset, α, group)
    ├── results_beta.csv                  # Raw β sweep results
    ├── wilcoxon_pvalues.csv              # Wilcoxon p-values: Distill-S vs Baseline-Half
    └── all_curves.pkl                    # Cached training curves for fast re-plotting
```

## Reproducing Results

### Install Dependencies

```bash
pip install torch numpy scipy scikit-learn matplotlib pandas
```

### Run Full Experiment

```bash
python distill_ablation.py
```

This runs α-sweep + β-sweep for all datasets (20 seeds), saves results to `results/`, and generates all 5 figures. **Expect ~30–60 minutes depending on hardware.**

### Add Label Smoothing Baseline

```bash
python run_ls_baseline.py
```

Adds Baseline-LS rows to `results.csv`, caches LS training curves, and regenerates all figures with the additional baseline.

### Re-plot from Cached Data (no re-training)

```bash
python distill_ablation.py --plot
```

Regenerates all 5 figures using cached results and training curves in `results/`.

## Figures

### Fig 1: α vs Best Validation Accuracy
Validation accuracy of each group as a function of data overlap α. Distill-S generally tracks near Baseline-Half.

### Fig 2: α vs e*
e* = epoch at which validation loss reaches minimum (proxy for overfitting onset). Baseline-Half and Baseline-LS bands shown for reference. Disjoint distillation (α = 0) delays overfitting across all datasets.

### Fig 3: Generalization Gap Curves
Per-epoch generalization gap (train_acc − val_acc) for α = 0.0 (disjoint) vs α = 1.0 (identical), with all baselines. Distillation reduces gap most when data is disjoint.

### Fig 4: α vs Final Generalization Gap
Final-epoch gen_gap with Wilcoxon significance markers (Distill-S vs Baseline-Half). Highlights that disjoint distillation significantly reduces overfitting, especially on Glass.

### Fig 5: β Sweep
Validation accuracy and generalization gap as functions of data allocation ratio β (BreastCancer + Glass).

## License

MIT