import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ======================== CONFIG ========================
CONFIG = {
    'alpha_values': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'beta_values': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'seeds': list(range(20)),
    'test_size': 0.2,
    'val_ratio': 0.2,
    'teacher_hidden': (64, 32),
    'teacher_max_iter': 300,
    'student_hidden': (64, 32),
    'batch_size': 32,
    'lr': 1e-3,
    'max_epochs': 300,
    'patience': 20,
    'lambda_mix': 0.5,
    'ls_smoothing': 0.1,
    'output_dir': 'results',
    'dpi': 300,
}


# ======================== PyTorch MLP ========================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


# ======================== DATA SPLITS ========================
def _train_test_split_deterministic(X, y, seed):
    rng = np.random.RandomState(seed)
    n_total = len(X)
    n_train = int((1 - CONFIG['test_size']) * n_total)
    indices = rng.permutation(n_total)
    return X[indices[:n_train]].copy(), y[indices[:n_train]].copy(), \
           X[indices[n_train:]].copy(), y[indices[n_train:]].copy()


def _val_pool_split(X_train_all, y_train_all, seed):
    """
    Split into val and pool sets using the same random permutation for X and y.
    NOTE: X_train_all must already be standardized (fit_transform preserves row order),
    and y_train_all must be the raw labels aligned to the same rows. Both are indexed
    by the identical `perm`, so X-y pairing is guaranteed correct.
    """
    pool_rng = np.random.RandomState(seed + 1000)
    n = len(X_train_all)
    n_val = int(CONFIG['val_ratio'] * n)
    perm = pool_rng.permutation(n)
    val_idx = perm[:n_val]
    pool_idx = perm[n_val:]
    return (X_train_all[val_idx].copy(), y_train_all[val_idx].copy(),
            X_train_all[pool_idx].copy(), y_train_all[pool_idx].copy())


def build_D_T(X_pool, y_pool, X_S, y_S, alpha, half, seed):
    """
    Build teacher training set D_T with controlled overlap with D_S.

    D_T consists of:
      - overlap_size samples drawn from the start of D_S (the overlapping portion)
      - (half - overlap_size) samples randomly drawn from pool-first-half
        (the non-overlapping portion, disjoint from D_S by construction).

    Random sampling of the non-overlap portion is seeded deterministically
    (using both `seed` and `alpha`) so that different alpha values draw
    independent random subsets rather than nested prefixes. This avoids
    systematic composition bias when comparing across alpha.
    """
    overlap_size = int(alpha * half)
    n_non_overlap = half - overlap_size

    # Overlap: first overlap_size samples from D_S
    X_T_overlap = X_S[:overlap_size].copy()
    y_T_overlap = y_S[:overlap_size].copy()

    # Non-overlap: randomly sample from pool first half (indices 0..half-1)
    sample_rng = np.random.RandomState(seed * 100 + int(alpha * 100))
    pool_head_indices = sample_rng.choice(half, size=n_non_overlap, replace=False)

    X_T_nonoverlap = X_pool[pool_head_indices].copy()
    y_T_nonoverlap = y_pool[pool_head_indices].copy()

    X_T = np.concatenate([X_T_overlap, X_T_nonoverlap], axis=0)
    y_T = np.concatenate([y_T_overlap, y_T_nonoverlap], axis=0)
    return X_T, y_T


# ======================== TEACHER ========================
def train_teacher(X_T, y_T, random_state=42):
    model = MLPClassifier(
        hidden_layer_sizes=CONFIG['teacher_hidden'],
        max_iter=CONFIG['teacher_max_iter'],
        random_state=random_state,
        early_stopping=False,
    )
    model.fit(X_T, y_T)
    return model


def get_soft_labels(teacher, X_S):
    return teacher.predict_proba(X_S)


# ======================== STUDENT TRAINING ========================
def train_student(X_train, y_train, X_val, y_val,
                  soft_labels=None, lambda_mix=None, seed=0,
                  label_smoothing=0.0):
    if lambda_mix is None:
        lambda_mix = CONFIG['lambda_mix']

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    use_distill = (soft_labels is not None)
    n_features = X_train.shape[1]

    n_classes = len(np.unique(np.concatenate([y_train, y_val])))
    if use_distill and soft_labels.shape[1] < n_classes:
        padded = np.full((soft_labels.shape[0], n_classes), 1e-8,
                         dtype=soft_labels.dtype)
        padded[:, :soft_labels.shape[1]] = soft_labels
        soft_labels = padded

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    if use_distill:
        soft_t = torch.FloatTensor(soft_labels)
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
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch_X, batch_y = batch[0].to(device), batch[1].to(device)
            batch_soft = batch[2].to(device)

            optimizer.zero_grad()
            logits = model(batch_X)

            if use_distill:
                hard_loss = F.cross_entropy(logits, batch_y)
                # Teacher soft labels are T=1 predict_proba outputs (already
                # probability distributions).  We compute KL(teacher || student)
                # via log_softmax, which at T=1 is equivalent to soft cross-entropy
                # up to the constant entropy of the teacher distribution.
                student_log_prob = F.log_softmax(logits, dim=1)
                soft_loss = F.kl_div(student_log_prob, batch_soft,
                                     reduction='batchmean')
                loss = lambda_mix * soft_loss + (1 - lambda_mix) * hard_loss
            else:
                loss = F.cross_entropy(logits, batch_y,
                                       label_smoothing=label_smoothing)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_X.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = F.cross_entropy(val_logits, y_val_t.to(device)).item()
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val_t.to(device)).float().mean().item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            break

    history['gen_gap_curve'] = [
        history['train_acc'][i] - history['val_acc'][i]
        for i in range(len(history['train_acc']))
    ]

    e_star = int(np.argmin(history['val_loss']))
    # e* is the epoch at which val_loss reaches its global minimum.  Due to
    # early-stopping with patience=CONFIG['patience'], training halts exactly
    # patience epochs after this minimum.  We interpret e* as a proxy for the
    # onset of overfitting: a larger e* implies that the model continued to
    # improve for more epochs before val_loss started degrading.
    #
    # NOTE: When a model stops early, e* ≈ (total_epochs - patience).  The
    # absolute value of e* is thus bounded by the early-stopping horizon;
    # comparisons across groups are still valid because all groups share the
    # same patience and max_epochs.

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_logits = model(X_val_t.to(device))
        best_val_pred = best_logits.argmax(dim=1)
        best_val_acc = (best_val_pred == y_val_t.to(device)).float().mean().item()

    auc_val = float(np.trapezoid(history['val_acc']))
    final_gen_gap = history['gen_gap_curve'][-1]

    history['e_star'] = e_star
    history['best_val_acc'] = best_val_acc
    history['auc_val'] = auc_val
    history['final_gen_gap'] = final_gen_gap

    return history


# ======================== RUN ALL EXPERIMENTS ========================
def run_all_experiments(dataset_name, X, y):
    results = []
    all_curves = {}

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"{'='*60}")

    for seed in CONFIG['seeds']:
        print(f"\n--- Seed {seed} ---")

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

        # Baseline-Half
        print(f"  Training Baseline-Half (|D_S|={len(X_S)})...")
        bh_history = train_student(X_S, y_S, X_val, y_val,
                                   soft_labels=None, seed=seed)
        curve_key = (dataset_name, 'Baseline-Half', seed)
        all_curves[curve_key] = {
            'gen_gap_curve': bh_history['gen_gap_curve'],
            'val_acc': bh_history['val_acc'],
        }

        # Baseline-LS (Label Smoothing — same D_S, soft uniform target)
        print(f"  Training Baseline-LS (|D_S|={len(X_S)}, ls={CONFIG['ls_smoothing']})...")
        ls_history = train_student(X_S, y_S, X_val, y_val,
                                   soft_labels=None, seed=seed,
                                   label_smoothing=CONFIG['ls_smoothing'])
        curve_key = (dataset_name, 'Baseline-LS', seed)
        all_curves[curve_key] = {
            'gen_gap_curve': ls_history['gen_gap_curve'],
            'val_acc': ls_history['val_acc'],
        }

        # Baseline-Full
        print(f"  Training Baseline-Full (|pool|={n_pool})...")
        bf_history = train_student(X_pool, y_pool, X_val, y_val,
                                   soft_labels=None, seed=seed)
        curve_key = (dataset_name, 'Baseline-Full', seed)
        all_curves[curve_key] = {
            'gen_gap_curve': bf_history['gen_gap_curve'],
            'val_acc': bf_history['val_acc'],
        }

        for alpha in CONFIG['alpha_values']:
            results.append({
                'dataset': dataset_name, 'alpha': alpha,
                'group': 'Baseline-Half', 'seed': seed,
                'e_star': bh_history['e_star'],
                'best_val_acc': bh_history['best_val_acc'],
                'auc_val': bh_history['auc_val'],
                'final_gen_gap': bh_history['final_gen_gap'],
            })
            results.append({
                'dataset': dataset_name, 'alpha': alpha,
                'group': 'Baseline-Full', 'seed': seed,
                'e_star': bf_history['e_star'],
                'best_val_acc': bf_history['best_val_acc'],
                'auc_val': bf_history['auc_val'],
                'final_gen_gap': bf_history['final_gen_gap'],
            })
            results.append({
                'dataset': dataset_name, 'alpha': alpha,
                'group': 'Baseline-LS', 'seed': seed,
                'e_star': ls_history['e_star'],
                'best_val_acc': ls_history['best_val_acc'],
                'auc_val': ls_history['auc_val'],
                'final_gen_gap': ls_history['final_gen_gap'],
            })

        # Distill-S for each alpha
        for alpha in CONFIG['alpha_values']:
            X_T, y_T = build_D_T(X_pool, y_pool, X_S, y_S, alpha, half, seed)
            overlap_size = int(alpha * half)
            teacher = train_teacher(X_T, y_T, random_state=seed)
            soft_labels = get_soft_labels(teacher, X_S)

            print(f"  Training Distill-S (alpha={alpha:.1f}, overlap={overlap_size}/{half})...")
            ds_history = train_student(X_S, y_S, X_val, y_val,
                                       soft_labels=soft_labels,
                                       lambda_mix=CONFIG['lambda_mix'],
                                       seed=seed)
            curve_key = (dataset_name, 'Distill-S', alpha, seed)
            all_curves[curve_key] = {
                'gen_gap_curve': ds_history['gen_gap_curve'],
                'val_acc': ds_history['val_acc'],
            }

            results.append({
                'dataset': dataset_name, 'alpha': alpha,
                'group': 'Distill-S', 'seed': seed,
                'e_star': ds_history['e_star'],
                'best_val_acc': ds_history['best_val_acc'],
                'auc_val': ds_history['auc_val'],
                'final_gen_gap': ds_history['final_gen_gap'],
            })

    df = pd.DataFrame(results)
    return df, all_curves


# ======================== BETA SWEEP (data allocation ratio) ========================
def run_beta_experiment(dataset_name, X, y):
    """
    Sweep the teacher-data allocation ratio beta = |D_T| / |pool|.

    Alpha is fixed at 0.0 (D_T and D_S partition the pool disjointly).
    For each beta, |D_T| = int(beta * |pool|) and D_S gets the remainder.

    Baselines:
      Baseline-Student — D_S (hard labels only), same data as Distill-S.
          This is the per-beta apples-to-apples control: it isolates the
          effect of distillation from the confounding effect of varying
          student sample size.
      Baseline-Full    — entire pool (hard labels only), upper bound.

    Returns (results_df, all_curves).
    """
    results = []
    all_curves = {}

    print(f"\n{'~'*60}")
    print(f"BETA SWEEP ─ Dataset: {dataset_name}")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"{'~'*60}")

    for seed in CONFIG['seeds']:
        print(f"\n--- Seed {seed} ---")

        X_train_all_raw, y_train_all_raw, X_test_raw, y_test_raw = \
            _train_test_split_deterministic(X, y, seed)

        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_all_raw)
        X_test = scaler.transform(X_test_raw)

        X_val, y_val, X_pool, y_pool = _val_pool_split(X_train_all, y_train_all_raw, seed)

        n_pool = len(X_pool)

        # Baseline-Full: train on entire pool (upper bound)
        print(f"  Training Baseline-Full (|pool|={n_pool})...")
        bf_history = train_student(X_pool, y_pool, X_val, y_val,
                                   soft_labels=None, seed=seed)

        for beta in CONFIG['beta_values']:
            n_T = max(1, int(beta * n_pool))
            n_S = n_pool - n_T

            # D_T: first n_T samples; D_S: the rest (disjoint by construction, alpha=0)
            X_T = X_pool[:n_T].copy()
            y_T = y_pool[:n_T].copy()
            X_S = X_pool[n_T:].copy()
            y_S = y_pool[n_T:].copy()

            # Baseline-Student: train on D_S with hard labels (per-beta control)
            bh_history = train_student(X_S, y_S, X_val, y_val,
                                        soft_labels=None, seed=seed)
            curve_key = (dataset_name, 'beta', 'Baseline-Student', beta, seed)
            all_curves[curve_key] = {
                'gen_gap_curve': bh_history['gen_gap_curve'],
                'val_acc': bh_history['val_acc'],
            }

            # Distill-S: train on D_S with mixed labels from teacher on D_T
            teacher = train_teacher(X_T, y_T, random_state=seed)
            soft_labels = get_soft_labels(teacher, X_S)

            print(f"  beta={beta:.1f}  |D_T|={n_T}  |D_S|={n_S} ...")
            ds_history = train_student(X_S, y_S, X_val, y_val,
                                       soft_labels=soft_labels,
                                       lambda_mix=CONFIG['lambda_mix'],
                                       seed=seed)
            curve_key = (dataset_name, 'beta', 'Distill-S', beta, seed)
            all_curves[curve_key] = {
                'gen_gap_curve': ds_history['gen_gap_curve'],
                'val_acc': ds_history['val_acc'],
            }

            for grp, hist in [('Baseline-Student', bh_history), ('Distill-S', ds_history)]:
                results.append({
                    'dataset': dataset_name,
                    'beta': beta,
                    'group': grp,
                    'seed': seed,
                    'e_star': hist['e_star'],
                    'best_val_acc': hist['best_val_acc'],
                    'auc_val': hist['auc_val'],
                    'final_gen_gap': hist['final_gen_gap'],
                })

            # Baseline-Full is independent of beta — store once per seed then re-attach
            results.append({
                'dataset': dataset_name,
                'beta': beta,
                'group': 'Baseline-Full',
                'seed': seed,
                'e_star': bf_history['e_star'],
                'best_val_acc': bf_history['best_val_acc'],
                'auc_val': bf_history['auc_val'],
                'final_gen_gap': bf_history['final_gen_gap'],
            })

    df = pd.DataFrame(results)
    return df, all_curves


# ======================== WILCOXON TEST ========================
def wilcoxon_test(df, dataset, alpha, group1, group2, metric='best_val_acc'):
    """
    Perform paired Wilcoxon signed-rank test between two groups across seeds.
    Returns (statistic, p_value).
    """
    vals1 = df[(df['dataset'] == dataset) & (df['alpha'] == alpha) & (df['group'] == group1)][metric].values
    vals2 = df[(df['dataset'] == dataset) & (df['alpha'] == alpha) & (df['group'] == group2)][metric].values
    if len(vals1) == 0 or len(vals2) == 0:
        return np.nan, np.nan
    # Ensure paired ordering by seed
    merged = pd.merge(
        df[(df['dataset'] == dataset) & (df['alpha'] == alpha) & (df['group'] == group1)][['seed', metric]],
        df[(df['dataset'] == dataset) & (df['alpha'] == alpha) & (df['group'] == group2)][['seed', metric]],
        on='seed', how='inner', suffixes=('_1', '_2')
    )
    if len(merged) < 3:
        return np.nan, np.nan
    diffs = merged[f'{metric}_2'].values - merged[f'{metric}_1'].values
    if np.allclose(diffs, 0):
        return np.nan, np.nan
    stat, p = wilcoxon(merged[f'{metric}_1'], merged[f'{metric}_2'])
    return stat, p


# ======================== CURVE AGGREGATION ========================
def aggregate_curves(all_curves, dataset, group, alpha=None):
    max_len = 0
    curves = []
    if alpha is not None:
        key_pattern = (dataset, group, alpha)
        for key, val in all_curves.items():
            if len(key) == 4 and key[0] == dataset and key[1] == group and key[3] != '_' and key[2] == alpha:
                curves.append(val['gen_gap_curve'])
                max_len = max(max_len, len(val['gen_gap_curve']))
    else:
        key_pattern = (dataset, group)
        for key, val in all_curves.items():
            if len(key) == 3 and key[0] == dataset and key[1] == group:
                curves.append(val['gen_gap_curve'])
                max_len = max(max_len, len(val['gen_gap_curve']))

    if not curves:
        return [], []

    padded = []
    for c in curves:
        if len(c) < max_len:
            c = c + [c[-1]] * (max_len - len(c))
        padded.append(c)

    arr = np.array(padded)
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0)
    return mean_curve, std_curve


# ======================== PLOTTING ========================
def _sig_stars(p):
    if p is None or np.isnan(p):
        return 'n.s.'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def make_plots(df, all_curves, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    datasets = sorted(df['dataset'].unique())
    groups_all = ['Baseline-Half', 'Baseline-Full', 'Baseline-LS', 'Distill-S']
    colors = {
        'Baseline-Half': '#2ca02c',
        'Baseline-Full': '#1f77b4',
        'Baseline-LS': '#ff7f0e',
        'Distill-S': '#d62728',
    }
    markers = {
        'Baseline-Half': 's',
        'Baseline-Full': '^',
        'Baseline-LS': 'D',
        'Distill-S': 'o',
    }
    linestyles = {
        'Baseline-Half': '--',
        'Baseline-Full': ':',
        'Baseline-LS': '-.',
        'Distill-S': '-',
    }

    agg = df.groupby(['dataset', 'alpha', 'group']).agg(
        mean_e_star=('e_star', 'mean'),
        std_e_star=('e_star', 'std'),
        mean_best_val_acc=('best_val_acc', 'mean'),
        std_best_val_acc=('best_val_acc', 'std'),
        mean_final_gen_gap=('final_gen_gap', 'mean'),
        std_final_gen_gap=('final_gen_gap', 'std'),
    ).reset_index()

    alphas = sorted(df['alpha'].unique())
    n_ds = len(datasets)
    n_cols = 4
    n_rows = 1

    # ======== FIGURE 1: alpha vs best_val_acc (1x4) ========
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(20, 4.2))
    axes1 = axes1.flatten()

    for ax, ds in zip(axes1, datasets):
        ds_agg = agg[agg['dataset'] == ds]
        for grp in groups_all:
            gd = ds_agg[ds_agg['group'] == grp].sort_values('alpha')
            ax.errorbar(gd['alpha'], gd['mean_best_val_acc'],
                        yerr=gd['std_best_val_acc'],
                        label=grp,
                        color=colors[grp],
                        marker=markers[grp],
                        linestyle=linestyles[grp],
                        capsize=3, markersize=5, linewidth=2.0)

        ds_distill = ds_agg[ds_agg['group'] == 'Distill-S'].sort_values('alpha')
        y_positions = ds_distill['mean_best_val_acc'].values
        yerrs = ds_distill['std_best_val_acc'].values
        for i, alpha in enumerate(ds_distill['alpha'].values):
            _, p = wilcoxon_test(df, ds, alpha, 'Distill-S', 'Baseline-Half', 'best_val_acc')
            stars = _sig_stars(p)
            if stars != 'n.s.':
                ax.annotate(stars,
                            xy=(alpha, y_positions[i] + yerrs[i] + 0.02),
                            fontsize=9, fontweight='bold',
                            color=colors['Distill-S'], ha='center')

        ax.set_xlabel('alpha (overlap)', fontsize=10)
        ax.set_ylabel('Best Validation Accuracy', fontsize=10)
        ax.set_title(ds)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0, top=1.05)

    for ax in axes1[n_ds:]:
        ax.set_visible(False)

    handles1 = [plt.Line2D([0], [0], color=colors[g], marker=markers[g],
                           linestyle=linestyles[g], linewidth=2, label=g)
                for g in groups_all]
    fig1.legend(handles=handles1, loc='upper center', bbox_to_anchor=(0.5, 0.94),
                ncol=4, fontsize=10, frameon=False)
    fig1.suptitle('alpha vs Best Validation Accuracy  (* p<.05  ** p<.01  *** p<.001 vs Baseline-Half)',
                  fontsize=12, fontweight='bold', y=0.99)
    fig1.tight_layout(rect=[0, 0, 1, 0.88])
    fig1.savefig(os.path.join(output_dir, 'fig1_alpha_vs_best_val_acc.png'),
                 dpi=CONFIG['dpi'])
    plt.close(fig1)
    print("  [Saved] fig1_alpha_vs_best_val_acc.png")

    # ======== FIGURE 2: alpha vs e_star (1x4) ========
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(20, 4.2))
    axes2 = axes2.flatten()

    for ax, ds in zip(axes2, datasets):
        ds_agg = agg[agg['dataset'] == ds]
        bh_data = ds_agg[ds_agg['group'] == 'Baseline-Half']
        bh_mean = bh_data['mean_e_star'].iloc[0]
        bh_std = bh_data['std_e_star'].iloc[0]
        ax.axhline(y=bh_mean, color=colors['Baseline-Half'], linestyle='--', alpha=0.7,
                   linewidth=2)
        ax.fill_between(alphas, bh_mean - bh_std, bh_mean + bh_std,
                        color=colors['Baseline-Half'], alpha=0.12,
                        label='Baseline-Half')

        ls_data = ds_agg[ds_agg['group'] == 'Baseline-LS']
        ls_mean = ls_data['mean_e_star'].iloc[0]
        ls_std = ls_data['std_e_star'].iloc[0]
        ax.axhline(y=ls_mean, color=colors['Baseline-LS'], linestyle='-.', alpha=0.7,
                   linewidth=2)
        ax.fill_between(alphas, ls_mean - ls_std, ls_mean + ls_std,
                        color=colors['Baseline-LS'], alpha=0.12,
                        label='Baseline-LS')

        ds_data = ds_agg[ds_agg['group'] == 'Distill-S'].sort_values('alpha')
        ax.errorbar(ds_data['alpha'], ds_data['mean_e_star'],
                    yerr=ds_data['std_e_star'],
                    label='Distill-S',
                    color=colors['Distill-S'],
                    marker=markers['Distill-S'],
                    capsize=3, markersize=5, linewidth=2.0)

        y_positions = ds_data['mean_e_star'].values
        yerrs = ds_data['std_e_star'].values
        for i, alpha in enumerate(ds_data['alpha'].values):
            _, p = wilcoxon_test(df, ds, alpha, 'Distill-S', 'Baseline-Half', 'e_star')
            stars = _sig_stars(p)
            if stars != 'n.s.':
                ax.annotate(stars,
                            xy=(alpha, y_positions[i] + yerrs[i] + 4),
                            fontsize=9, fontweight='bold',
                            color=colors['Distill-S'], ha='center')

        ax.set_xlabel('alpha (overlap)', fontsize=10)
        ax.set_ylabel('e* (epoch of min val_loss)', fontsize=10)
        ax.set_title(ds)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes2[n_ds:]:
        ax.set_visible(False)

    handles2 = [
        plt.Line2D([0], [0], color=colors['Baseline-Half'], linestyle='--', linewidth=2,
                   label='Baseline-Half'),
        plt.Line2D([0], [0], color=colors['Baseline-LS'], linestyle='-.', linewidth=2,
                   label='Baseline-LS'),
        plt.Line2D([0], [0], color=colors['Distill-S'], marker=markers['Distill-S'],
                   linewidth=2, label='Distill-S'),
    ]
    fig2.legend(handles=handles2, loc='upper center', bbox_to_anchor=(0.5, 0.94),
                ncol=3, fontsize=10, frameon=False)
    fig2.suptitle('alpha vs e*  (* p<.05  ** p<.01  *** p<.001 vs Baseline-Half)',
                  fontsize=12, fontweight='bold', y=0.99)
    fig2.tight_layout(rect=[0, 0, 1, 0.88])
    fig2.savefig(os.path.join(output_dir, 'fig2_alpha_vs_estar.png'),
                 dpi=CONFIG['dpi'])
    plt.close(fig2)
    print("  [Saved] fig2_alpha_vs_estar.png")

    # ======== FIGURE 3: gen_gap_curve alpha=0.0 / 1.0 (2x2) ========
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes3 = axes3.flatten()

    gen_gap_cfg = {
        ('Distill-S', 0.0):   {'color': '#d62728', 'ls': '-',  'label': 'Distill-S (a=0.0)'},
        ('Distill-S', 1.0):   {'color': '#ff7f0e', 'ls': '-',  'label': 'Distill-S (a=1.0)'},
        ('Baseline-LS', None): {'color': '#9467bd', 'ls': '-.', 'label': 'Baseline-LS'},
        ('Baseline-Half', None): {'color': '#2ca02c', 'ls': '--', 'label': 'Baseline-Half'},
    }

    for ax, ds in zip(axes3, datasets):
        for (grp, alp), cfg in gen_gap_cfg.items():
            if alp is not None:
                mean_c, std_c = aggregate_curves(all_curves, ds, grp, alpha=alp)
            else:
                mean_c, std_c = aggregate_curves(all_curves, ds, grp)
            if len(mean_c) == 0:
                continue
            epochs = np.arange(len(mean_c))
            ax.plot(epochs, mean_c, color=cfg['color'], linestyle=cfg['ls'],
                    linewidth=1.5, label=cfg['label'])
            show_fill = (grp != 'Baseline-Half')
            if show_fill and len(std_c) > 0:
                ax.fill_between(epochs, mean_c - std_c, mean_c + std_c,
                                color=cfg['color'], alpha=0.10)
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Gen Gap (train_acc - val_acc)', fontsize=10)
        ax.set_title(ds)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    for ax in axes3[n_ds:]:
        ax.set_visible(False)

    handles3 = [plt.Line2D([0], [0], color=cfg['color'], linestyle=cfg['ls'],
                           linewidth=2, label=cfg['label'])
                for cfg in gen_gap_cfg.values()]
    fig3.legend(handles=handles3, loc='upper center', bbox_to_anchor=(0.5, 0.94),
                ncol=4, fontsize=10, frameon=False)
    fig3.suptitle('Generalization Gap Curves', fontsize=12, fontweight='bold', y=0.99)
    fig3.tight_layout(rect=[0, 0, 1, 0.88])
    fig3.savefig(os.path.join(output_dir, 'fig3_gen_gap_curves.png'),
                 dpi=CONFIG['dpi'])
    plt.close(fig3)
    print("  [Saved] fig3_gen_gap_curves.png")


# ======================== BETA PLOTS ========================
def make_beta_plots(df_beta, output_dir):
    """
    Figure 4: beta-sweep results.
    Two vertically-stacked subplots (one per dataset) × best_val_acc & final_gen_gap
    side by side, giving a 2×2 layout.
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = sorted(df_beta['dataset'].unique())
    betas = sorted(df_beta['beta'].unique())
    groups = ['Baseline-Student', 'Distill-S', 'Baseline-Full']
    colors = {
        'Baseline-Student': '#2ca02c',
        'Baseline-Full': '#1f77b4',
        'Distill-S': '#d62728',
    }
    markers = {
        'Baseline-Student': 's',
        'Baseline-Full': '^',
        'Distill-S': 'o',
    }

    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 5 * len(datasets)),
                             squeeze=False)

    for row, ds in enumerate(datasets):
        ds_all = df_beta[df_beta['dataset'] == ds]
        agg = ds_all.groupby(['beta', 'group']).agg(
            mean_val=('best_val_acc', 'mean'),
            std_val=('best_val_acc', 'std'),
            mean_gap=('final_gen_gap', 'mean'),
            std_gap=('final_gen_gap', 'std'),
        ).reset_index()

        # --- best_val_acc ---
        ax_acc = axes[row][0]
        for grp in groups:
            gd = agg[agg['group'] == grp].sort_values('beta')
            ax_acc.errorbar(gd['beta'], gd['mean_val'], yerr=gd['std_val'],
                            label=grp, color=colors[grp], marker=markers[grp],
                            capsize=3, markersize=5, linewidth=1.3)

        # Annotate Distill-S - Baseline-Student delta
        ds_agg_dist = agg[agg['group'] == 'Distill-S'].sort_values('beta')
        y_pos = ds_agg_dist['mean_val'].values
        y_err = ds_agg_dist['std_val'].values
        for i, b in enumerate(ds_agg_dist['beta'].values):
            bh_val = agg[(agg['group'] == 'Baseline-Student') & (agg['beta'] == b)]['mean_val'].values
            if len(bh_val) > 0:
                delta = y_pos[i] - bh_val[0]
                sign = '+' if delta >= 0 else ''
                ax_acc.annotate(f'{sign}{delta:.3f}',
                                xy=(b, y_pos[i] + y_err[i] + 0.015),
                                fontsize=7, color=colors['Distill-S'], ha='center')

        ax_acc.set_xlabel('beta = |D_T| / |pool|', fontsize=10)
        ax_acc.set_ylabel('Best Validation Accuracy', fontsize=10)
        ax_acc.set_title(f'{ds}')
        ax_acc.tick_params(labelsize=9)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_xlim(betas[0] - 0.05, betas[-1] + 0.05)
        ax_acc.set_ylim(bottom=0.0, top=1.05)

        # --- final_gen_gap ---
        ax_gap = axes[row][1]
        for grp in groups:
            gd = agg[agg['group'] == grp].sort_values('beta')
            ax_gap.errorbar(gd['beta'], gd['mean_gap'], yerr=gd['std_gap'],
                            label=grp, color=colors[grp], marker=markers[grp],
                            capsize=3, markersize=5, linewidth=2.0)

        # Annotate Distill-S - Baseline-Student gap reduction
        for i, b in enumerate(ds_agg_dist['beta'].values):
            bh_gap = agg[(agg['group'] == 'Baseline-Student') & (agg['beta'] == b)]['mean_gap'].values
            gap_dist = ds_agg_dist['mean_gap'].values[i]
            gap_err = ds_agg_dist['std_gap'].values[i]
            if len(bh_gap) > 0:
                delta = gap_dist - bh_gap[0]
                sign = '+' if delta >= 0 else ''
                ax_gap.annotate(f'{sign}{delta:.3f}',
                                xy=(b, gap_dist + gap_err + 0.008),
                                fontsize=7, color=colors['Distill-S'], ha='center')

        ax_gap.set_xlabel('beta = |D_T| / |pool|', fontsize=10)
        ax_gap.set_ylabel('Final Gen Gap (train_acc - val_acc)', fontsize=10)
        ax_gap.set_title(f'{ds}  |  Gen Gap')
        ax_gap.tick_params(labelsize=9)
        ax_gap.grid(True, alpha=0.3)
        ax_gap.set_xlim(betas[0] - 0.05, betas[-1] + 0.05)

    handles4 = [plt.Line2D([0], [0], color=colors[g], marker=markers[g],
                           linewidth=2, label=g)
                for g in groups]
    fig.legend(handles=handles4, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               ncol=3, fontsize=10, frameon=False)
    fig.suptitle('beta Sweep — Data Allocation Ratio  (alpha=0, no overlap)',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(output_dir, 'fig4_beta_sweep.png'),
                dpi=CONFIG['dpi'])
    plt.close(fig)
    print("  [Saved] fig4_beta_sweep.png")


# ======================== ALPHA vs GEN GAP ========================
def make_alpha_gap_plot(df, output_dir):
    """
    alpha vs final_gen_gap across all four datasets (2x2 grid).
    Same layout as Fig 1 / Fig 2 so that every metric has a companion panel.
    Distill-S curve annotated with delta vs Baseline-Half and Wilcoxon stars.
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = sorted(df['dataset'].unique())
    groups_all = ['Baseline-Half', 'Baseline-Full', 'Baseline-LS', 'Distill-S']
    colors = {
        'Baseline-Half': '#2ca02c',
        'Baseline-Full': '#1f77b4',
        'Baseline-LS': '#ff7f0e',
        'Distill-S': '#d62728',
    }
    markers = {
        'Baseline-Half': 's',
        'Baseline-Full': '^',
        'Baseline-LS': 'D',
        'Distill-S': 'o',
    }
    linestyles = {
        'Baseline-Half': '--',
        'Baseline-Full': ':',
        'Baseline-LS': '-.',
        'Distill-S': '-',
    }

    agg = df.groupby(['dataset', 'alpha', 'group']).agg(
        mean_final_gen_gap=('final_gen_gap', 'mean'),
        std_final_gen_gap=('final_gen_gap', 'std'),
    ).reset_index()

    alphas = sorted(df['alpha'].unique())
    n_ds = len(datasets)
    n_cols = 4
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.2))
    axes = axes.flatten()

    for ax, ds in zip(axes, datasets):
        ds_agg = agg[agg['dataset'] == ds]
        for grp in groups_all:
            gd = ds_agg[ds_agg['group'] == grp].sort_values('alpha')
            ax.errorbar(gd['alpha'], gd['mean_final_gen_gap'],
                        yerr=gd['std_final_gen_gap'],
                        label=grp,
                        color=colors[grp],
                        marker=markers[grp],
                        linestyle=linestyles[grp],
                        capsize=3, markersize=5, linewidth=2.0)

        ds_distill = ds_agg[ds_agg['group'] == 'Distill-S'].sort_values('alpha')
        y_positions = ds_distill['mean_final_gen_gap'].values
        yerrs = ds_distill['std_final_gen_gap'].values
        
        ymin_cur, ymax_cur = ax.get_ylim()
        yrange = ymax_cur - ymin_cur
        rel_offset = 0.18 * yrange
        
        for i, alpha in enumerate(ds_distill['alpha'].values):
            bh_gap = ds_agg[(ds_agg['group'] == 'Baseline-Half') &
                            (ds_agg['alpha'] == alpha)]['mean_final_gen_gap'].values
            _, p = wilcoxon_test(df, ds, alpha, 'Distill-S', 'Baseline-Half', 'final_gen_gap')
            stars = _sig_stars(p)
            if len(bh_gap) > 0:
                delta = y_positions[i] - bh_gap[0]
                sign = '+' if delta >= 0 else ''
                ax.annotate(f'{sign}{delta:.3f}',
                            xy=(alpha, y_positions[i] + yerrs[i] + 0.5 * rel_offset),
                            fontsize=8, color=colors['Distill-S'], ha='center',
                            va='bottom')
            if stars != 'n.s.':
                ax.annotate(stars,
                            xy=(alpha, y_positions[i] + yerrs[i] + rel_offset),
                            fontsize=9, fontweight='bold',
                            color=colors['Distill-S'], ha='center')

        ax.set_xlabel('alpha (overlap)', fontsize=10)
        ax.set_ylabel('Final Generalization Gap', fontsize=10)
        ax.set_title(ds)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        
        _, ymax_after = ax.get_ylim()
        ax.set_ylim(ymin_cur, ymax_after * 1.24)

    for ax in axes[n_ds:]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], color=colors[g], marker=markers[g],
                          linestyle=linestyles[g], linewidth=2, label=g)
               for g in groups_all]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               ncol=4, fontsize=10, frameon=False)

    fig.suptitle('alpha vs Final Generalization Gap  (* p<.05  ** p<.01  *** p<.001 vs Baseline-Half)',
                 fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(output_dir, 'fig_alpha_vs_gen_gap.png'),
                dpi=CONFIG['dpi'])
    plt.close(fig)
    print("  [Saved] fig_alpha_vs_gen_gap.png")


# ======================== MAIN ========================
def _load_uci(name):
    """Load a UCI dataset via fetch_openml and label-encode categorical targets."""
    X, y = fetch_openml(name, version=1, return_X_y=True, parser='auto')
    if y.dtype.name == 'category' or y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    X = X.to_numpy(dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    return X, y


def main():
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()
    glass_X, glass_y = _load_uci('glass')

    datasets = {
        'Wine':         (wine.data, wine.target),
        'BreastCancer': (cancer.data, cancer.target),
        'Digits':       (digits.data, digits.target),
        'Glass':        (glass_X, glass_y),
    }

    all_dfs = []
    all_curves_combined = {}

    for ds_name, (X, y) in datasets.items():
        df, curves = run_all_experiments(ds_name, X, y)
        all_dfs.append(df)
        all_curves_combined.update(curves)

    full_df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(output_dir, 'results.csv')
    full_df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    print(f"\n=== Results Summary (mean ± std across {len(CONFIG['seeds'])} seeds) ===")
    summary = full_df.groupby(['dataset', 'alpha', 'group']).agg(
        e_star_mean=('e_star', 'mean'),
        e_star_std=('e_star', 'std'),
        best_val_acc_mean=('best_val_acc', 'mean'),
        best_val_acc_std=('best_val_acc', 'std'),
        final_gen_gap_mean=('final_gen_gap', 'mean'),
        final_gen_gap_std=('final_gen_gap', 'std'),
    ).round(4)
    print(summary.to_string())

    # ======== Wilcoxon significance summary ========
    print(f"\n=== Wilcoxon Significance: Distill-S vs Baseline-Half (p-values) ===")
    sig_rows = []
    for ds_name in datasets.keys():
        for alpha in CONFIG['alpha_values']:
            _, p_acc = wilcoxon_test(full_df, ds_name, alpha, 'Distill-S', 'Baseline-Half', 'best_val_acc')
            _, p_e = wilcoxon_test(full_df, ds_name, alpha, 'Distill-S', 'Baseline-Half', 'e_star')
            _, p_gap = wilcoxon_test(full_df, ds_name, alpha, 'Distill-S', 'Baseline-Half', 'final_gen_gap')
            sig_rows.append({
                'dataset': ds_name, 'alpha': alpha,
                'p_best_val_acc': round(p_acc, 4) if not np.isnan(p_acc) else '-',
                'p_e_star': round(p_e, 4) if not np.isnan(p_e) else '-',
                'p_final_gen_gap': round(p_gap, 4) if not np.isnan(p_gap) else '-',
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_path = os.path.join(output_dir, 'wilcoxon_pvalues.csv')
    sig_df.to_csv(sig_path, index=False)
    print(sig_df.to_string(index=False))
    print(f"\n[Saved] {sig_path}")

    # ======== Wilcoxon: Distill-S vs Baseline-LS ========
    print(f"\n=== Wilcoxon Significance: Distill-S vs Baseline-LS (p-values) ===")
    sig_ls_rows = []
    for ds_name in datasets.keys():
        for alpha in CONFIG['alpha_values']:
            _, p_acc = wilcoxon_test(full_df, ds_name, alpha, 'Distill-S', 'Baseline-LS', 'best_val_acc')
            _, p_e = wilcoxon_test(full_df, ds_name, alpha, 'Distill-S', 'Baseline-LS', 'e_star')
            _, p_gap = wilcoxon_test(full_df, ds_name, alpha, 'Distill-S', 'Baseline-LS', 'final_gen_gap')
            sig_ls_rows.append({
                'dataset': ds_name, 'alpha': alpha,
                'p_best_val_acc': round(p_acc, 4) if not np.isnan(p_acc) else '-',
                'p_e_star': round(p_e, 4) if not np.isnan(p_e) else '-',
                'p_final_gen_gap': round(p_gap, 4) if not np.isnan(p_gap) else '-',
            })
    sig_ls_df = pd.DataFrame(sig_ls_rows)
    sig_ls_path = os.path.join(output_dir, 'wilcoxon_pvalues_ls.csv')
    sig_ls_df.to_csv(sig_ls_path, index=False)
    print(sig_ls_df.to_string(index=False))
    print(f"\n[Saved] {sig_ls_path}")

    make_plots(full_df, all_curves_combined, output_dir)
    make_alpha_gap_plot(full_df, output_dir)

    # ======== BETA EXPERIMENT: data allocation ratio ========
    print(f"\n{'#'*60}")
    print(f"BETA SWEEP — Glass & BreastCancer only")
    print(f"alpha = 0.0 (disjoint partitions), sweeping beta")
    print(f"Each beta: |D_T| = int(beta * |pool|), D_S = rest of pool")
    print(f"{'#'*60}")

    beta_datasets = {
        'BreastCancer': (cancer.data, cancer.target),
        'Glass': (glass_X, glass_y),
    }

    beta_dfs = []
    for ds_name, (X, y) in beta_datasets.items():
        df_b, curves_b = run_beta_experiment(ds_name, X, y)
        beta_dfs.append(df_b)

    df_beta = pd.concat(beta_dfs, ignore_index=True)
    beta_csv = os.path.join(output_dir, 'results_beta.csv')
    df_beta.to_csv(beta_csv, index=False)
    print(f"\n[Saved] {beta_csv}")

    print(f"\n=== Beta Sweep Summary ===")
    beta_summary = df_beta.groupby(['dataset', 'beta', 'group']).agg(
        best_val_acc_mean=('best_val_acc', 'mean'),
        best_val_acc_std=('best_val_acc', 'std'),
        final_gen_gap_mean=('final_gen_gap', 'mean'),
        final_gen_gap_std=('final_gen_gap', 'std'),
    ).round(4)
    print(beta_summary.to_string())

    # Wilcoxon on beta: Distill-S vs Baseline-Student at each beta
    print(f"\n=== Beta Wilcoxon: Distill-S vs Baseline-Student (per-beta paired control) ===")
    for ds_name in beta_datasets.keys():
        for beta in CONFIG['beta_values']:
            vals_ds = df_beta[(df_beta['dataset'] == ds_name) & (df_beta['beta'] == beta) & (df_beta['group'] == 'Distill-S')]['best_val_acc'].values
            vals_bs = df_beta[(df_beta['dataset'] == ds_name) & (df_beta['beta'] == beta) & (df_beta['group'] == 'Baseline-Student')]['best_val_acc'].values
            vals_ds_gap = df_beta[(df_beta['dataset'] == ds_name) & (df_beta['beta'] == beta) & (df_beta['group'] == 'Distill-S')]['final_gen_gap'].values
            vals_bs_gap = df_beta[(df_beta['dataset'] == ds_name) & (df_beta['beta'] == beta) & (df_beta['group'] == 'Baseline-Student')]['final_gen_gap'].values
            diffs_acc = vals_ds - vals_bs
            diffs_gap = vals_ds_gap - vals_bs_gap
            p_acc = np.nan
            p_gap = np.nan
            if len(vals_ds) >= 3 and not np.allclose(diffs_acc, 0):
                _, p_acc = wilcoxon(vals_ds, vals_bs)
            if len(vals_ds) >= 3 and not np.allclose(diffs_gap, 0):
                _, p_gap = wilcoxon(vals_ds_gap, vals_bs_gap)
            stars_acc = _sig_stars(p_acc) if not np.isnan(p_acc) else 'n.s.'
            stars_gap = _sig_stars(p_gap) if not np.isnan(p_gap) else 'n.s.'
            flag = '  ***' if stars_acc != 'n.s.' or stars_gap != 'n.s.' else ''
            p_a = f"{p_acc:.4f}" if not np.isnan(p_acc) else ' - '
            p_g = f"{p_gap:.4f}" if not np.isnan(p_gap) else ' - '
            print(f"  {ds_name:12s}  beta={beta:.1f}  p_acc={p_a} {stars_acc}   p_gap={p_g} {stars_gap}{flag}")

    make_beta_plots(df_beta, output_dir)

    print("\nAll experiments completed!")


def plot_from_cache():
    """Load results.csv + all_curves.pkl and regenerate all plots."""
    output_dir = CONFIG['output_dir']
    import pickle
    df_path = os.path.join(output_dir, 'results.csv')
    pkl_path = os.path.join(output_dir, 'all_curves.pkl')
    if not os.path.exists(df_path) or not os.path.exists(pkl_path):
        print(f"Cache not ready — run main() first. Missing: {df_path} or {pkl_path}")
        return
    df = pd.read_csv(df_path)
    with open(pkl_path, 'rb') as f:
        all_curves = pickle.load(f)
    print(f"Loaded {len(df)} rows, {len(all_curves)} curves from cache.")
    make_plots(df, all_curves, output_dir)
    make_alpha_gap_plot(df, output_dir)
    beta_csv = os.path.join(output_dir, 'results_beta.csv')
    if os.path.exists(beta_csv):
        df_beta = pd.read_csv(beta_csv)
        make_beta_plots(df_beta, output_dir)
    print("Plots regenerated.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--plot':
        plot_from_cache()
    else:
        main()
