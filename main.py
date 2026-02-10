"""Complete pipeline for EEG electrode ablation study on SEED-IV."""

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm  # rebind to tqdm.notebook below if --notebook

from config import (
    CHANNEL_NAMES, DATA_ROOT, EMOTIV_EPOC, HEMISPHERES, MNE_NAME_MAP,
    MUSE_APPROX, N_BANDS, N_CHANNELS, N_CLASSES, N_SESSIONS, N_SUBJECTS,
    REGIONS_FINE, SESSION_LABELS, STANDARD_1020,
)
from models import ChannelAttentionEEGNet, MLPBaseline

# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_seed4_session(mat_path, session_idx):
    """Load one session's DE features from a SEED-IV .mat file.

    Args:
        mat_path: path to the .mat file
        session_idx: 0-based session index (for label lookup)
    Returns: X (n_samples, 62, 5), y (n_samples,), trial_ids (n_samples,)
    """
    data = sio.loadmat(mat_path)
    labels = SESSION_LABELS[session_idx]
    X_list, y_list, trial_id_list = [], [], []
    for trial_idx in range(24):
        key = f'de_LDS{trial_idx + 1}'
        if key not in data:
            continue
        trial_data = data[key]                       # (62, T, 5)
        trial_data = trial_data.transpose(1, 0, 2)  # -> (T, 62, 5)
        n_t = trial_data.shape[0]
        X_list.append(trial_data)
        y_list.append(np.full(n_t, labels[trial_idx]))
        trial_id_list.append(np.full(n_t, trial_idx))
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    trial_ids = np.concatenate(trial_id_list, axis=0).astype(np.int64)
    return X, y, trial_ids


# ────────────────────────────────────────────────────────────────────
# Masking utility
# ────────────────────────────────────────────────────────────────────

def make_channel_mask(active_indices, n_channels=N_CHANNELS, batch_size=1):
    """Create a binary mask: 1.0 = keep, 0.0 = ablated."""
    mask = torch.zeros(batch_size, n_channels)
    mask[:, active_indices] = 1.0
    return mask


# ────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        if isinstance(model, MLPBaseline):
            logits = model(X_batch.view(X_batch.size(0), -1))
        else:
            logits = model(X_batch)[0]
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, channel_mask=None):
    model.eval()
    correct, total = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if isinstance(model, MLPBaseline):
            logits = model(X_batch.view(X_batch.size(0), -1))
        else:
            B = X_batch.size(0)
            mask = channel_mask.expand(B, -1).to(device) if channel_mask is not None else None
            logits = model(X_batch, channel_mask=mask)[0]
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / total


# ────────────────────────────────────────────────────────────────────
# Cross-validation (Phase 1)
# ────────────────────────────────────────────────────────────────────

def cross_validate(X, y, model_cls, model_kwargs, train_kwargs, k=5, device='cuda',
                   groups=None):
    """Group k-fold CV (trial-level splits). Returns (mean_acc, std_acc)."""
    gkf = GroupKFold(n_splits=k)
    fold_accs = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr = torch.tensor(X[train_idx])
        y_tr = torch.tensor(y[train_idx])
        X_va = torch.tensor(X[val_idx])
        y_va = torch.tensor(y[val_idx])
        tr_loader = DataLoader(TensorDataset(X_tr, y_tr),
                               batch_size=train_kwargs['batch_size'], shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va, y_va),
                               batch_size=512, shuffle=False)
        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_kwargs['lr'],
                                     weight_decay=train_kwargs['wd'])
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        epoch_bar = tqdm(range(train_kwargs['max_epochs']),
                         desc=f'    Fold {fold+1}/{k}', leave=False)
        for epoch in epoch_bar:
            _, train_acc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
            val_acc = evaluate(model, va_loader, device)
            epoch_bar.set_postfix(train=f'{train_acc:.4f}', val=f'{val_acc:.4f}', best=f'{best_val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= train_kwargs['patience']:
                    break
        epoch_bar.close()
        model.load_state_dict(best_state)
        fold_accs.append(evaluate(model, va_loader, device))
    return float(np.mean(fold_accs)), float(np.std(fold_accs))


# ────────────────────────────────────────────────────────────────────
# Per-subject train & evaluate (Phase 2)
# ────────────────────────────────────────────────────────────────────

def train_and_evaluate(data, model_cls, model_kwargs, train_kwargs, device='cuda'):
    """Per-subject: train on sessions 1+2, test on session 3 with early stopping.

    Returns: (per_subject_accs dict, trained_models dict)
    """
    results = {}
    models = {}
    subj_bar = tqdm(range(1, N_SUBJECTS + 1), desc='Subjects')
    for subj in subj_bar:
        X_train_full = np.concatenate([data[subj][1][0], data[subj][2][0]])
        y_train_full = np.concatenate([data[subj][1][1], data[subj][2][1]])
        # Trial IDs with offset so session 2 trials are unique from session 1
        groups_full = np.concatenate([data[subj][1][2],
                                      data[subj][2][2] + 24])
        X_test, y_test, _ = data[subj][3]
        # Trial-level train/val split for early stopping (no temporal leakage)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(gss.split(X_train_full, y_train_full, groups=groups_full))
        X_tr, X_va = X_train_full[tr_idx], X_train_full[va_idx]
        y_tr, y_va = y_train_full[tr_idx], y_train_full[va_idx]
        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=train_kwargs['batch_size'], shuffle=True)
        va_loader = DataLoader(
            TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
            batch_size=512, shuffle=False)
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=512, shuffle=False)
        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_kwargs['lr'],
                                     weight_decay=train_kwargs['wd'])
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        epoch_bar = tqdm(range(train_kwargs['max_epochs']),
                         desc=f'  S{subj:02d} epochs', leave=False)
        for epoch in epoch_bar:
            _, train_acc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
            val_acc = evaluate(model, va_loader, device)
            epoch_bar.set_postfix(train=f'{train_acc:.4f}', val=f'{val_acc:.4f}', best=f'{best_val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= train_kwargs['patience']:
                    break
        epoch_bar.close()
        model.load_state_dict(best_state)
        test_acc = evaluate(model, te_loader, device)
        results[subj] = test_acc
        models[subj] = model
        subj_bar.set_postfix(last_test=f'{test_acc:.4f}')
        tqdm.write(f"  Subject {subj:2d}: val={best_val_acc:.4f}  test={test_acc:.4f}")
    accs = list(results.values())
    print(f"  Mean: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    return results, models


# ────────────────────────────────────────────────────────────────────
# Per-emotion importance (Phase 5 visualization helper)
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_emotion_importance(model, loader, device, n_classes=N_CLASSES):
    """Attention weights grouped by emotion class. Returns {class_id: (62,) array}."""
    model.eval()
    class_alphas = {c: [] for c in range(n_classes)}
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        _, alpha = model(X_batch)
        alpha = alpha.cpu()
        for c in range(n_classes):
            mask_c = (y_batch == c)
            if mask_c.any():
                class_alphas[c].append(alpha[mask_c])
    return {c: torch.cat(class_alphas[c], dim=0).mean(dim=0).numpy()
            for c in range(n_classes) if class_alphas[c]}


# ────────────────────────────────────────────────────────────────────
# Ablation study (Phase 4)
# ────────────────────────────────────────────────────────────────────

def _eval_all_subjects(models, data, mask, device):
    """Evaluate all subjects with a given channel mask. Returns list of 15 accuracies."""
    accs = []
    for subj in range(1, N_SUBJECTS + 1):
        X_test, y_test, _ = data[subj][3]
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=512, shuffle=False)
        accs.append(evaluate(models[subj], loader, device, channel_mask=mask))
    return accs


def run_full_ablation_study(models, data, grand_ranking, device='cuda'):
    """Run all ablation experiments. Returns dict of results."""
    all_results = {}
    all_ch = set(range(N_CHANNELS))

    # Region keep-only and remove
    for region_name, indices in REGIONS_FINE.items():
        mask_keep = make_channel_mask(indices, batch_size=1).to(device)
        accs_keep = _eval_all_subjects(models, data, mask_keep, device)
        all_results[f'keep_only_{region_name}'] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs_keep)), 'std': float(np.std(accs_keep)),
            'per_subj': accs_keep,
        }
        remaining = sorted(all_ch - set(indices))
        mask_rm = make_channel_mask(remaining, batch_size=1).to(device)
        accs_rm = _eval_all_subjects(models, data, mask_rm, device)
        all_results[f'remove_{region_name}'] = {
            'n_channels': len(remaining),
            'mean': float(np.mean(accs_rm)), 'std': float(np.std(accs_rm)),
            'per_subj': accs_rm,
        }
        print(f"  Region {region_name}: keep-only={np.mean(accs_keep):.4f}, "
              f"remove={np.mean(accs_rm):.4f}")

    # Hemisphere experiments
    for hemi_name, indices in HEMISPHERES.items():
        mask = make_channel_mask(indices, batch_size=1).to(device)
        accs = _eval_all_subjects(models, data, mask, device)
        all_results[f'hemisphere_{hemi_name}'] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_subj': accs,
        }
        print(f"  Hemisphere {hemi_name}: {np.mean(accs):.4f}")

    # Standard montage subsets
    montages = {
        'full_62': list(range(N_CHANNELS)),
        'standard_1020_19': STANDARD_1020,
        'emotiv_epoc_14': EMOTIV_EPOC,
        'muse_approx_4': MUSE_APPROX,
    }
    for mont_name, indices in montages.items():
        mask = make_channel_mask(indices, batch_size=1).to(device)
        accs = _eval_all_subjects(models, data, mask, device)
        all_results[f'montage_{mont_name}'] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_subj': accs,
        }
        print(f"  Montage {mont_name}: {np.mean(accs):.4f}")

    # Progressive ablation (attention-guided: least first, most first)
    n_keep_steps = [62, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 8, 5, 3, 1]
    for strategy_name, ranking in [
        ('attn_least_first', grand_ranking),
        ('attn_most_first', grand_ranking[::-1]),
    ]:
        curve = {}
        for n_keep in n_keep_steps:
            keep = ranking[:n_keep].tolist()
            mask = make_channel_mask(keep, batch_size=1).to(device)
            accs = _eval_all_subjects(models, data, mask, device)
            curve[n_keep] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
        all_results[f'progressive_{strategy_name}'] = curve
        print(f"  Progressive {strategy_name}: done")

    # Random ablation (20 repeats per step, averaged per subject first)
    curve_random = {}
    for n_keep in n_keep_steps:
        subj_means = np.zeros(N_SUBJECTS)
        for seed in range(20):
            rng = np.random.RandomState(seed)
            keep = rng.choice(N_CHANNELS, n_keep, replace=False).tolist()
            mask = make_channel_mask(keep, batch_size=1).to(device)
            subj_means += np.array(_eval_all_subjects(models, data, mask, device))
        subj_means /= 20
        curve_random[n_keep] = {
            'mean': float(np.mean(subj_means)), 'std': float(np.std(subj_means)),
        }
    all_results['progressive_random'] = curve_random
    print("  Progressive random: done")

    return all_results


# ────────────────────────────────────────────────────────────────────
# Visualization (Phase 5)
# ────────────────────────────────────────────────────────────────────

def plot_progressive_ablation_curves(results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    curves_to_plot = [
        ('progressive_attn_least_first', '#2ecc71', 'ChannelAttn (least first)', '-', 'o'),
        ('progressive_random',           '#95a5a6', 'ChannelAttn random',        '-', 'o'),
        ('progressive_attn_most_first',  '#e74c3c', 'ChannelAttn (most first)',  '-', 'o'),
    ]
    if 'dual_progressive_attn_least_first' in results:
        curves_to_plot.extend([
            ('dual_progressive_attn_least_first', '#2ecc71', 'DualAttn (least first)', '--', 's'),
            ('dual_progressive_random',           '#95a5a6', 'DualAttn random',        '--', 's'),
            ('dual_progressive_attn_most_first',  '#e74c3c', 'DualAttn (most first)',  '--', 's'),
        ])
    for key, color, label, ls, marker in curves_to_plot:
        curve = results[key]
        n_ch = sorted(curve.keys(), key=int)
        x_vals = [int(n) for n in n_ch]
        means = [curve[n]['mean'] for n in n_ch]
        stds = [curve[n]['std'] for n in n_ch]
        ax.plot(x_vals, means, marker=marker, color=color, label=label,
                linewidth=2, linestyle=ls)
        ax.fill_between(x_vals,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)
    ax.axhline(y=0.25, color='black', linestyle=':', alpha=0.5, label='Chance (4-class)')
    ax.set_xlabel('Number of Remaining Channels', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Progressive Electrode Ablation', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 65)
    ax.set_ylim(0.2, 0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('progressive_ablation.pdf', dpi=300, bbox_inches='tight')
    print("Saved progressive_ablation.pdf")
    plt.close()


def _make_topo_info():
    """Create MNE Info with valid montage positions for SEED-IV channels."""
    import mne
    mapped_names = [MNE_NAME_MAP.get(n, n) for n in CHANNEL_NAMES]
    info = mne.create_info(ch_names=mapped_names, sfreq=1, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage, on_missing='ignore')
    # Filter to channels that received valid montage positions
    picks = [i for i, ch in enumerate(info['chs'])
             if not np.allclose(ch['loc'][:3], 0)]
    if len(picks) < len(mapped_names):
        missing = [CHANNEL_NAMES[i] for i in range(len(CHANNEL_NAMES))
                   if i not in picks]
        print(f"  Warning: {len(missing)} channels without montage positions "
              f"excluded from topomap: {missing}")
    return mne.pick_info(info, picks), np.array(picks)


def plot_topographic_attention(grand_importance,
                               title='Channel Attention Weights (Grand Average)',
                               filename='topomap_attention.pdf'):
    import matplotlib.pyplot as plt
    import mne
    info, picks = _make_topo_info()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    mne.viz.plot_topomap(grand_importance[picks], info, axes=ax, show=False,
                         cmap='RdYlBu_r', contours=0)
    ax.set_title(title, fontsize=13)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def plot_region_ablation_table(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    regions = list(REGIONS_FINE.keys())

    # Keep-only
    ax = axes[0]
    means = [results[f'keep_only_{r}']['mean'] for r in regions]
    stds = [results[f'keep_only_{r}']['std'] for r in regions]
    n_chs = [results[f'keep_only_{r}']['n_channels'] for r in regions]
    labels = [f'{r}\n({n}ch)' for r, n in zip(regions, n_chs)]
    ax.bar(range(len(regions)), means, yerr=stds, capsize=3,
           color='#3498db', alpha=0.8)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Keep Only This Region')
    ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.5)

    # Remove
    ax = axes[1]
    means = [results[f'remove_{r}']['mean'] for r in regions]
    stds = [results[f'remove_{r}']['std'] for r in regions]
    n_chs = [results[f'remove_{r}']['n_channels'] for r in regions]
    labels = [f'w/o {r}\n({n}ch)' for r, n in zip(regions, n_chs)]
    ax.bar(range(len(regions)), means, yerr=stds, capsize=3,
           color='#e67e22', alpha=0.8)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Remove This Region')
    full_acc = results['montage_full_62']['mean']
    ax.axhline(y=full_acc, color='green', linestyle='--', alpha=0.7,
               label=f'Full 62ch ({full_acc:.3f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('region_ablation.pdf', dpi=300, bbox_inches='tight')
    print("Saved region_ablation.pdf")
    plt.close()


def plot_per_emotion_topomap(emotion_importances):
    import matplotlib.pyplot as plt
    import mne

    info, picks = _make_topo_info()
    emotion_labels = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(5 * N_CLASSES, 5))
    for c in range(N_CLASSES):
        mne.viz.plot_topomap(emotion_importances[c][picks], info, axes=axes[c],
                             show=False, cmap='RdYlBu_r', contours=0)
        axes[c].set_title(emotion_labels.get(c, str(c)), fontsize=13)
    plt.suptitle('Per-Emotion Channel Attention', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('per_emotion_topomap.pdf', dpi=300, bbox_inches='tight')
    print("Saved per_emotion_topomap.pdf")
    plt.close()


# ────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Electrode Ablation Study on SEED-IV')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT,
                        help='Path to SEED-IV eeg_feature_smooth directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    parser.add_argument('--skip_search', action='store_true',
                        help='Skip HP search, use defaults')
    parser.add_argument('--notebook', action='store_true',
                        help='Use tqdm.notebook for Colab/Jupyter')
    args = parser.parse_args()

    if args.notebook:
        from tqdm.notebook import tqdm  # noqa: F811 — rebinds module-level tqdm

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Phase 0: Load all data with per-session z-score normalization ──
    print("\n=== Phase 0: Loading data ===")
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    data = {}
    for sess in range(1, N_SESSIONS + 1):
        sess_dir = data_root / str(sess)
        if not sess_dir.is_dir():
            raise FileNotFoundError(
                f"Session directory not found: {sess_dir}\n"
                f"Expected {data_root}/1/, {data_root}/2/, {data_root}/3/")
        mat_files = sorted(sess_dir.glob('*.mat'),
                           key=lambda p: int(p.stem.split('_')[0]))
        for subj_idx, mat_path in enumerate(mat_files):
            subj = subj_idx + 1
            if subj not in data:
                data[subj] = {}
            X, y, trial_ids = load_seed4_session(str(mat_path), session_idx=sess - 1)
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std
            data[subj][sess] = (X, y, trial_ids)
    for subj in range(1, N_SUBJECTS + 1):
        sizes = [data[subj][s][0].shape[0] for s in range(1, N_SESSIONS + 1)]
        print(f"  Subject {subj:2d}: {sizes} samples per session")

    # Pool sessions 1+2 for HP search (trial IDs offset to be globally unique)
    X_pool = np.concatenate([data[s][sess][0]
                             for s in range(1, N_SUBJECTS + 1)
                             for sess in [1, 2]])
    y_pool = np.concatenate([data[s][sess][1]
                             for s in range(1, N_SUBJECTS + 1)
                             for sess in [1, 2]])
    trial_pool = np.concatenate([data[s][sess][2] + s * 1000 + sess * 100
                                 for s in range(1, N_SUBJECTS + 1)
                                 for sess in [1, 2]])
    print(f"  Pooled shape: {X_pool.shape}, labels: {np.bincount(y_pool)}")

    # ── Phase 1: Hyperparameter search ──
    if args.skip_search:
        best_model_kwargs = {'n_bands': N_BANDS, 'n_classes': N_CLASSES,
                             'd_hidden': 64, 'dropout': 0.3, 'n_heads': 4,
                             'dim_feedforward': 128}
        best_train_kwargs = {'lr': 5e-4, 'wd': 1e-4, 'batch_size': 128,
                             'max_epochs': 200, 'patience': 10}
        best_mlp_kwargs = {'input_dim': N_CHANNELS * N_BANDS, 'n_classes': N_CLASSES,
                           'h1': 128, 'h2': 64, 'dropout': 0.5}
        best_mlp_train_kwargs = {'lr': 5e-4, 'wd': 1e-4, 'batch_size': 128,
                                 'max_epochs': 200, 'patience': 10}
        print("\n=== Phase 1: Skipped (using defaults) ===")
    else:
        print("\n=== Phase 1a: Attention HP search (5-fold CV) ===")
        search_space = {
            'd_hidden': [64],
            'dim_feedforward': [64, 128],
            'dropout':  [0.3, 0.5],
            'lr':       [5e-4, 1e-4],
            'wd':       [1e-4, 5e-4],
            'batch_size': [64, 128],
        }
        best_score = 0.0
        best_model_kwargs = None
        best_train_kwargs = None
        configs_1a = list(itertools.product(
            search_space['d_hidden'], search_space['dim_feedforward'],
            search_space['dropout'],
            search_space['lr'], search_space['wd'], search_space['batch_size'],
        ))
        pbar_1a = tqdm(configs_1a, desc='Attention HP search')
        for d_h, d_ff, drop, lr, wd, bs in pbar_1a:
            mk = {'n_bands': N_BANDS, 'n_classes': N_CLASSES,
                   'd_hidden': d_h, 'dropout': drop, 'n_heads': 4,
                   'dim_feedforward': d_ff}
            tk = {'lr': lr, 'wd': wd, 'batch_size': bs,
                  'max_epochs': 200, 'patience': 10}
            mean_acc, std_acc = cross_validate(X_pool, y_pool,
                                               ChannelAttentionEEGNet, mk, tk,
                                               device=device, groups=trial_pool)
            if mean_acc > best_score:
                best_score = mean_acc
                best_model_kwargs = mk
                best_train_kwargs = tk
            pbar_1a.set_postfix(best=f'{best_score:.4f}', last=f'{mean_acc:.4f}')
        print(f"Best CV Acc: {best_score:.4f}")
        print(f"Best config: {best_model_kwargs}, {best_train_kwargs}")

        # Phase 1b: MLP HP search
        print("\n=== Phase 1b: MLP HP search (5-fold CV) ===")
        mlp_search_space = {
            'h1': [128, 256], 'h2': [64],
            'dropout': [0.3, 0.5], 'lr': [5e-4],
            'wd': [1e-4], 'batch_size': [128],
        }
        best_mlp_score = 0.0
        best_mlp_kwargs = None
        best_mlp_train_kwargs = None
        configs_1b = list(itertools.product(
            mlp_search_space['h1'], mlp_search_space['h2'],
            mlp_search_space['dropout'], mlp_search_space['lr'],
            mlp_search_space['wd'], mlp_search_space['batch_size'],
        ))
        pbar_1b = tqdm(configs_1b, desc='MLP HP search')
        for h1, h2, drop, lr, wd, bs in pbar_1b:
            mk = {'input_dim': N_CHANNELS * N_BANDS, 'n_classes': N_CLASSES,
                   'h1': h1, 'h2': h2, 'dropout': drop}
            tk = {'lr': lr, 'wd': wd, 'batch_size': bs,
                  'max_epochs': 200, 'patience': 10}
            mean_acc, std_acc = cross_validate(X_pool, y_pool,
                                               MLPBaseline, mk, tk,
                                               device=device, groups=trial_pool)
            if mean_acc > best_mlp_score:
                best_mlp_score = mean_acc
                best_mlp_kwargs = mk
                best_mlp_train_kwargs = tk
            pbar_1b.set_postfix(best=f'{best_mlp_score:.4f}', last=f'{mean_acc:.4f}')
        print(f"Best MLP CV Acc: {best_mlp_score:.4f}")
        print(f"Best MLP config: {best_mlp_kwargs}, {best_mlp_train_kwargs}")

    # ── Phase 2: Per-subject train/test ──
    print("\n=== Phase 2a: ChannelAttentionEEGNet ===")
    attn_results, attn_models = train_and_evaluate(
        data, ChannelAttentionEEGNet, best_model_kwargs, best_train_kwargs, device)

    print("\n=== Phase 2b: MLPBaseline ===")
    mlp_results, mlp_models = train_and_evaluate(
        data, MLPBaseline, best_mlp_kwargs, best_mlp_train_kwargs, device)

    # ── Phase 3: Extract channel importance ──
    print("\n=== Phase 3: Channel importance extraction ===")
    importances = np.zeros((N_SUBJECTS, N_CHANNELS))
    for subj in range(1, N_SUBJECTS + 1):
        X_test, y_test, _ = data[subj][3]
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=512, shuffle=False)
        mean_alpha, _ = attn_models[subj].get_channel_importance(loader, device)
        importances[subj - 1] = mean_alpha
    grand_importance = importances.mean(axis=0)
    grand_ranking = grand_importance.argsort()[::-1].copy()
    print("  Top-10 channels:", [CHANNEL_NAMES[i] for i in grand_ranking[:10]])

    # ── Phase 4: Full ablation study ──
    print("\n=== Phase 4: Ablation study ===")
    ablation_results = run_full_ablation_study(
        attn_models, data, grand_ranking, device)

    # ── Phase 5: Visualization & statistics ──
    print("\n=== Phase 5: Visualization ===")
    plot_progressive_ablation_curves(ablation_results)
    plot_topographic_attention(grand_importance)
    plot_region_ablation_table(ablation_results)

    # Per-emotion attention maps (grand average across subjects)
    print("Computing per-emotion attention maps...")
    grand_emotion_imp = {c: np.zeros(N_CHANNELS) for c in range(N_CLASSES)}
    for subj in range(1, N_SUBJECTS + 1):
        X_test, y_test, _ = data[subj][3]
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=512, shuffle=False)
        subj_emo = per_emotion_importance(attn_models[subj], loader, device)
        for c in subj_emo:
            grand_emotion_imp[c] += subj_emo[c]
    for c in grand_emotion_imp:
        grand_emotion_imp[c] /= N_SUBJECTS
    plot_per_emotion_topomap(grand_emotion_imp)

    # Wilcoxon signed-rank tests with Holm-Bonferroni correction
    print("\n=== Statistical tests (Wilcoxon, Holm-Bonferroni corrected) ===")
    from scipy import stats
    comparisons = [
        ('montage_full_62', 'montage_standard_1020_19'),
        ('montage_full_62', 'montage_emotiv_epoc_14'),
        ('montage_full_62', 'montage_muse_approx_4'),
        ('hemisphere_left', 'hemisphere_right'),
    ]
    raw_p_values = []
    test_rows = []
    for cfg_a, cfg_b in comparisons:
        accs_a = ablation_results[cfg_a]['per_subj']
        accs_b = ablation_results[cfg_b]['per_subj']
        stat, p = stats.wilcoxon(accs_a, accs_b)
        raw_p_values.append(p)
        test_rows.append((cfg_a, cfg_b, stat, p))
    # Holm-Bonferroni correction
    n_tests = len(raw_p_values)
    sorted_idx = np.argsort(raw_p_values)
    adjusted_p = np.ones(n_tests)
    for rank, idx in enumerate(sorted_idx):
        adjusted_p[idx] = raw_p_values[idx] * (n_tests - rank)
    for i in range(1, n_tests):
        adjusted_p[sorted_idx[i]] = max(adjusted_p[sorted_idx[i]],
                                         adjusted_p[sorted_idx[i - 1]])
    adjusted_p = np.minimum(adjusted_p, 1.0)
    for i, (cfg_a, cfg_b, stat, raw_p) in enumerate(test_rows):
        adj_p = adjusted_p[i]
        sig = '***' if adj_p < 0.001 else '**' if adj_p < 0.01 else '*' if adj_p < 0.05 else 'n.s.'
        print(f"  {cfg_a} vs {cfg_b}: p={raw_p:.4f} (adj={adj_p:.4f}) {sig}")

    # Save all results
    save_results = {
        'attention_per_subject': {str(k): v for k, v in attn_results.items()},
        'mlp_per_subject': {str(k): v for k, v in mlp_results.items()},
        'attention_mean': float(np.mean(list(attn_results.values()))),
        'mlp_mean': float(np.mean(list(mlp_results.values()))),
        'grand_ranking': grand_ranking.tolist(),
        'grand_importance': grand_importance.tolist(),
        'ablation': ablation_results,
        'best_model_kwargs': best_model_kwargs,
        'best_train_kwargs': best_train_kwargs,
        'best_mlp_kwargs': best_mlp_kwargs,
        'best_mlp_train_kwargs': best_mlp_train_kwargs,
    }
    with open('results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print("\nSaved results.json")
    print("Done.")
