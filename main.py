import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm  # rebind to tqdm.notebook below if --notebook

from config import (
    CHANNEL_NAMES, DATA_ROOT, EMOTIV_EPOC, HEMISPHERES, LOBES, MNE_NAME_MAP,
    MUSE_APPROX, N_BANDS, N_CHANNELS, N_CLASSES, N_SESSIONS, N_SUBJECTS,
    REGIONS_FINE, SESSION_LABELS, STANDARD_1020, T_FIXED,
)
from models import SOGNN


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_seed4_session(mat_path, session_idx):
    """Load one session's DE features from a SEED-IV .mat file.

    Each trial is one sample, zero-padded to T_FIXED and z-score normalized
    (statistics computed from all frames before padding).

    Args:
        mat_path: path to the .mat file
        session_idx: 0-based session index (for label lookup)
    Returns: X (24, 62, 5, T_FIXED), y (24,), trial_ids (24,)
    """
    data = sio.loadmat(mat_path)
    labels = SESSION_LABELS[session_idx]
    trials_raw = []  # list of (T_k, 62, 5) arrays
    y_list = []
    for trial_idx in range(24):
        key = f'de_LDS{trial_idx + 1}'
        if key not in data:
            continue
        trial_data = data[key]                       # (62, T_k, 5)
        trial_data = trial_data.transpose(1, 0, 2)  # -> (T_k, 62, 5)
        trials_raw.append(trial_data)
        y_list.append(labels[trial_idx])
    # Compute z-score statistics from ALL frames before padding
    all_frames = np.concatenate(trials_raw, axis=0)  # (N_total, 62, 5)
    mean = all_frames.mean(axis=0, keepdims=True)    # (1, 62, 5)
    std = all_frames.std(axis=0, keepdims=True) + 1e-8
    # Normalize each trial, zero-pad, and transpose to (62, 5, T_FIXED)
    X_list = []
    for trial in trials_raw:
        trial_normed = (trial - mean) / std           # (T_k, 62, 5)
        T_k = trial_normed.shape[0]
        if T_k < T_FIXED:
            pad_width = ((0, T_FIXED - T_k), (0, 0), (0, 0))
            trial_padded = np.pad(trial_normed, pad_width, mode='constant')
        elif T_k > T_FIXED:
            trial_padded = trial_normed[:T_FIXED]
        else:
            trial_padded = trial_normed
        # (T_FIXED, 62, 5) -> (62, 5, T_FIXED)
        X_list.append(trial_padded.transpose(1, 2, 0))
    X = np.stack(X_list, axis=0).astype(np.float32)  # (24, 62, 5, T_FIXED)
    y = np.array(y_list, dtype=np.int64)
    trial_ids = np.arange(len(y_list), dtype=np.int64)
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
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
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
        if channel_mask is not None:
            mask = channel_mask.to(device).unsqueeze(-1).unsqueeze(-1)  # (1,C,1,1)
            X_batch = X_batch * mask
        logits = model(X_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / total


@torch.no_grad()
def compute_training_auc(model, loader, device):
    """Compute macro-averaged training AUC (OvR)."""
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')


# ────────────────────────────────────────────────────────────────────
# LOSO train & evaluate (Phase 2)
# ────────────────────────────────────────────────────────────────────

def train_and_evaluate(data, model_cls, model_kwargs, train_kwargs, device='cuda'):
    """LOSO: for each subject, train on all other subjects, test on held-out.

    Returns: (per_subject_accs dict, trained_models dict)
    """
    results = {}
    models = {}
    subj_bar = tqdm(range(1, N_SUBJECTS + 1), desc='LOSO folds')
    for test_subj in subj_bar:
        # Pool ALL sessions from all OTHER subjects
        X_train = np.concatenate([data[s][sess][0]
                                  for s in range(1, N_SUBJECTS + 1) if s != test_subj
                                  for sess in range(1, N_SESSIONS + 1)])
        y_train = np.concatenate([data[s][sess][1]
                                  for s in range(1, N_SUBJECTS + 1) if s != test_subj
                                  for sess in range(1, N_SESSIONS + 1)])
        # Pool ALL sessions from held-out subject
        X_test = np.concatenate([data[test_subj][sess][0]
                                 for sess in range(1, N_SESSIONS + 1)])
        y_test = np.concatenate([data[test_subj][sess][1]
                                 for sess in range(1, N_SESSIONS + 1)])

        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=train_kwargs['batch_size'], shuffle=True)
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=512, shuffle=False)

        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_kwargs['lr'],
                                     weight_decay=train_kwargs['wd'])
        criterion = nn.CrossEntropyLoss()

        auc_threshold = train_kwargs.get('auc_threshold', 0.99)
        train_auc = 0.0
        epoch_bar = tqdm(range(train_kwargs['max_epochs']),
                         desc=f'  S{test_subj:02d} epochs', leave=False)
        for epoch in epoch_bar:
            _, train_acc = train_one_epoch(model, tr_loader, optimizer,
                                           criterion, device)
            train_auc = compute_training_auc(model, tr_loader, device)
            epoch_bar.set_postfix(train=f'{train_acc:.4f}',
                                  auc=f'{train_auc:.4f}')
            if train_auc >= auc_threshold and train_acc > 0.90:
                break
        epoch_bar.close()

        test_acc = evaluate(model, te_loader, device)
        results[test_subj] = test_acc
        models[test_subj] = model
        subj_bar.set_postfix(last_test=f'{test_acc:.4f}')
        tqdm.write(f"  LOSO fold {test_subj:2d}: auc={train_auc:.4f}  "
                   f"test={test_acc:.4f}")
    accs = list(results.values())
    print(f"  Mean: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    return results, models


# ────────────────────────────────────────────────────────────────────
# Permutation importance (Phase 3 / Phase 5)
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def permutation_importance(model, X, y, device, n_repeats=10):
    """Compute permutation importance for each channel (accuracy-based).

    Shuffles each channel's feature block across samples and measures accuracy
    drop. Positive = important (baseline_acc - shuffled_acc).

    Args:
        model: trained SOGNN
        X: numpy array (N, C, 5, T_FIXED)
        y: numpy array (N,)
        device: torch device
        n_repeats: number of shuffle repeats per channel
    Returns: (C,) array, positive = important
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    N = X_t.size(0)
    baseline_acc = (model(X_t).argmax(1) == y_t).float().mean().item()
    n_channels = X.shape[1]
    acc_importances = np.zeros(n_channels)
    for ch in range(n_channels):
        shuffled_accs = []
        for _ in range(n_repeats):
            X_perm = X_t.clone()
            perm_idx = torch.randperm(N, device=device)
            X_perm[:, ch, :, :] = X_perm[perm_idx, ch, :, :]
            logits = model(X_perm)
            shuffled_accs.append((logits.argmax(1) == y_t).float().mean().item())
        acc_importances[ch] = baseline_acc - float(np.mean(shuffled_accs))
    return acc_importances


@torch.no_grad()
def per_emotion_permutation_importance(model, X, y, device,
                                        n_repeats=10, n_classes=N_CLASSES):
    """Permutation importance per emotion class.
    Returns {class_id: (62,) array}."""
    result = {}
    for c in range(n_classes):
        mask_c = (y == c)
        if mask_c.sum() == 0:
            continue
        result[c] = permutation_importance(model, X[mask_c], y[mask_c],
                                           device, n_repeats=n_repeats)
    return result


def integrated_gradients_importance(model, X, y, device, n_steps=50,
                                    batch_size=24):
    """Compute Integrated Gradients channel importance.

    Args:
        model: trained SOGNN
        X: numpy array (N, 62, 5, T_FIXED)
        y: numpy array (N,)
        device: torch device
        n_steps: interpolation steps (default 50)
        batch_size: batch size for forward/backward passes (default 24)
    Returns: (62,) array — per-channel importance (positive = important)
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)  # (N, 62, 5, T)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    N = X_t.shape[0]

    accum = torch.zeros_like(X_t)  # (N, 62, 5, T)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_interp = (alpha * X_t[start:end]).detach().requires_grad_(True)
            logits = model(x_interp)
            target_logits = logits.gather(1, y_t[start:end].unsqueeze(1)).squeeze(1)
            target_logits.sum().backward()
            accum[start:end] += x_interp.grad.detach()

    # IG = (input - baseline) * avg_grad; baseline=0 → input * avg_grad
    avg_grad = accum / n_steps       # (N, 62, 5, T)
    ig = X_t * avg_grad              # (N, 62, 5, T)

    # Per-channel: sum |IG| over bands and time, mean over samples
    channel_importance = ig.abs().sum(dim=(2, 3)).mean(dim=0)  # (62,)
    return channel_importance.cpu().numpy()


# ────────────────────────────────────────────────────────────────────
# Ablation study (Phase 4)
# ────────────────────────────────────────────────────────────────────

def _eval_all_subjects(models, data, mask, device):
    """Evaluate all LOSO folds with a given channel mask. Returns list of 15 accuracies."""
    accs = []
    for subj in range(1, N_SUBJECTS + 1):
        X_test = np.concatenate([data[subj][sess][0]
                                 for sess in range(1, N_SESSIONS + 1)])
        y_test = np.concatenate([data[subj][sess][1]
                                 for sess in range(1, N_SESSIONS + 1)])
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=512, shuffle=False)
        accs.append(evaluate(models[subj], loader, device, channel_mask=mask))
    return accs


def _retrain_loso(data, channel_indices, model_kwargs, train_kwargs, device):
    """LOSO retrain: for each fold, train on 14 subjects' selected channels,
    test on held-out subject. Returns list of 15 accuracies."""
    n_ch = len(channel_indices)
    ch_idx = list(channel_indices)
    mk = {**model_kwargs, 'n_electrodes': n_ch}
    accs = []
    for test_subj in range(1, N_SUBJECTS + 1):
        X_train = np.concatenate([data[s][sess][0][:, ch_idx, :, :]
                                  for s in range(1, N_SUBJECTS + 1)
                                  if s != test_subj
                                  for sess in range(1, N_SESSIONS + 1)])
        y_train = np.concatenate([data[s][sess][1]
                                  for s in range(1, N_SUBJECTS + 1)
                                  if s != test_subj
                                  for sess in range(1, N_SESSIONS + 1)])
        X_test = np.concatenate([data[test_subj][sess][0][:, ch_idx, :, :]
                                 for sess in range(1, N_SESSIONS + 1)])
        y_test = np.concatenate([data[test_subj][sess][1]
                                 for sess in range(1, N_SESSIONS + 1)])

        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=train_kwargs['batch_size'], shuffle=True)
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=512, shuffle=False)

        model = SOGNN(**mk).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_kwargs['lr'],
                                     weight_decay=train_kwargs['wd'])
        criterion = nn.CrossEntropyLoss()
        auc_threshold = train_kwargs.get('auc_threshold', 0.99)
        for epoch in range(train_kwargs['max_epochs']):
            _, train_acc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
            train_auc = compute_training_auc(model, tr_loader, device)
            if train_auc >= auc_threshold and train_acc > 0.90:
                break
        accs.append(evaluate(model, te_loader, device))
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

    # Lobe keep-only and remove
    for lobe_name, indices in LOBES.items():
        mask_keep = make_channel_mask(indices, batch_size=1).to(device)
        accs_keep = _eval_all_subjects(models, data, mask_keep, device)
        all_results[f'lobe_keep_only_{lobe_name}'] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs_keep)), 'std': float(np.std(accs_keep)),
            'per_subj': accs_keep,
        }
        remaining = sorted(all_ch - set(indices))
        mask_rm = make_channel_mask(remaining, batch_size=1).to(device)
        accs_rm = _eval_all_subjects(models, data, mask_rm, device)
        all_results[f'lobe_remove_{lobe_name}'] = {
            'n_channels': len(remaining),
            'mean': float(np.mean(accs_rm)), 'std': float(np.std(accs_rm)),
            'per_subj': accs_rm,
        }
        print(f"  Lobe {lobe_name}: keep-only={np.mean(accs_keep):.4f}, "
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

    # Progressive ablation (PI-guided: least first, most first)
    n_keep_steps = [62, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 8, 5, 3, 1]
    for strategy_name, ranking in [
        ('pi_least_first', grand_ranking),
        ('pi_most_first', grand_ranking[::-1]),
    ]:
        curve = {}
        for n_keep in n_keep_steps:
            keep = ranking[:n_keep].tolist()
            mask = make_channel_mask(keep, batch_size=1).to(device)
            accs = _eval_all_subjects(models, data, mask, device)
            curve[n_keep] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
                            'per_subj': accs}
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
            'per_subj': subj_means.tolist(),
        }
    all_results['progressive_random'] = curve_random
    print("  Progressive random: done")

    return all_results


def run_retrain_ablation_study(data, grand_ranking, model_kwargs, train_kwargs,
                               device='cuda'):
    """Retrain-from-scratch ablation: fresh LOSO model per config. Returns dict of results."""
    all_results = {}
    all_ch = set(range(N_CHANNELS))

    # Region keep-only and remove
    region_configs = []
    for region_name, indices in REGIONS_FINE.items():
        region_configs.append((f'keep_only_{region_name}', indices))
        remaining = sorted(all_ch - set(indices))
        region_configs.append((f'remove_{region_name}', remaining))
    region_bar = tqdm(region_configs, desc='  Retrain regions')
    for config_name, indices in region_bar:
        accs = _retrain_loso(data, indices, model_kwargs, train_kwargs, device)
        all_results[config_name] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_subj': accs,
        }
        region_bar.set_postfix(config=config_name, acc=f'{np.mean(accs):.4f}')

    # Lobe keep-only and remove
    lobe_configs = []
    for lobe_name, indices in LOBES.items():
        lobe_configs.append((f'lobe_keep_only_{lobe_name}', indices))
        remaining = sorted(all_ch - set(indices))
        lobe_configs.append((f'lobe_remove_{lobe_name}', remaining))
    lobe_bar = tqdm(lobe_configs, desc='  Retrain lobes')
    for config_name, indices in lobe_bar:
        accs = _retrain_loso(data, indices, model_kwargs, train_kwargs, device)
        all_results[config_name] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_subj': accs,
        }
        lobe_bar.set_postfix(config=config_name, acc=f'{np.mean(accs):.4f}')

    # Hemisphere experiments
    hemi_bar = tqdm(HEMISPHERES.items(), desc='  Retrain hemispheres')
    for hemi_name, indices in hemi_bar:
        accs = _retrain_loso(data, indices, model_kwargs, train_kwargs, device)
        all_results[f'hemisphere_{hemi_name}'] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_subj': accs,
        }
        hemi_bar.set_postfix(hemi=hemi_name, acc=f'{np.mean(accs):.4f}')

    # Standard montage subsets
    montages = {
        'full_62': list(range(N_CHANNELS)),
        'standard_1020_19': STANDARD_1020,
        'emotiv_epoc_14': EMOTIV_EPOC,
        'muse_approx_4': MUSE_APPROX,
    }
    mont_bar = tqdm(montages.items(), desc='  Retrain montages')
    for mont_name, indices in mont_bar:
        accs = _retrain_loso(data, indices, model_kwargs, train_kwargs, device)
        all_results[f'montage_{mont_name}'] = {
            'n_channels': len(indices),
            'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_subj': accs,
        }
        mont_bar.set_postfix(montage=mont_name, acc=f'{np.mean(accs):.4f}')

    # Progressive attention-guided (fewer steps to manage cost)
    n_keep_steps = [62, 50, 40, 30, 20, 10, 5, 1]
    for strategy_name, ranking in [
        ('pi_least_first', grand_ranking),
        ('pi_most_first', grand_ranking[::-1]),
    ]:
        curve = {}
        step_bar = tqdm(n_keep_steps, desc=f'  Retrain {strategy_name}')
        for n_keep in step_bar:
            keep = ranking[:n_keep].tolist()
            accs = _retrain_loso(data, keep, model_kwargs, train_kwargs, device)
            curve[n_keep] = {'mean': float(np.mean(accs)),
                             'std': float(np.std(accs))}
            step_bar.set_postfix(n_keep=n_keep, acc=f'{np.mean(accs):.4f}')
        all_results[f'progressive_{strategy_name}'] = curve
        print(f"  Progressive retrain {strategy_name}: done")

    # Progressive random (5 seeds x 8 levels)
    curve_random = {}
    rand_bar = tqdm(n_keep_steps, desc='  Retrain random')
    for n_keep in rand_bar:
        subj_means = np.zeros(N_SUBJECTS)
        for seed in range(5):
            rng = np.random.RandomState(seed)
            keep = rng.choice(N_CHANNELS, n_keep, replace=False).tolist()
            accs = _retrain_loso(data, keep, model_kwargs, train_kwargs, device)
            subj_means += np.array(accs)
        subj_means /= 5
        curve_random[n_keep] = {
            'mean': float(np.mean(subj_means)),
            'std': float(np.std(subj_means)),
        }
        rand_bar.set_postfix(n_keep=n_keep, acc=f'{np.mean(subj_means):.4f}')
    all_results['progressive_random'] = curve_random
    print("  Progressive retrain random: done")

    return all_results


# ────────────────────────────────────────────────────────────────────
# Visualization (Phase 5)
# ────────────────────────────────────────────────────────────────────

def plot_progressive_ablation_curves(results, retrain_results=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    curves_to_plot = [
        ('progressive_pi_least_first', '#2ecc71', 'PI least first', '-', 'o'),
        ('progressive_random',         '#95a5a6', 'Random',         '-', 'o'),
        ('progressive_pi_most_first',  '#e74c3c', 'PI most first',  '-', 'o'),
    ]
    if retrain_results is not None:
        curves_to_plot.extend([
            ('retrain_progressive_pi_least_first', '#2ecc71', 'Retrain (least first)', '--', '^'),
            ('retrain_progressive_random',         '#95a5a6', 'Retrain random',        '--', '^'),
            ('retrain_progressive_pi_most_first',  '#e74c3c', 'Retrain (most first)',  '--', '^'),
        ])
    # Merge retrain curves into a combined dict for lookup
    _all_curves = dict(results)
    if retrain_results is not None:
        for k, v in retrain_results.items():
            if k.startswith('progressive_'):
                _all_curves[f'retrain_{k}'] = v
    for key, color, label, ls, marker in curves_to_plot:
        if key not in _all_curves:
            continue
        curve = _all_curves[key]
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


def plot_topographic_importance(grand_importance,
                                title='Permutation Importance (Grand Average)',
                                filename='topomap_importance.pdf'):
    import matplotlib.pyplot as plt
    import mne
    info, picks = _make_topo_info()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    mne.viz.plot_topomap(grand_importance[picks], info, axes=ax, show=False,
                         cmap='RdYlBu_r', contours=0, extrapolate='head',
                         outlines='head', sensors=True)
    ax.set_title(title, fontsize=13)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def plot_cumulative_topomaps(all_seed_arrays, seed_idx, metric_name):
    """Plot cumulative topomap averaging seeds 0..seed_idx.

    Args:
        all_seed_arrays: list of (N_SUBJECTS, N_CHANNELS) arrays, length >= seed_idx+1
        seed_idx: 0-based index of current seed
        metric_name: e.g. 'pi', 'ig'
    """
    import matplotlib.pyplot as plt
    import mne
    info, picks = _make_topo_info()
    # Cumulative grand average: mean across subjects per seed, then across seeds
    cumulative = np.array(
        [arr.mean(axis=0) for arr in all_seed_arrays[:seed_idx + 1]])
    grand = cumulative.mean(axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    mne.viz.plot_topomap(grand[picks], info, axes=ax, show=False,
                         cmap='RdYlBu_r', contours=0, extrapolate='head',
                         outlines='head', sensors=True)
    n_seeds_so_far = seed_idx + 1
    ax.set_title(f'{metric_name} — cumulative avg (seeds 1-{n_seeds_so_far})',
                 fontsize=13)
    fname = f'cumulative_{metric_name}_seed{n_seeds_so_far}.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
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


def plot_lobe_ablation_table(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lobes = list(LOBES.keys())

    # Keep-only
    ax = axes[0]
    means = [results[f'lobe_keep_only_{l}']['mean'] for l in lobes]
    stds = [results[f'lobe_keep_only_{l}']['std'] for l in lobes]
    n_chs = [results[f'lobe_keep_only_{l}']['n_channels'] for l in lobes]
    labels = [f'{l}\n({n}ch)' for l, n in zip(lobes, n_chs)]
    ax.bar(range(len(lobes)), means, yerr=stds, capsize=3,
           color='#3498db', alpha=0.8)
    ax.set_xticks(range(len(lobes)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('Keep Only This Lobe')
    ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.5)

    # Remove
    ax = axes[1]
    means = [results[f'lobe_remove_{l}']['mean'] for l in lobes]
    stds = [results[f'lobe_remove_{l}']['std'] for l in lobes]
    n_chs = [results[f'lobe_remove_{l}']['n_channels'] for l in lobes]
    labels = [f'w/o {l}\n({n}ch)' for l, n in zip(lobes, n_chs)]
    ax.bar(range(len(lobes)), means, yerr=stds, capsize=3,
           color='#e67e22', alpha=0.8)
    ax.set_xticks(range(len(lobes)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('Remove This Lobe')
    full_acc = results['montage_full_62']['mean']
    ax.axhline(y=full_acc, color='green', linestyle='--', alpha=0.7,
               label=f'Full 62ch ({full_acc:.3f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('lobe_ablation.pdf', dpi=300, bbox_inches='tight')
    print("Saved lobe_ablation.pdf")
    plt.close()


def plot_per_emotion_topomap(emotion_importances,
                            title='Per-Emotion Permutation Importance (rank-normalized)',
                            filename='per_emotion_topomap.pdf'):
    import matplotlib.pyplot as plt
    import mne
    from scipy.stats import rankdata

    info, picks = _make_topo_info()
    emotion_labels = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(5 * N_CLASSES, 5))
    for c in range(N_CLASSES):
        raw = emotion_importances[c][picks]
        ranked = rankdata(raw) / len(raw)  # rank-normalize to [0, 1]
        mne.viz.plot_topomap(ranked, info, axes=axes[c],
                             show=False, cmap='RdYlBu_r', contours=0,
                             vlim=(0, 1), extrapolate='head',
                             outlines='head', sensors=True)
        axes[c].set_title(emotion_labels.get(c, str(c)), fontsize=13)
    plt.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def plot_retrain_comparison(mask_results, retrain_results):
    """Grouped bar chart: mask-based vs retrain accuracy for montage subsets."""
    import matplotlib.pyplot as plt

    montage_keys = ['montage_full_62', 'montage_standard_1020_19',
                    'montage_emotiv_epoc_14', 'montage_muse_approx_4']
    labels = ['62ch (full)', '19ch (10-20)', '14ch (EPOC)', '4ch (Muse)']
    mask_means = [mask_results[k]['mean'] for k in montage_keys]
    mask_stds = [mask_results[k]['std'] for k in montage_keys]
    retrain_means = [retrain_results[k]['mean'] for k in montage_keys]
    retrain_stds = [retrain_results[k]['std'] for k in montage_keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mask_means, width, yerr=mask_stds, capsize=4,
           label='Mask-based', color='#3498db', alpha=0.85)
    ax.bar(x + width / 2, retrain_means, width, yerr=retrain_stds, capsize=4,
           label='Retrain from scratch', color='#e67e22', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy')
    ax.set_title('Mask-Based vs Retrain-from-Scratch Ablation')
    ax.legend()
    ax.axhline(y=0.25, color='black', linestyle=':', alpha=0.5)
    ax.set_ylim(0.15, 0.75)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('retrain_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Saved retrain_comparison.pdf")
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
    parser.add_argument('--notebook', action='store_true',
                        help='Use tqdm.notebook for Colab/Jupyter')
    parser.add_argument('--retrain_ablation', action='store_true',
                        help='Run retrain-from-scratch ablation (expensive)')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random seeds for ensemble stability (default=5)')
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
            data[subj][sess] = (X, y, trial_ids)
    for subj in range(1, N_SUBJECTS + 1):
        sizes = [data[subj][s][0].shape[0] for s in range(1, N_SESSIONS + 1)]
        print(f"  Subject {subj:2d}: {sizes} samples per session")

    # ── Fixed hyperparameters (matching SOGNN paper) ──
    model_kwargs = {'n_electrodes': N_CHANNELS, 'n_bands': N_BANDS,
                    'n_timeframes': T_FIXED, 'n_classes': N_CLASSES,
                    'top_k': 10, 'dropout': 0.1}
    train_kwargs = {'lr': 1e-5, 'wd': 1e-4, 'batch_size': 16,
                    'max_epochs': 200, 'auc_threshold': 0.99}

    # ── Phases 2+3+3b: Multi-seed ensemble (LOSO) ──
    n_seeds = args.n_seeds
    print(f"\n=== Multi-seed ensemble: {n_seeds} seed(s), LOSO ===")
    all_seed_models = []
    all_seed_results = []
    all_seed_importances = []
    all_seed_ig_importances = []
    all_seed_emotion_imp = []

    for seed_idx in range(n_seeds):
        seed = seed_idx + 1
        set_seed(seed)
        print(f"\n--- Seed {seed}/{n_seeds} ---")

        # Phase 2: LOSO train/test
        print(f"  Phase 2: LOSO training (seed {seed})")
        results_k, models_k = train_and_evaluate(
            data, SOGNN, model_kwargs, train_kwargs, device)
        all_seed_results.append(results_k)
        all_seed_models.append(models_k)

        # Phase 3: Permutation importance (on held-out subject's ALL sessions)
        print(f"  Phase 3: Permutation importance (seed {seed})")
        imp_k = np.zeros((N_SUBJECTS, N_CHANNELS))
        pi_bar = tqdm(range(1, N_SUBJECTS + 1), desc=f'PI seed {seed}')
        for subj in pi_bar:
            X_test = np.concatenate([data[subj][sess][0]
                                     for sess in range(1, N_SESSIONS + 1)])
            y_test = np.concatenate([data[subj][sess][1]
                                     for sess in range(1, N_SESSIONS + 1)])
            imp_k[subj - 1] = permutation_importance(
                models_k[subj], X_test, y_test, device, n_repeats=10)
        all_seed_importances.append(imp_k)

        # Phase 3b: Integrated Gradients importance
        print(f"  Phase 3b: Integrated Gradients (seed {seed})")
        ig_k = np.zeros((N_SUBJECTS, N_CHANNELS))
        ig_bar = tqdm(range(1, N_SUBJECTS + 1), desc=f'IG seed {seed}')
        for subj in ig_bar:
            X_test = np.concatenate([data[subj][sess][0]
                                     for sess in range(1, N_SESSIONS + 1)])
            y_test = np.concatenate([data[subj][sess][1]
                                     for sess in range(1, N_SESSIONS + 1)])
            ig_k[subj - 1] = integrated_gradients_importance(
                models_k[subj], X_test, y_test, device, n_steps=50)
        all_seed_ig_importances.append(ig_k)

        # Per-emotion PI
        emo_k = {c: np.zeros(N_CHANNELS) for c in range(N_CLASSES)}
        emo_bar = tqdm(range(1, N_SUBJECTS + 1), desc=f'Emo-PI seed {seed}')
        for subj in emo_bar:
            X_test = np.concatenate([data[subj][sess][0]
                                     for sess in range(1, N_SESSIONS + 1)])
            y_test = np.concatenate([data[subj][sess][1]
                                     for sess in range(1, N_SESSIONS + 1)])
            subj_emo = per_emotion_permutation_importance(
                models_k[subj], X_test, y_test, device, n_repeats=10)
            for c in subj_emo:
                emo_k[c] += subj_emo[c]
        for c in emo_k:
            emo_k[c] /= N_SUBJECTS
        all_seed_emotion_imp.append(emo_k)

        # Per-seed summary
        seed_grand = imp_k.mean(axis=0)
        seed_ranking = seed_grand.argsort()[::-1]
        print(f"  Seed {seed} top-5 PI: "
              f"{[CHANNEL_NAMES[i] for i in seed_ranking[:5]]}")

        # Cumulative topomaps (running average of seeds 1..current)
        plot_cumulative_topomaps(all_seed_importances, seed_idx, 'pi')
        plot_cumulative_topomaps(all_seed_ig_importances, seed_idx, 'ig')

    # ── Aggregate importance across seeds ──
    print("\n=== Aggregating across seeds ===")

    # PI: (K, 62) — each row is one seed's grand avg over subjects
    seed_grand_importances = np.array(
        [imp.mean(axis=0) for imp in all_seed_importances])
    grand_importance = seed_grand_importances.mean(axis=0)
    importance_std = seed_grand_importances.std(axis=0)
    grand_ranking = grand_importance.argsort()[::-1].copy()
    print("  Ensemble PI top-10:",
          [CHANNEL_NAMES[i] for i in grand_ranking[:10]])

    # IG: same logic
    seed_grand_ig = np.array(
        [imp.mean(axis=0) for imp in all_seed_ig_importances])
    grand_ig_importance = seed_grand_ig.mean(axis=0)
    ig_importance_std = seed_grand_ig.std(axis=0)
    grand_ig_ranking = grand_ig_importance.argsort()[::-1].copy()
    print("  Ensemble IG top-10:",
          [CHANNEL_NAMES[i] for i in grand_ig_ranking[:10]])

    # Test accuracy: per-subject average across seeds
    sognn_results = {}
    for subj in range(1, N_SUBJECTS + 1):
        sognn_results[subj] = float(np.mean(
            [sr[subj] for sr in all_seed_results]))
    accs = list(sognn_results.values())
    print(f"  Ensemble mean accuracy: {np.mean(accs):.4f} "
          f"+/- {np.std(accs):.4f}")

    # Per-emotion importance: average across seeds
    grand_emotion_imp = {c: np.zeros(N_CHANNELS) for c in range(N_CLASSES)}
    for emo_k in all_seed_emotion_imp:
        for c in emo_k:
            grand_emotion_imp[c] += emo_k[c]
    for c in grand_emotion_imp:
        grand_emotion_imp[c] /= n_seeds

    # ── Cross-seed stability diagnostics ──
    if n_seeds > 1:
        from scipy.stats import spearmanr
        rhos = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rho, _ = spearmanr(seed_grand_importances[i],
                                   seed_grand_importances[j])
                rhos.append(rho)
        print(f"  Cross-seed PI rank stability: mean Spearman rho = "
              f"{np.mean(rhos):.4f} "
              f"(min={np.min(rhos):.4f}, max={np.max(rhos):.4f})")
        ig_rhos = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rho, _ = spearmanr(seed_grand_ig[i], seed_grand_ig[j])
                ig_rhos.append(rho)
        print(f"  Cross-seed IG rank stability: mean Spearman rho = "
              f"{np.mean(ig_rhos):.4f} "
              f"(min={np.min(ig_rhos):.4f}, max={np.max(ig_rhos):.4f})")

    # ── Phase 4: Full ablation study (averaged across seeds) ──
    print("\n=== Phase 4: Ablation study ===")
    all_seed_ablations = []
    for seed_idx in range(n_seeds):
        print(f"  Ablation with seed {seed_idx + 1} models...")
        abl_k = run_full_ablation_study(
            all_seed_models[seed_idx], data, grand_ranking, device)
        all_seed_ablations.append(abl_k)

    # Average ablation results across seeds
    ablation_results = {}
    first_abl = all_seed_ablations[0]
    for key in first_abl:
        if key.startswith('progressive_'):
            curve = {}
            step_keys = list(first_abl[key].keys())
            for nk in step_keys:
                per_subj_stacked = np.array(
                    [abl[key][nk]['per_subj'] for abl in all_seed_ablations])
                avg_per_subj = per_subj_stacked.mean(axis=0).tolist()
                curve[nk] = {
                    'mean': float(np.mean(avg_per_subj)),
                    'std': float(np.std(avg_per_subj)),
                }
            ablation_results[key] = curve
        else:
            per_subj_stacked = np.array(
                [abl[key]['per_subj'] for abl in all_seed_ablations])
            avg_per_subj = per_subj_stacked.mean(axis=0).tolist()
            ablation_results[key] = {
                'n_channels': first_abl[key]['n_channels'],
                'mean': float(np.mean(avg_per_subj)),
                'std': float(np.std(avg_per_subj)),
                'per_subj': avg_per_subj,
            }

    # ── Phase 4b: Retrain-from-scratch ablation (optional) ──
    retrain_ablation_results = None
    if args.retrain_ablation:
        print("\n=== Phase 4b: Retrain-from-scratch ablation (LOSO) ===")
        retrain_ablation_results = run_retrain_ablation_study(
            data, grand_ranking, model_kwargs, train_kwargs, device)

    # ── Phase 5: Visualization & statistics ──
    print("\n=== Phase 5: Visualization ===")
    plot_progressive_ablation_curves(ablation_results, retrain_ablation_results)
    plot_topographic_importance(grand_importance)
    plot_topographic_importance(grand_ig_importance,
                                title='Integrated Gradients Importance (Grand Average)',
                                filename='topomap_importance_IG.pdf')
    plot_region_ablation_table(ablation_results)
    plot_lobe_ablation_table(ablation_results)
    plot_per_emotion_topomap(grand_emotion_imp)

    if retrain_ablation_results is not None:
        plot_retrain_comparison(ablation_results, retrain_ablation_results)

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
        sig = ('***' if adj_p < 0.001 else '**' if adj_p < 0.01
               else '*' if adj_p < 0.05 else 'n.s.')
        print(f"  {cfg_a} vs {cfg_b}: p={raw_p:.4f} "
              f"(adj={adj_p:.4f}) {sig}")

    # Save all results
    save_results = {
        'n_seeds': n_seeds,
        'evaluation': 'LOSO',
        'sognn_per_subject': {str(k): v for k, v in sognn_results.items()},
        'sognn_mean': float(np.mean(list(sognn_results.values()))),
        'grand_ranking': grand_ranking.tolist(),
        'grand_importance': grand_importance.tolist(),
        'importance_std': importance_std.tolist(),
        'grand_ig_importance': grand_ig_importance.tolist(),
        'grand_ig_ranking': grand_ig_ranking.tolist(),
        'ig_importance_std': ig_importance_std.tolist(),
        'ablation': ablation_results,
        'model_kwargs': model_kwargs,
        'train_kwargs': train_kwargs,
    }
    if retrain_ablation_results is not None:
        save_results['ablation_retrain'] = retrain_ablation_results
    with open('results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print("\nSaved results.json")
    print("Done.")
