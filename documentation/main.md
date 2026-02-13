# Pipeline Execution and Logic (`main.py`)

This document explains the 6-phase execution flow of the SOGNN electrode ablation pipeline, including data handling, importance metrics, and statistical validation.

## Phase 0: Data Loading and Preprocessing

The `load_seed4_session` function implements the data pipeline.

```python
def load_seed4_session(mat_path, session_idx):
    data = sio.loadmat(mat_path)
    labels = SESSION_LABELS[session_idx]
    # ... extraction of trials ...
    
    # Compute z-score statistics from ALL frames before padding
    all_frames = np.concatenate(trials_raw, axis=0)  # [Line 9]
    mean = all_frames.mean(axis=0, keepdims=True)    # [Line 10]
    std = all_frames.std(axis=0, keepdims=True) + 1e-8
    
    # Normalize each trial, zero-pad, and transpose to (62, 5, T_FIXED)
    X_list = []
    for trial in trials_raw:
        trial_normed = (trial - mean) / std          # [Line 11]
        # ... T_FIXED로 패딩/절단 ...
        X_list.append(trial_padded.transpose(1, 2, 0))
    # ...
```

### Line-by-Line Breakdown (Preprocessing)
- **Line 9 (`np.concatenate(trials_raw, axis=0)`)**: Stacks all trials (each with a different number of time frames) into one giant array. This is necessary to calculate global statistics for the entire session.
- **Line 10 (`all_frames.mean(...)`)**: Calculates the average Differential Entropy value for each (Channel, Band) pair across the whole session.
- **Line 11 (`(trial - mean) / std`)**: Performs the actual Z-score normalization. By using the session-wide mean/std, we preserve the relative intensity differences between different trials within the same session while removing session-level bias.

## Phase 1 & 2: Training and Ensemble

The pipeline uses **Leave-One-Subject-Out (LOSO)** cross-validation, repeated over multiple random seeds to ensure stability.

```python
def train_and_evaluate(data, model_cls, model_kwargs, train_kwargs, device='cuda'):
    # ...
    for test_subj in subj_bar:
        # [Line 12] Pool ALL sessions from all OTHER subjects for training
        X_train = np.concatenate([data[s][sess][0] ... if s != test_subj ...])
        # [Line 13] Pool ALL sessions from held-out subject for testing
        X_test = np.concatenate([data[test_subj][sess][0] ...])
        # ... training loop ...
```

### Line-by-Line Breakdown (LOSO)
- **Line 12 (`X_train = ...`)**: Creates the training set by aggregating data from 14 out of 15 subjects. This tests the model's ability to generalize to a completely new person (Subject-Independent task).
- **Line 13 (`X_test = ...`)**: The 15th subject is kept strictly for evaluation.

## Phase 3: Electrode Importance Metrics

Importance is calculated using **Permutation Importance (PI)**.

```python
def permutation_importance(model, X, y, device, n_repeats=10):
    # ...
    for ch in range(n_channels):
        # [Line 14] Shuffle channel 'ch' across the batch
        X_perm[:, ch, :, :] = X_perm[perm_idx, ch, :, :]
        # [Line 15] Measure drop in accuracy
        # ...
        acc_importances[ch] = baseline_acc - float(np.mean(shuffled_accs))
```

### Line-by-Line Breakdown (Importance)
- **Line 14 (`X_perm[:, ch, ...] = ...`)**: This is the core of the ablation logic. By shuffling the data of a specific channel across different samples, we destroy the relationship between that electrode's signal and the true emotion label, while keeping the signal's distribution (mean/variance) intact.
- **Line 15 (`baseline_acc - ...`)**: If the accuracy drops significantly after shuffling, it proves that the model was relying heavily on that specific electrode to make its decisions.

## Phase 4: Ablation Studies

Ablation is performed by applying a binary mask to the input of a trained model.

```python
def evaluate(model, loader, device, channel_mask=None):
    # ...
    if channel_mask is not None:
        mask = channel_mask.to(device).unsqueeze(-1).unsqueeze(-1)  # (1,C,1,1)
        X_batch = X_batch * mask
    logits = model(X_batch)
    # ...
```

- **Progressive Ablation**: Electrodes are removed one-by-one based on the "Grand Ranking" (average PI across all subjects and seeds). This generates an accuracy curve.
- **Knee Point Detection**: The "optimal" number of electrodes is found by finding the point of maximum curvature (knee) on the ablation curve.

## Phase 5: Visualization and Statistics

The results are synthesized into several plots using `matplotlib` and `mne`.

- **Topographic Maps**: Uses `mne.viz.plot_topomap` to project importance scores onto a 2D scalp representation.
- **Statistical Testing**: Uses `scipy.stats.wilcoxon` to compare different configurations (e.g., Full 62ch vs. 10-20 system) and applies **Holm-Bonferroni correction** to control the family-wise error rate.

```python
# Holm-Bonferroni correction snippet
n_tests = len(raw_p_values)
sorted_idx = np.argsort(raw_p_values)
for rank, idx in enumerate(sorted_idx):
    adjusted_p[idx] = raw_p_values[idx] * (n_tests - rank)
# ... ensure monotonicity ...
```
