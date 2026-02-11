# SOGNN Electrode Ablation Pipeline — Implementation Specification

> **Purpose.** This document specifies how to re-implement the EEG electrode ablation
> experiment pipeline (currently built around an MLP baseline) for a **Self-Organized
> Graph Neural Network (SOGNN)** operating on SEED-IV differential-entropy features.
> Every section maps one-to-one to a pipeline phase, states what the current MLP
> pipeline does, what must change for SOGNN, and the exact interface contracts the
> SOGNN code must satisfy so that downstream phases (ablation, importance,
> visualization, statistics) remain valid.
>
> **Reference.** Li et al., "Cross-Subject EEG Emotion Recognition With
> Self-Organized Graph Neural Network," *Frontiers in Neuroscience*, 2021.

---

## 0. Shared Constants (`config.py`)

| Constant | Value | Notes |
|----------|-------|-------|
| `N_CHANNELS` | 62 | SEED-IV electrode count |
| `N_BANDS` | 5 | delta, theta, alpha, beta, gamma |
| `N_CLASSES` | 4 | neutral / sad / fear / happy |
| `N_SUBJECTS` | 15 | |
| `N_SESSIONS` | 3 | Train on sess 1+2, test on sess 3 |
| `CHANNEL_NAMES` | 62 strings | 0-indexed, matches `.mat` row order |
| `SESSION_LABELS` | 3 × 24 list | Per-trial emotion label per session |
| `REGIONS_FINE` | 8 strips | Anterior → posterior non-overlapping |
| `LOBES` | 5 groups | Frontal / temporal / central / parietal / occipital |
| `HEMISPHERES` | 3 groups | Left / midline / right |
| `STANDARD_1020`, `EMOTIV_EPOC`, `MUSE_APPROX` | Index lists | Commercial montage subsets |
| `MNE_NAME_MAP` | Dict | For topographic plotting |
| **`T_FIXED`** | **64** | **New. Temporal padding length (matches SOGNN paper's SEED-IV setting)** |

These are all defined by electrode geometry and are model-agnostic. **Reuse as-is**
(adding only `T_FIXED`).

---

## 1. Data Loading (Phase 0)

### 1.1 Current MLP Pipeline

Each `.mat` file contains keys `de_LDS1` … `de_LDS24`. Each key holds one
trial's DE features with shape `(62, T_k, 5)` where `T_k` varies by trial.

```
load_seed4_session():
    for each trial k:
        trial = mat[f'de_LDS{k+1}']          # (62, T_k, 5)
        trial = trial.transpose(1, 0, 2)     # → (T_k, 62, 5)
        X_list.append(trial)                  # each frame = independent sample
    X = concatenate(X_list)                   # (N_total, 62, 5)  (N_total = ΣT_k)
```

**Key point:** The MLP treats each time frame as an independent sample.
Per-session z-score normalization is applied elementwise:
```
mean = X.mean(axis=0, keepdims=True)          # (1, 62, 5)
std  = X.std(axis=0, keepdims=True) + 1e-8
X    = (X - mean) / std
```

### 1.2 Adaptation for SOGNN

SOGNN's conv-pool blocks convolve along the **time axis** (and across bands)
independently for each electrode. It requires input shaped
`(Electrodes, Bands, TimeFrames)` = `(62, 5, T)` per sample.
**Each sample is an entire trial, not a single frame.**

#### Strategy: One trial = one sample (matching the original paper)

The SOGNN paper treats SEED-IV data as **72 samples per subject** (24 trials
× 3 sessions). Each trial is zero-padded to `T_fixed = 64` time frames. We
adopt the identical approach.

1. **Preserve trial boundaries.** Do not concatenate frames across trials.
   Each trial becomes one sample.
2. **Zero-pad to `T_fixed = 64`.** For each trial:
   - If `T_k < T_fixed`: right-pad with zeros along the time axis.
   - If `T_k > T_fixed`: truncate to `T_fixed` (rare in SEED-IV; typical
     trial lengths are 10–60 frames).
   - If `T_k == T_fixed`: no transformation needed.
3. **Per-session z-score normalization.** Compute statistics across the full
   session (all frames from all trials) *before* padding:
   ```python
   # Collect all frames from all trials in the session
   all_frames = concatenate([trial.transpose(1, 0, 2) for trial in trials])  # (N_frames, 62, 5)
   mean = all_frames.mean(axis=0, keepdims=True)   # (1, 62, 5)
   std  = all_frames.std(axis=0, keepdims=True) + 1e-8
   # Normalize each trial's frames, then pad
   for trial in trials:
       trial_normed = (trial.transpose(1, 0, 2) - mean) / std  # (T_k, 62, 5)
       trial_padded = pad_to(trial_normed, T_fixed)             # (T_fixed, 62, 5)
       trial_final  = trial_padded.transpose(1, 0, 2)           # (62, 5, T_fixed)
   ```
   Normalizing *before* padding ensures that the zero-padded region represents
   true zero (≈ session mean under z-score), not a distorted value.

4. **Output format per sample:** `(62, 5, T_fixed)`.

5. **Trial IDs.** Each sample has a unique trial index (0–23). Since there is
   exactly one sample per trial, trial IDs serve as group labels for
   GroupKFold / GroupShuffleSplit.

#### Resulting data structure

```python
data[subj][sess] = (X, y, trial_ids)
# X:         np.float32, shape (24, 62, 5, T_fixed)  — 24 trials per session
# y:         np.int64,   shape (24,)
# trial_ids: np.int64,   shape (24,)
```

**Sample counts:** 24 trials/session × 2 sessions (train) = 48 training samples;
24 trials (test). Total per subject: 72. This is small — see Section 11.5
for implications.

> **Why not sliding windows?** The SOGNN paper explicitly uses one sample per
> trial (72 per subject, 1,080 total). Introducing sliding windows would: (a)
> deviate from the published method, (b) create data leakage risks between
> windows from the same trial, (c) inflate sample counts non-uniformly across
> emotion classes if trial lengths vary by class. We match the paper exactly.

---

## 2. Model Definition

### 2.1 SOGNN Architecture Summary

The model processes input `(B, 62, 5, T_fixed)`. **Critically, the conv-pool
blocks process each electrode independently** — the paper states: "standard
convolution and max-pooling layers were applied to extract features for each
EEG electrode independently. Therefore, the features of different EEG
electrodes will not mix with each other."

This means each electrode's `(5, T)` feature map is treated as a separate
2D image. Implementation: reshape `(B, 62, 5, T)` → `(B*62, 1, 5, T)` before
conv layers, then reshape back after the conv-pool chain.

#### Dimension chain (SEED-IV, T_fixed = 64)

| Block | Operation | Per-electrode shape | Notes |
|-------|-----------|---------------------|-------|
| Input | Reshape | `(1, 5, 64)` | 1 input channel, 5 bands, 64 time steps |
| **Conv-Pool 1** | Conv2d(1, 32, 5×5, valid) → MaxPool(1, 2) | `(32, 1, 30)` | `5→1` across bands; `64→60→30` along time |
| **Conv-Pool 2** | Conv2d(32, 64, 1×5, valid) → MaxPool(1, 2) | `(64, 1, 13)` | `30→26→13` along time |
| **Conv-Pool 3** | Conv2d(64, 128, 1×5, valid) → MaxPool(1, 2) | `(128, 1, 4)` | `13→9→4` along time (floor division) |
| **Reshape** | Flatten per electrode | `(512,)` | D = 128 × 4 = 512 features per electrode |

After the conv-pool chain, reshape from `(B*62, 128, 1, 4)` back to
`(B, 62, 512)` — each of the 62 electrodes now has a 512-dimensional feature
vector. These become the **graph node features**.

| Block | Operation | Shape | Notes |
|-------|-----------|-------|-------|
| **Self-Org Graph 1** | `G = tanh(H W_s)`, `A = softmax(G G^T)`, top-k sparse | Adj: `(B, 62, 62)` | Dynamic per-sample adjacency |
| **Graph Conv 1** | `H' = ReLU(A_sparse H W_gc)` | `(B, 62, D')` | |
| **Self-Org Graph 2** | Same on updated H' | `(B, 62, 62)` | |
| **Graph Conv 2** | `H'' = ReLU(A_sparse H' W_gc)` | `(B, 62, D'')` | |
| **Self-Org Graph 3** | Same on H'' | `(B, 62, 62)` | |
| **Graph Conv 3** | `H''' = ReLU(A_sparse H'' W_gc)` | `(B, 62, D''')` | |
| **Flatten + Concat** | Flatten all node features | `(B, 62 * D''')` | **Not** global mean pooling |
| **FC + Output** | Linear → ReLU → Dropout → Linear → (4,) | `(B, 4)` | |

> **Important: Flatten, not mean pool.** The SOGNN paper states: "The outputs
> of the graph convolution layers were flattened and concatenated as a feature
> vector." This preserves per-electrode identity — the FC layer can learn
> electrode-specific weights. This has a critical consequence for the retrain
> ablation: with fewer electrodes, the flatten dimension changes, so the FC
> layer must be re-sized accordingly (see Section 5.2).

> **Top-k sparsification.** After computing `A = softmax(G G^T)`, only the
> top-k values per row are retained; the rest are set to zero. The paper does
> **not** re-normalize the remaining weights — the row sums become < 1.0. This
> can be interpreted as a "leak" to a virtual null node and is standard
> practice for sparse attention. **Do not re-normalize after sparsification.**

### 2.2 Interface Contract

```python
class SOGNN(nn.Module):
    def __init__(self, n_electrodes=62, n_bands=5, n_timeframes=64,
                 n_classes=4, top_k=10, dropout=0.1, **kwargs):
        ...

    def forward(self, x):
        """
        Args:
            x: (B, n_electrodes, n_bands, n_timeframes) float tensor
        Returns:
            logits: (B, n_classes) float tensor
        """
        ...
```

**No `.view(B, -1)` flattening in the training loop.** The model internally
handles all reshaping (including the `(B, E, 5, T) → (B*E, 1, 5, T)` reshape
for the conv-pool blocks and the final flatten).

### 2.3 Dependencies

SOGNN's graph convolution is `H' = sigma(AHW)` where `A` is computed internally.
This is implementable in **pure PyTorch** via batched matrix multiplication —
**no PyTorch Geometric or DGL dependency is required**.

```python
# Self-organized adjacency (batched)
G = torch.tanh(H @ W_self)                               # (B, E, d_graph)
A = torch.softmax(G @ G.transpose(-1, -2), dim=-1)       # (B, E, E)
# Top-k sparsification (NO re-normalization)
vals, idxs = A.topk(top_k, dim=-1)
A_sparse = torch.zeros_like(A).scatter_(-1, idxs, vals)   # sparse (B, E, E)
# Graph convolution
H_out = torch.relu(A_sparse @ H @ W_gc)                   # (B, E, D_out)
```

### 2.4 Weight Initialization

The SOGNN paper does not specify a weight initialization scheme. Use PyTorch
defaults (Kaiming uniform for Linear/Conv2d layers). Document this choice for
reproducibility.

---

## 3. Training Loop (Phase 1 + Phase 2)

### 3.1 Training Hyperparameters: Paper Values vs. Pipeline Defaults

The SOGNN paper uses specific training parameters that differ from the MLP
pipeline's defaults:

| Parameter | SOGNN Paper | MLP Pipeline | Recommendation |
|-----------|-------------|--------------|----------------|
| Learning rate | **1e-5** | 5e-4 | Use 1e-5 as default; include 5e-5 and 1e-4 in HP search |
| Weight decay | 1e-4 | 1e-4 | Same |
| Batch size | **16** | 128 | Use 16 as default; search [16, 32] |
| Dropout | **0.1** | 0.5 | Use 0.1 as default; search [0.1, 0.3] |
| Optimizer | Adam | Adam | Same |
| Stopping | **Training AUC >= 0.99** | Early stopping on val acc | **See note below** |
| Label smoothing | Not used | 0.1 | **See note below** |
| LR schedule | None mentioned | CosineAnnealingLR | **See note below** |

**Stopping criterion.** The paper stops when training AUC reaches 0.99. Our
pipeline uses early stopping on validation accuracy with patience. We
**retain early stopping** for the ablation study because:
(a) It prevents overfitting better than an AUC threshold on the training set.
(b) It is consistent across all ablation configurations (mask-based and retrain).
(c) The paper's AUC threshold was designed for cross-subject LOSO evaluation,
    not our within-subject protocol.

**Label smoothing.** The paper does not use it. We **retain label smoothing
= 0.1** as it provides regularization benefit that partially compensates for
the much smaller per-subject sample count (48 training samples vs. the paper's
~1,000+ in LOSO). Document this deviation.

**CosineAnnealingLR.** The paper does not mention it. We **retain it** as it
consistently improves convergence in our pipeline. Document this deviation.

> **Deviation disclosure.** Our training protocol intentionally differs from
> the original SOGNN paper in three ways: (1) within-subject evaluation (train
> sess 1+2, test sess 3) vs. cross-subject LOSO, (2) early stopping on
> validation accuracy vs. training AUC threshold, (3) label smoothing and
> cosine LR schedule added as regularization. These deviations are necessary
> because our ablation study requires a per-subject model trained within a
> single subject's data, whereas the paper's LOSO protocol trains on 14
> subjects' pooled data.

### 3.2 `train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0)`

#### Current (MLP)
```python
logits = model(X_batch.view(X_batch.size(0), -1))   # flatten to (B, 310)
```

#### SOGNN Adaptation
```python
logits = model(X_batch)   # X_batch is (B, 62, 5, T_fixed) — no reshape
```

**MixUp.** MixUp on 4D tensors works identically to 2D — element-wise convex
combination:
```python
X_batch = lam * X_batch + (1 - lam) * X_batch[idx]   # works for any shape
```

Note: MixUp on graph-structured data means the self-organized adjacency is
computed from the *mixed* input features, producing a non-linear interpolation
of the two source graphs. This is a valid augmentation strategy but differs
from MixUp on flat vectors. Consider setting `mixup_alpha = 0.0` as the
default for SOGNN and searching over `[0.0, 0.2]` during HP search.

**Everything else** (loss computation, gradient clipping with `max_norm=1.0`,
optimizer step) is unchanged.

### 3.3 `evaluate(model, loader, device, channel_mask=None)`

#### Current (MLP)
```python
if channel_mask is not None:
    mask = channel_mask.to(device).unsqueeze(-1)  # (1, C, 1)
    X_batch = X_batch * mask                       # (B, C, 5) * (1, C, 1)
logits = model(X_batch.view(X_batch.size(0), -1))
```

#### SOGNN Adaptation
```python
if channel_mask is not None:
    # mask shape: (1, C) → broadcast to (1, C, 1, 1) for (B, C, bands, T)
    mask = channel_mask.to(device).unsqueeze(-1).unsqueeze(-1)  # (1, C, 1, 1)
    X_batch = X_batch * mask
logits = model(X_batch)
```

**Ablation semantics under SOGNN (detailed analysis).** When a channel is
zeroed, two effects occur in the self-organized graph:

1. **Outgoing edges (the zeroed node's row in A).** The zeroed electrode has
   features H_i = 0. The graph construction computes G_i = tanh(0 · W_s) = 0.
   The dot product G_i · G_j^T = 0 for all j, so the softmax row becomes
   uniform: A_i,j = 1/E for all j. After top-k, the zeroed node attends
   uniformly to its k neighbors. It then aggregates their features, producing
   a non-zero representation after graph convolution.

2. **Incoming edges (other nodes' attention to the zeroed node).** For any
   non-zeroed node j, the dot product G_j · G_i^T = G_j · 0 = 0, giving the
   zeroed node the lowest unnormalized weight. After softmax, it receives
   negligible attention. After top-k, it is almost certainly excluded from
   other nodes' top-k neighbor sets.

**Net effect:** The zeroed electrode is effectively **disconnected from
incoming attention** but still produces a mildly non-zero output through its
uniform outgoing attention. Its contribution to the final flattened feature
vector is a weak average of neighbors' features — a form of noise. The
retrain-from-scratch ablation (where the node is truly removed from the graph)
serves as the ground-truth comparison.

### 3.4 `cross_validate(X, y, model_cls, model_kwargs, train_kwargs, k=5, groups=None)`

Model-agnostic — instantiates `model_cls(**model_kwargs)` and calls
`train_one_epoch` / `evaluate`. No changes needed beyond what is already
handled in those two callees.

**Note on small sample size.** With 48 training samples per subject and
`k=5` folds, each fold has ~38 train / ~10 val samples. This is small for a
100K+ parameter model. Consider using `k=3` for HP search to increase the
training set per fold (32 train / 16 val), or use leave-one-trial-out CV
(k=24 on pooled sessions).

### 3.5 Hyperparameter Search (Phase 1)

#### Current MLP grid
```python
{'h1': [128, 256], 'h2': [64], 'dropout': [0.3, 0.5],
 'lr': [5e-4], 'wd': [1e-4], 'batch_size': [128], 'mixup_alpha': [0.0, 0.2]}
```

#### SOGNN grid
```python
sognn_search_space = {
    'top_k':       [5, 10, 15],         # graph sparsity
    'dropout':     [0.1, 0.3],          # FC dropout (paper uses 0.1)
    'lr':          [1e-5, 5e-5, 1e-4],  # paper uses 1e-5
    'wd':          [1e-4],              # paper uses 1e-4
    'batch_size':  [16, 32],            # paper uses 16
    'mixup_alpha': [0.0, 0.2],
}
```

SOGNN model kwargs:
```python
sognn_model_kwargs = {
    'n_electrodes': N_CHANNELS,       # 62
    'n_bands':      N_BANDS,          # 5
    'n_timeframes': T_FIXED,          # 64
    'n_classes':    N_CLASSES,         # 4
    'top_k':        <from search>,
    'dropout':      <from search>,
}
```

**Skip-search defaults** (matching paper where possible):
```python
default_sognn_kwargs = {
    'n_electrodes': 62, 'n_bands': 5, 'n_timeframes': 64,
    'n_classes': 4, 'top_k': 10, 'dropout': 0.1,
}
default_sognn_train_kwargs = {
    'lr': 1e-5, 'wd': 1e-4, 'batch_size': 16,
    'max_epochs': 200, 'patience': 15, 'mixup_alpha': 0.0,
}
```

### 3.6 `train_and_evaluate(data, model_cls, model_kwargs, train_kwargs, device)`

Model-agnostic (uses `model_cls(**model_kwargs)`, calls `train_one_epoch` /
`evaluate`). After fixing those two callees, works as-is. Returns
`(per_subject_accs, trained_models)`.

### 3.7 Training Infrastructure (Unchanged)

| Component | Detail | Change needed? |
|-----------|--------|----------------|
| `set_seed(seed)` | Sets np/torch/cuda seeds | No |
| `GroupShuffleSplit(test_size=0.2)` | Trial-level train/val split | No |
| Early stopping with patience | Best val acc + deepcopy | No (patience=15 recommended) |
| `CosineAnnealingLR(T_max=max_epochs)` | LR schedule | No |
| `CrossEntropyLoss(label_smoothing=0.1)` | Loss function | No |
| Gradient clipping `max_norm=1.0` | Prevents exploding gradients | No |

---

## 4. Channel Importance Methods (Phase 3)

### 4.1 Permutation Importance (PI)

#### Current (MLP)
```python
X_t = tensor(X)                          # (N, 62, 5)
logits = model(X_t.reshape(N, -1))       # flatten
X_perm[:, ch, :] = X_perm[perm_idx, ch, :]  # shuffle channel ch across samples
logits = model(X_perm.reshape(N, -1))
```

#### SOGNN Adaptation
```python
X_t = tensor(X)                          # (N, 62, 5, T_fixed)
logits = model(X_t)                      # no flatten
X_perm[:, ch, :, :] = X_perm[perm_idx, ch, :, :]  # shuffle (bands, time) together
logits = model(X_perm)
```

**Semantic note.** Permutation importance shuffles an electrode's entire
feature vector (all bands × all time steps) across samples. This is correct
for SOGNN: shuffling one electrode's temporal profile disrupts both its
direct feature representation and its contribution to the self-organized
adjacency matrix (since the graph is recomputed per input).

**Batching consideration.** With only 24 test samples (session 3) and 4D
input, the full test set fits in GPU memory. No mini-batch loop needed for
the forward pass, but `n_repeats=10` shuffles per channel are still required.

### 4.2 Integrated Gradients (IG)

#### Current (MLP)
```python
x_interp = (alpha * X_t[start:end]).reshape(end - start, flat_dim)
x_interp.requires_grad_(True)
logits = model(x_interp)
# accumulate gradients on (N, 310) → reshape to (N, 62, 5)
```

#### SOGNN Adaptation
```python
x_interp = alpha * X_t[start:end]         # (batch, 62, 5, T_fixed)
x_interp = x_interp.detach().requires_grad_(True)
logits = model(x_interp)
target_logits = logits.gather(1, y_t[start:end].unsqueeze(1)).squeeze(1)
target_logits.sum().backward()
accum[start:end] += x_interp.grad.detach()  # (batch, 62, 5, T_fixed)
```

Channel-level aggregation:
```python
avg_grad = accum / n_steps                 # (N, 62, 5, T_fixed)
ig = X_t * avg_grad                        # element-wise
# Sum |IG| over bands and time, mean over samples
channel_importance = ig.abs().sum(dim=(2, 3)).mean(dim=0)  # (62,)
```

**Baseline = zero** is valid because z-score normalization ensures the data
mean ≈ 0. The zero baseline represents "no information" for each electrode.

**Note on zero-padded time steps.** For trials shorter than `T_fixed`, the
padded region already contains zeros (= baseline). IG attribution for these
time steps is exactly zero (since input = baseline), which is the correct
behavior — padded steps carry no information and receive no attribution.

### 4.3 Per-Emotion PI

No structural changes. `per_emotion_permutation_importance` filters samples
by class label and calls `permutation_importance` on each subset. With 24
test samples and 4 classes, each subset has ~6 samples (SEED-IV labels are
not perfectly balanced across trials). This is small; report per-emotion PI
as exploratory rather than definitive.

---

## 5. Ablation Study (Phase 4)

### 5.1 Mask-Based Ablation (`run_full_ablation_study`)

This function is **entirely model-agnostic.** It only calls:
- `make_channel_mask(indices)` → returns `(1, 62)` binary mask
- `_eval_all_subjects(models, data, mask, device)` → calls `evaluate()`

Since `evaluate()` is adapted (Section 3.3), the ablation study works as-is.
All ablation configurations are preserved:

| Configuration | Description |
|---------------|-------------|
| Region keep-only / remove | 8 fine strips × 2 directions |
| Lobe keep-only / remove | 5 lobes × 2 directions |
| Hemisphere (L/M/R) | 3 groups |
| Montage subsets | 62 / 19 / 14 / 4 channels |
| Progressive PI least-first | Remove least important channels incrementally |
| Progressive PI most-first | Remove most important channels incrementally |
| Progressive random | 20 random seeds per step |

**Progressive step schedule:**
`[62, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 8, 5, 3, 1]`

### 5.2 Retrain-from-Scratch Ablation (`run_retrain_ablation_study`)

#### Current (MLP)
```python
mk = {**model_kwargs, 'input_dim': n_ch * N_BANDS}
model = MLPBaseline(**mk)
```
The MLP simply takes a shorter flat vector when channels are removed.

#### SOGNN Adaptation
```python
mk = {**model_kwargs, 'n_electrodes': n_ch}
model = SOGNN(**mk)
```

**Critical detail: FC layer dimension depends on `n_electrodes`.** Because
SOGNN uses flatten+concat (not global pooling), the FC layer input dimension
is `n_electrodes * D_graph_out`. The `SOGNN.__init__` must compute this
dynamically:
```python
# In __init__:
self.fc_input_dim = n_electrodes * graph_output_dim
self.fc = nn.Linear(self.fc_input_dim, n_classes)
```

The conv-pool blocks are per-electrode and produce the same feature
dimension regardless of electrode count. Only the graph layers and FC
layer are affected.

**Data slicing:**
```python
# Current:   X_train[:, ch_idx, :]        → (N, n_ch, 5)
# SOGNN:     X_train[:, ch_idx, :, :]     → (N, n_ch, 5, T_fixed)
```

**`top_k` clamping:** Clamp to `min(top_k, n_ch)` when the number of
remaining electrodes is smaller than the default `top_k` value.

**Progressive step schedule for retrain** (coarser to manage cost):
`[62, 50, 40, 30, 20, 10, 5, 1]`

---

## 6. Multi-Seed Ensemble (Phases 2+3 combined)

### Protocol (Unchanged Logic)

```
for seed in 1..K:
    set_seed(seed)
    Phase 2: train_and_evaluate() → models_k, results_k
    Phase 3: PI, IG, per-emotion PI → imp_k, ig_k, emo_k
aggregate:
    grand_importance = mean(seed_grand_importances, axis=0)      # (62,)
    importance_std   = std(seed_grand_importances, axis=0)       # (62,)
    grand_ranking    = argsort(grand_importance)[::-1]
    cross-seed Spearman rho as stability diagnostic
```

**Default `n_seeds = 5`.** Each seed trains 15 independent SOGNN models (one
per subject). Total: `5 × 15 = 75` training runs in Phase 2+3.

> **Runtime estimate.** SOGNN is slower per epoch than MLP due to graph
> construction and batched matrix multiplication. However, the per-subject
> dataset is very small (48 train, 24 test samples), so each epoch is fast.
> With `max_epochs=200`, `patience=15`, `batch_size=16`, expect early
> stopping at ~50–100 epochs. Estimate ~1–2 hours per seed on a single GPU.
> Total for 5 seeds: ~5–10 hours.

---

## 7. Visualization (Phase 5)

### All visualization functions are model-agnostic.

They consume:
- `grand_importance`: `(62,)` array
- `grand_ig_importance`: `(62,)` array
- `ablation_results`: dict of dicts (same structure regardless of model)
- `grand_emotion_imp`: `{class_id: (62,) array}`

**No changes needed** for any plot function:

| Function | Output | Input contract |
|----------|--------|----------------|
| `plot_progressive_ablation_curves` | `progressive_ablation.pdf` | Dict with `progressive_*` keys |
| `plot_topographic_importance` | `topomap_importance.pdf` | `(62,)` importance array |
| `plot_region_ablation_table` | `region_ablation.pdf` | Ablation results dict |
| `plot_lobe_ablation_table` | `lobe_ablation.pdf` | Ablation results dict |
| `plot_per_emotion_topomap` | `per_emotion_topomap.pdf` | `{class: (62,)}` dict |
| `plot_retrain_comparison` | `retrain_comparison.pdf` | mask + retrain result dicts |

---

## 8. Statistical Tests (Phase 5 cont.)

Wilcoxon signed-rank tests with Holm-Bonferroni correction. Comparisons:

1. Full 62ch vs Standard 10-20 (19ch)
2. Full 62ch vs EMOTIV EPOC (14ch)
3. Full 62ch vs Muse (4ch)
4. Left hemisphere vs Right hemisphere

**No code changes needed.** The tests operate on `per_subj` accuracy lists.

**Limitation (to disclose in the paper).** With n = 15 subjects, the Wilcoxon
signed-rank test has limited statistical power. The minimum achievable
p-value is ~6.1 × 10⁻⁵ (when all 15 differences share the same sign). After
Holm-Bonferroni correction with 4 tests, small-to-medium effect sizes may not
reach significance. **Recommendation:** Report effect sizes (e.g., matched-
pairs rank-biserial correlation r = Z / sqrt(N)) alongside p-values to convey
practical significance even when statistical significance is borderline.

---

## 9. Output Artifacts

### `results.json` Structure

```json
{
  "n_seeds": 5,
  "sognn_per_subject": {"1": 0.72, "2": 0.68, ...},
  "sognn_mean": 0.70,
  "grand_ranking": [24, 7, 31, ...],
  "grand_importance": [0.012, 0.008, ...],
  "importance_std": [0.003, 0.002, ...],
  "grand_ig_importance": [0.15, 0.09, ...],
  "grand_ig_ranking": [24, 31, 7, ...],
  "ig_importance_std": [0.01, 0.008, ...],
  "ablation": { ... },
  "ablation_retrain": { ... },
  "best_sognn_kwargs": { ... },
  "best_sognn_train_kwargs": { ... }
}
```

Key name changes: `mlp_*` → `sognn_*`.

---

## 10. Summary of Required Code Changes

### Functions that need modification

| Function | File | Change |
|----------|------|--------|
| `load_seed4_session` | main.py | Keep each trial as one sample; zero-pad to `T_fixed=64`; output `(24, 62, 5, 64)` per session |
| `train_one_epoch` | main.py | Remove `.view(B, -1)`; pass `X_batch` directly to model |
| `evaluate` | main.py | Remove `.view(B, -1)`; adjust mask broadcast from `(1,C,1)` to `(1,C,1,1)` |
| `permutation_importance` | main.py | Remove `.reshape(N, -1)`; shuffle `[:, ch, :, :]` |
| `integrated_gradients_importance` | main.py | Operate on 4D tensor; sum over dims `(2, 3)` for channel aggregation |
| `_retrain_all_subjects` | main.py | Use `SOGNN(n_electrodes=n_ch, ...)`; slice `X[:, ch_idx, :, :]` |
| HP search grid | main.py | Replace `h1/h2` with `top_k`; use paper-derived lr/batch/dropout ranges |
| `results.json` keys | main.py | Rename `mlp_*` → `sognn_*` |

### Functions that need NO modification

| Function | Reason |
|----------|--------|
| `set_seed` | Model-agnostic |
| `make_channel_mask` | Returns `(1, 62)` — shape-agnostic |
| `cross_validate` | Calls `train_one_epoch` / `evaluate`; model-agnostic via `model_cls` |
| `train_and_evaluate` | Same — model-agnostic via `model_cls` |
| `per_emotion_permutation_importance` | Delegates to `permutation_importance` |
| `_eval_all_subjects` | Calls `evaluate`; mask is `(1, C)` |
| `run_full_ablation_study` | Calls `_eval_all_subjects`; all logic is index-based |
| `run_retrain_ablation_study` | After fixing `_retrain_all_subjects`, logic is unchanged |
| All `plot_*` functions | Consume `(62,)` arrays and result dicts |
| Wilcoxon / Holm-Bonferroni | Operates on accuracy lists |

### New code required

| Component | Notes |
|-----------|-------|
| `SOGNN` class in `models.py` | Conv-pool blocks (per-electrode) + self-org graph + graph conv + flatten + FC |
| `T_FIXED` constant in `config.py` | Default 64 |

---

## 11. Design Considerations for Paper Quality

### 11.1 Ablation Validity Under Dynamic Graphs

Unlike the MLP where zeroed channels are invisible post-flattening, SOGNN's
self-organized graph construction means zeroed electrodes still exist as
graph nodes. The detailed analysis (Section 3.3) shows that zeroed electrodes
are largely disconnected: they receive almost no incoming attention (dot product
= 0 pushes them below top-k threshold for all non-zeroed nodes) but produce a
mild non-zero output through their uniform outgoing attention.

This creates a small discrepancy vs. the MLP ablation. **Mitigation:** The
retrain-from-scratch ablation removes electrodes from the graph entirely,
providing a clean ground truth. The paper should compare mask-based and
retrain ablation curves and note any divergence.

### 11.2 Within-Subject vs. Cross-Subject Protocol

The SOGNN paper reports 75.27% accuracy on SEED-IV using **leave-one-subject-out
(LOSO)** cross-validation — training on 14 subjects and testing on the held-out
one. Our pipeline uses a **within-subject** protocol: train on sessions 1+2 of
one subject, test on session 3 of the same subject.

**These results are not directly comparable.** LOSO trains on ~1,000 samples
from 14 subjects (much more data, but cross-subject variation). Within-subject
trains on 48 samples from one subject (much less data, but no cross-subject
gap). Expect within-subject SOGNN accuracy to differ substantially from 75.27%.

Clearly state in the paper:
> *"We evaluate SOGNN in a within-subject paradigm (train sessions 1+2, test
> session 3) to obtain per-subject electrode importance rankings. This differs
> from the cross-subject LOSO protocol reported in [Li et al., 2021] and
> results are not directly comparable."*

### 11.3 Small Sample Size (48 train, 24 test per subject)

With the one-trial-one-sample approach, each subject has only 48 training
samples and 24 test samples. For a model with ~100K+ parameters, this is
severely data-limited. Expected consequences:
- Heavy overfitting risk → early stopping and dropout are critical.
- High variance across subjects and seeds → multi-seed ensemble is essential.
- HP search may be unreliable with small CV folds → consider fixing HPs to
  paper defaults (`--skip_search`) and validating on a few subjects first.

### 11.4 Comparability with MLP Results

If the paper includes both MLP and SOGNN results, ensure:

- **Same test split.** Use identical `random_state=42` for GroupShuffleSplit,
  identical seed sequence for multi-seed ensemble.
- **Same ablation configurations.** Identical channel index sets for every
  region/lobe/hemisphere/montage/progressive experiment.
- **Same importance methods.** PI with `n_repeats=10`, IG with `n_steps=50`.
- **Different data format.** MLP uses individual frames as samples; SOGNN uses
  whole trials. The training data is the same information, just structured
  differently. The test set covers the same session-3 trials.
- **Explicitly state** that SOGNN's mask-based ablation has subtly different
  semantics than MLP's (Section 11.1) and that retrain ablation provides the
  comparable ground truth.

### 11.5 Reproducibility Checklist

- [ ] Fixed `T_FIXED = 64` (matches SOGNN paper's SEED-IV setting)
- [ ] Per-session z-score normalization computed before zero-padding
- [ ] One trial = one sample (no sliding windows; matches original paper)
- [ ] Trial-level group splitting (no data leakage)
- [ ] `set_seed()` called before each seed iteration
- [ ] Cross-seed Spearman rho reported for both PI and IG
- [ ] Gradient clipping `max_norm=1.0`
- [ ] Label smoothing = 0.1 (deviation from paper, documented)
- [ ] CosineAnnealingLR with `T_max = max_epochs` (deviation, documented)
- [ ] Early stopping patience = 15 (deviation from paper's AUC threshold)
- [ ] `top_k` clamped to `min(top_k, n_electrodes)` during retrain ablation
- [ ] No re-normalization after top-k sparsification (matches paper)
- [ ] Effect sizes reported alongside Wilcoxon p-values
- [ ] Weight initialization: PyTorch defaults (documented)
- [ ] Training deviations from the original paper explicitly disclosed

---

## 12. CLI Interface

```bash
# Full pipeline
python main.py --data_root /path/to/SEED_IV/eeg_feature_smooth

# Skip HP search (use paper-derived defaults)
python main.py --data_root /path --skip_search

# Single seed (fast)
python main.py --data_root /path --skip_search --n_seeds 1

# 5-seed ensemble (default)
python main.py --data_root /path --skip_search --n_seeds 5

# With retrain ablation
python main.py --data_root /path --skip_search --retrain_ablation

# Force CPU
python main.py --data_root /path --device cpu
```

Same CLI structure as the MLP pipeline. No new flags required.

---

## Appendix A: Data Shape Reference

| Pipeline stage | MLP shape | SOGNN shape |
|----------------|-----------|-------------|
| Raw .mat trial | `(62, T_k, 5)` | `(62, T_k, 5)` |
| After transpose | `(T_k, 62, 5)` | `(T_k, 62, 5)` |
| After z-score | `(T_k, 62, 5)` | `(T_k, 62, 5)` |
| After pad/concat | `(N_total, 62, 5)` frames concat | `(24, 62, 5, 64)` per session |
| DataLoader batch | `(B, 62, 5)` | `(B, 62, 5, 64)` |
| Conv-pool reshape | N/A | `(B*62, 1, 5, 64)` |
| Conv-pool output | N/A | `(B*62, 128, 1, 4)` |
| Graph node features | N/A | `(B, 62, 512)` |
| Graph conv output | N/A | `(B, 62, D_gc)` |
| Pre-FC features | `(B, 310)` | `(B, 62 * D_gc)` flatten |
| Model output | `(B, 4)` logits | `(B, 4)` logits |
| Channel mask | `(1, 62)` | `(1, 62)` |
| Mask broadcast | `(1, 62, 1)` | `(1, 62, 1, 1)` |
| PI shuffle dim | `[:, ch, :]` | `[:, ch, :, :]` |
| IG accumulator | `(N, 310)` | `(N, 62, 5, 64)` |
| IG → channel | `.reshape(N,62,5).abs().sum(2).mean(0)` | `.abs().sum(dim=(2,3)).mean(0)` |
| Importance output | `(62,)` | `(62,)` |

## Appendix B: SOGNN Parameter Count Estimate

Assuming `D_gc` = 64 for graph convolution output (a typical choice):

| Component | Formula | Approximate params |
|-----------|---------|-------------------|
| Conv-Pool 1 (1→32, 5×5) | 1 × 32 × 25 + 32 | 832 |
| Conv-Pool 2 (32→64, 1×5) | 32 × 64 × 5 + 64 | 10,304 |
| Conv-Pool 3 (64→128, 1×5) | 64 × 128 × 5 + 128 | 41,088 |
| Self-Org 1 (W_s: 512 → d_g) | 512 × d_g | ~16K (d_g=32) |
| Graph Conv 1 (W_gc: 512 → 64) | 512 × 64 | 32,768 |
| Self-Org 2 | 64 × d_g | ~2K |
| Graph Conv 2 (64 → 64) | 64 × 64 | 4,096 |
| Self-Org 3 | 64 × d_g | ~2K |
| Graph Conv 3 (64 → 64) | 64 × 64 | 4,096 |
| FC (62 × 64 → 4) | 3968 × 4 + 4 | 15,876 |
| Intermediate FC layer | 3968 × hidden + hidden × 4 | varies |
| **Total (estimate)** | | **~130K** |

Compare: MLPBaseline = 48K params. SOGNN is ~2.5× larger.

## Appendix C: Key Deviations from Original SOGNN Paper

| Aspect | SOGNN Paper | Our Pipeline | Rationale |
|--------|-------------|--------------|-----------|
| Evaluation protocol | Cross-subject LOSO | Within-subject (sess 1+2 → sess 3) | Per-subject importance rankings needed for ablation |
| Stopping criterion | Training AUC >= 0.99 | Early stopping on val acc (patience 15) | Prevents overfitting with small per-subject data |
| Label smoothing | None | 0.1 | Regularization for small sample size |
| LR schedule | None specified | CosineAnnealingLR | Consistent with MLP pipeline; improves convergence |
| Learning rate | 1e-5 | 1e-5 (default), HP-searched | Matched to paper |
| Dropout | 0.1 | 0.1 (default), HP-searched | Matched to paper |
| Batch size | 16 | 16 (default), HP-searched | Matched to paper |
| Multi-seed ensemble | Not used | 5 seeds, averaged | Stability of importance rankings |
| Top-k re-normalization | Not re-normalized | Not re-normalized | Matched to paper |
