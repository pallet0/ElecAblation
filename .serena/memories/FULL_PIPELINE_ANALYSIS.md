# Full EEG Ablation Pipeline Analysis - Multi-Seed Ensembling

## Executive Summary

The EEG ablation pipeline is a complete 5-phase system for analyzing electrode importance in emotion recognition. The pipeline flows sequentially from data loading → hyperparameter search → per-subject training → permutation importance computation → ablation studies → visualization.

---

## Phase-by-Phase Flow

### Phase 0: Data Loading (lines 877-914)
- Loads SEED-IV .mat files from disk with per-session z-score normalization
- Creates a `data` dict: `data[subj][sess] = (X, y, trial_ids)` where:
  - X: (N, 62, 5) — samples × channels × bands
  - y: (N,) — class labels
  - trial_ids: (N,) — trial indices for group-level splits
- Pools sessions 1+2 across all subjects for HP search

### Phase 1: Hyperparameter Search (lines 916-957)
- **Optional** (`--skip_search` uses defaults)
- Grid searches MLPBaseline over 8 hyperparameters:
  - `h1` ∈ {128, 256}, `h2` ∈ {64}, `dropout` ∈ {0.3, 0.5}
  - `lr` ∈ {5e-4}, `wd` ∈ {1e-4}, `batch_size` ∈ {128}
  - `mixup_alpha` ∈ {0.0, 0.2}
- Uses 5-fold GroupKFold CV on pooled sessions 1+2
- Produces: `best_mlp_kwargs` (model config), `best_mlp_train_kwargs` (training config)

### Phase 2: Per-Subject Train/Test (lines 959-962)
```python
mlp_results, mlp_models = train_and_evaluate(data, MLPBaseline, best_mlp_kwargs, best_mlp_train_kwargs, device)
```
**Returns:**
- `mlp_results`: dict {subj → test_accuracy}
- `mlp_models`: dict {subj → trained MLPBaseline model}

**Details (lines 177-242):**
- Trains each of 15 subjects independently
- Uses **GroupShuffleSplit(test_size=0.2, random_state=42)** on pooled sessions 1+2 for early stopping
- Tests on session 3
- CosineAnnealingLR scheduler, label smoothing=0.1, gradient clipping (max_norm=1.0)
- Stores best checkpoint via deepcopy

---

## Phase 3: Permutation Importance (PI)

### Core PI Computation (lines 965-975)
```python
importances = np.zeros((N_SUBJECTS, N_CHANNELS))  # (15, 62)
for subj in range(1, N_SUBJECTS + 1):
    X_test, y_test, _ = data[subj][3]
    importances[subj - 1] = permutation_importance(mlp_models[subj], X_test, y_test, device, n_repeats=10)

grand_importance = importances.mean(axis=0)  # (62,) — mean over subjects
grand_ranking = grand_importance.argsort()[::-1].copy()  # (62,) — sorted channel indices, most→least important
```

**`permutation_importance()` function (lines 249-280):**
- Takes baseline accuracy on full channels
- For each channel: shuffle that channel 10 times, measure accuracy drop
- Returns (62,) array where `importance[c] = baseline_acc - mean(shuffled_accs)`
- Positive importance = important channel (shuffling hurts accuracy)

---

## Phase 3b: Integrated Gradients Importance (lines 977-988)

Similar structure to PI:
```python
ig_importances = np.zeros((N_SUBJECTS, N_CHANNELS))  # (15, 62)
for subj in range(1, N_SUBJECTS + 1):
    X_test, y_test, _ = data[subj][3]
    ig_importances[subj - 1] = integrated_gradients_importance(
        mlp_models[subj], X_test, y_test, device, n_steps=50)

grand_ig_importance = ig_importances.mean(axis=0)  # (62,) — mean over subjects
grand_ig_ranking = grand_ig_importance.argsort()[::-1].copy()  # (62,) — sorted channel indices
```

**`integrated_gradients_importance()` function (lines 297-336):**
- Computes IG via 50 interpolation steps from zero baseline to input
- For each step α ∈ [1/50, 50/50]:
  - Interpolates x_interp = (α * X)
  - Computes gradient w.r.t. target logit
  - Accumulates gradients
- Final importance = |X ⊙ avg_grad| summed over bands, averaged over samples
- Returns (62,) importance array

---

## Phase 4: Full Ablation Study (lines 990-993)

```python
ablation_results = run_full_ablation_study(mlp_models, data, grand_ranking, device)
```

**Function signature (lines 412-516):**
- **Input**: `mlp_models` (dict of 15 trained models), `data`, `grand_ranking` (62,) sorted channels
- **Uses**: `_eval_all_subjects()` helper to evaluate all 15 subjects with a given channel mask
- **Does NOT retrain** — reuses trained models with input masking

**Ablation configurations:**
1. **Region keep/remove** (16 configs): 8 regions × 2 (keep-only, remove)
2. **Lobe keep/remove** (10 configs): 5 lobes × 2
3. **Hemisphere** (3 configs): left, midline, right
4. **Standard montages** (4 configs): 62ch, 19ch (10-20), 14ch (EPOC), 4ch (Muse)
5. **Progressive PI-guided** (2 strategies × 15 steps = 30 configs):
   - `pi_least_first`: ranking → worst to best (least important first)
   - `pi_most_first`: ranking[::-1] → best to worst (most important first)
6. **Progressive random** (15 steps × 20 seeds = averaged per subject):
   - Random selection at each step, 20 repeats
   - Computes per-subject mean, then grand mean ± std

**Key detail:** Each config produces `per_subj` list of 15 accuracies, then `mean` and `std` across subjects.

---

## Phase 4b: Retrain-from-Scratch Ablation (lines 995-1000)

```python
if args.retrain_ablation:
    retrain_ablation_results = run_retrain_ablation_study(data, grand_ranking, best_mlp_kwargs, best_mlp_train_kwargs, device)
```

**Function signature (lines 519-627):**
- **Input**: `data`, `grand_ranking`, model/train configs
- **Process**: For EACH ablation config, trains a fresh MLPBaseline with **only the selected channels**
  - Input dim dynamically set to `n_channels * N_BANDS`
  - Same train/val/test splits as Phase 2
  - Same early stopping, CosineAnnealingLR, label smoothing

**`_retrain_all_subjects()` helper (lines 355-409):**
- Retrains MLPBaseline for each subject on subset of channels
- Returns list of 15 accuracies per config

**Configurations (fewer steps to manage cost):**
1. Region keep/remove (16)
2. Lobe keep/remove (10)
3. Hemisphere (3)
4. Standard montages (4)
5. Progressive PI-guided (2 strategies × 8 levels = 16)
6. Progressive random (8 levels × 5 seeds = averaged per subject)

---

## Phase 5: Visualization & Statistics (lines 1002-1078)

**Outputs 6 PDF files + results.json:**

1. **`progressive_ablation.pdf`** (lines 634-680):
   - Overlays mask-based and (optionally) retrain curves
   - 3 strategies: PI least first, random, PI most first

2. **`topomap_importance.pdf`** (lines 701-713):
   - Topographic map of `grand_importance` (PI)

3. **`topomap_importance_IG.pdf`** (lines 1006-1008):
   - Topographic map of `grand_ig_importance` (IG)

4. **`region_ablation.pdf`** (lines 716-756):
   - 2 bar charts: keep-only vs remove per region

5. **`lobe_ablation.pdf`** (lines 759-799):
   - 2 bar charts: keep-only vs remove per lobe

6. **`retrain_comparison.pdf`** (lines 820-850, optional):
   - Grouped bars: mask-based vs retrain accuracies for montages

7. **`per_emotion_topomap.pdf`** (lines 1012-1025):
   - Per-emotion topographic maps (4 subplots)
   - Computed via `per_emotion_permutation_importance()` (lines 284-294)

**Statistical tests (lines 1030-1060):**
- Wilcoxon signed-rank test (paired) on per-subject accuracies
- Holm-Bonferroni correction for 4 comparisons
- Comparisons: full vs 10-20, full vs EPOC, full vs Muse, left vs right

**Saved to `results.json` (lines 1062-1078):**
```json
{
  "mlp_per_subject": {str(subj): acc, ...},
  "mlp_mean": float,
  "grand_ranking": [ch_idx, ...],  # 62 channels sorted best→worst
  "grand_importance": [imp, ...],   # 62 values
  "grand_ig_importance": [imp, ...],
  "grand_ig_ranking": [ch_idx, ...],
  "ablation": {...},                 # All mask-based results
  "ablation_retrain": {...},         # (optional) All retrain results
  "best_mlp_kwargs": {...},
  "best_mlp_train_kwargs": {...}
}
```

---

## Function Signatures & Return Values

### `train_and_evaluate(data, model_cls, model_kwargs, train_kwargs, device='cuda')`
**Returns:** `(results, models)` tuple
- `results`: dict {subj (1-15) → test_accuracy (float)}
- `models`: dict {subj (1-15) → trained model instance}

### `permutation_importance(model, X, y, device, n_repeats=10)`
**Returns:** (62,) numpy array — per-channel importance

### `integrated_gradients_importance(model, X, y, device, n_steps=50, batch_size=256)`
**Returns:** (62,) numpy array — per-channel IG importance

### `_eval_all_subjects(models, data, mask, device)`
**Returns:** list of 15 floats — test accuracies per subject

### `_retrain_all_subjects(data, channel_indices, model_kwargs, train_kwargs, device)`
**Returns:** list of 15 floats — test accuracies per subject

### `run_full_ablation_study(models, data, grand_ranking, device='cuda')`
**Returns:** dict with ~40 keys, each value is:
```python
{
  'n_channels': int,
  'mean': float,
  'std': float,
  'per_subj': list of 15 floats
}
```

### `run_retrain_ablation_study(data, grand_ranking, model_kwargs, train_kwargs, device='cuda')`
**Returns:** Similar structure to `run_full_ablation_study()`

---

## Argparse Section (lines 857-875)

```python
parser.add_argument('--data_root', type=str, default=DATA_ROOT, ...)
parser.add_argument('--device', type=str, default=None, ...)
parser.add_argument('--skip_search', action='store_true', ...)  # Use HP defaults
parser.add_argument('--notebook', action='store_true', ...)      # Use tqdm.notebook
parser.add_argument('--retrain_ablation', action='store_true', ...) # Run Phase 4b
```

---

## Existing Seeding Infrastructure

**Random seeds used:**
- `GroupShuffleSplit(random_state=42)` — hardcoded in `train_and_evaluate()` and `_retrain_all_subjects()` (lines 193, 370)
- `np.random.RandomState(seed)` — for random ablation (lines 505, 613) with seed ∈ [0, 19] (Phase 4) or [0, 4] (Phase 4b)
- No torch.manual_seed() or np.random.seed() calls in main pipeline
- MixUp uses `torch.randperm()` without explicit seeding

**Note:** The pipeline uses:
- Fixed random_state=42 for train/val splits (deterministic)
- Loop-based seeds for random ablation strategies
- No global seed control at entry point

---

## Data Flow Summary

```
Phase 0: LOAD_DATA
  ↓ data[subj][sess] = (X, y, trial_ids)
  ↓ Per-session z-score normalization
  
Phase 1: HP_SEARCH (optional)
  ↓ Pooled sessions 1+2
  ↓ 5-fold GroupKFold CV
  ↓ best_mlp_kwargs, best_mlp_train_kwargs
  
Phase 2: TRAIN_AND_EVALUATE
  ↓ Train 15 models independently
  ↓ Session 1+2 train, Session 3 test
  ↓ mlp_results: {subj → acc}, mlp_models: {subj → model}
  
Phase 3: PERMUTATION_IMPORTANCE
  ↓ Use Phase 2 models
  ↓ importances: (15, 62) array
  ↓ grand_importance: (62,), grand_ranking: (62,)
  
Phase 3b: INTEGRATED_GRADIENTS
  ↓ Use Phase 2 models
  ↓ ig_importances: (15, 62) array
  ↓ grand_ig_importance: (62,), grand_ig_ranking: (62,)
  
Phase 4: ABLATION_STUDY
  ↓ Use Phase 2 models + grand_ranking
  ↓ Input masking (no retraining)
  ↓ ablation_results: ~40 configs with per_subj, mean, std
  
Phase 4b: RETRAIN_ABLATION (optional)
  ↓ Use grand_ranking only (not models)
  ↓ Train fresh models per config
  ↓ retrain_ablation_results: ~40 configs with per_subj, mean, std
  
Phase 5: VISUALIZATION & STATISTICS
  ↓ Plot curves, topomaps, bar charts
  ↓ Wilcoxon tests with Holm-Bonferroni correction
  ↓ Save results.json
```

---

## Key Design Principles

1. **Input masking for ablation**: Channels are zeroed before flattening. This works because data is z-score normalized (mean ≈ 0).

2. **Grand importance via averaging**: Per-subject PI/IG computed on test set, then grand average across subjects. Ranking derived from grand average.

3. **Deterministic train/val splits**: GroupShuffleSplit with fixed random_state=42 for reproducibility.

4. **Per-subject-first random ablation**: For random curves, accuracies averaged per-subject first, then std computed across subjects (reduces variance).

5. **Retrain validates mask importance**: Phase 4b retrains on subset channels to verify mask-based estimates reflect achievable performance.

6. **MLP-only pipeline**: Only MLPBaseline used. Attention models (ChannelAttentionEEGNet, DualAttentionEEGNet) defined but unused.

---

## Multi-Seed Ensembling Readiness

Current state:
- **No explicit multi-seed infrastructure** at entry point
- **Hardcoded random_state=42** for train/val splits
- **Loop-based seeds** used only in random ablation strategies (not for model ensemble)

To implement multi-seed ensembling would require:
1. Adding a `--n_seeds` CLI argument
2. Wrapping Phases 1–4 in a seed loop
3. Averaging results across seeds after each phase
4. Storing per-seed results if comparison is desired
