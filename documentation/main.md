# main.py Documentation

This script implements the complete pipeline for the EEG electrode ablation study using the SOGNN model on the SEED-IV dataset.

## Core Functions

### Data Loading and Preprocessing
- `load_seed4_session(mat_path, session_idx)`:
  - Loads Differential Entropy (DE) features from `.mat` files.
  - Transposes data to (Trial, Electrodes, Bands, Time).
  - Computes z-score normalization statistics across all trials within a session.
  - Zero-pads or truncates trials to a fixed length (`T_FIXED`).
- `make_channel_mask(active_indices, n_channels, batch_size)`:
  - Creates a binary mask (1.0 for kept channels, 0.0 for ablated) used during evaluation to simulate electrode removal.

### Training and Evaluation Helpers
- `train_one_epoch(...)`: Standard PyTorch training loop for one epoch.
- `evaluate(...)`: Evaluates the model accuracy. Supports applying a `channel_mask` to zero out specific electrodes.
- `compute_training_auc(...)`: Computes the macro-averaged Area Under the ROC Curve (AUC) for multi-class classification, used as a stopping criterion.
- `train_and_evaluate(...)`: Implements the **Leave-One-Subject-Out (LOSO)** validation scheme. Trains a model on 14 subjects and tests on the 15th, repeating for all folds.

### Importance Computation
- `permutation_importance(model, X, y, device, n_repeats)`:
  - Measures electrode importance by shuffling the data for a specific channel across samples and measuring the drop in accuracy.
- `per_emotion_permutation_importance(...)`: Computes permutation importance specifically for each emotion class.
- `integrated_gradients_importance(...)`:
  - Implements the Integrated Gradients (IG) attribution method.
  - Computes the integral of gradients along a path from a baseline (zero) to the actual input.
  - Aggregates the absolute IG values over time and frequency bands to get a per-channel importance score.

### Ablation Study Logic
- `run_full_ablation_study(...)`:
  - Evaluates pre-trained models using masks for different regions, lobes, and hemispheres.
  - Performs **Progressive Ablation**: incrementally removes electrodes based on importance rankings (Least Important First vs. Most Important First) or randomly.
- `run_retrain_ablation_study(...)`:
  - Unlike mask-based ablation, this **retrains the model from scratch** for each configuration (e.g., training a 4-channel model instead of masking a 62-channel model).
  - Used to verify if the model can adapt to reduced electrode sets.

### Visualization and Statistics
- `plot_progressive_ablation_curves(...)`: Plots accuracy vs. number of channels.
- `plot_topographic_importance(...)`: Generates heatmaps of electrode importance on a 2D scalp projection using `mne`.
- `plot_region_ablation_table(...)` & `plot_lobe_ablation_table(...)`: Bar charts comparing regional/lobar keep-only vs. removal.
- `plot_retrain_comparison(...)`: Compares mask-based ablation vs. retraining from scratch.

## Execution Flow (`if __name__ == '__main__':`)

1. **Phase 0 (Data Loading)**: Loads the entire SEED-IV dataset into memory.
2. **Multi-seed Ensemble**: Runs the LOSO pipeline multiple times (controlled by `--n_seeds`) to ensure statistical stability.
3. **Phase 2 (Training)**: Trains SOGNN models for each LOSO fold.
4. **Phase 3 (Importance)**: Calculates Permutation Importance and Integrated Gradients for each subject and electrode.
5. **Aggregation**: Averages importance scores across subjects and seeds to create a "Grand Ranking."
6. **Phase 4 (Ablation)**: Executes the ablation experiments (mask-based and optionally retrain-based).
7. **Phase 5 (Results)**:
   - Generates all PDF plots.
   - Performs statistical testing using the **Wilcoxon signed-rank test** with **Holm-Bonferroni correction**.
   - Saves all numerical results to `results.json`.
