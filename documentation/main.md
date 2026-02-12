# Pipeline Execution and Logic (`main.py`)

This document explains the 6-phase execution flow of the SOGNN electrode ablation pipeline, including data handling, importance metrics, and statistical validation.

## Phase 0: Data Loading and Preprocessing

The pipeline uses a **"one trial = one sample"** approach, consistent with the original SOGNN paper.

1. **Feature Extraction**: Loads Differential Entropy (DE) features from `.mat` files.
2. **Normalization**: Computes z-score statistics (mean, std) across all frames in a session *before* padding.
3. **Temporal Alignment**: Trials are zero-padded or truncated to `T_FIXED = 64` time frames.
4. **Reshaping**: Output shape per sample is `(62, 5, 64)` (Channels, Bands, Time).

## Phase 1 & 2: Training and Ensemble

To ensure results are statistically stable, the pipeline uses a **Multi-seed Ensemble**.

- **Evaluation Protocol**: Leave-One-Subject-Out (LOSO). For each subject, the model is trained on 14 participants and tested on the 15th.
- **Ensemble**: The LOSO process is repeated `n_seeds` times (default 5). Importance scores and accuracies are averaged across seeds.
- **Training Logic**: Uses Adam optimizer (`lr=1e-5`), early stopping based on training AUC (threshold 0.99) and accuracy, and Cross-Entropy loss.

## Phase 3: Electrode Importance Metrics

The pipeline calculates importance scores for each electrode:

1. **Permutation Importance (PI)**: Measures the drop in model accuracy when a specific channel's data is shuffled across samples. It is model-agnostic and causally valid, capturing the global contribution of the electrode.
2. **Per-Emotion Importance**: PI is calculated for each emotion class separately to identify emotion-specific brain regions.

## Phase 4: Ablation Studies

Ablation quantifies how model performance degrades as information is removed.

- **Mask-based (Instant)**: Applies a binary mask to the input of a pre-trained 62-channel model. Efficient for testing many configurations.
- **Retrain-from-scratch (Ground Truth)**: Re-initializes and trains a new SOGNN model using only the subset of electrodes. This verifies if the model can adapt to a reduced feature space.
- **Progressive Ablation**: Electrodes are removed one by one (or in steps) based on the "Grand Ranking" (averaged PI scores). This generates a curve that reveals the relationship between channel count and accuracy.

## Phase 5: Visualization and Statistics

The results are synthesized into several PDF reports:

- **Topographic Maps**: 2D scalp projections of PI importance scores.
- **Ablation Curves**: Accuracy vs. number of channels, highlighting the **Knee Point** (the point where adding more channels yields diminishing returns).
- **Region/Lobe Tables**: Bar charts and heatmaps comparing different anatomical subsets.
- **Statistical Testing**: **Wilcoxon signed-rank tests** are performed between major configurations (e.g., Full 62ch vs. 10-20 system) with **Holm-Bonferroni correction** for multiple comparisons.

## Command-Line Interface

```bash
python main.py [options]
```

- `--data_root`: Path to the SEED-IV features.
- `--n_seeds`: Number of ensemble iterations (default: 5).
- `--retrain_ablation`: Enable the expensive retrain-from-scratch experiments.
- `--notebook`: Formats progress bars for Jupyter/Colab environments.
