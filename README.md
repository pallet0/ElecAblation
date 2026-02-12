# SOGNN EEG Electrode Ablation Study (SEED-IV)

This project implements an advanced pipeline for EEG electrode ablation studies on the SEED-IV dataset using a **Self-Organized Graph Neural Network (SOGNN)**. It aims to identify the most critical brain regions and electrode configurations for emotion recognition while maintaining a high-performance, interpretable model.

## Project Overview

The core objective is to move beyond "black-box" classification by employing dynamic graph learning and rigorous ablation testing. The pipeline quantifies the contribution of individual electrodes and anatomical regions (lobes, hemispheres) to the classification of four emotions: neutral, sad, fear, and happy.

### Key Features

- **SOGNN Architecture**: Implements a multi-scale CNN-GCN hybrid that dynamically learns electrode adjacency matrices from data.
- **Importance Metrics**:
  - **Permutation Importance (PI)**: Model-agnostic method measuring accuracy drop when channel features are shuffled.
  - **Per-Emotion Importance**: Analyzes which electrodes are most predictive for specific emotions.
- **Rigorous Ablation Testing**:
  - **Mask-based Ablation**: Instant evaluation of pre-trained models with specific electrodes zeroed out.
  - **Retrain-from-scratch Ablation**: Re-training models on reduced electrode sets to verify actual performance potential.
  - **Progressive Ablation**: Incrementally removing electrodes based on importance rankings to find the "knee point" (optimal trade-off).
- **Statistical Validation**: Uses Wilcoxon signed-rank tests with Holm-Bonferroni correction to compare electrode configurations.
- **Rich Visualization**: Automated generation of topographic heatmaps, progressive ablation curves, and region-based performance tables.

## Project Structure

```text
.
├── main.py                # Main execution pipeline (Phases 0-5)
├── models.py              # SOGNN and SOGC layer implementations
├── config.py              # Electrode mappings, regions, and constants
├── documentation/         # Detailed technical and academic documentation
│   ├── config.md          # Constants and mapping details
│   ├── main.md            # Pipeline and execution logic
│   ├── models.md          # Architecture details
│   └── ko/                # Korean translations and academic justification
├── SEED_IV/               # Dataset directory (user-provided)
└── SOGNN_PIPELINE_SPEC.md # Technical implementation specification
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch, MNE, Scipy, Numpy, Matplotlib, Scikit-learn, Tqdm

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

The pipeline is structured into phases, from data loading to final visualization.

```bash
# Basic run with 5-seed ensemble (default)
python main.py --data_root ./SEED_IV/eeg_feature_smooth

# Fast run with a single seed and no retraining
python main.py --data_root ./SEED_IV/eeg_feature_smooth --n_seeds 1

# Comprehensive run including retrain-from-scratch ablation
python main.py --data_root ./SEED_IV/eeg_feature_smooth --retrain_ablation
```

## Documentation

For more detailed information, please refer to the files in the `documentation/` folder:

1. [**Configuration Guide**](documentation/config.md): Electrode indices, regional groupings, and dataset constants.
2. [**Pipeline Guide**](documentation/main.md): Explanation of the 6-phase execution flow, importance methods, and ablation logic.
3. [**Model Guide**](documentation/models.md): Deep dive into the SOGNN architecture and dynamic graph learning.
4. [**Academic Justification (KO)**](documentation/ko/justification.md): Literature review and theoretical basis for the interpretability-accuracy trade-off.

---
*Reference: Li et al., "Cross-Subject EEG Emotion Recognition With Self-Organized Graph Neural Network," Frontiers in Neuroscience, 2021.*
