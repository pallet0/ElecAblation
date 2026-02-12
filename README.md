# SOGNN EEG Electrode Ablation Study (SEED-IV)

This project implements a pipeline for EEG electrode ablation studies on the SEED-IV dataset using a **Self-Organized Graph Neural Network (SOGNN)**(Jingcong Li et al., 2021).

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Preparation
SEED-IV. Please use `--data_root` argument to locate the target folder.

### Running the Pipeline

The pipeline is structured into phases, from data loading to final visualization.

```bash
# Basic run with 5-seed ensemble (default)
python main.py --data_root ./SEED_IV/eeg_feature_smooth

# Fast run with a single seed and no retraining
python main.py --data_root ./SEED_IV/eeg_feature_smooth --n_seeds 1
```
