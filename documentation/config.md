# Configuration and Mappings (`config.py`)

This document details the constants, electrode layouts, and regional groupings used to ensure consistency across the SOGNN pipeline.

## Dataset Constants (SEED-IV)

| Constant | Value | Description |
| :--- | :--- | :--- |
| `N_CHANNELS` | 62 | Total number of EEG channels. |
| `N_BANDS` | 5 | Frequency bands: Delta, Theta, Alpha, Beta, Gamma. |
| `N_CLASSES` | 4 | Emotion classes: Neutral (0), Sad (1), Fear (2), Happy (3). |
| `N_SUBJECTS` | 15 | Total participants in the dataset. |
| `N_SESSIONS` | 3 | Separate recording sessions per participant. |
| `T_FIXED` | 64 | Temporal padding length for SOGNN input (64 frames). |

## Electrode Layout

`CHANNEL_NAMES` defines the 62-channel sequence as provided in the SEED-IV `.mat` files. These indices are used throughout the pipeline for masking and importance ranking.

```python
CHANNEL_NAMES = [
    'FP1','FPZ','FP2','AF3','AF4',                          # 0-4
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',           # 5-13
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',  # 14-22
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',           # 23-31
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',  # 32-40
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',           # 41-49
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',              # 50-56
    'CB1','O1','OZ','O2','CB2'                               # 57-61
]
```

## Regional and Anatomical Groupings

These groupings are used in Phase 4 (Ablation Study) to evaluate the impact of removing or keeping specific brain areas.

### 1. Fine-grained Regions (`REGIONS_FINE`)
8 non-overlapping longitudinal strips from anterior to posterior:
- `prefrontal`, `frontal`, `frontal_central`, `central`, `central_parietal`, `parietal`, `parietal_occipital`, `occipital`.

### 2. Hemispheres (`HEMISPHERES`)
- `left`: 27 channels.
- `midline`: 8 channels.
- `right`: 27 channels.

### 3. Anatomical Lobes (`LOBES`)
Groupings based on standard brain anatomy:
- `frontal`: Fp + AF + F electrodes.
- `temporal`: FT7/8, T7/8, TP7/8.
- `central`: FC, C, CP electrodes.
- `parietal`: P electrodes.
- `occipital`: PO, CB, O electrodes.

## Montage Subsets

Commercial and standard electrode configurations used for comparison:
- `STANDARD_1020`: International 10-20 system (19 channels).
- `EMOTIV_EPOC`: 14 channels used by Emotiv headsets.
- `MUSE_APPROX`: 4 channels approximating the Muse headband.

## Visualization Metadata

- `MNE_NAME_MAP`: Dictionary mapping SEED-IV names (e.g., `FPZ`) to standard MNE-compatible names (e.g., `Fpz`) for accurate topographic plotting.
- `SESSION_LABELS`: The ground-truth emotion labels for each of the 24 trials across the 3 sessions, as defined in the SEED-IV ReadMe.
