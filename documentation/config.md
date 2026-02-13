# Configuration and Mappings (`config.py`)

This document details the constants, electrode layouts, and regional groupings used to ensure consistency across the SOGNN pipeline.

## Dataset Constants (SEED-IV)

The following constants define the dimensions and parameters of the SEED-IV dataset as used in this study.

```python
N_CHANNELS = 62
N_BANDS = 5      # delta, theta, alpha, beta, gamma
N_CLASSES = 4    # 0=neutral, 1=sad, 2=fear, 3=happy
N_SUBJECTS = 15
N_SESSIONS = 3
T_FIXED = 64     # temporal padding length (SOGNN paper SEED-IV setting)
```

- `T_FIXED`: Trials in SEED-IV have varying lengths. Following the SOGNN protocol, all trials are zero-padded or truncated to exactly 64 frames.

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

These groupings are used in Phase 4 (Ablation Study) to evaluate the impact of removing or keeping specific brain areas. They are defined as lists of indices corresponding to `CHANNEL_NAMES`.

### 1. Fine-grained Regions (`REGIONS_FINE`)
Divided into 8 longitudinal strips from anterior to posterior to test sensitivity to "depth" in the brain.

```python
REGIONS_FINE = {
    'prefrontal':        [0, 1, 2, 3, 4],
    'frontal':           [5, 6, 7, 8, 9, 10, 11, 12, 13],
    'frontal_central':   [14, 15, 16, 17, 18, 19, 20, 21, 22],
    'central':           [23, 24, 25, 26, 27, 28, 29, 30, 31],
    'central_parietal':  [32, 33, 34, 35, 36, 37, 38, 39, 40],
    'parietal':          [41, 42, 43, 44, 45, 46, 47, 48, 49],
    'parietal_occipital':[50, 51, 52, 53, 54, 55, 56],
    'occipital':         [57, 58, 59, 60, 61],
}
```

### 2. Hemispheres (`HEMISPHERES`)
Used to test lateralization effects.

```python
HEMISPHERES = {
    'left':    [0,3,5,6,7,8,14,15,16,17,23,24,25,26,
                32,33,34,35,41,42,43,44,50,51,52,57,58],
    'midline': [1,9,18,27,36,45,53,59],
    'right':   [2,4,10,11,12,13,19,20,21,22,28,29,30,31,
                37,38,39,40,46,47,48,49,54,55,56,60,61],
}
```

### 3. Anatomical Lobes (`LOBES`)
Groupings based on standard brain anatomy.

```python
LOBES = {
    'frontal':   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'temporal':  [14, 22, 23, 31, 32, 40],
    'central':   [15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39],
    'parietal':  [41, 42, 43, 44, 45, 46, 47, 48, 49],
    'occipital': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
}
```

## Montage Subsets

Commercial and standard electrode configurations used for comparison. These allow the model to be evaluated on subsets of electrodes that might be available in consumer-grade EEG hardware.

```python
STANDARD_1020 = [0,2,5,7,9,11,13,23,25,27,29,31,41,43,45,47,49,58,60]
EMOTIV_EPOC   = [3,4,5,7,13,11,15,21,23,31,41,49,58,60]
MUSE_APPROX   = [3,4,32,40]
```

## Visualization Metadata

### `MNE_NAME_MAP`
Dictionary mapping SEED-IV names to standard MNE-compatible names. This is critical for generating correct topographic maps, as MNE needs standard names (e.g., `Fpz` instead of `FPZ`) to look up electrode coordinates.

### `SESSION_LABELS`
The ground-truth emotion labels (0=Neutral, 1=Sad, 2=Fear, 3=Happy) for each of the 24 trials across the 3 sessions.
