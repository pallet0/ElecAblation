# config.py Documentation

This file contains the constants, electrode mappings, and regional groupings used for the SEED-IV EEG emotion recognition ablation study.

## EEG Channel Layout
- `CHANNEL_NAMES`: A list of 62 EEG electrode names following the SEED-IV layout.
- `N_CHANNELS`: Total number of EEG channels (62).
- `N_BANDS`: Number of frequency bands (5: delta, theta, alpha, beta, gamma) used for Differential Entropy (DE) features.
- `N_CLASSES`: Number of emotion classes (4: neutral, sad, fear, happy).
- `N_SUBJECTS`: Number of subjects in the SEED-IV dataset (15).
- `N_SESSIONS`: Number of sessions per subject (3).
- `T_FIXED`: Temporal padding length (64) for the SOGNN model input.

## Data Configuration
- `DATA_ROOT`: The directory path where the SEED-IV preprocessed features (`eeg_feature_smooth`) are stored.
- `SESSION_LABELS`: A nested list containing the ground truth labels for the 24 trials in each of the 3 sessions.

## Regional Groupings (Electrode Subsets)
These groupings are used to analyze the contribution of different brain areas to emotion recognition.

### `REGIONS_FINE`
A dictionary mapping 8 non-overlapping longitudinal strips (from anterior to posterior) to their corresponding channel indices:
- `prefrontal`, `frontal`, `frontal_central`, `central`, `central_parietal`, `parietal`, `parietal_occipital`, `occipital`.

### `HEMISPHERES`
A dictionary mapping brain hemispheres to channel indices:
- `left`, `right`, and `midline`.

### `LOBES`
Anatomical lobe groupings:
- `frontal`, `temporal`, `central`, `parietal`, `occipital`.

## Standard Montage Subsets
Indices for common reduced electrode configurations:
- `STANDARD_1020`: The international 10-20 system (19 channels).
- `EMOTIV_EPOC`: The 14-channel layout used by the Emotiv EPOC headset.
- `MUSE_APPROX`: A 4-channel approximation of the Muse headband.

## MNE Mapping
- `MNE_NAME_MAP`: Maps SEED-IV channel names to standard MNE names for compatibility with topographic plotting libraries.
