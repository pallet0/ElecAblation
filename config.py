"""Constants and electrode mappings for SEED-IV ablation study."""

# ── SEED-IV channel layout (62 channels, 0-indexed) ──
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
N_CHANNELS = 62
N_BANDS = 5      # delta, theta, alpha, beta, gamma
N_CLASSES = 4    # 0=neutral, 1=sad, 2=fear, 3=happy
N_SUBJECTS = 15
N_SESSIONS = 3

DATA_ROOT = r'C:\Users\palle\Desktop\QIProject\ResStab_SEED-IV\SEED_IV\eeg_feature_smooth'

# Labels per session (from SEED-IV ReadMe)
SESSION_LABELS = [
    [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
]

# ── Non-overlapping regional groupings (8 strips, anterior→posterior) ──
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

# ── Hemisphere groupings ──
HEMISPHERES = {
    'left':    [0,3,5,6,7,8,14,15,16,17,23,24,25,26,
                32,33,34,35,41,42,43,44,50,51,52,57,58],
    'midline': [1,9,18,27,36,45,53,59],
    'right':   [2,4,10,11,12,13,19,20,21,22,28,29,30,31,
                37,38,39,40,46,47,48,49,54,55,56,60,61],
}

# ── Standard montage subsets ──
STANDARD_1020 = [0,2,5,7,9,11,13,23,25,27,29,31,41,43,45,47,49,58,60]
EMOTIV_EPOC   = [3,4,5,7,13,11,15,21,23,31,41,49,58,60]
MUSE_APPROX   = [3,4,32,40]

# ── MNE name mapping for topographic plotting ──
MNE_NAME_MAP = {
    'FP1': 'Fp1', 'FPZ': 'Fpz', 'FP2': 'Fp2',
    'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz',
    'PZ': 'Pz', 'POZ': 'POz', 'OZ': 'Oz',
    'CB1': 'PO9', 'CB2': 'PO10',
}
