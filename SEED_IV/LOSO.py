import os
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ====== labels from ReadMe (trial 1..24) ======
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# label mapping: 0 neutral, 1 sad, 2 fear, 3 happy
LABELS_BY_SESSION = {1: session1_label, 2: session2_label, 3: session3_label}

def load_subject_session(
    session: int,
    subject_file: str,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
    eeg_key_prefix="de_LDS",
):
    """
    Load ONE subject's ONE session data from:
      EEG: {eeg_root}/{session}/{subject_file}
      EYE: {eye_root}/{session}/{subject_file}

    Expects trial keys:
      EEG: de_LDS1..de_LDS24 each (62, W, 5)
      EYE: eye_1..eye_24     each (31, W)

    Returns:
      X: (num_windows_total, 341)   [EEG(310) + EYE(31)]
      y: (num_windows_total,)
      groups: (num_windows_total,)  subject_id repeated
    """
    eeg_path = os.path.join(eeg_root, str(session), subject_file)
    eye_path = os.path.join(eye_root, str(session), subject_file)

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    if not os.path.exists(eye_path):
        raise FileNotFoundError(f"Eye file not found: {eye_path}")

    eeg = sio.loadmat(eeg_path)
    eye = sio.loadmat(eye_path)

    # subject id: "1_20160518.mat" -> 1
    try:
        subj_id = int(subject_file.split("_")[0])
    except Exception as e:
        raise ValueError(f"Cannot parse subject id from filename: {subject_file}") from e

    labels = LABELS_BY_SESSION[session]  # length 24

    X_list, y_list, g_list = [], [], []

    for i in range(1, 25):
        eeg_key = f"{eeg_key_prefix}{i}"  # de_LDS1...
        eye_key = f"eye_{i}"              # eye_1...

        if eeg_key not in eeg:
            raise KeyError(f"Missing key '{eeg_key}' in {eeg_path}")
        if eye_key not in eye:
            raise KeyError(f"Missing key '{eye_key}' in {eye_path}")

        Xe = eeg[eeg_key]  # (62, W, 5)
        Xo = eye[eye_key]  # (31, W)

        if Xe.ndim != 3:
            raise ValueError(f"{eeg_key} expected 3D (62,W,5) but got {Xe.shape}")
        if Xo.ndim != 2:
            raise ValueError(f"{eye_key} expected 2D (31,W) but got {Xo.shape}")

        # alignment check: same W
        if Xe.shape[1] != Xo.shape[1]:
            raise ValueError(f"Window mismatch trial {i}: EEG {Xe.shape} vs EYE {Xo.shape}")

        W = Xe.shape[1]

        # EEG (62,W,5) -> (W, 310)
        Xe_w = np.transpose(Xe, (1, 0, 2)).reshape(W, -1)

        # EYE (31,W) -> (W,31)
        Xo_w = Xo.T

        # concat -> (W,341)
        Xw = np.concatenate([Xe_w, Xo_w], axis=1)

        yi = labels[i - 1]
        yv = np.full(W, yi, dtype=np.int64)
        gv = np.full(W, subj_id, dtype=np.int64)

        X_list.append(Xw)
        y_list.append(yv)
        g_list.append(gv)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return X, y, groups


def load_session_all_subjects(
    session: int,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
):
    """
    Load ALL subjects for a given session.
    Uses EEG folder listing as the canonical file list, and assumes
    the same filenames exist in the eye folder.

    Returns:
      X: (N,341), y: (N,), groups: (N,)
    """
    eeg_folder = os.path.join(eeg_root, str(session))
    eye_folder = os.path.join(eye_root, str(session))

    if not os.path.isdir(eeg_folder):
        raise FileNotFoundError(f"EEG session folder not found: {eeg_folder}")
    if not os.path.isdir(eye_folder):
        raise FileNotFoundError(f"Eye session folder not found: {eye_folder}")

    files = [f for f in os.listdir(eeg_folder) if f.endswith(".mat")]
    files.sort()

    if len(files) == 0:
        raise RuntimeError(f"No .mat files found in {eeg_folder}")

    Xs, ys, gs = [], [], []
    for f in files:
        # optional: ensure eye file exists too
        eye_path = os.path.join(eye_folder, f)
        if not os.path.exists(eye_path):
            raise FileNotFoundError(f"Eye file missing for {f}: {eye_path}")

        X, y, g = load_subject_session(session, f, eeg_root=eeg_root, eye_root=eye_root)
        Xs.append(X)
        ys.append(y)
        gs.append(g)

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(gs, axis=0)


def loso_cross_subject(session: int, n_estimators=500, random_state=42):
    X, y, groups = load_session_all_subjects(session)

    subject_ids = np.unique(groups)
    fold_accs = []

    for test_subj in subject_ids:
        test_mask = (groups == test_subj)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
        rf.fit(X_train, y_train)

        pred = rf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        fold_accs.append(acc)

        print(f"\n=== LOSO | session={session} | test_subject={test_subj} ===")
        print("Accuracy:", acc)
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        print("Report:\n", classification_report(y_test, pred, digits=4))

    print("\n========================")
    print(f"Session {session} LOSO mean acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print("========================")


# 실행
loso_cross_subject(session=1)
