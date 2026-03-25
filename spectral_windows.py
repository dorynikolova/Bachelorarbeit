import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.io import loadmat
from scipy.signal import welch
import scipy.signal as sps
import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# =========================================================
# CONFIG
# =========================================================

SFREQ = 1000
DS_FACTOR = 4
FEATURE_DIR = Path("Bachelorarbeit/features_subject_window_aggregated_spectral_5_50hz_250Hz")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 50),
}

# =========================================================
# HELPERS
# =========================================================

def get_task(eeg_id):
    eeg_id = str(eeg_id).upper()
    if eeg_id.startswith("AD"):
        return "AD"
    elif eeg_id.startswith("WY"):
        return "WY"
    return "UNKNOWN"

def bandpass_5_50(x, sfreq=SFREQ):
    return mne.filter.filter_data(x, sfreq=sfreq, l_freq=5, h_freq=50, verbose=False)

def downsample(x, factor):
    return sps.decimate(x, factor, axis=1, ftype="iir", zero_phase=True)

def zscore_per_channel(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)

def compute_psd(sig, fs):
    nperseg = min(len(sig), fs * 2)
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    return freqs, psd

def band_power(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapezoid(psd[idx], freqs[idx])

def relative_band_powers(freqs, psd, bands):
    abs_powers = {
        band: band_power(freqs, psd, fmin, fmax)
        for band, (fmin, fmax) in bands.items()
    }
    total_power = np.sum(list(abs_powers.values())) + 1e-12
    return {band: p / total_power for band, p in abs_powers.items()}

def make_windows(X, win_len, step):
    windows = []
    n_samples = X.shape[1]
    for start in range(0, n_samples - win_len + 1, step):
        stop = start + win_len
        windows.append(X[:, start:stop])
    return windows

def split_into_contiguous_blocks(X, y, target_label):
    blocks = []
    start = None

    for i, label in enumerate(y):
        if label == target_label and start is None:
            start = i
        elif label != target_label and start is not None:
            blocks.append(X[:, start:i])
            start = None

    if start is not None:
        blocks.append(X[:, start:len(y)])

    return blocks

# =========================================================
# FEATURES
# =========================================================

def extract_features_from_signal_spectral(X_window, sfreq):
    """
    Returns one feature vector for one window.
    X_window shape: (channels, samples)
    """
    all_feats = []

    for ch in range(X_window.shape[0]):
        sig = X_window[ch]
        freqs, psd = compute_psd(sig, sfreq)
        rel_p = relative_band_powers(freqs, psd, BANDS)

        feats = np.array([
            rel_p["theta"],
            rel_p["alpha"],
            rel_p["beta"],
            rel_p["gamma"],
            rel_p["theta"] / (rel_p["alpha"] + 1e-12),
            rel_p["beta"] / (rel_p["alpha"] + 1e-12),
            rel_p["gamma"] / (rel_p["alpha"] + 1e-12),
        ], dtype=np.float32)

        all_feats.append(feats)

    return np.hstack(all_feats)

def aggregate_window_features(W):
    """
    W shape: (n_windows, n_window_features)
    Returns one subject-level vector.
    """
    return np.hstack([
        W.mean(axis=0),
        W.std(axis=0),
        np.median(W, axis=0),
        np.percentile(W, 25, axis=0),
        np.percentile(W, 75, axis=0),
    ]).astype(np.float32)

def preprocess_signal(X, sfreq=SFREQ, ds_factor=DS_FACTOR):
    Xf = bandpass_5_50(X, sfreq=sfreq)

    if ds_factor > 1:
        Xf = downsample(Xf, ds_factor)
        sfreq_new = sfreq // ds_factor
    else:
        sfreq_new = sfreq

    Xf = zscore_per_channel(Xf)
    return Xf, sfreq_new

def extract_subject_features(mat_path, eeg_id, sfreq=SFREQ, ds_factor=DS_FACTOR,
                             win_sec=4.0, overlap=0.25):
    """
    Returns two subject-level vectors:
    - open aggregated over windows
    - closed aggregated over windows
    """
    feat_file = FEATURE_DIR / f"{eeg_id}.npz"

    if feat_file.exists():
        data = np.load(feat_file)
        return data["open"], data["closed"]

    m = loadmat(mat_path)
    Xraw = m["X"]
    y = m["y"].squeeze()

    vals = np.unique(y)
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f"{eeg_id}: expected binary labels 0/1, got {vals}")

    X = Xraw if Xraw.shape[0] < Xraw.shape[1] else Xraw.T

    # Keep consistent with your project
    open_blocks = split_into_contiguous_blocks(X, y, target_label=0)
    closed_blocks = split_into_contiguous_blocks(X, y, target_label=1)

    open_window_features = []
    closed_window_features = []

    for block in open_blocks:
        Xf, sfreq_new = preprocess_signal(block, sfreq=sfreq, ds_factor=ds_factor)

        win_len = int(win_sec * sfreq_new)
        step = int(win_len * (1 - overlap))
        if step <= 0:
            raise ValueError("overlap too large; step <= 0")

        windows = make_windows(Xf, win_len, step)

        if windows:
            block_feats = np.vstack([
                extract_features_from_signal_spectral(win, sfreq_new)
                for win in windows
            ])
            open_window_features.append(block_feats)

    for block in closed_blocks:
        Xf, sfreq_new = preprocess_signal(block, sfreq=sfreq, ds_factor=ds_factor)

        win_len = int(win_sec * sfreq_new)
        step = int(win_len * (1 - overlap))
        if step <= 0:
            raise ValueError("overlap too large; step <= 0")

        windows = make_windows(Xf, win_len, step)

        if windows:
            block_feats = np.vstack([
                extract_features_from_signal_spectral(win, sfreq_new)
                for win in windows
            ])
            closed_window_features.append(block_feats)

    if len(open_window_features) == 0:
        raise ValueError(f"No open windows created for {eeg_id}")
    if len(closed_window_features) == 0:
        raise ValueError(f"No closed windows created for {eeg_id}")

    W_open = np.vstack(open_window_features)
    W_closed = np.vstack(closed_window_features)

    feats_open = aggregate_window_features(W_open)
    feats_closed = aggregate_window_features(W_closed)

    np.savez_compressed(feat_file, open=feats_open, closed=feats_closed)
    return feats_open, feats_closed

# =========================================================
# LOAD LABELS
# =========================================================

df, meta = pyreadstat.read_sav(
    "Bachelorarbeit/Clinical/ADWY_Clinical_Data.sav",
    usecols=["EEG_ID", "Clinical"],
    apply_value_formats=False,
    formats_as_category=False
)

df["EEG_ID"] = df["EEG_ID"].astype(str).str.strip()
df["Clinical"] = pd.to_numeric(df["Clinical"], errors="coerce")

labels_df = (
    df.dropna(subset=["EEG_ID", "Clinical"])
      .drop_duplicates(subset=["EEG_ID"])[["EEG_ID", "Clinical"]]
)
labels_df["Clinical"] = labels_df["Clinical"].astype("int8")

print("Clinical label counts:")
print(labels_df["Clinical"].value_counts())

mat_paths = sorted(glob.glob("Bachelorarbeit/dataclean_2/dataclean/*.mat"))
mat_df = pd.DataFrame({
    "mat_path": mat_paths,
    "EEG_ID": [Path(p).stem for p in mat_paths]
})

data_index = mat_df.merge(labels_df, on="EEG_ID", how="inner")
data_index["Task"] = data_index["EEG_ID"].apply(get_task)

id2path = dict(zip(data_index["EEG_ID"], data_index["mat_path"]))
id2label = dict(zip(data_index["EEG_ID"], data_index["Clinical"]))

# =========================================================
# BUILD SUBJECT MATRICES
# =========================================================

def make_xy(subj_list, condition):
    X, y = [], []

    for sid in subj_list:
        feats_open, feats_closed = extract_subject_features(id2path[sid], sid)

        if condition == "open":
            X.append(feats_open)
        elif condition == "closed":
            X.append(feats_closed)
        else:
            raise ValueError("condition must be 'open' or 'closed'")

        y.append(id2label[sid])

    return np.vstack(X), np.array(y, dtype=int)

# =========================================================
# NESTED CV
# =========================================================

def run_nested_cv(subject_ids, task_name, condition, seeds=(42,)):
    print(f"\n========== {task_name} | {condition} ==========")
    subj_labels = np.array([id2label[s] for s in subject_ids])

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            tol=1e-3
        )
    )

    param_grid = {
        "logisticregression__C": [0.001, 0.01, 0.1, 1.0],
        "logisticregression__l1_ratio": [0.2, 0.5, 0.8]
    }

    all_seed_means = []

    for seed in seeds:
        print(f"\n--- Random state {seed} ---")
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        inner_scores = []
        outer_scores = []
        cm_total = np.zeros((2, 2), dtype=int)

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(subject_ids, subj_labels), 1):
            print(f"Outer fold {fold}")

            train_subjects = subject_ids[train_idx]
            test_subjects = subject_ids[test_idx]

            X_train, y_train = make_xy(train_subjects, condition)
            X_test, y_test = make_xy(test_subjects, condition)

            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=inner_cv,
                scoring="balanced_accuracy",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            fold_score = balanced_accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            cm_total += cm
            inner_scores.append(grid.best_score_)
            outer_scores.append(fold_score)

            print("  best params:", grid.best_params_)
            print("  subject-level test balanced accuracy:", fold_score)

        inner_scores = np.array(inner_scores)
        outer_scores = np.array(outer_scores)

        print("\nAggregated subject-level confusion matrix:")
        print(cm_total)
        print(f"Seed {seed} results:")
        print("Inner CV mean :", inner_scores.mean())
        print("Outer CV mean :", outer_scores.mean())
        print("Outer CV std  :", outer_scores.std())
        print("Generalization gap:", inner_scores.mean() - outer_scores.mean())

        all_seed_means.append(outer_scores.mean())

    all_seed_means = np.array(all_seed_means)
    print("\nAcross-seed summary:")
    print("Mean outer CV across seeds:", all_seed_means.mean())
    print("Std across seeds:", all_seed_means.std())

    return all_seed_means

# =========================================================
# RUN
# =========================================================

ad_subjects = data_index.loc[data_index["Task"] == "AD", "EEG_ID"].values
wy_subjects = data_index.loc[data_index["Task"] == "WY", "EEG_ID"].values

print("AD subjects:", len(ad_subjects))
print("WY subjects:", len(wy_subjects))

scores_ad_open = run_nested_cv(ad_subjects, "AD", "open")
scores_wy_open = run_nested_cv(wy_subjects, "WY", "open")

scores_ad_closed = run_nested_cv(ad_subjects, "AD", "closed")
scores_wy_closed = run_nested_cv(wy_subjects, "WY", "closed")