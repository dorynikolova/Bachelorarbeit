import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.io import loadmat
from scipy.signal import welch, hilbert
import scipy.signal as sps
import mne

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.svm import SVC

# =========================================================
# CONFIG
# =========================================================

SFREQ = 1000
DS_FACTOR = 4
FEATURE_DIR = Path("Bachelorarbeit/features_subject_spectral_hilbert_5_50hz_250Hz")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {
    "delta": (1, 4),
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
    # x shape: (channels, samples)
    return mne.filter.filter_data(
        x, sfreq=sfreq, l_freq=1, h_freq=50, verbose=False
    )

def downsample(x, factor):
    # x shape: (channels, samples)
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

def spectral_entropy_from_psd(psd):
    psd = np.asarray(psd, dtype=np.float64)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    se /= np.log2(len(psd_norm) + 1e-12)
    return float(se)

def hilbert_variance(sig):
    env = np.abs(hilbert(sig))
    return float(np.var(env))

def safe_ratio(a, b):
    return a / (b + 1e-12)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_channel_features(sig, fs):
    freqs, psd = compute_psd(sig, fs)

    abs_p = {
        band: band_power(freqs, psd, fmin, fmax)
        for band, (fmin, fmax) in BANDS.items()
    }
    rel_p = relative_band_powers(freqs, psd, BANDS)

    spec_ent = spectral_entropy_from_psd(psd)
    hvar = hilbert_variance(sig)

    return np.array([
        # absolute band powers
        abs_p["delta"],
        abs_p["theta"],
        abs_p["alpha"],
        abs_p["beta"],
        abs_p["gamma"],

        # relative band powers
        rel_p["delta"],
        rel_p["theta"],
        rel_p["alpha"],
        rel_p["beta"],
        rel_p["gamma"],

        # ratios
        safe_ratio(rel_p["delta"], rel_p["alpha"]),
        safe_ratio(rel_p["theta"], rel_p["alpha"]),
        safe_ratio(rel_p["beta"], rel_p["alpha"]),
        safe_ratio(rel_p["gamma"], rel_p["alpha"]),

        # other features
        spec_ent,
        hvar,
    ], dtype=np.float32)

"""def summarize_channelwise_features(channel_feat_matrix):
    #channel_feat_matrix: (n_channels, n_features_per_channel)
    #returns mean and std across channels
    
    mean_feats = channel_feat_matrix.mean(axis=0)
    std_feats = channel_feat_matrix.std(axis=0)
    return np.hstack([mean_feats, std_feats]).astype(np.float32)"""

def summarize_channelwise_features(channel_feat_matrix):
    return channel_feat_matrix.ravel().astype(np.float32)

def preprocess_signal(X, sfreq=SFREQ, ds_factor=DS_FACTOR):
    # X shape: (channels, samples)
    Xf = bandpass_5_50(X, sfreq=sfreq)

    if ds_factor > 1:
        Xf = downsample(Xf, ds_factor)
        sfreq_new = sfreq // ds_factor
    else:
        sfreq_new = sfreq

    Xf = zscore_per_channel(Xf)
    return Xf, sfreq_new

def extract_features_from_signal(X, sfreq=SFREQ, ds_factor=DS_FACTOR):
    """
    X shape: (channels, samples)
    returns one subject-level feature vector
    """
    Xf, sfreq_new = preprocess_signal(X, sfreq=sfreq, ds_factor=ds_factor)

    channel_feats = []
    for ch in range(Xf.shape[0]):
        channel_feats.append(extract_channel_features(Xf[ch], sfreq_new))

    channel_feats = np.vstack(channel_feats)
    feats = summarize_channelwise_features(channel_feats)
    return feats

def detect_label_mapping(y):
    """
    Your old code was inconsistent about open/closed.
    This function only checks that labels are binary.
    You still need to confirm once whether 0=open or 1=open.
    """
    vals = np.unique(y)
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f"Expected binary labels 0/1, got {vals}")
    return vals

def extract_subject_features(mat_path, eeg_id, sfreq=SFREQ, ds_factor=DS_FACTOR):
    feat_file = FEATURE_DIR / f"{eeg_id}.npz"

    if feat_file.exists():
        data = np.load(feat_file)
        return data["open"], data["closed"]

    m = loadmat(mat_path)
    Xraw = m["X"]
    y = m["y"].squeeze()

    detect_label_mapping(y)

    # make sure X is (channels, samples)
    X = Xraw if Xraw.shape[0] < Xraw.shape[1] else Xraw.T

    # IMPORTANT:
    # verify once whether 0=open and 1=closed in YOUR files.
    # I use the convention below because your later windowed pipeline used:
    # open -> 0, closed -> 1
    X_open = X[:, y == 0]
    X_closed = X[:, y == 1]

    if X_open.shape[1] == 0 or X_closed.shape[1] == 0:
        raise ValueError(f"{eeg_id}: one condition is empty. y unique/counts = {np.unique(y, return_counts=True)}")

    feats_open = extract_features_from_signal(X_open, sfreq=sfreq, ds_factor=ds_factor)
    feats_closed = extract_features_from_signal(X_closed, sfreq=sfreq, ds_factor=ds_factor)

    np.savez_compressed(
        feat_file,
        open=feats_open,
        closed=feats_closed
    )
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
# BUILD X/Y
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

def run_nested_cv(subject_ids, task_name, condition, outer_seed=42):
    print(f"\n========== {task_name} | {condition} ==========")

    subj_labels = np.array([id2label[s] for s in subject_ids])

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=15000,
            tol=1e-3
        )
    )

    """pipe = make_pipeline(
        StandardScaler(),
        PCA(random_state=42),
        SVC(kernel="linear", class_weight="balanced") # tried linear - AD RO2 doesnt have a stable signal, but rbf is compensating with complexity; with linear first and last are better.
    )

    param_grid = {
        "pca__n_components": [5, 10, 15, 20],
        "svc__C": [0.5, 1, 2, 3, 4, 5, 10],
        "svc__gamma": ["scale", 0.01, 0.001],
    }"""

    param_grid = {
        "logisticregression__C": [0.01, 0.1, 1.0, 10.0],
        "logisticregression__l1_ratio": [0.2, 0.5, 0.8],
    }

    """pipe = make_pipeline(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )

    param_grid = {
        "randomforestclassifier__n_estimators": [200, 500],
        "randomforestclassifier__max_depth": [None, 5, 10],
        "randomforestclassifier__min_samples_split": [2, 5],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4],
        "randomforestclassifier__max_features": ["sqrt", 0.5],
    }"""

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)

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
        print("  test balanced accuracy:", fold_score)

    inner_scores = np.array(inner_scores)
    outer_scores = np.array(outer_scores)

    print("\nAggregated confusion matrix:")
    print(cm_total)
    print(f"{task_name} results:")
    print("Inner CV mean :", inner_scores.mean())
    print("Outer CV mean :", outer_scores.mean())
    print("Outer CV std  :", outer_scores.std())
    print("Generalization gap:", inner_scores.mean() - outer_scores.mean())

    return outer_scores

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


# rbf maps data into very high dimensional space, which i dont have; PCA was reducing dimensionallity, keeping directions with maximum variance, but eeg disease often lie on low variance directions; harmful for linear models
# .sav pokazva po-malko bolesti otkolkoto pishe v maila
# matematikata ne izliza
# channels = number of electrodes (e.g. 61), samples = number of time points (e.g. 490 916 samples)
# check for label noise or errors
#CSP/LDA and random forest and riemann
# einzeln probieren - ohne dwt
# ICA and lda da probvam
# entropy with windows
#LZC and DFA
# data clean .mat hat eyes offen und closed
# Y raw ist es =1 =2

############################################################################## DONE #################################################################################################################
# relative band power - controls for overall signal amplitude differences between subjects - alpha, beta, theta, delta band power to the dwt - DONE
# AD und WY separat
# Linear SVC und logistic regression instead of rbf SVC
# tried also without PCA
# linearSVM - less risk of overfitting, good with PCA, fast and interpretable weights - DONE
# logistic regression with elastic net - linear with probabilistic output; combination of L1 and L2 regularization; for many correlated features - DONE
# 1 - 40Hz instead of 5 - 50 - Delta and theta rhythms (1–8 Hz) are often altered in disorders; by including 1–4 Hz, we capture those slow oscillations; frequencies above 40 Hz often contain muscle artifacts or line noise -> cleaner features - DIDNT HELP

############################################################################## Next Possible Steps #################################################################################################################
# 1. spectral entropy - a measure of how “disordered” the power spectrum of an EEG signal is, low spectral entropy - power concentrated in a few frequencies
# 2. ICA for artifact removal (eye blinks, muscle activity, heartbeats) before band power/DWT - decomposes EEG into statistically independent sources; separates artifacts into distinct components; cleaner signals -> better features 
# slower band features - instead of focusing only on mid‑range bands (alpha, beta), we compute features from slow oscillations: delta (0.5–4 Hz): Deep sleep, pathology markers; theta (4–8 Hz): Memory, drowsiness, often increased in Alzheimer’s; many neurological conditions show increased slow‑wave power (more delta/theta) and decreased fast‑wave power (less alpha/beta).
# permutation entropy - healthy brains have higher entropy
# sample entropy - a measure of signal irregularity, based on the probability that similar patterns remain similar when extended; low - more regular, predictable signal
# using LDA after PCA? - LDA finds directions that maximize class separation; supervised dimensionality reduction method; are classes linearly separable? is it useful for eegs?
# PCA(n_components=0.95)?
# Bayesianische logistische Regression instead of SVC? - posterior distributions, gives uncertainty estimates, better regularization and says more about the features?

# methode, resultate, diskussion, einleitung

# zweites, unabhängiges test?
# Gaussian Processes instead of SVC?
# coherence between channels - how consistently two eeg channels oscillate together at a given frequency (high coherence = strong functional connectivity between brain regions)
# correlation between channels?
# phase lag index (PLI) - Looks at whether one signal consistently leads/lags another in phase (phase synchronisation) - clinical relevance
# more folds for cv? - more reliable performance estimates, since each subject is tested more often; trade-off - smaller test sets per fold, so variance in each fold’s score increases; usually 10-fold cv
# random forest - decision trees trained on random subsets of data/features, feature importance scores; robust to noise; LESS STABLE WITH SMALL DATASETS
# or PCA Whitening - decorrelates features, but scales them by variance; equalizes feature scales; making features more balanced


