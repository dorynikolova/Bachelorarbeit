from joblib import Parallel, delayed
import numpy as np
from scipy.io import loadmat
import pandas as pd
import pyreadstat
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
import glob
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import scipy.signal as sps
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
import mne, pywt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

def compute_psd(sig, fs): # power spectral density using Welch’s method; robust spectral estimation, reduces noise compared to fft
    freqs, psd = welch(sig, fs=fs, nperseg=fs*2)
    return freqs, psd

def band_power(freqs, psd, fmin, fmax): # integrates PSD (power spectral density) within a frequency band; band power reflects rhytmic brain activity 
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def relative_band_powers(freqs, psd, bands): # normalizing band power by total power; comparable features
    abs_powers = {
        band: band_power(freqs, psd, fmin, fmax)
        for band, (fmin, fmax) in bands.items()
    }
    total_power = np.sum(list(abs_powers.values())) + 1e-12
    return {band: p / total_power for band, p in abs_powers.items()}

def get_rest_order(row):
    if row["Task"] == "AD":
        return row["ADRestOrder"]
    elif row["Task"] == "WY":
        return row["WYRestOrder"]
    else:
        return np.nan

def get_task(eeg_id):
    if eeg_id.upper().startswith("AD"):
        return "AD"
    elif eeg_id.upper().startswith("WY"):
        return "WY"
    else:
        return "UNKNOWN"

FEATURE_DIR = Path("features_dwt_5_50hz_250Hz") # caching the extracted features
FEATURE_DIR.mkdir(exist_ok=True)

# loading clinical information from .sav
df, meta = pyreadstat.read_sav(
    "Clinical/ADWY_Clinical_Data.sav",
    usecols=["EEG_ID", "Clinical", "ADRestOrder", "WYRestOrder"],
    apply_value_formats=False,
    formats_as_category=False
)

df["EEG_ID"] = df["EEG_ID"].astype(str).str.strip()
df["Clinical"] = pd.to_numeric(df["Clinical"], errors="coerce") # normalizing to 1/0 and handling NaN

#labels_df = (df
#             .dropna(subset=["EEG_ID", "Clinical"])
#             .drop_duplicates(subset=["EEG_ID"])[["EEG_ID","Clinical"]])

labels_df = (df
    .dropna(subset=["EEG_ID", "Clinical"])
    .drop_duplicates(subset=["EEG_ID"])
    [["EEG_ID", "Clinical", "ADRestOrder", "WYRestOrder"]]
)

labels_df["Clinical"] = labels_df["Clinical"].astype("int8")
print(labels_df["Clinical"].value_counts())

mat_paths = sorted(glob.glob("dataclean_2/dataclean/*.mat"))

# .mat files with .sav labels 
mat_df = pd.DataFrame({
    "mat_path": mat_paths,
    "EEG_ID": [Path(p).stem for p in mat_paths] 
})

data_index = mat_df.merge(labels_df, on="EEG_ID", how="inner")
data_index["Task"] = data_index["EEG_ID"].apply(get_task)
data_index["RestOrder"] = data_index.apply(get_rest_order, axis=1)
subjects = data_index["EEG_ID"].values
labels   = data_index["Clinical"].values

#subj_train, subj_test, y_train_subj, y_test_subj = train_test_split(
 #   subjects, labels, test_size=0.2, random_state=42, stratify=labels # splitting into 80% training and 20% test; we keep clinical / nonclinical ratio the same 
#)

def bandpass_5_50(x):
    return mne.filter.filter_data(x, sfreq=1000, l_freq=5, h_freq=50, verbose=False)

def downsample(x, factor):
    """Downsampling along the time axis using decimate (anti-alias)."""
    # x: (channels, samples)
    return sps.decimate(x, factor, axis=1, ftype='iir', zero_phase=True)

# each signal gets split into 5 detail levels and 1 approximation level; for each level we compute the 8 statistics (6 x 8 is 48 features per channel); 
def dwt_channel_feats(sig):
    coeffs = pywt.wavedec(sig, 'db4', level=5)
    f = []
    for c in coeffs:
        f += [np.mean(c), np.std(c), np.var(c),
              np.median(c), np.max(c), np.min(c),
              np.percentile(c, 25), np.percentile(c, 75)]
    return np.array(f, dtype=np.float32)

def extract_subject_features(mat_path, eeg_id, sfreq=1000, ds_factor=4):
    """
    Loading one .mat, filtering 5–50 Hz, downsampling, z-score (removes electrode-specific amplitude differences), DWT, relative band powers, returning feature vector.
    Uses caching so each subject is computed only once.
    """
    feat_file = FEATURE_DIR / f"{eeg_id}.npz"
    if feat_file.exists():
        return np.load(feat_file)["feats"]

    m = loadmat(mat_path)
    Xraw = m["X"]       

    # shape: (channels, samples)
    X = Xraw if Xraw.shape[0] < Xraw.shape[1] else Xraw.T

    # band-passing filter 5–50 Hz
    Xf = bandpass_5_50(X)

    # downsampling along time axis (1000 Hz -> 250 Hz if factor=4)
    if ds_factor > 1:
        Xf = downsample(Xf, ds_factor)
        sfreq_new = sfreq // ds_factor
    else:
        sfreq_new = sfreq

    # z-score per channel
    Xf = (Xf - Xf.mean(axis=1, keepdims=True)) / (Xf.std(axis=1, keepdims=True) + 1e-8)

    # DWT features per channel - concatenating to 1d vector
    #feats = np.hstack([dwt_channel_feats(Xf[ch]) for ch in range(Xf.shape[0])])
    all_feats = []

    for ch in range(Xf.shape[0]):
        sig = Xf[ch]

        # DWT features
        dwt_feats = dwt_channel_feats(sig)

        # Spectral features (NEW)
        freqs, psd = compute_psd(sig, sfreq_new)
        rel_p = relative_band_powers(freqs, psd, BANDS)

        spectral_feats = np.array([
            rel_p["delta"],
            rel_p["theta"],
            rel_p["alpha"],
            rel_p["beta"],
            rel_p["theta"] / (rel_p["alpha"] + 1e-12)  # theta/alpha ratio
        ])

        all_feats.append(np.hstack([dwt_feats, spectral_feats]))

    feats = np.hstack(all_feats)
    print(feats.shape)
    print(np.isnan(feats).any())
    print(np.isinf(feats).any())
    print("Feature dimension:", X.shape[1])

    np.savez_compressed(feat_file, feats=feats, sfreq=sfreq_new)
    return feats

id2path  = dict(zip(data_index["EEG_ID"], data_index["mat_path"]))
id2label = dict(zip(data_index["EEG_ID"], data_index["Clinical"]))

# building x and y matrices for train/test 
def make_xy(subj_list):
    X, y = [], []
    for sid in subj_list:
        print("Extracting", sid)
        feats = extract_subject_features(id2path[sid], sid)  
        X.append(feats)
        y.append(id2label[sid])
    return np.vstack(X), np.array(y, dtype=int)

def run_nested_cv(subject_ids, task_name):
    print(f"\n========== {task_name} ==========")

    X, y = make_xy(subject_ids)

    """pipe = make_pipeline(
        StandardScaler(),
        PCA(random_state=42),
        SVC(kernel="rbf", class_weight="balanced") # tried linear - AD RO2 doesnt have a stable signal, but rbf is compensating with complexity; with linear first and last are better.
    )"""
    # classification-relevant structure is largely linear or weakly separable => linear SVC has better performance
    # focusing on slow-bands
    pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression( # maximizing likelihood; L1 - feature selection, L2 is for stability with correlated features; perfect for high-dimensional eeg
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced",
        max_iter=10000,
        tol=1e-3
    )
    )
    param_grid = {
    "logisticregression__C": [0.01, 0.1, 1],
    "logisticregression__l1_ratio": [0.3, 0.5]
    }

    """param_grid = {
        "pca__n_components": [5, 10, 15, 20],
        "svc__C": [0.5, 1, 2, 3, 4, 5, 10],
        "svc__gamma": ["scale", 0.01, 0.001],
    }"""

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # each fold has the same class ratio

    all_seed_scores = []

    for seed in [0, 1, 42, 123]:
        print(f"\n--- Random state {seed} ---")
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # data split into 5 folds, each fold is 80% training, every subject becomes test data once; final performance is average over folds
        # generalization estimation - how well the best model generalizes to unseen subjects; and inner cv is for model selection - it never sees test data

    #outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # AD-RO2 und WY-RO1 sind besser, andere 2 sind so schlechter

        inner_scores = []
        outer_scores = []

        cm_total = np.zeros((2, 2), dtype=int)

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            print(f"Outer fold {fold}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=inner_cv,
                scoring="balanced_accuracy",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_total += cm


            inner_scores.append(grid.best_score_)
            outer_scores.append(grid.score(X_test, y_test))

    inner_scores = np.array(inner_scores)
    outer_scores = np.array(outer_scores)

    print("Aggregated confusion matrix:")
    print(cm_total)
    print(f"{task_name} results:")
    print("Inner CV mean :", inner_scores.mean())
    print("Outer CV mean :", outer_scores.mean())
    print("Outer CV std  :", outer_scores.std())
    print("Generalization gap:",
          inner_scores.mean() - outer_scores.mean())
    all_seed_scores.append(outer_scores.mean())

    return outer_scores

# =========================================================
# AD and WY separately
# =========================================================
ad_subjects = data_index.loc[
    data_index["Task"] == "AD", "EEG_ID"
].values

wy_subjects = data_index.loc[
    data_index["Task"] == "WY", "EEG_ID"
].values

ad_order1 = data_index.loc[
    (data_index["Task"] == "AD") &
    (data_index["RestOrder"] == 1),
    "EEG_ID"
].values

ad_order2 = data_index.loc[
    (data_index["Task"] == "AD") &
    (data_index["RestOrder"] == 2),
    "EEG_ID"
].values

wy_order1 = data_index.loc[
    (data_index["Task"] == "WY") &
    (data_index["RestOrder"] == 1),
    "EEG_ID"
].values

wy_order2 = data_index.loc[
    (data_index["Task"] == "WY") &
    (data_index["RestOrder"] == 2),
    "EEG_ID"
].values

print("AD subjects:", len(ad_subjects))
print("WY subjects:", len(wy_subjects))

scores_ad_o1 = run_nested_cv(ad_order1, "AD – RestOrder 1")
scores_ad_o2 = run_nested_cv(ad_order2, "AD – RestOrder 2")

scores_wy_o1 = run_nested_cv(wy_order1, "WY – RestOrder 1")
scores_wy_o2 = run_nested_cv(wy_order2, "WY – RestOrder 2")

print(len(ad_order1), len(ad_order2))
print(len(wy_order1), len(wy_order2))



