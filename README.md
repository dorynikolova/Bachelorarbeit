# Thesis in EEG Feature Extraction & Clinical Classification Pipeline

This project implements a **full EEG processing and machine learning pipeline** for clinical classification using spectral and wavelet-based features.  
It loads EEG `.mat` recordings, extracts physiologically meaningful features, and evaluates classification performance using **nested cross-validation**.

The goal is to **distinguish clinical vs. non-clinical subjects** while analyzing task conditions and rest order effects.

---

## Overview of the Pipeline

The pipeline performs:

1. EEG signal loading (`.mat`)
2. Band-pass filtering (5–50 Hz)
3. Downsampling (1000 Hz → 250 Hz)
4. Channel-wise z-score normalization
5. Feature extraction:
   - Discrete Wavelet Transform (DWT)
   - Power Spectral Density (Welch)
   - Relative EEG band power (delta, theta, alpha, beta)
6. Feature caching for efficiency
7. Nested cross-validation model training
8. Performance evaluation with confusion matrices

---

## EEG Feature Extraction

### Frequency Bands Used
| Band  |Range(Hz)|
|------ |---------|
| Delta |  1–4    |
| Theta |  4–8    |
| Alpha |  8–13   |
| Beta  | 13–30   |

### Extracted Feature Types

#### 1. Wavelet Features (DWT)
Each EEG channel is decomposed into 5 wavelet levels (`db4`), and the following statistics are computed per level:

- Mean  
- Standard deviation  
- Variance  
- Median  
- Min / Max  
- 25th & 75th percentiles  

#### 2. Spectral Features (Welch PSD)

- Relative band power for each EEG band  
- Theta / Alpha ratio  
- Normalized to total power for cross-subject comparability  

---

## Signal Processing Steps

- **Band-pass filter:** 5–50 Hz (removes drift & muscle artifacts)
- **Downsampling:** Anti-alias decimation (1000 Hz → 250 Hz)
- **Z-score normalization:** Reduces electrode amplitude bias
- **Feature caching:** Features stored in a folder

This ensures **efficient reuse and reproducibility**.

---

## Dataset Integration

Clinical metadata is loaded from:

Clinical/ADWY_Clinical_Data.sav

Labels include:
- Clinical condition (binary)
- Task type (`AD`, `WY`)
- Rest order (1 or 2)

EEG recordings are matched by `EEG_ID`.

---

## Machine Learning Pipeline

### Model
**Logistic Regression (Elastic Net)**  
Chosen for:
- High-dimensional EEG feature stability
- Feature sparsity (L1)
- Robust correlated feature handling (L2)
- Balanced clinical classification

### Pipeline Components

- StandardScaler  
- Logistic Regression (`saga` solver)  
- Class-balanced weighting  

### Hyperparameter Search
GridSearchCV over:
- Regularization strength (`C`)
- Elastic net mixing ratio (`l1_ratio`)

---

## Validation Strategy

### Nested Cross-Validation

- **Inner CV (3-fold):** Hyperparameter tuning  
- **Outer CV (5-fold):** Generalization estimation  
- **Multiple random seeds:** Robust stability measurement  

Ensures:
- No data leakage  
- Reliable generalization estimates  
- Fair subject-level evaluation  

---

## Task-Specific Evaluation

Performance is evaluated separately for:

- AD — Rest Order 1  
- AD — Rest Order 2  
- WY — Rest Order 1  
- WY — Rest Order 2  

Outputs include:
- Balanced accuracy  
- Confusion matrices  
- Generalization gap  
- Performance variance across seeds  

**Seeds:** 0, 1, 42, 123 

Metrics reported:
- Inner CV mean
- Outer CV mean 
- Outer CV standard deviation 
- Generalization gap 
- Aggregated confusion matrix

---

# Results (Until Now)
This section summarizes the **current intermediate results** of the EEG feature extraction and clinical classification pipeline. The project is **still ongoing**, and further analyses, model comparisons, and statistical evaluations will be added as development continues.

--- 

## Dataset Summary A total of **295 subjects** were included: 
| Group         | Count | 
|---------------|-------| 
| AD (Clinical) | 146   | 
| WY (Control)  | 149   |

## AD — Rest Order 1

**Confusion Matrix**

[[ 8  7]
[19 40]]

**Performance**

| Metric             | Value  |
|--------------------|--------|
| Inner CV mean      | 0.6518 |
| Outer CV mean      | 0.6045 |
| Outer CV std       | 0.1010 |
| Generalization gap | 0.0473 |

**Notes:**  
Moderate performance with slight overfitting. Variability across folds suggests sensitivity to subject distribution.

## AD — Rest Order 2

**Confusion Matrix**

[[ 2 16]
[ 6 48]]

**Performance**

| Metric             | Value  |
|--------------------|--------|
| Inner CV mean      | 0.5091 |
| Outer CV mean      | 0.4955 |
| Outer CV std       | 0.0091 |
| Generalization gap | 0.0137 |

**Notes:**  
Performance close to chance level. Very stable across folds but low discriminative power.

## WY — Rest Order 1

**Confusion Matrix**

[[28 15]
[ 3  1]]

**Performance**

| Metric             | Value  |
|--------------------|--------|
| Inner CV mean      | 0.5563 |
| Outer CV mean      | 0.4861 |
| Outer CV std       | 0.0596 |
| Generalization gap | 0.0702 |

**Notes:**  
Multiple warnings due to class imbalance. Some folds contain only a single class, affecting metric reliability.

## WY — Rest Order 2

**Confusion Matrix**

[[28 12]
[ 5  4]]

**Performance**

| Metric             | Value  |
|--------------------|--------|
| Inner CV mean      | 0.6229 |
| Outer CV mean      | 0.6000 |
| Outer CV std       | 0.2039 |
| Generalization gap | 0.0229 |

**Notes:**  
Best WY performance so far, but high variance indicates sensitivity to fold composition.

## Current Limitations

- Several folds triggered **class imbalance warnings**  
- Some WY folds contained **only one class**, affecting confusion matrix shape  
- Results may be influenced by:
  - subject distribution  
  - rest order effects  
  - feature dimensionality  
  - regularization strength  

These issues will be addressed in the next development phase.  

---

## Dependencies

Key libraries used:

- `numpy`, `scipy`, `pandas`
- `scikit-learn`
- `mne`
- `pywavelets`
- `pyreadstat`
- `joblib`