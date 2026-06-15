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
5. Feature extraction:
   - PyRiemann Covariance Matrices
   - Tangent Space
   - Standard Scaler
   - Logistic Regression with Elastic Net
6. Feature caching for efficiency
7. Nested cross-validation model training
8. Performance evaluation with confusion matrices

---

## EEG Feature Extraction

### Frequency Bands Used
| Band  |Range(Hz)|
|------ |---------|
| Theta |  4–8    |
| Alpha |  8–13   |
| Beta  | 13–30   |

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
- **Random seeds:** Robust stability measurement  

Ensures:
- No data leakage  
- Reliable generalization estimates  
- Fair subject-level evaluation  

---

## Task-Specific Evaluation

Outputs include:
- Balanced accuracy  
- Confusion matrices  
- Generalization gap  
- Performance variance across random seeds  

**Seed:** 42

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

---

## Dependencies

Key libraries used:

- `numpy`, `scipy`, `pandas`
- `scikit-learn`
- `mne`
- `pywavelets`
- `pyreadstat`
- `joblib`