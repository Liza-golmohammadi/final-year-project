# NeuroType — Parkinson's Disease Detection via Keystroke Dynamics

A machine learning system that analyses keystroke timing patterns to detect early motor symptoms associated with Parkinson's Disease (PD). Users type a short passage in the browser; the system extracts 26 clinically-motivated timing features and returns a risk score in real time.

---

## How it works

1. **Preprocessing** — Raw keystroke data from two datasets (Tappy and NeuroQWERTY/MIT-CS) is cleaned, standardised, and segmented into 200-keystroke windows with 75% overlap.
2. **Feature engineering** — Each window is converted into a 26-feature vector covering hold time (HT), down-down (DD), and up-down (UD) timing statistics, pause rates, and digraph transition rates.
3. **Model** — A Random Forest classifier trained on a balanced pooled dataset (Tappy + MIT-CS1 + MIT-CS2) using subject-disjoint 5-fold cross-validation (GroupKFold). Best window-level AUC: ~0.75.
4. **Web app** — A Flask server serves a live typing interface. Keystroke timing is captured in the browser, features are computed in JavaScript, and a prediction is returned from the trained model.

---

## Project structure

```
NeuroType-final/
│
├── app.py                        # Flask web server
├── export_model.py               # Train & save the RF model (run this first)
├── find_best_threshold.py        # Optimise decision threshold
├── requirements.txt
│
├── templates/
│   └── index.html                # Web UI (keystroke capture + visualisation)
│
├── preprocess_tappy.py           # Tappy dataset preprocessing
├── preprocess_neuroqwerty.py     # NeuroQWERTY/MIT-CS preprocessing
├── combine_features.py           # Feature fusion & schema alignment
├── build_dataset.py              # Build sequence arrays for CNN
│
├── train_baselines.py            # HT baseline — subject-level
├── train_baselines_window.py     # HT baseline — window-level
├── train_experiments.py          # LR/SVM/RF — subject-level
├── train_experiments_window.py   # LR/SVM/RF — window-level
├── train_cnn.py                  # 1D-CNN — subject-level
├── train_cnn_window.py           # 1D-CNN — window-level
├── eval_rf_pooled.py             # Full RF evaluation with confusion matrix
│
├── explainability.py             # SHAP + permutation importance
├── plot_confusion_matrix.py      # Confusion matrix figures
├── plot_metric_comparison.py     # Model comparison figures
├── early_detection.py            # Early detection experiments
│
├── figures/                      # Generated plots
├── Results/                      # Documentation and result reports
└── neurotype_model.joblib        # Trained RF model bundle
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Place the raw datasets in the project root:
- `tappy-keystroke-data-1.0.0/`
- `neuroqwerty-mit-csxpd-dataset-1.0.0/`

Then run preprocessing and feature fusion:
```bash
python preprocess_tappy.py
python preprocess_neuroqwerty.py
python combine_features.py
```

### 3. Train and export the model
```bash
python export_model.py
```
This trains the RF on the balanced pooled 26-feature common set and saves `neurotype_model.joblib`.

### 4. Run the web app
```bash
python app.py
```
Open `http://127.0.0.1:5000`, type 80+ keystrokes, then click **Analyse**.

---

## Features (26 common features)

| Group | Count | Examples |
|---|---|---|
| Hold Time (HT) | 6 | `hold_mean`, `hold_p95`, `hold_cv` |
| Down-Down (DD) | 6 | `dd_mean`, `dd_p95`, `dd_cv` |
| Up-Down / Flight (UD) | 6 | `ud_mean`, `ud_p95`, `ud_cv` |
| Pause Rates | 3 | `pause_rate_dd_gt_500/750/1000` |
| Digraph Patterns | 4 | `dig_KK_rate`, `dig_KS_rate`, `dig_SK_rate`, `dig_SS_rate` |
| Window Size | 1 | `n` |

---

## Datasets

- **Tappy** — naturalistic home typing, binary hand taxonomy (K/S)
- **MIT-CS1 / MIT-CS2 (NeuroQWERTY)** — clinical lab typing, ternary taxonomy (K/S/P)

Raw datasets are not included in this repository due to size. Download links are available from the respective dataset publishers.

---

## Results summary (window-level, no UPDRS)

| Model | MIT-only AUC | Balanced Pooled AUC |
|---|---|---|
| Random Forest | 0.7458 | 0.7479 |
| SVM | 0.7498 | 0.7339 |
| Logistic Regression | 0.7484 | 0.7149 |
| CNN | 0.6332 | 0.7140 |
| HT Baseline | 0.6449 | 0.5688 |
