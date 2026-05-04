"""
Train RF on balanced pooled 26-feature common set and export as neurotype_model.joblib.

Usage:
    python export_model.py

Output:
    neurotype_model.joblib  — bundle with keys: "model", "features", "threshold"
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

DATA_PATH = Path("combined/combined_features_common.csv")
OUT_PATH  = Path("neurotype_model.joblib")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, dtype={"subject_id": "string"}, low_memory=False)
df["subject_id"] = df["subject_id"].astype(str)

drop_cols = [c for c in ["window_id", "session_id", "updrs", "dataset", "subject_id"] if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_cols + ["label_pd"] and df[c].dtype != object]

df["group_id"] = df["dataset"] + "_" + df["subject_id"]
y_all    = df["label_pd"]
groups   = df["group_id"]

# ── Balanced pooled subset (equal Tappy / MIT window counts) ───────────────────
mit_df       = df[df["dataset"] != "tappy"].copy()
tappy_df     = df[df["dataset"] == "tappy"].copy()
tappy_sample = tappy_df.sample(n=len(mit_df), random_state=42)
bal_df       = pd.concat([tappy_sample, mit_df], ignore_index=True)

X_bal = bal_df[feature_cols]
y_bal = bal_df["label_pd"]
g_bal = bal_df["group_id"]

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Balanced pooled windows: {len(bal_df)}  |  PD: {y_bal.sum()}  HC: {(y_bal==0).sum()}")

# ── Cross-validation to report performance ────────────────────────────────────
rf_cv = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
gkf   = GroupKFold(n_splits=5)
aucs, f1s, accs = [], [], []

for fold, (tr, te) in enumerate(gkf.split(X_bal, y_bal, g_bal)):
    rf_cv.fit(X_bal.iloc[tr], y_bal.iloc[tr])
    y_prob = rf_cv.predict_proba(X_bal.iloc[te])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    aucs.append(roc_auc_score(y_bal.iloc[te], y_prob))
    f1s.append(f1_score(y_bal.iloc[te], y_pred, zero_division=0))
    accs.append(accuracy_score(y_bal.iloc[te], y_pred))
    print(f"  Fold {fold+1}  AUC={aucs[-1]:.4f}  F1={f1s[-1]:.4f}  Acc={accs[-1]:.4f}")

print(f"\nMean AUC={np.mean(aucs):.4f}  F1={np.mean(f1s):.4f}  Acc={np.mean(accs):.4f}")

# ── Train final model on ALL balanced pooled data ─────────────────────────────
rf_final = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf_final.fit(X_bal, y_bal)

# ── Save bundle ───────────────────────────────────────────────────────────────
bundle = {
    "model":     rf_final,
    "features":  feature_cols,
    "threshold": 0.5,
}
joblib.dump(bundle, OUT_PATH)
print(f"\nSaved: {OUT_PATH}  (features={len(feature_cols)})")
