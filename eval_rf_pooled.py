import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    recall_score, accuracy_score, balanced_accuracy_score
)
import pandas as pd

# ==============================
# 1. LOAD DATA
# ==============================
DATA_PATH = "combined/combined_features_common.csv"

df = pd.read_csv(DATA_PATH, dtype={"subject_id": "string"}, low_memory=False)
df["subject_id"] = df["subject_id"].astype(str)

drop_cols = [c for c in ["window_id", "session_id", "updrs"] if c in df.columns]
X_all = df.drop(columns=drop_cols + ["label_pd"])
X_all = X_all.select_dtypes(include=[np.number])

df["group_id"] = df["dataset"] + "_" + df["subject_id"]
y_all = df["label_pd"]
groups_all = df["group_id"]

# ==============================
# 2. BALANCED POOLED SUBSET
# ==============================
mit_df    = df[df["dataset"] != "tappy"].copy()
tappy_df  = df[df["dataset"] == "tappy"].copy()

mit_size     = len(mit_df)
tappy_sample = tappy_df.sample(n=mit_size, random_state=42)
balanced_df  = pd.concat([tappy_sample, mit_df], ignore_index=True)

X_bal = balanced_df[X_all.columns]
y_bal = balanced_df["label_pd"]
g_bal = balanced_df["group_id"]

# ==============================
# 3. RF MODEL
# ==============================
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)

# ==============================
# 4. CROSS-VALIDATION — WINDOW-LEVEL
# ==============================
print("=" * 60)
print("RF — Balanced Pooled — Window-level — Full Metrics")
print("=" * 60)

gkf = GroupKFold(n_splits=5)
aucs, f1s, recalls, specs, baccs, accs = [], [], [], [], [], []
cm_total = np.zeros((2, 2), dtype=int)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_bal, y_bal, g_bal)):
    X_train, X_test = X_bal.iloc[train_idx], X_bal.iloc[test_idx]
    y_train, y_test = y_bal.iloc[train_idx].values, y_bal.iloc[test_idx].values

    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_total += cm

    auc      = roc_auc_score(y_test, y_prob)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    recall   = recall_score(y_test, y_pred, zero_division=0)  # sensitivity
    spec     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc     = balanced_accuracy_score(y_test, y_pred)
    acc      = accuracy_score(y_test, y_pred)

    aucs.append(auc)
    f1s.append(f1)
    recalls.append(recall)
    specs.append(spec)
    baccs.append(bacc)
    accs.append(acc)

    print(f"\nFold {fold+1}")
    print(f"  Confusion Matrix: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  AUC={auc:.4f}  F1={f1:.4f}  Recall/Sensitivity={recall:.4f}  Specificity={spec:.4f}  Balanced Acc={bacc:.4f}")

# ==============================
# 5. FINAL AVERAGES
# ==============================
print("\n" + "=" * 60)
print("FINAL AVERAGES (mean across 5 folds)")
print("=" * 60)
print(f"  AUC                  : {np.mean(aucs):.4f}")
print(f"  F1-score             : {np.mean(f1s):.4f}")
print(f"  Sensitivity (Recall) : {np.mean(recalls):.4f}")
print(f"  Specificity          : {np.mean(specs):.4f}")
print(f"  Balanced Accuracy    : {np.mean(baccs):.4f}")
print(f"  Accuracy             : {np.mean(accs):.4f}")

print("\nAggregated Confusion Matrix (all folds combined):")
print(f"  TP={cm_total[1,1]}  FP={cm_total[0,1]}  TN={cm_total[0,0]}  FN={cm_total[1,0]}")
