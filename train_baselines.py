import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


# ==============================
# 1. LOAD DATA
# ==============================
DATA_PATH = "combined/combined_features_common.csv"

df = pd.read_csv(DATA_PATH, dtype={"subject_id": "string"}, low_memory=False)
df["subject_id"] = df["subject_id"].astype(str)

# Fix 1: Drop UPDRS and metadata columns — not available at inference time
drop_cols = [c for c in ["window_id", "session_id", "updrs"] if c in df.columns]
X_all = df.drop(columns=drop_cols + ["label_pd"])
X_all = X_all.select_dtypes(include=[np.number])

# Fix 2: Unique group IDs across datasets to prevent cross-dataset subject collision
df["group_id"] = df["dataset"] + "_" + df["subject_id"]

y_all = df["label_pd"]
groups_all = df["group_id"]


# ==============================
# 2. SUBJECT-LEVEL EVALUATION HELPER
# Aggregates window-level probabilities/scores to one score per subject
# then computes metrics at subject level
# ==============================
def subject_level_metrics(y_true_windows, y_score_windows, group_ids, threshold=0.5):
    results = pd.DataFrame({
        "group_id": group_ids,
        "y_true": y_true_windows,
        "y_score": y_score_windows
    })
    # One row per subject: mean score, majority label
    subj = results.groupby("group_id").agg(
        y_true=("y_true", "max"),    # subject is PD if any window is PD
        y_score=("y_score", "mean")  # mean probability across windows
    ).reset_index()

    y_pred = (subj["y_score"] >= threshold).astype(int)

    # Need at least one of each class for AUC
    if subj["y_true"].nunique() < 2:
        return None, None, None

    acc = accuracy_score(subj["y_true"], y_pred)
    f1 = f1_score(subj["y_true"], y_pred, zero_division=0)
    auc = roc_auc_score(subj["y_true"], subj["y_score"])
    return acc, f1, auc


# ==============================
# 3. HT VARIABILITY BASELINE
# Non-ML: threshold on hold_std — learned from training fold only
# ==============================
def run_ht_baseline(X, y, groups, title):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=5)
    accs, f1s, aucs = [], [], []

    ht_col = X["hold_std"].values

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        ht_train = ht_col[train_idx]
        y_train = y.iloc[train_idx].values

        pd_ht = ht_train[y_train == 1]
        hc_ht = ht_train[y_train == 0]
        threshold = (np.median(pd_ht) + np.median(hc_ht)) / 2

        ht_test = ht_col[test_idx]
        y_test = y.iloc[test_idx].values
        test_groups = groups.iloc[test_idx].values

        acc, f1, auc = subject_level_metrics(y_test, ht_test, test_groups, threshold=threshold)
        if acc is None:
            print(f"\nFold {fold+1}  |  Skipped (single class in test)")
            continue

        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

        print(f"\nFold {fold+1}  |  Threshold: {threshold:.4f}  |  Test subjects: {len(np.unique(test_groups))}")
        print(f"Subject-level  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    print("\n----- FINAL AVERAGE -----")
    print(f"Mean Accuracy : {np.mean(accs):.4f}")
    print(f"Mean F1-score : {np.mean(f1s):.4f}")
    print(f"Mean ROC-AUC  : {np.mean(aucs):.4f}")


def run_ht_baseline_cross_dataset(X_train_df, y_train, X_test_df, y_test, groups_test, title):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    ht_train = X_train_df["hold_std"].values
    ht_test = X_test_df["hold_std"].values
    y_train_arr = y_train.values
    y_test_arr = y_test.values

    pd_ht = ht_train[y_train_arr == 1]
    hc_ht = ht_train[y_train_arr == 0]
    threshold = (np.median(pd_ht) + np.median(hc_ht)) / 2

    acc, f1, auc = subject_level_metrics(y_test_arr, ht_test, groups_test.values, threshold=threshold)

    print(f"Threshold (from Tappy train): {threshold:.4f}")
    print(f"Subject-level  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")


# ==============================
# 4. LOGISTIC REGRESSION MODEL
# ==============================
lr_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])


def run_group_cv(model, X, y, groups, title):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=5)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_groups = groups.iloc[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc, f1, auc = subject_level_metrics(
            y_test.values, y_prob, test_groups.values
        )
        if acc is None:
            print(f"\nFold {fold+1}  |  Skipped (single class in test)")
            continue

        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

        print(f"\nFold {fold+1}  |  Test subjects: {test_groups.nunique()}")
        print(f"Subject-level  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    print("\n----- FINAL AVERAGE -----")
    print(f"Mean Accuracy : {np.mean(accs):.4f}")
    print(f"Mean F1-score : {np.mean(f1s):.4f}")
    print(f"Mean ROC-AUC  : {np.mean(aucs):.4f}")


# ==============================
# 5. EXPERIMENT 1 — MIT ONLY
# ==============================
mit_df = df[df["dataset"] != "tappy"].copy()
X_mit = mit_df[X_all.columns]
y_mit = mit_df["label_pd"]
g_mit = mit_df["group_id"]

run_ht_baseline(X_mit, y_mit, g_mit, "Exp 1 MIT-ONLY — HT Variability Baseline")
run_group_cv(lr_model, X_mit, y_mit, g_mit, "Exp 1 MIT-ONLY — Logistic Regression")


# ==============================
# 6. EXPERIMENT 2 — TAPPY ONLY
# ==============================
tappy_df = df[df["dataset"] == "tappy"].copy()
X_tappy = tappy_df[X_all.columns]
y_tappy = tappy_df["label_pd"]
g_tappy = tappy_df["group_id"]

run_ht_baseline(X_tappy, y_tappy, g_tappy, "Exp 2 TAPPY-ONLY — HT Variability Baseline")
run_group_cv(lr_model, X_tappy, y_tappy, g_tappy, "Exp 2 TAPPY-ONLY — Logistic Regression")


# ==============================
# 7. EXPERIMENT 3 — CROSS-DATASET
# ==============================
print("\n" + "="*60)
print("EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)")
print("="*60)

train_tappy = df[df["dataset"] == "tappy"]
test_mit = df[df["dataset"] != "tappy"]

X_train_cd = train_tappy[X_all.columns]
y_train_cd = train_tappy["label_pd"]
X_test_cd = test_mit[X_all.columns]
y_test_cd = test_mit["label_pd"]
g_test_cd = test_mit["group_id"]

run_ht_baseline_cross_dataset(
    X_train_cd, y_train_cd, X_test_cd, y_test_cd, g_test_cd,
    "Exp 3 CROSS-DATASET — HT Variability Baseline"
)

lr_model.fit(X_train_cd, y_train_cd)
y_prob_lr = lr_model.predict_proba(X_test_cd)[:, 1]
acc, f1, auc = subject_level_metrics(y_test_cd.values, y_prob_lr, g_test_cd.values)
print("\nExp 3 CROSS-DATASET — Logistic Regression")
print(f"Subject-level  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")


# ==============================
# 8. EXPERIMENT 4 — BALANCED POOLED
# ==============================
mit_size = len(mit_df)
tappy_sample = tappy_df.sample(n=mit_size, random_state=42)
balanced_df = pd.concat([tappy_sample, mit_df], ignore_index=True)

X_bal = balanced_df[X_all.columns]
y_bal = balanced_df["label_pd"]
g_bal = balanced_df["group_id"]

run_ht_baseline(X_bal, y_bal, g_bal, "Exp 4 BALANCED POOLED — HT Variability Baseline")
run_group_cv(lr_model, X_bal, y_bal, g_bal, "Exp 4 BALANCED POOLED — Logistic Regression")
