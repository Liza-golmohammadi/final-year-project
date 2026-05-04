import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ==============================
# 1. LOAD COMBINED FEATURE FILE
# ==============================
DATA_PATH = "combined/combined_features_common.csv"

df = pd.read_csv(DATA_PATH, dtype={"subject_id": "string"}, low_memory=False)
df["subject_id"] = df["subject_id"].astype(str)

print("\n===== DATA SUMMARY =====")
print("Total samples:", len(df))
print("\nDataset distribution:")
print(df["dataset"].value_counts())
print("\nLabel distribution:")
print(df["label_pd"].value_counts())


# ==============================
# 2. PREPARE FEATURES
# ==============================
# Drop UPDRS (clinical score — label leak) and metadata columns
drop_cols = [c for c in ["window_id", "session_id", "updrs"] if c in df.columns]
X = df.drop(columns=drop_cols + ["label_pd"])
X = X.select_dtypes(include=[np.number])

y = df["label_pd"]

# Unique group IDs across datasets — prevents subject ID collision between Tappy and MIT
df["group_id"] = df["dataset"] + "_" + df["subject_id"]
groups = df["group_id"]

print("\nNumber of features used:", X.shape[1])
print("Feature names:", list(X.columns))


# ==============================
# 3. SUBJECT-LEVEL EVALUATION HELPER
# ==============================
def subject_level_metrics(y_true_windows, y_score_windows, group_ids):
    results = pd.DataFrame({
        "group_id": group_ids,
        "y_true": y_true_windows,
        "y_score": y_score_windows
    })
    subj = results.groupby("group_id").agg(
        y_true=("y_true", "max"),
        y_score=("y_score", "mean")
    ).reset_index()

    if subj["y_true"].nunique() < 2:
        return None, None, None

    y_pred = (subj["y_score"] >= 0.5).astype(int)
    acc  = accuracy_score(subj["y_true"], y_pred)
    bacc = balanced_accuracy_score(subj["y_true"], y_pred)
    f1   = f1_score(subj["y_true"], y_pred, zero_division=0)
    auc  = roc_auc_score(subj["y_true"], subj["y_score"])
    return acc, bacc, f1, auc


# ==============================
# 4. DEFINE MODELS
# ==============================
svm_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ))
])


# ==============================
# 5. EVALUATION FUNCTION
# ==============================
def run_group_cv(model, X, y, groups, title):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=5)
    accs, baccs, f1s, aucs = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_groups = groups.iloc[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc, bacc, f1, auc = subject_level_metrics(
            y_test.values, y_prob, test_groups.values
        )
        if acc is None:
            print(f"\nFold {fold+1}  |  Skipped (single class in test)")
            continue

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
        aucs.append(auc)

        print(f"\nFold {fold+1}  |  Test subjects: {test_groups.nunique()}")
        print(f"Subject-level  Accuracy: {acc:.4f}  Balanced Acc: {bacc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    print("\n----- FINAL AVERAGE -----")
    print(f"Mean Accuracy         : {np.mean(accs):.4f}")
    print(f"Mean Balanced Accuracy: {np.mean(baccs):.4f}")
    print(f"Mean F1-score         : {np.mean(f1s):.4f}")
    print(f"Mean ROC-AUC          : {np.mean(aucs):.4f}")


# ==============================
# EXPERIMENT 1 — MIT ONLY
# ==============================
mit_df = df[df["dataset"] != "tappy"].copy()
X_mit = mit_df[X.columns]
y_mit = mit_df["label_pd"]
g_mit = mit_df["group_id"]

run_group_cv(svm_model, X_mit, y_mit, g_mit, "Exp 1 MIT-ONLY (SVM)")
run_group_cv(rf_model, X_mit, y_mit, g_mit, "Exp 1 MIT-ONLY (Random Forest)")


# ==============================
# EXPERIMENT 2 — TAPPY ONLY
# ==============================
tappy_df = df[df["dataset"] == "tappy"].copy()
X_tappy = tappy_df[X.columns]
y_tappy = tappy_df["label_pd"]
g_tappy = tappy_df["group_id"]

run_group_cv(svm_model, X_tappy, y_tappy, g_tappy, "Exp 2 TAPPY-ONLY (SVM)")
run_group_cv(rf_model, X_tappy, y_tappy, g_tappy, "Exp 2 TAPPY-ONLY (Random Forest)")


# ==============================
# EXPERIMENT 3 — CROSS-DATASET
# ==============================
print("\n" + "="*60)
print("EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)")
print("="*60)

train_tappy = df[df["dataset"] == "tappy"]
test_mit = df[df["dataset"] != "tappy"]

X_train = train_tappy[X.columns]
y_train = train_tappy["label_pd"]
X_test = test_mit[X.columns]
y_test = test_mit["label_pd"]
g_test = test_mit["group_id"]

svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

for name, model in [("SVM", svm_model), ("Random Forest", rf_model)]:
    y_prob = model.predict_proba(X_test)[:, 1]
    acc, bacc, f1, auc = subject_level_metrics(y_test.values, y_prob, g_test.values)
    print(f"\nExp 3 CROSS-DATASET — {name}")
    print(f"Subject-level  Accuracy: {acc:.4f}  Balanced Acc: {bacc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")


# ==============================
# EXPERIMENT 4 — BALANCED POOLED
# ==============================
print("\n" + "="*60)
print("EXPERIMENT 4: BALANCED POOLED MODEL")
print("="*60)

mit_size = len(mit_df)
tappy_sample = tappy_df.sample(n=mit_size, random_state=42)
balanced_df = pd.concat([tappy_sample, mit_df], ignore_index=True)

X_bal = balanced_df[X.columns]
y_bal = balanced_df["label_pd"]
g_bal = balanced_df["group_id"]

run_group_cv(svm_model, X_bal, y_bal, g_bal, "Exp 4 BALANCED POOLED (SVM)")
run_group_cv(rf_model, X_bal, y_bal, g_bal, "Exp 4 BALANCED POOLED (Random Forest)")
