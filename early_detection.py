import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# ==============================
# 1. LOAD DATA — MIT ONLY
# ==============================
DATA_PATH = "combined/combined_features_common.csv"

df = pd.read_csv(DATA_PATH, dtype={"subject_id": "string"}, low_memory=False)
df["subject_id"] = df["subject_id"].astype(str)

mit_df = df[df["dataset"] != "tappy"].copy()

drop_cols = [c for c in ["window_id", "session_id", "updrs"] if c in mit_df.columns]
X_mit = mit_df.drop(columns=drop_cols + ["label_pd"])
X_mit = X_mit.select_dtypes(include=[np.number])

y_mit = mit_df["label_pd"].values
updrs_mit = mit_df["updrs"].astype(float).values
groups_mit = (mit_df["dataset"] + "_" + mit_df["subject_id"]).values

print(f"MIT windows: {len(y_mit)}")
print(f"UPDRS range: {updrs_mit.min():.1f} – {updrs_mit.max():.1f}, mean={updrs_mit.mean():.1f}")
print(f"Mild PD windows (UPDRS ≤20): {((updrs_mit <= 20) & (y_mit == 1)).sum()}")
print(f"Moderate-Severe PD windows (UPDRS >20): {((updrs_mit > 20) & (y_mit == 1)).sum()}")

# ==============================
# 2. GROUPKFOLD — COLLECT ALL TEST PREDICTIONS
# ==============================
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
gkf = GroupKFold(n_splits=5)

all_y_true  = []
all_y_prob  = []
all_updrs   = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_mit, y_mit, groups_mit)):
    X_train, X_test = X_mit.iloc[train_idx], X_mit.iloc[test_idx]
    y_train, y_test = y_mit[train_idx], y_mit[test_idx]
    updrs_test = updrs_mit[test_idx]

    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]

    all_y_true.append(y_test)
    all_y_prob.append(y_prob)
    all_updrs.append(updrs_test)

all_y_true = np.concatenate(all_y_true)
all_y_prob = np.concatenate(all_y_prob)
all_updrs  = np.concatenate(all_updrs)

# ==============================
# 3. OVERALL AUC
# ==============================
auc_full = roc_auc_score(all_y_true, all_y_prob)
print(f"\nFull MIT-only AUC: {auc_full:.4f}")

# ==============================
# 4. SUBGROUP AUC BY UPDRS
# ==============================
hc_mask       = all_y_true == 0
mild_pd_mask  = (all_y_true == 1) & (all_updrs <= 20)
mod_pd_mask   = (all_y_true == 1) & (all_updrs > 20)

print(f"\nHC windows in test: {hc_mask.sum()}")
print(f"Mild PD windows (UPDRS ≤20): {mild_pd_mask.sum()}")
print(f"Moderate-Severe PD windows (UPDRS >20): {mod_pd_mask.sum()}")

# Mild PD vs HC
y_mild = np.concatenate([np.zeros(hc_mask.sum()), np.ones(mild_pd_mask.sum())])
p_mild = np.concatenate([all_y_prob[hc_mask], all_y_prob[mild_pd_mask]])
auc_mild = roc_auc_score(y_mild, p_mild)

# Moderate-Severe PD vs HC
y_mod = np.concatenate([np.zeros(hc_mask.sum()), np.ones(mod_pd_mask.sum())])
p_mod = np.concatenate([all_y_prob[hc_mask], all_y_prob[mod_pd_mask]])
auc_mod = roc_auc_score(y_mod, p_mod)

print(f"\n{'='*50}")
print(f"Full MIT-only AUC          : {auc_full:.4f}")
print(f"Mild PD (UPDRS ≤20) AUC    : {auc_mild:.4f}")
print(f"Moderate-Severe AUC        : {auc_mod:.4f}")
print(f"AUC gap (mod - mild)       : {auc_mod - auc_mild:.4f}")
