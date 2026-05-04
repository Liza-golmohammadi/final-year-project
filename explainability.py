import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

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

# ==============================
# 2. BALANCED POOLED SUBSET
# ==============================
mit_df   = df[df["dataset"] != "tappy"].copy()
tappy_df = df[df["dataset"] == "tappy"].copy()

mit_size     = len(mit_df)
tappy_sample = tappy_df.sample(n=mit_size, random_state=42)
balanced_df  = pd.concat([tappy_sample, mit_df], ignore_index=True)

X_bal = balanced_df[X_all.columns]
y_bal = balanced_df["label_pd"].values

# ==============================
# 3. TRAIN RF
# ==============================
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_bal, y_bal)
print("RF trained.")

# ==============================
# 4. FIGURE A — RF Feature Importance (Top 15)
# ==============================
fi_df = pd.DataFrame({
    "Feature": X_bal.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="#DE8F05", edgecolor="black")
for i, (val, name) in enumerate(zip(fi_df["Importance"][::-1], fi_df["Feature"][::-1])):
    ax1.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
ax1.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=12)
ax1.set_title("Top 15 Features — RF Feature Importance (Balanced Pooled)", fontsize=13, fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/figure_rf_importance.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: figure_rf_importance.png")

# ==============================
# 5. FIGURE B — Permutation Importance (Top 15)
# ==============================
print("\nRunning permutation importance (this may take a minute)...")
perm = permutation_importance(rf, X_bal, y_bal, n_repeats=10, random_state=42, n_jobs=-1, scoring="roc_auc")

perm_df = pd.DataFrame({
    "Feature": X_bal.columns,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std
}).sort_values("Importance", ascending=False).head(15)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(perm_df["Feature"][::-1], perm_df["Importance"][::-1],
         xerr=perm_df["Std"][::-1], color="#029E73", edgecolor="black", capsize=4)
ax2.set_xlabel("Mean AUC Decrease When Feature Shuffled", fontsize=12)
ax2.set_title("Top 15 Features — Permutation Importance (Balanced Pooled)", fontsize=13, fontweight="bold")
ax2.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/figure_permutation_importance.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: figure_permutation_importance.png")

print("\nPermutation Importance (top 15):")
print(perm_df.to_string(index=False))

# ==============================
# 6. FIGURE C — SHAP Beeswarm (Top 15)
# ==============================
print("\nRunning SHAP (this may take a few minutes)...")

# Use a sample for speed (500 windows)
sample_idx = np.random.RandomState(42).choice(len(X_bal), size=500, replace=False)
X_sample = X_bal.iloc[sample_idx]

explainer = shap.TreeExplainer(rf)
shap_explanation = explainer(X_sample)

# For binary RF, take class 1 (PD) SHAP values
shap_vals_pd = shap_explanation.values[:, :, 1] if shap_explanation.values.ndim == 3 else shap_explanation.values

fig3, ax3 = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_vals_pd, X_sample, plot_type="dot", max_display=15, show=False)
plt.title("SHAP Beeswarm — RF (Balanced Pooled, PD class)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/figure_shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: figure_shap_beeswarm.png")
