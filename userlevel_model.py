"""
User-Level Model — tappy_userlevel.csv
=======================================
Uses subject-level aggregated features including:
  - med_* : median of session features per subject
  - std_* : intra-subject variability across sessions (key PD signal)
  - digraph features: med_hand_L/R/S, med_dir_LL/LR/RL/RR/LS/SL/SS

One row = one subject → use StratifiedKFold (no group leakage risk).

Usage:
    python userlevel_model.py --csv tappy_userlevel.csv

Outputs (written to ./userlevel_output/):
    phase1_stats.csv         – feature stats sorted by effect size
    roc_curves.png
    confusion_matrices.png
    metrics_table.csv
    feature_importance.png
    userlevel_model.joblib   – best model bundle (features chosen here)
    report.txt
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

OUT_DIR = Path("userlevel_output")
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# All 48 feature columns (everything except UserKey, n_months, total_keys, label)
EXCLUDE_COLS = {"UserKey", "n_months", "total_keys", "label"}

# ── helpers ───────────────────────────────────────────────────────────────────

def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def preprocess(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        lo, hi = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(lower=lo, upper=hi)
        X[col] = X[col].fillna(X[col].median())
    return X


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def effect_label(d):
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def feature_stats(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for feat in X.columns:
        ctrl = X.loc[y==0, feat].dropna().values
        pd_  = X.loc[y==1, feat].dropna().values
        mw   = scipy_stats.mannwhitneyu(ctrl, pd_, alternative="two-sided")
        d    = cohens_d(pd_, ctrl)
        rows.append({
            "feature":     feat,
            "cohens_d":    round(d, 3),
            "effect_size": effect_label(d),
            "mw_p":        round(mw.pvalue, 4),
            "significant": mw.pvalue < 0.05,
            "ctrl_median": round(np.median(ctrl), 3),
            "pd_median":   round(np.median(pd_), 3),
        })
    return pd.DataFrame(rows).sort_values("mw_p")


def select_features_by_effect(stats_df: pd.DataFrame, min_abs_d: float = 0.1) -> list[str]:
    """Keep features with |d| >= min_abs_d (filters near-zero noise)."""
    return stats_df[stats_df["cohens_d"].abs() >= min_abs_d]["feature"].tolist()


# ── models ────────────────────────────────────────────────────────────────────

def build_models():
    svm = Pipeline([
        ("sc",  RobustScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                    probability=True, class_weight="balanced")),
    ])
    rf = Pipeline([
        ("sc",  RobustScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=600, random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=1, n_jobs=-1,
        )),
    ])
    mlp = Pipeline([
        ("sc",  RobustScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", max_iter=600,
            early_stopping=True, validation_fraction=0.15,
            random_state=42,
        )),
    ])
    ensemble = VotingClassifier(
        estimators=[
            ("svm", clone(svm)),
            ("rf",  clone(rf)),
            ("mlp", clone(mlp)),
        ],
        voting="soft",
    )
    return {"SVM": svm, "RF": rf, "MLP": mlp, "Ensemble": ensemble}


def cv_eval(X, y, model):
    probs_all, ys_all = [], []
    for tr, te in CV.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        probs_all.append(m.predict_proba(X.iloc[te])[:, 1])
        ys_all.append(y.iloc[te].to_numpy())
    probs = np.concatenate(probs_all)
    ys    = np.concatenate(ys_all)
    return probs, ys


def compute_metrics(probs, ys, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, preds).ravel()
    return {
        "AUC":          round(float(roc_auc_score(ys, probs)), 4),
        "BalancedAcc":  round(float(balanced_accuracy_score(ys, preds)), 4),
        "Sensitivity":  round(float(tp/(tp+fn)) if (tp+fn) else 0.0, 4),
        "Specificity":  round(float(tn/(tn+fp)) if (tn+fp) else 0.0, 4),
        "F1":           round(float(f1_score(ys, preds)), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ── figures ───────────────────────────────────────────────────────────────────

COLORS = {"SVM": "#2196F3", "RF": "#4CAF50", "MLP": "#FF9800", "Ensemble": "#9C27B0"}


def plot_roc(all_results, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in all_results.items():
        fpr, tpr, _ = roc_curve(data["ys"], data["probs"])
        auc = data["metrics"]["AUC"]
        ax.plot(fpr, tpr, lw=2.2, color=COLORS[name],
                label=f"{name}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — User-Level Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cm(all_results, out_path):
    names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.5))
    for ax, name in zip(axes, names):
        data  = all_results[name]
        preds = (data["probs"] >= 0.5).astype(int)
        cm    = confusion_matrix(data["ys"], preds)
        disp  = ConfusionMatrixDisplay(cm, display_labels=["Control", "PD"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        m = data["metrics"]
        ax.set_title(
            f"{name}\nAUC={m['AUC']:.3f}  Sens={m['Sensitivity']:.3f}  Spec={m['Specificity']:.3f}",
            fontsize=9,
        )
    fig.suptitle("Confusion Matrices — User-Level Model (threshold = 0.5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_feature_importance(rf_model, features, top_n=20, out_path=None):
    clf = rf_model.named_steps["clf"]
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    feats_sorted = [features[i] for i in idx[::-1]]
    imp_sorted   = imp[idx[::-1]]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    colors = ["#1a4d2e" if i < 3 else "#4f772d" if i < 8 else "#90be6d"
              for i in range(len(feats_sorted))]
    ax.barh(feats_sorted, imp_sorted, color=colors, edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title(f"Top {top_n} RF Feature Importances — User-Level", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics_bar(metrics_df, out_path):
    cols  = ["AUC", "BalancedAcc", "Sensitivity", "Specificity", "F1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x, w = np.arange(len(cols)), 0.18
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        offset = (i - len(metrics_df) / 2 + 0.5) * w
        vals   = [row[c] for c in cols]
        bars   = ax.bar(x + offset, vals, w, label=row["Model"],
                        color=COLORS.get(row["Model"], "gray"),
                        edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance — User-Level Features", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",      required=True)
    ap.add_argument("--min-d",    type=float, default=0.1,
                    help="Min |Cohen's d| to keep a feature (default 0.1)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading data …")
    df = pd.read_csv(args.csv).dropna(subset=["label"]).copy()
    y  = df["label"].astype(int)
    print(f"  {len(df)} subjects  |  {(y==0).sum()} control / {(y==1).sum()} PD")

    all_feats = get_features(df)
    print(f"  Available features: {len(all_feats)}")

    print("\nPreprocessing …")
    X_all = preprocess(df, all_feats)

    print("Computing feature statistics …")
    stats_df = feature_stats(X_all, y)
    stats_df.to_csv(OUT_DIR / "phase1_stats.csv", index=False)

    n_sig = stats_df["significant"].sum()
    print(f"\n  Significant features (p<0.05): {n_sig}/{len(all_feats)}")
    print(f"\n  Top 15 by p-value:")
    col = "{:<28} {:>8} {:>11} {:>7}"
    print(col.format("Feature", "Cohen's d", "effect", "p"))
    print("-" * 58)
    for _, r in stats_df.head(15).iterrows():
        print(col.format(r["feature"], f"{r['cohens_d']:+.3f}", r["effect_size"], f"{r['mw_p']:.4f}"))

    selected = select_features_by_effect(stats_df, min_abs_d=args.min_d)
    print(f"\n  Selected {len(selected)} features with |d| >= {args.min_d}")

    X = preprocess(df, selected)

    print("\n" + "=" * 60)
    print("EVALUATING MODELS (StratifiedKFold 5-fold)")
    print("=" * 60)

    models      = build_models()
    all_results = {}
    rows        = []

    for name, model in models.items():
        print(f"  {name} …", end=" ", flush=True)
        probs, ys = cv_eval(X, y, model)
        m         = compute_metrics(probs, ys)
        all_results[name] = {"probs": probs, "ys": ys, "metrics": m}
        rows.append({"Model": name, **m})
        print(f"AUC={m['AUC']:.4f}  BalAcc={m['BalancedAcc']:.4f}  "
              f"Sens={m['Sensitivity']:.4f}  Spec={m['Specificity']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "metrics_table.csv", index=False)

    print("\nGenerating figures …")
    plot_roc(all_results,    OUT_DIR / "roc_curves.png")
    plot_cm(all_results,     OUT_DIR / "confusion_matrices.png")
    plot_metrics_bar(metrics_df, OUT_DIR / "metrics_comparison.png")

    print("  Fitting RF on full data for feature importance …")
    rf_full = clone(models["RF"])
    rf_full.fit(X, y)
    plot_feature_importance(rf_full, selected, top_n=min(20, len(selected)),
                            out_path=OUT_DIR / "feature_importance.png")

    # Save best model
    best_name  = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]
    best_model = clone(models[best_name])
    best_model.fit(X, y)
    bundle = {"model": best_model, "features": selected}
    joblib.dump(bundle, OUT_DIR / "userlevel_model.joblib")

    # Print summary
    best = metrics_df.loc[metrics_df["AUC"].idxmax()]
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — User-Level Model")
    print(f"{'='*60}")
    print(f"Features used:   {len(selected)}  (|d|>={args.min_d})")
    print(f"Subjects:        {len(df)}  ({(y==0).sum()} ctrl / {(y==1).sum()} PD)")
    print(f"")
    col_fmt = "{:<12} {:>7} {:>11} {:>13} {:>13} {:>7}"
    print(col_fmt.format("Model", "AUC", "BalancedAcc", "Sensitivity", "Specificity", "F1"))
    print("-" * 68)
    for _, r in metrics_df.iterrows():
        print(col_fmt.format(r["Model"], f"{r['AUC']:.4f}", f"{r['BalancedAcc']:.4f}",
                             f"{r['Sensitivity']:.4f}", f"{r['Specificity']:.4f}", f"{r['F1']:.4f}"))
    print(f"\nBest model: {best['Model']}  AUC={best['AUC']:.4f}")
    print(f"Saved: {OUT_DIR}/userlevel_model.joblib")
    print(f"All outputs in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
