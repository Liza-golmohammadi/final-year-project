"""
Phase 4 – Final Model Training + Publication-Quality Figures
=============================================================
Usage:
    python phase4_final.py --csv tappy_features.csv
    python phase4_final.py --csv tappy_features.csv --features-json selection_output/optimal_features.json

If --features-json is omitted, falls back to the 3 most significant features
from Phase 1 (hold_mean, hold_std, active_kpm_mean) + flt_mean, flt_std.

Outputs (written to ./final_output/):
    metrics_table.csv        – AUC, sensitivity, specificity, balanced-acc, F1 per model
    roc_curves.png           – all 4 models on one plot
    confusion_matrices.png   – 2×2 grid of confusion matrices
    feature_importance.png   – RF feature importances
    final_model.joblib       – best model bundle (ready to drop into app.py)
    final_report.txt         – full performance report
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
FALLBACK_FEATURES = ["hold_mean", "hold_std", "active_kpm_mean", "flt_mean", "flt_std"]
CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
OUT_DIR = Path("final_output")

# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        lo, hi = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(lower=lo, upper=hi)
        X[col] = X[col].fillna(X[col].median())
    return X

# ── model definitions ─────────────────────────────────────────────────────────

def build_models(features: list[str]) -> dict:
    """Return fresh (name → Pipeline) dict for each model type."""
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
            min_samples_leaf=2, n_jobs=-1,
        )),
    ])
    mlp = Pipeline([
        ("sc",  RobustScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            learning_rate_init=0.001,
        )),
    ])
    # Ensemble uses clones of the above (same hyper-params, trained together)
    ensemble = VotingClassifier(
        estimators=[
            ("svm", Pipeline([
                ("sc",  RobustScaler()),
                ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                            probability=True, class_weight="balanced")),
            ])),
            ("rf", Pipeline([
                ("sc",  RobustScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=600, random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2, n_jobs=-1,
                )),
            ])),
            ("mlp", Pipeline([
                ("sc",  RobustScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32), activation="relu",
                    solver="adam", max_iter=600, early_stopping=True,
                    validation_fraction=0.15, random_state=42,
                )),
            ])),
        ],
        voting="soft",
    )
    return {"SVM": svm, "RF": rf, "MLP": mlp, "Ensemble": ensemble}

# ── CV evaluation ─────────────────────────────────────────────────────────────

def cv_eval(X, y, groups, model):
    probs_all, ys_all = [], []
    for tr, te in CV.split(X, y, groups=groups):
        m_clone = _clone(model)
        m_clone.fit(X.iloc[tr], y.iloc[tr])
        probs_all.append(m_clone.predict_proba(X.iloc[te])[:, 1])
        ys_all.append(y.iloc[te].to_numpy())
    probs = np.concatenate(probs_all)
    ys    = np.concatenate(ys_all)
    return probs, ys

def _clone(model):
    from sklearn.base import clone
    return clone(model)

def compute_metrics(probs: np.ndarray, ys: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "AUC":           round(float(roc_auc_score(ys, probs)), 4),
        "BalancedAcc":   round(float(balanced_accuracy_score(ys, preds)), 4),
        "Sensitivity":   round(float(sens), 4),
        "Specificity":   round(float(spec), 4),
        "F1":            round(float(f1_score(ys, preds)), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

# ── figures ───────────────────────────────────────────────────────────────────

_MODEL_COLORS = {
    "SVM":      "#2196F3",
    "RF":       "#4CAF50",
    "MLP":      "#FF9800",
    "Ensemble": "#9C27B0",
}

def plot_roc_curves(all_results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in all_results.items():
        fpr, tpr, _ = roc_curve(data["ys"], data["probs"])
        auc = data["metrics"]["AUC"]
        ax.plot(fpr, tpr, lw=2.2,
                color=_MODEL_COLORS.get(name, "gray"),
                label=f"{name}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Phase 4: ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confusion_matrices(all_results: dict, out_path: Path):
    names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        data = all_results[name]
        preds = (data["probs"] >= 0.5).astype(int)
        cm    = confusion_matrix(data["ys"], preds)
        disp  = ConfusionMatrixDisplay(cm, display_labels=["Control", "PD"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        m = data["metrics"]
        ax.set_title(
            f"{name}\nAUC={m['AUC']:.3f}  "
            f"Sens={m['Sensitivity']:.3f}  "
            f"Spec={m['Specificity']:.3f}",
            fontsize=9,
        )
    fig.suptitle("Phase 4: Confusion Matrices (threshold = 0.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_feature_importance(
    rf_model, features: list[str], out_path: Path
):
    """Plot mean decrease impurity from the trained RF (fitted on all data)."""
    clf = rf_model.named_steps["clf"]
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_feats = [features[i] for i in idx]
    sorted_imp   = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.5)))
    colors = ["#1a4d2e" if i == 0 else "#4f772d" if i < 3 else "#90be6d"
              for i in range(len(features))]
    ax.barh(sorted_feats[::-1], sorted_imp[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title("Phase 4: RF Feature Importances", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path: Path):
    metric_cols = ["AUC", "BalancedAcc", "Sensitivity", "Specificity", "F1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(metric_cols))
    width = 0.18
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        offset = (i - len(metrics_df) / 2 + 0.5) * width
        vals   = [row[m] for m in metric_cols]
        bars   = ax.bar(x + offset, vals, width,
                        label=row["Model"],
                        color=_MODEL_COLORS.get(row["Model"], "gray"),
                        edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Phase 4: Model Performance Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ── report ────────────────────────────────────────────────────────────────────

def write_report(
    metrics_df: pd.DataFrame,
    features: list[str],
    n_ctrl: int, n_pd: int,
    out_path: Path,
):
    best_row = metrics_df.loc[metrics_df["AUC"].idxmax()]
    lines = [
        "=" * 70,
        "PHASE 4 – FINAL MODEL PERFORMANCE REPORT",
        "=" * 70,
        "",
        f"Features used ({len(features)}):  {', '.join(features)}",
        f"Dataset:  {n_ctrl} control  /  {n_pd} PD  (total {n_ctrl + n_pd})",
        f"CV:  StratifiedGroupKFold(5, shuffle=True, random_state=42)",
        "",
        "-" * 70,
        "METRICS (5-fold subject-wise CV, threshold=0.5)",
        "-" * 70,
    ]
    col_fmt = "{:<12} {:>7} {:>11} {:>13} {:>13} {:>7}"
    lines.append(col_fmt.format(
        "Model", "AUC", "BalancedAcc", "Sensitivity", "Specificity", "F1"
    ))
    lines.append("-" * 70)
    for _, row in metrics_df.iterrows():
        lines.append(col_fmt.format(
            row["Model"],
            f"{row['AUC']:.4f}",
            f"{row['BalancedAcc']:.4f}",
            f"{row['Sensitivity']:.4f}",
            f"{row['Specificity']:.4f}",
            f"{row['F1']:.4f}",
        ))
    lines += [
        "",
        "-" * 70,
        f"Best model: {best_row['Model']}  (AUC={best_row['AUC']:.4f})",
        "",
        "Saved model: final_output/final_model.joblib",
        "(Compatible with app.py — drop-in replacement for tappy_web_model.joblib)",
        "",
        "=" * 70,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    out_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {out_path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",           required=True)
    ap.add_argument("--features-json", default=None,
                    help="JSON from Phase 3 (selection_output/optimal_features.json)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # ── load features ──────────────────────────────────────────────────────────
    if args.features_json and Path(args.features_json).exists():
        with open(args.features_json) as f:
            features = json.load(f)["optimal_features"]
        print(f"Loaded {len(features)} optimal features from {args.features_json}")
    else:
        features = FALLBACK_FEATURES
        print(f"No features JSON supplied — using fallback: {features}")

    print(f"Features: {features}")

    # ── load data ──────────────────────────────────────────────────────────────
    print("\nLoading data …")
    df     = pd.read_csv(args.csv).dropna(subset=["label", "UserKey"]).copy()
    y      = df["label"].astype(int)
    groups = df["UserKey"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    print(f"  {len(df)} sessions  |  {(y==0).sum()} control / {(y==1).sum()} PD")

    X = preprocess(df, features)

    # ── run CV for all models ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4 – FINAL MODEL TRAINING + EVALUATION")
    print("=" * 60)

    models      = build_models(features)
    all_results = {}
    rows        = []

    for name, model in models.items():
        print(f"  Evaluating {name} …", end=" ", flush=True)
        probs, ys = cv_eval(X, y, groups, model)
        m         = compute_metrics(probs, ys)
        all_results[name] = {"probs": probs, "ys": ys, "metrics": m}
        rows.append({"Model": name, **m})
        print(f"AUC={m['AUC']:.4f}  "
              f"BalAcc={m['BalancedAcc']:.4f}  "
              f"Sens={m['Sensitivity']:.4f}  "
              f"Spec={m['Specificity']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'metrics_table.csv'}")

    # ── figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_roc_curves(all_results,          OUT_DIR / "roc_curves.png")
    plot_confusion_matrices(all_results,  OUT_DIR / "confusion_matrices.png")
    plot_metrics_bar(metrics_df,          OUT_DIR / "metrics_comparison.png")

    # Feature importance from RF trained on ALL data
    print("  Fitting RF on full dataset for feature importance …")
    rf_full = _clone(models["RF"])
    rf_full.fit(X, y)
    plot_feature_importance(rf_full, features, OUT_DIR / "feature_importance.png")

    # ── save best model ────────────────────────────────────────────────────────
    best_name = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]
    best_model = _clone(models[best_name])
    best_model.fit(X, y)
    bundle = {"model": best_model, "features": features}
    joblib.dump(bundle, OUT_DIR / "final_model.joblib")
    print(f"\n  Best model: {best_name} → saved as {OUT_DIR / 'final_model.joblib'}")
    print(f"  (Copy to project root and rename to tappy_web_model.joblib to deploy)")

    # ── report ─────────────────────────────────────────────────────────────────
    write_report(
        metrics_df, features,
        n_ctrl=int((y == 0).sum()),
        n_pd=int((y == 1).sum()),
        out_path=OUT_DIR / "final_report.txt",
    )


if __name__ == "__main__":
    main()
