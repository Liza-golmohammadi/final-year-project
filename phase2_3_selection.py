"""
Phase 2 & 3 – Minimal Feature Test + Forward Feature Selection
===============================================================
Usage:
    python phase2_3_selection.py --csv tappy_features.csv

Outputs (written to ./selection_output/):
    phase2_comparison.csv    – 4-feature vs 13-feature AUC / metrics table
    phase2_roc.png           – ROC curves for both feature sets
    phase3_selection.png     – AUC vs number of features added
    optimal_features.json    – final feature subset for Phase 4
    phase3_report.txt        – human-readable selection log
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
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
WEB_FEATURES = [
    "hold_mean", "hold_std",
    "flt_mean",  "flt_std",  "flt_neg_rate",
    "lat_mean",  "lat_std",
    "active_kpm_mean", "active_kpm_std",
    "burst_count", "active_minutes",
    "long_gap_rate", "n_keys",
]

CORE_FEATURES = ["hold_mean", "hold_std", "flt_mean", "flt_std"]

CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

OUT_DIR = Path("selection_output")

# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Clip 1st/99th percentile, fill NaN with median. Scaler is inside pipelines."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        lo, hi = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(lower=lo, upper=hi)
        X[col] = X[col].fillna(X[col].median())
    return X

# ── model factories ───────────────────────────────────────────────────────────

def make_svm() -> Pipeline:
    return Pipeline([
        ("sc",  RobustScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                    probability=True, class_weight="balanced")),
    ])

def make_rf() -> Pipeline:
    return Pipeline([
        ("sc",  RobustScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1,
        )),
    ])

# ── CV evaluation ─────────────────────────────────────────────────────────────

def cv_eval(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_factory,
) -> tuple[float, np.ndarray, np.ndarray]:
    """5-fold subject-wise CV. Returns (mean_auc, probs_all, ys_all)."""
    probs_all, ys_all = [], []
    for tr, te in CV.split(X, y, groups=groups):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        probs_all.append(m.predict_proba(X.iloc[te])[:, 1])
        ys_all.append(y.iloc[te].to_numpy())
    probs = np.concatenate(probs_all)
    ys    = np.concatenate(ys_all)
    return float(roc_auc_score(ys, probs)), probs, ys

def metrics_from_probs(probs: np.ndarray, ys: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auc":      roc_auc_score(ys, probs),
        "bal_acc":  balanced_accuracy_score(ys, preds),
        "f1":       f1_score(ys, preds),
        "sensitivity": sens,
        "specificity": spec,
    }

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_phase2_roc(results: dict, out_path: Path):
    """ROC curves for all (feature_set, model) combos."""
    colors = {
        "4_core_SVM": "#2196F3", "4_core_RF":  "#03A9F4",
        "13_all_SVM": "#F44336", "13_all_RF":  "#FF9800",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, data in results.items():
        fpr, tpr, _ = roc_curve(data["ys"], data["probs"])
        auc = data["metrics"]["auc"]
        label = f"{key.replace('_', ' ')}  AUC={auc:.3f}"
        ls = "-" if "SVM" in key else "--"
        ax.plot(fpr, tpr, lw=2, linestyle=ls, color=colors.get(key, "gray"), label=label)

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Phase 2: ROC Curves — 4 Core Features vs 13 Features", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_selection_history(history: list[dict], out_path: Path):
    steps  = [h["step"]         for h in history]
    aucs   = [h["auc"]          for h in history]
    labels = [h["added"]        for h in history]
    nfeats = [h["n_features"]   for h in history]

    fig, ax = plt.subplots(figsize=(max(7, len(history) * 1.2), 5))
    ax.plot(steps, aucs, "o-", color="#1a4d2e", lw=2, ms=7)
    for s, a, lbl in zip(steps, aucs, labels):
        ax.annotate(lbl, (s, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, rotation=30)
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{n}\nfeats" for n in nfeats], fontsize=8)
    ax.set_ylabel("CV AUC (RF)", fontsize=12)
    ax.set_title("Phase 3: Forward Feature Selection Path", fontsize=12)
    ax.set_ylim(max(0, min(aucs) - 0.05), min(1.0, max(aucs) + 0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ── phase 2 ───────────────────────────────────────────────────────────────────

def run_phase2(df: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 2 – MINIMAL FEATURE TEST")
    print("=" * 60)
    print(f"  Core features:  {CORE_FEATURES}")
    print(f"  Full features:  {WEB_FEATURES}\n")

    feature_sets  = [("4_core", CORE_FEATURES), ("13_all", WEB_FEATURES)]
    model_configs = [("SVM", make_svm), ("RF", make_rf)]

    results = {}
    rows    = []

    for fs_name, feats in feature_sets:
        X = preprocess(df, feats)
        for mname, factory in model_configs:
            key = f"{fs_name}_{mname}"
            auc, probs, ys = cv_eval(X, y, groups, factory)
            m = metrics_from_probs(probs, ys)
            results[key] = {"probs": probs, "ys": ys, "metrics": m, "features": feats}
            rows.append({
                "feature_set":  fs_name,
                "n_features":   len(feats),
                "model":        mname,
                "AUC":          round(m["auc"],         3),
                "BalancedAcc":  round(m["bal_acc"],     3),
                "Sensitivity":  round(m["sensitivity"], 3),
                "Specificity":  round(m["specificity"], 3),
                "F1":           round(m["f1"],          3),
            })
            print(f"  {key:<18}  AUC={m['auc']:.3f}  "
                  f"BalAcc={m['bal_acc']:.3f}  "
                  f"Sens={m['sensitivity']:.3f}  "
                  f"Spec={m['specificity']:.3f}")

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUT_DIR / "phase2_comparison.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'phase2_comparison.csv'}")

    plot_phase2_roc(results, OUT_DIR / "phase2_roc.png")

    # Print verdict
    core_auc = results["4_core_RF"]["metrics"]["auc"]
    full_auc = results["13_all_RF"]["metrics"]["auc"]
    delta    = full_auc - core_auc
    print(f"\n  VERDICT (RF): 4-core AUC={core_auc:.3f}  |  "
          f"13-all AUC={full_auc:.3f}  |  Δ={delta:+.3f}")
    if delta < -0.005:
        print("  → Extra features HURT performance. Forward selection will prune from 13.")
    elif delta > 0.005:
        print("  → Extra features help. Forward selection will identify the useful ones.")
    else:
        print("  → Negligible difference. Forward selection will confirm optimal subset.")

    return results

# ── phase 3 ───────────────────────────────────────────────────────────────────

def run_phase3(df: pd.DataFrame, y: pd.Series, groups: pd.Series) -> list[str]:
    print("\n" + "=" * 60)
    print("PHASE 3 – FORWARD FEATURE SELECTION")
    print("=" * 60)

    MIN_DELTA = 0.002   # minimum AUC improvement to accept a feature

    current    = list(CORE_FEATURES)
    candidates = [f for f in WEB_FEATURES if f not in current]

    X0       = preprocess(df, current)
    base_auc, _, _ = cv_eval(X0, y, groups, make_rf)
    print(f"  Start: {current}")
    print(f"  Baseline AUC = {base_auc:.4f}\n")

    history = [{
        "step":       0,
        "added":      "(core 4)",
        "n_features": len(current),
        "auc":        base_auc,
        "delta":      0.0,
    }]
    report_lines = [
        "PHASE 3 – FORWARD FEATURE SELECTION LOG",
        "=" * 50,
        f"Start: {current}  AUC={base_auc:.4f}",
        "",
    ]

    step = 0
    while candidates:
        step += 1
        round_res = []
        for cand in candidates:
            X_test = preprocess(df, current + [cand])
            auc_test, _, _ = cv_eval(X_test, y, groups, make_rf)
            round_res.append((cand, auc_test, auc_test - base_auc))

        round_res.sort(key=lambda x: x[1], reverse=True)

        best_feat, best_auc_cand, best_delta = round_res[0]

        report_lines.append(f"Round {step} — candidates tried:")
        for feat, a, d in round_res:
            marker = " ← BEST" if feat == best_feat else ""
            report_lines.append(f"  add {feat:<22} AUC={a:.4f}  Δ={d:+.4f}{marker}")

        if best_delta > MIN_DELTA:
            current.append(best_feat)
            candidates.remove(best_feat)
            base_auc = best_auc_cand
            history.append({
                "step":       step,
                "added":      best_feat,
                "n_features": len(current),
                "auc":        base_auc,
                "delta":      best_delta,
            })
            msg = f"  ADDED {best_feat}  → AUC={base_auc:.4f}  (Δ={best_delta:+.4f})"
            print(msg)
            report_lines.append(msg)
        else:
            msg = (f"  No candidate improved AUC by >{MIN_DELTA:.3f}. "
                   f"Best was {best_feat} (Δ={best_delta:+.4f}). Stopping.")
            print(msg)
            report_lines.append(msg)
            break

        report_lines.append("")

    report_lines += [
        "",
        "=" * 50,
        f"OPTIMAL FEATURE SUBSET ({len(current)} features):",
        *[f"  {f}" for f in current],
        f"Final CV AUC (RF): {base_auc:.4f}",
    ]

    report_text = "\n".join(report_lines)
    print(f"\n  Optimal subset ({len(current)} features): {current}")
    print(f"  Final AUC (RF): {base_auc:.4f}")

    (OUT_DIR / "phase3_report.txt").write_text(report_text, encoding="utf-8")
    print(f"  Saved: {OUT_DIR / 'phase3_report.txt'}")

    optimal = {"optimal_features": current, "final_rf_auc": round(base_auc, 4)}
    with open(OUT_DIR / "optimal_features.json", "w") as f:
        json.dump(optimal, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'optimal_features.json'}")

    plot_selection_history(history, OUT_DIR / "phase3_selection.png")

    return current

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading data …")
    df     = pd.read_csv(args.csv).dropna(subset=["label", "UserKey"]).copy()
    y      = df["label"].astype(int)
    groups = df["UserKey"]

    missing = [f for f in WEB_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    print(f"  {len(df)} sessions  |  {(y==0).sum()} control / {(y==1).sum()} PD")

    run_phase2(df, y, groups)
    optimal = run_phase3(df, y, groups)

    print("\n" + "=" * 60)
    print(f"Done. Run phase4_final.py next with optimal features:")
    print(f"  python phase4_final.py --csv {args.csv} "
          f"--features-json {OUT_DIR / 'optimal_features.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
