"""
Phase 1 – Data Quality Audit
=============================
Usage:
    python phase1_audit.py --csv tappy_features.csv

Outputs (written to ./audit_output/):
    feature_stats.csv        – per-class stats + effect size + p-value for all 13 features
    boxplots.png             – 13-panel boxplot grid (Control vs PD)
    correlation_control.png  – feature correlation matrix for controls
    correlation_pd.png       – feature correlation matrix for PD subjects
    audit_report.txt         – human-readable diagnostic summary
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ── constants ─────────────────────────────────────────────────────────────────
WEB_FEATURES = [
    "hold_mean", "hold_std",
    "flt_mean",  "flt_std",  "flt_neg_rate",
    "lat_mean",  "lat_std",
    "active_kpm_mean", "active_kpm_std",
    "burst_count", "active_minutes",
    "long_gap_rate", "n_keys",
]

ALPHA = 0.05
OUT_DIR = Path("audit_output")

# ── helpers ───────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clip at 1st/99th percentile, fill NaN with per-feature median."""
    X = df[WEB_FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        lo, hi = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(lower=lo, upper=hi)
        X[col] = X[col].fillna(X[col].median())
    return X


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d (positive = group a > group b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-12
    return float((a.mean() - b.mean()) / pooled_std)


def effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── phase 1 ───────────────────────────────────────────────────────────────────

def compute_feature_stats(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for feat in WEB_FEATURES:
        ctrl = X.loc[y == 0, feat].dropna().values
        pd_   = X.loc[y == 1, feat].dropna().values

        mw = stats.mannwhitneyu(ctrl, pd_, alternative="two-sided")
        d  = cohens_d(pd_, ctrl)   # PD – Control

        rows.append({
            "feature":        feat,
            "ctrl_n":         len(ctrl),
            "pd_n":           len(pd_),
            "ctrl_mean":      ctrl.mean(),
            "ctrl_std":       ctrl.std(ddof=1),
            "ctrl_median":    np.median(ctrl),
            "pd_mean":        pd_.mean(),
            "pd_std":         pd_.std(ddof=1),
            "pd_median":      np.median(pd_),
            "cohens_d":       d,
            "effect_size":    effect_label(d),
            "mw_p":           mw.pvalue,
            "significant":    mw.pvalue < ALPHA,
            "sig_stars":      sig_stars(mw.pvalue),
        })

    df = pd.DataFrame(rows).sort_values("mw_p")
    return df


def plot_boxplots(X: pd.DataFrame, y: pd.Series, stats_df: pd.DataFrame, out_path: Path):
    n = len(WEB_FEATURES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    palette = {"Control": "#4a9eff", "PD": "#ff6b6b"}

    for idx, feat in enumerate(WEB_FEATURES):
        ax = axes[idx]
        data_plot = pd.DataFrame({
            "value": X[feat].values,
            "group": y.map({0: "Control", 1: "PD"}).values,
        })
        sns.boxplot(
            data=data_plot, x="group", y="value", hue="group",
            palette=palette, order=["Control", "PD"],
            width=0.5, fliersize=3, legend=False, ax=ax
        )
        ax.set_xticklabels(["Control", "PD"])
        ax.set_xlabel("")
        ax.set_ylabel("")

        row = stats_df[stats_df["feature"] == feat].iloc[0]
        d_str = f"d={row['cohens_d']:.2f} ({row['effect_size']})"
        p_str = f"p={row['mw_p']:.3f} {row['sig_stars']}"
        ax.set_title(f"{feat}\n{d_str}  {p_str}", fontsize=8, pad=4)

        # colour frame for significant features
        for spine in ax.spines.values():
            spine.set_linewidth(2.0 if row["significant"] else 0.8)
            spine.set_edgecolor("#e74c3c" if row["significant"] else "#aaaaaa")

    # hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    ctrl_patch = mpatches.Patch(color=palette["Control"], label="Control")
    pd_patch   = mpatches.Patch(color=palette["PD"], label="PD")
    fig.legend(handles=[ctrl_patch, pd_patch], loc="lower right", fontsize=10)
    fig.suptitle("Feature Distributions: Control vs PD  (red border = significant at p<0.05)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_correlation(X: pd.DataFrame, y: pd.Series, label: int, out_path: Path):
    subset = X[y == label]
    corr = subset.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.4, ax=ax,
        annot_kws={"size": 7}
    )
    group_name = "Control" if label == 0 else "PD"
    ax.set_title(f"Spearman Correlation Matrix – {group_name}  (n={len(subset)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_report(stats_df: pd.DataFrame, y: pd.Series, out_path: Path):
    n_ctrl = int((y == 0).sum())
    n_pd   = int((y == 1).sum())
    n_sig  = int(stats_df["significant"].sum())
    sig_feats   = stats_df[stats_df["significant"]]["feature"].tolist()
    insig_feats = stats_df[~stats_df["significant"]]["feature"].tolist()

    large_medium = stats_df[stats_df["effect_size"].isin(["large", "medium"])].sort_values("mw_p")
    small_       = stats_df[stats_df["effect_size"] == "small"].sort_values("mw_p")
    negligible   = stats_df[stats_df["effect_size"] == "negligible"].sort_values("mw_p")

    lines = [
        "=" * 70,
        "PHASE 1 – DATA QUALITY AUDIT REPORT",
        "=" * 70,
        "",
        f"Dataset summary",
        f"  Control sessions : {n_ctrl}",
        f"  PD sessions      : {n_pd}",
        f"  Total sessions   : {n_ctrl + n_pd}",
        f"  Class ratio PD   : {n_pd / (n_ctrl + n_pd):.1%}",
        "",
        "-" * 70,
        f"FEATURE SIGNAL SUMMARY  (Mann-Whitney U, α={ALPHA})",
        "-" * 70,
        f"  Significant features ({n_sig}/{len(WEB_FEATURES)}) : {', '.join(sig_feats) if sig_feats else 'none'}",
        f"  Non-significant        : {', '.join(insig_feats) if insig_feats else 'none'}",
        "",
        "-" * 70,
        "PER-FEATURE DETAILS  (sorted by p-value)",
        "-" * 70,
    ]

    col_fmt = "{:<20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>6} {:>5}"
    lines.append(col_fmt.format(
        "Feature", "ctrl_med", "pd_med", "d", "effect", "p-value", "sig", "stars"
    ))
    lines.append("-" * 80)
    for _, row in stats_df.iterrows():
        lines.append(col_fmt.format(
            row["feature"],
            f"{row['ctrl_median']:.2f}",
            f"{row['pd_median']:.2f}",
            f"{row['cohens_d']:.2f}",
            row["effect_size"],
            f"{row['mw_p']:.4f}",
            "YES" if row["significant"] else "no",
            row["sig_stars"],
        ))

    lines += [
        "",
        "-" * 70,
        "RECOMMENDATIONS FOR PHASE 2/3",
        "-" * 70,
    ]

    if large_medium.empty and small_.empty:
        lines.append("  WARNING: No features show meaningful effect sizes. "
                     "Consider dataset quality or feature engineering.")
    else:
        if not large_medium.empty:
            lines.append("  Priority features (medium/large effect, keep in all models):")
            for _, r in large_medium.iterrows():
                lines.append(f"    {r['feature']:<22} d={r['cohens_d']:+.2f}  p={r['mw_p']:.4f}  {r['sig_stars']}")
        if not small_.empty:
            lines.append("  Secondary features (small effect, include in forward selection):")
            for _, r in small_.iterrows():
                lines.append(f"    {r['feature']:<22} d={r['cohens_d']:+.2f}  p={r['mw_p']:.4f}  {r['sig_stars']}")
        if not negligible.empty:
            lines.append("  Weak features (negligible effect, likely to add noise):")
            for _, r in negligible.iterrows():
                lines.append(f"    {r['feature']:<22} d={r['cohens_d']:+.2f}  p={r['mw_p']:.4f}  {r['sig_stars']}")

    lines.append("")
    lines.append("=" * 70)
    report = "\n".join(lines)
    print(report)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 1 – Data quality audit")
    ap.add_argument("--csv", required=True, help="Path to tappy_features.csv")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading data …")
    df = pd.read_csv(args.csv).dropna(subset=["label", "UserKey"]).copy()
    y  = df["label"].astype(int)

    missing = [f for f in WEB_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    print(f"  {len(df)} sessions  |  {y.value_counts()[0]} control  /  {y.value_counts()[1]} PD")

    print("\nPreprocessing (clip 1st/99th percentile, fill NaN with median) …")
    X = preprocess(df)

    print("\nComputing feature statistics …")
    stats_df = compute_feature_stats(X, y)
    stats_df.to_csv(OUT_DIR / "feature_stats.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'feature_stats.csv'}")

    print("\nGenerating boxplots …")
    plot_boxplots(X, y, stats_df, OUT_DIR / "boxplots.png")

    print("\nGenerating correlation matrices …")
    plot_correlation(X, y, label=0, out_path=OUT_DIR / "correlation_control.png")
    plot_correlation(X, y, label=1, out_path=OUT_DIR / "correlation_pd.png")

    print("\nWriting diagnostic report …")
    write_report(stats_df, y, OUT_DIR / "audit_report.txt")

    print(f"\nAll outputs written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
