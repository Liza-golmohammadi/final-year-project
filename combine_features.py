#!/usr/bin/env python3

# Command in terminal to combine
# python3 combine_features.py \
#   --tappy "out_tappy/tappy_window_features.csv" \
#   --cs1 "out_neuroqwerty/mit_cs1_window_features.csv" \
#   --cs2 "out_neuroqwerty/mit_cs2_window_features.csv" \
#   --out_dir "combined" \
#   --mode "both"


from __future__ import annotations

import os
import argparse
import pandas as pd
import numpy as np


ID_COLS = ["dataset", "subject_id", "session_id", "window_id"]
LABEL_COLS = ["label_pd", "updrs"]  # updrs may be missing for tappy


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase, strip spaces
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    if "subject_id" not in df.columns:
        raise ValueError(f"{dataset_name}: missing required column 'subject_id'")
    if "label_pd" not in df.columns:
        raise ValueError(f"{dataset_name}: missing required column 'label_pd'")
    return df


def numeric_feature_cols(df: pd.DataFrame) -> set[str]:
    # keep only numeric columns excluding IDs/labels
    cols = set(df.select_dtypes(include=[np.number]).columns)
    cols -= set(LABEL_COLS)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tappy", required=True, help="Path to tappy_window_features.csv")
    ap.add_argument("--cs1", required=True, help="Path to mit_cs1_window_features.csv")
    ap.add_argument("--cs2", required=True, help="Path to mit_cs2_window_features.csv")
    ap.add_argument("--out_dir", default="combined", help="Output folder")
    ap.add_argument("--mode", choices=["common", "union", "both"], default="both",
                    help="common = only shared features; union = all features; both = save both")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tappy = ensure_columns(normalize_columns(pd.read_csv(args.tappy)), "tappy")
    cs1 = ensure_columns(normalize_columns(pd.read_csv(args.cs1)), "mit_cs1")
    cs2 = ensure_columns(normalize_columns(pd.read_csv(args.cs2)), "mit_cs2")

    # Identify numeric feature sets
    ft = numeric_feature_cols(tappy)
    f1 = numeric_feature_cols(cs1)
    f2 = numeric_feature_cols(cs2)

    common = ft & f1 & f2
    union = ft | f1 | f2

    # Always keep required id/label columns if they exist
    keep_id = [c for c in ID_COLS if c in (set(tappy.columns) | set(cs1.columns) | set(cs2.columns))]
    keep_label = [c for c in LABEL_COLS if c in (set(tappy.columns) | set(cs1.columns) | set(cs2.columns))]
    keep_required = list(dict.fromkeys(keep_id + keep_label))  # preserve order, de-dup

    # Report
    report_path = os.path.join(args.out_dir, "feature_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== FEATURE SET REPORT ===\n\n")
        f.write(f"Tappy numeric features: {len(ft)}\n")
        f.write(f"MIT-CS1 numeric features: {len(f1)}\n")
        f.write(f"MIT-CS2 numeric features: {len(f2)}\n\n")
        f.write(f"COMMON features (all 3): {len(common)}\n")
        f.write(f"UNION features (any): {len(union)}\n\n")

        f.write("Common features:\n")
        for c in sorted(common):
            f.write(f"  {c}\n")

        f.write("\nFeatures only in Tappy:\n")
        for c in sorted(ft - common):
            f.write(f"  {c}\n")

        f.write("\nFeatures only in MIT-CS1:\n")
        for c in sorted(f1 - common):
            f.write(f"  {c}\n")

        f.write("\nFeatures only in MIT-CS2:\n")
        for c in sorted(f2 - common):
            f.write(f"  {c}\n")

    print(f"[OK] Wrote report: {report_path}")

    def subset(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        # Some datasets might not have some cols -> add them
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]

    # Save common-features dataset (recommended)
    if args.mode in ("common", "both"):
        common_cols = keep_required + sorted(common)
        combined_common = pd.concat([
            subset(tappy, common_cols),
            subset(cs1, common_cols),
            subset(cs2, common_cols),
        ], ignore_index=True)

        out_common = os.path.join(args.out_dir, "combined_features_common.csv")
        combined_common.to_csv(out_common, index=False)
        print(f"[OK] Saved: {out_common}  shape={combined_common.shape}")

    # Save union-features dataset (optional)
    if args.mode in ("union", "both"):
        union_cols = keep_required + sorted(union)
        combined_union = pd.concat([
            subset(tappy, union_cols),
            subset(cs1, union_cols),
            subset(cs2, union_cols),
        ], ignore_index=True)

        out_union = os.path.join(args.out_dir, "combined_features_union.csv")
        combined_union.to_csv(out_union, index=False)
        print(f"[OK] Saved: {out_union}  shape={combined_union.shape}")


if __name__ == "__main__":
    main()