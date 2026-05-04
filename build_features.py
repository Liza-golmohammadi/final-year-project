"""
Build tappy_features.csv using a sliding window over keystrokes.

Each row = one window of WINDOW_SIZE keystrokes (stride = STRIDE keystrokes).
This gives many more samples than session-level aggregation.

Usage:
    python build_features.py --data tappy-keystroke-data-1.0.0
    python build_features.py --data tappy-keystroke-data-1.0.0 --window 100 --stride 50

Outputs:
    tappy_features.csv   (window-level, used by train_web_model.py / phase scripts)
    tappy_userlevel.csv  (user-level aggregated, used by userlevel_model.py)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


WINDOW_SIZE = 100   # keystrokes per window
STRIDE      = 50    # step between windows (50% overlap by default)


def parse_keystroke_file(path):
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            try:
                rows.append({
                    "UserKey":   parts[0],
                    "Date":      parts[1],
                    "Timestamp": parts[2],
                    "Hand":      parts[3],
                    "Hold":      float(parts[4]),
                    "Direction": parts[5],
                    "Latency":   float(parts[6]),
                    "Flight":    float(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def parse_user_file(path):
    info = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                info[k.strip()] = v.strip()
    return info


def window_features(df_win):
    """Compute features for one window of keystrokes."""
    hold = df_win["Hold"].values
    lat  = df_win["Latency"].values
    flt  = df_win["Flight"].values

    # Active duration from timestamps in this window
    try:
        times = pd.to_datetime(df_win["Timestamp"], format="%H:%M:%S.%f", errors="coerce")
        duration_min = (times.max() - times.min()).total_seconds() / 60.0
        if pd.isna(duration_min) or duration_min <= 0:
            duration_min = 1.0
    except Exception:
        duration_min = 1.0

    n_keys = len(df_win)
    kpm    = n_keys / duration_min

    long_gap_rate = float(np.mean(lat > 2000))
    burst_count   = int(np.sum(lat < 500))

    return {
        "active_kpm_mean": kpm,
        "active_kpm_std":  float(np.std(hold)),
        "burst_count":     burst_count,
        "active_minutes":  duration_min,
        "long_gap_rate":   long_gap_rate,
        "hold_mean":       float(np.mean(hold)),
        "hold_std":        float(np.std(hold)),
        "lat_mean":        float(np.mean(lat)),
        "lat_std":         float(np.std(lat)),
        "flt_mean":        float(np.mean(flt)),
        "flt_std":         float(np.std(flt)),
        "flt_neg_rate":    float(np.mean(flt < 0)),
        "n_keys":          n_keys,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="tappy-keystroke-data-1.0.0")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE,
                    help="Keystrokes per window (default: 100)")
    ap.add_argument("--stride", type=int, default=STRIDE,
                    help="Stride between windows (default: 50)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    key_dir  = data_dir / "Archived-Data"
    user_dir = data_dir / "Archived-Users"

    # Load PD labels
    user_info = {}
    for ufile in user_dir.glob("User_*.txt"):
        uid  = ufile.stem.replace("User_", "")
        info = parse_user_file(ufile)
        pd_val = info.get("Parkinsons", "False").strip().lower()
        user_info[uid] = 1 if pd_val == "true" else 0

    print(f"Found {len(user_info)} users  |  window={args.window}  stride={args.stride}")

    all_windows = []
    skipped = 0

    for kfile in sorted(key_dir.glob("*.txt")):
        stem  = kfile.stem          # e.g. 0EA27ICBLF_1607
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        uid   = parts[0]
        month = parts[1]

        if uid not in user_info:
            skipped += 1
            continue

        raw = parse_keystroke_file(kfile)
        if len(raw) < args.window:
            continue                # not enough keystrokes for even one window

        df = pd.DataFrame(raw)

        # Slide window over keystrokes
        n = len(df)
        for start in range(0, n - args.window + 1, args.stride):
            df_win = df.iloc[start : start + args.window]
            feats  = window_features(df_win)
            feats["UserKey"] = uid
            feats["month"]   = month
            feats["window"]  = start // args.stride
            feats["label"]   = user_info[uid]
            all_windows.append(feats)

    print(f"Skipped {skipped} files (no matching user label)")

    session_df = pd.DataFrame(all_windows)
    session_df.to_csv("tappy_features.csv", index=False)
    print(f"Saved tappy_features.csv  —  {len(session_df)} windows  "
          f"({session_df['label'].sum()} PD / {(session_df['label']==0).sum()} HC)")

    # User-level CSV: median + std of window features per user
    feature_cols = [
        "hold_mean", "hold_std", "lat_mean", "lat_std",
        "flt_mean", "flt_std", "flt_neg_rate",
        "active_kpm_mean", "active_kpm_std", "burst_count",
        "active_minutes", "long_gap_rate", "n_keys",
    ]
    agg = {}
    for col in feature_cols:
        agg[f"med_{col}"] = (col, "median")
        agg[f"std_{col}"] = (col, "std")

    user_df = session_df.groupby("UserKey").agg(**agg).reset_index()
    user_df["label"]    = user_df["UserKey"].map(user_info)
    user_df["n_months"] = session_df.groupby("UserKey")["month"].nunique().values
    user_df["total_keys"] = session_df.groupby("UserKey")["n_keys"].sum().values
    user_df = user_df.fillna(0)
    user_df.to_csv("tappy_userlevel.csv", index=False)
    print(f"Saved tappy_userlevel.csv —  {len(user_df)} users")


if __name__ == "__main__":
    main()
