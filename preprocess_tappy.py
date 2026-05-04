#!/usr/bin/env python3

#To Run the code
# python3 preprocess_tappy.py \
#   --tappy_root "/path/to/tappy-keystroke-data-1.0.0" \
#   --out_dir "out_tappy"



from __future__ import annotations

import os
import glob
import argparse
import numpy as np
import pandas as pd

UNIFIED_COLS = [
    "dataset", "subject_id", "session_id", "label_pd",
    "hold_ms", "dd_ms", "ud_ms", "digraph_class"
]

def safe_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan

def hand_to_class_tappy(h: str) -> str:
    h = str(h).strip().upper()
    return "S" if h == "S" else "K"

def parse_user_file(path: str) -> dict:
    meta = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta

def read_transition_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", header=None, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    # Expected 8 columns; trim/pad
    if df.shape[1] < 8:
        for _ in range(8 - df.shape[1]):
            df[df.shape[1]] = np.nan
    df = df.iloc[:, :8]
    df.columns = ["subject_id", "date", "time", "hand", "hold_ms", "digraph", "dd_ms", "ud_ms"]
    return df

def parse_session(path: str, user_meta: dict) -> pd.DataFrame:
    df = read_transition_file(path)
    if df.empty:
        return pd.DataFrame(columns=UNIFIED_COLS)

    df["hold_ms"] = df["hold_ms"].map(safe_float)
    df["dd_ms"] = df["dd_ms"].map(safe_float)
    df["ud_ms"] = df["ud_ms"].map(safe_float)

    label_pd = str(user_meta.get("Parkinsons", "")).strip().lower() in {"true", "1", "yes"}
    df["label_pd"] = 1 if label_pd else 0

    df["session_id"] = os.path.splitext(os.path.basename(path))[0]

    # Collapse digraph into {K,S}{K,S}
    df["k1"] = df["hand"].map(hand_to_class_tappy)
    dig = df["digraph"].astype(str).str.strip().str.upper()
    second = dig.str[1].fillna("L")  # fallback -> K
    df["k2"] = second.map(hand_to_class_tappy)
    df["digraph_class"] = df["k1"] + df["k2"]

    df = df[np.isfinite(df["hold_ms"]) & np.isfinite(df["dd_ms"]) & np.isfinite(df["ud_ms"])].copy()
    df = df[(df["hold_ms"].between(10, 2000)) & (df["dd_ms"].between(10, 5000)) & (df["ud_ms"].between(-500, 5000))]

    df["dataset"] = "tappy"
    return df[UNIFIED_COLS]

def make_windows(transitions: pd.DataFrame, window_len=200, stride=50) -> pd.DataFrame:
    windows = []
    for (sub, sess), g in transitions.groupby(["subject_id", "session_id"], sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < window_len:
            continue
        for w, s in enumerate(range(0, n - window_len + 1, stride)):
            e = s + window_len
            windows.append({
                "dataset": "tappy",
                "subject_id": sub,
                "session_id": sess,
                "window_id": f"tappy:{sub}:{sess}:{w}",
                "label_pd": int(g.loc[0, "label_pd"]),
                "idx_start": s,
                "idx_end": e
            })
    return pd.DataFrame(windows)

def summarize(x: np.ndarray, prefix: str) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean","std","median","iqr","cv","p95"]}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    med = float(np.median(x))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    cv = float(std/mean) if mean else np.nan
    p95 = float(np.percentile(x, 95))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": med,
        f"{prefix}_iqr": iqr,
        f"{prefix}_cv": cv,
        f"{prefix}_p95": p95,
    }

def featurize(transitions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    grouped = {(sub, sess): g.reset_index(drop=True)
               for (sub, sess), g in transitions.groupby(["subject_id","session_id"], sort=False)}
    out = []
    for _, w in windows.iterrows():
        g = grouped[(w["subject_id"], w["session_id"])]
        seg = g.iloc[int(w["idx_start"]):int(w["idx_end"])]

        hold = seg["hold_ms"].to_numpy(float)
        dd = seg["dd_ms"].to_numpy(float)
        ud = seg["ud_ms"].to_numpy(float)

        row = {
            "window_id": w["window_id"],
            "dataset": "tappy",
            "subject_id": w["subject_id"],
            "session_id": w["session_id"],
            "label_pd": int(w["label_pd"]),
            "n": len(seg)
        }
        row.update(summarize(hold, "hold"))
        row.update(summarize(dd, "dd"))
        row.update(summarize(ud, "ud"))

        for thr in (500, 750, 1000):
            row[f"pause_rate_dd_gt_{thr}"] = float(np.mean(dd > thr))

        dig = seg["digraph_class"].astype(str).to_numpy()
        for cls in ["KK","KS","SK","SS"]:
            row[f"dig_{cls}_rate"] = float(np.mean(dig == cls))

        out.append(row)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tappy_root", required=True, help="Path to tappy-keystroke-data-1.0.0 root")
    ap.add_argument("--out_dir", default="out_tappy", help="Output folder")
    ap.add_argument("--window_len", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    user_files = glob.glob(os.path.join(args.tappy_root, "**", "User_*.txt"), recursive=True)
    meta_by_id = {}
    for up in user_files:
        uid = os.path.basename(up).replace("User_", "").replace(".txt", "")
        meta_by_id[uid] = parse_user_file(up)

    session_files = [p for p in glob.glob(os.path.join(args.tappy_root, "**", "*.txt"), recursive=True)
                     if not os.path.basename(p).startswith("User_")]

    rows = []
    bad = 0
    for sp in session_files:
        uid = os.path.basename(sp).split("_")[0]
        meta = meta_by_id.get(uid)
        if meta is None:
            continue
        try:
            df = parse_session(sp, meta)
            if not df.empty:
                rows.append(df)
        except Exception as e:
            bad += 1
            print(f"[WARN] failed {sp}: {e}")

    if not rows:
        raise RuntimeError("Loaded 0 tappy rows. Check your folder path.")

    trans = pd.concat(rows, ignore_index=True)
    trans_path = os.path.join(args.out_dir, "tappy_transitions.csv")
    trans.to_csv(trans_path, index=False)
    print(f"[OK] transitions: {len(trans):,} -> {trans_path} (bad files: {bad})")

    windows = make_windows(trans, window_len=args.window_len, stride=args.stride)
    win_path = os.path.join(args.out_dir, "tappy_windows.csv")
    windows.to_csv(win_path, index=False)
    print(f"[OK] windows: {len(windows):,} -> {win_path}")

    feats = featurize(trans, windows)
    feat_path = os.path.join(args.out_dir, "tappy_window_features.csv")
    feats.to_csv(feat_path, index=False)
    print(f"[OK] features: {len(feats):,} -> {feat_path}")

if __name__ == "__main__":
    main()