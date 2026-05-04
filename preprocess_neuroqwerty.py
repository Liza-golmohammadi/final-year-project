#!/usr/bin/env python3
"""
preprocess_neuroqwerty.py  (RECODED / more robust)

Purpose
-------
Create a clean transition-level dataset (hold_ms, dd_ms, ud_ms, digraph_class)
from NeuroQWERTY MIT-CS1PD / MIT-CS2PD.

Key differences vs previous version
-----------------------------------
- Robust per-keystroke file parsing:
  * Tries whitespace/tab parsing first (matches your pasted examples)
  * Falls back to comma-delimited parsing
  * Forces exactly 4 columns, drops malformed rows
- Does NOT over-filter early; uses conservative sanity filters only
- Uses a directory index (basename -> full path) so path joining never fails
- Produces:
  * <dataset_tag>_transitions.csv
  * <dataset_tag>_windows.csv
  * <dataset_tag>_window_features.csv

Usage
-----
MIT-CS1:
  python3 preprocess_neuroqwerty.py \
    --gt "neuroqwerty-mit-csxpd-dataset-1.0.0/neuroQWERTY/MIT-CS1PD/GT_DataPD_MIT-CS1PD.csv" \
    --data_dir "neuroqwerty-mit-csxpd-dataset-1.0.0/neuroQWERTY/MIT-CS1PD/data_MIT-CS1PD" \
    --dataset_tag "mit_cs1" \
    --out_dir "out_neuroqwerty"

MIT-CS2:
  python3 preprocess_neuroqwerty.py \
    --gt "neuroqwerty-mit-csxpd-dataset-1.0.0/neuroQWERTY/MIT-CS2PD/GT_DataPD_MIT-CS2PD.csv" \
    --data_dir "neuroqwerty-mit-csxpd-dataset-1.0.0/neuroQWERTY/MIT-CS2PD/data_MIT-CS2PD" \
    --dataset_tag "mit_cs2" \
    --out_dir "out_neuroqwerty"
"""

from __future__ import annotations

import os
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


UNIFIED_COLS = [
    "dataset", "subject_id", "session_id", "label_pd", "updrs",
    "hold_ms", "dd_ms", "ud_ms", "digraph_class"
]


# ----------------------------
# Helpers
# ----------------------------

def safe_float(x) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan


def key_to_class_mit(key: str) -> str:
    """
    Collapse MIT key identity into {K,S,P}:
      K = alnum single char, S = space, P = punctuation/other token
    """
    if key is None:
        return "K"
    k = str(key).strip().lower()
    if k == "space":
        return "S"
    if len(k) == 1 and k.isalnum():
        return "K"
    return "P"


def summarize(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "std", "median", "iqr", "cv", "p95"]}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    median = float(np.median(x))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    cv = float(std / mean) if mean else np.nan
    p95 = float(np.percentile(x, 95))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": median,
        f"{prefix}_iqr": iqr,
        f"{prefix}_cv": cv,
        f"{prefix}_p95": p95,
    }


def build_file_index(data_dir: str) -> Dict[str, str]:
    """
    basename -> absolute path
    Avoids all filename/path join issues.
    """
    paths = glob.glob(os.path.join(data_dir, "*.csv"))
    return {os.path.basename(p): p for p in paths}


# ----------------------------
# GT parsing
# ----------------------------

def read_gt(gt_path: str) -> pd.DataFrame:
    """
    Reads the GT file robustly (TSV or CSV).
    Expected columns include: pID, gt, updrs108, file_1, file_2 (optional).
    """
    gt = pd.read_csv(gt_path, sep=None, engine="python")
    gt.columns = [c.strip().replace("\ufeff", "") for c in gt.columns]

    gt = gt.rename(columns={"pID": "subject_id", "gt": "gt_pd", "updrs108": "updrs"})
    if "subject_id" not in gt.columns or "gt_pd" not in gt.columns:
        raise ValueError(f"GT missing required columns. Found: {gt.columns.tolist()}")

    gt["label_pd"] = gt["gt_pd"].astype(str).str.strip().str.upper().eq("TRUE").astype(int)

    file_cols = [c for c in gt.columns if c.lower().startswith("file_")]
    if not file_cols:
        raise ValueError(f"No file_* columns found in GT. Found: {gt.columns.tolist()}")

    keep = ["subject_id", "label_pd", "updrs"] + file_cols
    return gt[keep].copy()


# ----------------------------
# MIT per-keystroke file parsing
# ----------------------------

def read_mit_file(path: str) -> pd.DataFrame:
    """
    Robustly reads a per-keystroke file into 4 columns:
      key, hold_sec, press_sec, release_sec

    Tries whitespace/tab first (most common in your samples).
    Falls back to comma-separated.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["key", "hold_sec", "press_sec", "release_sec"])

    df = None

    # 1) whitespace/tab separated
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", header=None, dtype=str)
    except Exception:
        df = None

    # If that failed or has too few cols, try comma
    if df is None or df.empty or df.shape[1] < 4:
        try:
            df = pd.read_csv(path, sep=",", engine="python", header=None, dtype=str)
        except Exception:
            return pd.DataFrame(columns=["key", "hold_sec", "press_sec", "release_sec"])

    if df.empty or df.shape[1] < 4:
        return pd.DataFrame(columns=["key", "hold_sec", "press_sec", "release_sec"])

    # Force exactly 4 columns
    df = df.iloc[:, :4].copy()
    df.columns = ["key", "hold_sec", "press_sec", "release_sec"]

    # Coerce numeric columns
    df["hold_sec"] = df["hold_sec"].map(safe_float)
    df["press_sec"] = df["press_sec"].map(safe_float)
    df["release_sec"] = df["release_sec"].map(safe_float)

    # Drop malformed rows (handles CS1 weird first line)
    df = df[np.isfinite(df["hold_sec"]) & np.isfinite(df["press_sec"]) & np.isfinite(df["release_sec"])].copy()
    return df


def keystrokes_to_transitions(
    ks: pd.DataFrame,
    dataset_tag: str,
    subject_id: str,
    session_id: str,
    label_pd: int,
    updrs: float
) -> pd.DataFrame:
    """
    Convert keystroke events to transitions:
      hold_ms (of current key),
      dd_ms = down(i+1) - down(i),
      ud_ms = down(i+1) - up(i),
      digraph_class from collapsed key classes.
    """
    if ks.shape[0] < 2:
        return pd.DataFrame(columns=UNIFIED_COLS)

    hold_ms = ks["hold_sec"].to_numpy(dtype=float) * 1000.0
    t_down = ks["press_sec"].to_numpy(dtype=float) * 1000.0
    t_up = ks["release_sec"].to_numpy(dtype=float) * 1000.0

    # Conservative sanity only (avoid dropping everything)
    # hold must be positive and not absurd
    ok = np.isfinite(hold_ms) & np.isfinite(t_down) & np.isfinite(t_up) & (hold_ms > 0) & (hold_ms < 5000)
    hold_ms, t_down, t_up = hold_ms[ok], t_down[ok], t_up[ok]
    keys = ks.loc[ok, "key"].astype(str).to_numpy()

    if hold_ms.size < 2:
        return pd.DataFrame(columns=UNIFIED_COLS)

    # If press/release are swapped in some rows, fix by swapping
    swap = t_up < t_down
    if np.any(swap):
        tmp = t_down[swap].copy()
        t_down[swap] = t_up[swap]
        t_up[swap] = tmp

    # Sort by keydown time
    order = np.argsort(t_down)
    hold_ms, t_down, t_up, keys = hold_ms[order], t_down[order], t_up[order], keys[order]

    # Compute transitions
    dd = t_down[1:] - t_down[:-1]
    ud = t_down[1:] - t_up[:-1]

    # Key classes for digraph
    kclass = np.array([key_to_class_mit(k) for k in keys], dtype=str)
    digraph = np.char.add(kclass[:-1], kclass[1:])

    out = pd.DataFrame({
        "dataset": dataset_tag,
        "subject_id": str(subject_id),
        "session_id": str(session_id),
        "label_pd": int(label_pd),
        "updrs": safe_float(updrs),
        "hold_ms": hold_ms[:-1],
        "dd_ms": dd,
        "ud_ms": ud,
        "digraph_class": digraph
    })

    # Light plausibility bounds (not too strict)
    out = out[np.isfinite(out["hold_ms"]) & np.isfinite(out["dd_ms"]) & np.isfinite(out["ud_ms"])].copy()
    out = out[(out["dd_ms"] > -2000) & (out["dd_ms"] < 20000)]
    out = out[(out["ud_ms"] > -2000) & (out["ud_ms"] < 20000)]
    return out[UNIFIED_COLS]


# ----------------------------
# Windowing + features
# ----------------------------

@dataclass
class WindowConfig:
    window_len: int = 200
    stride: int = 50


def make_windows(transitions: pd.DataFrame, cfg: WindowConfig) -> pd.DataFrame:
    windows = []
    for (sub, sess), g in transitions.groupby(["subject_id", "session_id"], sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < cfg.window_len:
            continue
        for w, s in enumerate(range(0, n - cfg.window_len + 1, cfg.stride)):
            windows.append({
                "dataset": g.loc[0, "dataset"],
                "subject_id": sub,
                "session_id": sess,
                "window_id": f"{g.loc[0, 'dataset']}:{sub}:{sess}:{w}",
                "label_pd": int(g.loc[0, "label_pd"]),
                "updrs": safe_float(g.loc[0, "updrs"]),
                "idx_start": s,
                "idx_end": s + cfg.window_len
            })
    return pd.DataFrame(windows)


def featurize(transitions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    grouped = {(sub, sess): g.reset_index(drop=True)
               for (sub, sess), g in transitions.groupby(["subject_id", "session_id"], sort=False)}

    out = []
    for _, w in windows.iterrows():
        g = grouped[(w["subject_id"], w["session_id"])]
        seg = g.iloc[int(w["idx_start"]):int(w["idx_end"])]

        hold = seg["hold_ms"].to_numpy(float)
        dd = seg["dd_ms"].to_numpy(float)
        ud = seg["ud_ms"].to_numpy(float)

        row = {
            "window_id": w["window_id"],
            "dataset": w["dataset"],
            "subject_id": w["subject_id"],
            "session_id": w["session_id"],
            "label_pd": int(w["label_pd"]),
            "updrs": safe_float(w["updrs"]),
            "n": len(seg)
        }
        row.update(summarize(hold, "hold"))
        row.update(summarize(dd, "dd"))
        row.update(summarize(ud, "ud"))

        for thr in (500, 750, 1000):
            row[f"pause_rate_dd_gt_{thr}"] = float(np.mean(dd > thr)) if dd.size else np.nan

        dig = seg["digraph_class"].astype(str).to_numpy()
        for cls in ["KK", "KS", "SK", "SS", "KP", "PK", "SP", "PS", "PP"]:
            row[f"dig_{cls}_rate"] = float(np.mean(dig == cls))

        out.append(row)

    return pd.DataFrame(out)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset_tag", required=True, help="mit_cs1 or mit_cs2")
    ap.add_argument("--out_dir", default="out_neuroqwerty")
    ap.add_argument("--window_len", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--debug_one", action="store_true", help="Print debug info for first parsed file")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt = read_gt(args.gt)
    idx = build_file_index(args.data_dir)

    file_cols = [c for c in gt.columns if c.lower().startswith("file_")]
    print(f"[INFO] GT rows: {len(gt)} | file cols: {file_cols}")
    print(f"[INFO] Indexed files in data_dir: {len(idx)}")

    transitions_parts = []
    miss = 0
    parsed_files = 0

    for _, r in gt.iterrows():
        sid = r["subject_id"]
        label = int(r["label_pd"])
        updrs = r.get("updrs", np.nan)

        for fc in file_cols:
            fname = r.get(fc, None)
            if pd.isna(fname):
                continue
            base = os.path.basename(str(fname).strip().replace("\ufeff", ""))
            fpath = idx.get(base)

            if fpath is None:
                miss += 1
                continue

            session_id = os.path.splitext(base)[0]
            ks = read_mit_file(fpath)

            if args.debug_one and parsed_files == 0:
                print("[DEBUG] First file:", base)
                print("[DEBUG] Parsed keystrokes shape:", ks.shape)
                print(ks.head(5))

            trans = keystrokes_to_transitions(
                ks=ks,
                dataset_tag=args.dataset_tag,
                subject_id=str(sid),
                session_id=session_id,
                label_pd=label,
                updrs=updrs
            )

            parsed_files += 1
            if not trans.empty:
                transitions_parts.append(trans)

    if not transitions_parts:
        raise RuntimeError(
            f"Loaded 0 transitions. Missing files: {miss}. "
            f"Try running with --debug_one to see how a file is parsed."
        )

    transitions = pd.concat(transitions_parts, ignore_index=True)
    out_trans = os.path.join(args.out_dir, f"{args.dataset_tag}_transitions.csv")
    transitions.to_csv(out_trans, index=False)
    print(f"[OK] transitions rows: {len(transitions):,} -> {out_trans} | missing files: {miss}")

    cfg = WindowConfig(window_len=args.window_len, stride=args.stride)
    windows = make_windows(transitions, cfg)
    out_win = os.path.join(args.out_dir, f"{args.dataset_tag}_windows.csv")
    windows.to_csv(out_win, index=False)
    print(f"[OK] windows: {len(windows):,} -> {out_win}")

    feats = featurize(transitions, windows)
    out_feat = os.path.join(args.out_dir, f"{args.dataset_tag}_window_features.csv")
    feats.to_csv(out_feat, index=False)
    print(f"[OK] features: {len(feats):,} -> {out_feat}")


if __name__ == "__main__":
    main()