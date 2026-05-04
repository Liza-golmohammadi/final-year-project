"""
Keystroke PD project — unified parsing + feature extraction

Supports:
- Tappy-style dataset:
  - User files:     User_<ID>.txt   (metadata + PD label)
  - Session files:  <ID>_<YYMM>.txt (tab/space delimited transitions)
    Example row:
      0EA27ICBLF  160722  18:41:04.336  L  0101.6  LL  0234.4  0156.3

- NeuroQWERTY MIT-CS1PD / MIT-CS2PD:
  - GT files: GT_DataPD_MIT-CS1PD.csv / GT_DataPD_MIT-CS2PD.csv
  - Data files: each row:
      key  hold_sec  press_sec  release_sec
    (MIT-CS1 sometimes has a malformed first row; we drop malformed rows safely)

Outputs:
- Transition-level table (common denominator):
  subject_id, session_id, label_pd, updrs, hold_ms, dd_ms, ud_ms, digraph_class
- Window-level features for SVM/RF
- Optional sequences for LSTM

Dependencies: pandas, numpy, scipy (optional), scikit-learn (optional)
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

def safe_float(x) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan

def robust_read_whitespace(path: str, ncols: int) -> pd.DataFrame:
    """
    Reads whitespace/tab-delimited rows with unknown spacing.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        dtype=str,
        comment="#",
    )
    # Pad / trim columns to expected ncols
    if df.shape[1] < ncols:
        for _ in range(ncols - df.shape[1]):
            df[df.shape[1]] = np.nan
    elif df.shape[1] > ncols:
        df = df.iloc[:, :ncols]
    return df


def key_to_class_mit(key: str) -> str:
    """
    Collapse MIT key identity into {K,S,P} to match cross-dataset constraints.
    K = any letter/digit key (general "key"), S = space, P = punctuation/other
    """
    if key is None:
        return "K"
    k = str(key).strip().strip('"').lower()
    if k == "space":
        return "S"
    if len(k) == 1 and k.isalnum():
        return "K"
    return "P"


def hand_to_class_tappy(hand: str) -> str:
    """
    Tappy uses L/R/S. Collapse to {K,S}.
    """
    h = str(hand).strip().upper()
    if h == "S":
        return "S"
    return "K"


def summarize_series(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Robust summary stats. Skew is computed via moment (no scipy).
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean","median","std","iqr","cv","p95","skew"]}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    median = float(np.median(x))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    cv = float(std / mean) if mean != 0 else np.nan
    p95 = float(np.percentile(x, 95))
    # skewness (moment-based)
    if x.size > 2 and std > 0:
        m3 = np.mean((x - mean) ** 3)
        skew = float(m3 / (std ** 3))
    else:
        skew = np.nan
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": median,
        f"{prefix}_std": std,
        f"{prefix}_iqr": iqr,
        f"{prefix}_cv": cv,
        f"{prefix}_p95": p95,
        f"{prefix}_skew": skew,
    }


# ----------------------------
# Tappy parsing
# ----------------------------

def parse_tappy_user_file(user_path: str) -> Dict[str, str]:
    """
    Parses key: value lines from User_<ID>.txt
    """
    meta = {}
    with open(user_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def parse_tappy_session_file(session_path: str, user_meta: Dict[str, str]) -> pd.DataFrame:
    """
    Parses <ID>_<YYMM>.txt transition rows into a common transition table.

    Expected columns:
      0 user_id
      1 date (YYMMDD)
      2 time (HH:MM:SS.mmm)
      3 hand/keycat (L/R/S)
      4 hold_ms
      5 digraph (LL/LR/LS/...)
      6 dd_ms
      7 ud_ms
    """
    df = robust_read_whitespace(session_path, ncols=8)
    df.columns = ["subject_id", "date", "time", "hand", "hold_ms", "digraph", "dd_ms", "ud_ms"]

    # Convert to floats (ms)
    df["hold_ms"] = df["hold_ms"].map(safe_float)
    df["dd_ms"] = df["dd_ms"].map(safe_float)
    df["ud_ms"] = df["ud_ms"].map(safe_float)

    # Label
    label_pd = str(user_meta.get("Parkinsons", "")).strip().lower() in {"true", "1", "yes"}
    df["label_pd"] = 1 if label_pd else 0
    df["updrs"] = np.nan  # not present in sample; keep column for unification

    # Session id from filename or date/time
    base = os.path.basename(session_path)
    session_id = os.path.splitext(base)[0]
    df["session_id"] = session_id

    # Digraph class collapsed to {K,S}{K,S}
    # current class from 'hand'; next class from digraph second char
    df["k1"] = df["hand"].map(hand_to_class_tappy)
    df["k2"] = df["digraph"].astype(str).str.strip().str.upper().str[1].map(hand_to_class_tappy)
    df["digraph_class"] = df["k1"] + df["k2"]
    df.drop(columns=["k1", "k2"], inplace=True)

    # Basic filtering
    df = df[np.isfinite(df["hold_ms"]) & np.isfinite(df["dd_ms"]) & np.isfinite(df["ud_ms"])].copy()
    df = df[(df["hold_ms"].between(10, 2000)) & (df["dd_ms"].between(10, 5000)) & (df["ud_ms"].between(-500, 5000))]

    df["dataset"] = "tappy"
    return df[[
        "dataset","subject_id","session_id","label_pd","updrs",
        "hold_ms","dd_ms","ud_ms","digraph_class"
    ]]


def load_tappy_dataset(root: str) -> pd.DataFrame:
    """
    root should contain Archived Users + Archived Data (or similar).
    We’ll match User_<ID>.txt to all session files that start with <ID>_.
    """
    user_files = glob.glob(os.path.join(root, "**", "User_*.txt"), recursive=True)
    all_rows = []

    # Index user meta by ID
    user_meta_by_id = {}
    for up in user_files:
        uid = os.path.basename(up).replace("User_", "").replace(".txt", "")
        user_meta_by_id[uid] = parse_tappy_user_file(up)

    # Find candidate session files: <ID>_*.txt excluding User_*.txt
    session_files = [p for p in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)
                     if os.path.basename(p).startswith(tuple(user_meta_by_id.keys())) and not os.path.basename(p).startswith("User_")]

    for sp in session_files:
        uid = os.path.basename(sp).split("_")[0]
        meta = user_meta_by_id.get(uid, {})
        try:
            all_rows.append(parse_tappy_session_file(sp, meta))
        except Exception as e:
            print(f"[WARN] Failed parsing Tappy session {sp}: {e}")

    if not all_rows:
        return pd.DataFrame(columns=[
            "dataset","subject_id","session_id","label_pd","updrs","hold_ms","dd_ms","ud_ms","digraph_class"
        ])
    return pd.concat(all_rows, ignore_index=True)


# ----------------------------
# MIT NeuroQWERTY parsing
# ----------------------------

def load_mit_gt(gt_csv_path: str, dataset_tag: str) -> pd.DataFrame:
    """
    Loads GT_DataPD_MIT-CS1PD.csv or GT_DataPD_MIT-CS2PD.csv

    Expected columns (as shown):
      pID, gt, updrs108, ..., file_1, (file_2)
    """
    gt = pd.read_csv(gt_csv_path)
    gt = gt.rename(columns={
        "pID": "subject_id",
        "gt": "gt_pd",
        "updrs108": "updrs",
    })
    gt["label_pd"] = gt["gt_pd"].astype(str).str.upper().eq("TRUE").astype(int)
    gt["dataset"] = dataset_tag

    # Normalize file columns that exist
    file_cols = [c for c in gt.columns if c.lower().startswith("file_")]
    keep = ["dataset","subject_id","label_pd","updrs"] + file_cols
    return gt[keep].copy()


def parse_mit_data_file(data_path: str, subject_id: int | str, session_id: str, label_pd: int, updrs: float, dataset_tag: str) -> pd.DataFrame:
    """
    Reads MIT file rows into event-level then derives transitions.

    Expected per-row content:
      key  hold_sec  press_sec  release_sec
    Some MIT-CS1 files may include a malformed first row: we coerce and drop invalid.
    """
    # MIT files are comma-delimited CSVs
    # Column order per PhysioNet docs: key, hold_sec, release_sec, press_sec
    df = pd.read_csv(data_path, header=None, dtype=str,
                     names=["key", "hold_sec", "release_sec", "press_sec"])

    df["hold_sec"]    = df["hold_sec"].map(safe_float)
    df["press_sec"]   = df["press_sec"].map(safe_float)
    df["release_sec"] = df["release_sec"].map(safe_float)

    # Drop rows that can't be parsed
    df = df[np.isfinite(df["hold_sec"]) & np.isfinite(df["press_sec"]) & np.isfinite(df["release_sec"])].copy()

    # Convert to ms (times are seconds from time 0)
    df["t_down_ms"] = df["press_sec"]   * 1000.0
    df["t_up_ms"]   = df["release_sec"] * 1000.0
    df["hold_ms"]   = df["hold_sec"]    * 1000.0

    # Validity filters matching nqDataLoader sanity checks
    df = df[(df["hold_sec"] > 0) & (df["hold_sec"] < 5)]
    df = df[df["t_up_ms"] >= df["t_down_ms"]]
    df = df[df["hold_ms"].between(10, 2000)]

    # Sort by press time (keystroke order)
    df = df.sort_values("t_down_ms").reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame(columns=[
            "dataset","subject_id","session_id","label_pd","updrs","hold_ms","dd_ms","ud_ms","digraph_class"
        ])

    # Compute transitions
    t_down = df["t_down_ms"].to_numpy()
    t_up = df["t_up_ms"].to_numpy()
    hold = df["hold_ms"].to_numpy()

    dd = t_down[1:] - t_down[:-1]
    ud = t_down[1:] - t_up[:-1]

    # Digraph class collapsed: {K,S,P}{K,S,P} then optionally collapse to {K,S} for maximum parity
    kclass = df["key"].map(key_to_class_mit).to_numpy()
    digraph = np.char.add(kclass[:-1].astype(str), kclass[1:].astype(str))

    out = pd.DataFrame({
        "dataset": dataset_tag,
        "subject_id": str(subject_id),
        "session_id": session_id,
        "label_pd": int(label_pd),
        "updrs": safe_float(updrs),
        "hold_ms": hold[:-1],   # align with transition (current key)
        "dd_ms": dd,
        "ud_ms": ud,
        "digraph_class": digraph,
    })

    # Filter plausible ranges
    out = out[(out["dd_ms"].between(10, 5000)) & (out["ud_ms"].between(-500, 5000))].copy()
    return out


def load_mit_dataset(gt_csv_path: str, data_dir: str, dataset_tag: str) -> pd.DataFrame:
    """
    Loads GT then parses all referenced data files from the data_dir.
    """
    gt = load_mit_gt(gt_csv_path, dataset_tag)
    all_rows = []

    file_cols = [c for c in gt.columns if c.lower().startswith("file_")]
    for _, row in gt.iterrows():
        sid = row["subject_id"]
        label_pd = row["label_pd"]
        updrs = row.get("updrs", np.nan)

        for fc in file_cols:
            fname = row.get(fc, None)
            if pd.isna(fname) or not str(fname).strip():
                continue
            fpath = os.path.join(data_dir, str(fname))
            if not os.path.exists(fpath):
                print(f"[MISS] not found: {fpath}")
                # allow user to pass absolute paths or different structure
                continue

            session_id = os.path.splitext(str(fname))[0]
            try:
                all_rows.append(parse_mit_data_file(
                    data_path=fpath,
                    subject_id=sid,
                    session_id=session_id,
                    label_pd=label_pd,
                    updrs=updrs,
                    dataset_tag=dataset_tag
                ))
            except Exception as e:
                print(f"[WARN] Failed parsing MIT file {fpath}: {e}")

    if not all_rows:
        return pd.DataFrame(columns=[
            "dataset","subject_id","session_id","label_pd","updrs","hold_ms","dd_ms","ud_ms","digraph_class"
        ])
    return pd.concat(all_rows, ignore_index=True)


# ----------------------------
# Windowing + feature extraction
# ----------------------------

@dataclass
class WindowConfig:
    window_len: int = 200
    stride: int = 50


def make_windows(transitions: pd.DataFrame, cfg: WindowConfig) -> pd.DataFrame:
    """
    Splits transition rows into fixed-length windows per (dataset, subject_id, session_id).

    Output: window table with columns:
      dataset, subject_id, session_id, window_id, label_pd, updrs, idx_start, idx_end
    """
    required = {"dataset","subject_id","session_id","label_pd","updrs","hold_ms","dd_ms","ud_ms","digraph_class"}
    missing = required - set(transitions.columns)
    if missing:
        raise ValueError(f"Transitions missing columns: {missing}")

    windows = []
    for (ds, sub, sess), g in transitions.groupby(["dataset","subject_id","session_id"], sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < cfg.window_len:
            continue
        starts = range(0, n - cfg.window_len + 1, cfg.stride)
        for w, s in enumerate(starts):
            e = s + cfg.window_len
            windows.append({
                "dataset": ds,
                "subject_id": sub,
                "session_id": sess,
                "window_id": f"{ds}:{sub}:{sess}:{w}",
                "label_pd": int(g.loc[0, "label_pd"]),
                "updrs": safe_float(g.loc[0, "updrs"]),
                "idx_start": s,
                "idx_end": e,
            })
    return pd.DataFrame(windows)


def featurize_windows(transitions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """
    Computes tabular features per window for SVM/RF.
    """
    feats = []
    # Pre-index groups for speed
    grouped = {
        (ds, sub, sess): g.reset_index(drop=True)
        for (ds, sub, sess), g in transitions.groupby(["dataset","subject_id","session_id"], sort=False)
    }

    for _, w in windows.iterrows():
        key = (w["dataset"], w["subject_id"], w["session_id"])
        g = grouped.get(key)
        if g is None:
            continue
        seg = g.iloc[int(w["idx_start"]):int(w["idx_end"])]

        hold = seg["hold_ms"].to_numpy(dtype=float)
        dd = seg["dd_ms"].to_numpy(dtype=float)
        ud = seg["ud_ms"].to_numpy(dtype=float)

        row = {
            "window_id": w["window_id"],
            "dataset": w["dataset"],
            "subject_id": w["subject_id"],
            "session_id": w["session_id"],
            "label_pd": int(w["label_pd"]),
            "updrs": safe_float(w["updrs"]),
            "n": len(seg),
        }
        row.update(summarize_series(hold, "hold"))
        row.update(summarize_series(dd, "dd"))
        row.update(summarize_series(ud, "ud"))

        # Pause features (tune thresholds later)
        for thr in (500, 750, 1000):
            row[f"pause_rate_dd_gt_{thr}"] = float(np.mean(dd > thr)) if len(dd) else np.nan
            row[f"pause_p95_dd_gt_{thr}"] = float(np.percentile(dd[dd > thr], 95)) if np.any(dd > thr) else 0.0

        # Digraph proportions (works for both datasets; MIT includes P too)
        dig = seg["digraph_class"].astype(str).to_numpy()
        for cls in ["KK","KS","SK","SS","KP","PK","SP","PS","PP"]:
            row[f"dig_{cls}_rate"] = float(np.mean(dig == cls))

        feats.append(row)

    return pd.DataFrame(feats)


def build_sequences(transitions: pd.DataFrame, windows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds sequence arrays for CNN:
      X: (num_windows, window_len, 3)  -> [hold_ms, dd_ms, ud_ms]
      y: (num_windows,)
      groups: (num_windows,) subject_id for subject-disjoint splitting

    Per-window z-scoring is applied to reduce typing-speed confound across subjects.
    """
    grouped = {
        (ds, sub, sess): g.reset_index(drop=True)
        for (ds, sub, sess), g in transitions.groupby(["dataset","subject_id","session_id"], sort=False)
    }
    X_list, y_list, grp_list = [], [], []
    for _, w in windows.iterrows():
        key = (w["dataset"], w["subject_id"], w["session_id"])
        g = grouped.get(key)
        if g is None:
            continue
        seg = g.iloc[int(w["idx_start"]):int(w["idx_end"])]
        arr = seg[["hold_ms","dd_ms","ud_ms"]].to_numpy(dtype=float)

        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0) + 1e-6
        arr = (arr - mu) / sd

        X_list.append(arr)
        y_list.append(int(w["label_pd"]))
        grp_list.append(str(w["subject_id"]))

    return np.stack(X_list), np.array(y_list, dtype=int), np.array(grp_list, dtype=str)


# ----------------------------
# Example usage (edit paths)
# ----------------------------

if __name__ == "__main__":
    # --- EDIT THESE PATHS ---
    TAPPY_ROOT = r"c:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs\tappy-keystroke-data-1.0.0"  # contains Archived Data + Archived Users
    MIT_CS1_GT = r"c:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs\neuroqwerty-mit-csxpd-dataset-1.0.0\neuroQWERTY\MIT-CS1PD\GT_DataPD_MIT-CS1PD.csv"
    MIT_CS1_DATA = r"c:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs\neuroqwerty-mit-csxpd-dataset-1.0.0\neuroQWERTY\MIT-CS1PD\data_MIT-CS1PD"
    MIT_CS2_GT = r"c:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs\neuroqwerty-mit-csxpd-dataset-1.0.0\neuroQWERTY\MIT-CS2PD\GT_DataPD_MIT-CS2PD.csv"
    MIT_CS2_DATA = r"c:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs\neuroqwerty-mit-csxpd-dataset-1.0.0\neuroQWERTY\MIT-CS2PD\data_MIT-CS2PD"

    # Load transitions
    tappy_trans = load_tappy_dataset(TAPPY_ROOT)
    cs1_trans = load_mit_dataset(MIT_CS1_GT, MIT_CS1_DATA, dataset_tag="mit_cs1")
    cs2_trans = load_mit_dataset(MIT_CS2_GT, MIT_CS2_DATA, dataset_tag="mit_cs2")

    transitions = pd.concat([tappy_trans, cs1_trans, cs2_trans], ignore_index=True)

    # Save transition-level unified table
    transitions.to_csv("unified_transitions.csv", index=False)
    print("Saved unified_transitions.csv", transitions.shape)

    # Window + features
    cfg = WindowConfig(window_len=200, stride=50)
    windows = make_windows(transitions, cfg)
    feats = featurize_windows(transitions, windows)

    feats.to_csv("window_features.csv", index=False)
    windows.to_csv("windows_index.csv", index=False)
    print("Saved window_features.csv", feats.shape)

    # Sequence arrays for CNN
    X, y, groups = build_sequences(transitions, windows)
    np.save("X_sequences.npy", X)
    np.save("y_labels.npy", y)
    np.save("groups_subject.npy", groups)
    print("Saved sequence arrays:", X.shape, y.shape, groups.shape)
