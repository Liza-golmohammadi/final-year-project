from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, render_template, request, jsonify

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "neurotype_model.joblib"
THRESH_PATH = APP_DIR / "threshold.json"

DEFAULT_THRESHOLD = 0.5

FEATURE_LABELS = {
    "hold_mean":   "average hold time",
    "hold_std":    "hold time spread",
    "hold_cv":     "how unevenly you held keys",
    "hold_iqr":    "hold time range",
    "hold_median": "median hold time",
    "hold_p95":    "peak hold time",
    "dd_mean":     "average typing speed",
    "dd_std":      "typing speed spread",
    "dd_cv":       "how variable your typing rhythm was",
    "dd_iqr":      "typing cadence range",
    "dd_median":   "median typing cadence",
    "dd_p95":      "slowest keystroke intervals",
    "ud_mean":     "average gap between keystrokes",
    "ud_std":      "keystroke gap spread",
    "ud_cv":       "how irregular your keystroke gaps were",
    "ud_iqr":      "keystroke gap range",
    "ud_median":   "median gap between keystrokes",
    "ud_p95":      "longest inter-key gaps",
    "pause_rate_dd_gt_500":  "how often you paused mid-typing (>500 ms)",
    "pause_rate_dd_gt_750":  "how often you paused mid-typing (>750 ms)",
    "pause_rate_dd_gt_1000": "how often you paused for over a second",
    "dig_KK_rate": "key-to-key transition rate",
    "dig_KS_rate": "key-to-space transition rate",
    "dig_SK_rate": "space-to-key transition rate",
    "dig_SS_rate": "space-to-space transition rate",
    "n":           "number of keystrokes",
}


def load_threshold() -> float:
    if THRESH_PATH.exists():
        try:
            data = json.loads(THRESH_PATH.read_text(encoding="utf-8"))
            return float(data.get("threshold", DEFAULT_THRESHOLD))
        except Exception:
            return DEFAULT_THRESHOLD
    return DEFAULT_THRESHOLD


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.name}. Run export_model.py first to train and save it."
        )
    bundle = joblib.load(MODEL_PATH)
    if not (isinstance(bundle, dict) and "model" in bundle and "features" in bundle):
        raise ValueError("Model bundle must be a dict with keys: 'model' and 'features'.")
    return bundle["model"], bundle["features"]


MODEL, FEATURES = None, None
EXPLAINER = None


def ensure_model_loaded():
    global MODEL, FEATURES, EXPLAINER
    if MODEL is None:
        MODEL, FEATURES = load_bundle()
        EXPLAINER = shap.TreeExplainer(MODEL)


def build_X(payload: Dict[str, Any]) -> "pd.DataFrame":
    ensure_model_loaded()
    row = {f: float(payload.get(f, 0.0)) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


def get_top_contributors(X: "pd.DataFrame", n: int = 6) -> List[Dict]:
    global EXPLAINER, FEATURES
    shap_vals = EXPLAINER.shap_values(X)

    # Handle both old API (list of arrays per class) and new API (ndarray)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        sv = np.array(shap_vals[1][0])
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]
    else:
        sv = np.array(shap_vals[0])

    pairs = sorted(
        zip(FEATURES, sv, X.iloc[0].values),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return [
        {
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat.replace("_", " ")),
            "shap": float(val),
            "value": float(fval),
        }
        for feat, val, fval in pairs[:n]
    ]


app = Flask(__name__)


@app.get("/")
def index():
    ensure_model_loaded()
    return render_template("index.html")


@app.get("/speech")
def speech():
    return render_template("speech.html")


@app.post("/predict")
def predict():
    ensure_model_loaded()
    payload = request.get_json(force=True, silent=False) or {}
    try:
        X = build_X(payload)
        proba = float(MODEL.predict_proba(X)[0, 1])
        t = load_threshold()
        pred = int(proba >= t)
        contributors = get_top_contributors(X)
        return jsonify({
            "ok": True,
            "threshold": t,
            "prob_pd": proba,
            "prediction": pred,
            "label": "PD-likely" if pred == 1 else "Control-likely",
            "top_contributors": contributors,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
