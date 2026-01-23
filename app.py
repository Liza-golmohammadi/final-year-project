from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "tappy_web_model.joblib"   # expected model name
THRESH_PATH = APP_DIR / "threshold.json"

DEFAULT_THRESHOLD = 0.75

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
            f"Model not found: {MODEL_PATH.name}. Train it with train_web_model.py "
            "and place it in this folder."
        )
    bundle = joblib.load(MODEL_PATH)
    if not (isinstance(bundle, dict) and "model" in bundle and "features" in bundle):
        raise ValueError("Model bundle must be a dict with keys: 'model' and 'features'.")
    return bundle["model"], bundle["features"]

MODEL, FEATURES = None, None

def ensure_model_loaded():
    global MODEL, FEATURES
    if MODEL is None:
        MODEL, FEATURES = load_bundle()

def build_X(payload: Dict[str, Any]) -> "pd.DataFrame":
    ensure_model_loaded()
    row = {f: float(payload.get(f, 0.0)) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


app = Flask(__name__)

@app.get("/")
def index():
    ensure_model_loaded()
    return render_template("index.html")

@app.post("/predict")
def predict():
    ensure_model_loaded()
    payload = request.get_json(force=True, silent=False) or {}
    try:
        X = build_X(payload)
        proba = float(MODEL.predict_proba(X)[0, 1])
        t = load_threshold()
        pred = int(proba >= t)
        return jsonify({
            "ok": True,
            "threshold": t,
            "prob_pd": proba,
            "prediction": pred,
            "label": "PD-likely" if pred == 1 else "Control-likely"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
