# Final Year Project: NeuroType (Tappy Web Demo)

This repository contains a small Flask web demo for predicting Parkinson’s vs control from **derived typing features** (e.g., hold time, flight time, typing rate).  
It loads a scikit-learn model (`tappy_web_model.joblib`) and an optional decision threshold (`threshold.json`).

## Requirements
- Python 3.10+ recommended
- Packages in `requirements.txt`

## Quick start (run the web app)
```bash
pip install -r requirements.txt
python app.py
```

Open in your browser:
- http://127.0.0.1:5000

## Training the model (if you want to retrain)
If you already have `tappy_web_model.joblib`, you can skip this section.

Your CSV should include:
- `label` (0/1)
- `UserKey` (participant/user id for group split)
- feature columns used by the web demo

Train:
```bash
python train_web_model.py --csv PATH_TO_YOUR_tappy_features.csv --out tappy_web_model.joblib
```

## Choose a better threshold (optional, recommended)
This writes `threshold.json`, which the web app reads at runtime:
```bash
python find_best_threshold.py --csv PATH_TO_YOUR_tappy_features.csv --model tappy_web_model.joblib --out threshold.json
```

## Project structure
- `app.py` — Flask server (`/` and `/predict`)
- `templates/index.html` — simple front-end UI
- `train_web_model.py` — trains and saves the model
- `find_best_threshold.py` — finds a threshold to improve balanced accuracy
- `requirements.txt` — dependencies
- `tappy_web_model.joblib` — saved model (if already trained)
- `threshold.json` — saved threshold (optional)

## Notes
- The demo sends **derived features** to the backend (not raw keystrokes).
- If `threshold.json` is missing, the app falls back to a default threshold.
