\
import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="threshold.json")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feats = bundle["features"]

    df = pd.read_csv(args.csv).dropna(subset=["label", "UserKey"]).copy()
    y = df["label"].astype(int)
    groups = df["UserKey"]

    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    probs_all, ys_all = [], []
    for tr, te in cv.split(X, y, groups=groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        probs_all.append(p)
        ys_all.append(y.iloc[te].to_numpy())

    probs_all = np.concatenate(probs_all)
    ys_all = np.concatenate(ys_all)

    auc = float(roc_auc_score(ys_all, probs_all))
    print("CV AUC:", round(auc, 3))

    ths = np.linspace(0.05, 0.95, 19)
    best_bal, best_t = -1.0, 0.5
    for t in ths:
        pred = (probs_all >= t).astype(int)
        bal = balanced_accuracy_score(ys_all, pred)
        if bal > best_bal:
            best_bal, best_t = float(bal), float(t)

    print("BEST BalancedAcc:", round(best_bal, 3), "at threshold", round(best_t, 2))

    pred = (probs_all >= best_t).astype(int)
    cm = confusion_matrix(ys_all, pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
    print("Sensitivity:", round(float(sens), 3))
    print("Specificity:", round(float(spec), 3))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_t, "cv_auc": auc, "best_balanced_accuracy": best_bal}, f, indent=2)

    print("Saved", args.out)


if __name__ == "__main__":
    main()
