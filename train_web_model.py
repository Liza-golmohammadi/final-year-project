\
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


WEB_FEATURES = [
    "active_kpm_mean",
    "active_kpm_std",
    "burst_count",
    "active_minutes",
    "long_gap_rate",
    "hold_mean",
    "hold_std",
    "lat_mean",
    "lat_std",
    "flt_mean",
    "flt_std",
    "flt_neg_rate",
    "n_keys",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="tappy_features.csv (session/month-level) or similar")
    ap.add_argument("--out", default="tappy_web_model.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["label", "UserKey"]).copy()
    y = df["label"].astype(int)
    groups = df["UserKey"]

    missing = [c for c in WEB_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(
            "Missing required columns for web demo model:\n"
            f"{missing}\n\n"
            "Tip: regenerate tappy_features.csv using your feature builder so it contains these columns."
        )

    X = df[WEB_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced"))
        ]),
        "rf": RandomForestClassifier(
            n_estimators=800,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1
        )
    }

    results = {}
    for name, model in models.items():
        probs_all, ys_all = [], []
        for tr, te in cv.split(X, y, groups=groups):
            model.fit(X.iloc[tr], y.iloc[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            probs_all.append(p)
            ys_all.append(y.iloc[te].to_numpy())

        probs_all = np.concatenate(probs_all)
        ys_all = np.concatenate(ys_all)

        auc = roc_auc_score(ys_all, probs_all)
        bal = balanced_accuracy_score(ys_all, (probs_all >= 0.5).astype(int))
        results[name] = (auc, bal)
        print(f"{name}: AUC={auc:.3f}  BalancedAcc@0.5={bal:.3f}")

    best = max(results.items(), key=lambda kv: kv[1][0])[0]
    best_model = models[best]
    best_model.fit(X, y)

    joblib.dump({"model": best_model, "features": WEB_FEATURES}, args.out)
    print(f"Saved {args.out} (best={best})")


if __name__ == "__main__":
    main()
