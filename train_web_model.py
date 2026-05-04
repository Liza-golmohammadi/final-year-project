import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
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

# Hyperparameter search spaces
PARAM_DISTS = {
    "svm_rbf": {
        "clf__C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "rf": {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
    },
}

N_ITER_SEARCH = 20   # random combinations to try per model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="tappy_features.csv (session/month-level) or similar")
    ap.add_argument("--out", default="tappy_web_model.joblib")
    ap.add_argument("--n-iter", type=int, default=N_ITER_SEARCH,
                    help="Number of random hyperparameter combinations to try (default: 20)")
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

    # Inner CV for hyperparameter search (3-fold, group-safe)
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    # Outer CV for held-out evaluation (5-fold, group-safe)
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    base_estimators = {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]),
        "rf": RandomForestClassifier(
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    results = {}
    tuned_estimators = {}

    for name, base in base_estimators.items():
        print(f"\n[{name}] Searching {args.n_iter} hyperparameter combinations ...")
        search = RandomizedSearchCV(
            base,
            PARAM_DISTS[name],
            n_iter=args.n_iter,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            refit=True,
            verbose=0,
        )
        search.fit(X, y, groups=groups)
        print(f"[{name}] Best params: {search.best_params_}")
        print(f"[{name}] Inner CV AUC (mean): {search.best_score_:.3f}")

        # Outer evaluation using the best hyperparameters found above
        best_est = search.best_estimator_
        probs_all, ys_all = [], []
        for tr, te in outer_cv.split(X, y, groups=groups):
            best_est.fit(X.iloc[tr], y.iloc[tr])
            p = best_est.predict_proba(X.iloc[te])[:, 1]
            probs_all.append(p)
            ys_all.append(y.iloc[te].to_numpy())

        probs_all = np.concatenate(probs_all)
        ys_all = np.concatenate(ys_all)

        auc = roc_auc_score(ys_all, probs_all)
        bal = balanced_accuracy_score(ys_all, (probs_all >= 0.5).astype(int))
        results[name] = (auc, bal)
        tuned_estimators[name] = search.best_estimator_
        print(f"[{name}] Outer CV — AUC={auc:.3f}  BalancedAcc@0.5={bal:.3f}")

    best = max(results.items(), key=lambda kv: kv[1][0])[0]
    print(f"\nBest model: {best} (AUC={results[best][0]:.3f})")

    # Refit best model on all data before saving
    best_model = tuned_estimators[best]
    best_model.fit(X, y)

    joblib.dump({"model": best_model, "features": WEB_FEATURES}, args.out)
    print(f"Saved {args.out} (best={best})")


if __name__ == "__main__":
    main()
