import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ==============================
# 1. LOAD SEQUENCE DATA
# ==============================
SEQ_PATH = "."

X = np.load(f"{SEQ_PATH}/X_sequences.npy")       # (N, 200, 3)
y = np.load(f"{SEQ_PATH}/y_labels.npy")           # (N,)
groups = np.load(f"{SEQ_PATH}/groups_subject.npy", allow_pickle=True)  # (N,)

print("\n===== DATA SUMMARY =====")
print("X shape:", X.shape)
print("y distribution:", dict(zip(*np.unique(y, return_counts=True))))

dataset_per_window = np.load(f"{SEQ_PATH}/dataset_sequences.npy", allow_pickle=True)
print("Dataset distribution:", dict(zip(*np.unique(dataset_per_window, return_counts=True))))


# ==============================
# 2. BUILD CNN MODEL
# ==============================
def build_cnn(input_shape=(200, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==============================
# 3. SUBJECT-LEVEL EVALUATION
# ==============================
def subject_level_metrics(y_true_windows, y_score_windows, group_ids, min_windows=10):
    results = pd.DataFrame({
        "group_id": group_ids,
        "y_true":   y_true_windows,
        "y_score":  y_score_windows
    })

    subj = results.groupby("group_id").agg(
        y_true=("y_true",  "max"),
        y_score=("y_score", "mean"),
        n_windows=("y_score", "count")
    ).reset_index()

    # Drop subjects with too few windows
    dropped = subj[subj["n_windows"] < min_windows]
    if len(dropped) > 0:
        print(f"  [eval] Dropping {len(dropped)} subject(s) with <{min_windows} windows")
    subj = subj[subj["n_windows"] >= min_windows]

    # Warn if any subject has mixed labels (data bug)
    label_check = results.groupby("group_id")["y_true"].nunique()
    inconsistent = label_check[label_check > 1]
    if len(inconsistent) > 0:
        print(f"  [eval] WARNING: {len(inconsistent)} subject(s) have mixed labels: {list(inconsistent.index)}")

    if subj["y_true"].nunique() < 2:
        return None, None, None

    y_pred = (subj["y_score"] >= 0.5).astype(int)
    acc = accuracy_score(subj["y_true"], y_pred)
    f1  = f1_score(subj["y_true"], y_pred, zero_division=0)
    auc = roc_auc_score(subj["y_true"], subj["y_score"])
    return acc, f1, auc


# ==============================
# 4. CROSS-VAL RUNNER
# ==============================
def run_cnn_cv(X_data, y_data, groups_data, title, epochs=20, batch_size=64):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=5)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_data, y_data, groups_data)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        test_groups = groups_data[test_idx]

        # Class weights to handle imbalance
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        class_weight = {0: 1.0, 1: n_neg / n_pos}

        model = build_cnn()
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0
        )

        y_prob = model.predict(X_test, verbose=0).flatten()
        acc, f1, auc = subject_level_metrics(y_test, y_prob, test_groups, min_windows=3)

        if acc is None:
            print(f"\nFold {fold+1}  |  Skipped (single class in test)")
            continue

        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

        n_subjects = len(np.unique(test_groups))
        print(f"\nFold {fold+1}  |  Test subjects: {n_subjects}")
        print(f"Subject-level  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    print("\n----- FINAL AVERAGE -----")
    print(f"Mean Accuracy : {np.mean(accs):.4f}")
    print(f"Mean F1-score : {np.mean(f1s):.4f}")
    print(f"Mean ROC-AUC  : {np.mean(aucs):.4f}")
    return np.mean(accs), np.mean(f1s), np.mean(aucs)


# ==============================
# EXPERIMENT 1 — MIT ONLY
# ==============================
mit_mask = dataset_per_window != "tappy"
run_cnn_cv(X[mit_mask], y[mit_mask], groups[mit_mask], "Exp 1 MIT-ONLY (CNN)")


# ==============================
# EXPERIMENT 2 — TAPPY ONLY
# ==============================
tappy_mask = dataset_per_window == "tappy"
run_cnn_cv(X[tappy_mask], y[tappy_mask], groups[tappy_mask], "Exp 2 TAPPY-ONLY (CNN)")


# ==============================
# EXPERIMENT 3 — CROSS-DATASET (Train: Tappy → Test: MIT)
# ==============================
print(f"\n{'='*60}")
print("EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)")
print(f"{'='*60}")

X_train_c = X[tappy_mask]
y_train_c = y[tappy_mask]
X_test_c  = X[mit_mask]
y_test_c  = y[mit_mask]
g_test_c  = groups[mit_mask]

n_neg = np.sum(y_train_c == 0)
n_pos = np.sum(y_train_c == 1)
cw = {0: 1.0, 1: n_neg / n_pos}

cross_model = build_cnn()
cross_model.fit(X_train_c, y_train_c, epochs=20, batch_size=64,
                class_weight=cw, verbose=0)

y_prob_c = cross_model.predict(X_test_c, verbose=0).flatten()
acc, f1, auc = subject_level_metrics(y_test_c, y_prob_c, g_test_c, min_windows=3)
print(f"\nExp 3 CROSS-DATASET — CNN")
print(f"Subject-level  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")


# ==============================
# EXPERIMENT 4 — BALANCED POOLED
# ==============================
mit_idx    = np.where(mit_mask)[0]
tappy_idx  = np.where(tappy_mask)[0]
sample_idx = np.random.RandomState(42).choice(tappy_idx, size=len(mit_idx), replace=False)
bal_idx    = np.concatenate([mit_idx, sample_idx])

run_cnn_cv(X[bal_idx], y[bal_idx], groups[bal_idx], "Exp 4 BALANCED POOLED (CNN)")
