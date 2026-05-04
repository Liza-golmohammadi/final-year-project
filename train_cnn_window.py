import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


# ==============================
# 1. LOAD SEQUENCE DATA
# ==============================
SEQ_PATH = "."

X = np.load(f"{SEQ_PATH}/X_sequences.npy")
y = np.load(f"{SEQ_PATH}/y_labels.npy")
groups = np.load(f"{SEQ_PATH}/groups_subject.npy", allow_pickle=True)

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
# 3. WINDOW-LEVEL EVALUATION
# Each window is an independent test sample — no aggregation
# ==============================
def window_level_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        return None, None, None, None
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    return acc, bacc, f1, auc


# ==============================
# 4. CROSS-VAL RUNNER — WINDOW-LEVEL
# ==============================
def run_cnn_cv_window(X_data, y_data, groups_data, title, epochs=20, batch_size=64):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=5)
    accs, baccs, f1s, aucs = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_data, y_data, groups_data)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        test_groups = groups_data[test_idx]

        n_test_windows   = len(test_idx)
        n_test_subjects  = len(np.unique(test_groups))

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
        acc, bacc, f1, auc = window_level_metrics(y_test, y_prob)

        if acc is None:
            print(f"\nFold {fold+1}  |  Skipped (single class in test)")
            continue

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
        aucs.append(auc)

        print(f"\nFold {fold+1}  |  Test windows: {n_test_windows}  |  Test subjects: {n_test_subjects}")
        print(f"Window-level  Accuracy: {acc:.4f}  Balanced Acc: {bacc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    print("\n----- FINAL AVERAGE -----")
    print(f"Mean Accuracy         : {np.mean(accs):.4f}")
    print(f"Mean Balanced Accuracy: {np.mean(baccs):.4f}")
    print(f"Mean F1-score         : {np.mean(f1s):.4f}")
    print(f"Mean ROC-AUC          : {np.mean(aucs):.4f}")
    return np.mean(accs), np.mean(baccs), np.mean(f1s), np.mean(aucs)


# ==============================
# EXPERIMENT 1 — MIT ONLY
# ==============================
mit_mask = dataset_per_window != "tappy"
run_cnn_cv_window(X[mit_mask], y[mit_mask], groups[mit_mask], "Exp 1 MIT-ONLY (CNN) — Window-level")


# ==============================
# EXPERIMENT 2 — TAPPY ONLY
# ==============================
tappy_mask = dataset_per_window == "tappy"
run_cnn_cv_window(X[tappy_mask], y[tappy_mask], groups[tappy_mask], "Exp 2 TAPPY-ONLY (CNN) — Window-level")


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

n_neg = np.sum(y_train_c == 0)
n_pos = np.sum(y_train_c == 1)
cw = {0: 1.0, 1: n_neg / n_pos}

cross_model = build_cnn()
cross_model.fit(X_train_c, y_train_c, epochs=20, batch_size=64,
                class_weight=cw, verbose=0)

y_prob_c = cross_model.predict(X_test_c, verbose=0).flatten()
acc, bacc, f1, auc = window_level_metrics(y_test_c, y_prob_c)
print(f"\nExp 3 CROSS-DATASET — CNN (Window-level)")
print(f"Test windows: {len(y_test_c)}")
print(f"Window-level  Accuracy: {acc:.4f}  Balanced Acc: {bacc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")


# ==============================
# EXPERIMENT 4 — BALANCED POOLED
# ==============================
mit_idx    = np.where(mit_mask)[0]
tappy_idx  = np.where(tappy_mask)[0]
sample_idx = np.random.RandomState(42).choice(tappy_idx, size=len(mit_idx), replace=False)
bal_idx    = np.concatenate([mit_idx, sample_idx])

run_cnn_cv_window(X[bal_idx], y[bal_idx], groups[bal_idx], "Exp 4 BALANCED POOLED (CNN) — Window-level")
