"""
lstm_financial.py

LSTM base con datos puramente financieros (Yahoo Finance).
Sin noticias, sin Reddit.
Arquitectura conservadora: 2 capas LSTM pequeñas + Dropout.

Output:
  scripts/financial/outputs/LSTM/lstm_financial_results.csv
  scripts/financial/outputs/LSTM/lstm_financial_plot.png
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score
)

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH   = "data/dataset_nvda_financial.csv"
OUTPUT_CSV  = "scripts/financial/outputs/LSTM/lstm_financial_results.csv"
OUTPUT_PLOT = "scripts/financial/outputs/LSTM/lstm_financial_plot.png"

FINANCIAL_FEATURES = [
    "Close", "High", "Low", "Volume",
    "log_return", "volatility_7d", "ema_7", "ema_14", "volume_change",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "bb_width", "bb_position",
    "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
    "log_return_lag_4", "log_return_lag_5",
    "momentum_5", "momentum_10",
]
TIME_STEPS   = 15
EPOCHS       = 100
BATCH_SIZE   = 32
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    available = [c for c in df.columns if c not in {"date", "target"}]
    X = df[available].values
    y = df["target"].values
    print(f"Dataset: {df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Features financieras: {len(available)}")
    print(f"Target: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, df["date"], available


def make_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def build_model(n_features, time_steps):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_steps, n_features),
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),
        LSTM(16, return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(8, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=["accuracy"],
    )
    return model


def evaluate(model, X_seq, y_seq, threshold, label=""):
    proba  = model.predict(X_seq, verbose=0).flatten()
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_seq, y_pred)
    f1  = f1_score(y_seq, y_pred, average="weighted")
    rec = recall_score(y_seq, y_pred, pos_label=1)
    pre = precision_score(y_seq, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_seq, proba)
    print(f"[{label}] thr={threshold:.2f}  acc={acc:.4f}  auc={auc:.4f}  recall={rec:.4f}  prec={pre:.4f}")
    print(classification_report(y_seq, y_pred, target_names=["Baja", "Sube"]))
    return {"label": label, "threshold": threshold, "accuracy": acc,
            "f1_weighted": f1, "recall_sube": rec, "precision_sube": pre, "roc_auc": auc}


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV),  exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    print("=" * 60)
    print(f"LSTM BASE — DATOS FINANCIEROS  (time_steps={TIME_STEPS})")
    print("=" * 60)

    X, y, dates, features = load_data(DATA_PATH)

    n         = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"\nTrain: {dates.iloc[0].date()} → {dates.iloc[train_end-1].date()}")
    print(f"Val:   {dates.iloc[train_end].date()} → {dates.iloc[val_end-1].date()}")
    print(f"Test:  {dates.iloc[val_end].date()} → {dates.iloc[-1].date()}")

    X_tr_seq, y_tr_seq = make_sequences(X_train_s, y_train, TIME_STEPS)
    X_vl_seq, y_vl_seq = make_sequences(X_val_s,   y_val,   TIME_STEPS)
    X_te_seq, y_te_seq = make_sequences(X_test_s,  y_test,  TIME_STEPS)

    print(f"\nSecuencias train: {X_tr_seq.shape}")
    print(f"Secuencias val:   {X_vl_seq.shape}")
    print(f"Secuencias test:  {X_te_seq.shape}")

    n_neg = (y_tr_seq == 0).sum()
    n_pos = (y_tr_seq == 1).sum()
    class_weight = {0: 1.0, 1: round(n_neg / n_pos, 2)}
    print(f"Class weights: {class_weight}")

    model = build_model(X_tr_seq.shape[2], TIME_STEPS)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
    ]

    print(f"\nEntrenando ({EPOCHS} epochs máx)...")
    history = model.fit(
        X_tr_seq, y_tr_seq,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_vl_seq, y_vl_seq),
        callbacks=callbacks, class_weight=class_weight, verbose=1,
    )
    print(f"Épocas entrenadas: {len(history.history['loss'])}")

    print(f"\n{'='*60}\nEVALUACIÓN TEST\n{'='*60}")
    res_base = evaluate(model, X_te_seq, y_te_seq, 0.50, "baseline_0.50")
    res_tr   = evaluate(model, X_tr_seq, y_tr_seq, 0.50, "train")
    print(f"Gap acc: {res_tr['accuracy'] - res_base['accuracy']:.4f}  Gap auc: {res_tr['roc_auc'] - res_base['roc_auc']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"],     label="Train", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val",   linewidth=2)
    axes[0].set_title("Loss durante entrenamiento", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    proba_test = model.predict(X_te_seq, verbose=0).flatten()
    axes[1].hist(proba_test[y_te_seq == 0], bins=25, alpha=0.6, label="Baja", color="#e74c3c")
    axes[1].hist(proba_test[y_te_seq == 1], bins=25, alpha=0.6, label="Sube", color="#2ecc71")
    axes[1].axvline(0.5, color="black", linestyle="--", lw=2)
    axes[1].set_title("Distribución probabilidades (test)", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nGráfico: {OUTPUT_PLOT}")

    results = pd.DataFrame([res_base, res_tr])
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV:     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
