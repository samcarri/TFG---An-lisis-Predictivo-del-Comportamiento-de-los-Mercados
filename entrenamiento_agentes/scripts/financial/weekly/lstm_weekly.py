"""
lstm_weekly.py

LSTM para predicción a 1 semana (5 días hábiles) con datos financieros.
Predice si el precio subirá en los próximos 5 días.

Output:
  scripts/financial/outputs/weekly/lstm_weekly_results.csv
  scripts/financial/outputs/weekly/lstm_weekly_plot.png
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
import joblib
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
OUTPUT_CSV  = "scripts/financial/outputs/weekly/lstm_weekly_results.csv"
OUTPUT_PLOT = "scripts/financial/outputs/weekly/lstm_weekly_plot.png"

PREDICTION_HORIZON = 5  # 1 semana = 5 días hábiles
TIME_STEPS   = 15
EPOCHS       = 100
BATCH_SIZE   = 32
RANDOM_STATE = 42


def load_data(path, horizon=5):
    """Carga datos y crea target para predicción a N días"""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Crear target: ¿sube en los próximos N días?
    df["future_close"] = df["Close"].shift(-horizon)
    df["target_weekly"] = (df["future_close"] > df["Close"]).astype(int)
    
    # Eliminar últimas N filas (sin target válido)
    df = df[:-horizon].copy()
    
    available = [c for c in df.columns if c not in {"date", "target", "target_weekly", "future_close"}]
    X = df[available].values
    y = df["target_weekly"].values
    
    print(f"Dataset: {df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Horizonte predicción: {horizon} días")
    print(f"Features financieras: {len(available)}")
    print(f"Target distribución: {dict(zip(*np.unique(y, return_counts=True)))}")
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
            "f1_weighted": f1, "recall_sube": rec, "precision_sube": pre, "roc_auc": auc,
            "horizon_days": PREDICTION_HORIZON}


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV),  exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    print("=" * 60)
    print(f"LSTM WEEKLY — PREDICCIÓN A {PREDICTION_HORIZON} DÍAS (time_steps={TIME_STEPS})")
    print("=" * 60)

    X, y, dates, features = load_data(DATA_PATH, horizon=PREDICTION_HORIZON)

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
    axes[0].set_title(f"Loss - Predicción {PREDICTION_HORIZON} días", fontweight="bold")
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

    # ── Walk-Forward ──────────────────────────────────────────────
    print(f"\n{'='*60}\nWALK-FORWARD  min_train=70%  step=10\n{'='*60}")
    min_train = int(len(X) * 0.70)
    step = 10
    all_preds, all_true, all_probas = [], [], []
    model_cache, scaler_cache = None, None

    for t in range(min_train, len(X) - TIME_STEPS - 1):
        if (t - min_train) % step == 0:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X[:t])
            X_tr_seq_wf, y_tr_seq_wf = make_sequences(X_tr_s, y[:t], TIME_STEPS)
            n_neg_wf = (y_tr_seq_wf == 0).sum()
            n_pos_wf = (y_tr_seq_wf == 1).sum()
            cw = {0: 1.0, 1: round(n_neg_wf / max(n_pos_wf, 1), 2)}
            m = build_model(X_tr_seq_wf.shape[2], TIME_STEPS)
            m.fit(X_tr_seq_wf, y_tr_seq_wf, epochs=50, batch_size=32,
                  callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
                  class_weight=cw, verbose=0)
            model_cache, scaler_cache = m, sc

        X_point = scaler_cache.transform(X[t - TIME_STEPS + 1:t + 1])
        X_seq_point = X_point.reshape(1, TIME_STEPS, -1)
        proba = float(model_cache.predict(X_seq_point, verbose=0).flatten()[0])
        all_probas.append(proba)
        all_preds.append(int(proba >= 0.50))
        all_true.append(y[t])

    all_preds  = np.array(all_preds)
    all_true   = np.array(all_true)
    all_probas = np.array(all_probas)

    wf_acc = accuracy_score(all_true, all_preds)
    wf_f1  = f1_score(all_true, all_preds, average="weighted")
    wf_rec = recall_score(all_true, all_preds, pos_label=1)
    wf_pre = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    wf_auc = roc_auc_score(all_true, all_probas)
    wf_baseline = accuracy_score(all_true, np.full_like(all_preds, int(np.bincount(all_true).argmax())))

    print(f"OOS: {len(all_preds)} predicciones")
    print(f"Acc={wf_acc:.4f}  AUC={wf_auc:.4f}  Recall={wf_rec:.4f}  Prec={wf_pre:.4f}")
    print(f"Baseline={wf_baseline:.4f}  Mejora={wf_acc - wf_baseline:+.4f}")
    print(classification_report(all_true, all_preds, target_names=["Baja", "Sube"]))

    wf_result = {"label": "walk_forward", "threshold": 0.50,
                 "accuracy": wf_acc, "f1_weighted": wf_f1, "recall_sube": wf_rec,
                 "precision_sube": wf_pre, "roc_auc": wf_auc,
                 "horizon_days": PREDICTION_HORIZON,
                 "wf_n_preds": len(all_preds), "wf_baseline": wf_baseline}

    results = pd.DataFrame([res_base, res_tr, wf_result])
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV:     {OUTPUT_CSV}")

    # Guardar modelo y scaler entrenados con todos los datos
    print(f"\n{'='*60}\nGUARDANDO MODELO FINAL\n{'='*60}")
    scaler_final = StandardScaler()
    X_all_scaled = scaler_final.fit_transform(X)
    
    X_all_seq, y_all_seq = make_sequences(X_all_scaled, y, TIME_STEPS)
    
    model_final = build_model(X_all_seq.shape[2], TIME_STEPS)
    
    n_neg_all = (y_all_seq == 0).sum()
    n_pos_all = (y_all_seq == 1).sum()
    class_weight_final = {0: 1.0, 1: round(n_neg_all / n_pos_all, 2)}
    
    model_final.fit(
        X_all_seq, y_all_seq,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)],
        class_weight=class_weight_final, verbose=0,
    )
    
    MODEL_PATH = "trained_models/lstm_weekly_model.h5"
    SCALER_PATH = "trained_models/lstm_weekly_scaler.joblib"
    
    os.makedirs("trained_models", exist_ok=True)
    model_final.save(MODEL_PATH)
    joblib.dump(scaler_final, SCALER_PATH)
    print(f"Modelo guardado: {MODEL_PATH}")
    print(f"Scaler guardado: {SCALER_PATH}")
    
    # Métricas detalladas por clase
    y_pred_proba = model_final.predict(X_all_seq, verbose=0).flatten()
    y_pred_final = (y_pred_proba >= 0.5).astype(int)
    print(f"\n{'='*60}\nMÉTRICAS DETALLADAS POR CLASE (Datos completos)\n{'='*60}")
    print(classification_report(y_all_seq, y_pred_final, target_names=["Baja", "Sube"], digits=4))


if __name__ == "__main__":
    main()
