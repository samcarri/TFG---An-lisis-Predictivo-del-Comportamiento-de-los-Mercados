"""
rf_weekly.py

Random Forest para predicción a 1 semana (5 días hábiles) con datos financieros.

Output:
  scripts/financial/outputs/weekly/rf_weekly_results.csv
  scripts/financial/outputs/weekly/rf_weekly_plot.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

DATA_PATH   = "data/dataset_nvda_financial.csv"
OUTPUT_CSV  = "scripts/financial/outputs/weekly/rf_weekly_results.csv"
OUTPUT_PLOT = "scripts/financial/outputs/weekly/rf_weekly_plot.png"
MODEL_PATH  = "trained_models/rf_weekly_model.joblib"
SCALER_PATH = "trained_models/rf_weekly_scaler.joblib"
PREDICTION_HORIZON = 5
STEP        = 10
RANDOM_STATE = 42


def load_data(path, horizon=5):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    df["future_close"] = df["Close"].shift(-horizon)
    df["target_weekly"] = (df["future_close"] > df["Close"]).astype(int)
    df = df[:-horizon].copy()
    
    available = [c for c in df.columns if c not in {"date", "target", "target_weekly", "future_close"}]
    X = df[available].values
    y = df["target_weekly"].values
    
    print(f"Dataset: {df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Horizonte: {horizon} días | Features: {len(available)}")
    print(f"Target: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, df["date"], available


def walk_forward(X, y, model_params, threshold=0.50, min_train=None, step=STEP):
    if min_train is None:
        min_train = int(len(X) * 0.70)
    print(f"\n{'='*60}\nWALK-FORWARD  min_train={min_train}  step={step}  thr={threshold}\n{'='*60}")

    all_preds, all_true, all_probas = [], [], []
    model_cache, scaler_cache = None, None

    for t in range(min_train, len(X) - 1):
        if (t - min_train) % step == 0:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X[:t])
            m = RandomForestClassifier(**model_params)
            m.fit(X_tr_s, y[:t])
            model_cache, scaler_cache = m, sc

        proba = model_cache.predict_proba(scaler_cache.transform(X[t:t+1]))[0, 1]
        all_probas.append(proba)
        all_preds.append(int(proba >= threshold))
        all_true.append(y[t])

    all_preds  = np.array(all_preds)
    all_true   = np.array(all_true)
    all_probas = np.array(all_probas)

    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="weighted")
    rec_sube = recall_score(all_true, all_preds, pos_label=1)
    pre_sube = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    rec_baja = recall_score(all_true, all_preds, pos_label=0)
    pre_baja = precision_score(all_true, all_preds, pos_label=0, zero_division=0)
    auc = roc_auc_score(all_true, all_probas)
    baseline = accuracy_score(all_true, np.full_like(all_preds, int(np.bincount(all_true).argmax())))

    print(f"OOS: {len(all_preds)} días")
    print(f"Acc={acc:.4f}  AUC={auc:.4f}  Recall_Sube={rec_sube:.4f}  Prec_Sube={pre_sube:.4f}")
    print(f"Recall_Baja={rec_baja:.4f}  Prec_Baja={pre_baja:.4f}")
    print(f"Baseline={baseline:.4f}  Mejora={acc-baseline:+.4f}")
    print(classification_report(all_true, all_preds, target_names=["Baja", "Sube"]))

    return {"label": "walk_forward", "threshold": threshold,
            "accuracy": acc, "f1_weighted": f1, "recall_sube": rec_sube,
            "precision_sube": pre_sube, "recall_baja": rec_baja, "precision_baja": pre_baja,
            "roc_auc": auc, "wf_n_preds": len(all_preds), "wf_baseline": baseline,
            "horizon_days": PREDICTION_HORIZON}


def evaluate(model, X, y, threshold, label=""):
    proba  = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="weighted")
    rec_sube = recall_score(y, y_pred, pos_label=1)
    pre_sube = precision_score(y, y_pred, pos_label=1, zero_division=0)
    rec_baja = recall_score(y, y_pred, pos_label=0)
    pre_baja = precision_score(y, y_pred, pos_label=0, zero_division=0)
    auc = roc_auc_score(y, proba)
    print(f"[{label}] thr={threshold:.2f}  acc={acc:.4f}  auc={auc:.4f}")
    print(f"  Sube: recall={rec_sube:.4f}  prec={pre_sube:.4f}")
    print(f"  Baja: recall={rec_baja:.4f}  prec={pre_baja:.4f}")
    print(classification_report(y, y_pred, target_names=["Baja", "Sube"]))
    return {"label": label, "threshold": threshold, "accuracy": acc,
            "f1_weighted": f1, "recall_sube": rec_sube, "precision_sube": pre_sube, "recall_baja": rec_baja, "precision_baja": pre_baja, "roc_auc": auc,
            "horizon_days": PREDICTION_HORIZON}


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV),  exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    print("=" * 60)
    print(f"RANDOM FOREST WEEKLY — PREDICCIÓN A {PREDICTION_HORIZON} DÍAS")
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

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_split=20,
        min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    print(f"\n{'='*60}\nEVALUACIÓN TEST\n{'='*60}")
    res_base = evaluate(model, X_test_s, y_test, 0.50, "baseline_0.50")
    res_tr   = evaluate(model, X_train_s, y_train, 0.50, "train")
    print(f"Gap acc: {res_tr['accuracy'] - res_base['accuracy']:.4f}  Gap auc: {res_tr['roc_auc'] - res_base['roc_auc']:.4f}")

    imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\n=== TOP 10 FEATURES ===")
    for _, row in imp.head(10).iterrows():
        bar = "█" * int(row["importance"] * 500)
        print(f"  {row['feature']:25s} {row['importance']:.5f}  {bar}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(range(len(imp)), imp["importance"], color="#2ecc71", edgecolor="white")
    axes[0].set_yticks(range(len(imp))); axes[0].set_yticklabels(imp["feature"], fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_xlabel("Importancia (Gini)")
    axes[0].set_title(f"Random Forest Weekly ({PREDICTION_HORIZON}d) — Feature Importance", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    proba_test = model.predict_proba(X_test_s)[:, 1]
    axes[1].hist(proba_test[y_test == 0], bins=25, alpha=0.6, label="Baja", color="#e74c3c")
    axes[1].hist(proba_test[y_test == 1], bins=25, alpha=0.6, label="Sube", color="#2ecc71")
    axes[1].axvline(0.5, color="black", linestyle="--", lw=2)
    axes[1].set_title("Distribución probabilidades (test)", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nGráfico: {OUTPUT_PLOT}")

    wf_params = {
        "n_estimators": 300, "max_depth": 6, "min_samples_split": 20,
        "min_samples_leaf": 10, "max_features": "sqrt",
        "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1,
    }
    wf = walk_forward(X, y, wf_params)
    results = pd.DataFrame([res_base, res_tr, wf])
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV:     {OUTPUT_CSV}")

    # Guardar modelo y scaler entrenados con todos los datos
    print(f"\n{'='*60}\nGUARDANDO MODELO FINAL\n{'='*60}")
    scaler_final = StandardScaler()
    X_all_scaled = scaler_final.fit_transform(X)
    
    model_final = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_split=20,
        min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model_final.fit(X_all_scaled, y)
    
    joblib.dump(model_final, MODEL_PATH)
    joblib.dump(scaler_final, SCALER_PATH)
    print(f"Modelo guardado: {MODEL_PATH}")
    print(f"Scaler guardado: {SCALER_PATH}")
    
    # Métricas detalladas por clase
    y_pred_final = model_final.predict(X_all_scaled)
    print(f"\n{'='*60}\nMÉTRICAS DETALLADAS POR CLASE (Datos completos)\n{'='*60}")
    print(classification_report(y, y_pred_final, target_names=["Baja", "Sube"], digits=4))


if __name__ == "__main__":
    main()
