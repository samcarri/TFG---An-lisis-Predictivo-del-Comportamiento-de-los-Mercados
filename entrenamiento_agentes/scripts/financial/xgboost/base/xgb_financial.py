"""
xgb_financial.py

XGBoost base con datos puramente financieros (Yahoo Finance).
Features: indicadores técnicos derivados de OHLCV.
Sin noticias, sin Reddit.

Output:
  scripts/financial/outputs/xgboost/xgb_financial_results.csv
  scripts/financial/outputs/xgboost/xgb_financial_plot.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

DATA_PATH   = "data/dataset_nvda_financial.csv"
OUTPUT_CSV  = "scripts/financial/outputs/xgboost/xgb_financial_results.csv"
OUTPUT_PLOT = "scripts/financial/outputs/xgboost/xgb_financial_plot.png"
STEP        = 10   # reentrenar cada N días en walk-forward

FINANCIAL_FEATURES = [
    "Close", "High", "Low", "Volume",
    "log_return", "volatility_7d", "ema_7", "ema_14", "volume_change",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "bb_width", "bb_position",
    "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
    "log_return_lag_4", "log_return_lag_5",
    "momentum_5", "momentum_10",
]
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # CSV puramente financiero: usar todas las columnas excepto date y target
    available = [c for c in df.columns if c not in {"date", "target"}]
    X = df[available].values
    y = df["target"].values
    print(f"Dataset: {df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Features financieras: {len(available)}")
    print(f"Target: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, df["date"], available


def walk_forward(X, y, model_params, threshold=0.50, min_train=None, step=STEP):
    """Reentrenar cada `step` días, predice t+1. Test definitivo sin leakage."""
    if min_train is None:
        min_train = int(len(X) * 0.70)
    print(f"\n{'='*60}\nWALK-FORWARD  min_train={min_train}  step={step}  thr={threshold}\n{'='*60}")

    all_preds, all_true, all_probas = [], [], []
    model_cache, scaler_cache = None, None

    for t in range(min_train, len(X) - 1):
        if (t - min_train) % step == 0:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X[:t])
            m = xgb.XGBClassifier(**model_params)
            m.fit(X_tr_s, y[:t], verbose=False)
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
    rec = recall_score(all_true, all_preds, pos_label=1)
    pre = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(all_true, all_probas)
    baseline = accuracy_score(all_true, np.full_like(all_preds, int(np.bincount(all_true).argmax())))

    print(f"OOS: {len(all_preds)} días")
    print(f"Acc={acc:.4f}  AUC={auc:.4f}  Recall={rec:.4f}  Prec={pre:.4f}")
    print(f"Baseline={baseline:.4f}  Mejora={acc-baseline:+.4f}")
    print(classification_report(all_true, all_preds, target_names=["Baja", "Sube"]))

    return {"label": "walk_forward", "threshold": threshold,
            "accuracy": acc, "f1_weighted": f1, "recall_sube": rec,
            "precision_sube": pre, "roc_auc": auc,
            "wf_n_preds": len(all_preds), "wf_baseline": baseline}


def evaluate(model, X, y, threshold, label=""):
    proba  = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="weighted")
    rec = recall_score(y, y_pred, pos_label=1)
    pre = precision_score(y, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y, proba)
    print(f"[{label}] thr={threshold:.2f}  acc={acc:.4f}  auc={auc:.4f}  recall={rec:.4f}  prec={pre:.4f}")
    print(classification_report(y, y_pred, target_names=["Baja", "Sube"]))
    return {"label": label, "threshold": threshold, "accuracy": acc,
            "f1_weighted": f1, "recall_sube": rec, "precision_sube": pre, "roc_auc": auc}


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV),  exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    print("=" * 60)
    print("XGBOOST BASE — DATOS FINANCIEROS")
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

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=round(n_neg / n_pos, 2),
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        early_stopping_rounds=30,
    )
    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    print(f"Best iteration: {model.best_iteration}")

    print(f"\n{'='*60}\nEVALUACIÓN TEST\n{'='*60}")
    res_base = evaluate(model, X_test_s, y_test, 0.50, "baseline_0.50")
    res_tr   = evaluate(model, X_train_s, y_train, 0.50, "train")
    print(f"Gap acc: {res_tr['accuracy'] - res_base['accuracy']:.4f}  Gap auc: {res_tr['roc_auc'] - res_base['roc_auc']:.4f}")

    # Importancia
    imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\n=== TOP 10 FEATURES ===")
    for _, row in imp.head(10).iterrows():
        bar = "█" * int(row["importance"] * 500)
        print(f"  {row['feature']:25s} {row['importance']:.5f}  {bar}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(range(len(imp)), imp["importance"], color="#3498db", edgecolor="white")
    axes[0].set_yticks(range(len(imp))); axes[0].set_yticklabels(imp["feature"], fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_xlabel("Importancia (gain)")
    axes[0].set_title("XGBoost Financial — Feature Importance", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    proba_test = model.predict_proba(X_test_s)[:, 1]
    axes[1].hist(proba_test[y_test == 0], bins=25, alpha=0.6, label="Baja", color="#e74c3c")
    axes[1].hist(proba_test[y_test == 1], bins=25, alpha=0.6, label="Sube", color="#2ecc71")
    axes[1].axvline(0.5, color="black", linestyle="--", lw=2, label="thr=0.50")
    axes[1].set_title("Distribución probabilidades (test)", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nGráfico: {OUTPUT_PLOT}")

    results = pd.DataFrame([res_base, res_tr])
    # Walk-forward
    wf_params = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
        "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "scale_pos_weight": round((y_train == 0).sum() / (y_train == 1).sum(), 2),
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
    }
    wf = walk_forward(X, y, wf_params)
    results = pd.DataFrame([res_base, res_tr, wf])
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV:     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
