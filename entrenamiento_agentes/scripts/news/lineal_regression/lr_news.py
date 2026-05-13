"""
lr_news.py

Regresión Logística base con datos financieros + features de noticias (GDELT).
Dataset: data/dataset_nvda_news.csv

Output:
  scripts/news/outputs/lineal_regression/lr_news_results.csv
  scripts/news/outputs/lineal_regression/lr_news_plot.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

DATA_PATH   = "data/dataset_nvda_news.csv"
OUTPUT_CSV  = "scripts/news/outputs/lineal_regression/lr_news_results.csv"
OUTPUT_PLOT = "scripts/news/outputs/lineal_regression/lr_news_plot.png"
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    available = [c for c in df.columns if c not in {"date", "target"}]
    X = df[available].values
    y = df["target"].values
    print(f"Dataset: {df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Features totales: {len(available)}")
    print(f"Target: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, df["date"], available


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
    print("REGRESIÓN LOGÍSTICA BASE — DATOS FINANCIEROS + NOTICIAS")
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

    model = LogisticRegression(
        C=0.1, penalty="l2", class_weight="balanced",
        solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train)

    print(f"\n{'='*60}\nEVALUACIÓN TEST\n{'='*60}")
    res_base = evaluate(model, X_test_s, y_test, 0.50, "baseline_0.50")
    res_tr   = evaluate(model, X_train_s, y_train, 0.50, "train")
    print(f"Gap acc: {res_tr['accuracy'] - res_base['accuracy']:.4f}  Gap auc: {res_tr['roc_auc'] - res_base['roc_auc']:.4f}")

    coef = np.abs(model.coef_[0])
    imp  = pd.DataFrame({"feature": features, "coef_abs": coef})
    imp  = imp.sort_values("coef_abs", ascending=False).reset_index(drop=True)
    print("\n=== TOP 10 COEFICIENTES (|coef|) ===")
    for _, row in imp.head(10).iterrows():
        bar = "█" * int(row["coef_abs"] / imp["coef_abs"].max() * 30)
        print(f"  {row['feature']:30s} {row['coef_abs']:.4f}  {bar}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(range(len(imp)), imp["coef_abs"], color="#e67e22", edgecolor="white")
    axes[0].set_yticks(range(len(imp))); axes[0].set_yticklabels(imp["feature"], fontsize=7)
    axes[0].invert_yaxis(); axes[0].set_xlabel("|Coeficiente|")
    axes[0].set_title("Regresión Logística News — Coeficientes", fontweight="bold")
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

    results = pd.DataFrame([res_base, res_tr])
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV:     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
