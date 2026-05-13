"""
correlation_analysis.py

Genera la matriz de correlación entre las predicciones (probabilidades)
de los tres modelos base del ensemble: RF_News, LightGBM_News, XGBoost_Reddit.

Objetivo: Verificar que los modelos cometen errores en momentos distintos.
Si la correlación entre pares es > 0.8, el Soft Voting es redundante.

Outputs:
  outputs/voting_ensemble/correlation_matrix.png   — Heatmap
  outputs/voting_ensemble/correlation_report.txt   — Reporte numérico
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir el directorio padre para importar voting_ensemble
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voting_ensemble import (
    MODEL_CONFIGS, THRESHOLD, load_and_split, train_base_model
)

# ── Configuración de salida ───────────────────────────────────────────────────

OUTPUT_DIR    = "outputs/voting_ensemble"
OUTPUT_PLOT   = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "correlation_report.txt")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ANÁLISIS DE CORRELACIÓN — PREDICCIONES MODELOS BASE")
    print("=" * 60)

    # ── 1. Cargar y dividir datasets ──────────────────────────────────────────
    loaded = []
    for cfg in MODEL_CONFIGS:
        print(f"\n--- {cfg['name']} ({cfg['data']}) ---")
        splits = load_and_split(cfg["data"])
        loaded.append(splits)

    # ── 2. Entrenar/cargar modelos base ───────────────────────────────────────
    trained = []
    for cfg, splits in zip(MODEL_CONFIGS, loaded):
        X_train, X_val, X_test, y_train, y_val, y_test, dates, features = splits
        print(f"\n--- Cargando/Entrenando {cfg['name']} ---")
        try:
            model, scaler, X_train_s, X_val_s = train_base_model(
                cfg["type"], X_train, y_train, X_val, y_val,
                model_path=cfg.get("model"), scaler_path=cfg.get("scaler"),
            )
            # Verificar que predict_proba funciona
            scaler.transform(X_test[:1])
            model.predict_proba(scaler.transform(X_test[:1]))
        except Exception as e:
            print(f"  ⚠️  Error cargando modelo ({e}). Entrenando desde cero...")
            model, scaler, X_train_s, X_val_s = train_base_model(
                cfg["type"], X_train, y_train, X_val, y_val,
                model_path=None, scaler_path=None,
            )
        X_test_s = scaler.transform(X_test)
        trained.append({
            "cfg":      cfg,
            "model":    model,
            "scaler":   scaler,
            "X_test_s": X_test_s,
            "y_test":   y_test,
            "dates":    dates,
        })

    # ── 3. Alinear predicciones por fechas comunes en test ────────────────────
    test_date_sets = []
    for t in trained:
        n       = len(t["dates"])
        val_end = int(np.floor(n * 0.85))
        test_dates = t["dates"].iloc[val_end:].dt.date
        test_date_sets.append(set(test_dates))

    common_test_dates = sorted(test_date_sets[0].intersection(*test_date_sets[1:]))
    print(f"\nFechas comunes en test: {common_test_dates[0]} → {common_test_dates[-1]}  ({len(common_test_dates)} días)")

    # Obtener probabilidades alineadas
    model_names = []
    aligned_probas = []

    for i, t in enumerate(trained):
        n       = len(t["dates"])
        val_end = int(np.floor(n * 0.85))
        test_dates_series = t["dates"].iloc[val_end:].dt.date.reset_index(drop=True)
        mask = test_dates_series.isin(set(common_test_dates))

        proba_test = t["model"].predict_proba(t["X_test_s"])[:, 1]
        aligned_probas.append(proba_test[mask.values])
        model_names.append(t["cfg"]["name"])

    # ── 4. Calcular matriz de correlación ─────────────────────────────────────
    # Correlación entre probabilidades continuas (Pearson)
    df_probas = pd.DataFrame(
        {name: proba for name, proba in zip(model_names, aligned_probas)}
    )
    corr_probas = df_probas.corr(method="pearson")

    # Correlación entre predicciones binarias (Pearson sobre 0/1)
    df_preds = pd.DataFrame(
        {name: (proba >= THRESHOLD).astype(int) for name, proba in zip(model_names, aligned_probas)}
    )
    corr_preds = df_preds.corr(method="pearson")

    print(f"\n{'='*60}")
    print("MATRIZ DE CORRELACIÓN — PROBABILIDADES (Pearson)")
    print("=" * 60)
    print(corr_probas.to_string(float_format="{:.4f}".format))

    print(f"\n{'='*60}")
    print("MATRIZ DE CORRELACIÓN — PREDICCIONES BINARIAS (Pearson)")
    print("=" * 60)
    print(corr_preds.to_string(float_format="{:.4f}".format))

    # ── 5. Evaluar redundancia ────────────────────────────────────────────────
    # Extraer pares únicos (sin diagonal)
    pairs = []
    n_models = len(model_names)
    max_corr_proba = 0.0
    max_corr_pred  = 0.0

    for i in range(n_models):
        for j in range(i + 1, n_models):
            cp = corr_probas.iloc[i, j]
            cb = corr_preds.iloc[i, j]
            pairs.append((model_names[i], model_names[j], cp, cb))
            max_corr_proba = max(max_corr_proba, cp)
            max_corr_pred  = max(max_corr_pred, cb)

    print(f"\n{'='*60}")
    print("ANÁLISIS DE DIVERSIDAD DEL ENSEMBLE")
    print("=" * 60)
    for m1, m2, cp, cb in pairs:
        print(f"  {m1} vs {m2}:")
        print(f"    Corr. probabilidades: {cp:.4f}")
        print(f"    Corr. predicciones:   {cb:.4f}")
        status = "⚠️  ALTA (>0.8) — posible redundancia" if cp > 0.8 else "✅ ACEPTABLE (≤0.8) — diversidad suficiente"
        print(f"    Estado: {status}")
        print()

    is_redundant = max_corr_proba > 0.8


    # ── 6. Generar heatmap ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Matriz de Correlación entre Modelos Base del Ensemble",
                 fontsize=13, fontweight="bold")

    # Heatmap de probabilidades
    sns.heatmap(corr_probas, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=1, square=True, linewidths=1,
                cbar_kws={"label": "Correlación Pearson"},
                ax=axes[0])
    axes[0].set_title("Correlación de Probabilidades", fontweight="bold")

    # Heatmap de predicciones binarias
    sns.heatmap(corr_preds, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=1, square=True, linewidths=1,
                cbar_kws={"label": "Correlación Pearson"},
                ax=axes[1])
    axes[1].set_title("Correlación de Predicciones Binarias", fontweight="bold")

    # Línea de referencia 0.8 en la barra de color (anotación)
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap guardado: {OUTPUT_PLOT}")

    # ── 7. Guardar reporte de texto ──────────────────────────────────────────
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE CORRELACIÓN — MODELOS BASE DEL ENSEMBLE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Modelos: {', '.join(model_names)}\n")
        f.write(f"Fechas de test: {common_test_dates[0]} → {common_test_dates[-1]}\n")
        f.write(f"Días evaluados: {len(common_test_dates)}\n\n")

        f.write("CORRELACIÓN DE PROBABILIDADES (Pearson):\n")
        f.write(corr_probas.to_string(float_format="{:.4f}".format) + "\n\n")

        f.write("CORRELACIÓN DE PREDICCIONES BINARIAS (Pearson):\n")
        f.write(corr_preds.to_string(float_format="{:.4f}".format) + "\n\n")

        f.write("ANÁLISIS POR PARES:\n")
        for m1, m2, cp, cb in pairs:
            f.write(f"  {m1} vs {m2}: corr_proba={cp:.4f}, corr_pred={cb:.4f}\n")
        f.write(f"\nCorrelación máxima (probabilidades): {max_corr_proba:.4f}\n")
        f.write(f"Umbral de redundancia: 0.80\n\n")
     

    print(f"Reporte guardado: {OUTPUT_REPORT}")
    print(f"\n{'='*60}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
