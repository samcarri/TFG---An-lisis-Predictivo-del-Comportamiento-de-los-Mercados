"""
overfitting_analysis.py

Genera gráficas para analizar el desfase (gap) entre Train, Test y Walk-Forward
del Voting Ensemble. Visualiza:
  1. Comparativa de métricas Train vs Test vs Walk-Forward (barras agrupadas)
  2. Gap de sobreajuste por métrica
  3. Tabla resumen con todas las métricas

Outputs:
  outputs/voting_ensemble/overfitting_gap_analysis.png
  outputs/voting_ensemble/train_test_wf_comparison.png
  outputs/voting_ensemble/metrics_summary_table.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Configuración ─────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs/voting_ensemble"
RESULTS_CSV = "scripts/ensemble/outputs/voting_ensemble_results.csv"

# Métricas del ensemble (de voting_ensemble_results.csv)
METRICS = {
    "Train": {
        "Accuracy": 0.8871,
        "ROC-AUC": 0.9651,
        "F1 Weighted": 0.8884,
        "Recall (Sube)": 0.9667,
        "Precision (Sube)": 0.7923,
    },
    "Test": {
        "Accuracy": 0.6914,
        "ROC-AUC": 0.7628,
        "F1 Weighted": 0.6937,
        "Recall (Sube)": 0.5893,
        "Precision (Sube)": 0.5500,
    },
    "Walk-Forward": {
        "Accuracy": 0.6656,
        "ROC-AUC": 0.7144,
        "F1 Weighted": 0.6630,
        "Recall (Sube)": 0.5085,
        "Precision (Sube)": 0.5455,
    },
}

WF_BASELINE = 0.6347
WF_N_PREDS = 323


def plot_metrics_comparison():
    """Gráfico 1: Barras agrupadas Train vs Test vs Walk-Forward."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_names = list(METRICS["Train"].keys())
    x = np.arange(len(metrics_names))
    width = 0.25

    train_vals = [METRICS["Train"][m] for m in metrics_names]
    test_vals = [METRICS["Test"][m] for m in metrics_names]
    wf_vals = [METRICS["Walk-Forward"][m] for m in metrics_names]

    bars1 = ax.bar(x - width, train_vals, width, label="Train",
                   color="#2ecc71", edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x, test_vals, width, label="Test (estático)",
                   color="#3498db", edgecolor="white", alpha=0.9)
    bars3 = ax.bar(x + width, wf_vals, width, label="Walk-Forward",
                   color="#e74c3c", edgecolor="white", alpha=0.9)

    # Valores sobre las barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Métrica", fontsize=11)
    ax.set_ylabel("Valor", fontsize=11)
    ax.set_title("Voting Ensemble — Comparativa Train vs Test vs Walk-Forward",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.4, label="Baseline (0.50)")
    ax.axhline(WF_BASELINE, color="orange", linestyle=":", lw=1.5, alpha=0.7,
               label=f"WF Baseline ({WF_BASELINE:.4f})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Anotación del gap
    ax.annotate("Gap ROC-AUC\n0.2023",
                xy=(1, 0.96), xytext=(1.4, 1.02),
                fontsize=9, ha="center", color="#c0392b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "train_test_wf_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado: {path}")
    plt.close()


def plot_overfitting_gap():
    """Gráfico 2: Gap de sobreajuste (Train - Test) y (Train - WF)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Análisis del Gap de Sobreajuste — Voting Ensemble",
                 fontsize=13, fontweight="bold")

    metrics_names = list(METRICS["Train"].keys())
    x = np.arange(len(metrics_names))

    # Gap Train → Test
    gap_test = [METRICS["Train"][m] - METRICS["Test"][m] for m in metrics_names]
    colors_test = ["#e74c3c" if g > 0.20 else "#f39c12" if g > 0.15 else "#27ae60"
                   for g in gap_test]

    bars1 = axes[0].bar(x, gap_test, 0.6, color=colors_test, edgecolor="white", alpha=0.9)
    axes[0].set_title("Gap: Train → Test (estático)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Diferencia (Train - Test)")
    axes[0].axhline(0.20, color="red", linestyle="--", lw=1.2, alpha=0.7, label="Umbral 0.20")
    axes[0].axhline(0.15, color="orange", linestyle="--", lw=1, alpha=0.5, label="Umbral 0.15")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, gap_test):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Gap Train → Walk-Forward
    gap_wf = [METRICS["Train"][m] - METRICS["Walk-Forward"][m] for m in metrics_names]
    colors_wf = ["#e74c3c" if g > 0.25 else "#f39c12" if g > 0.20 else "#27ae60"
                 for g in gap_wf]

    bars2 = axes[1].bar(x, gap_wf, 0.6, color=colors_wf, edgecolor="white", alpha=0.9)
    axes[1].set_title("Gap: Train → Walk-Forward", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Diferencia (Train - WF)")
    axes[1].axhline(0.25, color="red", linestyle="--", lw=1.2, alpha=0.7, label="Umbral 0.25")
    axes[1].axhline(0.20, color="orange", linestyle="--", lw=1, alpha=0.5, label="Umbral 0.20")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, gap_wf):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "overfitting_gap_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado: {path}")
    plt.close()


def plot_metrics_table():
    """Gráfico 3: Tabla visual con todas las métricas y gaps."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    metrics_names = list(METRICS["Train"].keys())

    # Construir datos de la tabla
    col_labels = ["Métrica", "Train", "Test", "Walk-Forward",
                  "Gap (T→Test)", "Gap (T→WF)", "WF vs Baseline"]
    table_data = []

    for m in metrics_names:
        train_v = METRICS["Train"][m]
        test_v = METRICS["Test"][m]
        wf_v = METRICS["Walk-Forward"][m]
        gap_test = train_v - test_v
        gap_wf = train_v - wf_v

        if m == "Accuracy":
            wf_vs_bl = f"+{(wf_v - WF_BASELINE)*100:.2f}%"
        else:
            wf_vs_bl = "—"

        table_data.append([
            m,
            f"{train_v:.4f}",
            f"{test_v:.4f}",
            f"{wf_v:.4f}",
            f"{gap_test:.4f}",
            f"{gap_wf:.4f}",
            wf_vs_bl,
        ])

    # Añadir fila de resumen
    table_data.append([
        "N predicciones WF", "—", "162 días", f"{WF_N_PREDS} días",
        "—", "—", f"Baseline: {WF_BASELINE:.4f}"
    ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Estilizar cabecera
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colorear gaps
    for i in range(len(metrics_names)):
        # Gap Train→Test
        gap_test = METRICS["Train"][metrics_names[i]] - METRICS["Test"][metrics_names[i]]
        if gap_test > 0.20:
            table[i + 1, 4].set_facecolor("#fadbd8")
        elif gap_test > 0.15:
            table[i + 1, 4].set_facecolor("#fdebd0")
        else:
            table[i + 1, 4].set_facecolor("#d5f5e3")

        # Gap Train→WF
        gap_wf = METRICS["Train"][metrics_names[i]] - METRICS["Walk-Forward"][metrics_names[i]]
        if gap_wf > 0.25:
            table[i + 1, 5].set_facecolor("#fadbd8")
        elif gap_wf > 0.20:
            table[i + 1, 5].set_facecolor("#fdebd0")
        else:
            table[i + 1, 5].set_facecolor("#d5f5e3")

    # Última fila en gris
    for j in range(len(col_labels)):
        table[len(metrics_names) + 1, j].set_facecolor("#ecf0f1")

    ax.set_title("Resumen de Métricas — Voting Ensemble (Train / Test / Walk-Forward)",
                 fontsize=12, fontweight="bold", pad=20)

    # Leyenda de colores
    fig.text(0.15, 0.02, "🟢 Gap ≤ 0.15 (aceptable)", fontsize=8, color="#27ae60")
    fig.text(0.40, 0.02, "🟡 Gap 0.15–0.20 (moderado)", fontsize=8, color="#f39c12")
    fig.text(0.65, 0.02, "🔴 Gap > 0.20 (alto)", fontsize=8, color="#e74c3c")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(OUTPUT_DIR, "metrics_summary_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado: {path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ANÁLISIS DE SOBREAJUSTE — VOTING ENSEMBLE")
    print("=" * 60)

    # Intentar cargar métricas reales del CSV si existe
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        print(f"\nMétricas cargadas de: {RESULTS_CSV}")
        print(df.to_string(index=False))

        # Actualizar METRICS con valores reales
        for _, row in df.iterrows():
            label = row["label"]
            if label == "baseline_0.50":
                target = "Test"
            elif label == "train":
                target = "Train"
            elif label == "walk_forward":
                target = "Walk-Forward"
            else:
                continue

            METRICS[target]["Accuracy"] = round(row["accuracy"], 4)
            METRICS[target]["ROC-AUC"] = round(row["roc_auc"], 4)
            METRICS[target]["F1 Weighted"] = round(row["f1_weighted"], 4)
            METRICS[target]["Recall (Sube)"] = round(row["recall_sube"], 4)
            METRICS[target]["Precision (Sube)"] = round(row["precision_sube"], 4)

    print("\n--- Métricas utilizadas ---")
    for split, metrics in METRICS.items():
        print(f"\n  {split}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    # Generar gráficas
    print(f"\n{'='*60}")
    print("GENERANDO GRÁFICAS...")
    print("=" * 60)

    plot_metrics_comparison()
    plot_overfitting_gap()
    plot_metrics_table()

    print(f"\n{'='*60}")
    print("RESUMEN DEL ANÁLISIS DE SOBREAJUSTE")
    print("=" * 60)
    gap_auc = METRICS["Train"]["ROC-AUC"] - METRICS["Test"]["ROC-AUC"]
    gap_acc = METRICS["Train"]["Accuracy"] - METRICS["Test"]["Accuracy"]
    wf_mejora = METRICS["Walk-Forward"]["Accuracy"] - WF_BASELINE

    print(f"  Gap ROC-AUC (Train→Test):     {gap_auc:.4f}")
    print(f"  Gap Accuracy (Train→Test):    {gap_acc:.4f}")
    print(f"  WF Accuracy vs Baseline:      +{wf_mejora*100:.2f}%")
    print(f"  WF Predicciones:              {WF_N_PREDS} días")
    print(f"\n  Interpretación:")
    if gap_auc > 0.25:
        print("    ⚠️  Sobreajuste ALTO — considerar regularización adicional")
    elif gap_auc > 0.15:
        print("    🟡 Sobreajuste MODERADO — aceptable para mercados financieros")
    else:
        print("    ✅ Sobreajuste BAJO — buena generalización")

    print(f"\n{'='*60}")
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
