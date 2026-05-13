"""
compare_results.py

Compara resultados de todos los modelos entre:
  - financial  (solo features técnicas, 2019-2026)
  - news       (financial + noticias GDELT, 2019-2026)
  - reddit     (financial + Reddit FinBERT, 2023-2025)

Si algún CSV no existe todavía, se omite sin error.

Output:
  outputs/comparison_results.csv
  outputs/comparison_plot.png
  outputs/comparison_plot_reddit.png

Uso:
  python scripts/compare_results.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR        = "outputs"
OUTPUT_CSV        = f"{OUTPUT_DIR}/comparison_results.csv"
OUTPUT_PLOT_FN    = f"{OUTPUT_DIR}/comparison_plot.png"
OUTPUT_PLOT_R     = f"{OUTPUT_DIR}/comparison_plot_reddit.png"

# ── Mapa de resultados ────────────────────────────────────────────────────────
RESULTS = {
    "LR": {
        "financial": "scripts/financial/outputs/lineal_regression/lr_financial_results.csv",
        "news":      "scripts/news/outputs/lineal_regression/lr_news_results.csv",
        "reddit":    "scripts/reddit/outputs/lineal_regression/lr_reddit2_results.csv",
    },
    "RF": {
        "financial": "scripts/financial/outputs/random_forest/rf_financial_results.csv",
        "news":      "scripts/news/outputs/random_forest/rf_news_results.csv",
        "reddit":    "scripts/reddit/outputs/random_forest/rf_reddit2_results.csv",
    },
    "XGBoost": {
        "financial": "scripts/financial/outputs/xgboost/xgb_financial_results.csv",
        "news":      "scripts/news/outputs/xgboost/xgb_news_results.csv",
        "reddit":    "scripts/reddit/outputs/xgboost/xgb_reddit2_results.csv",
    },
    "LightGBM": {
        "financial": "scripts/financial/outputs/lightGBM/lgbm_financial_results.csv",
        "news":      "scripts/news/outputs/lightGBM/lgbm_news_results.csv",
        "reddit":    "scripts/reddit/outputs/lightGBM/lgbm_reddit2_results.csv",
    },
    "LSTM": {
        "financial": "scripts/financial/outputs/LSTM/lstm_financial_results.csv",
        "news":      "scripts/news/outputs/lstm/lstm_news_results.csv",
        "reddit":    "scripts/reddit/outputs/lstm/lstm_reddit2_results.csv",
    },
    "GRU": {
        "financial": "scripts/financial/outputs/GRU/gru_financial_results.csv",
        "news":      "scripts/news/outputs/GRU/gru_news_results.csv",
        "reddit":    "scripts/reddit/outputs/GRU/gru_reddit2_results.csv",
    },
    "Ensemble": {
        "ensemble": "scripts/ensemble/outputs/voting_ensemble_results.csv",
    },
}

DATASETS = ["financial", "news", "reddit", "ensemble"]
METRICS  = ["accuracy", "roc_auc", "f1_weighted", "recall_sube", "precision_sube"]

COLORS = {
    "financial": "#3498db",
    "news":      "#e67e22",
    "reddit":    "#e74c3c",
    "ensemble":  "#8e44ad",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_row(path, label):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["label"] == label]
    return row.iloc[0] if not row.empty else None


def build_summary(label="baseline_0.50"):
    rows = []
    for model, paths in RESULTS.items():
        for dataset in DATASETS:
            path = paths.get(dataset)
            row  = load_row(path, label)
            if row is None:
                continue
            entry = {"model": model, "dataset": dataset}
            for m in METRICS:
                entry[m] = round(float(row[m]), 4) if m in row and not pd.isna(row[m]) else None
            rows.append(entry)
    return pd.DataFrame(rows)

def get_val(df, model, dataset, metric):
    sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
    if sub.empty or sub[metric].values[0] is None:
        return float("nan")
    return float(sub[metric].values[0])


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(df, title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    available_datasets = [d for d in DATASETS if d in df["dataset"].values]
    for model in RESULTS.keys():
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        print(f"\n  {model}")
        model_datasets = [d for d in available_datasets if d in sub["dataset"].values]
        header = f"  {'Métrica':<18}" + "".join(f" {d:>11}" for d in model_datasets)
        print(header)
        print(f"  {'-'*60}")
        for m in METRICS:
            vals = [get_val(df, model, d, m) for d in model_datasets]
            line = f"  {m:<18}"
            for v in vals:
                line += f" {v:>11.4f}" if not np.isnan(v) else f" {'N/A':>11}"
            # Δ news-fin y reddit-fin si existen (no aplica para ensemble)
            deltas = []
            fin_v = get_val(df, model, "financial", m)
            for d in model_datasets[1:]:
                if d == "ensemble":
                    continue
                dv = get_val(df, model, d, m)
                if not np.isnan(fin_v) and not np.isnan(dv):
                    delta = dv - fin_v
                    sign  = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
                    deltas.append(f"{sign}{delta:+.4f}")
                else:
                    deltas.append("  N/A  ")
            if deltas:
                line += "   " + "  ".join(deltas)
            print(line)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(df_test):
    print(f"\n{'='*75}")
    print("  RESUMEN — ¿Añadir señal externa mejora vs solo financial?")
    print(f"{'='*75}")
    print(f"  {'Modelo':<12} {'Dataset':<12} {'Acc':>8} {'AUC':>8} {'F1':>8}  {'Δ Acc':>8} {'Δ AUC':>8}")
    print(f"  {'-'*68}")
    for model in RESULTS.keys():
        fin_acc = get_val(df_test, model, "financial", "accuracy")
        fin_auc = get_val(df_test, model, "financial", "roc_auc")
        for dataset in ["news", "reddit", "ensemble"]:
            acc = get_val(df_test, model, dataset, "accuracy")
            auc = get_val(df_test, model, dataset, "roc_auc")
            f1  = get_val(df_test, model, dataset, "f1_weighted")
            if np.isnan(acc):
                continue
            da = acc - fin_acc if not np.isnan(fin_acc) else float("nan")
            dr = auc - fin_auc if not np.isnan(fin_auc) else float("nan")
            verdict = "✅" if (not np.isnan(da) and da > 0) else ("➖" if np.isnan(da) else "❌")
            print(f"  {model:<12} {dataset:<12} {acc:>8.4f} {auc:>8.4f} {f1:>8.4f}  {da:>+8.4f} {dr:>+8.4f}  {verdict}"
                  if not np.isnan(da) else
                  f"  {model:<12} {dataset:<12} {acc:>8.4f} {auc:>8.4f} {f1:>8.4f}  {'N/A':>8} {'N/A':>8}  {verdict}")
        # separador entre modelos
        print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_financial_vs_news(df_test, df_wf):
    """Gráfico original: financial vs news (mantiene compatibilidad)."""
    models  = [m for m in RESULTS.keys() if m != "Ensemble"]
    x       = np.arange(len(models))
    width   = 0.35
    metrics = ["accuracy", "roc_auc", "f1_weighted"]
    titles  = ["Accuracy (test)", "ROC-AUC (test)", "F1 Weighted (test)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Financial vs News — Comparativa de modelos", fontsize=14, fontweight="bold")

    for col, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[0, col]
        fin_vals  = [get_val(df_test, m, "financial", metric) for m in models]
        news_vals = [get_val(df_test, m, "news",      metric) for m in models]

        b1 = ax.bar(x - width/2, fin_vals,  width, label="Financial", color=COLORS["financial"], alpha=0.85)
        b2 = ax.bar(x + width/2, news_vals, width, label="News",      color=COLORS["news"],      alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0.3, 0.85)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color=COLORS["financial"])
        for bar in b2:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, color=COLORS["news"])

    wf_models = [m for m in models if not np.isnan(get_val(df_wf, m, "financial", "accuracy"))
                                    or not np.isnan(get_val(df_wf, m, "news", "accuracy"))]
    x_wf = np.arange(len(wf_models))

    for col, (metric, title) in enumerate(zip(["accuracy", "roc_auc"], ["WF Accuracy", "WF ROC-AUC"])):
        ax = axes[1, col]
        fin_vals  = [get_val(df_wf, m, "financial", metric) for m in wf_models]
        news_vals = [get_val(df_wf, m, "news",      metric) for m in wf_models]
        ax.bar(x_wf - width/2, fin_vals,  width, label="Financial", color=COLORS["financial"], alpha=0.85)
        ax.bar(x_wf + width/2, news_vals, width, label="News",      color=COLORS["news"],      alpha=0.85)
        ax.set_title(f"{title} (walk-forward)", fontweight="bold")
        ax.set_xticks(x_wf); ax.set_xticklabels(wf_models, rotation=20, ha="right")
        ax.set_ylim(0.3, 0.85)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Delta table
    ax = axes[1, 2]
    ax.axis("off")
    delta_rows = []
    for m in models:
        fin_row  = df_test[(df_test["model"] == m) & (df_test["dataset"] == "financial")]
        news_row = df_test[(df_test["model"] == m) & (df_test["dataset"] == "news")]
        if fin_row.empty or news_row.empty:
            continue
        row = {"Model": m}
        for met in ["accuracy", "roc_auc", "f1_weighted"]:
            fv = fin_row[met].values[0]
            nv = news_row[met].values[0]
            row[f"Δ {met[:3].upper()}"] = round(nv - fv, 4) if (fv and nv) else None
        delta_rows.append(row)

    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        tbl = ax.table(cellText=delta_df.values, colLabels=delta_df.columns,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.6)
        for (ri, ci), cell in tbl.get_celld().items():
            if ri == 0:
                cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
            elif ci > 0:
                try:
                    val = float(cell.get_text().get_text())
                    cell.set_facecolor("#2ecc71" if val > 0 else ("#e74c3c" if val < 0 else "white"))
                    cell.set_alpha(min(abs(val) * 10 + 0.2, 0.8))
                except (ValueError, TypeError):
                    pass
    ax.set_title("Δ News − Financial (test)", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FN, dpi=150, bbox_inches="tight")
    print(f"Gráfico financial vs news: {OUTPUT_PLOT_FN}")


def plot_three_way(df_test, df_wf):
    """Gráfico de 3 datasets: financial / news / reddit."""
    models  = [m for m in RESULTS.keys() if m != "Ensemble"]
    x       = np.arange(len(models))
    width   = 0.25
    metrics = ["accuracy", "roc_auc", "f1_weighted"]
    titles  = ["Accuracy (test)", "ROC-AUC (test)", "F1 Weighted (test)"]
    base_datasets = ["financial", "news", "reddit"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Financial vs News vs Reddit — Comparativa de modelos", fontsize=14, fontweight="bold")

    for col, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[0, col]
        for i, dataset in enumerate(base_datasets):
            vals = [get_val(df_test, m, dataset, metric) for m in models]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=dataset.capitalize(),
                          color=COLORS[dataset], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h) and h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=6.5,
                            color=COLORS[dataset])
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0.3, 0.90)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Walk-forward: accuracy y AUC
    wf_models = [m for m in models
                 if any(not np.isnan(get_val(df_wf, m, d, "accuracy")) for d in base_datasets)]
    x_wf = np.arange(len(wf_models))

    for col, (metric, title) in enumerate(zip(["accuracy", "roc_auc"], ["WF Accuracy", "WF ROC-AUC"])):
        ax = axes[1, col]
        for i, dataset in enumerate(base_datasets):
            vals = [get_val(df_wf, m, dataset, metric) for m in wf_models]
            offset = (i - 1) * width
            ax.bar(x_wf + offset, vals, width, label=dataset.capitalize(),
                   color=COLORS[dataset], alpha=0.85)
        ax.set_title(f"{title} (walk-forward)", fontweight="bold")
        ax.set_xticks(x_wf); ax.set_xticklabels(wf_models, rotation=20, ha="right")
        ax.set_ylim(0.3, 0.90)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Tabla delta doble: Δ news-fin y Δ reddit-fin
    ax = axes[1, 2]
    ax.axis("off")
    delta_rows = []
    for m in models:
        fin_acc = get_val(df_test, m, "financial", "accuracy")
        fin_auc = get_val(df_test, m, "financial", "roc_auc")
        row = {"Model": m}
        for dataset, label in [("news", "News"), ("reddit", "Reddit")]:
            acc = get_val(df_test, m, dataset, "accuracy")
            auc = get_val(df_test, m, dataset, "roc_auc")
            row[f"Δ Acc\n({label})"]  = round(acc - fin_acc, 4) if not (np.isnan(acc) or np.isnan(fin_acc)) else None
            row[f"Δ AUC\n({label})"]  = round(auc - fin_auc, 4) if not (np.isnan(auc) or np.isnan(fin_auc)) else None
        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows).fillna("N/A")
    tbl = ax.table(cellText=delta_df.values, colLabels=delta_df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1, 1.5)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif ci > 0:
            try:
                val = float(cell.get_text().get_text())
                cell.set_facecolor("#2ecc71" if val > 0 else ("#e74c3c" if val < 0 else "white"))
                cell.set_alpha(min(abs(val) * 10 + 0.2, 0.8))
            except (ValueError, TypeError):
                pass
    ax.set_title("Δ vs Financial (test)", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_R, dpi=150, bbox_inches="tight")
    print(f"Gráfico 3-way:             {OUTPUT_PLOT_R}")


OUTPUT_PLOT_ENS = f"{OUTPUT_DIR}/comparison_plot_ensemble.png"


def plot_ensemble_comparison(df_test, df_wf):
    """Gráfico comparativo: Voting Ensemble vs mejores modelos base (XGBoost_News, LightGBM_News, XGBoost_Reddit)."""
    ens_row_test = df_test[df_test["dataset"] == "ensemble"]
    if ens_row_test.empty:
        print("Sin datos del ensemble — omitiendo gráfico de ensemble.")
        return

    # Modelos base de referencia y sus datasets
    base_entries = [
        ("XGBoost",  "news",    "XGBoost_News"),
        ("LightGBM", "news",    "LightGBM_News"),
        ("XGBoost",  "reddit",  "XGBoost_Reddit"),
    ]
    metrics = ["accuracy", "roc_auc", "f1_weighted"]
    titles  = ["Accuracy (test)", "ROC-AUC (test)", "F1 Weighted (test)"]

    # Construir tabla de comparación
    labels = [e[2] for e in base_entries] + ["Voting_Ensemble"]
    colors = ["#3498db", "#e67e22", "#e74c3c", "#8e44ad"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Voting Ensemble vs Modelos Base — Comparativa", fontsize=14, fontweight="bold")

    x     = np.arange(len(labels))
    width = 0.55

    for ax, metric, title in zip(axes, metrics, titles):
        vals = []
        for model, dataset, _ in base_entries:
            vals.append(get_val(df_test, model, dataset, metric))
        ens_val = float(ens_row_test[metric].values[0]) if metric in ens_row_test.columns else float("nan")
        vals.append(ens_val)

        bars = ax.bar(x, vals, width, color=colors, edgecolor="white", alpha=0.9)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0.3, 0.90)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#8e44ad", label="Voting Ensemble"),
        Patch(facecolor="#3498db", label="Modelos Base"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUTPUT_PLOT_ENS, dpi=150, bbox_inches="tight")
    print(f"Gráfico ensemble:          {OUTPUT_PLOT_ENS}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_test = build_summary("baseline_0.50")
    df_wf   = build_summary("walk_forward")

    print_table(df_test, "TEST (threshold=0.50)")
    print_table(df_wf,   "WALK-FORWARD")
    print_summary(df_test)

    # Guardar CSV completo
    all_df = pd.concat([
        df_test.assign(eval_type="test"),
        df_wf.assign(eval_type="walk_forward"),
    ], ignore_index=True)
    all_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV guardado: {OUTPUT_CSV}")

    plot_financial_vs_news(df_test, df_wf)
    plot_three_way(df_test, df_wf)
    plot_ensemble_comparison(df_test, df_wf)


if __name__ == "__main__":
    main()
