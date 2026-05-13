"""
reliability_test.py

Test de fiabilidad del Voting Ensemble: compara las métricas del ensemble
frente a cada modelo base individual.

Lee los CSVs generados por los scripts base y por voting_ensemble.py.

Inputs:
  scripts/ensemble/outputs/voting_ensemble_results.csv
  scripts/news/outputs/xgboost/xgb_news_results.csv
  scripts/news/outputs/lightGBM/lgbm_news_results.csv
  scripts/reddit/outputs/xgboost/xgb_reddit2_results.csv

Output:
  outputs/voting_ensemble/reliability_report.txt

Uso:
  python scripts/ensemble/reliability_test.py
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ENSEMBLE_CSV = "scripts/ensemble/outputs/voting_ensemble_results.csv"

BASE_CSVS = {
    "XGBoost_News":   "scripts/news/outputs/xgboost/xgb_news_results.csv",
    "LightGBM_News":  "scripts/news/outputs/lightGBM/lgbm_news_results.csv",
    "XGBoost_Reddit": "scripts/reddit/outputs/xgboost/xgb_reddit2_results.csv",
}

REPORT_PATH = "outputs/voting_ensemble/reliability_report.txt"
METRICS     = ["roc_auc", "accuracy", "f1_weighted"]


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_results(csv_path: str) -> pd.DataFrame:
    """
    Carga el CSV de resultados. Requisito 5.1
    Raises FileNotFoundError si no existe.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV no encontrado: {csv_path}\n"
            f"Ejecuta primero: python scripts/ensemble/voting_ensemble.py"
        )
    return pd.read_csv(csv_path)


def get_metric(df: pd.DataFrame, label: str, metric: str) -> float:
    """Extrae el valor de una métrica para un label dado."""
    row = df[df["label"] == label]
    if row.empty or metric not in row.columns:
        return float("nan")
    val = row.iloc[0][metric]
    return float(val) if not pd.isna(val) else float("nan")


# ── Comparación ───────────────────────────────────────────────────────────────

def compare_metrics(results_df: pd.DataFrame, eval_type: str = "baseline_0.50") -> dict:
    """
    Compara ROC-AUC, accuracy y F1 weighted del ensemble frente a cada modelo base.
    Requisitos 5.1, 5.2, 5.3, 5.4, 5.6

    Returns:
        {
            "verdict": "PASSED" | "FAILED",
            "comparisons": [...],
            "no_improvement": bool,
            "ensemble_metrics": {...},
        }
    """
    # Cargar métricas del ensemble
    ens_metrics = {m: get_metric(results_df, eval_type, m) for m in METRICS}

    # Cargar métricas de los modelos base
    base_metrics = {}
    for name, path in BASE_CSVS.items():
        if not os.path.exists(path):
            print(f"  [AVISO] CSV no encontrado para {name}: {path}")
            base_metrics[name] = {m: float("nan") for m in METRICS}
            continue
        df_base = pd.read_csv(path)
        base_metrics[name] = {m: get_metric(df_base, eval_type, m) for m in METRICS}

    comparisons = []
    any_improvement = False

    for model_name, bm in base_metrics.items():
        for metric in METRICS:
            ens_val  = ens_metrics[metric]
            base_val = bm[metric]
            delta    = ens_val - base_val if not (np.isnan(ens_val) or np.isnan(base_val)) else float("nan")
            passed   = bool(delta > 0) if not np.isnan(delta) else False
            if passed:
                any_improvement = True
            comparisons.append({
                "metric":      metric,
                "model":       model_name,
                "ensemble_val": ens_val,
                "base_val":    base_val,
                "delta":       delta,
                "passed":      passed,
            })

    # Veredicto: PASSED si y solo si ROC-AUC del ensemble > ROC-AUC de los tres modelos base
    base_aucs = [base_metrics[n]["roc_auc"] for n in BASE_CSVS]
    ens_auc   = ens_metrics["roc_auc"]
    verdict   = "PASSED" if (not np.isnan(ens_auc) and all(
        not np.isnan(b) and ens_auc > b for b in base_aucs
    )) else "FAILED"

    no_improvement = not any_improvement

    if no_improvement:
        print("\n⚠️  ADVERTENCIA: El ensemble no mejora en ninguna métrica respecto a ningún modelo base.")
        print("   Considera revisar los pesos del ensemble o los hiperparámetros de los modelos.")

    return {
        "verdict":         verdict,
        "comparisons":     comparisons,
        "no_improvement":  no_improvement,
        "ensemble_metrics": ens_metrics,
        "base_metrics":    base_metrics,
        "eval_type":       eval_type,
    }


# ── Informe ───────────────────────────────────────────────────────────────────

def save_report(report: dict, output_path: str) -> None:
    """
    Guarda el informe de fiabilidad en texto plano. Requisito 5.5
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("VOTING ENSEMBLE — RELIABILITY REPORT")
    lines.append("=" * 45)
    lines.append(f"Eval type: {report['eval_type']}")
    lines.append("")
    lines.append(f"VEREDICTO: {report['verdict']}")
    lines.append("")

    ens_m = report["ensemble_metrics"]
    base_m = report["base_metrics"]

    for metric in METRICS:
        lines.append(f"{metric.upper().replace('_', ' ')}:")
        lines.append(f"  Ensemble:       {ens_m[metric]:.4f}")
        for model_name in BASE_CSVS:
            bv    = base_m[model_name][metric]
            delta = ens_m[metric] - bv if not (np.isnan(ens_m[metric]) or np.isnan(bv)) else float("nan")
            icon  = "✅" if delta > 0 else "❌"
            delta_str = f"Δ = {delta:+.4f}" if not np.isnan(delta) else "Δ = N/A"
            lines.append(f"  {model_name:<16} {bv:.4f}  {delta_str}  {icon}")
        lines.append("")

    if report["no_improvement"]:
        lines.append("[ADVERTENCIA: El ensemble no mejora en ninguna métrica respecto a ningún modelo base.]")
        lines.append("Considera revisar los pesos del ensemble o los hiperparámetros de los modelos.")

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Informe guardado: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RELIABILITY TEST — VOTING ENSEMBLE")
    print("=" * 60)

    results_df = load_results(ENSEMBLE_CSV)
    print(f"CSV cargado: {ENSEMBLE_CSV}")
    print(f"Labels disponibles: {results_df['label'].tolist()}")

    report = compare_metrics(results_df, eval_type="baseline_0.50")

    print(f"\n{'='*60}")
    print(f"VEREDICTO: {report['verdict']}")
    print(f"{'='*60}")

    ens_m  = report["ensemble_metrics"]
    base_m = report["base_metrics"]

    for metric in METRICS:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Ensemble:       {ens_m[metric]:.4f}")
        for model_name in BASE_CSVS:
            bv    = base_m[model_name][metric]
            delta = ens_m[metric] - bv if not (np.isnan(ens_m[metric]) or np.isnan(bv)) else float("nan")
            icon  = "✅" if delta > 0 else "❌"
            delta_str = f"Δ = {delta:+.4f}" if not np.isnan(delta) else "Δ = N/A"
            print(f"  {model_name:<16} {bv:.4f}  {delta_str}  {icon}")

    save_report(report, REPORT_PATH)


if __name__ == "__main__":
    main()
