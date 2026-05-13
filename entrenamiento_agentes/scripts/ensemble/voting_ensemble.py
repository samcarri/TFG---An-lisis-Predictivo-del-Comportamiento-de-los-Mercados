"""
voting_ensemble.py

Voting Ensemble System: combina XGBoost_News, LightGBM_News y XGBoost_Reddit
mediante soft voting para predecir el movimiento diario de NVIDIA.

Datasets:
  data/dataset_nvda_news.csv    (XGBoost_News, LightGBM_News)
  data/dataset_nvda_reddit.csv  (XGBoost_Reddit)

Output:
  scripts/ensemble/outputs/voting_ensemble_results.csv
  outputs/voting_ensemble/ensemble_comparison_plot.png
  outputs/voting_ensemble/reliability_report.txt
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score,
)
import warnings
warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
STEP         = 10
THRESHOLD    = 0.50

OUTPUT_CSV   = "scripts/ensemble/outputs/voting_ensemble_results.csv"
OUTPUT_PLOT  = "outputs/voting_ensemble/ensemble_comparison_plot.png"

XGBOOST_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
    "early_stopping_rounds": 30,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "binary", "random_state": RANDOM_STATE,
    "n_jobs": -1, "verbose": -1,
}

MODEL_CONFIGS = [
    {
        "name":   "RF_News",
        "type":   "randomforest",
        "data":   "data/dataset_nvda_news.csv",
        "model":  "trained_models/rf_news_model.joblib",
        "scaler": "trained_models/rf_news_scaler.joblib",
    },
    {
        "name":   "LightGBM_News",
        "type":   "lightgbm",
        "data":   "data/dataset_nvda_news.csv",
        "model":  "trained_models/lgbm_news_model.joblib",
        "scaler": "trained_models/lgbm_news_scaler.joblib",
    },
    {
        "name":   "XGBoost_Reddit",
        "type":   "xgboost",
        "data":   "data/dataset_nvda_reddit.csv",
        "model":  "trained_models/xgb_reddit2_model.joblib",
        "scaler": "trained_models/xgb_reddit2_scaler.joblib",
    },
]


# ── Utilidades ────────────────────────────────────────────────────────────────

def log_library_versions():
    """Imprime versiones de las librerías principales. Requisito 8.2"""
    print("=" * 60)
    print("VERSIONES DE LIBRERÍAS")
    print("=" * 60)
    print(f"  xgboost:      {xgb.__version__}")
    print(f"  lightgbm:     {lgb.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")
    print(f"  pandas:       {pd.__version__}")
    print(f"  numpy:        {np.__version__}")


# ── Carga y split ─────────────────────────────────────────────────────────────

def load_and_split(path: str):
    """
    Carga un dataset CSV, ordena por fecha y aplica split temporal 70/15/15.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test, dates, feature_names
    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el dataset tiene menos de 10 filas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Dataset insuficiente para split 70/15/15: solo {len(df)} filas en {path}")

    feature_names = [c for c in df.columns if c not in {"date", "target"}]
    X = df[feature_names].values
    y = df["target"].values
    dates = df["date"]

    n         = len(X)
    train_end = int(np.floor(n * 0.70))
    val_end   = int(np.floor(n * 0.85))

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"Dataset: {df.shape}  |  {dates.min().date()} → {dates.max().date()}")
    print(f"Features: {len(feature_names)}  |  Target: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Train: {dates.iloc[0].date()} → {dates.iloc[train_end-1].date()}  ({train_end})")
    print(f"Val:   {dates.iloc[train_end].date()} → {dates.iloc[val_end-1].date()}  ({val_end - train_end})")
    print(f"Test:  {dates.iloc[val_end].date()} → {dates.iloc[-1].date()}  ({n - val_end})")

    return X_train, X_val, X_test, y_train, y_val, y_test, dates, feature_names


# ── Construcción y entrenamiento de modelos base ──────────────────────────────

def build_model(model_type: str, n_neg: int, n_pos: int):
    """
    Instancia XGBClassifier o LGBMClassifier con hiperparámetros fijos.
    Requisitos 2.1, 2.2, 2.3, 2.5
    """
    if n_pos == 0:
        raise ValueError("No hay muestras de clase positiva en train (n_pos == 0)")

    spw = round(n_neg / n_pos, 4)

    if model_type == "xgboost":
        params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}
        return xgb.XGBClassifier(**params)
    elif model_type == "lightgbm":
        params = {**LIGHTGBM_PARAMS, "scale_pos_weight": spw}
        return lgb.LGBMClassifier(**params)
    elif model_type == "randomforest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_split=20,
            min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        raise ValueError(f"model_type desconocido: {model_type}")


def train_base_model(model_type: str, X_train, y_train, X_val, y_val,
                     model_path: str = None, scaler_path: str = None):
    """
    Carga modelo y scaler desde disco si existen; si no, entrena desde cero.
    Requisitos 2.4, 2.5
    Returns: (modelo_entrenado, scaler, X_train_s, X_val_s)
    """
    if model_path and scaler_path and os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"  Cargando modelo guardado: {model_path}")
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)
        return model, scaler, X_train_s, X_val_s

    print(f"  Entrenando desde cero (no se encontró modelo guardado)")
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    model = build_model(model_type, n_neg, n_pos)

    if model_type == "xgboost":
        model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    elif model_type == "lightgbm":
        model.fit(X_train_s, y_train,
                  eval_set=[(X_val_s, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])
    else:  # randomforest
        model.fit(X_train_s, y_train)

    return model, scaler, X_train_s, X_val_s


# ── Soft voting y umbralización ───────────────────────────────────────────────

def soft_vote(probas: list, threshold: float = 0.50):
    """
    Combina probabilidades mediante promedio aritmético y aplica umbral.
    Requisitos 3.1, 3.2, 3.3, 3.4, 3.5

    Returns: (y_pred, avg_proba)
    """
    avg_proba = np.mean(probas, axis=0)
    y_pred    = (avg_proba >= threshold).astype(int)
    return y_pred, avg_proba


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba, label: str) -> dict:
    """
    Calcula accuracy, ROC-AUC, F1 weighted, recall y precision.
    Imprime classification_report. Requisitos 4.1, 4.2, 4.3
    """
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, pos_label=1)
    pre = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print(f"[{label}] thr=0.50  acc={acc:.4f}  auc={auc:.4f}  recall={rec:.4f}  prec={pre:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Baja", "Sube"]))

    return {
        "label":          label,
        "threshold":      THRESHOLD,
        "accuracy":       acc,
        "f1_weighted":    f1,
        "recall_sube":    rec,
        "precision_sube": pre,
        "roc_auc":        auc,
    }


# ── Persistencia ──────────────────────────────────────────────────────────────

def save_results(results: list, output_path: str) -> None:
    """
    Guarda métricas en CSV, creando el directorio si no existe.
    Requisitos 4.4, 4.5, 8.3
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    # Asegurar orden de columnas compatible con compare_results.py
    cols = ["label", "threshold", "accuracy", "f1_weighted",
            "recall_sube", "precision_sube", "roc_auc", "wf_n_preds", "wf_baseline"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"CSV guardado: {output_path}")


# ── Visualización ─────────────────────────────────────────────────────────────

def plot_comparison(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Gráfico de barras comparando accuracy, ROC-AUC y F1 weighted.
    Requisitos 7.1, 7.2, 7.3, 7.4
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_df = results_df[results_df["label"] == "baseline_0.50"].copy()
    models  = test_df["model"].tolist()
    metrics = ["accuracy", "roc_auc", "f1_weighted"]
    titles  = ["Accuracy (test)", "ROC-AUC (test)", "F1 Weighted (test)"]

    x     = np.arange(len(models))
    width = 0.55

    # Color diferenciado para el ensemble
    colors = ["#e74c3c" if m == "Voting_Ensemble" else "#3498db" for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Voting Ensemble vs Modelos Base", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        vals = test_df[metric].tolist()
        bars = ax.bar(x, vals, width, color=colors, edgecolor="white", alpha=0.9)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0.3, 0.90)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Voting Ensemble"),
        Patch(facecolor="#3498db", label="Modelo Base"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado: {output_path}")


# ── Walk-forward ensemble ─────────────────────────────────────────────────────

def walk_forward_ensemble(datasets: list, model_configs: list,
                          threshold: float = 0.50, step: int = 10,
                          min_train_ratio: float = 0.70) -> dict:
    """
    Evaluación walk-forward sobre los tres modelos base simultáneamente.
    Usa la intersección de fechas entre los tres datasets.
    Requisitos 6.1, 6.2, 6.3, 6.4, 6.5
    """
    # datasets: lista de (X_full, y_full, dates_full) para cada modelo
    # Encontrar intersección de fechas
    date_sets = [set(d[2].dt.date) for d in datasets]
    common_dates = sorted(date_sets[0].intersection(*date_sets[1:]))
    print(f"\nFechas comunes para walk-forward: {common_dates[0]} → {common_dates[-1]}  ({len(common_dates)} días)")

    # Filtrar cada dataset a las fechas comunes
    aligned = []
    for X_full, y_full, dates_full in datasets:
        mask = dates_full.dt.date.isin(set(common_dates))
        idx  = np.where(mask)[0]
        aligned.append((X_full[idx], y_full[idx]))

    N         = len(aligned[0][0])
    min_train = int(np.floor(N * min_train_ratio))

    print(f"WALK-FORWARD ENSEMBLE  N={N}  min_train={min_train}  step={step}  thr={threshold}")

    all_preds, all_true, all_probas = [], [], []
    model_caches  = [None] * len(aligned)
    scaler_caches = [None] * len(aligned)

    for t in range(min_train, N - 1):
        if (t - min_train) % step == 0:
            for i, ((X_a, y_a), cfg) in enumerate(zip(aligned, model_configs)):
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_a[:t])
                n_neg  = int((y_a[:t] == 0).sum())
                n_pos  = int((y_a[:t] == 1).sum())
                if n_pos == 0:
                    n_pos = 1
                m = build_model(cfg["type"], n_neg, n_pos)
                # Quitar early_stopping para walk-forward (sin val set)
                if cfg["type"] == "xgboost":
                    params = {k: v for k, v in XGBOOST_PARAMS.items()
                              if k != "early_stopping_rounds"}
                    params["scale_pos_weight"] = round(n_neg / n_pos, 4)
                    m = xgb.XGBClassifier(**params)
                    m.fit(X_tr_s, y_a[:t], verbose=False)
                elif cfg["type"] == "lightgbm":
                    params = {**LIGHTGBM_PARAMS, "scale_pos_weight": round(n_neg / n_pos, 4)}
                    m = lgb.LGBMClassifier(**params)
                    m.fit(X_tr_s, y_a[:t], callbacks=[lgb.log_evaluation(-1)])
                else:  # randomforest
                    m = RandomForestClassifier(
                        n_estimators=300, max_depth=6, min_samples_split=20,
                        min_samples_leaf=10, max_features="sqrt",
                        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
                    )
                    m.fit(X_tr_s, y_a[:t])
                model_caches[i]  = m
                scaler_caches[i] = sc

        # Predicción combinada
        probas_t = []
        for i, (X_a, _) in enumerate(aligned):
            p = model_caches[i].predict_proba(scaler_caches[i].transform(X_a[t:t+1]))[0, 1]
            probas_t.append(p)

        avg_p = float(np.mean(probas_t))
        all_probas.append(avg_p)
        all_preds.append(int(avg_p >= threshold))
        all_true.append(int(aligned[0][1][t]))

    all_preds  = np.array(all_preds)
    all_true   = np.array(all_true)
    all_probas = np.array(all_probas)

    acc      = accuracy_score(all_true, all_preds)
    f1       = f1_score(all_true, all_preds, average="weighted")
    rec      = recall_score(all_true, all_preds, pos_label=1)
    pre      = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    auc      = roc_auc_score(all_true, all_probas)
    baseline = accuracy_score(all_true, np.full_like(all_preds, int(np.bincount(all_true).argmax())))

    print(f"OOS: {len(all_preds)} días")
    print(f"Acc={acc:.4f}  AUC={auc:.4f}  Recall={rec:.4f}  Prec={pre:.4f}")
    print(f"Baseline={baseline:.4f}  Mejora={acc - baseline:+.4f}")
    print(classification_report(all_true, all_preds, target_names=["Baja", "Sube"]))

    return {
        "label":          "walk_forward",
        "threshold":      threshold,
        "accuracy":       acc,
        "f1_weighted":    f1,
        "recall_sube":    rec,
        "precision_sube": pre,
        "roc_auc":        auc,
        "wf_n_preds":     len(all_preds),
        "wf_baseline":    baseline,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_library_versions()

    print("\n" + "=" * 60)
    print("VOTING ENSEMBLE — XGBoost_News + LightGBM_News + XGBoost_Reddit")
    print("=" * 60)

    # ── 1. Cargar y dividir datasets ──────────────────────────────────────────
    loaded = []
    for cfg in MODEL_CONFIGS:
        print(f"\n--- {cfg['name']} ({cfg['data']}) ---")
        splits = load_and_split(cfg["data"])
        loaded.append(splits)

    # ── 2. Entrenar modelos base ──────────────────────────────────────────────
    trained = []
    for cfg, splits in zip(MODEL_CONFIGS, loaded):
        X_train, X_val, X_test, y_train, y_val, y_test, dates, features = splits
        print(f"\n--- Entrenando {cfg['name']} ---")
        model, scaler, X_train_s, X_val_s = train_base_model(
            cfg["type"], X_train, y_train, X_val, y_val,
            model_path=cfg.get("model"), scaler_path=cfg.get("scaler"),
        )
        X_test_s = scaler.transform(X_test)
        trained.append({
            "cfg":       cfg,
            "model":     model,
            "scaler":    scaler,
            "X_train_s": X_train_s,
            "X_val_s":   X_val_s,
            "X_test_s":  X_test_s,
            "y_train":   y_train,
            "y_val":     y_val,
            "y_test":    y_test,
            "dates":     dates,
            "features":  features,
            "X_full":    splits[0],   # X_train original (sin escalar)
            "y_full":    np.concatenate([splits[3], splits[4], splits[5]]),
            "dates_full": splits[6],
            "X_full_raw": np.concatenate([splits[0], splits[1], splits[2]]),
        })

    # ── 3. Evaluación estática en test ────────────────────────────────────────
    print(f"\n{'='*60}\nEVALUACIÓN TEST — MODELOS BASE\n{'='*60}")
    base_results = []
    test_probas  = []

    for t in trained:
        name = t["cfg"]["name"]
        proba_test = t["model"].predict_proba(t["X_test_s"])[:, 1]
        test_probas.append(proba_test)

        y_pred_base = (proba_test >= THRESHOLD).astype(int)
        r = compute_metrics(t["y_test"], y_pred_base, proba_test, f"{name}_baseline_0.50")
        r["model"] = name
        base_results.append(r)

    # Verificar que todos los test sets tienen el mismo tamaño (para soft voting)
    # Los datasets news y reddit tienen tamaños distintos; usamos el test del dataset news
    # como referencia para el ensemble estático (intersección de fechas en test)
    # Alineamos por fecha para el test estático
    print(f"\n{'='*60}\nEVALUACIÓN TEST — VOTING ENSEMBLE\n{'='*60}")

    # Obtener fechas de test de cada modelo
    test_date_sets = []
    for t in trained:
        n         = len(t["dates"])
        val_end   = int(np.floor(n * 0.85))
        test_dates = t["dates"].iloc[val_end:].dt.date
        test_date_sets.append(set(test_dates))

    common_test_dates = sorted(test_date_sets[0].intersection(*test_date_sets[1:]))
    print(f"Fechas comunes en test: {common_test_dates[0]} → {common_test_dates[-1]}  ({len(common_test_dates)} días)")

    aligned_test_probas = []
    aligned_y_test      = None

    for i, t in enumerate(trained):
        n       = len(t["dates"])
        val_end = int(np.floor(n * 0.85))
        test_dates_series = t["dates"].iloc[val_end:].dt.date.reset_index(drop=True)
        mask = test_dates_series.isin(set(common_test_dates))
        aligned_test_probas.append(test_probas[i][mask.values])
        if aligned_y_test is None:
            aligned_y_test = t["y_test"][mask.values]

    y_pred_ens, avg_proba_ens = soft_vote(aligned_test_probas, THRESHOLD)
    res_ens_test = compute_metrics(aligned_y_test, y_pred_ens, avg_proba_ens, "baseline_0.50")
    res_ens_test["model"] = "Voting_Ensemble"

    # Train metrics del ensemble (promedio de probabilidades en train)
    train_date_sets = []
    for t in trained:
        n         = len(t["dates"])
        train_end = int(np.floor(n * 0.70))
        train_dates = t["dates"].iloc[:train_end].dt.date
        train_date_sets.append(set(train_dates))

    common_train_dates = sorted(train_date_sets[0].intersection(*train_date_sets[1:]))
    aligned_train_probas = []
    aligned_y_train      = None

    for i, t in enumerate(trained):
        n         = len(t["dates"])
        train_end = int(np.floor(n * 0.70))
        train_dates_series = t["dates"].iloc[:train_end].dt.date.reset_index(drop=True)
        mask = train_dates_series.isin(set(common_train_dates))
        proba_train = t["model"].predict_proba(t["X_train_s"])[:, 1]
        aligned_train_probas.append(proba_train[mask.values])
        if aligned_y_train is None:
            aligned_y_train = t["y_train"][mask.values]

    y_pred_tr, avg_proba_tr = soft_vote(aligned_train_probas, THRESHOLD)
    res_ens_train = compute_metrics(aligned_y_train, y_pred_tr, avg_proba_tr, "train")
    res_ens_train["model"] = "Voting_Ensemble"

    # ── 4. Walk-forward ensemble ──────────────────────────────────────────────
    print(f"\n{'='*60}\nWALK-FORWARD ENSEMBLE\n{'='*60}")
    wf_datasets = [
        (t["X_full_raw"], t["y_full"], t["dates_full"])
        for t in trained
    ]
    res_wf = walk_forward_ensemble(wf_datasets, MODEL_CONFIGS, THRESHOLD, STEP)
    res_wf["model"] = "Voting_Ensemble"

    # ── 5. Guardar resultados ─────────────────────────────────────────────────
    ensemble_rows = [res_ens_test, res_ens_train, res_wf]
    save_results(ensemble_rows, OUTPUT_CSV)

    # ── 6. Gráfico comparativo ────────────────────────────────────────────────
    # Construir DataFrame con ensemble + modelos base para el gráfico
    plot_rows = []
    for r in base_results:
        plot_rows.append({
            "model":    r["model"],
            "label":    "baseline_0.50",
            "accuracy": r["accuracy"],
            "roc_auc":  r["roc_auc"],
            "f1_weighted": r["f1_weighted"],
        })
    plot_rows.append({
        "model":    "Voting_Ensemble",
        "label":    "baseline_0.50",
        "accuracy": res_ens_test["accuracy"],
        "roc_auc":  res_ens_test["roc_auc"],
        "f1_weighted": res_ens_test["f1_weighted"],
    })
    plot_df = pd.DataFrame(plot_rows)
    plot_comparison(plot_df, OUTPUT_PLOT)

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Voting Ensemble — Test:")
    print(f"    Accuracy:    {res_ens_test['accuracy']:.4f}")
    print(f"    ROC-AUC:     {res_ens_test['roc_auc']:.4f}")
    print(f"    F1 Weighted: {res_ens_test['f1_weighted']:.4f}")
    print(f"  Voting Ensemble — Walk-Forward:")
    print(f"    Accuracy:    {res_wf['accuracy']:.4f}")
    print(f"    ROC-AUC:     {res_wf['roc_auc']:.4f}")
    print(f"    F1 Weighted: {res_wf['f1_weighted']:.4f}")
    print(f"\nCSV: {OUTPUT_CSV}")
    print(f"Plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
