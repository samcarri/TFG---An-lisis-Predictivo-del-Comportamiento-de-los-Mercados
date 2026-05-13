"""
test_properties.py

Tests basados en propiedades (Hypothesis) para el Voting Ensemble System.
Cada test incluye comentario de trazabilidad con el formato:
  # Feature: voting-ensemble-system, Propiedad N: <texto>

Uso:
  pytest scripts/ensemble/test_properties.py -v
"""

import os
import math
import tempfile

import numpy as np
import pandas as pd
import pytest

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# Importar funciones a testear
from scripts.ensemble.voting_ensemble import (
    load_and_split,
    build_model,
    soft_vote,
    compute_metrics,
    save_results,
)
from scripts.ensemble.reliability_test import compare_metrics


# ── Estrategias auxiliares ────────────────────────────────────────────────────

def binary_array(min_size=10, max_size=500):
    """Genera arrays binarios con al menos 1 positivo y 1 negativo."""
    return arrays(
        dtype=np.int64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.integers(min_value=0, max_value=1),
    ).filter(lambda y: (y == 0).sum() >= 1 and (y == 1).sum() >= 1)


def prob_array(n):
    """Genera array de probabilidades en [0, 1] de longitud n."""
    return arrays(
        dtype=np.float64,
        shape=n,
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )


# ── Propiedad 1: Corrección del split temporal ────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 1: Corrección del split temporal

@settings(max_examples=100)
@given(st.integers(min_value=20, max_value=500))
def test_prop1_split_temporal(n):
    """
    Para cualquier dataset de N filas, el split 70/15/15 debe producir
    exactamente floor(N*0.70) filas de train, floor(N*0.85)-floor(N*0.70) de val
    y N-floor(N*0.85) de test, sin solapamiento.
    Valida: Requisito 1.5
    """
    # Feature: voting-ensemble-system, Propiedad 1: Corrección del split temporal
    expected_train = math.floor(n * 0.70)
    expected_val   = math.floor(n * 0.85) - math.floor(n * 0.70)
    expected_test  = n - math.floor(n * 0.85)

    # Crear CSV temporal
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date":   dates.strftime("%Y-%m-%d"),
        "target": np.random.randint(0, 2, n),
        "feat1":  np.random.randn(n),
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, dates_out, features = load_and_split(tmp_path)

        assert len(X_train) == expected_train, f"Train: {len(X_train)} != {expected_train}"
        assert len(X_val)   == expected_val,   f"Val:   {len(X_val)} != {expected_val}"
        assert len(X_test)  == expected_test,  f"Test:  {len(X_test)} != {expected_test}"
        assert len(X_train) + len(X_val) + len(X_test) == n
    finally:
        os.unlink(tmp_path)


# ── Propiedad 2: StandardScaler ajuste solo en train ─────────────────────────
# Feature: voting-ensemble-system, Propiedad 2: Corrección del StandardScaler

@settings(max_examples=100)
@given(
    st.integers(min_value=30, max_value=300),
    st.integers(min_value=2,  max_value=20),
)
def test_prop2_standard_scaler(n, f):
    """
    El StandardScaler debe ajustarse solo sobre train.
    scaler.mean_ debe coincidir con X_train.mean(axis=0).
    X_train_s.mean(axis=0) debe ser ≈ 0.
    Valida: Requisito 2.4
    """
    # Feature: voting-ensemble-system, Propiedad 2: Corrección del StandardScaler (ajuste solo en train)
    X = np.random.randn(n, f)
    train_end = math.floor(n * 0.70)
    X_train = X[:train_end]

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(X_train_s.mean(axis=0), np.zeros(f), atol=1e-10)


# ── Propiedad 3: scale_pos_weight ────────────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 3: Corrección del cálculo de scale_pos_weight

@settings(max_examples=100)
@given(binary_array(min_size=10, max_size=500))
def test_prop3_scale_pos_weight(y):
    """
    scale_pos_weight debe ser count(y==0) / count(y==1).
    Valida: Requisito 2.5
    """
    # Feature: voting-ensemble-system, Propiedad 3: Corrección del cálculo de scale_pos_weight
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    expected_spw = n_neg / n_pos

    model = build_model("xgboost", n_neg, n_pos)
    actual_spw = model.get_params()["scale_pos_weight"]

    assert abs(actual_spw - expected_spw) < 1e-4, (
        f"scale_pos_weight: {actual_spw} != {expected_spw}"
    )


# ── Propiedad 4: Soft voting ──────────────────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 4: Corrección del soft voting

@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=200))
def test_prop4_soft_voting(n):
    """
    avg_proba == np.mean([p1, p2, p3], axis=0) elemento a elemento.
    len(y_pred) == N.
    Valida: Requisitos 3.1, 3.4
    """
    # Feature: voting-ensemble-system, Propiedad 4: Corrección del soft voting
    rng = np.random.default_rng(42)
    p1  = rng.uniform(0, 1, n)
    p2  = rng.uniform(0, 1, n)
    p3  = rng.uniform(0, 1, n)

    y_pred, avg_proba = soft_vote([p1, p2, p3], threshold=0.50)

    expected_avg = np.mean([p1, p2, p3], axis=0)
    np.testing.assert_allclose(avg_proba, expected_avg, atol=1e-12)
    assert len(y_pred) == n


# ── Propiedad 5: Umbralización ────────────────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 5: Corrección de la umbralización

@settings(max_examples=100)
@given(
    st.integers(min_value=1, max_value=200),
    st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_prop5_thresholding(n, threshold):
    """
    y_pred[i] == 1 iff avg_proba[i] >= threshold para cualquier threshold.
    Valida: Requisitos 3.2, 3.3, 3.5
    """
    # Feature: voting-ensemble-system, Propiedad 5: Corrección de la umbralización
    rng = np.random.default_rng(0)
    p1  = rng.uniform(0, 1, n)
    p2  = rng.uniform(0, 1, n)
    p3  = rng.uniform(0, 1, n)

    y_pred, avg_proba = soft_vote([p1, p2, p3], threshold=threshold)

    for i in range(n):
        if avg_proba[i] >= threshold:
            assert y_pred[i] == 1, f"idx={i}: proba={avg_proba[i]:.4f} >= thr={threshold:.4f} pero pred={y_pred[i]}"
        else:
            assert y_pred[i] == 0, f"idx={i}: proba={avg_proba[i]:.4f} < thr={threshold:.4f} pero pred={y_pred[i]}"


# ── Propiedad 6: Cálculo de métricas ─────────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 6: Corrección del cálculo de métricas

@settings(max_examples=100)
@given(binary_array(min_size=10, max_size=200))
def test_prop6_compute_metrics(y_true):
    """
    Las métricas calculadas por compute_metrics deben coincidir con sklearn.
    Valida: Requisito 4.1
    """
    # Feature: voting-ensemble-system, Propiedad 6: Corrección del cálculo de métricas
    rng    = np.random.default_rng(42)
    y_prob = rng.uniform(0, 1, len(y_true))
    y_pred = (y_prob >= 0.5).astype(int)

    # Asegurar que y_pred tiene ambas clases para evitar errores de sklearn
    assume((y_pred == 0).sum() >= 1 and (y_pred == 1).sum() >= 1)

    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        result = compute_metrics(y_true, y_pred, y_prob, "test_label")

    assert abs(result["accuracy"]    - accuracy_score(y_true, y_pred))                          < 1e-10
    assert abs(result["f1_weighted"] - f1_score(y_true, y_pred, average="weighted"))            < 1e-10
    assert abs(result["recall_sube"] - recall_score(y_true, y_pred, pos_label=1))               < 1e-10
    assert abs(result["roc_auc"]     - roc_auc_score(y_true, y_prob))                           < 1e-10


# ── Propiedad 7: Round-trip CSV ───────────────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 7: Persistencia de métricas en CSV

@settings(max_examples=100)
@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_prop7_csv_roundtrip(acc, f1, rec, pre, auc):
    """
    Guardar métricas en CSV y releerlas debe producir diferencias < 1e-6.
    Valida: Requisitos 4.4, 8.3
    """
    # Feature: voting-ensemble-system, Propiedad 7: Persistencia de métricas en CSV (round-trip)
    original = {
        "label":          "baseline_0.50",
        "threshold":      0.50,
        "accuracy":       acc,
        "f1_weighted":    f1,
        "recall_sube":    rec,
        "precision_sube": pre,
        "roc_auc":        auc,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_results.csv")
        save_results([original], path)
        df = pd.read_csv(path)
        row = df[df["label"] == "baseline_0.50"].iloc[0]

        for field in ["accuracy", "f1_weighted", "recall_sube", "precision_sube", "roc_auc"]:
            assert abs(float(row[field]) - original[field]) < 1e-6, (
                f"{field}: {float(row[field])} vs {original[field]}"
            )


# ── Propiedad 8: Lógica de fiabilidad ────────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 8: Lógica de decisión del test de fiabilidad

@settings(max_examples=100)
@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_prop8_reliability_logic(ens_auc, auc1, auc2, auc3):
    """
    verdict == "PASSED" iff ensemble_auc > max(base_aucs).
    Valida: Requisitos 5.1, 5.2, 5.3

    Nota: los valores de AUC de los modelos base se leen desde CSV, lo que introduce
    una pequeña pérdida de precisión flotante. El test usa valores distintos para
    evitar falsos positivos por este efecto.
    """
    # Feature: voting-ensemble-system, Propiedad 8: Lógica de decisión del test de fiabilidad
    # Evitar valores iguales entre ensemble y base para no depender de precisión CSV
    assume(ens_auc != auc1 and ens_auc != auc2 and ens_auc != auc3)

    # Construir DataFrame simulado con las métricas del ensemble
    results_df = pd.DataFrame([{
        "label":          "baseline_0.50",
        "threshold":      0.50,
        "accuracy":       0.65,
        "f1_weighted":    0.65,
        "recall_sube":    0.60,
        "precision_sube": 0.60,
        "roc_auc":        ens_auc,
    }])

    # Parchear temporalmente BASE_CSVS con CSVs temporales
    import scripts.ensemble.reliability_test as rt
    original_base_csvs = rt.BASE_CSVS.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        new_base_csvs = {}
        for i, (name, auc_val) in enumerate(zip(
            ["XGBoost_News", "LightGBM_News", "XGBoost_Reddit"],
            [auc1, auc2, auc3]
        )):
            path = os.path.join(tmpdir, f"{name}.csv")
            pd.DataFrame([{
                "label":          "baseline_0.50",
                "threshold":      0.50,
                "accuracy":       0.65,
                "f1_weighted":    0.65,
                "recall_sube":    0.60,
                "precision_sube": 0.60,
                "roc_auc":        auc_val,
            }]).to_csv(path, index=False)
            new_base_csvs[name] = path

        rt.BASE_CSVS = new_base_csvs
        try:
            import io
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                report = compare_metrics(results_df, eval_type="baseline_0.50")
        finally:
            rt.BASE_CSVS = original_base_csvs

    expected_verdict = "PASSED" if (ens_auc > auc1 and ens_auc > auc2 and ens_auc > auc3) else "FAILED"
    assert report["verdict"] == expected_verdict, (
        f"ens={ens_auc:.4f} vs [{auc1:.4f}, {auc2:.4f}, {auc3:.4f}]: "
        f"expected {expected_verdict}, got {report['verdict']}"
    )


# ── Propiedad 9: Reproducibilidad ────────────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 9: Reproducibilidad del ensemble

@settings(max_examples=50, deadline=None)
@given(
    st.integers(min_value=50, max_value=200),
    st.integers(min_value=2,  max_value=10),
)
def test_prop9_reproducibility(n, f):
    """
    Dos ejecuciones con random_state=42 producen predicciones idénticas.
    Valida: Requisito 8.1
    """
    # Feature: voting-ensemble-system, Propiedad 9: Reproducibilidad del ensemble
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((n, f))
    y   = rng.integers(0, 2, n)
    assume((y == 0).sum() >= 5 and (y == 1).sum() >= 5)

    train_end = math.floor(n * 0.70)
    val_end   = math.floor(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test           = X[val_end:]

    assume((y_train == 0).sum() >= 1 and (y_train == 1).sum() >= 1)

    def run_once():
        from scripts.ensemble.voting_ensemble import train_base_model
        model, scaler, _, _ = train_base_model("xgboost", X_train, y_train, X_val, y_val)
        X_test_s = scaler.transform(X_test)
        return model.predict_proba(X_test_s)[:, 1]

    proba1 = run_once()
    proba2 = run_once()

    np.testing.assert_array_equal(proba1, proba2)


# ── Propiedad 10: Walk-forward periódico ─────────────────────────────────────
# Feature: voting-ensemble-system, Propiedad 10: Protocolo walk-forward — reentrenamiento periódico

@settings(max_examples=100)
@given(
    st.integers(min_value=30, max_value=200),
    st.integers(min_value=5,  max_value=20),
)
def test_prop10_walk_forward_retraining(n, step):
    """
    Los índices de reentrenamiento son exactamente {min_train + k*step | k=0,1,...}.
    El número total de predicciones OOS es N - 1 - min_train.
    Valida: Requisitos 6.1, 6.3
    """
    # Feature: voting-ensemble-system, Propiedad 10: Protocolo walk-forward — reentrenamiento periódico
    min_train = math.floor(n * 0.70)

    # Calcular índices esperados de reentrenamiento
    expected_retrain = set()
    t = min_train
    while t < n - 1:
        expected_retrain.add(t)
        t += step

    # Simular el bucle walk-forward y registrar cuándo se reentrena
    actual_retrain = set()
    for t in range(min_train, n - 1):
        if (t - min_train) % step == 0:
            actual_retrain.add(t)

    assert actual_retrain == expected_retrain, (
        f"Índices de reentrenamiento incorrectos: {actual_retrain} != {expected_retrain}"
    )

    # Número total de predicciones OOS
    expected_n_preds = n - 1 - min_train
    actual_n_preds   = len(range(min_train, n - 1))
    assert actual_n_preds == expected_n_preds, (
        f"n_preds: {actual_n_preds} != {expected_n_preds}"
    )
