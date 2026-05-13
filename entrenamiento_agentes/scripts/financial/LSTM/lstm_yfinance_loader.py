#!/usr/bin/env python3
"""
lstm_yfinance_loader.py

Evaluador de modelos LSTM usando las funciones originales del proyecto YFinance.
Combina la precisión del script original con métricas de evaluación completas.

Output:
  scripts/financial/outputs/LSTM/yfinance_loader_results.csv
  scripts/financial/outputs/LSTM/yfinance_loader_plot.png
"""

import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score
)

# Importar funciones del proyecto original YFinance
YFINANCE_PROJECT_PATH = '/Users/samuel/Desktop/TFG---Stonks/Modulo2_YFinance'
sys.path.insert(0, YFINANCE_PROJECT_PATH)

try:
    from stock_predictor import add_features, create_windows, download_data
    print("✓ Funciones importadas del proyecto YFinance")
except ImportError as e:
    print(f"❌ Error importando del proyecto YFinance: {e}")
    print(f"   Verifica que existe: {YFINANCE_PROJECT_PATH}/stock_predictor.py")
    sys.exit(1)

# Configuración
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

MODELS_DIR = "trained_models/window"
MANIFEST_PATH = os.path.join(YFINANCE_PROJECT_PATH, "results/saved_models/models_manifest_latest.json")
OUTPUT_CSV = "scripts/financial/outputs/LSTM/yfinance_loader_results.csv"
OUTPUT_PLOT = "scripts/financial/outputs/LSTM/yfinance_loader_plot.png"

# Configuración de modelos
MODEL_CONFIGS = {
    "7d": {
        "name": "LSTM_YF_7d",
        "model_file": "7d/FULL_Features_LSTM_top10_w25_p50_0_f7.joblib",
        "scaler_file": "7d/FULL_Features_LSTM_top10_w25_p50_0_f7_scalers.joblib",
    },
    "30d": {
        "name": "LSTM_YF_30d",
        "model_file": "30d/FULL_Features_LSTM_w20_p40_0_f30.joblib",
        "scaler_file": "30d/FULL_Features_LSTM_w20_p40_0_f30_scalers.joblib",
    }
}


# ── Arquitectura LSTM ──────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Arquitectura LSTM del proyecto YFinance"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.3, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# ── Utilidades ─────────────────────────────────────────────────────────────────

def load_manifest():
    """Carga el manifest con información de los modelos"""
    if not os.path.exists(MANIFEST_PATH):
        print(f"⚠️  Manifest no encontrado: {MANIFEST_PATH}")
        return []
    
    with open(MANIFEST_PATH, 'r') as f:
        return json.load(f)


def get_selected_features_from_manifest(model_file):
    """Obtiene las features seleccionadas del manifest para un modelo específico"""
    manifest = load_manifest()
    model_name = os.path.basename(model_file).replace('.joblib', '')
    
    for exp in manifest:
        saved_model_path = exp.get('saved_model', {}).get('model_path', '')
        if model_name in saved_model_path:
            selected_features = exp.get('selected_features', [])
            print(f"  ✓ Features del manifest: {len(selected_features)} features")
            return selected_features
    
    print(f"  ⚠️  No se encontraron selected_features en manifest para {model_name}")
    return None


def load_yfinance_model(model_path, scaler_path, device='cpu'):
    """Carga modelo LSTM y scaler desde archivos joblib"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
    
    artifact = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = artifact['metadata']
    
    model = LSTMClassifier(
        input_size=metadata['input_size'],
        hidden_size=metadata['hidden_size'],
        num_layers=metadata['num_layers'],
        dropout=metadata['dropout'],
        num_classes=metadata['num_classes']
    )
    
    model.load_state_dict(artifact['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Modelo cargado: {metadata['experiment_name']}")
    print(f"  - Window size: {metadata['window_size']}")
    print(f"  - Forecast days: {metadata['forecast_days']}")
    print(f"  - Input features: {metadata['input_size']}")
    print(f"  - Best fold accuracy: {metadata['best_fold_accuracy']:.4f}")
    
    return model, scaler, metadata


def prepare_data_with_yfinance_functions(ticker, start_date, end_date, window_size, selected_features):
    """
    Prepara datos usando las funciones originales del proyecto YFinance.
    Garantiza que las features sean idénticas al entrenamiento.
    """
    print(f"📊 Descargando datos de {ticker} ({start_date} → {end_date})...")
    data = download_data(ticker, start_date, end_date)
    
    if len(data) < window_size:
        raise ValueError(f"Datos insuficientes. Se necesitan al menos {window_size} días.")
    
    print(f"🔧 Calculando features con add_features()...")
    data, feature_list = add_features(data, window_size)
    data = data.dropna()
    
    if len(data) == 0:
        raise ValueError("No hay datos después de calcular features.")
    
    # Verificar que todas las selected_features existen
    available_features = [f for f in selected_features if f in data.columns]
    missing = set(selected_features) - set(available_features)
    
    if missing:
        print(f"  ⚠️  Features faltantes: {missing}")
        raise ValueError(f"Faltan features críticas: {missing}")
    
    print(f"  ✓ Usando {len(available_features)} features")
    print(f"\nDataset: {data.shape}  |  {data.index[0].date()} → {data.index[-1].date()}")
    
    return data, available_features


def create_target_from_future_return(data, forecast_days, split_pct):
    """Crea target binario basado en FutureReturn y percentil"""
    # Calcular FutureReturn
    data['FutureReturn'] = data['Close'].pct_change(forecast_days).shift(-forecast_days)
    data = data.dropna(subset=['FutureReturn'])
    
    # Calcular umbral de percentil
    threshold = np.percentile(data['FutureReturn'], split_pct)
    
    # Crear target binario
    data['target'] = (data['FutureReturn'] > threshold).astype(int)
    
    print(f"  ✓ Target creado (percentil {split_pct}%: {threshold:.4f})")
    print(f"  ✓ Distribución: {dict(zip(*np.unique(data['target'], return_counts=True)))}")
    
    return data


def evaluate_model(y_true, y_pred, y_proba, label=""):
    """Calcula métricas de evaluación"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, pos_label=1)
    pre = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"[{label}] acc={acc:.4f}  auc={auc:.4f}  recall={rec:.4f}  prec={pre:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Baja", "Sube"]))
    
    return {
        "label": label,
        "accuracy": acc,
        "f1_weighted": f1,
        "recall_sube": rec,
        "precision_sube": pre,
        "roc_auc": auc,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    
    print("=" * 70)
    print("LSTM YFINANCE LOADER — Usando Funciones Originales")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    all_results = []
    
    # Evaluar cada modelo
    for horizon, config in MODEL_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Evaluando: {config['name']} (horizonte {horizon})")
        print(f"{'='*70}")
        
        model_path = os.path.join(MODELS_DIR, config['model_file'])
        scaler_path = os.path.join(MODELS_DIR, config['scaler_file'])
        
        try:
            # 1. Cargar modelo
            model, scaler, metadata = load_yfinance_model(model_path, scaler_path, device)
            
            # 2. Obtener selected_features del manifest
            selected_features = get_selected_features_from_manifest(config['model_file'])
            if not selected_features:
                print(f"  ⚠️  Saltando modelo (no hay selected_features)")
                continue
            
            # 3. Preparar datos con funciones originales
            data, features = prepare_data_with_yfinance_functions(
                ticker=metadata['ticker'],
                start_date="2019-01-01",
                end_date="2026-04-01",
                window_size=metadata['window_size'],
                selected_features=selected_features
            )
            
            # 4. Crear target
            data = create_target_from_future_return(
                data,
                forecast_days=metadata['forecast_days'],
                split_pct=metadata['split_pct']
            )
            
            # 5. Crear ventanas con función original
            print(f"📐 Creando ventanas (window_size={metadata['window_size']})...")
            X, y = create_windows(data, selected_features, metadata['window_size'], target_col='target')
            
            print(f"  ✓ X shape: {X.shape}, y shape: {y.shape}")
            
            # 6. Split temporal
            n = len(X)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            X_test, y_test = X[val_end:], y[val_end:]
            
            print(f"\nTest: {val_end} → {n} ({n-val_end} ventanas)")
            
            # 7. Normalizar
            print(f"⚖️  Normalizando con scaler entrenado...")
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            
            # 8. Predecir
            print(f"🔮 Realizando predicción...")
            X_tensor = torch.FloatTensor(X_test_scaled).to(device)
            
            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            y_proba = probabilities[:, 1].cpu().numpy()
            
            # 9. Evaluar
            result = evaluate_model(
                y_test, predictions, y_proba,
                label=f"{config['name']}_test"
            )
            result['model'] = config['name']
            result['horizon'] = horizon
            result['window_size'] = metadata['window_size']
            all_results.append(result)
            
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print(f"   Saltando modelo {config['name']}")
            continue
        except Exception as e:
            print(f"❌ Error procesando {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Guardar resultados
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ Resultados guardados: {OUTPUT_CSV}")
        
        # Gráfico comparativo
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Modelos LSTM YFinance - Comparación", fontsize=14, fontweight="bold")
        
        models = results_df['model'].tolist()
        x = np.arange(len(models))
        width = 0.6
        
        metrics = ['accuracy', 'roc_auc', 'f1_weighted']
        titles = ['Accuracy', 'ROC-AUC', 'F1 Weighted']
        
        for ax, metric, title in zip(axes, metrics, titles):
            vals = results_df[metric].tolist()
            bars = ax.bar(x, vals, width, color='#3498db', alpha=0.8, edgecolor='white')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=20, ha='right')
            ax.set_ylim(0.3, 0.8)
            ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                       f'{h:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {OUTPUT_PLOT}")
        
        # Resumen
        print(f"\n{'='*70}")
        print("RESUMEN")
        print(f"{'='*70}")
        print(results_df[['model', 'horizon', 'accuracy', 'roc_auc', 'f1_weighted']].to_string(index=False))
    else:
        print("\n⚠️  No se pudo evaluar ningún modelo")


if __name__ == "__main__":
    main()