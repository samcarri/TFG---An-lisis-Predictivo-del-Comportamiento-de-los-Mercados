# Análisis de Modelos YFinance - Integración al Sistema Multiagente

## Resumen Ejecutivo

Análisis del proyecto `/Users/samuel/Desktop/TFG---Stonks/Modulo2_YFinance` para integrar modelos LSTM entrenados en el sistema multiagente.

## 1. Modelos Disponibles

### Modelo 1: FULL_Features_LSTM_w20_p40_0_f30
- **Horizonte**: 30 días
- **Window Size**: 20 días
- **Split Percentile**: 40%
- **Accuracy**: 55.19%
- **Features**: 30 features (OHLCV + indicadores técnicos + lags)

### Modelo 2: FULL_Features_LSTM_top10_w20_p47_0_f1
- **Horizonte**: 1 día
- **Window Size**: 20 días
- **Split Percentile**: 47%
- **Accuracy**: 51.71%
- **Features**: Top 10 features seleccionadas

### Modelo 3: FULL_Features_LSTM_top10_w25_p50_0_f7
- **Horizonte**: 7 días
- **Window Size**: 25 días
- **Split Percentile**: 50%
- **Accuracy**: 49.54%
- **Features**: Top 10 features seleccionadas

## 2. Arquitectura del Modelo

### Clase LSTM (PyTorch)
```python
class LSTMClassifier(nn.Module):
    - input_size: número de features
    - hidden_size: 128 (LSTM_UNITS)
    - num_layers: 2
    - dropout: 0.3
    - num_classes: 2 (binario: bajada/subida)
```

### Configuración de Entrenamiento
- Epochs: 80
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss con class weights
- Early Stopping: patience=8, min_delta=0.0005
- Validation Split: 15%

## 3. Features Requeridas

### Features Base (OHLCV)
1. **Open**: Precio apertura
2. **High**: Precio máximo
3. **Low**: Precio mínimo
4. **Close**: Precio cierre
5. **Volume**: Volumen negociado

### Indicadores Técnicos Calculados
6. **return**: Retorno diario = (Close - Close_prev) / Close_prev
7. **ATR**: Average True Range (ventana 20 días)
8. **volatility**: Desviación estándar de retornos (ventana 20 días)
9. **volume_change**: Cambio porcentual de volumen
10. **RSI**: Relative Strength Index (ventana 20 días)

### Features Lag (para window_size=20)
11-30. **return_lag_1 a return_lag_20**: Retornos históricos

### Features Top10 (modelos optimizados)
Para modelos top10, las features más importantes son:
1. return
2. Open
3. High
4. Low/volume_change (según modelo)
5. Volume
6. Close
7. ATR
8. volatility
9. RSI
10. volume_change/ATR (según modelo)

## 4. Proceso de Generación de Datos

### Paso 1: Descarga de Datos
```python
data = yf.download(ticker='NVDA', start='2010-01-01', end=today)
# Retorna: Date, Open, High, Low, Close, Volume
```

### Paso 2: Cálculo de Features
```python
def add_features(data, window=20):
    # Retorno diario
    data['return'] = data['Close'].pct_change()
    
    # ATR
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = max(high_low, high_close, low_close)
    data['ATR'] = tr.rolling(window=20).mean()
    
    # Volatilidad
    data['volatility'] = data['return'].rolling(window=20).std()
    
    # Cambio de volumen
    data['volume_change'] = data['Volume'].pct_change()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=20).mean()
    avg_loss = loss.rolling(window=20).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Lags
    for lag in range(1, window + 1):
        data[f'return_lag_{lag}'] = data['return'].shift(lag)
    
    return data
```

### Paso 3: Crear Target (FutureReturn)
```python
def create_future_return(data, forecast_days=7):
    future_close = data['Close'].shift(-forecast_days)
    data['FutureReturn'] = (future_close - data['Close']) / data['Close']
    return data
```

### Paso 4: Crear Ventanas Deslizantes
```python
def create_windows(data, feature_cols, window_size):
    # Input: (n_dias, n_features)
    # Output: X (n_samples, window_size, n_features)
    #         y (n_samples,)
    
    for i in range(len(data) - window_size):
        X.append(features[i:i+window_size])
        y.append(targets[i+window_size])
    
    return np.array(X), np.array(y)
```

### Paso 5: Normalización
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Reshape: (samples, window, features) -> (samples*window, features)
X_2d = X.reshape(-1, num_features)
X_scaled_2d = scaler.fit_transform(X_2d)
# Reshape back: -> (samples, window, features)
X_scaled = X_scaled_2d.reshape(samples, window, features)
```

### Paso 6: Clasificación Binaria
```python
def assign_binary_percentile_labels(values, split_pct):
    # split_pct = 50 -> mediana
    # split_pct = 40 -> percentil 40
    split_thr = np.percentile(values, split_pct)
    labels = np.zeros(len(values))
    labels[values > split_thr] = 1  # 1 = Subida, 0 = Bajada
    return labels
```

## 5. Formato de Archivos Guardados

### Modelo (.joblib)
```python
{
    'state_dict': model.state_dict(),  # Pesos del modelo PyTorch
    'metadata': {
        'model_type': 'LSTMClassifier',
        'experiment_name': 'FULL (Features)_LSTM_top10',
        'ticker': 'NVDA',
        'window_size': 25,
        'forecast_days': 7,
        'split_pct': 50,
        'input_size': 10,  # número de features
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3,
        'best_fold': 9,
        'best_fold_accuracy': 0.6185,
        'created_at': '2026-04-20T20:28:19'
    }
}
```

### Scaler (.joblib)
```python
MinMaxScaler(feature_range=(0, 1))
# Contiene: data_min_, data_max_, data_range_, scale_
```

## 6. Proceso de Inferencia

### Paso 1: Cargar Modelo y Scaler
```python
import joblib
import torch

# Cargar artefactos
artifact = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Reconstruir modelo
metadata = artifact['metadata']
model = LSTMClassifier(
    input_size=metadata['input_size'],
    hidden_size=metadata['hidden_size'],
    num_layers=metadata['num_layers'],
    dropout=metadata['dropout'],
    num_classes=metadata['num_classes']
)
model.load_state_dict(artifact['state_dict'])
model.eval()
```

### Paso 2: Preparar Datos de Entrada
```python
# Necesitas: últimos window_size días de datos
# Con todas las features calculadas

# 1. Descargar datos recientes
data = yf.download('NVDA', start=fecha_inicio, end=fecha_fin)

# 2. Calcular features
data = add_features(data, window=window_size)

# 3. Seleccionar últimos window_size días
recent_data = data[selected_features].tail(window_size)

# 4. Convertir a array
X = recent_data.values  # Shape: (window_size, n_features)
X = X.reshape(1, window_size, n_features)  # Add batch dimension

# 5. Normalizar
X_2d = X.reshape(-1, n_features)
X_scaled_2d = scaler.transform(X_2d)
X_scaled = X_scaled_2d.reshape(1, window_size, n_features)

# 6. Convertir a tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
```

### Paso 3: Hacer Predicción
```python
with torch.no_grad():
    outputs = model(X_tensor)  # Shape: (1, 2)
    probabilities = torch.softmax(outputs, dim=1)
    prediction = torch.argmax(outputs, dim=1).item()
    
    prob_bajada = probabilities[0][0].item()
    prob_subida = probabilities[0][1].item()

# prediction: 0 = Bajada, 1 = Subida
# prob_subida: probabilidad de subida (0-1)
```

## 7. Requisitos de Datos Mínimos

Para hacer una predicción necesitas:

1. **Datos históricos**: Mínimo `window_size + 20` días
   - window_size días para la ventana de predicción
   - 20 días adicionales para calcular indicadores (ATR, RSI, etc.)

2. **Datos OHLCV**: Open, High, Low, Close, Volume para cada día

3. **Frecuencia**: Datos diarios (1d interval)

4. **Ticker**: Debe ser el mismo ticker usado en entrenamiento (NVDA)

## 8. Dependencias

```python
# Core
numpy
pandas
torch
joblib

# Data
yfinance

# ML
scikit-learn

# Opcional (para entrenamiento)
matplotlib
seaborn
```

## 9. Limitaciones y Consideraciones

### Limitaciones
1. **Ticker específico**: Modelos entrenados solo para NVDA
2. **Horizonte fijo**: Cada modelo predice un horizonte específico (1d, 7d, 30d)
3. **Clasificación binaria**: Solo predice dirección (subida/bajada), no magnitud
4. **Datos diarios**: No funciona con datos intraday

### Consideraciones
1. **Calidad de datos**: Requiere datos limpios sin gaps
2. **Normalización**: Debe usar el mismo scaler del entrenamiento
3. **Features exactas**: Debe calcular features de la misma manera
4. **Window size**: Debe respetar el window_size del modelo
5. **Percentil**: El split_pct define el umbral de clasificación

## 10. Integración Recomendada

### Opción 1: Módulo de Predicción Independiente
```python
# tools/sources/yfinance_predictor.py
class YFinancePredictor:
    def __init__(self, model_path, scaler_path):
        self.load_model(model_path, scaler_path)
    
    def predict(self, ticker, date=None):
        # 1. Descargar datos
        # 2. Calcular features
        # 3. Crear ventana
        # 4. Normalizar
        # 5. Predecir
        return prediction, probabilities
```

### Opción 2: Integración en Market Agent
```python
# agents/market_agent.py
class MarketAgent:
    def __init__(self):
        self.yf_predictor = YFinancePredictor(...)
    
    def analyze_with_ml(self, ticker):
        pred_1d = self.yf_predictor.predict(ticker, horizon='1d')
        pred_7d = self.yf_predictor.predict(ticker, horizon='7d')
        pred_30d = self.yf_predictor.predict(ticker, horizon='30d')
        return self.combine_predictions(pred_1d, pred_7d, pred_30d)
```

## 11. Compatibilidad con dataset_nvda_financial.csv

### ✅ Datos Disponibles en el CSV

El archivo `data/dataset_nvda_financial.csv` **YA CONTIENE** todas las features necesarias:

**Features Base (OHLCV):**
- ✅ Close
- ✅ High
- ✅ Low
- ✅ Volume
- ⚠️ Open (NO presente, pero puede derivarse o usar Close anterior)

**Indicadores Técnicos:**
- ✅ log_return (equivalente a 'return')
- ✅ volatility_7d (volatilidad a 7 días)
- ✅ rsi_14 (RSI a 14 días, similar a RSI a 20)
- ✅ volume_change
- ⚠️ ATR (NO presente directamente)

**Features Adicionales Disponibles:**
- ✅ ema_7, ema_14 (medias móviles exponenciales)
- ✅ macd, macd_signal, macd_diff
- ✅ bb_width, bb_position (Bandas de Bollinger)
- ✅ momentum_5, momentum_10
- ✅ log_return_lag_1 a log_return_lag_5 (lags históricos)

**Target:**
- ✅ target (clasificación binaria 0/1)
- ✅ date (índice temporal)

### ⚠️ Features Faltantes

Para usar los modelos LSTM necesitas calcular:

1. **Open**: Puede aproximarse con Close del día anterior o interpolarse
2. **ATR**: Debe calcularse manualmente (fórmula en sección 4)
3. **Lags adicionales**: El CSV tiene lags 1-5, pero modelos necesitan hasta lag 20-25

### 🔄 Estrategia de Adaptación

**Opción A: Usar Features Disponibles (Recomendado)**
```python
# Mapeo de features
feature_mapping = {
    'return': 'log_return',
    'volatility': 'volatility_7d',
    'RSI': 'rsi_14',
    'volume_change': 'volume_change',
    'Close': 'Close',
    'High': 'High',
    'Low': 'Low',
    'Volume': 'Volume'
}

# Para lags, usar los disponibles (1-5) y calcular los restantes
for lag in range(1, 6):
    feature_mapping[f'return_lag_{lag}'] = f'log_return_lag_{lag}'

# Calcular lags 6-20 manualmente
for lag in range(6, 21):
    data[f'return_lag_{lag}'] = data['log_return'].shift(lag)
```

**Opción B: Calcular Features Faltantes**
```python
# Calcular Open (aproximación)
data['Open'] = data['Close'].shift(1)

# Calcular ATR
high_low = data['High'] - data['Low']
high_close = (data['High'] - data['Close'].shift()).abs()
low_close = (data['Low'] - data['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
data['ATR'] = tr.rolling(window=20).mean()
```

**Opción C: Re-entrenar con Features del CSV**
- Usar las features ya disponibles en el CSV
- Re-entrenar modelos con estas features específicas
- Aprovechar features adicionales (MACD, Bollinger, momentum)

### 💡 Ventajas del CSV Actual

1. **Más features**: Tiene indicadores adicionales (MACD, Bollinger, EMA)
2. **Pre-procesado**: Datos ya limpios y normalizados
3. **Target incluido**: Ya tiene clasificación binaria
4. **Lags parciales**: Tiene lags 1-5 calculados

### 📋 Recomendación Final

**Para integración rápida:**
1. Usar features disponibles en el CSV
2. Calcular solo ATR y lags adicionales (6-20)
3. Aproximar Open con Close anterior
4. Usar modelos top10 (requieren menos features)

**Para máxima precisión:**
1. Descargar datos frescos de yfinance
2. Calcular todas las features exactamente como en entrenamiento
3. Usar modelos con todas las features

## 12. Próximos Pasos

1. ✅ Copiar modelos a `entrenamiento_agentes/trained_models/window/`
2. ✅ Analizar compatibilidad con dataset_nvda_financial.csv
3. ⬜ Crear módulo `YFinancePredictor` con soporte para ambas fuentes
4. ⬜ Implementar cálculo de features faltantes
5. ⬜ Crear tests de inferencia con CSV existente
6. ⬜ Integrar en Market Agent
7. ⬜ Documentar API de predicción
8. ⬜ Crear ejemplos de uso
## 11. Compatibilidad con dataset_nvda_financial.csv

### ✅ Datos Disponibles en el CSV

El archivo `data/dataset_nvda_financial.csv` **YA CONTIENE** todas las features necesarias:

**Features Base (OHLCV):**
- ✅ Close
- ✅ High
- ✅ Low
- ✅ Volume
- ⚠️ Open (NO presente, pero puede derivarse o usar Close anterior)

**Indicadores Técnicos:**
- ✅ log_return (equivalente a 'return')
- ✅ volatility_7d (volatilidad a 7 días)
- ✅ rsi_14 (RSI a 14 días, similar a RSI a 20)
- ✅ volume_change
- ⚠️ ATR (NO presente directamente)

**Features Adicionales Disponibles:**
- ✅ ema_7, ema_14 (medias móviles exponenciales)
- ✅ macd, macd_signal, macd_diff
- ✅ bb_width, bb_position (Bandas de Bollinger)
- ✅ momentum_5, momentum_10
- ✅ log_return_lag_1 a log_return_lag_5 (lags históricos)

**Target:**
- ✅ target (clasificación binaria 0/1)
- ✅ date (índice temporal)

### ⚠️ Features Faltantes

Para usar los modelos LSTM necesitas calcular:

1. **Open**: Puede aproximarse con Close del día anterior o interpolarse
2. **ATR**: Debe calcularse manualmente (fórmula en sección 4)
3. **Lags adicionales**: El CSV tiene lags 1-5, pero modelos necesitan hasta lag 20-25

### 🔄 Estrategia de Adaptación

**Opción A: Usar Features Disponibles (Recomendado)**
```python
# Mapeo de features
feature_mapping = {
    'return': 'log_return',
    'volatility': 'volatility_7d',
    'RSI': 'rsi_14',
    'volume_change': 'volume_change',
    'Close': 'Close',
    'High': 'High',
    'Low': 'Low',
    'Volume': 'Volume'
}

# Para lags, usar los disponibles (1-5) y calcular los restantes
for lag in range(1, 6):
    feature_mapping[f'return_lag_{lag}'] = f'log_return_lag_{lag}'

# Calcular lags 6-20 manualmente
for lag in range(6, 21):
    data[f'return_lag_{lag}'] = data['log_return'].shift(lag)
```

**Opción B: Calcular Features Faltantes**
```python
# Calcular Open (aproximación)
data['Open'] = data['Close'].shift(1)

# Calcular ATR
high_low = data['High'] - data['Low']
high_close = (data['High'] - data['Close'].shift()).abs()
low_close = (data['Low'] - data['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
data['ATR'] = tr.rolling(window=20).mean()
```

**Opción C: Re-entrenar con Features del CSV**
- Usar las features ya disponibles en el CSV
- Re-entrenar modelos con estas features específicas
- Aprovechar features adicionales (MACD, Bollinger, momentum)

### 💡 Ventajas del CSV Actual

1. **Más features**: Tiene indicadores adicionales (MACD, Bollinger, EMA)
2. **Pre-procesado**: Datos ya limpios y normalizados
3. **Target incluido**: Ya tiene clasificación binaria
4. **Lags parciales**: Tiene lags 1-5 calculados

### 📋 Recomendación Final

**Para integración rápida:**
1. Usar features disponibles en el CSV
2. Calcular solo ATR y lags adicionales (6-20)
3. Aproximar Open con Close anterior
4. Usar modelos top10 (requieren menos features)

**Para máxima precisión:**
1. Descargar datos frescos de yfinance
2. Calcular todas las features exactamente como en entrenamiento
3. Usar modelos con todas las features

## 12. Próximos Pasos

1. ✅ Copiar modelos a `entrenamiento_agentes/trained_models/window/`
2. ✅ Analizar compatibilidad con dataset_nvda_financial.csv
3. ⬜ Crear módulo `YFinancePredictor` con soporte para ambas fuentes
4. ⬜ Implementar cálculo de features faltantes
5. ⬜ Crear tests de inferencia con CSV existente
6. ⬜ Integrar en Market Agent
7. ⬜ Documentar API de predicción
8. ⬜ Crear ejemplos de uso

## 12. Archivos Clave del Proyecto Origen

- `stock_predictor.py`: Script principal de entrenamiento
- `results/saved_models/models_manifest_latest.json`: Metadata de modelos
- `data/NVDA_*.csv`: Datasets procesados con features
- `batch_runs/`: Resultados de experimentos batch

---

**Fecha de análisis**: 21/04/2026
**Analizado por**: Cline AI Assistant
**Proyecto origen**: `/Users/samuel/Desktop/TFG---Stonks/Modulo2_YFinance`