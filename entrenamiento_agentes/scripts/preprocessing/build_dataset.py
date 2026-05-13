"""
Script para construir el dataset conjunto de NVIDIA
Combina datos financieros (yfinance) con datos de sentimiento de noticias
Siguiendo el plan de acción definido
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CONSTRUCCIÓN DEL DATASET NVDA + SENTIMIENTO")
print("=" * 60)

# ============================================================================
# FASE 2: EXTRACCIÓN DE DATOS
# ============================================================================

print("\n📥 FASE 2: Extracción de datos")
print("-" * 60)

# 2.1 Datos financieros (yfinance)
print("\n2.1 Descargando datos financieros de NVIDIA...")
nvda = yf.download("NVDA", start="2019-01-01", end="2026-12-31", progress=False)
print(f"✓ Datos financieros descargados: {len(nvda)} registros")
print(f"  Rango: {nvda.index.min()} a {nvda.index.max()}")

# 2.2 Datos de noticias (ya disponibles)
print("\n2.2 Cargando datos de sentimiento de noticias...")
news = pd.read_csv("data/nvidia_sentiment_2019_2026.csv")
print(f"✓ Datos de sentimiento cargados: {len(news)} registros")
print(f"  Columnas: {list(news.columns)}")

# ============================================================================
# FASE 3: LIMPIEZA Y ALINEACIÓN TEMPORAL
# ============================================================================

print("\n\n🧹 FASE 3: Limpieza y alineación temporal")
print("-" * 60)

# Preparar datos financieros
nvda = nvda.reset_index()
# yfinance ahora devuelve columnas con MultiIndex, necesitamos aplanarlas
if isinstance(nvda.columns, pd.MultiIndex):
    nvda.columns = nvda.columns.get_level_values(0)
print(f"  Columnas de mercado: {list(nvda.columns)}")
nvda['date'] = pd.to_datetime(nvda['Date']).dt.date

# Preparar datos de noticias
news['date'] = pd.to_datetime(news['date']).dt.date

print(f"\n✓ Fechas convertidas correctamente")
print(f"  Mercado: {nvda['date'].min()} a {nvda['date'].max()}")
print(f"  Noticias: {news['date'].min()} a {news['date'].max()}")

# Expandir mercado a días diarios (forward fill para fines de semana)
print("\n⏳ Expandiendo datos de mercado a días diarios...")
# Convertir date a datetime para poder usar asfreq
nvda['date_dt'] = pd.to_datetime(nvda['date'])
nvda_daily = nvda.set_index('date_dt')
nvda_daily = nvda_daily.asfreq('D')
nvda_daily = nvda_daily.fillna(method='ffill').reset_index()
# Convertir de vuelta a date para el merge
nvda_daily['date'] = nvda_daily['date_dt'].dt.date
nvda_daily = nvda_daily.drop('date_dt', axis=1)

print(f"✓ Datos expandidos: {len(nvda_daily)} días (incluyendo fines de semana)")

# Merge de datos
print("\n🔗 Combinando datos de mercado y sentimiento...")
df = pd.merge(news, nvda_daily, on='date', how='left')

print(f"✓ Dataset combinado: {len(df)} registros")
print(f"  Valores nulos en Close: {df['Close'].isna().sum()}")

# Eliminar filas sin datos de mercado
df = df.dropna(subset=['Close'])
print(f"✓ Después de eliminar nulos: {len(df)} registros")

# ============================================================================
# FASE 4: FEATURE ENGINEERING
# ============================================================================

print("\n\n🧠 FASE 4: Feature Engineering")
print("-" * 60)

# 4.1 Features Financieras (datos HASTA el final del día t)
print("\n4.1 Creando features financieras...")
print("  ✅ Features usan datos HASTA el final del día t")
print("  ✅ Target predice el día t+1")

# Returns (logarítmicos) - del día ACTUAL (t)
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
print("  ✓ log_return (día t)")

# Volatilidad (rolling std de 7 días) - calculada HASTA el día t
df['volatility_7d'] = df['log_return'].rolling(7).std()
print("  ✓ volatility_7d (hasta día t)")

# Tendencia (EMAs) - calculadas HASTA el día t
df['ema_7'] = df['Close'].ewm(span=7, adjust=False).mean()
df['ema_14'] = df['Close'].ewm(span=14, adjust=False).mean()
print("  ✓ ema_7, ema_14 (hasta día t)")

# Volumen - del día ACTUAL (t)
df['volume_change'] = df['Volume'].pct_change()
print("  ✓ volume_change (día t)")

# 4.1.1 Features técnicas avanzadas
print("\n4.1.1 Creando features técnicas avanzadas...")

# RSI (Relative Strength Index) - 14 días
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi_14'] = calculate_rsi(df['Close'], period=14)
print("  ✓ rsi_14")

# MACD (Moving Average Convergence Divergence)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_diff'] = df['macd'] - df['macd_signal']
print("  ✓ macd, macd_signal, macd_diff")

# Bollinger Bands
df['bb_middle'] = df['Close'].rolling(window=20).mean()
df['bb_std'] = df['Close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
print("  ✓ bollinger_bands (width, position)")

# Lag features (retornos pasados)
for lag in range(1, 6):
    df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)
print("  ✓ log_return_lag_1 a lag_5")

# Momentum features
df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
print("  ✓ momentum_5, momentum_10")

# 4.2 Features de Sentimiento (con shift para evitar data leakage)
print("\n4.2 Preparando features de sentimiento...")

# Verificar qué columnas de sentimiento tenemos
sentiment_cols = [col for col in df.columns if 'tone' in col.lower() or col == 'n_news']
print(f"  Columnas de sentimiento disponibles: {sentiment_cols}")

# Aplicar shift a las features de sentimiento (usar datos del día anterior)
if 'mean_tone' in df.columns:
    df['mean_tone_shifted'] = df['mean_tone'].shift(1)
    print("  ✓ mean_tone_shifted")

if 'weighted_tone' in df.columns:
    df['weighted_tone_shifted'] = df['weighted_tone'].shift(1)
    print("  ✓ weighted_tone_shifted")

if 'tone_momentum' in df.columns:
    df['tone_momentum_shifted'] = df['tone_momentum'].shift(1)
    print("  ✓ tone_momentum_shifted")

if 'n_news' in df.columns:
    df['n_news_shifted'] = df['n_news'].shift(1)
    print("  ✓ n_news_shifted")

# 4.3 Target (clasificación binaria: ¿sube mañana?)
print("\n4.3 Creando variable target...")
df['target'] = (df['log_return'].shift(-1) > 0).astype(int)
print(f"  ✓ target creado")
print(f"  Distribución: {df['target'].value_counts().to_dict()}")

# ============================================================================
# FASE 5: CONSTRUCCIÓN DEL CSV FINAL
# ============================================================================

print("\n\n📊 FASE 5: Construcción del CSV final")
print("-" * 60)

# Seleccionar features finales
features = [
    # Precio original (para referencia y cálculos adicionales)
    'Close',
    'High',
    'Low',
    'Volume',
    # Features básicas
    'log_return',
    'volatility_7d',
    'ema_7',
    'ema_14',
    'volume_change',
    # Features técnicas avanzadas
    'rsi_14',
    'macd',
    'macd_signal',
    'macd_diff',
    'bb_width',
    'bb_position',
    'log_return_lag_1',
    'log_return_lag_2',
    'log_return_lag_3',
    'log_return_lag_4',
    'log_return_lag_5',
    'momentum_5',
    'momentum_10',
    # Features de sentimiento
    'mean_tone_shifted',
    'weighted_tone_shifted',
    'tone_momentum_shifted',
    'n_news_shifted'
]

# Verificar qué features existen realmente
available_features = [f for f in features if f in df.columns]
missing_features = [f for f in features if f not in df.columns]

if missing_features:
    print(f"\n⚠️  Features no disponibles: {missing_features}")
    print(f"✓ Features disponibles: {available_features}")
    features = available_features

# Crear dataset final
df_final = df[features + ['target', 'date']].copy()

# Eliminar filas con valores nulos
print(f"\nAntes de dropna: {len(df_final)} registros")
df_final = df_final.dropna()
print(f"Después de dropna: {len(df_final)} registros")

# Estadísticas del dataset
print("\n📈 Estadísticas del dataset final:")
print(f"  Total de registros: {len(df_final)}")
print(f"  Rango temporal: {df_final['date'].min()} a {df_final['date'].max()}")
print(f"  Features: {len(features)}")
print(f"\n  Distribución del target:")
print(f"    Sube (1): {(df_final['target'] == 1).sum()} ({(df_final['target'] == 1).sum() / len(df_final) * 100:.1f}%)")
print(f"    Baja (0): {(df_final['target'] == 0).sum()} ({(df_final['target'] == 0).sum() / len(df_final) * 100:.1f}%)")

# Guardar dataset
output_path = "data/dataset_nvda_lstm.csv"
df_final.to_csv(output_path, index=False)
print(f"\n✅ Dataset guardado en: {output_path}")

# Mostrar primeras filas
print("\n📋 Primeras 5 filas del dataset:")
print(df_final.head())

# Mostrar información de las columnas
print("\n📊 Información de las columnas:")
print(df_final.info())

print("\n" + "=" * 60)
print("✅ CONSTRUCCIÓN DEL DATASET COMPLETADA")
print("=" * 60)