"""
Feature Engineering - Creación de indicadores técnicos
Añade RSI, MACD, Bollinger Bands y otros indicadores al dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_rsi(data, window=14):
    """
    Calcula el Relative Strength Index (RSI)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calcula el MACD (Moving Average Convergence Divergence)
    """
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calcula las Bandas de Bollinger
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, rolling_mean, lower_band

def calculate_moving_averages(data):
    """
    Calcula medias móviles simples y exponenciales
    """
    sma_5 = data.rolling(window=5).mean()
    sma_10 = data.rolling(window=10).mean()
    sma_20 = data.rolling(window=20).mean()
    sma_50 = data.rolling(window=50).mean()
    
    ema_5 = data.ewm(span=5, adjust=False).mean()
    ema_10 = data.ewm(span=10, adjust=False).mean()
    ema_20 = data.ewm(span=20, adjust=False).mean()
    
    return sma_5, sma_10, sma_20, sma_50, ema_5, ema_10, ema_20

def calculate_momentum_indicators(df):
    """
    Calcula indicadores de momentum
    """
    # Rate of Change (ROC)
    roc_5 = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
    roc_10 = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Momentum
    momentum_5 = df['Close'] - df['Close'].shift(5)
    momentum_10 = df['Close'] - df['Close'].shift(10)
    
    return roc_5, roc_10, momentum_5, momentum_10

def calculate_volume_indicators(df):
    """
    Calcula indicadores basados en volumen
    """
    # Volume Moving Average
    volume_ma_5 = df['Volume'].rolling(window=5).mean()
    volume_ma_10 = df['Volume'].rolling(window=10).mean()
    
    # Volume Rate of Change
    volume_roc = ((df['Volume'] - df['Volume'].shift(5)) / df['Volume'].shift(5)) * 100
    
    # On-Balance Volume (OBV)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return volume_ma_5, volume_ma_10, volume_roc, obv

def add_technical_indicators(df):
    """
    Añade todos los indicadores técnicos al dataset
    """
    print("\n🔧 Calculando indicadores técnicos...")
    
    # RSI
    print("  - RSI (Relative Strength Index)")
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    print("  - MACD (Moving Average Convergence Divergence)")
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    print("  - Bollinger Bands")
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Moving Averages
    print("  - Moving Averages (SMA y EMA)")
    df['SMA_5'], df['SMA_10'], df['SMA_20'], df['SMA_50'], df['EMA_5'], df['EMA_10'], df['EMA_20'] = calculate_moving_averages(df['Close'])
    
    # Momentum Indicators
    print("  - Momentum Indicators")
    df['ROC_5'], df['ROC_10'], df['Momentum_5'], df['Momentum_10'] = calculate_momentum_indicators(df)
    
    # Volume Indicators
    print("  - Volume Indicators")
    df['Volume_MA_5'], df['Volume_MA_10'], df['Volume_ROC'], df['OBV'] = calculate_volume_indicators(df)
    
    # Price-based features
    print("  - Price-based features")
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Open'] * 100
    
    # Lag features (valores previos)
    print("  - Lag features")
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        df[f'Sentiment_Lag_{lag}'] = df['sentiment_score'].shift(lag)
    
    # Rolling statistics
    print("  - Rolling statistics")
    for window in [5, 10, 20]:
        df[f'Returns_Rolling_Mean_{window}'] = df['Returns'].rolling(window=window).mean()
        df[f'Returns_Rolling_Std_{window}'] = df['Returns'].rolling(window=window).std()
        df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
    
    return df

def main():
    """
    Ejecuta el feature engineering completo
    """
    print("\n" + "="*60)
    print("🔧 FEATURE ENGINEERING - INDICADORES TÉCNICOS")
    print("="*60)
    
    # Cargar dataset unificado
    print("\n📊 Cargando dataset unificado...")
    df = pd.read_csv('data/dataset_unified.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Dataset original: {df.shape}")
    print(f"Columnas originales: {len(df.columns)}")
    
    # Añadir indicadores técnicos
    df = add_technical_indicators(df)
    
    # Eliminar filas con NaN (debido a ventanas de cálculo)
    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)
    
    print(f"\n✅ Feature engineering completado")
    print(f"Dataset final: {df.shape}")
    print(f"Columnas totales: {len(df.columns)}")
    print(f"Nuevas features: {len(df.columns) - 11}")  # 11 columnas originales
    print(f"Filas eliminadas (NaN): {rows_removed}")
    
    # Guardar dataset con features
    output_path = 'data/dataset_with_features.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset guardado en: {output_path}")
    
    # Mostrar lista de todas las columnas
    print("\n📋 Lista completa de features:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Estadísticas básicas
    print("\n📊 Estadísticas del dataset final:")
    print(f"  Período: {df['Date'].min()} a {df['Date'].max()}")
    print(f"  Total de días: {len(df)}")
    print(f"  Valores nulos: {df.isnull().sum().sum()}")
    
    return df

if __name__ == "__main__":
    df = main()