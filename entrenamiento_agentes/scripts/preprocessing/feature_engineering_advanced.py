#!/usr/bin/env python3
"""
Feature Engineering Avanzado para Predicción de Precios NVDA
Incluye indicadores técnicos completos, lag features y regime features
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula el Relative Strength Index (RSI)
    
    Args:
        prices: Serie de precios
        period: Período para el cálculo (default 14)
    
    Returns:
        Serie con valores RSI (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Serie de precios
        fast: Período EMA rápida
        slow: Período EMA lenta
        signal: Período línea de señal
    
    Returns:
        Tupla (MACD, Signal Line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, 
                               period: int = 20, 
                               std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula Bandas de Bollinger
    
    Args:
        prices: Serie de precios
        period: Período para media móvil
        std_dev: Número de desviaciones estándar
    
    Returns:
        Tupla (Upper Band, Middle Band, Lower Band)
    """
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_momentum(prices: pd.Series, periods: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Calcula momentum en múltiples horizontes temporales
    
    Args:
        prices: Serie de precios
        periods: Lista de períodos para calcular momentum
    
    Returns:
        DataFrame con columnas de momentum para cada período
    """
    momentum_df = pd.DataFrame(index=prices.index)
    
    for period in periods:
        momentum_df[f'momentum_{period}'] = prices.pct_change(periods=period)
    
    return momentum_df


def calculate_log_returns_lags(prices: pd.Series, lags: list = [1, 2, 5, 10]) -> pd.DataFrame:
    """
    Calcula retornos logarítmicos con diferentes lags
    
    Args:
        prices: Serie de precios
        lags: Lista de lags a calcular
    
    Returns:
        DataFrame con retornos logarítmicos para cada lag
    """
    log_returns = np.log(prices / prices.shift(1))
    
    lag_df = pd.DataFrame(index=prices.index)
    lag_df['log_return'] = log_returns
    
    for lag in lags:
        lag_df[f'log_return_lag_{lag}'] = log_returns.shift(lag)
    
    return lag_df


def calculate_regime_features(prices: pd.Series, 
                               short_window: int = 20,
                               long_window: int = 50) -> pd.DataFrame:
    """
    Calcula features de régimen de mercado
    
    Args:
        prices: Serie de precios
        short_window: Ventana corta para tendencia
        long_window: Ventana larga para volatilidad
    
    Returns:
        DataFrame con features de régimen
    """
    regime_df = pd.DataFrame(index=prices.index)
    
    # Tendencia: SMA corta vs SMA larga
    sma_short = prices.rolling(window=short_window).mean()
    sma_long = prices.rolling(window=long_window).mean()
    regime_df['trend_signal'] = (sma_short > sma_long).astype(int)
    regime_df['trend_strength'] = (sma_short - sma_long) / sma_long
    
    # Volatilidad rolling larga
    returns = prices.pct_change()
    regime_df['volatility_20d'] = returns.rolling(window=20).std()
    regime_df['volatility_50d'] = returns.rolling(window=50).std()
    regime_df['volatility_ratio'] = regime_df['volatility_20d'] / regime_df['volatility_50d']
    
    return regime_df


def calculate_sentiment_features(sentiment: pd.Series, 
                                  windows: list = [7, 14, 30]) -> pd.DataFrame:
    """
    Calcula features avanzadas de sentimiento
    
    Args:
        sentiment: Serie de sentimiento
        windows: Ventanas para rolling statistics
    
    Returns:
        DataFrame con features de sentimiento mejoradas
    """
    sent_df = pd.DataFrame(index=sentiment.index)
    sent_df['sentiment'] = sentiment
    
    for window in windows:
        # Rolling mean y std
        sent_df[f'sentiment_mean_{window}d'] = sentiment.rolling(window=window).mean()
        sent_df[f'sentiment_std_{window}d'] = sentiment.rolling(window=window).std()
        
        # Momentum del sentimiento
        sent_df[f'sentiment_momentum_{window}d'] = sentiment.diff(periods=window)
    
    # Diferencia vs media histórica (expanding)
    sent_df['sentiment_vs_historical'] = sentiment - sentiment.expanding().mean()
    
    # Z-score del sentimiento
    sent_df['sentiment_zscore'] = (sentiment - sentiment.rolling(window=30).mean()) / sentiment.rolling(window=30).std()
    
    return sent_df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade todas las features avanzadas al dataset
    
    Args:
        df: DataFrame con datos base (debe incluir 'Close' y 'sentiment')
    
    Returns:
        DataFrame con todas las features añadidas
    """
    print("🔧 Añadiendo features técnicas avanzadas...")
    
    df = df.copy()
    
    # 1. RSI
    print("  📊 Calculando RSI...")
    df['rsi_14'] = calculate_rsi(df['Close'], period=14)
    df['rsi_7'] = calculate_rsi(df['Close'], period=7)
    
    # 2. MACD
    print("  📊 Calculando MACD...")
    macd, signal, histogram = calculate_macd(df['Close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram
    
    # 3. Bollinger Bands
    print("  📊 Calculando Bollinger Bands...")
    upper, middle, lower = calculate_bollinger_bands(df['Close'])
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle
    df['bb_position'] = (df['Close'] - lower) / (upper - lower)
    
    # 4. Momentum múltiple
    print("  📊 Calculando Momentum...")
    momentum_df = calculate_momentum(df['Close'], periods=[5, 10, 20, 30])
    df = pd.concat([df, momentum_df], axis=1)
    
    # 5. Log returns con lags
    print("  📊 Calculando Log Returns y Lags...")
    lag_df = calculate_log_returns_lags(df['Close'], lags=[1, 2, 5, 10, 20])
    df = pd.concat([df, lag_df], axis=1)
    
    # 6. Regime features
    print("  📊 Calculando Regime Features...")
    regime_df = calculate_regime_features(df['Close'])
    df = pd.concat([df, regime_df], axis=1)
    
    # 7. Sentiment features mejoradas
    if 'sentiment' in df.columns:
        print("  📊 Calculando Sentiment Features avanzadas...")
        sentiment_df = calculate_sentiment_features(df['sentiment'])
        df = pd.concat([df, sentiment_df.drop('sentiment', axis=1)], axis=1)
    
    # 8. Features adicionales útiles
    print("  📊 Calculando features adicionales...")
    
    # Rango diario (High-Low)
    if 'High' in df.columns and 'Low' in df.columns:
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
    
    # Distancia a máximos/mínimos recientes
    df['distance_to_52w_high'] = (df['Close'].rolling(window=252).max() - df['Close']) / df['Close']
    df['distance_to_52w_low'] = (df['Close'] - df['Close'].rolling(window=252).min()) / df['Close']
    
    # Volumen relativo
    if 'Volume' in df.columns:
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    print(f"✅ Features añadidas. Total de columnas: {len(df.columns)}")
    
    return df


def prepare_dataset_for_lstm(input_file: str, output_file: str):
    """
    Prepara el dataset completo con todas las features avanzadas
    
    Args:
        input_file: Ruta al archivo CSV de entrada
        output_file: Ruta al archivo CSV de salida
    """
    print(f"📂 Cargando dataset desde {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"📊 Dataset original: {df.shape}")
    print(f"   Columnas: {list(df.columns)}")
    
    # Añadir features avanzadas
    df = add_advanced_features(df)
    
    # Eliminar filas con NaN (debido a rolling windows)
    initial_rows = len(df)
    df = df.dropna()
    print(f"🧹 Eliminadas {initial_rows - len(df)} filas con NaN")
    
    # Guardar dataset procesado
    df.to_csv(output_file, index=False)
    print(f"💾 Dataset guardado en {output_file}")
    print(f"📊 Dataset final: {df.shape}")
    print(f"   Features: {len(df.columns) - 1}")  # -1 por la columna Date
    
    # Mostrar resumen de features
    print("\n📋 Resumen de features por categoría:")
    
    technical = [col for col in df.columns if any(x in col for x in ['rsi', 'macd', 'bb_', 'momentum'])]
    print(f"   Técnicas: {len(technical)}")
    
    lag_features = [col for col in df.columns if 'lag' in col or 'log_return' in col]
    print(f"   Lag/Returns: {len(lag_features)}")
    
    regime = [col for col in df.columns if any(x in col for x in ['trend', 'volatility', 'regime'])]
    print(f"   Régimen: {len(regime)}")
    
    sentiment = [col for col in df.columns if 'sentiment' in col]
    print(f"   Sentimiento: {len(sentiment)}")
    
    return df


if __name__ == "__main__":
    # Configuración
    INPUT_FILE = "data/dataset_nvda_lstm.csv"
    OUTPUT_FILE = "data/dataset_nvda_advanced.csv"
    
    # Procesar dataset
    df = prepare_dataset_for_lstm(INPUT_FILE, OUTPUT_FILE)
    
    print("\n✅ Feature engineering avanzado completado!")
    print(f"📊 Dataset listo para entrenamiento: {OUTPUT_FILE}")