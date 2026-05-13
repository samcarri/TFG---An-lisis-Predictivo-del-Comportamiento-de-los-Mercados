"""
Market Agent Tools
Herramientas para el agente de análisis de mercado
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys

# Importar desde el mismo directorio tools
from .market_financial_data import FinancialDataCollector

try:
    from strands import tool
except ImportError:
    # Fallback si Strands no está disponible
    def tool(func):
        return func


@tool
def get_market_features(ticker: str, period: str = '3mo', end_date: Optional[str] = None) -> Dict:
    """
    Extrae features de mercado para el Market Agent.
    
    Args:
        ticker: Símbolo bursátil (ej: 'NVDA')
        period: Período de análisis ('1mo', '3mo', '6mo', '1y')
        end_date: Fecha final opcional en formato 'YYYY-MM-DD'. Si no se proporciona, usa hoy.
    
    Returns:
        Dict con features de mercado procesadas
    """
    collector = FinancialDataCollector()
    
    # Calcular fechas
    if end_date is None:
        end_date_dt = datetime.now()
    else:
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    end_date_str = end_date_dt.strftime('%Y-%m-%d')
    
    if period == '1mo':
        start_date = (end_date_dt - timedelta(days=30)).strftime('%Y-%m-%d')
    elif period == '3mo':
        start_date = (end_date_dt - timedelta(days=90)).strftime('%Y-%m-%d')
    elif period == '6mo':
        start_date = (end_date_dt - timedelta(days=180)).strftime('%Y-%m-%d')
    elif period == '1y':
        start_date = (end_date_dt - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        start_date = (end_date_dt - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Obtener datos de precio
    df = collector.collect_stock_data(ticker, start_date, end_date_str)
    
    if df.empty:
        return {
            'error': True,
            'message': f'No se pudieron obtener datos para {ticker}'
        }
    
    # Calcular indicadores técnicos
    latest = df.iloc[-1]
    
    # SMA 50 y 200
    sma_50 = df['close'].tail(50).mean() if len(df) >= 50 else df['close'].mean()
    sma_200 = df['close'].tail(200).mean() if len(df) >= 200 else df['close'].mean()
    
    # RSI
    rsi = calculate_rsi(df['close'])
    
    # MACD
    macd_data = calculate_macd(df['close'])
    
    # Volatilidad robusta (con manejo de outliers)
    returns = df['close'].pct_change().dropna()
    # Eliminar outliers extremos (> 3 desviaciones estándar)
    returns_clean = returns[np.abs(returns - returns.mean()) <= (3 * returns.std())]
    volatility = returns_clean.std() * np.sqrt(252)  # Anualizada
    
    # Tendencia
    trend = detect_trend(df['close'], sma_50, sma_200)
    
    # Volumen promedio
    avg_volume = df['volume'].mean()
    volume_trend = 'high' if latest['volume'] > avg_volume * 1.2 else 'normal' if latest['volume'] > avg_volume * 0.8 else 'low'
    
    # Returns en múltiples horizontes temporales
    returns_1d = float((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) >= 2 else 0.0
    returns_5d = float((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100) if len(df) >= 6 else 0.0
    returns_period = float((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
    
    # Obtener fecha de forma segura
    analysis_date = latest['date'] if isinstance(latest['date'], str) else str(latest['date'])
    
    features = {
        'ticker': ticker,
        'current_price': float(latest['close']),
        'returns': {
            '1d': returns_1d,
            '5d': returns_5d,
            'period': returns_period
        },
        'trend': trend,
        'rsi': float(rsi),
        'macd': macd_data,
        'sma_50': float(sma_50),
        'sma_200': float(sma_200),
        'volatility': float(volatility),
        'volume': int(latest['volume']),
        'avg_volume': float(avg_volume),
        'volume_trend': volume_trend,
        'date': analysis_date
    }
    
    return features


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calcula el RSI (Relative Strength Index) usando Wilder smoothing (estándar de mercado).
    
    Args:
        prices: Serie de precios
        period: Período para el cálculo (default: 14)
    
    Returns:
        Valor del RSI (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Valor neutral si no hay suficientes datos
    
    # Calcular cambios
    delta = prices.diff().dropna()
    
    # Separar ganancias y pérdidas
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Wilder smoothing (EMA con alpha = 1/period)
    # Primera media simple
    avg_gain = gains.iloc[:period].mean()
    avg_loss = losses.iloc[:period].mean()
    
    # Aplicar Wilder smoothing para el resto
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
    
    # Calcular RS y RSI
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi) if not pd.isna(rsi) else 50.0


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """
    Calcula el MACD (Moving Average Convergence Divergence) con detección de cruces.
    
    Args:
        prices: Serie de precios
        fast: Período EMA rápida (default: 12)
        slow: Período EMA lenta (default: 26)
        signal: Período línea de señal (default: 9)
    
    Returns:
        Dict con macd_line, signal_line, histogram y crossover
    """
    if len(prices) < slow + signal:
        return {
            'macd_line': 0.0,
            'signal_line': 0.0,
            'histogram': 0.0,
            'crossover': 'none'
        }
    
    # Calcular EMAs
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    # Detectar cruce reciente (últimos 2 períodos)
    crossover = 'none'
    if len(histogram) >= 2:
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # Cruce alcista: de negativo a positivo
        if prev_hist < 0 and current_hist > 0:
            crossover = 'bullish'
        # Cruce bajista: de positivo a negativo
        elif prev_hist > 0 and current_hist < 0:
            crossover = 'bearish'
    
    return {
        'macd_line': float(macd_line.iloc[-1]),
        'signal_line': float(signal_line.iloc[-1]),
        'histogram': float(histogram.iloc[-1]),
        'crossover': crossover
    }


def detect_trend(prices: pd.Series, sma_50: float, sma_200: float) -> str:
    """
    Detecta la tendencia del mercado con análisis de pendientes.
    
    Args:
        prices: Serie de precios
        sma_50: Media móvil de 50 períodos
        sma_200: Media móvil de 200 períodos
    
    Returns:
        'bullish', 'bearish', o 'sideways'
    """
    current_price = prices.iloc[-1]
    
    # Calcular pendiente de SMA 50 (últimos 10 días)
    if len(prices) >= 60:
        sma_50_series = prices.rolling(window=50).mean()
        recent_sma50 = sma_50_series.tail(10)
        slope_sma50 = (recent_sma50.iloc[-1] - recent_sma50.iloc[0]) / recent_sma50.iloc[0]
    else:
        slope_sma50 = 0
    
    # Golden Cross / Death Cross con confirmación de pendiente
    if current_price > sma_50 > sma_200:
        # Confirmar con pendiente positiva
        if slope_sma50 > 0.01:  # Pendiente > 1%
            return 'bullish'
        else:
            return 'sideways'  # Precio arriba pero sin momentum
    elif current_price < sma_50 < sma_200:
        # Confirmar con pendiente negativa
        if slope_sma50 < -0.01:  # Pendiente < -1%
            return 'bearish'
        else:
            return 'sideways'  # Precio abajo pero sin momentum bajista fuerte
    else:
        return 'sideways'


@tool
def detect_overbought_oversold(rsi: float) -> str:
    """
    Detecta condiciones de sobrecompra/sobreventa.
    
    Args:
        rsi: Valor del RSI
    
    Returns:
        'overbought', 'oversold', o 'neutral'
    """
    if rsi > 70:
        return 'overbought'
    elif rsi < 30:
        return 'oversold'
    else:
        return 'neutral'


@tool
def get_technical_analysis(ticker: str, period: str = '3mo') -> Dict:
    """
    Realiza análisis técnico completo de una acción.
    
    Args:
        ticker: Símbolo bursátil
        period: Período de análisis
    
    Returns:
        Dict con análisis técnico completo
    """
    features = get_market_features(ticker, period)
    
    if features.get('error'):
        return features
    
    # Análisis de RSI
    rsi_status = detect_overbought_oversold(features['rsi'])
    
    # Análisis de MACD
    macd = features['macd']
    macd_signal = 'bullish' if macd['histogram'] > 0 else 'bearish'
    
    # Señales de trading
    signals = []
    
    if features['trend'] == 'bullish':
        signals.append('Precio por encima de SMA 50 y SMA 200 (alcista)')
    elif features['trend'] == 'bearish':
        signals.append('Precio por debajo de SMA 50 y SMA 200 (bajista)')
    
    if rsi_status == 'overbought':
        signals.append(f'RSI en {features["rsi"]:.1f} - Zona de sobrecompra')
    elif rsi_status == 'oversold':
        signals.append(f'RSI en {features["rsi"]:.1f} - Zona de sobreventa')
    else:
        signals.append(f'RSI en {features["rsi"]:.1f} - Zona neutral')
    
    if macd_signal == 'bullish':
        signals.append('MACD muestra momentum positivo')
    else:
        signals.append('MACD muestra momentum negativo')
    
    if features['volume_trend'] == 'high':
        signals.append('Volumen por encima del promedio')
    elif features['volume_trend'] == 'low':
        signals.append('Volumen por debajo del promedio')
    
    analysis = {
        'ticker': ticker,
        'current_price': features['current_price'],
        'returns': features['returns'],
        'trend': features['trend'],
        'rsi': features['rsi'],
        'rsi_status': rsi_status,
        'macd_signal': macd_signal,
        'volatility': features['volatility'],
        'volume_trend': features['volume_trend'],
        'signals': signals,
        'interpretation': {
            'trend_strength': 'strong' if abs(features['returns']['period']) > 10 else 'moderate' if abs(features['returns']['period']) > 5 else 'weak',
            'momentum': macd_signal,
            'risk_level': 'high' if features['volatility'] > 0.4 else 'moderate' if features['volatility'] > 0.25 else 'low'
        }
    }
    
    return analysis


@tool
def compare_to_baseline(ticker: str, baseline_price: float) -> Dict:
    """
    Compara el precio actual con un precio base.
    
    Args:
        ticker: Símbolo bursátil
        baseline_price: Precio de referencia
    
    Returns:
        Dict con comparación
    """
    collector = FinancialDataCollector()
    current_price = collector.get_latest_price(ticker)
    
    if current_price is None:
        return {
            'error': True,
            'message': f'No se pudo obtener precio actual de {ticker}'
        }
    
    change = current_price - baseline_price
    change_pct = (change / baseline_price) * 100
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'baseline_price': baseline_price,
        'change': change,
        'change_pct': change_pct,
        'status': 'above' if change > 0 else 'below' if change < 0 else 'equal'
    }


# ============================================================================
# FUNCIONES HELPER PARA INTERPRETACIÓN
# ============================================================================

def _interpret_rsi(rsi: float) -> str:
    """Interpreta el valor del RSI"""
    if rsi > 70:
        return "overbought"
    elif rsi > 60:
        return "approaching_overbought"
    elif rsi < 30:
        return "oversold"
    elif rsi < 40:
        return "approaching_oversold"
    else:
        return "neutral"


def _interpret_volatility(vol: float) -> str:
    """Interpreta la volatilidad anualizada"""
    if vol > 0.4:
        return "very_high"
    elif vol > 0.25:
        return "high"
    elif vol > 0.15:
        return "moderate"
    else:
        return "low"


def _interpret_trend_strength(change_pct: float) -> str:
    """Interpreta la fuerza de la tendencia"""
    abs_change = abs(change_pct)
    if abs_change > 15:
        return "very_strong"
    elif abs_change > 10:
        return "strong"
    elif abs_change > 5:
        return "moderate"
    else:
        return "weak"


def _generate_key_signals(features: Dict) -> List[str]:
    """
    Genera lista de señales clave (máximo 5)
    
    Args:
        features: Dict con features de mercado
    
    Returns:
        Lista de señales en lenguaje natural
    """
    signals = []
    
    # Señal de precio (usar returns del período)
    change_pct = features['returns']['period']
    if abs(change_pct) > 5:
        direction = "subió" if change_pct > 0 else "cayó"
        signals.append(f"Precio {direction} {abs(change_pct):.1f}% en el período")
    
    # Señal de RSI
    rsi = features['rsi']
    rsi_status = _interpret_rsi(rsi)
    if rsi_status != "neutral":
        signals.append(f"RSI en {rsi:.1f} - {rsi_status.replace('_', ' ')}")
    
    # Señal de MACD con cruce si existe
    macd = features['macd']
    if macd.get('crossover') == 'bullish':
        signals.append("MACD cruce alcista (señal de compra)")
    elif macd.get('crossover') == 'bearish':
        signals.append("MACD cruce bajista (señal de venta)")
    else:
        macd_trend = "positivo" if macd['histogram'] > 0 else "negativo"
        signals.append(f"MACD con momentum {macd_trend}")
    
    # Señal de tendencia vs SMAs
    if features['trend'] == 'bullish':
        signals.append("Precio sobre SMA 50 y SMA 200 (alcista)")
    elif features['trend'] == 'bearish':
        signals.append("Precio bajo SMA 50 y SMA 200 (bajista)")
    
    # Señal de volumen
    vol_trend = features['volume_trend']
    if vol_trend != 'normal':
        signals.append(f"Volumen {vol_trend.replace('_', ' ')}")
    
    return signals[:5]  # Máximo 5 señales


# ============================================================================
# HERRAMIENTA PRINCIPAL: RESUMEN AGREGADO OPTIMIZADO
# ============================================================================

@tool
def get_market_summary(ticker: str, period: str = '3mo') -> Dict:
    """
    Obtiene resumen agregado y optimizado de datos de mercado.
    Diseñado para consumo eficiente por LLM - reduce tokens ~70%.
    
    Args:
        ticker: Símbolo bursátil (ej: 'NVDA', 'AAPL')
        period: Período de análisis ('1mo', '3mo', '6mo', '1y')
    
    Returns:
        Dict con estructura agregada y pre-procesada
    """
    # Obtener features completas
    features = get_market_features(ticker, period)
    
    if features.get('error'):
        return features
    
    # Construir resumen compacto y optimizado
    summary = {
        "ticker": ticker,
        "period": period,
        "data_points": len(features.get('raw_data', [])) if 'raw_data' in features else 0,
        
        # Resumen de precios con múltiples horizontes
        "price_summary": {
            "current": features['current_price'],
            "returns": features['returns'],  # Incluye 1d, 5d, period
            "trend_direction": features['trend'],
            "trend_strength": _interpret_trend_strength(features['returns']['period'])
        },
        
        # Indicadores técnicos pre-interpretados
        "technical_indicators": {
            "rsi": {
                "value": features['rsi'],
                "status": _interpret_rsi(features['rsi'])
            },
            "macd": {
                "histogram": features['macd']['histogram'],
                "crossover": features['macd']['crossover'],
                "trend": "bullish" if features['macd']['histogram'] > 0 else "bearish"
            },
            "moving_averages": {
                "sma_50": features['sma_50'],
                "sma_200": features['sma_200'],
                "price_vs_sma50_pct": ((features['current_price'] - features['sma_50']) / features['sma_50'] * 100),
                "price_vs_sma200_pct": ((features['current_price'] - features['sma_200']) / features['sma_200'] * 100)
            }
        },
        
        # Análisis de riesgo
        "risk_analysis": {
            "volatility": features['volatility'],
            "volatility_level": _interpret_volatility(features['volatility']),
            "risk_rating": "high" if features['volatility'] > 0.3 else "moderate" if features['volatility'] > 0.2 else "low"
        },
        
        # Volumen
        "volume_analysis": {
            "current": features['volume'],
            "average": features['avg_volume'],
            "trend": features['volume_trend']
        },
        
        # Señales clave (máximo 5, pre-generadas)
        "key_signals": _generate_key_signals(features),
        
        # Timestamp
        "analysis_date": features['date']
    }
    
    return summary
