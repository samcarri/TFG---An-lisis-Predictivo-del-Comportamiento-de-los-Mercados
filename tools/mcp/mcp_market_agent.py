"""
MCP Market Agent
MCP server del MarketAgent — agrupa todas las tools de análisis de mercado.
Cada función está decorada con @tool para ser invocable por el agente.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

try:
    from strands import tool
except ImportError:
    def tool(func):
        return func

from ..time_window_utils import validate_time_window, normalize_date_range, PERIOD_TO_DAYS
from ..sources.market_financial_data import FinancialDataCollector
from ..sources.market_technical_tools import calculate_rsi, calculate_macd, detect_trend, _interpret_rsi

log = logging.getLogger(__name__)

DATE_FORMAT = '%Y-%m-%d'


def _get_collector() -> FinancialDataCollector:
    return FinancialDataCollector()


def _adjust_to_last_trading_day(collector: FinancialDataCollector, end_date: str) -> tuple:
    """
    Si end_date no es día de trading, retrocede al último día de trading disponible.

    Returns:
        (adjusted_end_date, was_adjusted)
    """
    if collector.nyse is None:
        return end_date, False

    if collector.is_trading_day(end_date):
        return end_date, False

    # Buscar el último día de trading en los 14 días anteriores
    end_dt = datetime.strptime(end_date, DATE_FORMAT)
    lookback_start = (end_dt - timedelta(days=14)).strftime(DATE_FORMAT)

    try:
        schedule = collector.nyse.schedule(start_date=lookback_start, end_date=end_date)
        if not schedule.empty:
            adjusted = schedule.index[-1].strftime(DATE_FORMAT)
            return adjusted, True
    except Exception as e:
        log.warning(f"Error ajustando end_date: {e}")

    return end_date, False


@tool
def get_market_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = '3mo'
) -> Dict:
    """
    Obtiene precio, RSI, MACD, SMA, volatilidad y retornos para un ticker.
    Si end_date cae en día no laborable, retrocede automáticamente al último día de trading.
    Output optimizado para qwen2.5:7b (~80 tokens).

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, se deriva de period.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.
        period: Período ('1mo', '3mo', '6mo', '1y'). Usado si start_date es None.

    Returns:
        Dict con datos de mercado y campos analysis_start_date, analysis_end_date.
    """
    try:
        # Normalizar ventana temporal
        start_str, end_str = normalize_date_range(end_date=end_date, period=period)
        if start_date is not None:
            if not validate_time_window(start_date, end_str):
                return {'error': True, 'message': f'Invalid time window: {start_date} > {end_str}'}
            start_str = start_date

        collector = _get_collector()

        # Ajustar end_date si no es día de trading
        adjusted_end, was_adjusted = _adjust_to_last_trading_day(collector, end_str)

        print(f"📂 Buscando datos en caché para {ticker} ({start_str} → {adjusted_end})...")
        df = collector.collect_stock_data(ticker, start_str, adjusted_end)

        if df.empty:
            print(f"❌ Sin datos disponibles para {ticker} en este período")
            return {
                'error': True,
                'message': f'No data available for {ticker} between {start_str} and {adjusted_end}',
                'analysis_start_date': start_str,
                'analysis_end_date': adjusted_end,
            }

        print(f"✅ {len(df)} sesiones obtenidas — calculando indicadores técnicos...")

        import numpy as np
        latest = df.iloc[-1]
        sma_50 = float(df['close'].tail(50).mean()) if len(df) >= 50 else float(df['close'].mean())
        sma_200 = float(df['close'].tail(200).mean()) if len(df) >= 200 else float(df['close'].mean())
        rsi = calculate_rsi(df['close'])
        macd_data = calculate_macd(df['close'])
        returns = df['close'].pct_change().dropna()
        returns_clean = returns[abs(returns - returns.mean()) <= (3 * returns.std())]
        volatility = float(returns_clean.std() * np.sqrt(252))
        trend = detect_trend(df['close'], sma_50, sma_200)
        returns_1d = float((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) >= 2 else 0.0
        returns_period = float((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)

        result = {
            'ticker': ticker,
            'current_price': float(latest['close']),
            'trend': trend,
            'rsi': round(float(rsi), 2),
            'rsi_status': _interpret_rsi(float(rsi)),
            'macd_histogram': round(float(macd_data['histogram']), 4),
            'macd_crossover': macd_data['crossover'],
            'sma_50': round(sma_50, 2),
            'sma_200': round(sma_200, 2),
            'volatility': round(volatility, 4),
            'return_1d_pct': round(returns_1d, 2),
            'return_period_pct': round(returns_period, 2),
            'volume': int(latest['volume']),
            'analysis_start_date': start_str,
            'analysis_end_date': adjusted_end,
            '_steps': [
                f'{"⚡ Caché" if "caché" in str(df.shape) else "📊 Yahoo Finance"}: {len(df)} registros para {ticker}',
                f'🔢 RSI={round(float(rsi),1)}, MACD={round(float(macd_data["histogram"]),4)}, Tendencia={trend}',
            ],
        }

        if was_adjusted:
            result['adjusted_end_date'] = adjusted_end
            result['original_end_date'] = end_str

        return result

    except ValueError as e:
        return {'error': True, 'message': str(e)}
    except Exception as e:
        log.error(f"get_market_data error: {e}")
        return {'error': True, 'message': str(e)}


@tool
def get_technical_analysis(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = '3mo'
) -> Dict:
    """
    Análisis técnico completo con señales interpretadas (RSI, MACD, SMA, tendencia).
    Output optimizado para qwen2.5:7b (~100 tokens).

    Args:
        ticker: Símbolo bursátil.
        start_date: Fecha de inicio YYYY-MM-DD. Si None, se deriva de period.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.
        period: Período de análisis.

    Returns:
        Dict con análisis técnico y campos analysis_start_date, analysis_end_date.
    """
    data = get_market_data(ticker=ticker, start_date=start_date, end_date=end_date, period=period)

    if data.get('error'):
        return data

    signals = []
    if data['trend'] == 'bullish':
        signals.append('Price above SMA50 and SMA200 (bullish)')
    elif data['trend'] == 'bearish':
        signals.append('Price below SMA50 and SMA200 (bearish)')

    rsi_status = data['rsi_status']
    if rsi_status != 'neutral':
        signals.append(f"RSI {data['rsi']:.1f} — {rsi_status.replace('_', ' ')}")

    if data['macd_crossover'] == 'bullish':
        signals.append('MACD bullish crossover')
    elif data['macd_crossover'] == 'bearish':
        signals.append('MACD bearish crossover')
    else:
        momentum = 'positive' if data['macd_histogram'] > 0 else 'negative'
        signals.append(f'MACD momentum {momentum}')

    risk = 'high' if data['volatility'] > 0.4 else 'moderate' if data['volatility'] > 0.25 else 'low'

    return {
        'ticker': ticker,
        'trend': data['trend'],
        'rsi': data['rsi'],
        'rsi_status': rsi_status,
        'macd_signal': 'bullish' if data['macd_histogram'] > 0 else 'bearish',
        'macd_crossover': data['macd_crossover'],
        'volatility': data['volatility'],
        'risk_level': risk,
        'return_period_pct': data['return_period_pct'],
        'signals': signals[:5],
        'analysis_start_date': data['analysis_start_date'],
        'analysis_end_date': data['analysis_end_date'],
    }


@tool
def get_market_calendar_info(date: str) -> Dict:
    """
    Consulta si una fecha es día de trading en NYSE y cuál es el próximo/anterior día de trading.
    Output ~20 tokens.

    Args:
        date: Fecha en formato YYYY-MM-DD.

    Returns:
        Dict con is_trading_day, previous_trading_day, next_trading_day.
    """
    try:
        collector = _get_collector()

        if collector.nyse is None:
            return {
                'date': date,
                'is_trading_day': True,
                'note': 'NYSE calendar not available',
            }

        is_trading = collector.is_trading_day(date)

        date_dt = datetime.strptime(date, DATE_FORMAT)

        # Día de trading anterior
        prev_start = (date_dt - timedelta(days=10)).strftime(DATE_FORMAT)
        prev_end = (date_dt - timedelta(days=1)).strftime(DATE_FORMAT)
        prev_schedule = collector.nyse.schedule(start_date=prev_start, end_date=prev_end)
        previous_trading_day = prev_schedule.index[-1].strftime(DATE_FORMAT) if not prev_schedule.empty else None

        # Próximo día de trading
        next_start = (date_dt + timedelta(days=1)).strftime(DATE_FORMAT)
        next_end = (date_dt + timedelta(days=10)).strftime(DATE_FORMAT)
        next_schedule = collector.nyse.schedule(start_date=next_start, end_date=next_end)
        next_trading_day = next_schedule.index[0].strftime(DATE_FORMAT) if not next_schedule.empty else None

        return {
            'date': date,
            'is_trading_day': is_trading,
            'previous_trading_day': previous_trading_day,
            'next_trading_day': next_trading_day,
        }

    except ValueError:
        return {'error': True, 'message': f'Invalid date format: {date!r}. Expected YYYY-MM-DD.'}
    except Exception as e:
        log.error(f"get_market_calendar_info error: {e}")
        return {'error': True, 'message': str(e)}
