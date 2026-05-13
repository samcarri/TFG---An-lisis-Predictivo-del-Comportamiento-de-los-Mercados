"""
MCP Debate System
MCP server orquestador — consolida señales de los tres MCP servers en un único dict.
Diseñado para alimentar el DebateSystem con ~300 tokens máximo.
"""

import logging
from typing import Dict, Optional

try:
    from strands import tool
except ImportError:
    def tool(func):
        return func

from ..time_window_utils import validate_time_window, normalize_date_range
from .mcp_market_agent import get_market_data
from .mcp_news_agent import get_news_sentiment_summary
from .mcp_reddit_agent import get_reddit_sentiment_summary

log = logging.getLogger(__name__)


@tool
def get_unified_signals(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = '3mo',
) -> Dict:
    """
    Consolida señales de mercado, noticias y Reddit en un único dict normalizado.
    Es la tool principal del DebateSystem — usar por defecto para no saturar el contexto.
    Si alguna fuente falla, la incluye como unavailable sin propagar la excepción.
    Output máximo ~300 tokens. Optimizado para qwen2.5:7b.

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, se deriva de period.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.
        period: Período de análisis para mercado ('1mo', '3mo', '6mo', '1y').

    Returns:
        Dict consolidado con market, news, reddit y campos analysis_start_date, analysis_end_date.
    """
    # Normalizar ventana temporal
    start_str, end_str = normalize_date_range(end_date=end_date, period=period)
    if start_date is not None:
        if not validate_time_window(start_date, end_str):
            return {
                'error': True,
                'message': f'Invalid time window: {start_date} > {end_str}',
                'analysis_start_date': start_date,
                'analysis_end_date': end_date,
            }
        start_str = start_date

    # --- Market signals ---
    try:
        market_raw = get_market_data(
            ticker=ticker, start_date=start_str, end_date=end_str, period=period
        )
        if market_raw.get('error'):
            market = {'available': False, 'error': market_raw.get('message', 'unknown')}
        else:
            market = {
                'available': True,
                'price': market_raw.get('current_price'),
                'trend': market_raw.get('trend'),
                'rsi': market_raw.get('rsi'),
                'rsi_status': market_raw.get('rsi_status'),
                'macd_histogram': market_raw.get('macd_histogram'),
                'macd_crossover': market_raw.get('macd_crossover'),
                'volatility': market_raw.get('volatility'),
                'return_period_pct': market_raw.get('return_period_pct'),
            }
    except Exception as e:
        log.error(f"get_unified_signals market error: {e}")
        market = {'available': False, 'error': str(e)}

    # --- News signals ---
    try:
        news_raw = get_news_sentiment_summary(
            ticker=ticker, start_date=start_str, end_date=end_str
        )
        if news_raw.get('error'):
            news = {'available': False, 'error': news_raw.get('message', 'unknown')}
        else:
            alpaca = news_raw.get('alpaca', {})
            gdelt = news_raw.get('gdelt', {})
            news = {
                'available': True,
                'alpaca_score': alpaca.get('sentiment_score'),
                'alpaca_articles': alpaca.get('total_articles', 0),
                'top_headlines': alpaca.get('top_headlines', []),
                'gdelt_score': gdelt.get('sentiment_mean'),
                'gdelt_trend': gdelt.get('sentiment_trend', 'unknown'),
                'gdelt_volume_spike': gdelt.get('volume_spike', False),
            }
    except Exception as e:
        log.error(f"get_unified_signals news error: {e}")
        news = {'available': False, 'error': str(e)}

    # --- Reddit signals ---
    try:
        reddit_raw = get_reddit_sentiment_summary(
            ticker=ticker, start_date=start_str, end_date=end_str
        )
        if reddit_raw.get('error'):
            reddit = {'available': False, 'error': reddit_raw.get('message', 'unknown')}
        else:
            reddit = {
                'available': True,
                'sentiment_score': reddit_raw.get('sentiment_score'),
                'trend': reddit_raw.get('trend'),
                'post_volume_avg': reddit_raw.get('post_volume_avg'),
                'total_posts': reddit_raw.get('total_posts'),
            }
    except Exception as e:
        log.error(f"get_unified_signals reddit error: {e}")
        reddit = {'available': False, 'error': str(e)}

    return {
        'ticker': ticker,
        'market': market,
        'news': news,
        'reddit': reddit,
        'analysis_start_date': start_str,
        'analysis_end_date': end_str,
    }


@tool
def validate_time_window_tool(start_date: str, end_date: str) -> Dict:
    """
    Valida y normaliza una ventana temporal.
    Wrapper de validate_time_window de time_window_utils.
    Output: ~20 tokens.

    Args:
        start_date: Fecha de inicio YYYY-MM-DD.
        end_date: Fecha de fin YYYY-MM-DD.

    Returns:
        Dict con is_valid, start_date, end_date y mensaje de error si aplica.
    """
    is_valid = validate_time_window(start_date, end_date)
    result: Dict = {
        'is_valid': is_valid,
        'start_date': start_date,
        'end_date': end_date,
    }
    if not is_valid:
        result['error'] = (
            f"Invalid time window: '{start_date}' to '{end_date}'. "
            "Both must be YYYY-MM-DD and start_date <= end_date."
        )
    return result
