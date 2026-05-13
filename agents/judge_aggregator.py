"""
judge_aggregator.py — Agregación determinista de señales de los 3 agentes.
Calcula ponderación, dirección y recomendación sin LLM.
El juez LLM solo añade el razonamiento narrativo sobre este resultado.
"""

import json
from typing import Dict, Any


def signal_to_score(signal: str) -> float:
    """Convierte una señal textual a score numérico [-1, 0, 1]."""
    s = str(signal).upper()
    if any(x in s for x in ['BULL', 'POSIT', 'SUBE', 'UP', 'COMPRAR']):
        return 1.0
    if any(x in s for x in ['BEAR', 'NEGAT', 'BAJA', 'DOWN', 'VENDER']):
        return -1.0
    return 0.0


def compute_structured_summary(
    market: Dict[str, Any],
    news: Dict[str, Any],
    reddit: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Agrega las señales de los 3 agentes con pesos dinámicos.
    - Si Reddit no tiene datos (reliability NONE/NO_DATA), su peso se redistribuye.
    - Completamente determinista — sin LLM.
    """
    m_score = signal_to_score(market.get('signal', ''))
    n_score = signal_to_score(news.get('sentiment', ''))
    r_score = signal_to_score(reddit.get('community_sentiment', ''))

    r_reliability = str(reddit.get('reliability', 'LOW')).upper()
    reddit_has_data = r_reliability not in ('NONE', 'NO_DATA', 'UNAVAILABLE')
    r_weight_base = 0.10 if reddit_has_data else 0.0

    # Pesos base: market 60%, news 30%, reddit 10%
    # Si Reddit no tiene datos, redistribuir su peso proporcionalmente
    total = 0.60 + 0.30 + r_weight_base
    w_m = round(0.60 / total, 4)
    w_n = round(0.30 / total, 4)
    w_r = round(r_weight_base / total, 4)

    # Scores ponderados por horizonte
    short_score  = m_score * 0.60 + n_score * 0.30 + r_score * r_weight_base
    medium_score = m_score * 0.40 + n_score * 0.40 + r_score * r_weight_base
    long_score   = m_score * 0.50 + n_score * 0.40 + r_score * r_weight_base

    # Confianza media ponderada
    m_conf = _safe_float(market.get('confidence'), 0.5)
    n_conf = _safe_float(news.get('confidence'), 0.5)
    r_conf = _safe_float(reddit.get('confidence'), 0.5) if reddit_has_data else 0.0
    avg_conf = round(m_conf * w_m + n_conf * w_n + r_conf * w_r, 3)

    # Alineación de consenso: diferencia entre señales
    signal_spread = abs(m_score - n_score) + abs(m_score - r_score) * w_r
    consensus_alignment = 'ALTO' if signal_spread < 0.4 else ('MEDIO' if signal_spread < 1.0 else 'BAJO')

    return {
        'short_term_direction':  _score_to_direction(short_score),
        'medium_term_direction': _score_to_direction(medium_score),
        'long_term_direction':   _score_to_direction(long_score),
        'recommendation':        _score_to_recommendation(short_score, avg_conf),
        'aggregate_confidence':  avg_conf,
        'worth_investing':       short_score > 0.1 and avg_conf > 0.5,
        'consensus_alignment':   consensus_alignment,
        'reddit_excluded':       not reddit_has_data,
        'scores': {
            'market': m_score,
            'news':   n_score,
            'reddit': r_score if reddit_has_data else None,
        },
        'weights': {'market': w_m, 'news': w_n, 'reddit': w_r},
        'raw_signals': {
            'market': market.get('signal', ''),
            'news':   news.get('sentiment', ''),
            'reddit': reddit.get('community_sentiment', '') if reddit_has_data else 'N/A',
        },
    }


def _score_to_direction(score: float) -> str:
    if score > 0.15:  return 'SUBE'
    if score < -0.15: return 'BAJA'
    return 'LATERAL'


def _score_to_recommendation(score: float, confidence: float) -> str:
    if score > 0.3 and confidence > 0.6:  return 'COMPRAR'
    if score < -0.3 and confidence > 0.6: return 'VENDER'
    if abs(score) < 0.15:                 return 'MANTENER'
    return 'ESPERAR'


def _safe_float(val, default: float) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default
