"""
output_parser.py — Validación y normalización de outputs de agentes.
Extrae JSON del texto libre, valida campos requeridos y devuelve fallbacks estructurados.
"""

import json
import re
from typing import Dict, Any

REQUIRED_FIELDS = {
    'market': ['signal', 'confidence', 'reasons'],
    'news':   ['sentiment', 'confidence', 'prediction'],
    'reddit': ['community_sentiment', 'sentiment_score', 'reliability'],
}

COMPRESS_FIELDS = {
    'market': ['signal', 'confidence', 'trend', 'rsi', 'reasons', 'risks'],
    'news':   ['sentiment', 'confidence', 'prediction', 'key_headlines', 'gdelt_trend'],
    'reddit': ['community_sentiment', 'sentiment_score', 'reliability', 'trend', 'post_volume_avg'],
}

_DEFAULTS: Dict[str, Any] = {
    'signal': 'NEUTRAL', 'sentiment': 'NEUTRAL', 'community_sentiment': 'NEUTRAL',
    'confidence': 0.5, 'sentiment_score': 0.0, 'reliability': 'LOW',
    'prediction': 'LATERAL', 'trend': 'stable', 'reasons': [], 'risks': [],
    'key_headlines': [], 'gdelt_trend': 'stable', 'rsi': None,
    'post_volume_avg': 0,
}


def parse_agent_output(text: str, agent_type: str) -> Dict[str, Any]:
    """
    Extrae el JSON del output del agente, valida campos mínimos y rellena defaults.
    Siempre devuelve un dict válido — nunca lanza excepción.
    """
    data = _extract_json(text)

    if data is None:
        return _fallback(agent_type, reason=f"No JSON parseable en output: {text[:120]}")

    # Rellenar campos faltantes con defaults
    for field in REQUIRED_FIELDS.get(agent_type, []):
        if field not in data:
            data[field] = _DEFAULTS.get(field)

    data['_raw'] = text
    data['_parsed_ok'] = True
    return data


def compress_for_context(parsed: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
    """Devuelve solo los campos esenciales para inyectar en el contexto del debate."""
    keep = COMPRESS_FIELDS.get(agent_type, list(parsed.keys()))
    return {k: parsed[k] for k in keep if k in parsed and not k.startswith('_')}


def _extract_json(text: str):
    for pattern in [r'```json\s*([\s\S]*?)```', r'(\{[\s\S]*\})']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _fallback(agent_type: str, reason: str = '') -> Dict[str, Any]:
    base: Dict[str, Any] = {'_parsed_ok': False, '_error': reason}
    specifics = {
        'market': {'signal': 'NEUTRAL', 'confidence': 0.0, 'reasons': [], 'risks': [],
                   'trend': 'stable', 'rsi': None},
        'news':   {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'prediction': 'LATERAL',
                   'key_headlines': [], 'gdelt_trend': 'stable'},
        'reddit': {'community_sentiment': 'NEUTRAL', 'sentiment_score': 0.0,
                   'reliability': 'NONE', 'trend': 'stable', 'post_volume_avg': 0,
                   'prediction': 'LATERAL'},
    }
    return {**base, **specifics.get(agent_type, {})}
