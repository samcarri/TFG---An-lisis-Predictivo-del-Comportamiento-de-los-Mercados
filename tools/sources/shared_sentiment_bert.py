"""Analizador de sentimiento de texto para posts de Reddit (FinBERT únicamente).

Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

MODEL_ID = "ProsusAI/finbert"

# ---------------------------------------------------------------------------
# Lazy-loaded pipeline cache (cargado una sola vez por proceso)
# ---------------------------------------------------------------------------
_FINBERT_PIPELINE: Any = None


def _get_finbert_pipeline():
    """Obtiene o crea el pipeline de FinBERT (lazy loading)."""
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        from transformers import pipeline
        _FINBERT_PIPELINE = pipeline(
            "text-classification",
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            top_k=None,
            truncation=True,
            max_length=512,  # truncación real se hace antes
        )
    return _FINBERT_PIPELINE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text(post: dict) -> str:
    """Concatena title + selftext; campos ausentes → ''."""
    title = str(post.get("title") or "")
    selftext = str(post.get("selftext") or "")
    return (title + " " + selftext).strip()


def _apply_finbert(text: str, max_length: int) -> dict:
    """Devuelve campos sent_finbert_* para un texto no vacío."""
    clf = _get_finbert_pipeline()
    # Truncamos el texto a max_length tokens usando el tokenizer del pipeline
    tokenizer = clf.tokenizer
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    truncated = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

    result = clf(truncated)[0]  # lista de {label, score}
    score_map = {d["label"].lower(): float(d["score"]) for d in result}
    pos = score_map.get("positive", 0.0)
    neg = score_map.get("negative", 0.0)
    neu = score_map.get("neutral", 0.0)
    label = max([("positive", pos), ("negative", neg), ("neutral", neu)], key=lambda x: x[1])[0]
    return {
        "sent_finbert_label": label,
        "sent_finbert_pos": pos,
        "sent_finbert_neg": neg,
        "sent_finbert_neu": neu,
    }


# ---------------------------------------------------------------------------
# Valores por defecto para texto vacío
# ---------------------------------------------------------------------------

_EMPTY_DEFAULTS = {
    "sent_finbert_label": "neutral",
    "sent_finbert_pos": 0.0,
    "sent_finbert_neg": 0.0,
    "sent_finbert_neu": 1.0,
}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def analyze_text_sentiment(post: dict, max_length: int = 256) -> dict:
    """Analiza el sentimiento del texto (title + selftext) del post con FinBERT.

    - Concatena title + selftext (campos ausentes = "")
    - Texto vacío → neutral con scores por defecto
    - Añade campos sent_finbert_label, sent_finbert_pos, sent_finbert_neg, sent_finbert_neu
    - Retorna el post actualizado (copia superficial con campos añadidos)

    Args:
        post: Dict con al menos los campos 'title' y 'selftext'.
        max_length: Número máximo de tokens para truncar el texto.

    Returns:
        Post actualizado con los campos de sentimiento añadidos.
    """
    result = dict(post)
    text = _build_text(post)

    if not text:
        result.update(_EMPTY_DEFAULTS)
        return result

    try:
        result.update(_apply_finbert(text, max_length))
    except Exception as exc:
        logger.error(
            "[text_analyzer] post_id=%s: %s",
            post.get("id", "?"),
            exc,
        )
        result.update(_EMPTY_DEFAULTS)

    return result


def analyze_batch(posts: list[dict], max_length: int = 256) -> list[dict]:
    """Analiza una lista de posts en batch para mayor eficiencia.

    Args:
        posts: Lista de posts como dicts.
        max_length: Número máximo de tokens.

    Returns:
        Lista de posts actualizados con campos de sentimiento.
    """
    return [analyze_text_sentiment(post, max_length) for post in posts]


# ---------------------------------------------------------------------------
# Compatibilidad con API anterior
# ---------------------------------------------------------------------------

def analyze_sentiment_batch(texts: list[str]) -> list[tuple[float, str]]:
    """
    Analiza una lista de textos con FinBERT (compatibilidad con API anterior).

    Returns:
        Lista de tuplas (probability_of_winner, label)
        donde label es 'positive' | 'neutral' | 'negative'
    """
    results = []
    for text in texts:
        post = {"title": text, "selftext": ""}
        analyzed = analyze_text_sentiment(post)
        
        label = analyzed["sent_finbert_label"]
        # Obtener la probabilidad del label ganador
        if label == "positive":
            prob = analyzed["sent_finbert_pos"]
        elif label == "negative":
            prob = analyzed["sent_finbert_neg"]
        else:
            prob = analyzed["sent_finbert_neu"]
        
        results.append((prob, label))
    
    return results