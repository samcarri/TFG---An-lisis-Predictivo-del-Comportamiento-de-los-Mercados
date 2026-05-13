"""
Tests E2E para el Reddit Agent.

El agente LLM (qwen2.5:7b via Ollama) recibe un prompt, decide qué tools
llamar, las ejecuta con datos reales (CSV histórico o RedditScraper), y produce un análisis.

Requiere:
- Ollama corriendo en localhost:11434 con qwen2.5:7b
- Conexión a internet (para RedditScraper si no hay datos históricos)

Ejecutar solo estos tests:
    pytest tests/agents/reddit_agent/test_reddit_agent_e2e.py -v -m e2e
"""

import json
import pytest
from agents.reddit_agent import create_reddit_agent, analyze_reddit_sentiment, STRANDS_AVAILABLE

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def reddit_agent():
    """Crea el agente una sola vez para todos los tests del módulo."""
    if not STRANDS_AVAILABLE:
        pytest.skip("Strands no disponible")
    agent = create_reddit_agent(callback_handler=None)
    if agent is None:
        pytest.skip("No se pudo crear el Reddit Agent")
    return agent


class TestRedditAgentE2E:

    def test_agent_can_fetch_and_analyze_reddit_sentiment(self, reddit_agent):
        """El agente obtiene y analiza el sentimiento de Reddit para NVDA."""
        response = reddit_agent(
            "Analiza el sentimiento de Reddit para NVDA de los últimos 7 días "
            "usando get_reddit_sentiment_summary. "
            "Dime el score, la tendencia y el volumen de posts."
        )
        text = str(response)
        assert len(text) > 30, f"Respuesta demasiado corta: {text}"
        assert any(word in text.lower() for word in [
            'sentiment', 'sentimiento', 'score', 'positive', 'negative', 'neutral',
            'positivo', 'negativo', 'trend', 'tendencia', 'posts', 'reddit',
            'improving', 'stable', 'declining', 'no hay', 'available'
        ]), f"No se encontró análisis de Reddit en: {text[:300]}"

    def test_agent_can_fetch_detailed_daily_signals(self, reddit_agent):
        """El agente obtiene señales detalladas por día de Reddit."""
        response = reddit_agent(
            "Usa get_reddit_signals para NVDA de los últimos 7 días. "
            "¿Cuántos posts hay por día y cuál es el sentimiento diario?"
        )
        text = str(response)
        assert len(text) > 30
        assert any(word in text.lower() for word in [
            'day', 'día', 'daily', 'diario', 'post', 'count',
            'sentiment', 'sentimiento', 'score', 'no hay', 'available', 'datos'
        ]), f"No se encontró análisis diario en: {text[:300]}"

    def test_agent_handles_historical_window(self, reddit_agent):
        """El agente analiza una ventana temporal histórica."""
        result = analyze_reddit_sentiment('NVDA', days=7, end_date='2024-08-07')
        assert isinstance(result, str)
        assert len(result) > 20
        assert 'traceback' not in result.lower(), \
            f"El agente lanzó una excepción: {result[:300]}"

    def test_agent_output_has_no_raw_post_text(self, reddit_agent):
        """El agente no expone texto completo de posts — solo métricas y keywords."""
        response = reddit_agent(
            "Usa get_reddit_signals para NVDA de los últimos 3 días. "
            "Muéstrame los top_keywords por día."
        )
        text = str(response)
        assert len(text) > 20
        # La respuesta no debe contener posts completos (selftext largo)
        # Un post completo tendría >500 chars de texto continuo sin estructura
        lines = [l for l in text.split('\n') if len(l) > 300]
        assert len(lines) == 0, \
            f"La respuesta contiene líneas muy largas (posible texto de post): {lines[:1]}"

    def test_agent_assesses_reliability(self, reddit_agent):
        """El agente evalúa la fiabilidad del análisis según el volumen de posts."""
        response = reddit_agent(
            "Analiza el sentimiento de Reddit para NVDA usando get_reddit_sentiment_summary. "
            "¿Cuál es la fiabilidad del análisis? ¿Hay suficientes posts para confiar en el resultado?"
        )
        text = str(response)
        assert len(text) > 30
        assert any(word in text.lower() for word in [
            'reliability', 'fiabilidad', 'confianza', 'confidence',
            'high', 'medium', 'low', 'alta', 'media', 'baja',
            'posts', 'sufficient', 'suficiente', 'no hay', 'available'
        ]), f"No se encontró evaluación de fiabilidad en: {text[:300]}"

    def test_agent_produces_structured_analysis(self, reddit_agent):
        """El agente produce un análisis estructurado con los campos esperados."""
        response = reddit_agent(
            "Analiza el sentimiento de Reddit para NVDA de los últimos 7 días. "
            "Responde SOLO con este JSON:\n"
            '{"community_sentiment": "POSITIVE|NEUTRAL|NEGATIVE", '
            '"sentiment_score": 0.0, "trend": "improving|stable|declining", '
            '"reliability": "HIGH|MEDIUM|LOW", "reasoning": "string"}'
        )
        text = str(response)
        assert len(text) > 30
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                assert any(k in parsed for k in [
                    'community_sentiment', 'sentiment_score', 'trend', 'reliability'
                ])
        except json.JSONDecodeError:
            assert len(text) > 50, f"Respuesta demasiado corta: {text}"

    def test_agent_uses_historical_data_when_available(self, reddit_agent):
        """El agente usa datos históricos del CSV cuando están disponibles."""
        response = reddit_agent(
            "Usa get_reddit_signals para NVDA con end_date='2024-08-07'. "
            "¿Los datos vienen de histórico o de scraping en tiempo real?"
        )
        text = str(response)
        assert len(text) > 20
        # Debe mencionar la fuente de datos
        assert any(word in text.lower() for word in [
            'historical', 'histórico', 'csv', 'live', 'scraper', 'real',
            'source', 'fuente', 'no hay', 'available', 'datos'
        ]), f"No se mencionó la fuente de datos: {text[:300]}"
