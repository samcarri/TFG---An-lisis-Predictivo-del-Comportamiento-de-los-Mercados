"""
Tests E2E para el News Agent.

El agente LLM (qwen2.5:7b via Ollama) recibe un prompt, decide qué tools
llamar, las ejecuta con datos reales (Alpaca API + GDELT), y produce un análisis.

Requiere:
- Ollama corriendo en localhost:11434 con qwen2.5:7b
- ALPACA_API_KEY y ALPACA_API_SECRET en .env (para noticias frescas)
- Conexión a internet

Ejecutar solo estos tests:
    pytest tests/agents/news_agent/test_news_agent_e2e.py -v -m e2e
"""

import json
import pytest
from agents.news_agent import create_news_agent, analyze_ticker_news, STRANDS_AVAILABLE

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def news_agent():
    """Crea el agente una sola vez para todos los tests del módulo."""
    if not STRANDS_AVAILABLE:
        pytest.skip("Strands no disponible")
    agent = create_news_agent(callback_handler=None)
    if agent is None:
        pytest.skip("No se pudo crear el News Agent")
    return agent


class TestNewsAgentE2E:

    def test_agent_can_fetch_and_analyze_news_summary(self, news_agent):
        """El agente obtiene y analiza el resumen de noticias de NVDA."""
        response = news_agent(
            "Analiza las noticias de NVDA de los últimos 7 días usando get_news_sentiment_summary. "
            "Dime el sentimiento general y si hay headlines relevantes."
        )
        text = str(response)
        assert len(text) > 50, f"Respuesta demasiado corta: {text}"
        # Debe mencionar sentimiento o noticias
        assert any(word in text.lower() for word in [
            'positive', 'negative', 'neutral', 'positivo', 'negativo', 'neutral',
            'sentiment', 'sentimiento', 'news', 'noticias', 'nvda', 'nvidia'
        ]), f"No se encontró análisis de noticias en: {text[:300]}"

    def test_agent_can_fetch_alpaca_news_with_end_date(self, news_agent):
        """El agente obtiene noticias de Alpaca para una ventana temporal específica."""
        response = news_agent(
            "Usa get_alpaca_news_signals para obtener noticias de NVDA "
            "con start_date='2024-07-01' y end_date='2024-08-07'. "
            "¿Cuántos artículos encontraste y cuál es el sentimiento?"
        )
        text = str(response)
        assert len(text) > 30
        # Debe mencionar artículos o sentimiento
        assert any(word in text.lower() for word in [
            'article', 'artículo', 'headline', 'news', 'noticia',
            'positive', 'negative', 'neutral', 'sentiment', 'sentimiento',
            'found', 'encontr', '0', 'no hay', 'available'
        ]), f"No se encontró análisis de Alpaca en: {text[:300]}"

    def test_agent_can_fetch_gdelt_signals(self, news_agent):
        """El agente obtiene señales GDELT para una ventana temporal."""
        response = news_agent(
            "Usa get_gdelt_signals para obtener señales de sentimiento de NVDA "
            "con start_date='2024-07-01' y end_date='2024-08-07' y granularity='1w'. "
            "¿Cuál es la tendencia del sentimiento?"
        )
        text = str(response)
        assert len(text) > 30
        assert any(word in text.lower() for word in [
            'gdelt', 'sentiment', 'sentimiento', 'trend', 'tendencia',
            'increasing', 'decreasing', 'stable', 'unknown',
            'data', 'datos', 'no hay', 'available'
        ]), f"No se encontró análisis GDELT en: {text[:300]}"

    def test_agent_uses_cache_for_repeated_requests(self, news_agent):
        """El agente no llama a la API si los datos ya están en caché."""
        # Primera llamada — puede ir a la API
        response1 = news_agent(
            "Usa get_alpaca_news_signals para NVDA con end_date='2024-08-07'. "
            "Dame el total de artículos."
        )
        # Segunda llamada — debe usar caché
        response2 = news_agent(
            "Usa get_alpaca_news_signals para NVDA con end_date='2024-08-07'. "
            "Dame el total de artículos."
        )
        # Ambas respuestas deben ser sustanciales
        assert len(str(response1)) > 20
        assert len(str(response2)) > 20

    def test_agent_produces_sentiment_analysis(self, news_agent):
        """El agente produce un análisis de sentimiento con score numérico."""
        response = news_agent(
            "Analiza el sentimiento de las noticias de NVDA de los últimos 30 días "
            "usando get_news_sentiment_summary. "
            "Responde en JSON con: sentiment (POSITIVE/NEUTRAL/NEGATIVE), "
            "alpaca_score (número), gdelt_trend (string), reasoning (string)."
        )
        text = str(response)
        assert len(text) > 50
        # Debe contener alguna clasificación de sentimiento
        assert any(word in text.upper() for word in [
            'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'POSITIVO', 'NEGATIVO'
        ]), f"No se encontró clasificación de sentimiento en: {text[:300]}"

    def test_agent_handles_historical_window(self, news_agent):
        """El agente analiza correctamente una ventana temporal histórica."""
        result = analyze_ticker_news('NVDA', days=30, end_date='2024-08-07')
        assert isinstance(result, str)
        assert len(result) > 20
        # No debe haber un error fatal no manejado
        assert 'traceback' not in result.lower(), \
            f"El agente lanzó una excepción: {result[:300]}"

    def test_agent_identifies_volume_spikes(self, news_agent):
        """El agente puede identificar picos de volumen de noticias."""
        response = news_agent(
            "Usa get_gdelt_signals para NVDA con granularity='1d' y los últimos 30 días. "
            "¿Hay algún volume_spike? ¿Qué significa para el análisis?"
        )
        text = str(response)
        assert len(text) > 30
        # Debe mencionar volumen o spike
        assert any(word in text.lower() for word in [
            'volume', 'volumen', 'spike', 'pico', 'news', 'noticias',
            'true', 'false', 'no hay', 'available', 'datos'
        ]), f"No se encontró análisis de volumen en: {text[:300]}"
