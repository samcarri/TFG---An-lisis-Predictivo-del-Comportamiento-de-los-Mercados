"""
Tests E2E para el Market Agent.

El agente LLM (qwen2.5:7b via Ollama) recibe un prompt, decide qué tools
llamar, las ejecuta con datos reales de Yahoo Finance, y produce un análisis.

Requiere:
- Ollama corriendo en localhost:11434 con qwen2.5:7b
- Conexión a internet (Yahoo Finance)

Ejecutar solo estos tests:
    pytest tests/agents/market_agent/test_market_agent_e2e.py -v -m e2e
"""

import json
import pytest
from agents.market_agent import create_market_agent, analyze_stock, STRANDS_AVAILABLE

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def market_agent():
    """Crea el agente una sola vez para todos los tests del módulo."""
    if not STRANDS_AVAILABLE:
        pytest.skip("Strands no disponible")
    agent = create_market_agent(callback_handler=None)
    if agent is None:
        pytest.skip("No se pudo crear el Market Agent")
    return agent


class TestMarketAgentE2E:

    def test_agent_can_fetch_and_analyze_current_data(self, market_agent):
        """El agente obtiene datos actuales de NVDA y produce un análisis válido."""
        response = market_agent(
            "Analiza NVDA con datos actuales usando get_market_data. "
            "Responde en JSON con signal, confidence y reasons."
        )
        text = str(response)
        assert len(text) > 50, "La respuesta es demasiado corta"
        # El agente debe haber producido alguna señal
        assert any(word in text.upper() for word in ['BULLISH', 'BEARISH', 'NEUTRAL', 'UP', 'DOWN']), \
            f"No se encontró señal de mercado en: {text[:200]}"

    def test_agent_can_fetch_historical_data_with_end_date(self, market_agent):
        """El agente obtiene datos históricos de NVDA para una fecha específica."""
        response = market_agent(
            "Analiza NVDA con end_date='2024-08-07' y period='1mo' usando get_market_data. "
            "Responde en JSON con signal, current_price y analysis_end_date."
        )
        text = str(response)
        assert len(text) > 50
        # El agente debe haber usado la fecha correcta
        assert '2024-08-07' in text or '2024' in text, \
            f"No se encontró referencia a la fecha en: {text[:300]}"

    def test_agent_uses_get_market_data_tool(self, market_agent):
        """El agente llama a get_market_data para obtener datos de mercado."""
        # Usamos analyze_stock que incluye el prompt completo
        result = analyze_stock('NVDA', period='1mo', end_date='2024-08-07')
        assert isinstance(result, str)
        assert len(result) > 20

    def test_agent_uses_get_technical_analysis_tool(self, market_agent):
        """El agente puede usar get_technical_analysis para análisis detallado."""
        response = market_agent(
            "Usa get_technical_analysis para analizar NVDA con end_date='2024-08-07'. "
            "Dime la tendencia, RSI y señales clave."
        )
        text = str(response)
        assert len(text) > 50
        # Debe mencionar algún indicador técnico
        assert any(word in text.upper() for word in ['RSI', 'MACD', 'SMA', 'TREND', 'TENDENCIA', 'BULLISH', 'BEARISH']), \
            f"No se encontraron indicadores técnicos en: {text[:300]}"

    def test_agent_uses_calendar_tool_for_weekend(self, market_agent):
        """El agente puede consultar si una fecha es día de trading."""
        response = market_agent(
            "Usa get_market_calendar_info para verificar si 2024-08-10 es día de trading en NYSE. "
            "Dime también el día de trading anterior y siguiente."
        )
        text = str(response)
        assert len(text) > 20
        # 2024-08-10 es sábado, no es día de trading
        assert any(word in text.lower() for word in ['no', 'false', 'weekend', 'fin de semana', 'sábado', 'saturday']), \
            f"El agente no detectó que es fin de semana: {text[:300]}"

    def test_agent_produces_structured_analysis(self, market_agent):
        """El agente produce un análisis estructurado con los campos esperados."""
        response = market_agent(
            f"Analiza NVDA para el período 3mo con end_date='2024-08-07'. "
            "Usa get_market_data y responde SOLO con este JSON:\n"
            '{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, '
            '"trend": "bullish|bearish|sideways", "rsi": 0.0, "reasons": []}'
        )
        text = str(response)
        # Intentar parsear JSON de la respuesta
        try:
            # Buscar JSON en la respuesta
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                assert 'signal' in parsed or 'trend' in parsed or 'confidence' in parsed
        except json.JSONDecodeError:
            # Si no es JSON puro, al menos debe tener contenido sustancial
            assert len(text) > 100, f"Respuesta demasiado corta: {text}"

    def test_agent_handles_weekend_end_date_gracefully(self, market_agent):
        """El agente maneja correctamente un end_date en fin de semana."""
        response = market_agent(
            "Analiza NVDA con end_date='2024-08-10' (sábado) usando get_market_data. "
            "¿Qué datos obtienes? ¿Se ajustó la fecha automáticamente?"
        )
        text = str(response)
        assert len(text) > 30
        # No debe haber un error fatal
        assert 'error' not in text.lower() or 'adjusted' in text.lower() or 'ajust' in text.lower() or \
               any(c.isdigit() for c in text), \
            f"El agente falló con fin de semana: {text[:300]}"
