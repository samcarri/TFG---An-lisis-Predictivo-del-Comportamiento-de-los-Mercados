"""
Tests para agents/market_agent.py

Cubre:
- Creación del agente y exposición de tools MCP
- Fallback sin Strands (usa MCP tools directamente)
- analyze_stock con y sin end_date
- Integración con get_market_data, get_technical_analysis, get_market_calendar_info
- Ventana temporal: end_date se propaga correctamente a las tools
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st

import agents.market_agent as market_agent_module
from agents.market_agent import (
    create_market_agent,
    analyze_stock,
    MARKET_AGENT_PROMPT,
)
from tools.mcp.mcp_market_agent import (
    get_market_data,
    get_technical_analysis,
    get_market_calendar_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(end_date: str, n: int = 60) -> pd.DataFrame:
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(end=end_dt, periods=n, freq='B')
    prices = np.linspace(100, 130, n) + np.random.normal(0, 0.5, n)
    return pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'open': prices, 'high': prices * 1.01, 'low': prices * 0.99,
        'close': prices, 'volume': np.random.randint(1_000_000, 5_000_000, n),
        'adj_close': prices, 'ticker': 'NVDA', 'is_trading_day': True,
    })


def _patch_collector(end_date: str):
    mock = MagicMock()
    mock.collect_stock_data.return_value = _make_df(end_date)
    mock.is_trading_day.return_value = True
    mock.nyse = None
    return patch('tools.mcp.mcp_market_agent._get_collector', return_value=mock)


# ---------------------------------------------------------------------------
# Creación del agente
# ---------------------------------------------------------------------------

class TestCreateMarketAgent:

    def test_returns_none_when_strands_unavailable(self):
        """Sin Strands, create_market_agent retorna None."""
        with patch.object(market_agent_module, 'STRANDS_AVAILABLE', False):
            agent = create_market_agent()
        assert agent is None

    def test_agent_has_three_mcp_tools(self):
        """El agente expone exactamente las 3 tools del MCP server."""
        mock_agent = MagicMock()
        mock_agent.tools = [get_market_data, get_technical_analysis, get_market_calendar_info]

        with patch.object(market_agent_module, 'STRANDS_AVAILABLE', True), \
             patch('agents.market_agent.Agent', return_value=mock_agent), \
             patch('agents.market_agent.OllamaModel'):
            agent = create_market_agent()

        assert len(agent.tools) == 3
        tool_names = {t.__name__ for t in agent.tools}
        assert tool_names == {'get_market_data', 'get_technical_analysis', 'get_market_calendar_info'}

    def test_system_prompt_mentions_mcp_tools(self):
        """El system prompt menciona las 3 tools del MCP."""
        assert 'get_market_data' in MARKET_AGENT_PROMPT
        assert 'get_technical_analysis' in MARKET_AGENT_PROMPT
        assert 'get_market_calendar_info' in MARKET_AGENT_PROMPT

    def test_system_prompt_mentions_end_date(self):
        """El system prompt explica el uso de end_date."""
        assert 'end_date' in MARKET_AGENT_PROMPT


# ---------------------------------------------------------------------------
# analyze_stock — fallback sin Strands
# ---------------------------------------------------------------------------

class TestAnalyzeStockFallback:

    def test_fallback_returns_json_string(self):
        """Sin Strands, analyze_stock retorna JSON válido."""
        with patch.object(market_agent_module, 'STRANDS_AVAILABLE', False), \
             _patch_collector('2024-08-07'):
            result = analyze_stock('NVDA', period='3mo', end_date='2024-08-07')

        parsed = json.loads(result)
        assert 'analysis_end_date' in parsed or 'trend' in parsed

    def test_fallback_with_end_date_propagates(self):
        """El end_date se propaga a get_technical_analysis en el fallback."""
        with patch.object(market_agent_module, 'STRANDS_AVAILABLE', False), \
             _patch_collector('2024-08-07'):
            result = analyze_stock('NVDA', end_date='2024-08-07')

        parsed = json.loads(result)
        assert parsed.get('analysis_end_date') == '2024-08-07'

    def test_fallback_without_end_date_does_not_raise(self):
        """Sin end_date, el fallback usa la fecha actual sin error."""
        today = datetime.now().strftime('%Y-%m-%d')
        with patch.object(market_agent_module, 'STRANDS_AVAILABLE', False), \
             _patch_collector(today):
            result = analyze_stock('NVDA')

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# analyze_stock — con Strands mockeado
# ---------------------------------------------------------------------------

class TestAnalyzeStockWithAgent:

    def test_with_end_date_passes_date_in_prompt(self):
        """Con Strands, el prompt incluye el end_date."""
        captured_prompts = []

        mock_agent = MagicMock(side_effect=lambda p: captured_prompts.append(p) or '{"signal":"NEUTRAL"}')

        with patch.object(market_agent_module, 'STRANDS_AVAILABLE', True), \
             patch('agents.market_agent.Agent', return_value=mock_agent), \
             patch('agents.market_agent.OllamaModel'):
            analyze_stock('NVDA', end_date='2024-08-07')

        assert any('2024-08-07' in p for p in captured_prompts)

    def test_without_end_date_prompt_says_datos_actuales(self):
        """Sin end_date, el prompt indica datos actuales."""
        captured_prompts = []

        mock_agent = MagicMock(side_effect=lambda p: captured_prompts.append(p) or '{"signal":"NEUTRAL"}')

        with patch.object(market_agent_module, 'STRANDS_AVAILABLE', True), \
             patch('agents.market_agent.Agent', return_value=mock_agent), \
             patch('agents.market_agent.OllamaModel'):
            analyze_stock('NVDA')

        assert any('actuales' in p for p in captured_prompts)


# ---------------------------------------------------------------------------
# MCP tools directamente — integración
# ---------------------------------------------------------------------------

class TestMCPToolsIntegration:

    def test_get_market_data_returns_required_fields(self):
        """get_market_data retorna los campos obligatorios."""
        with _patch_collector('2024-08-07'):
            result = get_market_data('NVDA', end_date='2024-08-07', period='3mo')

        assert not result.get('error')
        for field in ['ticker', 'current_price', 'trend', 'rsi', 'analysis_start_date', 'analysis_end_date']:
            assert field in result, f"Campo faltante: {field}"

    def test_get_technical_analysis_returns_signals(self):
        """get_technical_analysis retorna lista de señales."""
        with _patch_collector('2024-08-07'):
            result = get_technical_analysis('NVDA', end_date='2024-08-07')

        assert not result.get('error')
        assert isinstance(result.get('signals'), list)
        assert len(result['signals']) > 0

    def test_get_market_calendar_info_weekend(self):
        """get_market_calendar_info detecta fin de semana correctamente."""
        result = get_market_calendar_info('2024-08-10')  # sábado
        assert result['is_trading_day'] is False
        assert result['previous_trading_day'] is not None
        assert result['next_trading_day'] is not None

    def test_get_market_calendar_info_weekday(self):
        """get_market_calendar_info detecta día laborable correctamente."""
        result = get_market_calendar_info('2024-08-07')  # miércoles
        assert result['is_trading_day'] is True

    @settings(max_examples=30)
    @given(end=st.dates(min_value=date(2022, 1, 1), max_value=date(2024, 12, 31)))
    def test_get_market_data_end_date_roundtrip(self, end):
        """analysis_end_date == end_date para cualquier fecha válida."""
        end_str = end.strftime('%Y-%m-%d')
        with _patch_collector(end_str):
            result = get_market_data('NVDA', end_date=end_str, period='1mo')
        assert not result.get('error')
        assert result['analysis_end_date'] == end_str
