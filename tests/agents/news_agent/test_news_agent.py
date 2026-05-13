"""
Tests para agents/news_agent.py

Cubre:
- Creación del agente y exposición de tools MCP
- Fallback sin Strands (usa MCP tools directamente)
- analyze_ticker_news con y sin end_date
- Integración con get_news_sentiment_summary, get_alpaca_news_signals, get_gdelt_signals
- Ventana temporal: end_date se propaga correctamente
- Caché inteligente: no llama a Alpaca si los datos ya están en caché
"""

import json
import pytest
import pandas as pd
from datetime import date, datetime
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st

import agents.news_agent as news_agent_module
from agents.news_agent import (
    create_news_agent,
    analyze_ticker_news,
    NEWS_AGENT_PROMPT,
)
from tools.mcp.mcp_news_agent import (
    get_alpaca_news_signals,
    get_gdelt_signals,
    get_news_sentiment_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_news_df(end_date: str, n: int = 20) -> pd.DataFrame:
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(end=end_dt, periods=n, freq='D')
    labels = (['positive'] * (n // 2)) + (['negative'] * (n // 4)) + (['neutral'] * (n - n // 2 - n // 4))
    return pd.DataFrame({
        'date': dates,
        'title': [f'NVDA news {i}' for i in range(n)],
        'summary': ['summary'] * n,
        'url': [f'http://example.com/{i}' for i in range(n)],
        'source': ['alpaca'] * n,
        'financialbert_label': labels,
    })


def _make_gdelt_df(start_date: str, end_date: str) -> pd.DataFrame:
    import numpy as np
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return pd.DataFrame({
        'date': dates,
        'mean_tone': np.random.uniform(-2, 2, len(dates)),
        'n_news': np.random.randint(50, 200, len(dates)),
        'tone_momentum': np.random.uniform(-0.5, 0.5, len(dates)),
    })


# ---------------------------------------------------------------------------
# Creación del agente
# ---------------------------------------------------------------------------

class TestCreateNewsAgent:

    def test_returns_none_when_strands_unavailable(self):
        with patch.object(news_agent_module, 'STRANDS_AVAILABLE', False):
            agent = create_news_agent()
        assert agent is None

    def test_agent_has_three_mcp_tools(self):
        mock_agent = MagicMock()
        mock_agent.tools = [get_news_sentiment_summary, get_alpaca_news_signals, get_gdelt_signals]

        with patch.object(news_agent_module, 'STRANDS_AVAILABLE', True), \
             patch('agents.news_agent.Agent', return_value=mock_agent), \
             patch('agents.news_agent.OllamaModel'):
            agent = create_news_agent()

        assert len(agent.tools) == 3
        tool_names = {t.__name__ for t in agent.tools}
        assert tool_names == {'get_news_sentiment_summary', 'get_alpaca_news_signals', 'get_gdelt_signals'}

    def test_system_prompt_mentions_all_tools(self):
        assert 'get_news_sentiment_summary' in NEWS_AGENT_PROMPT
        assert 'get_alpaca_news_signals' in NEWS_AGENT_PROMPT
        assert 'get_gdelt_signals' in NEWS_AGENT_PROMPT

    def test_system_prompt_mentions_end_date(self):
        assert 'end_date' in NEWS_AGENT_PROMPT

    def test_system_prompt_mentions_financialbert(self):
        """El agente debe saber que Alpaca usa FinancialBERT."""
        assert 'FinancialBERT' in NEWS_AGENT_PROMPT


# ---------------------------------------------------------------------------
# analyze_ticker_news — fallback sin Strands
# ---------------------------------------------------------------------------

class TestAnalyzeTickerNewsFallback:

    def test_fallback_returns_json_string(self):
        mock_df = _make_news_df('2024-08-07')
        with patch.object(news_agent_module, 'STRANDS_AVAILABLE', False), \
             patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]):
            result = analyze_ticker_news('NVDA', end_date='2024-08-07')

        parsed = json.loads(result)
        assert 'analysis_end_date' in parsed

    def test_fallback_end_date_propagates(self):
        mock_df = _make_news_df('2024-08-07')
        with patch.object(news_agent_module, 'STRANDS_AVAILABLE', False), \
             patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]):
            result = analyze_ticker_news('NVDA', end_date='2024-08-07')

        parsed = json.loads(result)
        assert parsed['analysis_end_date'] == '2024-08-07'

    def test_fallback_without_end_date_does_not_raise(self):
        mock_df = _make_news_df(datetime.now().strftime('%Y-%m-%d'))
        with patch.object(news_agent_module, 'STRANDS_AVAILABLE', False), \
             patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]):
            result = analyze_ticker_news('NVDA')

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# analyze_ticker_news — con Strands mockeado
# ---------------------------------------------------------------------------

class TestAnalyzeTickerNewsWithAgent:

    def test_with_end_date_passes_date_in_prompt(self):
        captured = []
        mock_agent = MagicMock(side_effect=lambda p: captured.append(p) or '{"sentiment":"NEUTRAL"}')

        with patch.object(news_agent_module, 'STRANDS_AVAILABLE', True), \
             patch('agents.news_agent.Agent', return_value=mock_agent), \
             patch('agents.news_agent.OllamaModel'):
            analyze_ticker_news('NVDA', end_date='2024-08-07')

        assert any('2024-08-07' in p for p in captured)


# ---------------------------------------------------------------------------
# MCP tools directamente — integración
# ---------------------------------------------------------------------------

class TestMCPToolsIntegration:

    def test_get_alpaca_news_signals_structure(self):
        """get_alpaca_news_signals retorna estructura correcta."""
        mock_df = _make_news_df('2024-08-07')
        with patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]):
            result = get_alpaca_news_signals('NVDA', end_date='2024-08-07')

        assert not result.get('error')
        assert 'headlines' in result
        assert 'daily_metrics' in result
        assert 'overall_sentiment' in result
        assert 'analysis_start_date' in result
        assert 'analysis_end_date' in result

    def test_get_alpaca_news_signals_max_15_headlines(self):
        """get_alpaca_news_signals retorna máximo 15 headlines."""
        mock_df = _make_news_df('2024-08-07', n=50)
        with patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]):
            result = get_alpaca_news_signals('NVDA', end_date='2024-08-07')

        assert len(result['headlines']) <= 15

    def test_get_alpaca_news_signals_no_api_call_when_cached(self):
        """No llama a Alpaca API si todos los datos están en caché."""
        mock_df = _make_news_df('2024-08-07')
        with patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]), \
             patch('tools.mcp.mcp_news_agent._fetch_alpaca_and_update_cache') as mock_fetch:
            get_alpaca_news_signals('NVDA', end_date='2024-08-07')

        mock_fetch.assert_not_called()

    def test_get_alpaca_news_signals_calls_api_for_missing_dates(self):
        """Llama a Alpaca API cuando hay fechas faltantes en caché."""
        mock_df = _make_news_df('2024-08-07')
        with patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=['2024-08-06', '2024-08-07']), \
             patch('tools.mcp.mcp_news_agent._fetch_alpaca_and_update_cache', return_value=mock_df) as mock_fetch:
            get_alpaca_news_signals('NVDA', end_date='2024-08-07')

        mock_fetch.assert_called_once()

    def test_get_gdelt_signals_structure(self):
        """get_gdelt_signals retorna estructura correcta."""
        mock_agg = MagicMock()
        mock_agg.load_sentiment_data.return_value = _make_gdelt_df('2024-05-09', '2024-08-07')
        mock_agg._calculate_trend.return_value = 'stable'

        with patch('tools.sources.news_gdelt_aggregator.GDELTAggregator', mock_agg), \
             patch('tools.mcp.mcp_news_agent.GDELTAggregator', mock_agg, create=True):
            from tools.mcp.mcp_news_agent import get_gdelt_signals as _get
            with patch('tools.sources.news_gdelt_aggregator.GDELTAggregator', return_value=mock_agg):
                result = _get('NVDA', start_date='2024-05-09', end_date='2024-08-07')

        assert not result.get('error')
        assert 'sentiment_mean' in result
        assert 'sentiment_trend' in result
        assert result['analysis_start_date'] == '2024-05-09'
        assert result['analysis_end_date'] == '2024-08-07'

    def test_get_news_sentiment_summary_combines_sources(self):
        """get_news_sentiment_summary combina Alpaca y GDELT."""
        mock_df = _make_news_df('2024-08-07')
        mock_agg = MagicMock()
        mock_agg.load_sentiment_data.return_value = _make_gdelt_df('2024-08-01', '2024-08-07')
        mock_agg._calculate_trend.return_value = 'stable'

        with patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]), \
             patch('tools.sources.news_gdelt_aggregator.GDELTAggregator', return_value=mock_agg):
            from tools.mcp.mcp_news_agent import get_news_sentiment_summary as _get
            result = _get('NVDA', end_date='2024-08-07')

        assert not result.get('error')
        assert 'alpaca' in result
        assert 'gdelt' in result
        assert result['analysis_end_date'] == '2024-08-07'

    @settings(max_examples=30)
    @given(end=st.dates(min_value=date(2022, 1, 1), max_value=date(2024, 12, 31)))
    def test_alpaca_end_date_roundtrip(self, end):
        """analysis_end_date == end_date para cualquier fecha válida."""
        end_str = end.strftime('%Y-%m-%d')
        mock_df = _make_news_df(end_str)
        with patch('tools.mcp.mcp_news_agent._load_news_cache', return_value=mock_df), \
             patch('tools.mcp.mcp_news_agent._missing_dates', return_value=[]):
            result = get_alpaca_news_signals('NVDA', end_date=end_str)
        assert result.get('analysis_end_date') == end_str
