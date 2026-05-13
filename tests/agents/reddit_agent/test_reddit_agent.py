"""
Tests para agents/reddit_agent.py

Cubre:
- Creación del agente y exposición de tools MCP
- Fallback sin Strands (usa MCP tools directamente)
- analyze_reddit_sentiment con y sin end_date
- Integración con get_reddit_signals, get_reddit_sentiment_summary
- Ventana temporal: end_date se propaga correctamente
- Fallback histórico → scraper en tiempo real
- Sin texto de posts en el output (privacidad + tokens)
"""

import json
import pytest
import pandas as pd
from datetime import date, datetime, timezone
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st

import agents.reddit_agent as reddit_agent_module
from agents.reddit_agent import (
    create_reddit_agent,
    analyze_reddit_sentiment,
    REDDIT_AGENT_PROMPT,
)
from tools.mcp.mcp_reddit_agent import (
    get_reddit_signals,
    get_reddit_sentiment_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_posts_df(end_date: str, n: int = 20) -> pd.DataFrame:
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    from datetime import timedelta
    timestamps = [int((end_dt - timedelta(days=i % 7)).timestamp()) for i in range(n)]
    labels = (['positive'] * (n // 2)) + (['negative'] * (n // 4)) + (['neutral'] * (n - n // 2 - n // 4))
    return pd.DataFrame({
        'id': [str(i) for i in range(n)],
        'subreddit': ['wallstreetbets'] * n,
        'title': [f'NVDA post {i}' for i in range(n)],
        'selftext': [''] * n,
        'created_utc': timestamps,
        'score': [100] * n,
        'num_comments': [10] * n,
        'author': ['user'] * n,
        'permalink': ['/r/wsb/'] * n,
        'url': [f'http://reddit.com/{i}' for i in range(n)],
        'sent_finbert_label': labels,
    })


def _mock_no_historical():
    return patch('tools.mcp.mcp_reddit_agent._historical_reddit', return_value=None)


# ---------------------------------------------------------------------------
# Creación del agente
# ---------------------------------------------------------------------------

class TestCreateRedditAgent:

    def test_returns_none_when_strands_unavailable(self):
        with patch.object(reddit_agent_module, 'STRANDS_AVAILABLE', False):
            agent = create_reddit_agent()
        assert agent is None

    def test_agent_has_two_mcp_tools(self):
        mock_agent = MagicMock()
        mock_agent.tools = [get_reddit_sentiment_summary, get_reddit_signals]

        with patch.object(reddit_agent_module, 'STRANDS_AVAILABLE', True), \
             patch('agents.reddit_agent.Agent', return_value=mock_agent), \
             patch('agents.reddit_agent.OllamaModel'):
            agent = create_reddit_agent()

        assert len(agent.tools) == 2
        tool_names = {t.__name__ for t in agent.tools}
        assert tool_names == {'get_reddit_sentiment_summary', 'get_reddit_signals'}

    def test_system_prompt_mentions_both_tools(self):
        assert 'get_reddit_sentiment_summary' in REDDIT_AGENT_PROMPT
        assert 'get_reddit_signals' in REDDIT_AGENT_PROMPT

    def test_system_prompt_mentions_end_date(self):
        assert 'end_date' in REDDIT_AGENT_PROMPT

    def test_system_prompt_warns_about_reliability(self):
        """El prompt debe advertir sobre limitaciones del sentimiento Reddit."""
        assert 'retail' in REDDIT_AGENT_PROMPT.lower() or 'fiabilidad' in REDDIT_AGENT_PROMPT.lower()

    def test_no_external_module_dependency(self):
        """El agente no debe depender de Modulo4_MultiAgent."""
        import agents.reddit_agent as mod
        source = open(mod.__file__).read()
        assert 'Modulo4_MultiAgent' not in source
        assert 'OriginalRedditAgent' not in source


# ---------------------------------------------------------------------------
# analyze_reddit_sentiment — fallback sin Strands
# ---------------------------------------------------------------------------

class TestAnalyzeRedditSentimentFallback:

    def test_fallback_returns_json_string(self):
        mock_df = _make_posts_df('2024-08-07')
        with patch.object(reddit_agent_module, 'STRANDS_AVAILABLE', False), \
             _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df):
            result = analyze_reddit_sentiment('NVDA', end_date='2024-08-07')

        parsed = json.loads(result)
        assert 'analysis_end_date' in parsed

    def test_fallback_end_date_propagates(self):
        mock_df = _make_posts_df('2024-08-07')
        with patch.object(reddit_agent_module, 'STRANDS_AVAILABLE', False), \
             _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df):
            result = analyze_reddit_sentiment('NVDA', end_date='2024-08-07')

        parsed = json.loads(result)
        assert parsed['analysis_end_date'] == '2024-08-07'

    def test_fallback_without_end_date_does_not_raise(self):
        with patch.object(reddit_agent_module, 'STRANDS_AVAILABLE', False), \
             _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=pd.DataFrame()):
            result = analyze_reddit_sentiment('NVDA')

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# analyze_reddit_sentiment — con Strands mockeado
# ---------------------------------------------------------------------------

class TestAnalyzeRedditSentimentWithAgent:

    def test_with_end_date_passes_date_in_prompt(self):
        captured = []
        mock_agent = MagicMock(side_effect=lambda p: captured.append(p) or '{"community_sentiment":"NEUTRAL"}')

        with patch.object(reddit_agent_module, 'STRANDS_AVAILABLE', True), \
             patch('agents.reddit_agent.Agent', return_value=mock_agent), \
             patch('agents.reddit_agent.OllamaModel'):
            analyze_reddit_sentiment('NVDA', end_date='2024-08-07')

        assert any('2024-08-07' in p for p in captured)


# ---------------------------------------------------------------------------
# MCP tools directamente — integración
# ---------------------------------------------------------------------------

class TestMCPToolsIntegration:

    def test_get_reddit_signals_no_post_text(self):
        """El output no contiene texto de posts — solo métricas."""
        mock_df = _make_posts_df('2024-08-07')
        with _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df):
            result = get_reddit_signals('NVDA', end_date='2024-08-07')

        assert not result.get('error')
        # Verificar que no hay texto de posts en el output
        for day in result.get('daily_metrics', []):
            assert 'selftext' not in day
            assert 'title' not in day
            assert 'top_keywords' in day  # solo keywords, no texto completo

    def test_get_reddit_signals_structure(self):
        """get_reddit_signals retorna estructura correcta."""
        mock_df = _make_posts_df('2024-08-07')
        with _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df):
            result = get_reddit_signals('NVDA', end_date='2024-08-07')

        assert not result.get('error')
        assert 'daily_metrics' in result
        assert 'overall_sentiment' in result
        assert 'analysis_start_date' in result
        assert 'analysis_end_date' in result

    def test_get_reddit_signals_uses_historical_first(self):
        """Intenta datos históricos antes de usar el scraper."""
        historical_data = {
            'count': 10,
            'data': [{'date': '2024-08-07', 'sentiment_score': 0.3}],
            'summary': {'average_sentiment': 0.3, 'total_posts': 100},
        }
        with patch('tools.mcp.mcp_reddit_agent._historical_reddit', return_value=historical_data), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date') as mock_scraper:
            result = get_reddit_signals('NVDA', end_date='2024-08-07')

        mock_scraper.assert_not_called()
        assert result['source'] == 'historical_csv'

    def test_get_reddit_signals_falls_back_to_scraper(self):
        """Usa el scraper cuando no hay datos históricos."""
        mock_df = _make_posts_df('2024-08-07')
        with _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df) as mock_scraper:
            result = get_reddit_signals('NVDA', end_date='2024-08-07')

        mock_scraper.assert_called_once()
        assert result['source'] == 'live_scraper'

    def test_get_reddit_sentiment_summary_structure(self):
        """get_reddit_sentiment_summary retorna estructura condensada."""
        mock_df = _make_posts_df('2024-08-07')
        with _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df):
            result = get_reddit_sentiment_summary('NVDA', end_date='2024-08-07')

        assert not result.get('error')
        for field in ['sentiment_score', 'trend', 'post_volume_avg', 'total_posts',
                      'analysis_start_date', 'analysis_end_date']:
            assert field in result, f"Campo faltante: {field}"

    def test_get_reddit_sentiment_summary_trend_values(self):
        """El campo trend solo puede ser improving, stable o declining."""
        mock_df = _make_posts_df('2024-08-07', n=30)
        with _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df):
            result = get_reddit_sentiment_summary('NVDA', end_date='2024-08-07')

        assert result.get('trend') in ('improving', 'stable', 'declining')

    @settings(max_examples=30)
    @given(end=st.dates(min_value=date(2022, 1, 1), max_value=date(2024, 12, 31)))
    def test_end_date_roundtrip(self, end):
        """analysis_end_date == end_date para cualquier fecha válida."""
        end_str = end.strftime('%Y-%m-%d')
        mock_df = _make_posts_df(end_str)
        with _mock_no_historical(), \
             patch('tools.mcp.mcp_reddit_agent._scraper_with_end_date', return_value=mock_df):
            result = get_reddit_signals('NVDA', end_date=end_str)
        assert result.get('analysis_end_date') == end_str
