"""
Tools package for Multi-Agent Financial System
"""

# Time window utilities
from .time_window_utils import (
    validate_time_window,
    normalize_date_range,
    PERIOD_TO_DAYS,
    TimeWindowDict,
)

# MCP server — MarketAgent
from .mcp.mcp_market_agent import (
    get_market_data,
    get_technical_analysis as get_technical_analysis_mcp,
    get_market_calendar_info,
)

# MCP server — NewsAgent
from .mcp.mcp_news_agent import (
    get_alpaca_news_signals,
    get_gdelt_signals,
    get_news_sentiment_summary,
)

# MCP server — RedditAgent
from .mcp.mcp_reddit_agent import (
    get_reddit_signals,
    get_reddit_sentiment_summary,
)

# MCP server — DebateSystem (orquestador)
from .mcp.mcp_debate_system import (
    get_unified_signals,
    validate_time_window_tool,
)

# Legacy sources — market
from .sources.market_technical_tools import (
    get_market_features,
    get_technical_analysis,
    calculate_rsi,
    calculate_macd,
    detect_trend,
    detect_overbought_oversold,
    compare_to_baseline,
)

# Legacy sources — reddit
from .sources.reddit_tools import (
    RedditTools,
    create_reddit_tools,
)

__all__ = [
    # Time window utilities
    'validate_time_window',
    'normalize_date_range',
    'PERIOD_TO_DAYS',
    'TimeWindowDict',
    # MCP — MarketAgent
    'get_market_data',
    'get_technical_analysis_mcp',
    'get_market_calendar_info',
    # MCP — NewsAgent
    'get_alpaca_news_signals',
    'get_gdelt_signals',
    'get_news_sentiment_summary',
    # MCP — RedditAgent
    'get_reddit_signals',
    'get_reddit_sentiment_summary',
    # MCP — DebateSystem
    'get_unified_signals',
    'validate_time_window_tool',
    # Legacy market tools
    'get_market_features',
    'get_technical_analysis',
    'calculate_rsi',
    'calculate_macd',
    'detect_trend',
    'detect_overbought_oversold',
    'compare_to_baseline',
    # Reddit tools
    'RedditTools',
    'create_reddit_tools',
]
