"""
Agents package for Multi-Agent Financial System
"""

from .market_agent import create_market_agent, analyze_stock
from .news_agent import create_news_agent
from .reddit_agent import create_reddit_agent
from .debate_system import DebateSystem

__all__ = [
    'create_market_agent',
    'analyze_stock',
    'create_news_agent',
    'create_reddit_agent',
    'DebateSystem'
]
