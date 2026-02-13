"""
Kalshi integration module.

Provides authentication, market discovery, and order execution for
Kalshi weather temperature markets (KXHIGH series).
"""

from .auth import KalshiAuth
from .client import KalshiClient, OrderResult
from .markets import (
    KalshiMarketFinder,
    WeatherMarket,
    TemperatureBracket,
    SameDayUncertainty,
    SameDayTradingChecker,
    today_est,
)
from .rounding import (
    kalshi_round,
    get_settlement_range,
    get_bracket_probability_with_rounding,
    get_determined_bracket,
    is_bracket_relevant,
    validate_edge_with_rounding,
    format_bracket,
    get_best_bracket_for_forecast,
)

__all__ = [
    "KalshiAuth",
    "KalshiClient",
    "OrderResult",
    "KalshiMarketFinder",
    "WeatherMarket",
    "TemperatureBracket",
    "SameDayUncertainty",
    "SameDayTradingChecker",
    "today_est",
    # Rounding utilities
    "kalshi_round",
    "get_settlement_range",
    "get_bracket_probability_with_rounding",
    "get_determined_bracket",
    "is_bracket_relevant",
    "validate_edge_with_rounding",
    "format_bracket",
    "get_best_bracket_for_forecast",
]
