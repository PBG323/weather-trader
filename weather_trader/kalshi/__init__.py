"""
Kalshi integration module.

Provides authentication, market discovery, and order execution for
Kalshi weather temperature markets (KXHIGH series).
"""

from .auth import KalshiAuth
from .client import KalshiClient, OrderResult
from .markets import KalshiMarketFinder, WeatherMarket, TemperatureBracket

__all__ = [
    "KalshiAuth",
    "KalshiClient",
    "OrderResult",
    "KalshiMarketFinder",
    "WeatherMarket",
    "TemperatureBracket",
]
