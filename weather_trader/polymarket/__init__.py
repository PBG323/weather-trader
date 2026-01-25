"""
Polymarket integration module.

Handles:
- Authentication and wallet management
- Market discovery for weather contracts
- Order execution via CLOB API
"""

from .auth import PolymarketAuth, create_wallet
from .client import PolymarketClient
from .markets import WeatherMarketFinder, WeatherMarket

__all__ = [
    "PolymarketAuth",
    "create_wallet",
    "PolymarketClient",
    "WeatherMarketFinder",
    "WeatherMarket",
]
