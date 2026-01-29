"""
Weather API clients for forecast data collection.

Supports:
- Open-Meteo: Free access to ECMWF, GFS, HRRR models
- Tomorrow.io: Proprietary hyperlocal forecasts
- NWS: Official settlement source data
"""

from .open_meteo import OpenMeteoClient, reset_rate_limit as _reset_open_meteo
from .tomorrow_io import TomorrowIOClient
from .nws import NWSClient


def reset_all_api_state():
    """
    Reset all API rate limiting and caching state.

    Bug #9 fix: Provides a single function to reset all API clients,
    ensuring a clean slate when the user requests a cache refresh.
    """
    # Reset Open-Meteo rate limiting state
    _reset_open_meteo()
    # Note: Tomorrow.io and NWS don't have module-level rate limit state


__all__ = ["OpenMeteoClient", "TomorrowIOClient", "NWSClient", "reset_all_api_state"]
