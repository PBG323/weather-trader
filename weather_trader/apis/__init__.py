"""
Weather API clients for forecast data collection.

Supports:
- Open-Meteo: Free access to ECMWF, ECMWF AIFS, GFS, GFS Ensemble, HRRR models
- Visual Crossing: Backup/validation forecasts (free tier: 1,000/day)
- Tomorrow.io: Proprietary hyperlocal forecasts
- NWS: Official settlement source data
- Aviation Weather: METAR/TAF real-time observations (pilot edge)
"""

from .open_meteo import OpenMeteoClient, WeatherModel, reset_rate_limit as _reset_open_meteo
from .visual_crossing import VisualCrossingClient, reset_rate_limit as _reset_visual_crossing
from .tomorrow_io import TomorrowIOClient
from .nws import NWSClient
from .aviation_weather import (
    AviationWeatherClient,
    METARObservation,
    CITY_AIRPORTS,
    get_aviation_edge,
)


def reset_all_api_state():
    """
    Reset all API rate limiting and caching state.

    Bug #9 fix: Provides a single function to reset all API clients,
    ensuring a clean slate when the user requests a cache refresh.
    """
    # Reset Open-Meteo rate limiting state
    _reset_open_meteo()
    # Reset Visual Crossing daily counter
    _reset_visual_crossing()
    # Note: Tomorrow.io, NWS, and Aviation don't have module-level rate limit state


__all__ = [
    "OpenMeteoClient",
    "WeatherModel",
    "VisualCrossingClient",
    "TomorrowIOClient",
    "NWSClient",
    "AviationWeatherClient",
    "METARObservation",
    "CITY_AIRPORTS",
    "get_aviation_edge",
    "reset_all_api_state",
]
