"""
Weather API clients for forecast data collection.

Supports:
- Open-Meteo: Free access to ECMWF, GFS, HRRR models
- Tomorrow.io: Proprietary hyperlocal forecasts
- NWS: Official settlement source data
"""

from .open_meteo import OpenMeteoClient
from .tomorrow_io import TomorrowIOClient
from .nws import NWSClient

__all__ = ["OpenMeteoClient", "TomorrowIOClient", "NWSClient"]
