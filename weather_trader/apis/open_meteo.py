"""
Open-Meteo API Client

Provides access to multiple weather models:
- ECMWF IFS: Best overall accuracy, 9km resolution
- GFS: Good US coverage, 27km resolution
- HRRR: Best 0-48hr US forecasts, 3km resolution

Free API with no key required.
"""

import httpx
import asyncio
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
from enum import Enum

from ..config import config, CityConfig


# Rate limiting state
_rate_limit_until: Optional[datetime] = None
_consecutive_failures: int = 0
MAX_RETRIES = 3
BASE_BACKOFF = 2.0  # seconds


def reset_rate_limit():
    """Reset the rate limit state to allow fresh API calls."""
    global _rate_limit_until, _consecutive_failures
    _rate_limit_until = None
    _consecutive_failures = 0


class WeatherModel(Enum):
    """Available weather models on Open-Meteo."""
    ECMWF = "ecmwf"           # ECMWF IFS - 9km global, best overall
    ECMWF_AIFS = "ecmwf_aifs"  # ECMWF AIFS - AI-powered, 25km, experimental
    GFS = "gfs"               # NCEP GFS - 27km global, good US coverage
    GFS_ENSEMBLE = "gfs_ensemble"  # GFS Ensemble - probabilistic, 31 members
    HRRR = "hrrr"             # HRRR - 3km US, 0-48hr, 15-min updates
    BEST_MATCH = "best_match"  # Auto-selects best model blend


@dataclass
class ForecastPoint:
    """A single forecast data point."""
    timestamp: datetime
    temperature_high: float  # Daily high in Fahrenheit
    temperature_low: float   # Daily low in Fahrenheit
    model: str
    city: str
    latitude: float
    longitude: float


@dataclass
class HourlyForecast:
    """Hourly forecast data point."""
    timestamp: datetime
    temperature: float  # Temperature in Fahrenheit
    model: str
    city: str


class OpenMeteoClient:
    """Client for Open-Meteo weather API."""

    def __init__(self):
        self.base_url = config.api.open_meteo_base_url
        self.historical_url = config.api.open_meteo_historical_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _celsius_to_fahrenheit(self, celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9/5) + 32

    async def _request_with_retry(self, endpoint: str, params: dict) -> dict:
        """Make a request with retry logic and exponential backoff."""
        global _rate_limit_until, _consecutive_failures

        # Check if we're in a rate limit cooldown
        if _rate_limit_until and datetime.now() < _rate_limit_until:
            wait_seconds = (_rate_limit_until - datetime.now()).total_seconds()
            print(f"[Open-Meteo] Rate limited, waiting {wait_seconds:.0f}s...")
            raise Exception(f"Rate limited - retry after {wait_seconds:.0f}s")

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.get(endpoint, params=params)

                if response.status_code == 429:
                    # Rate limited - set cooldown
                    _consecutive_failures += 1
                    cooldown = min(300, BASE_BACKOFF ** (_consecutive_failures + 2))  # Max 5 min
                    _rate_limit_until = datetime.now() + timedelta(seconds=cooldown)
                    print(f"[Open-Meteo] Rate limited (429). Cooldown: {cooldown}s")
                    raise Exception(f"Rate limited - cooldown {cooldown}s")

                response.raise_for_status()
                _consecutive_failures = 0  # Reset on success
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    raise  # Don't retry 429s, let the cooldown handle it
                # Other errors - retry with backoff
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_BACKOFF ** attempt
                    print(f"[Open-Meteo] Request failed, retrying in {wait}s...")
                    await asyncio.sleep(wait)
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_BACKOFF ** attempt
                    await asyncio.sleep(wait)

        raise last_error or Exception("Request failed after retries")

    async def get_daily_forecast(
        self,
        city_config: CityConfig,
        model: WeatherModel = WeatherModel.BEST_MATCH,
        days: int = 7
    ) -> list[ForecastPoint]:
        """
        Get daily temperature forecasts for a city.

        Args:
            city_config: City configuration with coordinates
            model: Weather model to use
            days: Number of forecast days (1-16)

        Returns:
            List of ForecastPoint objects with daily high/low temperatures
        """
        # Build endpoint and params based on model
        # Open-Meteo API endpoints:
        # - /forecast: Auto-selects best model (blends multiple sources)
        # - /ecmwf: ECMWF IFS model (9km, best accuracy)
        # - /gfs: NCEP GFS model (27km)
        # - /ensemble: Probabilistic ensemble forecasts
        # - /forecast with models=ecmwf_aifs025: ECMWF AIFS AI model
        # - /forecast with models=hrrr_conus: HRRR for US (3km, 15-min updates)
        model_param = None

        if model == WeatherModel.ECMWF:
            endpoint = f"{self.base_url}/ecmwf"
        elif model == WeatherModel.ECMWF_AIFS:
            endpoint = f"{self.base_url}/forecast"
            model_param = "ecmwf_aifs025"
        elif model == WeatherModel.GFS:
            endpoint = f"{self.base_url}/gfs"
        elif model == WeatherModel.GFS_ENSEMBLE:
            # GFS Ensemble - use GFS endpoint (ensemble endpoint doesn't support daily aggregates)
            # Note: True ensemble spread data requires hourly fetching
            endpoint = f"{self.base_url}/gfs"
        elif model == WeatherModel.HRRR:
            # HRRR only available for US via forecast endpoint
            if city_config.country != "US":
                raise ValueError("HRRR model only available for US cities")
            endpoint = f"{self.base_url}/forecast"
            model_param = "hrrr_conus"
        else:
            endpoint = f"{self.base_url}/forecast"

        params = {
            "latitude": city_config.latitude,
            "longitude": city_config.longitude,
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": city_config.timezone,
            "forecast_days": min(days, 16),
        }

        # Add model parameter if specified
        if model_param:
            params["models"] = model_param

        # Use retry logic for rate limiting
        data = await self._request_with_retry(endpoint, params)

        forecasts = []
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])

        for i, date_str in enumerate(dates):
            if i < len(highs) and i < len(lows):
                forecasts.append(ForecastPoint(
                    timestamp=datetime.fromisoformat(date_str),
                    temperature_high=highs[i],
                    temperature_low=lows[i],
                    model=model.value,
                    city=city_config.name,
                    latitude=city_config.latitude,
                    longitude=city_config.longitude,
                ))

        return forecasts

    async def get_hourly_forecast(
        self,
        city_config: CityConfig,
        model: WeatherModel = WeatherModel.BEST_MATCH,
        hours: int = 48
    ) -> list[HourlyForecast]:
        """
        Get hourly temperature forecasts for a city.

        Args:
            city_config: City configuration with coordinates
            model: Weather model to use
            hours: Number of forecast hours

        Returns:
            List of HourlyForecast objects
        """
        model_param = None

        if model == WeatherModel.ECMWF:
            endpoint = f"{self.base_url}/ecmwf"
        elif model == WeatherModel.ECMWF_AIFS:
            endpoint = f"{self.base_url}/forecast"
            model_param = "ecmwf_aifs025"
        elif model == WeatherModel.GFS:
            endpoint = f"{self.base_url}/gfs"
        elif model == WeatherModel.GFS_ENSEMBLE:
            endpoint = f"{self.base_url}/ensemble"
            model_param = "gfs_seamless"
        elif model == WeatherModel.HRRR:
            if city_config.country != "US":
                raise ValueError("HRRR model only available for US cities")
            endpoint = f"{self.base_url}/forecast"
            # Use 15-minutely HRRR for highest resolution when available
            model_param = "hrrr_conus"
        else:
            endpoint = f"{self.base_url}/forecast"

        params = {
            "latitude": city_config.latitude,
            "longitude": city_config.longitude,
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "timezone": city_config.timezone,
            "forecast_hours": hours,
        }

        if model_param:
            params["models"] = model_param

        # Use retry logic for rate limiting
        data = await self._request_with_retry(endpoint, params)

        forecasts = []
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])

        for i, time_str in enumerate(times):
            if i < len(temps) and temps[i] is not None:
                forecasts.append(HourlyForecast(
                    timestamp=datetime.fromisoformat(time_str),
                    temperature=temps[i],
                    model=model.value,
                    city=city_config.name,
                ))

        return forecasts

    async def get_ensemble_forecast(
        self,
        city_config: CityConfig,
        days: int = 7,
        include_experimental: bool = True
    ) -> dict[str, list[ForecastPoint]]:
        """
        Get forecasts from multiple models for ensemble analysis.

        Args:
            city_config: City configuration with coordinates
            days: Number of forecast days
            include_experimental: Include ECMWF AIFS (experimental AI model)

        Returns:
            Dictionary mapping model names to forecast lists
        """
        results = {}

        # Core models - always included
        # Note: HRRR only supports hourly data, not daily aggregates like temp_max
        # Note: GFS_ENSEMBLE and ECMWF_AIFS removed - don't support daily aggregates well
        models = [
            WeatherModel.ECMWF,       # Primary: ECMWF IFS 9km
            WeatherModel.GFS,         # Secondary: GFS 27km
        ]

        # US-specific models
        if city_config.country == "US":
            models.append(WeatherModel.BEST_MATCH)  # Open-Meteo auto-blend

        for model in models:
            try:
                forecasts = await self.get_daily_forecast(city_config, model, days)
                results[model.value] = forecasts
            except Exception as e:
                # Log but continue with other models
                print(f"Warning: Failed to fetch {model.value} forecast: {e}")

        return results

    async def get_gfs_ensemble_members(
        self,
        city_config: CityConfig,
        days: int = 7
    ) -> dict:
        """
        Get all GFS ensemble member forecasts for probabilistic analysis.

        The GFS ensemble has 31 members (1 control + 30 perturbed).
        Returns spread information useful for confidence estimation.

        Args:
            city_config: City configuration with coordinates
            days: Number of forecast days

        Returns:
            Dict with mean, min, max, spread for high/low temps
        """
        endpoint = f"{self.base_url}/ensemble"

        params = {
            "latitude": city_config.latitude,
            "longitude": city_config.longitude,
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": city_config.timezone,
            "forecast_days": min(days, 16),
            "models": "gfs_seamless",
        }

        data = await self._request_with_retry(endpoint, params)

        # Parse ensemble statistics
        daily = data.get("daily", {})
        dates = daily.get("time", [])

        # GFS ensemble returns arrays - calculate statistics
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])

        return {
            "dates": dates,
            "high_temps": highs,
            "low_temps": lows,
            "city": city_config.name,
            "model": "gfs_ensemble",
        }

    async def get_historical_weather(
        self,
        city_config: CityConfig,
        start_date: date,
        end_date: date
    ) -> list[ForecastPoint]:
        """
        Get historical actual weather data.

        Args:
            city_config: City configuration with coordinates
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            List of ForecastPoint objects with actual temperatures
        """
        endpoint = f"{self.historical_url}/archive"

        params = {
            "latitude": city_config.latitude,
            "longitude": city_config.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": city_config.timezone,
        }

        # Use retry logic for rate limiting
        data = await self._request_with_retry(endpoint, params)

        results = []
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])

        for i, date_str in enumerate(dates):
            if i < len(highs) and i < len(lows):
                results.append(ForecastPoint(
                    timestamp=datetime.fromisoformat(date_str),
                    temperature_high=highs[i],
                    temperature_low=lows[i],
                    model="historical",
                    city=city_config.name,
                    latitude=city_config.latitude,
                    longitude=city_config.longitude,
                ))

        return results
