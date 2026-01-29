"""
Open-Meteo API Client

Provides access to multiple weather models:
- ECMWF IFS: Best overall accuracy, 9km resolution
- GFS: Good US coverage, 27km resolution
- HRRR: Best 0-48hr US forecasts, 3km resolution

Free API with no key required.
"""

import httpx
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
from enum import Enum

from ..config import config, CityConfig


class WeatherModel(Enum):
    """Available weather models on Open-Meteo."""
    ECMWF = "ecmwf"
    GFS = "gfs"
    HRRR = "hrrr"
    BEST_MATCH = "best_match"  # Auto-selects best model


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
        # Build endpoint based on model
        # Open-Meteo API endpoints:
        # - /forecast: Auto-selects best model (blends multiple sources)
        # - /ecmwf: ECMWF IFS model
        # - /gfs: NCEP GFS model
        # - /forecast with models=hrrr_conus: HRRR for US
        if model == WeatherModel.ECMWF:
            endpoint = f"{self.base_url}/ecmwf"
        elif model == WeatherModel.GFS:
            endpoint = f"{self.base_url}/gfs"
        elif model == WeatherModel.HRRR:
            # HRRR only available for US via forecast endpoint with model parameter
            if city_config.country != "US":
                raise ValueError("HRRR model only available for US cities")
            endpoint = f"{self.base_url}/forecast"
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

        # HRRR requires specifying the model explicitly
        if model == WeatherModel.HRRR:
            params["models"] = "hrrr_conus"

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

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
        if model == WeatherModel.ECMWF:
            endpoint = f"{self.base_url}/ecmwf"
        elif model == WeatherModel.GFS:
            endpoint = f"{self.base_url}/gfs"
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

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

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
        days: int = 7
    ) -> dict[str, list[ForecastPoint]]:
        """
        Get forecasts from multiple models for ensemble analysis.

        Args:
            city_config: City configuration with coordinates
            days: Number of forecast days

        Returns:
            Dictionary mapping model names to forecast lists
        """
        results = {}

        # Models to fetch based on city location
        models = [WeatherModel.ECMWF, WeatherModel.GFS]
        if city_config.country == "US":
            models.append(WeatherModel.HRRR)  # Best for short-term US forecasts
            models.append(WeatherModel.BEST_MATCH)  # Auto-blend model

        for model in models:
            try:
                forecasts = await self.get_daily_forecast(city_config, model, days)
                results[model.value] = forecasts
            except Exception as e:
                # Log but continue with other models
                print(f"Warning: Failed to fetch {model.value} forecast: {e}")

        return results

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

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

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
