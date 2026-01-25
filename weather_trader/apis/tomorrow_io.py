"""
Tomorrow.io API Client

Proprietary weather forecasting with:
- Hyperlocal predictions
- 99.9% uptime SLA
- Minute-by-minute nowcasting

Requires API key.
"""

import httpx
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..config import config, CityConfig


@dataclass
class TomorrowForecast:
    """Forecast data from Tomorrow.io."""
    timestamp: datetime
    temperature_high: float  # Daily high in Fahrenheit
    temperature_low: float   # Daily low in Fahrenheit
    temperature_apparent_high: float
    temperature_apparent_low: float
    humidity_avg: float
    precipitation_probability: float
    city: str


@dataclass
class TomorrowHourly:
    """Hourly forecast from Tomorrow.io."""
    timestamp: datetime
    temperature: float
    temperature_apparent: float
    humidity: float
    precipitation_probability: float
    city: str


class TomorrowIOClient:
    """Client for Tomorrow.io weather API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.api.tomorrow_io_api_key
        if not self.api_key:
            raise ValueError("Tomorrow.io API key is required")

        self.base_url = config.api.tomorrow_io_base_url
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
        days: int = 5
    ) -> list[TomorrowForecast]:
        """
        Get daily temperature forecasts for a city.

        Args:
            city_config: City configuration with coordinates
            days: Number of forecast days (1-5 for free tier)

        Returns:
            List of TomorrowForecast objects
        """
        endpoint = f"{self.base_url}/weather/forecast"

        params = {
            "location": f"{city_config.latitude},{city_config.longitude}",
            "apikey": self.api_key,
            "units": "imperial",
            "timesteps": "1d",
        }

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        forecasts = []
        timelines = data.get("timelines", {})
        daily = timelines.get("daily", [])

        for day in daily[:days]:
            values = day.get("values", {})
            forecasts.append(TomorrowForecast(
                timestamp=datetime.fromisoformat(day["time"].replace("Z", "+00:00")),
                temperature_high=values.get("temperatureMax", 0),
                temperature_low=values.get("temperatureMin", 0),
                temperature_apparent_high=values.get("temperatureApparentMax", 0),
                temperature_apparent_low=values.get("temperatureApparentMin", 0),
                humidity_avg=values.get("humidityAvg", 0),
                precipitation_probability=values.get("precipitationProbabilityMax", 0),
                city=city_config.name,
            ))

        return forecasts

    async def get_hourly_forecast(
        self,
        city_config: CityConfig,
        hours: int = 24
    ) -> list[TomorrowHourly]:
        """
        Get hourly temperature forecasts.

        Args:
            city_config: City configuration with coordinates
            hours: Number of forecast hours

        Returns:
            List of TomorrowHourly objects
        """
        endpoint = f"{self.base_url}/weather/forecast"

        params = {
            "location": f"{city_config.latitude},{city_config.longitude}",
            "apikey": self.api_key,
            "units": "imperial",
            "timesteps": "1h",
        }

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        forecasts = []
        timelines = data.get("timelines", {})
        hourly = timelines.get("hourly", [])

        for hour in hourly[:hours]:
            values = hour.get("values", {})
            forecasts.append(TomorrowHourly(
                timestamp=datetime.fromisoformat(hour["time"].replace("Z", "+00:00")),
                temperature=values.get("temperature", 0),
                temperature_apparent=values.get("temperatureApparent", 0),
                humidity=values.get("humidity", 0),
                precipitation_probability=values.get("precipitationProbability", 0),
                city=city_config.name,
            ))

        return forecasts

    async def get_realtime(
        self,
        city_config: CityConfig
    ) -> dict:
        """
        Get current/realtime weather conditions.

        Args:
            city_config: City configuration with coordinates

        Returns:
            Dictionary with current weather data
        """
        endpoint = f"{self.base_url}/weather/realtime"

        params = {
            "location": f"{city_config.latitude},{city_config.longitude}",
            "apikey": self.api_key,
            "units": "imperial",
        }

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        return data.get("data", {}).get("values", {})

    async def get_forecast_with_confidence(
        self,
        city_config: CityConfig,
        target_date: datetime
    ) -> dict:
        """
        Get forecast with confidence intervals for a specific date.

        Tomorrow.io provides ensemble-based uncertainty.

        Args:
            city_config: City configuration
            target_date: Date to forecast

        Returns:
            Dictionary with forecast and confidence data
        """
        forecasts = await self.get_daily_forecast(city_config, days=5)

        # Find forecast for target date
        target = target_date.date()
        for forecast in forecasts:
            if forecast.timestamp.date() == target:
                # Tomorrow.io doesn't provide explicit confidence intervals
                # We estimate based on forecast horizon
                days_ahead = (target - datetime.now().date()).days

                # Uncertainty increases with forecast horizon
                # Approximate standard deviation in Fahrenheit
                base_uncertainty = 2.0
                uncertainty = base_uncertainty * (1 + 0.3 * days_ahead)

                return {
                    "forecast": forecast,
                    "high_mean": forecast.temperature_high,
                    "low_mean": forecast.temperature_low,
                    "high_std": uncertainty,
                    "low_std": uncertainty,
                    "confidence": max(0.5, 1.0 - 0.1 * days_ahead),
                }

        raise ValueError(f"No forecast available for {target_date}")
