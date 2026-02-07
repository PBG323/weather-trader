"""
Visual Crossing Weather API Client

Provides backup/validation weather forecasts.
Free tier: 1,000 records/day - sufficient for validation purposes.

Features:
- 15-day forecasts
- Historical data
- Global coverage
- Weather alerts
"""

import httpx
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional

from ..config import config, CityConfig


# Rate limiting state
_daily_calls: int = 0
_last_reset_date: Optional[date] = None
DAILY_LIMIT = 950  # Stay under 1000 limit with buffer


@dataclass
class VisualCrossingForecast:
    """Forecast data from Visual Crossing."""
    date: date
    temperature_high: float  # Daily high in Fahrenheit
    temperature_low: float   # Daily low in Fahrenheit
    feels_like_high: float
    feels_like_low: float
    humidity: float
    precip_prob: float
    conditions: str
    city: str
    source: str = "visual_crossing"


class VisualCrossingClient:
    """Client for Visual Crossing Weather API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Visual Crossing client.

        Args:
            api_key: API key (or set VISUAL_CROSSING_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("VISUAL_CROSSING_API_KEY", "")
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _check_rate_limit(self) -> bool:
        """Check if we're within daily rate limits."""
        global _daily_calls, _last_reset_date

        today = date.today()
        if _last_reset_date != today:
            _daily_calls = 0
            _last_reset_date = today

        return _daily_calls < DAILY_LIMIT

    def _increment_calls(self, count: int = 1):
        """Track API calls for rate limiting."""
        global _daily_calls
        _daily_calls += count

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def get_remaining_calls(self) -> int:
        """Get remaining API calls for today."""
        global _daily_calls, _last_reset_date

        today = date.today()
        if _last_reset_date != today:
            return DAILY_LIMIT
        return max(0, DAILY_LIMIT - _daily_calls)

    async def get_forecast(
        self,
        city_config: CityConfig,
        days: int = 7
    ) -> list[VisualCrossingForecast]:
        """
        Get daily temperature forecasts for a city.

        Args:
            city_config: City configuration with coordinates
            days: Number of forecast days (1-15)

        Returns:
            List of VisualCrossingForecast objects
        """
        if not self.api_key:
            raise ValueError("Visual Crossing API key not configured")

        if not self._check_rate_limit():
            raise Exception(f"Daily API limit reached ({DAILY_LIMIT} calls)")

        # Build location string
        location = f"{city_config.latitude},{city_config.longitude}"

        # Build date range
        start_date = date.today()
        end_date = start_date + timedelta(days=min(days, 15) - 1)

        url = f"{self.base_url}/{location}/{start_date.isoformat()}/{end_date.isoformat()}"

        params = {
            "unitGroup": "us",  # Fahrenheit
            "include": "days",
            "key": self.api_key,
            "contentType": "json",
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            self._increment_calls()

            forecasts = []
            for day_data in data.get("days", []):
                forecasts.append(VisualCrossingForecast(
                    date=date.fromisoformat(day_data["datetime"]),
                    temperature_high=day_data.get("tempmax", 0),
                    temperature_low=day_data.get("tempmin", 0),
                    feels_like_high=day_data.get("feelslikemax", 0),
                    feels_like_low=day_data.get("feelslikemin", 0),
                    humidity=day_data.get("humidity", 0),
                    precip_prob=day_data.get("precipprob", 0),
                    conditions=day_data.get("conditions", ""),
                    city=city_config.name,
                ))

            return forecasts

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"[Visual Crossing] Rate limited (429)")
            raise

    async def get_current_conditions(
        self,
        city_config: CityConfig
    ) -> dict:
        """
        Get current weather conditions for validation.

        Args:
            city_config: City configuration

        Returns:
            Dict with current conditions
        """
        if not self.api_key:
            raise ValueError("Visual Crossing API key not configured")

        if not self._check_rate_limit():
            raise Exception(f"Daily API limit reached")

        location = f"{city_config.latitude},{city_config.longitude}"
        url = f"{self.base_url}/{location}/today"

        params = {
            "unitGroup": "us",
            "include": "current",
            "key": self.api_key,
            "contentType": "json",
        }

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        self._increment_calls()

        current = data.get("currentConditions", {})
        return {
            "temperature": current.get("temp"),
            "feels_like": current.get("feelslike"),
            "humidity": current.get("humidity"),
            "conditions": current.get("conditions"),
            "observation_time": current.get("datetime"),
            "city": city_config.name,
            "source": "visual_crossing",
        }

    async def get_historical(
        self,
        city_config: CityConfig,
        target_date: date
    ) -> Optional[VisualCrossingForecast]:
        """
        Get historical weather data for a specific date.

        Args:
            city_config: City configuration
            target_date: Date to get historical data for

        Returns:
            VisualCrossingForecast with actual temperatures, or None
        """
        if not self.api_key:
            raise ValueError("Visual Crossing API key not configured")

        if not self._check_rate_limit():
            return None

        location = f"{city_config.latitude},{city_config.longitude}"
        url = f"{self.base_url}/{location}/{target_date.isoformat()}"

        params = {
            "unitGroup": "us",
            "include": "days",
            "key": self.api_key,
            "contentType": "json",
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            self._increment_calls()

            days = data.get("days", [])
            if days:
                day_data = days[0]
                return VisualCrossingForecast(
                    date=date.fromisoformat(day_data["datetime"]),
                    temperature_high=day_data.get("tempmax", 0),
                    temperature_low=day_data.get("tempmin", 0),
                    feels_like_high=day_data.get("feelslikemax", 0),
                    feels_like_low=day_data.get("feelslikemin", 0),
                    humidity=day_data.get("humidity", 0),
                    precip_prob=day_data.get("precipprob", 0),
                    conditions=day_data.get("conditions", ""),
                    city=city_config.name,
                )
        except Exception as e:
            print(f"[Visual Crossing] Historical fetch failed: {e}")

        return None


def reset_rate_limit():
    """Reset the daily call counter (for testing)."""
    global _daily_calls, _last_reset_date
    _daily_calls = 0
    _last_reset_date = None
