"""
Aviation Weather API Client (METAR/TAF)

Accesses real-time weather observations from aviation meteorological stations.
This data updates every 1-3 hours and provides actual sensor readings,
not forecasts - giving a significant edge over public weather forecasts.

Data sources:
- METAR: Meteorological Aerodrome Report (current conditions)
- TAF: Terminal Aerodrome Forecast (airport-specific forecast)

The key insight from successful weather traders:
"Pilots get weather updates 12 hours before any public forecast.
Not forecasts. Observations. Real readings from real sensors."
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import re

import httpx


# NOAA Aviation Weather Center - Free, no API key required
AVIATIONWEATHER_BASE_URL = "https://aviationweather.gov/api/data"

# CheckWX API - More reliable, requires free API key
CHECKWX_BASE_URL = "https://api.checkwx.com"


@dataclass
class METARObservation:
    """Decoded METAR observation with temperature data."""
    station_id: str
    observation_time: datetime
    temperature_c: float
    temperature_f: float
    dewpoint_c: Optional[float]
    wind_speed_kt: Optional[int]
    wind_direction: Optional[int]
    visibility_miles: Optional[float]
    altimeter_inhg: Optional[float]
    sky_condition: str
    flight_category: str  # VFR, MVFR, IFR, LIFR
    raw_metar: str
    age_minutes: int  # How old is this observation

    @property
    def is_fresh(self) -> bool:
        """Check if observation is less than 2 hours old."""
        return self.age_minutes < 120

    @property
    def is_very_fresh(self) -> bool:
        """Check if observation is less than 30 minutes old."""
        return self.age_minutes < 30


@dataclass
class TAFForecast:
    """Decoded TAF forecast for an airport."""
    station_id: str
    issue_time: datetime
    valid_from: datetime
    valid_to: datetime
    temperature_forecasts: list[dict]  # List of {time, temp_c, temp_f}
    raw_taf: str


# Airport codes for cities we trade
# Maps city keys to their primary airport ICAO codes
CITY_AIRPORTS = {
    "nyc": ["KJFK", "KLGA", "KEWR"],  # JFK, LaGuardia, Newark
    "chicago": ["KORD", "KMDW"],  # O'Hare, Midway
    "miami": ["KMIA", "KFLL"],  # Miami Int'l, Fort Lauderdale
    "la": ["KLAX", "KBUR"],  # LAX, Burbank
    "denver": ["KDEN"],  # Denver Int'l
    "philadelphia": ["KPHL"],  # Philadelphia Int'l
    "austin": ["KAUS"],  # Austin-Bergstrom
    "london": ["EGLL", "EGKK"],  # Heathrow, Gatwick
    "buenos_aires": ["SAEZ"],  # Ezeiza
}


class AviationWeatherClient:
    """
    Client for fetching real-time aviation weather observations.

    This provides the "pilot edge" - accessing METAR/TAF data that
    updates every 1-3 hours with actual sensor readings.
    """

    def __init__(self, checkwx_api_key: Optional[str] = None):
        """
        Initialize aviation weather client.

        Args:
            checkwx_api_key: Optional API key for CheckWX (free tier available)
        """
        self.checkwx_api_key = checkwx_api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

    async def get_metar(self, station_id: str) -> Optional[METARObservation]:
        """
        Fetch current METAR observation for an airport.

        Args:
            station_id: ICAO airport code (e.g., "KJFK")

        Returns:
            METARObservation or None if unavailable
        """
        self._ensure_client()

        try:
            # Try NOAA Aviation Weather Center first (free, no key)
            url = f"{AVIATIONWEATHER_BASE_URL}/metar"
            params = {
                "ids": station_id,
                "format": "json",
                "hours": 3,  # Get last 3 hours of observations
            }

            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            # Get most recent observation
            latest = data[0] if isinstance(data, list) else data

            return self._parse_metar(latest, station_id)

        except Exception as e:
            print(f"[AviationWeather] METAR fetch failed for {station_id}: {e}")
            return None

    def _parse_metar(self, data: dict, station_id: str) -> Optional[METARObservation]:
        """Parse METAR JSON response into structured observation."""
        try:
            # Extract observation time
            # Aviation Weather API returns obsTime as Unix timestamp (int)
            obs_time_val = data.get("obsTime") or data.get("reportTime")
            if obs_time_val:
                if isinstance(obs_time_val, (int, float)):
                    # Unix timestamp
                    obs_time = datetime.fromtimestamp(obs_time_val, tz=ZoneInfo("UTC"))
                else:
                    # ISO string format
                    obs_time = datetime.fromisoformat(str(obs_time_val).replace("Z", "+00:00"))
            else:
                obs_time = datetime.now(ZoneInfo("UTC"))

            # Calculate age
            now = datetime.now(ZoneInfo("UTC"))
            age_minutes = int((now - obs_time).total_seconds() / 60)

            # Temperature (METAR reports in Celsius)
            temp_c = data.get("temp")
            if temp_c is None:
                # Try parsing from raw METAR
                raw = data.get("rawOb", "")
                temp_match = re.search(r'\b(M?\d{2})/(M?\d{2})\b', raw)
                if temp_match:
                    temp_str = temp_match.group(1)
                    temp_c = -int(temp_str[1:]) if temp_str.startswith('M') else int(temp_str)

            if temp_c is None:
                return None

            temp_f = temp_c * 9/5 + 32

            # Dewpoint
            dewpoint_c = data.get("dewp")

            # Wind
            wind_speed = data.get("wspd")
            wind_dir = data.get("wdir")

            # Visibility - handle values like '10+' meaning "> 10 miles"
            visibility = data.get("visib")
            if visibility is not None and isinstance(visibility, str):
                visibility = visibility.rstrip('+')  # Remove trailing '+'

            # Altimeter
            altimeter = data.get("altim")

            # Sky condition
            sky = data.get("cover", "")
            if isinstance(sky, list) and sky:
                sky = sky[0].get("cover", "CLR")

            # Flight category
            flight_cat = data.get("fltcat", "VFR")

            # Raw METAR string
            raw_metar = data.get("rawOb", "")

            return METARObservation(
                station_id=station_id,
                observation_time=obs_time,
                temperature_c=float(temp_c),
                temperature_f=float(temp_f),
                dewpoint_c=float(dewpoint_c) if dewpoint_c else None,
                wind_speed_kt=int(wind_speed) if wind_speed else None,
                wind_direction=int(wind_dir) if wind_dir else None,
                visibility_miles=float(visibility) if visibility else None,
                altimeter_inhg=float(altimeter) if altimeter else None,
                sky_condition=str(sky),
                flight_category=str(flight_cat),
                raw_metar=raw_metar,
                age_minutes=age_minutes,
            )

        except Exception as e:
            print(f"[AviationWeather] METAR parse failed: {e}")
            return None

    async def get_city_observations(self, city_key: str) -> list[METARObservation]:
        """
        Get all METAR observations for a city's airports.

        Args:
            city_key: City key (e.g., "nyc", "chicago")

        Returns:
            List of METARObservation from all airports for that city
        """
        airports = CITY_AIRPORTS.get(city_key.lower(), [])
        if not airports:
            return []

        observations = []
        for station in airports:
            obs = await self.get_metar(station)
            if obs:
                observations.append(obs)

        return observations

    async def get_current_temperature(self, city_key: str) -> Optional[dict]:
        """
        Get current observed temperature for a city.

        Returns the most recent observation from any of the city's airports.

        This is the "pilot edge" - real sensor data, not forecasts.

        Args:
            city_key: City key (e.g., "nyc")

        Returns:
            Dict with temperature_f, temperature_c, observation_time, age_minutes, station
        """
        observations = await self.get_city_observations(city_key)

        if not observations:
            return None

        # Get the freshest observation
        freshest = min(observations, key=lambda x: x.age_minutes)

        return {
            "temperature_f": freshest.temperature_f,
            "temperature_c": freshest.temperature_c,
            "observation_time": freshest.observation_time,
            "age_minutes": freshest.age_minutes,
            "station": freshest.station_id,
            "is_fresh": freshest.is_fresh,
            "is_very_fresh": freshest.is_very_fresh,
            "raw_metar": freshest.raw_metar,
        }

    async def get_all_city_temperatures(self) -> dict[str, dict]:
        """
        Get current temperatures for all tracked cities.

        Returns:
            Dict mapping city_key to temperature data
        """
        results = {}

        tasks = [
            self.get_current_temperature(city)
            for city in CITY_AIRPORTS.keys()
        ]

        city_keys = list(CITY_AIRPORTS.keys())
        observations = await asyncio.gather(*tasks, return_exceptions=True)

        for city_key, obs in zip(city_keys, observations):
            if isinstance(obs, dict):
                results[city_key] = obs
            elif isinstance(obs, Exception):
                print(f"[AviationWeather] Failed to get {city_key}: {obs}")

        return results


async def get_aviation_edge(city_key: str, forecast_high: float) -> dict:
    """
    Calculate the "aviation edge" - difference between forecast and current observation.

    If the current observed temperature is already higher than forecast high,
    and it's early in the day, there's likely an edge on the high side.

    Args:
        city_key: City key (e.g., "nyc")
        forecast_high: Forecasted daily high in Fahrenheit

    Returns:
        Dict with edge analysis
    """
    async with AviationWeatherClient() as client:
        current = await client.get_current_temperature(city_key)

        if not current:
            return {"has_edge": False, "reason": "No observation available"}

        current_temp = current["temperature_f"]
        age = current["age_minutes"]

        # If current temp already exceeds forecast high, there's an edge
        if current_temp >= forecast_high:
            return {
                "has_edge": True,
                "edge_type": "HIGH_EXCEEDED",
                "current_temp": current_temp,
                "forecast_high": forecast_high,
                "difference": current_temp - forecast_high,
                "observation_age": age,
                "recommendation": f"Current temp {current_temp:.1f}°F already >= forecast high {forecast_high:.1f}°F. HIGH brackets above {current_temp:.0f}°F have edge.",
                "station": current["station"],
            }

        # If current temp is close to forecast high early in day
        # (within 2°F), high side may have edge
        if current_temp >= forecast_high - 2:
            return {
                "has_edge": True,
                "edge_type": "HIGH_APPROACHING",
                "current_temp": current_temp,
                "forecast_high": forecast_high,
                "difference": current_temp - forecast_high,
                "observation_age": age,
                "recommendation": f"Current temp {current_temp:.1f}°F approaching forecast high. Monitor for overshoot.",
                "station": current["station"],
            }

        return {
            "has_edge": False,
            "current_temp": current_temp,
            "forecast_high": forecast_high,
            "difference": current_temp - forecast_high,
            "observation_age": age,
            "station": current["station"],
        }
