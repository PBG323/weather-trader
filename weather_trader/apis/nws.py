"""
National Weather Service (NWS) API Client

Provides access to official weather station data that Polymarket uses for settlement.
This is the source of truth for US cities.

For international cities (Toronto, London), we use METAR data from aviation sources.
"""

import httpx
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
import re

from ..config import config, CityConfig


@dataclass
class StationObservation:
    """Observation data from a weather station."""
    timestamp: datetime
    temperature: float  # In Fahrenheit
    temperature_celsius: float
    station_id: str
    station_name: str
    description: str
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[int] = None


@dataclass
class DailyClimateSummary:
    """Daily climate summary from official records."""
    date: date
    temperature_high: float  # Daily high in Fahrenheit
    temperature_low: float   # Daily low in Fahrenheit
    station_id: str
    observation_count: int


class NWSClient:
    """Client for NWS weather data and METAR observations."""

    def __init__(self):
        self.nws_base_url = config.api.nws_base_url
        self.user_agent = config.api.nws_user_agent
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": self.user_agent}
        )

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

    async def get_station_observations(
        self,
        city_config: CityConfig,
        hours: int = 24
    ) -> list[StationObservation]:
        """
        Get recent observations from the official weather station.

        Args:
            city_config: City configuration with station ID
            hours: Number of hours of observations to retrieve

        Returns:
            List of StationObservation objects
        """
        station_id = city_config.nws_station_id

        # For US stations, use NWS API
        if city_config.country == "US":
            return await self._get_nws_observations(station_id, hours)
        else:
            # For international stations, use METAR data
            return await self._get_metar_observations(station_id, hours)

    async def _get_nws_observations(
        self,
        station_id: str,
        hours: int
    ) -> list[StationObservation]:
        """Get observations from NWS API for US stations."""
        endpoint = f"{self.nws_base_url}/stations/{station_id}/observations"

        params = {
            "limit": min(hours * 2, 500),  # Observations may be more frequent than hourly
        }

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        observations = []
        features = data.get("features", [])

        for feature in features:
            props = feature.get("properties", {})

            # Get temperature (may be None)
            temp_c = props.get("temperature", {}).get("value")
            if temp_c is None:
                continue

            temp_f = self._celsius_to_fahrenheit(temp_c)

            # Parse timestamp
            timestamp_str = props.get("timestamp")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Check if within requested time range
            cutoff = datetime.now(timestamp.tzinfo) - timedelta(hours=hours)
            if timestamp < cutoff:
                break  # Observations are in reverse chronological order

            observations.append(StationObservation(
                timestamp=timestamp,
                temperature=temp_f,
                temperature_celsius=temp_c,
                station_id=station_id,
                station_name=props.get("station", "").split("/")[-1],
                description=props.get("textDescription", ""),
                humidity=props.get("relativeHumidity", {}).get("value"),
                wind_speed=props.get("windSpeed", {}).get("value"),
                wind_direction=props.get("windDirection", {}).get("value"),
            ))

        return observations

    async def _get_metar_observations(
        self,
        station_id: str,
        hours: int
    ) -> list[StationObservation]:
        """
        Get METAR observations for international stations.

        Uses the aviationweather.gov METAR API.
        """
        # Aviation Weather Center API for METAR data
        endpoint = "https://aviationweather.gov/api/data/metar"

        params = {
            "ids": station_id,
            "format": "json",
            "hours": hours,
        }

        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception:
            # Fallback: try alternative endpoint
            return await self._get_metar_fallback(station_id, hours)

        observations = []

        for metar in data if isinstance(data, list) else []:
            temp_c = metar.get("temp")
            if temp_c is None:
                continue

            temp_f = self._celsius_to_fahrenheit(float(temp_c))

            # Parse observation time
            obs_time = metar.get("obsTime")
            if obs_time:
                timestamp = datetime.fromtimestamp(obs_time)
            else:
                continue

            observations.append(StationObservation(
                timestamp=timestamp,
                temperature=temp_f,
                temperature_celsius=float(temp_c),
                station_id=station_id,
                station_name=metar.get("name", station_id),
                description=metar.get("rawOb", ""),
                humidity=None,  # METAR humidity requires dewpoint calculation
                wind_speed=metar.get("wspd"),
                wind_direction=metar.get("wdir"),
            ))

        return observations

    async def _get_metar_fallback(
        self,
        station_id: str,
        hours: int
    ) -> list[StationObservation]:
        """Fallback METAR source using weather.gov."""
        # Use ADDS TDS for METAR data
        endpoint = "https://www.aviationweather.gov/adds/dataserver_current/httpparam"

        params = {
            "dataSource": "metars",
            "requestType": "retrieve",
            "format": "csv",
            "stationString": station_id,
            "hoursBeforeNow": hours,
        }

        response = await self.client.get(endpoint, params=params)

        # Parse CSV response
        observations = []
        lines = response.text.split("\n")

        for line in lines:
            if line.startswith(station_id):
                parts = line.split(",")
                if len(parts) > 5:
                    try:
                        temp_c = float(parts[5]) if parts[5] else None
                        if temp_c is not None:
                            observations.append(StationObservation(
                                timestamp=datetime.fromisoformat(parts[2]),
                                temperature=self._celsius_to_fahrenheit(temp_c),
                                temperature_celsius=temp_c,
                                station_id=station_id,
                                station_name=station_id,
                                description=parts[0] if parts else "",
                            ))
                    except (ValueError, IndexError):
                        continue

        return observations

    async def get_daily_summary(
        self,
        city_config: CityConfig,
        target_date: Optional[date] = None
    ) -> Optional[DailyClimateSummary]:
        """
        Get official daily climate summary (high/low temperatures).

        This is what Polymarket uses for settlement.

        Args:
            city_config: City configuration with station ID
            target_date: Date to get summary for (defaults to yesterday)

        Returns:
            DailyClimateSummary or None if not available
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        # Get all observations for the day
        observations = await self.get_station_observations(city_config, hours=48)

        # Filter to target date
        day_obs = [
            obs for obs in observations
            if obs.timestamp.date() == target_date
        ]

        if not day_obs:
            return None

        # Calculate high and low
        temps = [obs.temperature for obs in day_obs]

        return DailyClimateSummary(
            date=target_date,
            temperature_high=max(temps),
            temperature_low=min(temps),
            station_id=city_config.nws_station_id,
            observation_count=len(day_obs),
        )

    async def get_historical_daily_data(
        self,
        city_config: CityConfig,
        start_date: date,
        end_date: date
    ) -> list[DailyClimateSummary]:
        """
        Get historical daily climate data for bias correction training.

        Note: This may require accessing NOAA Climate Data Online or
        other historical archives for data more than a few days old.

        Args:
            city_config: City configuration
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of DailyClimateSummary objects
        """
        # For recent data (last 7 days), we can use observations
        # For older data, we'd need to integrate with NOAA CDO API
        summaries = []

        # Get what we can from recent observations
        days_diff = (date.today() - start_date).days
        if days_diff <= 7:
            observations = await self.get_station_observations(city_config, hours=days_diff * 24)

            # Group by date
            by_date: dict[date, list[float]] = {}
            for obs in observations:
                d = obs.timestamp.date()
                if start_date <= d <= end_date:
                    if d not in by_date:
                        by_date[d] = []
                    by_date[d].append(obs.temperature)

            for d, temps in sorted(by_date.items()):
                if temps:
                    summaries.append(DailyClimateSummary(
                        date=d,
                        temperature_high=max(temps),
                        temperature_low=min(temps),
                        station_id=city_config.nws_station_id,
                        observation_count=len(temps),
                    ))

        return summaries

    async def get_forecast(
        self,
        city_config: CityConfig
    ) -> list[dict]:
        """
        Get NWS point forecast for a location.

        This gives us the NWS's own forecast for comparison.

        Args:
            city_config: City configuration with coordinates

        Returns:
            List of forecast periods
        """
        if city_config.country != "US":
            return []  # NWS forecasts only for US

        # First, get the forecast grid endpoint
        points_url = f"{self.nws_base_url}/points/{city_config.latitude},{city_config.longitude}"

        response = await self.client.get(points_url)
        response.raise_for_status()
        points_data = response.json()

        forecast_url = points_data.get("properties", {}).get("forecast")
        if not forecast_url:
            return []

        # Get the forecast
        response = await self.client.get(forecast_url)
        response.raise_for_status()
        forecast_data = response.json()

        return forecast_data.get("properties", {}).get("periods", [])
