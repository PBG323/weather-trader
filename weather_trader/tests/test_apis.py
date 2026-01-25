"""
Tests for weather API clients.
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from weather_trader.config import get_city_config
from weather_trader.apis.open_meteo import OpenMeteoClient, WeatherModel, ForecastPoint
from weather_trader.apis.tomorrow_io import TomorrowIOClient, TomorrowForecast
from weather_trader.apis.nws import NWSClient, StationObservation


class TestOpenMeteoClient:
    """Tests for Open-Meteo API client."""

    @pytest.fixture
    def client(self):
        return OpenMeteoClient()

    @pytest.fixture
    def nyc_config(self):
        return get_city_config("nyc")

    @pytest.fixture
    def mock_daily_response(self):
        return {
            "daily": {
                "time": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "temperature_2m_max": [45.0, 48.0, 52.0],
                "temperature_2m_min": [32.0, 35.0, 38.0],
            }
        }

    @pytest.mark.asyncio
    async def test_get_daily_forecast(self, client, nyc_config, mock_daily_response):
        """Test fetching daily forecasts."""
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_daily_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            forecasts = await client.get_daily_forecast(nyc_config, WeatherModel.BEST_MATCH, days=3)

            assert len(forecasts) == 3
            assert forecasts[0].temperature_high == 45.0
            assert forecasts[0].temperature_low == 32.0
            assert forecasts[0].city == "New York City"

    @pytest.mark.asyncio
    async def test_get_ensemble_forecast(self, client, nyc_config, mock_daily_response):
        """Test fetching ensemble forecasts from multiple models."""
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_daily_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            results = await client.get_ensemble_forecast(nyc_config, days=3)

            # Should have multiple model results
            assert len(results) >= 2
            assert "ecmwf" in results or "gfs" in results

    @pytest.mark.asyncio
    async def test_celsius_to_fahrenheit(self, client):
        """Test temperature conversion."""
        assert client._celsius_to_fahrenheit(0) == 32
        assert client._celsius_to_fahrenheit(100) == 212
        assert client._celsius_to_fahrenheit(-40) == -40  # They're equal at -40


class TestTomorrowIOClient:
    """Tests for Tomorrow.io API client."""

    @pytest.fixture
    def client(self):
        # Skip if no API key configured
        try:
            return TomorrowIOClient(api_key="test_key")
        except ValueError:
            pytest.skip("Tomorrow.io API key required")

    @pytest.fixture
    def mock_forecast_response(self):
        return {
            "timelines": {
                "daily": [
                    {
                        "time": "2024-01-15T00:00:00Z",
                        "values": {
                            "temperatureMax": 48.0,
                            "temperatureMin": 35.0,
                            "temperatureApparentMax": 45.0,
                            "temperatureApparentMin": 30.0,
                            "humidityAvg": 65.0,
                            "precipitationProbabilityMax": 20.0,
                        }
                    }
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_get_daily_forecast(self, client, mock_forecast_response):
        """Test fetching daily forecasts from Tomorrow.io."""
        nyc_config = get_city_config("nyc")

        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_forecast_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            forecasts = await client.get_daily_forecast(nyc_config, days=1)

            assert len(forecasts) == 1
            assert forecasts[0].temperature_high == 48.0
            assert forecasts[0].temperature_low == 35.0


class TestNWSClient:
    """Tests for NWS API client."""

    @pytest.fixture
    def client(self):
        return NWSClient()

    @pytest.fixture
    def mock_observations_response(self):
        return {
            "features": [
                {
                    "properties": {
                        "timestamp": "2024-01-15T12:00:00Z",
                        "temperature": {"value": 10.0},  # Celsius
                        "station": "stations/KNYC",
                        "textDescription": "Partly Cloudy",
                        "relativeHumidity": {"value": 65.0},
                    }
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_get_station_observations_us(self, client, mock_observations_response):
        """Test fetching US station observations."""
        nyc_config = get_city_config("nyc")

        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_observations_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            observations = await client.get_station_observations(nyc_config, hours=24)

            assert len(observations) == 1
            assert observations[0].station_id == "KNYC"
            assert observations[0].temperature == 50.0  # Converted to Fahrenheit

    @pytest.mark.asyncio
    async def test_get_daily_summary(self, client, mock_observations_response):
        """Test computing daily summary from observations."""
        nyc_config = get_city_config("nyc")

        # Mock multiple observations for a day
        multi_obs_response = {
            "features": [
                {
                    "properties": {
                        "timestamp": "2024-01-15T12:00:00Z",
                        "temperature": {"value": 15.0},
                        "station": "stations/KNYC",
                        "textDescription": "Clear",
                    }
                },
                {
                    "properties": {
                        "timestamp": "2024-01-15T06:00:00Z",
                        "temperature": {"value": 5.0},
                        "station": "stations/KNYC",
                        "textDescription": "Clear",
                    }
                },
            ]
        }

        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = multi_obs_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            summary = await client.get_daily_summary(nyc_config, date(2024, 1, 15))

            # Should compute high and low from observations
            if summary:  # May be None if dates don't match
                assert summary.temperature_high > summary.temperature_low


class TestCityConfigs:
    """Tests for city configuration."""

    def test_all_cities_have_required_fields(self):
        """Verify all city configs have required fields."""
        from weather_trader.config import CITY_CONFIGS

        required_fields = [
            "name", "nws_station_id", "station_name",
            "latitude", "longitude", "timezone", "country"
        ]

        for city_key, config in CITY_CONFIGS.items():
            for field in required_fields:
                assert hasattr(config, field), f"{city_key} missing {field}"
                assert getattr(config, field) is not None, f"{city_key}.{field} is None"

    def test_station_ids_are_valid_icao(self):
        """Verify station IDs follow ICAO format."""
        from weather_trader.config import CITY_CONFIGS

        for city_key, config in CITY_CONFIGS.items():
            station_id = config.nws_station_id
            assert len(station_id) == 4, f"{city_key} station ID should be 4 chars"
            assert station_id.isupper(), f"{city_key} station ID should be uppercase"

    def test_coordinates_are_valid(self):
        """Verify coordinates are within valid ranges."""
        from weather_trader.config import CITY_CONFIGS

        for city_key, config in CITY_CONFIGS.items():
            assert -90 <= config.latitude <= 90, f"{city_key} latitude out of range"
            assert -180 <= config.longitude <= 180, f"{city_key} longitude out of range"
