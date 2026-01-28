"""
Configuration module for Weather Trader.

Contains API configurations, city/station mappings, and trading parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CityConfig:
    """Configuration for a tradeable city."""
    name: str
    station_id: str  # ICAO code for NWS station
    station_name: str
    latitude: float
    longitude: float
    timezone: str
    country: str
    temp_unit: str = "F"
    kalshi_series_ticker: str = ""  # Kalshi KXHIGH series ticker for this city

    @property
    def nws_station_id(self) -> str:
        """Backward compatibility alias for station_id."""
        return self.station_id


# NWS Station Mappings - Critical for Kalshi settlement accuracy
# Kalshi settles on NWS Daily Climate Report data
CITY_CONFIGS: dict[str, CityConfig] = {
    "nyc": CityConfig(
        name="New York City",
        station_id="KNYC",
        station_name="Central Park",
        latitude=40.7828,
        longitude=-73.9653,
        timezone="America/New_York",
        country="US",
        temp_unit="F",
        kalshi_series_ticker="KXHIGHNY",
    ),
    "chicago": CityConfig(
        name="Chicago",
        station_id="KMDW",
        station_name="Midway International Airport",
        latitude=41.7868,
        longitude=-87.7522,
        timezone="America/Chicago",
        country="US",
        temp_unit="F",
        kalshi_series_ticker="KXHIGHCHI",
    ),
    "miami": CityConfig(
        name="Miami",
        station_id="KMIA",
        station_name="Miami International Airport",
        latitude=25.7959,
        longitude=-80.2870,
        timezone="America/New_York",
        country="US",
        temp_unit="F",
        kalshi_series_ticker="KXHIGHMIA",
    ),
    "austin": CityConfig(
        name="Austin",
        station_id="KAUS",
        station_name="Austin-Bergstrom International Airport",
        latitude=30.1945,
        longitude=-97.6699,
        timezone="America/Chicago",
        country="US",
        temp_unit="F",
        kalshi_series_ticker="KXHIGHAUS",
    ),
    "la": CityConfig(
        name="Los Angeles",
        station_id="KLAX",
        station_name="Los Angeles International Airport",
        latitude=33.9416,
        longitude=-118.4085,
        timezone="America/Los_Angeles",
        country="US",
        temp_unit="F",
        kalshi_series_ticker="KXHIGHLA",
    ),
    "denver": CityConfig(
        name="Denver",
        station_id="KDEN",
        station_name="Denver International Airport",
        latitude=39.8561,
        longitude=-104.6737,
        timezone="America/Denver",
        country="US",
        temp_unit="F",
        kalshi_series_ticker="KXHIGHDEN",
    ),
    "philadelphia": CityConfig(
        name="Philadelphia",
        station_id="KPHL",
        station_name="Philadelphia International Airport",
        latitude=39.8721,
        longitude=-75.2411,
        timezone="America/New_York",
        country="US",
        temp_unit="F",
        kalshi_series_ticker="KXHIGHPHL",
    ),
}


@dataclass
class APIConfig:
    """API configuration settings."""
    # Open-Meteo (no API key required)
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"
    open_meteo_historical_url: str = "https://archive-api.open-meteo.com/v1"

    # Tomorrow.io
    tomorrow_io_api_key: str = field(default_factory=lambda: os.getenv("TOMORROW_IO_API_KEY", ""))
    tomorrow_io_base_url: str = "https://api.tomorrow.io/v4"

    # NWS API (no API key required, but needs User-Agent)
    nws_base_url: str = "https://api.weather.gov"
    nws_user_agent: str = "(WeatherTrader, contact@example.com)"

    # Kalshi API
    kalshi_api_base_url: str = field(
        default_factory=lambda: os.getenv("KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
    )


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing
    max_position_percent: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PERCENT", "5.0"))
    )
    kelly_fraction: float = field(
        default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25"))
    )

    # Risk management
    daily_loss_limit_percent: float = field(
        default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT_PERCENT", "10.0"))
    )

    # Edge thresholds
    min_edge_threshold: float = field(
        default_factory=lambda: float(os.getenv("MIN_EDGE_THRESHOLD", "0.05"))
    )
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.70"))
    )

    # Execution
    use_limit_orders: bool = True
    limit_order_offset: float = 0.01  # Place limits 1% better than current price


@dataclass
class KalshiConfig:
    """Kalshi API authentication configuration."""
    key_id: str = field(
        default_factory=lambda: os.getenv("KALSHI_KEY_ID", "")
    )
    private_key_path: str = field(
        default_factory=lambda: os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    )
    api_base_url: str = field(
        default_factory=lambda: os.getenv("KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
    )


@dataclass
class AlertConfig:
    """Alert configuration for notifications."""
    discord_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL") or None
    )
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN") or None
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID") or None
    )


@dataclass
class Config:
    """Main configuration container."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    env: str = field(
        default_factory=lambda: os.getenv("ENV", "development")
    )

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# Global configuration instance
config = Config()


def get_city_config(city_key: str) -> CityConfig:
    """Get configuration for a specific city."""
    city_key = city_key.lower()
    if city_key not in CITY_CONFIGS:
        raise ValueError(f"Unknown city: {city_key}. Valid options: {list(CITY_CONFIGS.keys())}")
    return CITY_CONFIGS[city_key]


def get_all_cities() -> list[str]:
    """Get list of all configured city keys."""
    return list(CITY_CONFIGS.keys())
