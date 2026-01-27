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
    station_id: str  # ICAO code for Weather Underground station
    station_name: str
    latitude: float
    longitude: float
    timezone: str
    country: str  # For API routing (US cities use NWS, international use different sources)
    temp_unit: str = "F"  # F for Fahrenheit, C for Celsius
    polymarket_slug: str = ""  # City name as used in Polymarket slugs

    @property
    def nws_station_id(self) -> str:
        """Backward compatibility alias for station_id."""
        return self.station_id


# Weather Underground Station Mappings - Critical for settlement accuracy
# Polymarket uses Weather Underground data for resolution
CITY_CONFIGS: dict[str, CityConfig] = {
    "nyc": CityConfig(
        name="New York City",
        station_id="KLGA",
        station_name="LaGuardia Airport",
        latitude=40.7769,
        longitude=-73.8740,
        timezone="America/New_York",
        country="US",
        temp_unit="F",
        polymarket_slug="nyc"
    ),
    "atlanta": CityConfig(
        name="Atlanta",
        station_id="KATL",
        station_name="Hartsfield-Jackson Airport",
        latitude=33.6407,
        longitude=-84.4277,
        timezone="America/New_York",
        country="US",
        temp_unit="F",
        polymarket_slug="atlanta"
    ),
    "seattle": CityConfig(
        name="Seattle",
        station_id="KSEA",
        station_name="Seattle-Tacoma Airport",
        latitude=47.4502,
        longitude=-122.3088,
        timezone="America/Los_Angeles",
        country="US",
        temp_unit="F",
        polymarket_slug="seattle"
    ),
    "toronto": CityConfig(
        name="Toronto",
        station_id="CYYZ",
        station_name="Pearson International Airport",
        latitude=43.6777,
        longitude=-79.6248,
        timezone="America/Toronto",
        country="CA",
        temp_unit="C",
        polymarket_slug="toronto"
    ),
    "london": CityConfig(
        name="London",
        station_id="EGLC",
        station_name="London City Airport",
        latitude=51.5048,
        longitude=0.0495,
        timezone="Europe/London",
        country="UK",
        temp_unit="C",
        polymarket_slug="london"
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

    # Polymarket CLOB
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"


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
class PolygonConfig:
    """Polygon blockchain configuration."""
    private_key: str = field(
        default_factory=lambda: os.getenv("POLYGON_PRIVATE_KEY", "")
    )
    rpc_url: str = field(
        default_factory=lambda: os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
    )
    chain_id: int = 137  # Polygon mainnet


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
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
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
