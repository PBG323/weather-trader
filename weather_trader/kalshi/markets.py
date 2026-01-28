"""
Kalshi weather market discovery and data structures.

Discovers KXHIGH (high temperature) events and parses bracket markets.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import httpx

from ..config import CityConfig, CITY_CONFIGS, get_city_config, config


@dataclass
class TemperatureBracket:
    """A single temperature bracket (contract) within a weather event.

    Kalshi weather events consist of ~6 mutually exclusive brackets,
    each covering a temperature range. Prices are in cents (1-99).
    """
    ticker: str
    event_ticker: str
    description: str
    temp_low: Optional[float]  # None for "≤X" brackets
    temp_high: Optional[float]  # None for "≥X" brackets
    yes_price_cents: int = 50  # 1-99 cents
    no_price_cents: int = 50
    volume: int = 0
    open_interest: int = 0

    @property
    def yes_price(self) -> float:
        """Price as 0-1 probability (for strategy layer compatibility)."""
        return self.yes_price_cents / 100.0

    @property
    def no_price(self) -> float:
        """NO price as 0-1 probability."""
        return self.no_price_cents / 100.0

    @property
    def midpoint(self) -> Optional[float]:
        """Temperature midpoint for sorting/display."""
        if self.temp_low is not None and self.temp_high is not None:
            return (self.temp_low + self.temp_high) / 2
        elif self.temp_low is not None:
            return self.temp_low + 1
        elif self.temp_high is not None:
            return self.temp_high - 1
        return None

    # Aliases for backward compatibility with TemperatureOutcome
    @property
    def condition_id(self) -> str:
        return self.ticker

    @property
    def outcome_id(self) -> str:
        return self.ticker

    @property
    def liquidity(self) -> int:
        return self.open_interest


@dataclass
class WeatherMarket:
    """A complete weather event with multiple temperature brackets.

    Represents a single day's high-temperature event for a city on Kalshi.
    """
    event_ticker: str
    series_ticker: str
    city: str
    city_config: Optional[CityConfig]
    target_date: date
    brackets: list[TemperatureBracket] = field(default_factory=list)
    is_active: bool = True
    close_time: Optional[datetime] = None
    resolution_source: str = "NWS Daily Climate Report"
    question: str = ""

    @property
    def outcomes(self) -> list[TemperatureBracket]:
        """Alias for brackets (backward compat with strategy layer)."""
        return self.brackets

    @property
    def temp_unit(self) -> str:
        return "F"

    @property
    def total_volume(self) -> int:
        return sum(b.volume for b in self.brackets)

    @property
    def event_slug(self) -> str:
        """Backward compat alias."""
        return self.event_ticker

    def get_probability_distribution(self) -> list[dict]:
        """Return probability distribution across all brackets."""
        return [
            {
                "ticker": b.ticker,
                "description": b.description,
                "temp_low": b.temp_low,
                "temp_high": b.temp_high,
                "probability": b.yes_price,
            }
            for b in sorted(self.brackets, key=lambda x: x.midpoint or 0)
        ]

    def get_expected_temperature(self) -> Optional[float]:
        """Estimate expected temperature from market prices."""
        total_prob = 0
        weighted_temp = 0
        for b in self.brackets:
            mid = b.midpoint
            if mid is not None:
                prob = b.yes_price
                weighted_temp += mid * prob
                total_prob += prob
        if total_prob > 0:
            return weighted_temp / total_prob
        return None

    def find_best_bracket(self) -> Optional[TemperatureBracket]:
        """Find the bracket with highest YES price (market favorite)."""
        if not self.brackets:
            return None
        return max(self.brackets, key=lambda b: b.yes_price)


class KalshiMarketFinder:
    """Discovers and parses Kalshi weather temperature markets."""

    def __init__(self, base_url: str = ""):
        self._base_url = base_url or config.api.kalshi_api_base_url
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    async def find_weather_markets(
        self,
        city_filter: Optional[str] = None,
        days_ahead: int = 3,
        active_only: bool = True,
    ) -> list[WeatherMarket]:
        """Discover weather temperature markets across all configured cities.

        Args:
            city_filter: If provided, only fetch markets for this city key.
            days_ahead: How many days ahead to look for markets.
            active_only: Only return active/open markets.

        Returns:
            List of WeatherMarket objects with populated brackets.
        """
        # Auto-create client if not using context manager
        created_client = False
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
            created_client = True

        try:
            return await self._find_weather_markets_impl(city_filter, days_ahead, active_only)
        finally:
            if created_client and self._client:
                await self._client.aclose()
                self._client = None

    async def _find_weather_markets_impl(
        self,
        city_filter: Optional[str] = None,
        days_ahead: int = 3,
        active_only: bool = True,
    ) -> list[WeatherMarket]:
        """Internal implementation of market discovery."""
        markets = []
        cities = [city_filter] if city_filter else list(CITY_CONFIGS.keys())

        for city_key in cities:
            try:
                city_config = get_city_config(city_key)
                if not city_config.kalshi_series_ticker:
                    continue

                series_ticker = city_config.kalshi_series_ticker
                events = await self._fetch_events(series_ticker, active_only)

                for event in events:
                    event_ticker = event.get("event_ticker", "")
                    target_date = self._extract_date_from_ticker(event_ticker)

                    if target_date is None:
                        continue

                    # Filter by date range
                    days_until = (target_date - date.today()).days
                    if days_until < 0 or days_until > days_ahead:
                        continue

                    bracket_markets = await self._fetch_brackets(event_ticker)
                    brackets = []

                    for m in bracket_markets:
                        bracket = self._parse_bracket(m, event_ticker)
                        if bracket:
                            brackets.append(bracket)

                    if brackets:
                        question = event.get("title", f"High temperature in {city_config.name} on {target_date}")
                        market = WeatherMarket(
                            event_ticker=event_ticker,
                            series_ticker=series_ticker,
                            city=city_key,
                            city_config=city_config,
                            target_date=target_date,
                            brackets=brackets,
                            is_active=event.get("status", "") in ("open", "active"),
                            question=question,
                        )
                        markets.append(market)

            except Exception as e:
                print(f"Error fetching markets for {city_key}: {e}")
                continue

        return markets

    async def _fetch_events(self, series_ticker: str, active_only: bool = True) -> list[dict]:
        """Fetch events for a series ticker."""
        params = {"series_ticker": series_ticker}
        if active_only:
            params["status"] = "open"

        resp = await self._client.get("/events", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", [])

    async def _fetch_brackets(self, event_ticker: str) -> list[dict]:
        """Fetch all bracket markets for an event."""
        resp = await self._client.get("/markets", params={"event_ticker": event_ticker})
        resp.raise_for_status()
        data = resp.json()
        return data.get("markets", [])

    def _extract_date_from_ticker(self, event_ticker: str) -> Optional[date]:
        """Parse date from event ticker like 'KXHIGHNY-26JAN28' → date(2026, 1, 28).

        Format: SERIES-YYMMMDD where MMM is a 3-letter month abbreviation.
        """
        match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})$', event_ticker)
        if not match:
            return None

        year = 2000 + int(match.group(1))
        month_str = match.group(2)
        day = int(match.group(3))

        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        month = months.get(month_str)
        if month is None:
            return None

        try:
            return date(year, month, day)
        except ValueError:
            return None

    def _parse_bracket(self, market_data: dict, event_ticker: str) -> Optional[TemperatureBracket]:
        """Parse a single bracket market from Kalshi API response."""
        ticker = market_data.get("ticker", "")
        subtitle = market_data.get("subtitle", "") or market_data.get("title", "")

        # Extract temperature range from floor_strike / cap_strike
        floor_strike = market_data.get("floor_strike")
        cap_strike = market_data.get("cap_strike")

        temp_low = None
        temp_high = None

        if floor_strike is not None and cap_strike is not None:
            temp_low = float(floor_strike)
            temp_high = float(cap_strike)
        elif floor_strike is not None:
            # "X or higher" bracket
            temp_low = float(floor_strike)
            temp_high = None
        elif cap_strike is not None:
            # "X or lower" bracket
            temp_low = None
            temp_high = float(cap_strike)
        else:
            # Try to parse from subtitle text
            parsed = self._parse_bracket_range(subtitle)
            if parsed:
                temp_low, temp_high = parsed
            else:
                return None

        # Build description
        if temp_low is None and temp_high is not None:
            description = f"{temp_high:.0f}°F or below"
        elif temp_high is None and temp_low is not None:
            description = f"{temp_low:.0f}°F or above"
        elif temp_low is not None and temp_high is not None:
            description = f"{temp_low:.0f}-{temp_high:.0f}°F"
        else:
            description = subtitle or ticker

        # Extract prices (Kalshi returns yes_bid, yes_ask, last_price in cents)
        yes_price = market_data.get("yes_bid", 50)
        if yes_price is None or yes_price == 0:
            yes_price = market_data.get("last_price", 50) or 50

        no_price = 100 - yes_price

        return TemperatureBracket(
            ticker=ticker,
            event_ticker=event_ticker,
            description=description,
            temp_low=temp_low,
            temp_high=temp_high,
            yes_price_cents=int(yes_price),
            no_price_cents=int(no_price),
            volume=market_data.get("volume", 0) or 0,
            open_interest=market_data.get("open_interest", 0) or 0,
        )

    @staticmethod
    def _parse_bracket_range(text: str) -> Optional[tuple[Optional[float], Optional[float]]]:
        """Parse temperature range from bracket description text.

        Handles formats like:
            "32°F or lower" → (None, 32)
            "33 to 37°F"    → (33, 37)
            "52°F or higher" → (52, None)
        """
        text = text.strip()

        # "X°F or lower" / "X°F or below"
        m = re.match(r'(\d+)\s*°?\s*F?\s+or\s+(lower|below)', text, re.IGNORECASE)
        if m:
            return None, float(m.group(1))

        # "X°F or higher" / "X°F or above"
        m = re.match(r'(\d+)\s*°?\s*F?\s+or\s+(higher|above)', text, re.IGNORECASE)
        if m:
            return float(m.group(1)), None

        # "X to Y°F" or "X-Y°F"
        m = re.match(r'(\d+)\s*(?:to|-)\s*(\d+)\s*°?\s*F?', text, re.IGNORECASE)
        if m:
            return float(m.group(1)), float(m.group(2))

        return None
