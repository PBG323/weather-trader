"""
Weather Market Discovery

Finds and parses Polymarket weather temperature markets.

Polymarket weather markets structure:
- URL: highest-temperature-in-{city}-on-{month}-{day}
- Multi-outcome markets with temperature ranges (e.g., "20-21°F", "≤15°F", "≥26°F")
- Resolution via Weather Underground station data
"""

import re
import httpx
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional
from enum import Enum

from ..config import config, CITY_CONFIGS, CityConfig


@dataclass
class TemperatureOutcome:
    """A single temperature outcome in a multi-outcome market."""
    outcome_id: str  # Market ID for this outcome
    condition_id: str
    description: str  # e.g., "20-21°F" or "≤15°F"
    temp_low: Optional[float]  # Lower bound (None if "≤X")
    temp_high: Optional[float]  # Upper bound (None if "≥X")
    yes_token_id: str
    no_token_id: str
    yes_price: float  # 0-1, probability
    volume: float
    liquidity: float

    @property
    def midpoint(self) -> float:
        """Get midpoint temperature for this range."""
        if self.temp_low is None:
            return self.temp_high - 1  # For "≤X" ranges
        if self.temp_high is None:
            return self.temp_low + 1  # For "≥X" ranges
        return (self.temp_low + self.temp_high) / 2


@dataclass
class WeatherMarket:
    """A weather temperature market on Polymarket with multiple outcomes."""
    # Event identifiers
    event_id: str
    event_slug: str

    # Market details
    city: str
    city_config: Optional[CityConfig]
    target_date: date
    temp_unit: str  # "F" or "C"

    # All temperature outcome options
    outcomes: list[TemperatureOutcome] = field(default_factory=list)

    # Aggregate info
    total_volume: float = 0.0
    total_liquidity: float = 0.0

    # Market state
    is_active: bool = True
    end_date: Optional[datetime] = None

    # Raw question text
    question: str = ""

    # Resolution source
    resolution_source: str = ""

    def get_probability_distribution(self) -> dict[float, float]:
        """
        Get probability distribution over temperature midpoints.

        Returns:
            Dict mapping temperature midpoint to probability
        """
        return {o.midpoint: o.yes_price for o in self.outcomes}

    def get_expected_temperature(self) -> float:
        """Calculate expected temperature from market prices."""
        total_prob = sum(o.yes_price for o in self.outcomes)
        if total_prob == 0:
            return 0
        return sum(o.midpoint * o.yes_price for o in self.outcomes) / total_prob

    def find_best_outcome(self, forecast_temp: float) -> Optional[TemperatureOutcome]:
        """Find the outcome that contains the forecasted temperature."""
        for o in self.outcomes:
            if o.temp_low is None:  # "≤X" range
                if forecast_temp <= o.temp_high:
                    return o
            elif o.temp_high is None:  # "≥X" range
                if forecast_temp >= o.temp_low:
                    return o
            else:  # Normal range
                if o.temp_low <= forecast_temp <= o.temp_high:
                    return o
        return None


class WeatherMarketFinder:
    """
    Discovers weather temperature markets on Polymarket.

    Uses the Gamma API with slug-based queries for reliable market discovery.
    """

    def __init__(self):
        self.gamma_url = config.api.polymarket_gamma_url
        self.client = httpx.AsyncClient(timeout=30.0)

        # Regex for parsing temperature ranges like "20-21°F", "≤15°F", "≥26°F"
        self.range_pattern = re.compile(
            r"(?:(\d+)-(\d+)|[≤<]=?(\d+)|[≥>]=?(\d+))\s*°?([FC])?",
            re.IGNORECASE
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _build_market_slug(self, city_slug: str, target_date: date) -> str:
        """Build the Polymarket slug for a weather market."""
        month_name = target_date.strftime("%B").lower()
        day = target_date.day
        return f"highest-temperature-in-{city_slug}-on-{month_name}-{day}"

    async def find_weather_markets(
        self,
        city_filter: Optional[str] = None,
        days_ahead: int = 3,
        active_only: bool = True
    ) -> list[WeatherMarket]:
        """
        Find all weather temperature markets.

        Args:
            city_filter: Optional city key to filter by
            days_ahead: Number of days to look ahead for markets
            active_only: Only return active markets

        Returns:
            List of WeatherMarket objects
        """
        markets = []
        cities = [city_filter] if city_filter else list(CITY_CONFIGS.keys())
        errors = []

        for city_key in cities:
            city_config = CITY_CONFIGS.get(city_key)
            if not city_config:
                continue

            # Check markets for today and upcoming days
            for days in range(days_ahead + 1):
                target_date = date.today() + timedelta(days=days)
                slug = self._build_market_slug(city_config.polymarket_slug, target_date)

                try:
                    market = await self._fetch_market_by_slug(slug, city_key, city_config)
                    if market:
                        if not active_only or market.is_active:
                            markets.append(market)
                            print(f"[Markets] Found: {slug} with {len(market.outcomes)} outcomes")
                except Exception as e:
                    errors.append(f"{slug}: {str(e)}")

        if errors and len(errors) < 5:
            print(f"[Markets] Errors: {errors}")

        return markets

    async def get_market_for_city_date(
        self,
        city_key: str,
        target_date: date
    ) -> Optional[WeatherMarket]:
        """
        Get the weather market for a specific city and date.

        Args:
            city_key: City key (e.g., "nyc", "london")
            target_date: The date to get the market for

        Returns:
            WeatherMarket or None if not found
        """
        city_config = CITY_CONFIGS.get(city_key.lower())
        if not city_config:
            return None

        slug = self._build_market_slug(city_config.polymarket_slug, target_date)
        return await self._fetch_market_by_slug(slug, city_key, city_config)

    async def _fetch_market_by_slug(
        self,
        slug: str,
        city_key: str,
        city_config: CityConfig
    ) -> Optional[WeatherMarket]:
        """Fetch a market from Polymarket by its slug."""
        endpoint = f"{self.gamma_url}/events"
        params = {"slug": slug}

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()

        # The API returns a list; we want the first match
        if not data or len(data) == 0:
            return None

        event = data[0] if isinstance(data, list) else data
        return self._parse_event(event, city_key, city_config)

    def _parse_event(
        self,
        event: dict,
        city_key: str,
        city_config: CityConfig
    ) -> Optional[WeatherMarket]:
        """Parse a Polymarket event into a WeatherMarket."""
        # Extract target date from slug
        slug = event.get("slug", "")
        target_date = self._extract_date_from_slug(slug)

        if not target_date:
            # Try from end date
            end_str = event.get("endDate") or event.get("end_date_iso")
            if end_str:
                try:
                    target_date = datetime.fromisoformat(
                        end_str.replace("Z", "+00:00")
                    ).date()
                except ValueError:
                    return None
            else:
                return None

        # Parse end datetime
        end_date = None
        end_str = event.get("endDate") or event.get("end_date_iso")
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Parse all outcome markets
        outcomes = []
        sub_markets = event.get("markets", [])

        for market in sub_markets:
            outcome = self._parse_outcome(market, city_config.temp_unit)
            if outcome:
                outcomes.append(outcome)

        # Sort outcomes by temperature
        outcomes.sort(key=lambda o: o.midpoint)

        # Calculate totals
        total_volume = sum(o.volume for o in outcomes)
        total_liquidity = sum(o.liquidity for o in outcomes)

        # Check if active
        is_active = event.get("active", True) and not event.get("closed", False)

        # Get resolution source from description
        description = event.get("description", "")
        resolution_source = "Weather Underground"
        if city_config.station_name:
            resolution_source = f"Weather Underground - {city_config.station_name}"

        return WeatherMarket(
            event_id=str(event.get("id", "")),
            event_slug=slug,
            city=city_key,
            city_config=city_config,
            target_date=target_date,
            temp_unit=city_config.temp_unit,
            outcomes=outcomes,
            total_volume=total_volume,
            total_liquidity=total_liquidity,
            is_active=is_active,
            end_date=end_date,
            question=event.get("title", ""),
            resolution_source=resolution_source,
        )

    def _parse_outcome(self, market: dict, temp_unit: str) -> Optional[TemperatureOutcome]:
        """Parse a single outcome market."""
        # Try groupItemTitle first (more reliable for range), then question
        question = market.get("groupItemTitle", "") or market.get("question", "")

        # Parse temperature range from question/title
        temp_low, temp_high = self._parse_temp_range(question)

        if temp_low is None and temp_high is None:
            # Try parsing from the full question if groupItemTitle didn't work
            full_question = market.get("question", "")
            temp_low, temp_high = self._parse_temp_range(full_question)

        if temp_low is None and temp_high is None:
            return None

        # Get token IDs
        tokens = market.get("tokens", [])
        clob_token_ids = market.get("clobTokenIds", [])

        # Try to get Yes/No token IDs
        yes_token_id = ""
        no_token_id = ""

        if clob_token_ids and len(clob_token_ids) >= 2:
            yes_token_id = str(clob_token_ids[0])
            no_token_id = str(clob_token_ids[1])
        elif tokens:
            for t in tokens:
                if t.get("outcome") == "Yes":
                    yes_token_id = t.get("token_id", "")
                elif t.get("outcome") == "No":
                    no_token_id = t.get("token_id", "")

        # Get price (probability) - outcomePrices is array of strings like ["0.725", "0.275"]
        yes_price = 0.5
        outcome_prices = market.get("outcomePrices", [])
        if isinstance(outcome_prices, list) and len(outcome_prices) > 0:
            try:
                yes_price = float(outcome_prices[0])
            except (ValueError, TypeError):
                pass
        elif isinstance(outcome_prices, str):
            try:
                yes_price = float(outcome_prices.split(",")[0])
            except (ValueError, TypeError):
                pass

        # Fallback to other price fields
        if yes_price == 0.5:
            try:
                yes_price = float(market.get("bestAsk") or market.get("lastTradePrice") or 0.5)
            except (ValueError, TypeError):
                pass

        # Use groupItemTitle as description if available
        description = market.get("groupItemTitle", "") or question

        return TemperatureOutcome(
            outcome_id=str(market.get("id", "")),
            condition_id=market.get("conditionId", ""),
            description=description,
            temp_low=temp_low,
            temp_high=temp_high,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_price=yes_price,
            volume=float(market.get("volume") or 0),
            liquidity=float(market.get("liquidity") or 0),
        )

    def _parse_temp_range(self, text: str) -> tuple[Optional[float], Optional[float]]:
        """
        Parse temperature range from text.

        Examples:
            "20-21°F" -> (20, 21)
            "≤15°F" or "15°F or below" -> (None, 15)
            "≥26°F" or "26°F or higher" -> (26, None)
        """
        # Handle "X or below" / "X or lower"
        below_match = re.search(r"(\d+)\s*°?[FC]?\s+or\s+(?:below|lower)", text, re.IGNORECASE)
        if below_match:
            return (None, float(below_match.group(1)))

        # Handle "X or above" / "X or higher"
        above_match = re.search(r"(\d+)\s*°?[FC]?\s+or\s+(?:above|higher)", text, re.IGNORECASE)
        if above_match:
            return (float(above_match.group(1)), None)

        # Handle "≤X" or "<=X"
        lte_match = re.search(r"[≤<]=?\s*(\d+)", text)
        if lte_match:
            return (None, float(lte_match.group(1)))

        # Handle "≥X" or ">=X"
        gte_match = re.search(r"[≥>]=?\s*(\d+)", text)
        if gte_match:
            return (float(gte_match.group(1)), None)

        # Handle "X-Y" range
        range_match = re.search(r"(\d+)\s*-\s*(\d+)", text)
        if range_match:
            return (float(range_match.group(1)), float(range_match.group(2)))

        # Handle single temperature (rare)
        single_match = re.search(r"(\d+)\s*°?[FC]", text)
        if single_match:
            temp = float(single_match.group(1))
            return (temp, temp)

        return (None, None)

    def _extract_date_from_slug(self, slug: str) -> Optional[date]:
        """Extract the target date from a market slug."""
        # Pattern: highest-temperature-in-city-on-month-day
        match = re.search(r"on-(\w+)-(\d+)$", slug)
        if not match:
            return None

        month_name = match.group(1)
        day = int(match.group(2))

        # Map month name to number
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

        month = months.get(month_name.lower())
        if not month:
            return None

        # Assume current year, but handle year boundary
        year = date.today().year
        try:
            target = date(year, month, day)
            # If date is more than 6 months in the past, assume next year
            if (date.today() - target).days > 180:
                target = date(year + 1, month, day)
            return target
        except ValueError:
            return None

    async def get_orderbook(self, token_id: str) -> dict:
        """
        Get the orderbook for a specific token.

        Args:
            token_id: The CLOB token ID

        Returns:
            Orderbook with bids, asks, best prices
        """
        clob_url = config.api.polymarket_clob_url
        endpoint = f"{clob_url}/book"

        try:
            response = await self.client.get(endpoint, params={"token_id": token_id})
            response.raise_for_status()
            data = response.json()

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else 0
            best_ask = float(asks[0]["price"]) if asks else 1

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": (best_bid + best_ask) / 2,
                "spread": best_ask - best_bid,
                "bid_depth": sum(float(b.get("size", 0)) for b in bids[:5]),
                "ask_depth": sum(float(a.get("size", 0)) for a in asks[:5]),
            }
        except Exception as e:
            return {
                "best_bid": 0,
                "best_ask": 1,
                "mid_price": 0.5,
                "spread": 1.0,
                "bid_depth": 0,
                "ask_depth": 0,
                "error": str(e),
            }
