"""
Weather Market Discovery

Finds and parses Polymarket weather temperature markets.

Weather markets are typically structured as:
- "Will [City] high temperature be over/under X°F on [Date]?"
- Binary YES/NO contracts
- Settlement based on official NWS/NOAA readings
"""

import re
import httpx
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
from enum import Enum

from ..config import config, CITY_CONFIGS, CityConfig


class MarketType(Enum):
    """Type of temperature market."""
    HIGH_OVER = "high_over"      # High temp over threshold
    HIGH_UNDER = "high_under"    # High temp under threshold
    LOW_OVER = "low_over"        # Low temp over threshold
    LOW_UNDER = "low_under"      # Low temp under threshold
    RANGE = "range"              # Temp in a range


@dataclass
class WeatherMarket:
    """A weather temperature market on Polymarket."""
    # Market identifiers
    condition_id: str
    question_id: str
    market_slug: str

    # Market details
    city: str
    city_config: Optional[CityConfig]
    target_date: date
    threshold: float  # Temperature threshold in Fahrenheit
    market_type: MarketType

    # Token info
    yes_token_id: str
    no_token_id: str

    # Current prices
    yes_price: float  # 0-1
    no_price: float   # 0-1

    # Liquidity info
    volume: float
    liquidity: float

    # Market state
    is_active: bool
    end_date: datetime

    # Raw question text
    question: str

    def implied_probability(self, for_yes: bool = True) -> float:
        """Get market-implied probability."""
        return self.yes_price if for_yes else self.no_price

    @property
    def is_over_market(self) -> bool:
        """Check if this is an 'over' market."""
        return self.market_type in [MarketType.HIGH_OVER, MarketType.LOW_OVER]

    @property
    def is_high_temp_market(self) -> bool:
        """Check if this is a high temperature market."""
        return self.market_type in [MarketType.HIGH_OVER, MarketType.HIGH_UNDER]


class WeatherMarketFinder:
    """
    Discovers weather temperature markets on Polymarket.
    """

    def __init__(self):
        self.gamma_url = config.api.polymarket_gamma_url
        self.client = httpx.AsyncClient(timeout=30.0)

        # Regex patterns for parsing weather questions
        self.city_patterns = {
            "nyc": r"(?:NYC|New York|New York City)",
            "atlanta": r"Atlanta",
            "seattle": r"Seattle",
            "toronto": r"Toronto",
            "london": r"London",
        }

        self.temp_pattern = re.compile(
            r"(?:high|low)?\s*(?:temperature|temp)?\s*(?:be\s+)?(?:over|under|above|below)\s+(\d+(?:\.\d+)?)\s*(?:°?F|degrees?)?",
            re.IGNORECASE
        )

        self.date_pattern = re.compile(
            r"(?:on|for)\s+(\w+\s+\d{1,2}(?:,?\s*\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)",
            re.IGNORECASE
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def find_weather_markets(
        self,
        city_filter: Optional[str] = None,
        active_only: bool = True
    ) -> list[WeatherMarket]:
        """
        Find all weather temperature markets.

        Args:
            city_filter: Optional city key to filter by
            active_only: Only return active markets

        Returns:
            List of WeatherMarket objects
        """
        # Search for weather-related markets
        search_terms = ["temperature", "weather", "degrees"]
        all_markets = []

        for term in search_terms:
            try:
                markets = await self._search_markets(term)
                all_markets.extend(markets)
            except Exception as e:
                print(f"Warning: Failed to search for '{term}': {e}")

        # Deduplicate by condition_id
        seen = set()
        unique_markets = []
        for m in all_markets:
            if m.condition_id not in seen:
                seen.add(m.condition_id)
                unique_markets.append(m)

        # Filter by city if specified
        if city_filter:
            city_filter = city_filter.lower()
            unique_markets = [
                m for m in unique_markets
                if m.city.lower() == city_filter
            ]

        # Filter active only
        if active_only:
            unique_markets = [m for m in unique_markets if m.is_active]

        return unique_markets

    async def _search_markets(self, search_term: str) -> list[WeatherMarket]:
        """Search Polymarket for markets matching a term."""
        endpoint = f"{self.gamma_url}/markets"

        params = {
            "search": search_term,
            "active": "true",
            "closed": "false",
        }

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        markets = []
        for market_data in data:
            parsed = self._parse_market(market_data)
            if parsed:
                markets.append(parsed)

        return markets

    async def get_market_by_id(self, condition_id: str) -> Optional[WeatherMarket]:
        """
        Get a specific market by condition ID.

        Args:
            condition_id: The market condition ID

        Returns:
            WeatherMarket or None if not found
        """
        endpoint = f"{self.gamma_url}/markets/{condition_id}"

        try:
            response = await self.client.get(endpoint)
            response.raise_for_status()
            data = response.json()
            return self._parse_market(data)
        except httpx.HTTPStatusError:
            return None

    async def get_market_prices(self, market: WeatherMarket) -> dict:
        """
        Get current orderbook prices for a market.

        Args:
            market: WeatherMarket to get prices for

        Returns:
            Dictionary with bid/ask prices and depth
        """
        clob_url = config.api.polymarket_clob_url
        endpoint = f"{clob_url}/book"

        params = {"token_id": market.yes_token_id}

        try:
            response = await self.client.get(endpoint, params=params)
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
                "bid_depth": sum(float(b["size"]) for b in bids[:5]),
                "ask_depth": sum(float(a["size"]) for a in asks[:5]),
            }
        except Exception as e:
            print(f"Error fetching orderbook: {e}")
            return {
                "best_bid": market.yes_price * 0.98,
                "best_ask": market.yes_price * 1.02,
                "mid_price": market.yes_price,
                "spread": 0.04,
                "bid_depth": 0,
                "ask_depth": 0,
            }

    def _parse_market(self, data: dict) -> Optional[WeatherMarket]:
        """
        Parse market data into a WeatherMarket object.

        Returns None if the market is not a temperature market.
        """
        question = data.get("question", "")

        # Check if this is a temperature market
        if not self._is_temperature_market(question):
            return None

        # Extract city
        city = self._extract_city(question)
        if not city:
            return None

        city_config = CITY_CONFIGS.get(city)

        # Extract threshold temperature
        threshold = self._extract_threshold(question)
        if threshold is None:
            return None

        # Extract target date
        target_date = self._extract_date(question, data)
        if target_date is None:
            return None

        # Determine market type
        market_type = self._extract_market_type(question)

        # Extract token IDs
        tokens = data.get("tokens", [])
        yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
        no_token = next((t for t in tokens if t.get("outcome") == "No"), None)

        if not yes_token or not no_token:
            return None

        # Parse end date
        end_date_str = data.get("endDate") or data.get("end_date_iso")
        try:
            if end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            else:
                end_date = datetime.now()
        except ValueError:
            end_date = datetime.now()

        return WeatherMarket(
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            question_id=data.get("questionId", data.get("question_id", "")),
            market_slug=data.get("slug", ""),
            city=city,
            city_config=city_config,
            target_date=target_date,
            threshold=threshold,
            market_type=market_type,
            yes_token_id=yes_token.get("token_id", ""),
            no_token_id=no_token.get("token_id", ""),
            yes_price=float(yes_token.get("price", 0.5)),
            no_price=float(no_token.get("price", 0.5)),
            volume=float(data.get("volume", 0)),
            liquidity=float(data.get("liquidity", 0)),
            is_active=data.get("active", True) and not data.get("closed", False),
            end_date=end_date,
            question=question,
        )

    def _is_temperature_market(self, question: str) -> bool:
        """Check if question is about temperature."""
        temp_keywords = [
            "temperature", "degrees", "°F", "°C",
            "high temp", "low temp", "thermometer"
        ]
        question_lower = question.lower()
        return any(kw in question_lower for kw in temp_keywords)

    def _extract_city(self, question: str) -> Optional[str]:
        """Extract city from question text."""
        for city_key, pattern in self.city_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                return city_key
        return None

    def _extract_threshold(self, question: str) -> Optional[float]:
        """Extract temperature threshold from question."""
        match = self.temp_pattern.search(question)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Try to find any number followed by degree indicators
        number_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|degrees?|F)", question)
        if number_match:
            try:
                return float(number_match.group(1))
            except ValueError:
                pass

        return None

    def _extract_date(self, question: str, data: dict) -> Optional[date]:
        """Extract target date from question or market data."""
        # Try to find date in question
        match = self.date_pattern.search(question)
        if match:
            date_str = match.group(1)
            # Try various date formats
            for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%m/%d/%Y", "%m/%d/%y"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue

        # Fall back to market end date
        end_date_str = data.get("endDate") or data.get("end_date_iso")
        if end_date_str:
            try:
                return datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).date()
            except ValueError:
                pass

        return None

    def _extract_market_type(self, question: str) -> MarketType:
        """Determine market type from question."""
        question_lower = question.lower()

        is_high = "high" in question_lower or "maximum" in question_lower
        is_low = "low" in question_lower or "minimum" in question_lower
        is_over = any(w in question_lower for w in ["over", "above", "exceed", "more than"])

        if is_high:
            return MarketType.HIGH_OVER if is_over else MarketType.HIGH_UNDER
        elif is_low:
            return MarketType.LOW_OVER if is_over else MarketType.LOW_UNDER
        else:
            # Default to high temp over
            return MarketType.HIGH_OVER if is_over else MarketType.HIGH_UNDER
