"""
Tests for Kalshi integration module.
"""

import pytest
from datetime import date

from weather_trader.kalshi.markets import (
    KalshiMarketFinder, WeatherMarket, TemperatureBracket
)
from weather_trader.kalshi.client import OrderResult


class TestTemperatureBracket:
    """Tests for TemperatureBracket price conversion and properties."""

    def test_yes_price_conversion(self):
        """Cents → 0-1 float conversion."""
        b = TemperatureBracket(
            ticker="T", event_ticker="E", description="50-52°F",
            temp_low=50, temp_high=52,
            yes_price_cents=42, no_price_cents=58,
        )
        assert b.yes_price == pytest.approx(0.42)
        assert b.no_price == pytest.approx(0.58)

    def test_midpoint_range(self):
        b = TemperatureBracket(
            ticker="T", event_ticker="E", description="50-54°F",
            temp_low=50, temp_high=54,
        )
        assert b.midpoint == 52.0

    def test_midpoint_lower_bound(self):
        """'X or below' bracket."""
        b = TemperatureBracket(
            ticker="T", event_ticker="E", description="45°F or below",
            temp_low=None, temp_high=45,
        )
        assert b.midpoint == 44.0

    def test_midpoint_upper_bound(self):
        """'X or above' bracket."""
        b = TemperatureBracket(
            ticker="T", event_ticker="E", description="60°F or above",
            temp_low=60, temp_high=None,
        )
        assert b.midpoint == 61.0

    def test_backward_compat_aliases(self):
        b = TemperatureBracket(
            ticker="KXHIGHNY-26JAN28-T50", event_ticker="E",
            description="d", temp_low=50, temp_high=52,
            open_interest=123,
        )
        assert b.condition_id == "KXHIGHNY-26JAN28-T50"
        assert b.outcome_id == "KXHIGHNY-26JAN28-T50"
        assert b.liquidity == 123


class TestTickerDateParsing:
    """Tests for extracting dates from Kalshi event tickers."""

    def setup_method(self):
        self.finder = KalshiMarketFinder.__new__(KalshiMarketFinder)

    def test_parse_standard_ticker(self):
        d = self.finder._extract_date_from_ticker("KXHIGHNY-26JAN28")
        assert d == date(2026, 1, 28)

    def test_parse_feb(self):
        d = self.finder._extract_date_from_ticker("KXHIGHCHI-26FEB05")
        assert d == date(2026, 2, 5)

    def test_parse_dec(self):
        d = self.finder._extract_date_from_ticker("KXHIGHLA-25DEC31")
        assert d == date(2025, 12, 31)

    def test_invalid_ticker(self):
        d = self.finder._extract_date_from_ticker("RANDOM_TICKER")
        assert d is None


class TestBracketRangeParsing:
    """Tests for parsing temperature ranges from text descriptions."""

    def test_or_lower(self):
        result = KalshiMarketFinder._parse_bracket_range("32°F or lower")
        assert result == (None, 32.0)

    def test_or_below(self):
        result = KalshiMarketFinder._parse_bracket_range("25 or below")
        assert result == (None, 25.0)

    def test_or_higher(self):
        result = KalshiMarketFinder._parse_bracket_range("52°F or higher")
        assert result == (52.0, None)

    def test_or_above(self):
        result = KalshiMarketFinder._parse_bracket_range("60 or above")
        assert result == (60.0, None)

    def test_range_to(self):
        result = KalshiMarketFinder._parse_bracket_range("33 to 37°F")
        assert result == (33.0, 37.0)

    def test_range_dash(self):
        result = KalshiMarketFinder._parse_bracket_range("40-45°F")
        assert result == (40.0, 45.0)

    def test_unrecognized(self):
        result = KalshiMarketFinder._parse_bracket_range("something weird")
        assert result is None


class TestOrderResult:
    """Tests for OrderResult price conversion."""

    def test_filled_price_property(self):
        r = OrderResult(success=True, filled_price_cents=75)
        assert r.filled_price == pytest.approx(0.75)

    def test_filled_size_property(self):
        r = OrderResult(success=True, filled_count=10)
        assert r.filled_size == 10.0

    def test_failed_order(self):
        r = OrderResult(success=False, message="Insufficient balance")
        assert r.filled_price == 0.0
        assert r.filled_size == 0.0


class TestWeatherMarket:
    """Tests for WeatherMarket."""

    def _make_market(self):
        from weather_trader.config import get_city_config
        brackets = [
            TemperatureBracket(
                ticker="T1", event_ticker="E", description="45°F or below",
                temp_low=None, temp_high=45, yes_price_cents=10,
            ),
            TemperatureBracket(
                ticker="T2", event_ticker="E", description="46-50°F",
                temp_low=46, temp_high=50, yes_price_cents=30,
            ),
            TemperatureBracket(
                ticker="T3", event_ticker="E", description="51-55°F",
                temp_low=51, temp_high=55, yes_price_cents=40,
            ),
            TemperatureBracket(
                ticker="T4", event_ticker="E", description="56°F or above",
                temp_low=56, temp_high=None, yes_price_cents=20,
            ),
        ]
        return WeatherMarket(
            event_ticker="KXHIGHNY-26JAN28",
            series_ticker="KXHIGHNY",
            city="nyc",
            city_config=get_city_config("nyc"),
            target_date=date(2026, 1, 28),
            brackets=brackets,
        )

    def test_outcomes_alias(self):
        m = self._make_market()
        assert m.outcomes is m.brackets

    def test_temp_unit(self):
        m = self._make_market()
        assert m.temp_unit == "F"

    def test_total_volume(self):
        m = self._make_market()
        assert m.total_volume == 0  # No volume set in fixtures

    def test_find_best_bracket(self):
        m = self._make_market()
        best = m.find_best_bracket()
        assert best.ticker == "T3"  # 40 cents is highest

    def test_probability_distribution(self):
        m = self._make_market()
        dist = m.get_probability_distribution()
        assert len(dist) == 4
        assert all("probability" in d for d in dist)

    def test_expected_temperature(self):
        m = self._make_market()
        expected = m.get_expected_temperature()
        assert expected is not None
        assert 40 < expected < 60
