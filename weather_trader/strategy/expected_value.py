"""
Expected Value Calculator

Compares forecast probabilities against market odds to identify
positive expected value (+EV) trading opportunities.

Edge = Forecast Probability - Market Implied Probability

When edge is positive and significant, we have a trading opportunity.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
from enum import Enum

from ..models.ensemble import EnsembleForecast
from ..kalshi.markets import WeatherMarket, TemperatureBracket


class SignalStrength(Enum):
    """Strength of trading signal."""
    STRONG_BUY_YES = "strong_buy_yes"
    BUY_YES = "buy_yes"
    NEUTRAL = "neutral"
    BUY_NO = "buy_no"
    STRONG_BUY_NO = "strong_buy_no"


@dataclass
class TradeSignal:
    """A trading signal based on EV analysis."""
    market: WeatherMarket
    bracket: TemperatureBracket  # The specific bracket being evaluated
    forecast: EnsembleForecast

    # Probabilities
    forecast_probability: float  # Our forecast P(YES)
    market_probability: float    # Market implied P(YES)

    # Edge analysis
    edge: float                  # forecast_prob - market_prob
    expected_value: float        # Expected profit per dollar

    # Signal
    signal: SignalStrength
    side: str                    # "YES" or "NO"

    # Confidence metrics
    confidence: float           # Forecast confidence (0-1)
    model_agreement: float      # How much models agree (0-1)

    # Metadata
    threshold: float
    is_high_temp: bool
    is_over_market: bool

    @property
    def is_tradeable(self) -> bool:
        """Check if this signal is strong enough to trade."""
        return self.signal != SignalStrength.NEUTRAL

    @property
    def edge_percent(self) -> str:
        """Get edge as percentage string."""
        return f"{self.edge * 100:.1f}%"


class ExpectedValueCalculator:
    """
    Calculates expected value for weather market trades.
    """

    def __init__(
        self,
        min_edge: float = 0.05,        # Minimum 5% edge
        min_confidence: float = 0.70,   # Minimum 70% confidence
        strong_edge: float = 0.15,      # 15%+ edge is strong signal
    ):
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.strong_edge = strong_edge

    def calculate_ev(
        self,
        market: WeatherMarket,
        bracket: TemperatureBracket,
        forecast: EnsembleForecast
    ) -> TradeSignal:
        """
        Calculate expected value for a single bracket given our forecast.

        Kalshi weather events consist of mutually exclusive temperature brackets.
        Each bracket is a binary contract: will the temp land in this range?

        Args:
            market: The WeatherMarket (event context)
            bracket: The specific TemperatureBracket to evaluate
            forecast: Our ensemble forecast

        Returns:
            TradeSignal with EV analysis
        """
        # All KXHIGH series are high temperature markets
        is_high = True

        # Calculate forecast probability for this bracket's range
        if bracket.temp_low is not None and bracket.temp_high is not None:
            # Range bracket: e.g. "50-52°F"
            forecast_prob = forecast.get_probability_in_range(
                bracket.temp_low, bracket.temp_high, for_high=is_high
            )
        elif bracket.temp_high is not None:
            # Lower-bound bracket: e.g. "≤45°F"
            forecast_prob = forecast.get_probability_below(
                bracket.temp_high, for_high=is_high
            )
        elif bracket.temp_low is not None:
            # Upper-bound bracket: e.g. "≥56°F"
            forecast_prob = forecast.get_probability_above(
                bracket.temp_low, for_high=is_high
            )
        else:
            # Malformed bracket with no temperature bounds - use market price as fair value
            # This prevents false signals; edge will be 0
            forecast_prob = bracket.yes_price

        # Market implied probability from bracket price
        market_prob = bracket.yes_price

        # Calculate edge
        edge = forecast_prob - market_prob

        # Determine signal strength
        signal = self._determine_signal(edge, forecast.confidence)

        # Determine side
        if edge > 0:
            side = "YES"
            expected_value = edge  # EV per dollar on YES
        else:
            side = "NO"
            expected_value = -edge  # EV per dollar on NO

        # Use forecast's pre-calculated confidence as model agreement
        # This is computed in ensemble.py using consensus-based formula
        model_agreement = forecast.confidence

        return TradeSignal(
            market=market,
            bracket=bracket,
            forecast=forecast,
            forecast_probability=forecast_prob,
            market_probability=market_prob,
            edge=edge,
            expected_value=expected_value,
            signal=signal,
            side=side,
            confidence=forecast.confidence,
            model_agreement=model_agreement,
            threshold=bracket.midpoint or 0.0,
            is_high_temp=is_high,
            is_over_market=forecast_prob > market_prob,
        )

    def _determine_signal(self, edge: float, confidence: float) -> SignalStrength:
        """Determine signal strength from edge and confidence."""
        # Check minimum confidence
        if confidence < self.min_confidence:
            return SignalStrength.NEUTRAL

        # Check edge thresholds
        if edge >= self.strong_edge:
            return SignalStrength.STRONG_BUY_YES
        elif edge >= self.min_edge:
            return SignalStrength.BUY_YES
        elif edge <= -self.strong_edge:
            return SignalStrength.STRONG_BUY_NO
        elif edge <= -self.min_edge:
            return SignalStrength.BUY_NO
        else:
            return SignalStrength.NEUTRAL

    def analyze_markets(
        self,
        markets: list[WeatherMarket],
        forecasts: dict[str, EnsembleForecast]  # city -> forecast
    ) -> list[TradeSignal]:
        """
        Analyze multiple markets and return all signals.

        Args:
            markets: List of WeatherMarket objects
            forecasts: Dictionary mapping city keys to forecasts

        Returns:
            List of TradeSignal objects, sorted by edge strength
        """
        from datetime import datetime
        signals = []

        for market in markets:
            # Bug #5 fix: Skip inactive/expired markets
            if not market.is_active:
                continue

            # Check if market has closed (close_time in the past)
            if market.close_time and market.close_time < datetime.now():
                continue

            # Find matching forecast
            city_key = market.city.lower()
            forecast = forecasts.get(city_key)

            if not forecast:
                continue

            # Bug #10 fix: Check dates match with timezone awareness
            # Both forecast.date and market.target_date should be date objects
            # Normalize both to date objects for comparison (handles datetime vs date)
            forecast_date = forecast.date
            market_date = market.target_date
            if hasattr(forecast_date, 'date'):
                forecast_date = forecast_date.date()
            if hasattr(market_date, 'date'):
                market_date = market_date.date()
            if forecast_date != market_date:
                continue

            # Evaluate each bracket independently
            for bracket in market.brackets:
                signal = self.calculate_ev(market, bracket, forecast)
                signals.append(signal)

        # Sort by absolute edge (strongest signals first)
        signals.sort(key=lambda s: abs(s.edge), reverse=True)

        return signals

    def get_tradeable_signals(
        self,
        signals: list[TradeSignal]
    ) -> list[TradeSignal]:
        """
        Filter signals to only tradeable ones.

        Args:
            signals: List of all TradeSignal objects

        Returns:
            List of tradeable signals
        """
        return [s for s in signals if s.is_tradeable]

    def summarize_opportunities(self, signals: list[TradeSignal]) -> dict:
        """
        Generate summary of trading opportunities.

        Args:
            signals: List of TradeSignal objects

        Returns:
            Summary dictionary
        """
        tradeable = self.get_tradeable_signals(signals)

        return {
            "total_markets": len(signals),
            "tradeable_markets": len(tradeable),
            "strong_signals": len([s for s in tradeable
                                   if s.signal in [SignalStrength.STRONG_BUY_YES,
                                                   SignalStrength.STRONG_BUY_NO]]),
            "avg_edge": sum(abs(s.edge) for s in tradeable) / len(tradeable) if tradeable else 0,
            "avg_confidence": sum(s.confidence for s in tradeable) / len(tradeable) if tradeable else 0,
            "yes_signals": len([s for s in tradeable if s.side == "YES"]),
            "no_signals": len([s for s in tradeable if s.side == "NO"]),
        }
