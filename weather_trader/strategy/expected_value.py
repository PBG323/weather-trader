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
from ..polymarket.markets import WeatherMarket


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
        forecast: EnsembleForecast
    ) -> TradeSignal:
        """
        Calculate expected value for a market given our forecast.

        Args:
            market: The WeatherMarket to analyze
            forecast: Our ensemble forecast

        Returns:
            TradeSignal with EV analysis
        """
        # Get forecast probability for this specific market outcome
        is_high = market.is_high_temp_market
        is_over = market.is_over_market
        threshold = market.threshold

        if is_over:
            forecast_prob = forecast.get_probability_above(threshold, for_high=is_high)
        else:
            forecast_prob = forecast.get_probability_below(threshold, for_high=is_high)

        # Market implied probability
        market_prob = market.yes_price

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

        # Calculate model agreement
        if forecast.model_forecasts:
            temps = [f.forecast_high if is_high else f.forecast_low
                     for f in forecast.model_forecasts]
            if temps:
                import numpy as np
                model_agreement = 1 / (1 + np.std(temps) / 5)
            else:
                model_agreement = 0.5
        else:
            model_agreement = 0.5

        return TradeSignal(
            market=market,
            forecast=forecast,
            forecast_probability=forecast_prob,
            market_probability=market_prob,
            edge=edge,
            expected_value=expected_value,
            signal=signal,
            side=side,
            confidence=forecast.confidence,
            model_agreement=model_agreement,
            threshold=threshold,
            is_high_temp=is_high,
            is_over_market=is_over,
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
        signals = []

        for market in markets:
            # Find matching forecast
            city_key = market.city.lower()
            forecast = forecasts.get(city_key)

            if not forecast:
                continue

            # Check dates match
            if forecast.date != market.target_date:
                continue

            signal = self.calculate_ev(market, forecast)
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
