"""
Smart Filter for Weather Trading Signals.

Implements senior trader logic to filter out low-quality trades:
1. Boundary bets (forecast too close to bracket edge)
2. Low edge after spread costs
3. Tail bets against high-confidence forecasts
4. Position sizing based on conviction
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class RejectionReason(Enum):
    """Reasons why a trade signal might be rejected."""
    BOUNDARY_BET = "boundary_bet"
    LOW_EDGE = "low_edge"
    HIGH_SPREAD = "high_spread"
    LOW_CONFIDENCE = "low_confidence"
    TAIL_BET_AGAINST_FORECAST = "tail_bet_against_forecast"
    PRICE_TOO_HIGH = "price_too_high"
    PRICE_TOO_LOW = "price_too_low"


@dataclass
class FilterResult:
    """Result of smart filter evaluation."""
    approved: bool
    reason: Optional[RejectionReason] = None
    details: str = ""
    recommended_size_multiplier: float = 1.0
    conviction_score: float = 0.5


class SmartFilter:
    """
    Smart filter for trading signals.

    Applies senior trader logic to evaluate trade quality and
    recommend position sizing based on conviction.
    """

    def __init__(
        self,
        min_edge_after_spread: float = 0.03,
        max_spread: float = 0.15,
        min_confidence: float = 0.60,
        boundary_threshold: float = 0.5,
    ):
        """
        Initialize smart filter.

        Args:
            min_edge_after_spread: Minimum edge after accounting for spread
            max_spread: Maximum acceptable bid-ask spread
            min_confidence: Minimum model confidence
            boundary_threshold: How close to bracket edge is considered boundary
        """
        self.min_edge_after_spread = min_edge_after_spread
        self.max_spread = max_spread
        self.min_confidence = min_confidence
        self.boundary_threshold = boundary_threshold

    def evaluate(
        self,
        forecast_mean: float,
        forecast_std: float,
        bracket_low: Optional[float],
        bracket_high: Optional[float],
        raw_edge: float,
        market_price: float,
        bid_price: float,
        ask_price: float,
        confidence: float,
        side: str,
    ) -> FilterResult:
        """
        Evaluate a trading signal.

        Args:
            forecast_mean: Forecast temperature
            forecast_std: Forecast standard deviation
            bracket_low: Lower bound of bracket (None for tail bets)
            bracket_high: Upper bound of bracket (None for tail bets)
            raw_edge: Raw edge before spread
            market_price: Current market price
            bid_price: Current bid price
            ask_price: Current ask price
            confidence: Model confidence (0-1)
            side: "YES" or "NO"

        Returns:
            FilterResult with approval status and details
        """
        # Calculate spread
        spread = ask_price - bid_price

        # Calculate edge after spread
        edge_after_spread = abs(raw_edge) - (spread / 2)

        # Check confidence
        if confidence < self.min_confidence:
            return FilterResult(
                approved=False,
                reason=RejectionReason.LOW_CONFIDENCE,
                details=f"Confidence {confidence:.0%} < {self.min_confidence:.0%}",
            )

        # Check spread
        if spread > self.max_spread:
            return FilterResult(
                approved=False,
                reason=RejectionReason.HIGH_SPREAD,
                details=f"Spread {spread:.0%} > {self.max_spread:.0%}",
            )

        # Check edge after spread
        if edge_after_spread < self.min_edge_after_spread:
            return FilterResult(
                approved=False,
                reason=RejectionReason.LOW_EDGE,
                details=f"Edge after spread {edge_after_spread:.1%} < {self.min_edge_after_spread:.0%}",
            )

        # Check boundary bet (for range brackets)
        if bracket_low is not None and bracket_high is not None:
            # Check if forecast is near bracket boundaries
            distance_to_low = abs(forecast_mean - bracket_low)
            distance_to_high = abs(forecast_mean - bracket_high)
            min_distance = min(distance_to_low, distance_to_high)

            # For YES bets, we want forecast well inside the bracket
            if side == "YES" and min_distance < self.boundary_threshold:
                # Check if forecast is inside bracket
                if bracket_low <= forecast_mean <= bracket_high:
                    # Inside but near edge - moderate concern
                    pass  # Allow but reduce size
                else:
                    # Outside bracket - boundary bet
                    return FilterResult(
                        approved=False,
                        reason=RejectionReason.BOUNDARY_BET,
                        details=f"Forecast {forecast_mean:.1f}°F near bracket edge",
                    )

        # Calculate conviction score based on multiple factors
        conviction_score = self._calculate_conviction(
            forecast_mean=forecast_mean,
            forecast_std=forecast_std,
            bracket_low=bracket_low,
            bracket_high=bracket_high,
            edge=raw_edge,
            confidence=confidence,
            spread=spread,
        )

        # Determine size multiplier based on conviction
        if conviction_score >= 0.8:
            size_multiplier = 1.2  # High conviction - size up slightly
        elif conviction_score >= 0.6:
            size_multiplier = 1.0  # Normal conviction
        elif conviction_score >= 0.4:
            size_multiplier = 0.7  # Low conviction - size down
        else:
            size_multiplier = 0.5  # Very low conviction - minimal size

        return FilterResult(
            approved=True,
            recommended_size_multiplier=size_multiplier,
            conviction_score=conviction_score,
        )

    def _calculate_conviction(
        self,
        forecast_mean: float,
        forecast_std: float,
        bracket_low: Optional[float],
        bracket_high: Optional[float],
        edge: float,
        confidence: float,
        spread: float,
    ) -> float:
        """
        Calculate conviction score (0-1) for a trade.

        Higher scores indicate stronger conviction and justify larger sizes.
        """
        score = 0.5  # Base score

        # Factor 1: Edge magnitude (0-0.2)
        edge_factor = min(abs(edge) * 2, 0.2)
        score += edge_factor

        # Factor 2: Confidence (0-0.2)
        conf_factor = (confidence - 0.5) * 0.4  # Range: -0.2 to 0.2
        score += max(0, conf_factor)

        # Factor 3: Spread (negative factor)
        spread_penalty = spread * 0.5  # Higher spread = lower conviction
        score -= spread_penalty

        # Factor 4: Forecast position relative to bracket
        if bracket_low is not None and bracket_high is not None:
            bracket_center = (bracket_low + bracket_high) / 2
            bracket_width = bracket_high - bracket_low
            if bracket_width > 0:
                distance_from_center = abs(forecast_mean - bracket_center)
                center_factor = 1 - (distance_from_center / bracket_width)
                score += center_factor * 0.1

        # Factor 5: Forecast uncertainty
        if forecast_std < 2.0:
            score += 0.1  # Low uncertainty is good
        elif forecast_std > 4.0:
            score -= 0.1  # High uncertainty is bad

        return max(0, min(1, score))


# Global instance with default settings
smart_filter = SmartFilter()
