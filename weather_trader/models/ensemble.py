"""
Ensemble Forecaster

Combines forecasts from multiple weather models to produce:
- A weighted average forecast
- Confidence intervals
- Probability distributions for temperature outcomes

The ensemble approach outperforms any single model because:
1. Different models have different strengths
2. Combining reduces individual model errors
3. Spread of forecasts indicates uncertainty
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import numpy as np
from scipy import stats

from ..config import CityConfig
from .bias_correction import BiasCorrector, CorrectedForecast


@dataclass
class ModelForecast:
    """Individual model forecast."""
    model_name: str
    forecast_high: float
    forecast_low: float
    weight: float = 1.0
    bias_corrected: bool = False


@dataclass
class EnsembleForecast:
    """Combined forecast from multiple models."""
    date: date
    city: str

    # Point estimates
    high_mean: float
    high_median: float
    low_mean: float
    low_median: float

    # Uncertainty estimates
    high_std: float
    low_std: float

    # Confidence intervals (90%)
    high_ci_lower: float
    high_ci_upper: float
    low_ci_lower: float
    low_ci_upper: float

    # Model details
    model_count: int
    model_forecasts: list[ModelForecast]

    # Overall confidence (0-1)
    confidence: float

    def get_probability_above(self, threshold: float, for_high: bool = True) -> float:
        """
        Calculate probability that actual temperature will be above threshold.

        Uses normal distribution assumption based on ensemble mean and std.

        Args:
            threshold: Temperature threshold in Fahrenheit
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability (0-1) of exceeding threshold
        """
        if for_high:
            mean, std = self.high_mean, self.high_std
        else:
            mean, std = self.low_mean, self.low_std

        # Add minimum uncertainty floor
        std = max(std, 1.5)

        # P(X > threshold) = 1 - CDF(threshold)
        return 1 - stats.norm.cdf(threshold, loc=mean, scale=std)

    def get_probability_below(self, threshold: float, for_high: bool = True) -> float:
        """
        Calculate probability that actual temperature will be below threshold.

        Args:
            threshold: Temperature threshold in Fahrenheit
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability (0-1) of being below threshold
        """
        return 1 - self.get_probability_above(threshold, for_high)

    def get_probability_in_range(
        self,
        lower: float,
        upper: float,
        for_high: bool = True
    ) -> float:
        """
        Calculate probability that temperature falls within a range.

        Args:
            lower: Lower bound (inclusive)
            upper: Upper bound (inclusive)
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability of temperature being in [lower, upper]
        """
        if for_high:
            mean, std = self.high_mean, self.high_std
        else:
            mean, std = self.low_mean, self.low_std

        std = max(std, 1.5)

        p_below_upper = stats.norm.cdf(upper, loc=mean, scale=std)
        p_below_lower = stats.norm.cdf(lower, loc=mean, scale=std)

        return p_below_upper - p_below_lower


# Default model weights based on historical accuracy
# ECMWF is generally most accurate, followed by GFS
DEFAULT_WEIGHTS = {
    "ecmwf": 1.5,
    "gfs": 1.0,
    "hrrr": 1.2,  # Best for short-term US
    "best_match": 1.1,
    "tomorrow": 1.3,
    "nws": 1.0,  # NWS point forecast
    "unknown": 0.8,
}


class EnsembleForecaster:
    """
    Combines multiple weather model forecasts into an ensemble prediction.
    """

    def __init__(
        self,
        bias_corrector: Optional[BiasCorrector] = None,
        weights: Optional[dict[str, float]] = None
    ):
        self.bias_corrector = bias_corrector or BiasCorrector()
        self.weights = weights or DEFAULT_WEIGHTS

    def create_ensemble(
        self,
        city_config: CityConfig,
        forecasts: list[ModelForecast],
        forecast_date: date,
        apply_bias_correction: bool = True
    ) -> EnsembleForecast:
        """
        Create an ensemble forecast from multiple model forecasts.

        Args:
            city_config: City configuration
            forecasts: List of individual model forecasts
            forecast_date: Date being forecast
            apply_bias_correction: Whether to apply bias correction

        Returns:
            EnsembleForecast with combined predictions
        """
        if not forecasts:
            raise ValueError("No forecasts provided for ensemble")

        # Apply bias correction if available and requested
        corrected_forecasts = []
        for f in forecasts:
            if apply_bias_correction and not f.bias_corrected:
                corrected = self.bias_corrector.correct_forecast(
                    city_config,
                    f.forecast_high,
                    f.forecast_low,
                    forecast_date,
                    f.model_name
                )
                corrected_forecasts.append(ModelForecast(
                    model_name=f.model_name,
                    forecast_high=corrected.corrected_high,
                    forecast_low=corrected.corrected_low,
                    weight=f.weight * corrected.confidence,
                    bias_corrected=True,
                ))
            else:
                corrected_forecasts.append(f)

        # Apply model weights
        highs = []
        lows = []
        weights = []

        for f in corrected_forecasts:
            model_weight = self.weights.get(f.model_name.lower(), 1.0)
            total_weight = f.weight * model_weight
            highs.append(f.forecast_high)
            lows.append(f.forecast_low)
            weights.append(total_weight)

        highs = np.array(highs)
        lows = np.array(lows)
        weights = np.array(weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted statistics
        high_mean = np.average(highs, weights=weights)
        low_mean = np.average(lows, weights=weights)

        # Weighted standard deviation
        high_var = np.average((highs - high_mean) ** 2, weights=weights)
        low_var = np.average((lows - low_mean) ** 2, weights=weights)

        # Add inherent forecast uncertainty (minimum std of ~2F)
        min_std = 2.0
        high_std = max(np.sqrt(high_var), min_std)
        low_std = max(np.sqrt(low_var), min_std)

        # If models disagree more, increase uncertainty
        spread_factor = 1 + (np.std(highs) / 5)  # Increase std if spread is large
        high_std *= spread_factor
        low_std *= spread_factor

        # Confidence intervals (90%)
        z_90 = 1.645
        high_ci_lower = high_mean - z_90 * high_std
        high_ci_upper = high_mean + z_90 * high_std
        low_ci_lower = low_mean - z_90 * low_std
        low_ci_upper = low_mean + z_90 * low_std

        # Overall confidence
        # Higher if models agree, lower if they spread
        model_agreement = 1 / (1 + np.std(highs) / 5)
        confidence = min(0.95, model_agreement * 0.9)

        return EnsembleForecast(
            date=forecast_date,
            city=city_config.name,
            high_mean=high_mean,
            high_median=np.median(highs),
            low_mean=low_mean,
            low_median=np.median(lows),
            high_std=high_std,
            low_std=low_std,
            high_ci_lower=high_ci_lower,
            high_ci_upper=high_ci_upper,
            low_ci_lower=low_ci_lower,
            low_ci_upper=low_ci_upper,
            model_count=len(corrected_forecasts),
            model_forecasts=corrected_forecasts,
            confidence=confidence,
        )

    def forecast_for_market(
        self,
        ensemble: EnsembleForecast,
        market_threshold: float,
        is_over_market: bool = True,
        for_high: bool = True
    ) -> dict:
        """
        Generate forecast probability for a specific market contract.

        Polymarket temperature markets are typically structured as:
        "Will NYC high temperature be over/under X degrees?"

        Args:
            ensemble: Ensemble forecast
            market_threshold: Temperature threshold in the market
            is_over_market: True if market is "over X", False if "under X"
            for_high: True for daily high markets, False for daily low

        Returns:
            Dictionary with forecast probability and metadata
        """
        if is_over_market:
            prob = ensemble.get_probability_above(market_threshold, for_high)
        else:
            prob = ensemble.get_probability_below(market_threshold, for_high)

        return {
            "probability": prob,
            "threshold": market_threshold,
            "is_over": is_over_market,
            "for_high": for_high,
            "forecast_mean": ensemble.high_mean if for_high else ensemble.low_mean,
            "forecast_std": ensemble.high_std if for_high else ensemble.low_std,
            "confidence": ensemble.confidence,
            "model_count": ensemble.model_count,
        }

    def calculate_edge(
        self,
        ensemble: EnsembleForecast,
        market_threshold: float,
        market_price: float,
        is_over_market: bool = True,
        for_high: bool = True
    ) -> dict:
        """
        Calculate trading edge versus market price.

        Edge = Forecast Probability - Market Implied Probability

        Args:
            ensemble: Ensemble forecast
            market_threshold: Temperature threshold
            market_price: Current market price (0-1)
            is_over_market: True for "over" markets
            for_high: True for high temperature markets

        Returns:
            Dictionary with edge calculation and recommendation
        """
        forecast = self.forecast_for_market(
            ensemble, market_threshold, is_over_market, for_high
        )

        forecast_prob = forecast["probability"]
        edge = forecast_prob - market_price

        # Determine if this is a tradeable opportunity
        min_edge = 0.05  # 5% minimum edge
        min_confidence = 0.7

        should_trade = (
            abs(edge) >= min_edge and
            forecast["confidence"] >= min_confidence
        )

        # Direction: buy YES if positive edge, buy NO (or sell YES) if negative
        if edge > 0:
            direction = "BUY_YES"
            expected_profit = edge
        else:
            direction = "BUY_NO"
            expected_profit = -edge

        return {
            "edge": edge,
            "forecast_probability": forecast_prob,
            "market_probability": market_price,
            "direction": direction,
            "expected_profit_per_dollar": expected_profit,
            "should_trade": should_trade,
            "confidence": forecast["confidence"],
            "reason": self._get_edge_reason(edge, forecast["confidence"]),
        }

    def _get_edge_reason(self, edge: float, confidence: float) -> str:
        """Generate human-readable reason for trading decision."""
        if abs(edge) < 0.05:
            return "Edge too small (< 5%)"
        if confidence < 0.7:
            return "Model confidence too low"
        if edge > 0.15:
            return f"Strong buy YES signal: {edge:.1%} edge"
        if edge > 0.05:
            return f"Buy YES: {edge:.1%} edge"
        if edge < -0.15:
            return f"Strong buy NO signal: {-edge:.1%} edge"
        if edge < -0.05:
            return f"Buy NO: {-edge:.1%} edge"
        return "No clear signal"
