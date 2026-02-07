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
from scipy.stats import skewnorm

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

    # Skewness (for skew-normal distribution) - fields with defaults must come last
    # Positive = warm tail extends further, Negative = cold tail extends further
    high_skew: float = 0.0
    low_skew: float = 0.0

    # Model consensus ratio (0-1, higher = more agreement)
    consensus_ratio: float = 1.0

    def _get_std_floor(self) -> float:
        """
        Get minimum std floor based on model consensus.

        When models strongly agree, we can use a tighter distribution.
        When models disagree, we need more uncertainty buffer.

        Returns:
            Minimum std dev floor (1.0 to 2.0)
        """
        if self.consensus_ratio > 0.9:
            return 1.0  # Very strong consensus - allow tight distribution
        elif self.consensus_ratio > 0.8:
            return 1.5  # Strong consensus
        else:
            return 2.0  # Normal case - keep conservative floor

    def get_probability_above(self, threshold: float, for_high: bool = True) -> float:
        """
        Calculate probability that actual temperature will be above threshold.

        Uses skew-normal distribution when skew is significant, otherwise normal.
        The std already includes model disagreement, base forecast uncertainty,
        and city-specific station bias uncertainty from 180-day historical analysis.

        Args:
            threshold: Temperature threshold in Fahrenheit
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability (0-1) of exceeding threshold
        """
        if for_high:
            mean, std, skew = self.high_mean, self.high_std, self.high_skew
        else:
            mean, std, skew = self.low_mean, self.low_std, self.low_skew

        # Apply consensus-aware std floor
        std_floor = self._get_std_floor()
        std = max(std, std_floor)

        # Use skew-normal when skew is significant, otherwise normal (faster)
        if abs(skew) >= 0.1:
            # P(X > threshold) = 1 - CDF(threshold) using skew-normal
            return 1 - skewnorm.cdf(threshold, a=skew, loc=mean, scale=std)
        else:
            # P(X > threshold) = 1 - CDF(threshold) using normal
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

        Uses skew-normal distribution when skew is significant, otherwise normal.

        Args:
            lower: Lower bound (inclusive)
            upper: Upper bound (inclusive)
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability of temperature being in [lower, upper]
        """
        if for_high:
            mean, std, skew = self.high_mean, self.high_std, self.high_skew
        else:
            mean, std, skew = self.low_mean, self.low_std, self.low_skew

        # Apply consensus-aware std floor
        std_floor = self._get_std_floor()
        std = max(std, std_floor)

        # Use skew-normal when skew is significant
        if abs(skew) >= 0.1:
            p_below_upper = skewnorm.cdf(upper, a=skew, loc=mean, scale=std)
            p_below_lower = skewnorm.cdf(lower, a=skew, loc=mean, scale=std)
        else:
            p_below_upper = stats.norm.cdf(upper, loc=mean, scale=std)
            p_below_lower = stats.norm.cdf(lower, loc=mean, scale=std)

        return p_below_upper - p_below_lower


def estimate_skew(values: np.ndarray) -> float:
    """
    Estimate distribution skew from model spread using Pearson's coefficient.

    Temperature distributions are often skewed:
    - Cold fronts create sharp drops (negative/left skew)
    - Warm spells have long tails (positive/right skew)

    Args:
        values: Array of model forecast values

    Returns:
        Skew estimate, clamped to [-2.0, 2.0]
    """
    if len(values) < 3:
        return 0.0  # Not enough data for skew estimation

    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)

    if std < 0.5:
        return 0.0  # Very tight consensus, normal is appropriate

    # Pearson's second skewness coefficient: 3 * (mean - median) / std
    # Positive when mean > median (right skew, warm tail)
    # Negative when mean < median (left skew, cold tail)
    skew = 3 * (mean - median) / std

    # Clamp to reasonable range for scipy.stats.skewnorm
    return float(np.clip(skew, -2.0, 2.0))


# Default model weights based on historical accuracy
# GFS and Tomorrow.io weighted higher for US weather markets
# ECMWF reduced due to larger spread/variance observed
DEFAULT_WEIGHTS = {
    "ecmwf": 0.8,   # Reduced - shows larger spread
    "gfs": 1.2,     # Increased - good for US
    "hrrr": 1.3,    # Best for short-term US
    "best_match": 1.0,
    "tomorrow.io": 1.4,  # Tomorrow.io strong for US cities
    "nws": 1.1,     # NWS point forecast - local expertise
    "unknown": 0.7,
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

        # Apply model weights with outlier dampening
        # When a model diverges from the consensus, reduce its weight
        # proportionally to how far it deviates. This prevents a single
        # wide-spread model (e.g. ECMWF) from dragging the mean and
        # inflating uncertainty.
        highs = []
        lows = []
        raw_weights = []

        for f in corrected_forecasts:
            model_weight = self.weights.get(f.model_name.lower(), 1.0)
            total_weight = f.weight * model_weight
            highs.append(f.forecast_high)
            lows.append(f.forecast_low)
            raw_weights.append(total_weight)

        highs = np.array(highs)
        lows = np.array(lows)
        raw_weights = np.array(raw_weights)

        # Outlier dampening: reduce weight of models that deviate from
        # the unweighted median. Uses a Gaussian kernel so moderate
        # disagreement is tolerated but extreme outliers are suppressed.
        #
        # OUTLIER_DAMPENING_SCALE = 2.0°F (aggressive)
        # This is the "characteristic distance" in the Gaussian kernel.
        # - At 2°F deviation from median: weight drops to ~61% (e^-0.5)
        # - At 4°F deviation: weight drops to ~14%
        # - At 6°F deviation: weight drops to ~1%
        # Aggressive setting: outliers nearly ignored, consensus dominates.
        OUTLIER_DAMPENING_SCALE = 2.0  # degrees F

        if len(highs) >= 3:
            median_high = np.median(highs)
            deviations = np.abs(highs - median_high)
            dampening = np.exp(-0.5 * (deviations / OUTLIER_DAMPENING_SCALE) ** 2)
            weights = raw_weights * dampening
        else:
            weights = raw_weights

        # Normalize weights (guard against division by zero if all weights are zero)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback to equal weights if all dampened weights are zero
            weights = np.ones(len(weights)) / len(weights)

        # Weighted statistics
        high_mean = np.average(highs, weights=weights)
        low_mean = np.average(lows, weights=weights)

        # Apply station bias correction
        # NWS station observations often differ systematically from grid model data
        # due to local effects (urban heat island, elevation, microclimate).
        # This bias is added to grid model forecasts to better predict NWS settlement.
        station_bias = getattr(city_config, 'station_bias', 0.0)
        station_bias_std = getattr(city_config, 'station_bias_std', 2.0)
        if station_bias != 0.0:
            high_mean += station_bias
            low_mean += station_bias

        # Weighted standard deviation
        high_var = np.average((highs - high_mean) ** 2, weights=weights)
        low_var = np.average((lows - low_mean) ** 2, weights=weights)

        # Add inherent forecast uncertainty
        #
        # Total uncertainty combines:
        # 1. Model disagreement (high_var from weighted variance)
        # 2. Minimum forecast uncertainty (weather is inherently unpredictable)
        # 3. Station bias uncertainty (how variable the station-to-grid relationship is)
        #
        # We combine these in quadrature (sqrt of sum of squares) since they're
        # independent sources of uncertainty.
        #
        # MIN_FORECAST_UNCERTAINTY = 1.5°F base forecast uncertainty
        # station_bias_std = city-specific (from 180-day historical analysis)
        #   - Miami/Chicago/Austin: ~1.5F (consistent, good for trading)
        #   - NYC/Philly/Denver: ~2.0F (moderate)
        #   - LA: 2.8F (high variability, marine layer effects)
        MIN_FORECAST_UNCERTAINTY = 1.5  # degrees F (reduced since station_bias_std adds uncertainty)

        # Combine uncertainties in quadrature
        # Total std = sqrt(model_variance + min_uncertainty^2 + station_bias_std^2)
        combined_uncertainty = np.sqrt(
            high_var + MIN_FORECAST_UNCERTAINTY**2 + station_bias_std**2
        )
        high_std = combined_uncertainty

        combined_uncertainty_low = np.sqrt(
            low_var + MIN_FORECAST_UNCERTAINTY**2 + station_bias_std**2
        )
        low_std = combined_uncertainty_low

        # Model disagreement increases uncertainty, but cap the amplification
        # so a single outlier model can't blow up the std.
        #
        # UNCERTAINTY AMPLIFICATION CONSTANTS:
        # - IQR_HIGH_DISAGREEMENT = 8°F: When IQR reaches 8°F, models strongly
        #   disagree and we apply maximum uncertainty amplification.
        # - STD_HIGH_DISAGREEMENT = 5°F: For <4 models, 5°F std dev indicates
        #   high disagreement (fallback metric, less robust than IQR).
        # - MAX_SPREAD_AMPLIFICATION = 0.5: Maximum 50% increase to base std.
        #   spread_factor ranges from 1.0 (perfect agreement) to 1.5 (high disagreement).
        #   Caps prevent a single outlier from making uncertainty unrealistically large.
        #
        # Example: If IQR=4°F → spread_factor=1.25 (25% uncertainty boost)
        #          If IQR=8°F → spread_factor=1.5 (50% max boost)
        IQR_HIGH_DISAGREEMENT = 8.0      # degrees F
        STD_HIGH_DISAGREEMENT = 5.0      # degrees F
        MAX_SPREAD_AMPLIFICATION = 0.5   # 50% max increase

        if len(highs) >= 4:
            iqr = np.percentile(highs, 75) - np.percentile(highs, 25)
            spread_factor = 1 + min(iqr / IQR_HIGH_DISAGREEMENT, MAX_SPREAD_AMPLIFICATION)
        else:
            raw_spread = np.std(highs)
            spread_factor = 1 + min(raw_spread / STD_HIGH_DISAGREEMENT, MAX_SPREAD_AMPLIFICATION)

        high_std *= spread_factor
        low_std *= spread_factor

        # Confidence intervals (90%)
        z_90 = 1.645
        high_ci_lower = high_mean - z_90 * high_std
        high_ci_upper = high_mean + z_90 * high_std
        low_ci_lower = low_mean - z_90 * low_std
        low_ci_upper = low_mean + z_90 * low_std

        # Estimate skewness from model spread
        # Used for skew-normal probability calculations
        high_skew = estimate_skew(highs)
        low_skew = estimate_skew(lows)

        # Overall confidence based on:
        # 1. Model consensus (do models agree?)
        # 2. Station reliability (how predictable is the station bias?)
        #
        # Count how many models are within 3°F of the median — the
        # "consensus core". One outlier shouldn't tank confidence if the
        # majority agrees.
        if len(highs) >= 3:
            median_high = np.median(highs)
            consensus_count = np.sum(np.abs(highs - median_high) <= 3.0)
            consensus_ratio = consensus_count / len(highs)
            # Consensus ratio 1.0 → agreement 0.95, ratio 0.5 → agreement 0.65
            model_agreement = 0.5 + 0.45 * consensus_ratio
        else:
            consensus_ratio = 1.0  # Assume consensus with few models
            model_agreement = 1 / (1 + np.std(highs) / 5)

        # Station reliability factor based on bias std dev
        # Low std (1.5F) = very reliable = factor ~1.0
        # High std (2.8F) = less reliable = factor ~0.85
        # This penalizes confidence for cities with unpredictable station behavior
        BASELINE_STD = 1.5  # Miami/Chicago level - most reliable
        station_reliability = 1.0 - 0.1 * max(0, (station_bias_std - BASELINE_STD))
        station_reliability = max(0.8, station_reliability)  # Floor at 0.8

        # Horizon decay: forecasts further out are less reliable
        # ~8% decay per day, floor at 50%
        # Day 0 (today): 1.0, Day 1: 0.92, Day 2: 0.84, Day 7: 0.50
        horizon_days = (forecast_date - date.today()).days
        horizon_factor = max(0.5, 1.0 - 0.08 * max(0, horizon_days))

        confidence = min(0.95, model_agreement * 0.9 * station_reliability * horizon_factor)

        return EnsembleForecast(
            date=forecast_date,
            city=city_config.name,
            high_mean=high_mean,
            high_median=np.median(highs),
            low_mean=low_mean,
            low_median=np.median(lows),
            high_std=high_std,
            low_std=low_std,
            high_skew=high_skew,
            low_skew=low_skew,
            consensus_ratio=consensus_ratio,
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

        Kalshi temperature markets are typically structured as:
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
