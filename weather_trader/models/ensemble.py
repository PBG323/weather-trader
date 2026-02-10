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

For same-day trading, we use Bayesian conditioning on METAR observations:
- Observations provide a hard floor (high cannot be below current temp)
- Time-of-day determines how much additional warming is possible
- City-specific afternoon warming patterns refine estimates
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import Optional
from zoneinfo import ZoneInfo
import numpy as np
from scipy import stats
from scipy.stats import skewnorm, truncnorm

from ..config import CityConfig
from .bias_correction import BiasCorrector, CorrectedForecast


# Timezone for US weather markets
EST = ZoneInfo("America/New_York")

# City-specific afternoon warming patterns (degrees F typically gained after observation)
# Based on climatology: how much does temp typically rise from current hour to daily max
# Format: {city: {hour: expected_additional_warming}}
AFTERNOON_WARMING = {
    "nyc": {8: 8, 9: 7, 10: 6, 11: 4, 12: 3, 13: 2, 14: 1, 15: 0.5, 16: 0, 17: 0},
    "chicago": {8: 9, 9: 8, 10: 6, 11: 5, 12: 3, 13: 2, 14: 1, 15: 0.5, 16: 0, 17: 0},
    "miami": {8: 6, 9: 5, 10: 4, 11: 3, 12: 2, 13: 1.5, 14: 1, 15: 0.5, 16: 0, 17: 0},
    "la": {8: 10, 9: 9, 10: 7, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1, 16: 0.5, 17: 0},
    "denver": {8: 12, 9: 10, 10: 8, 11: 6, 12: 4, 13: 3, 14: 2, 15: 1, 16: 0, 17: 0},
    "philadelphia": {8: 8, 9: 7, 10: 6, 11: 4, 12: 3, 13: 2, 14: 1, 15: 0.5, 16: 0, 17: 0},
    "austin": {8: 10, 9: 9, 10: 7, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1, 16: 0.5, 17: 0},
}

# Default warming pattern for cities not explicitly defined
DEFAULT_WARMING = {8: 8, 9: 7, 10: 6, 11: 4, 12: 3, 13: 2, 14: 1, 15: 0.5, 16: 0, 17: 0}


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

    def adjust_for_observation(
        self,
        observed_temp: float,
        observation_age_minutes: int = 60,
        observation_hour: Optional[int] = None,
        city_key: Optional[str] = None,
        is_same_day: bool = True,
    ) -> "EnsembleForecast":
        """
        Bayesian update of forecast using real-time METAR observation.

        Uses truncated normal distribution for statistically rigorous probability
        calculations. The observation provides a HARD FLOOR - the daily high
        cannot be below the current observed temperature.

        Key insight: P(high | forecast, observation) ≠ P(high | forecast)

        The posterior distribution is:
        1. Truncated at observed_temp (hard floor)
        2. Mean shifted based on observation vs forecast delta
        3. Std reduced based on time-of-day (less uncertainty later)
        4. City-specific afternoon warming patterns applied

        Args:
            observed_temp: Current observed temperature in Fahrenheit
            observation_age_minutes: How old the observation is
            observation_hour: Hour of observation (0-23, EST). Auto-detected if None.
            city_key: City identifier for warming patterns. Uses default if None.
            is_same_day: Whether this is for same-day bracket trading

        Returns:
            New EnsembleForecast with Bayesian-updated probabilities
        """
        # Don't adjust if observation is too old (>3 hours)
        if observation_age_minutes > 180:
            return self

        # Get current hour if not provided
        if observation_hour is None:
            now = datetime.now(EST)
            observation_hour = now.hour

        # For non-same-day, observations are informational only
        if not is_same_day:
            return self

        # === BAYESIAN UPDATE FOR SAME-DAY TRADING ===

        # 1. HARD FLOOR: The high CANNOT be below current observation
        floor_temp = observed_temp

        # 2. EXPECTED ADDITIONAL WARMING based on time of day and city
        city_lower = (city_key or self.city or "").lower()
        warming_schedule = AFTERNOON_WARMING.get(city_lower, DEFAULT_WARMING)

        # Get expected additional warming from current hour to daily max
        # Daily max typically occurs between 2-5pm depending on city
        expected_additional = warming_schedule.get(observation_hour, 0)

        # Add uncertainty to the additional warming estimate
        # Earlier in day = more uncertainty about afternoon warming
        warming_uncertainty = max(0.5, expected_additional * 0.3)

        # 3. POSTERIOR MEAN CALCULATION
        # The posterior mean is influenced by:
        # a) Original forecast (prior)
        # b) Current observation + expected additional warming (likelihood)

        # Weight observation more heavily as day progresses
        # 8am: 30% observation weight, 3pm: 90% observation weight
        if observation_hour <= 8:
            obs_weight = 0.3
        elif observation_hour >= 15:
            obs_weight = 0.9
        else:
            # Linear interpolation from 8am to 3pm
            obs_weight = 0.3 + (observation_hour - 8) * (0.6 / 7)

        # Observation-based estimate of final high
        obs_estimated_high = observed_temp + expected_additional

        # Bayesian posterior mean (weighted combination)
        posterior_mean = (1 - obs_weight) * self.high_mean + obs_weight * obs_estimated_high

        # Ensure posterior mean is at least the floor
        posterior_mean = max(posterior_mean, floor_temp + 0.5)

        # 4. POSTERIOR STD CALCULATION
        # Uncertainty decreases as we get more data through the day
        # Also incorporates warming uncertainty

        # Base reduction: later in day = less uncertainty
        if observation_hour <= 8:
            std_reduction = 0.1  # 10% reduction at 8am
        elif observation_hour >= 16:
            std_reduction = 0.7  # 70% reduction at 4pm (near daily max)
        else:
            std_reduction = 0.1 + (observation_hour - 8) * (0.6 / 8)

        # Posterior std combines reduced forecast uncertainty with warming uncertainty
        reduced_forecast_std = self.high_std * (1 - std_reduction)
        posterior_std = np.sqrt(reduced_forecast_std**2 + warming_uncertainty**2)

        # Minimum std floor (some uncertainty always remains)
        posterior_std = max(posterior_std, 0.8)

        # 5. SKEW ADJUSTMENT
        # If observation is running hot, distribution should skew warm
        diff = observed_temp - self.high_mean
        if diff > 2:
            # Running hot - positive skew (warm tail)
            posterior_skew = min(self.high_skew + 0.5, 2.0)
        elif diff < -2:
            # Running cold - but floor constrains, so slight negative skew
            posterior_skew = max(self.high_skew - 0.3, -1.0)
        else:
            # Confirming forecast - reduce skew toward normal
            posterior_skew = self.high_skew * 0.7

        # 6. CONFIDENCE UPDATE
        # Confidence increases when we have real data
        # But decreases if observation conflicts significantly with forecast

        if abs(diff) <= 1.5:
            # Observation confirms forecast - high confidence
            confidence_boost = 0.15 * obs_weight
        elif abs(diff) <= 3.0:
            # Minor deviation - moderate confidence boost
            confidence_boost = 0.08 * obs_weight
        elif abs(diff) <= 5.0:
            # Significant deviation - small boost (we have data, just unexpected)
            confidence_boost = 0.03 * obs_weight
        else:
            # Major deviation - slight reduction (something unusual happening)
            confidence_boost = -0.05

        posterior_confidence = min(0.98, max(0.5, self.confidence + confidence_boost))

        # 7. CONFIDENCE INTERVALS (accounting for truncation)
        z_90 = 1.645
        ci_lower = max(floor_temp, posterior_mean - z_90 * posterior_std)
        ci_upper = posterior_mean + z_90 * posterior_std

        # Create conditioned forecast
        return ObservationConditionedForecast(
            date=self.date,
            city=self.city,
            high_mean=posterior_mean,
            high_median=max(self.high_median, floor_temp + expected_additional * 0.5),
            low_mean=self.low_mean,
            low_median=self.low_median,
            high_std=posterior_std,
            low_std=self.low_std,
            high_ci_lower=ci_lower,
            high_ci_upper=ci_upper,
            low_ci_lower=self.low_ci_lower,
            low_ci_upper=self.low_ci_upper,
            model_count=self.model_count,
            model_forecasts=self.model_forecasts,
            confidence=posterior_confidence,
            high_skew=posterior_skew,
            low_skew=self.low_skew,
            consensus_ratio=self.consensus_ratio,
            # Observation-specific fields
            observation_floor=floor_temp,
            observed_temp=observed_temp,
            observation_hour=observation_hour,
            expected_additional_warming=expected_additional,
            obs_weight=obs_weight,
        )


@dataclass
class ObservationConditionedForecast(EnsembleForecast):
    """
    Forecast conditioned on real-time METAR observation.

    Extends EnsembleForecast with:
    1. Truncated normal distribution (floor at observed temp)
    2. Observation metadata for transparency
    3. Specialized probability calculations accounting for truncation

    This provides statistically rigorous probability calculations that
    properly account for the constraint that high >= observed_temp.
    """
    # Observation-specific fields
    observation_floor: float = 0.0  # Hard floor (observed temp)
    observed_temp: float = 0.0
    observation_hour: int = 12
    expected_additional_warming: float = 0.0
    obs_weight: float = 0.5

    def get_probability_above(self, threshold: float, for_high: bool = True) -> float:
        """
        Calculate P(high > threshold) using truncated normal distribution.

        The distribution is truncated at the observation floor since the
        daily high cannot be below what's already been observed.

        Args:
            threshold: Temperature threshold in Fahrenheit
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability (0-1) of exceeding threshold
        """
        if not for_high:
            # Low temperature not affected by high temp observation
            return super().get_probability_above(threshold, for_high=False)

        # For high temperature with observation, use truncated normal
        floor = self.observation_floor
        mean = self.high_mean
        std = max(self.high_std, 0.5)

        # If threshold is below floor, probability is 1.0
        # (we know the high is at least the floor)
        if threshold <= floor:
            return 1.0

        # Use truncated normal: X ~ TruncNorm(mean, std, a=floor, b=inf)
        # P(X > threshold) = 1 - P(X <= threshold)

        # Standardize for truncnorm
        a = (floor - mean) / std  # Lower bound in standard units
        b = np.inf  # No upper bound

        # Create truncated normal distribution
        try:
            trunc_dist = truncnorm(a=a, b=b, loc=mean, scale=std)
            prob = 1 - trunc_dist.cdf(threshold)
        except Exception:
            # Fallback to regular normal if truncnorm fails
            prob = 1 - stats.norm.cdf(threshold, loc=mean, scale=std)

        return float(np.clip(prob, 0.001, 0.999))

    def get_probability_below(self, threshold: float, for_high: bool = True) -> float:
        """
        Calculate P(high < threshold) using truncated normal distribution.

        Args:
            threshold: Temperature threshold in Fahrenheit
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability (0-1) of being below threshold
        """
        if not for_high:
            return super().get_probability_below(threshold, for_high=False)

        # P(X < threshold) = 1 - P(X >= threshold)
        return 1 - self.get_probability_above(threshold, for_high=True)

    def get_probability_in_range(
        self,
        lower: float,
        upper: float,
        for_high: bool = True
    ) -> float:
        """
        Calculate P(lower <= high <= upper) using truncated normal.

        This is the key function for bracket probability calculation.

        Args:
            lower: Lower bound (inclusive)
            upper: Upper bound (inclusive)
            for_high: If True, calculate for daily high; if False, for daily low

        Returns:
            Probability of temperature being in [lower, upper]
        """
        if not for_high:
            return super().get_probability_in_range(lower, upper, for_high=False)

        floor = self.observation_floor
        mean = self.high_mean
        std = max(self.high_std, 0.5)

        # If entire range is below floor, probability is 0
        if upper < floor:
            return 0.0

        # Adjust lower bound to floor if needed
        effective_lower = max(lower, floor)

        # Use truncated normal
        a = (floor - mean) / std
        b = np.inf

        try:
            trunc_dist = truncnorm(a=a, b=b, loc=mean, scale=std)
            prob = trunc_dist.cdf(upper) - trunc_dist.cdf(effective_lower)
        except Exception:
            # Fallback
            prob = stats.norm.cdf(upper, mean, std) - stats.norm.cdf(effective_lower, mean, std)

        return float(np.clip(prob, 0.0, 1.0))

    def get_observation_summary(self) -> dict:
        """Get summary of observation conditioning for logging/display."""
        return {
            "observed_temp": self.observed_temp,
            "observation_floor": self.observation_floor,
            "observation_hour": self.observation_hour,
            "expected_additional_warming": self.expected_additional_warming,
            "obs_weight": self.obs_weight,
            "posterior_mean": self.high_mean,
            "posterior_std": self.high_std,
            "confidence": self.confidence,
            "is_truncated": True,
        }


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
    "ecmwf": 0.8,         # Reduced - shows larger spread
    "ecmwf_aifs": 1.0,    # ECMWF AI model - experimental, equal weight
    "gfs": 1.2,           # Increased - good for US
    "gfs_ensemble": 1.1,  # GFS ensemble mean - probabilistic
    "hrrr": 1.3,          # Best for short-term US
    "best_match": 1.0,
    "tomorrow.io": 1.4,   # Tomorrow.io strong for US cities
    "nws": 1.1,           # NWS point forecast - local expertise
    "visual_crossing": 1.0,  # Visual Crossing - backup/validation source
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
        # ~4% decay per day, floor at 70% (reduced from 8% to avoid over-penalizing)
        # Day 0 (today): 1.0, Day 1: 0.96, Day 2: 0.92, Day 7: 0.72
        horizon_days = (forecast_date - date.today()).days
        horizon_factor = max(0.70, 1.0 - 0.04 * max(0, horizon_days))

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


def detect_weather_regime(forecasts: dict) -> dict:
    """
    Classify current weather regime for strategy adjustment.

    Weather regimes:
    - stable: High-confidence stable pattern → tighter spreads, smaller edge needed
    - volatile: Active pattern (fronts, storms) → wider spreads, require larger edge
    - normal: Typical conditions

    Args:
        forecasts: Dict of city -> EnsembleForecast

    Returns:
        Dict with regime classification and recommended parameters
    """
    if not forecasts:
        return {
            "regime": "unknown",
            "min_edge": 0.05,
            "position_scale": 1.0,
            "reason": "No forecast data"
        }

    # Extract std and consensus from all forecasts
    stds = []
    consensus_ratios = []

    for key, fc in forecasts.items():
        if isinstance(fc, dict):
            stds.append(fc.get("high_std", 3.0))
            # Get consensus from confidence breakdown if available
            breakdown = fc.get("confidence_breakdown", {})
            consensus_ratios.append(breakdown.get("model_agreement", 0.75))
        elif hasattr(fc, "high_std"):
            stds.append(fc.high_std)
            consensus_ratios.append(getattr(fc, "consensus_ratio", 0.75))

    if not stds:
        return {
            "regime": "unknown",
            "min_edge": 0.05,
            "position_scale": 1.0,
            "reason": "No valid forecasts"
        }

    avg_std = np.mean(stds)
    avg_consensus = np.mean(consensus_ratios)

    if avg_std < 2.5 and avg_consensus > 0.85:
        # High confidence, stable weather pattern
        return {
            "regime": "stable",
            "min_edge": 0.04,  # Can trade smaller edges
            "position_scale": 1.2,  # Slightly larger positions OK
            "avg_std": float(avg_std),
            "avg_consensus": float(avg_consensus),
            "reason": f"Stable pattern: avg_std={avg_std:.1f}°F, consensus={avg_consensus:.0%}"
        }
    elif avg_std > 4.0 or avg_consensus < 0.6:
        # Volatile weather, high uncertainty
        return {
            "regime": "volatile",
            "min_edge": 0.07,  # Require larger edge
            "position_scale": 0.7,  # Smaller positions
            "avg_std": float(avg_std),
            "avg_consensus": float(avg_consensus),
            "reason": f"Volatile pattern: avg_std={avg_std:.1f}°F, consensus={avg_consensus:.0%}"
        }
    else:
        # Normal conditions
        return {
            "regime": "normal",
            "min_edge": 0.05,
            "position_scale": 1.0,
            "avg_std": float(avg_std),
            "avg_consensus": float(avg_consensus),
            "reason": f"Normal pattern: avg_std={avg_std:.1f}°F, consensus={avg_consensus:.0%}"
        }



async def adjust_forecasts_with_metar(
    forecasts: dict[str, EnsembleForecast],
    is_same_day: bool = True,
) -> tuple[dict[str, EnsembleForecast], dict[str, dict]]:
    """
    Adjust ensemble forecasts using real-time METAR observations.

    This integrates the "pilot edge" into probability calculations by:
    1. Fetching current METAR observations for all cities
    2. Applying Bayesian conditioning with truncated normal distribution
    3. Returning both adjusted forecasts and observation metadata

    The adjustment uses:
    - Hard floor constraint (high cannot be below observation)
    - Time-of-day weighting (afternoon obs matter more)
    - City-specific afternoon warming patterns
    - Truncated normal for rigorous probability calculations

    Args:
        forecasts: Dict of city_key -> EnsembleForecast
        is_same_day: Whether these are same-day bracket forecasts

    Returns:
        Tuple of (adjusted_forecasts, metar_data)
        - adjusted_forecasts: Dict of city_key -> ObservationConditionedForecast
        - metar_data: Dict of city_key -> observation info for UI display
    """
    from ..apis.aviation_weather import AviationWeatherClient

    adjusted = {}
    metar_data = {}

    # Get current hour for time-of-day weighting
    now = datetime.now(EST)
    current_hour = now.hour

    try:
        async with AviationWeatherClient() as client:
            observations = await client.get_all_city_temperatures()

            for city_key, forecast in forecasts.items():
                city_lower = city_key.lower()

                if city_lower in observations:
                    obs = observations[city_lower]
                    observed_temp = obs["temperature_f"]
                    age_minutes = obs.get("age_minutes", 60)

                    # Calculate observation hour (accounting for age)
                    obs_hour = current_hour
                    if age_minutes > 30:
                        obs_hour = max(8, current_hour - (age_minutes // 60))

                    # Adjust forecast with Bayesian conditioning
                    adjusted_forecast = forecast.adjust_for_observation(
                        observed_temp=observed_temp,
                        observation_age_minutes=age_minutes,
                        observation_hour=obs_hour,
                        city_key=city_lower,
                        is_same_day=is_same_day,
                    )
                    adjusted[city_key] = adjusted_forecast

                    # Store METAR data for UI/logging
                    diff = observed_temp - forecast.high_mean
                    expected_warming = AFTERNOON_WARMING.get(
                        city_lower, DEFAULT_WARMING
                    ).get(obs_hour, 0)

                    metar_data[city_key] = {
                        "observed_temp": observed_temp,
                        "forecast_high": forecast.high_mean,
                        "adjusted_high": adjusted_forecast.high_mean,
                        "adjusted_std": adjusted_forecast.high_std,
                        "difference": diff,
                        "station": obs.get("station", "unknown"),
                        "age_minutes": age_minutes,
                        "observation_hour": obs_hour,
                        "expected_additional_warming": expected_warming,
                        "is_fresh": obs.get("is_fresh", False),
                        "adjustment_applied": True,
                        "original_confidence": forecast.confidence,
                        "adjusted_confidence": adjusted_forecast.confidence,
                        "uses_truncated_normal": isinstance(
                            adjusted_forecast, ObservationConditionedForecast
                        ),
                        "observation_floor": observed_temp,
                    }
                else:
                    # No observation available, use original forecast
                    adjusted[city_key] = forecast
                    metar_data[city_key] = {
                        "adjustment_applied": False,
                        "reason": "No METAR observation available",
                    }

    except Exception as e:
        # If METAR fetch fails, return original forecasts
        for city_key, forecast in forecasts.items():
            adjusted[city_key] = forecast
            metar_data[city_key] = {
                "adjustment_applied": False,
                "reason": f"METAR fetch error: {str(e)}",
            }

    return adjusted, metar_data


def get_metar_edge_summary(metar_data: dict[str, dict]) -> list[str]:
    """
    Generate actionable trading insights from METAR observations.

    Provides specific guidance on:
    - Which brackets are affected by current observations
    - Hard floor constraints from observed temperatures
    - Expected additional warming potential

    Args:
        metar_data: Dict from adjust_forecasts_with_metar

    Returns:
        List of human-readable edge descriptions with trading implications
    """
    summaries = []

    for city_key, data in metar_data.items():
        if not data.get("adjustment_applied"):
            continue

        diff = data.get("difference", 0)
        observed = data.get("observed_temp", 0)
        forecast = data.get("forecast_high", 0)
        adjusted = data.get("adjusted_high", observed)
        expected_warming = data.get("expected_additional_warming", 0)
        obs_hour = data.get("observation_hour", 12)

        # Calculate trading-relevant floor
        floor = observed  # Hard floor - high cannot be below this

        if diff >= 5.0:
            # Major bullish signal
            summaries.append(
                f"🔥 {city_key.upper()} STRONG BULLISH: Already at {observed:.0f}°F "
                f"({diff:.0f}°F above {forecast:.0f}°F forecast). "
                f"Floor locked at {floor:.0f}°F. Brackets ≤{floor:.0f}°F now have ~0% probability."
            )
        elif diff >= 3.0:
            summaries.append(
                f"📈 {city_key.upper()} BULLISH: Running hot at {observed:.0f}°F "
                f"(+{diff:.0f}°F vs forecast). Expected final high: {adjusted:.0f}°F. "
                f"Hard floor: {floor:.0f}°F."
            )
        elif diff <= -5.0:
            summaries.append(
                f"❄️ {city_key.upper()} STRONG BEARISH: Only at {observed:.0f}°F "
                f"({abs(diff):.0f}°F below {forecast:.0f}°F forecast). "
                f"May still warm +{expected_warming:.0f}°F to ~{observed + expected_warming:.0f}°F."
            )
        elif diff <= -3.0:
            summaries.append(
                f"📉 {city_key.upper()} BEARISH: Running cold at {observed:.0f}°F "
                f"(-{abs(diff):.0f}°F vs forecast). "
                f"Potential recovery to ~{observed + expected_warming:.0f}°F if warming continues."
            )
        elif data.get("is_fresh") and abs(diff) <= 1.5:
            summaries.append(
                f"✅ {city_key.upper()}: On track at {observed:.0f}°F "
                f"(forecast: {forecast:.0f}°F). High confidence. "
                f"Floor: {floor:.0f}°F, expected final: {adjusted:.0f}°F."
            )
        elif obs_hour >= 14:
            # Late observation - near final high
            summaries.append(
                f"⏰ {city_key.upper()}: Late observation ({obs_hour}:00) at {observed:.0f}°F. "
                f"Likely near final high. Limited additional warming expected (+{expected_warming:.0f}°F max)."
            )

    return summaries
