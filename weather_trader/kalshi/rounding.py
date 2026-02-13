"""
Kalshi Temperature Rounding and Bracket Probability Calculations

Kalshi settles temperature markets based on ROUNDED integers from NWS:
- 37.4°F → settles as 37°F
- 37.5°F → settles as 38°F (standard rounding: ≥0.5 rounds up)

This module provides functions to:
1. Calculate bracket probabilities accounting for rounding
2. Determine which bracket a temperature settles into
3. Validate if an edge is real after rounding
4. Filter brackets to relevant temperature ranges
"""

from typing import Optional, Tuple
import numpy as np
from scipy import stats
from scipy.stats import skewnorm


def kalshi_round(temp: float) -> int:
    """
    Round temperature using Kalshi/NWS settlement rules.

    Standard rounding: ≥0.5 rounds up, <0.5 rounds down.

    Args:
        temp: Temperature in Fahrenheit

    Returns:
        Rounded integer temperature

    Examples:
        37.4 → 37
        37.5 → 38
        37.9 → 38
        36.0 → 36
    """
    return int(np.round(temp))


def get_settlement_range(bracket_low: Optional[float], bracket_high: Optional[float]) -> Tuple[float, float]:
    """
    Get the continuous temperature range that settles into a bracket.

    For bracket "36-37°F":
    - 35.5°F rounds to 36°F (IN bracket)
    - 37.4°F rounds to 37°F (IN bracket)
    - 37.5°F rounds to 38°F (OUT of bracket)

    Args:
        bracket_low: Lower bound of bracket (None for "≤X" brackets)
        bracket_high: Upper bound of bracket (None for "≥X" brackets)

    Returns:
        Tuple of (continuous_low, continuous_high) for probability calculation
    """
    if bracket_low is None and bracket_high is not None:
        # "≤X" bracket: anything that rounds to X or below
        # e.g., "≤34°F" means actual temp < 34.5°F
        return (-np.inf, bracket_high + 0.5)

    elif bracket_high is None and bracket_low is not None:
        # "≥X" bracket: anything that rounds to X or above
        # e.g., "≥42°F" means actual temp ≥ 41.5°F
        return (bracket_low - 0.5, np.inf)

    elif bracket_low is not None and bracket_high is not None:
        # Range bracket "X-Y°F"
        # e.g., "36-37°F" means 35.5 ≤ actual < 37.5
        return (bracket_low - 0.5, bracket_high + 0.5)

    else:
        # Invalid bracket
        return (0, 0)


def get_bracket_probability_with_rounding(
    forecast_mean: float,
    forecast_std: float,
    bracket_low: Optional[float],
    bracket_high: Optional[float],
    skew: float = 0.0,
) -> float:
    """
    Calculate probability of landing in a bracket, accounting for Kalshi rounding.

    This is the KEY function that fixes the rounding issue.

    Args:
        forecast_mean: Forecast temperature mean
        forecast_std: Forecast standard deviation
        bracket_low: Lower bound (None for "≤X")
        bracket_high: Upper bound (None for "≥X")
        skew: Distribution skewness (0 for normal)

    Returns:
        Probability (0-1) of settling in this bracket

    Examples:
        Forecast 37.9°F, std 2.0, bracket 38-39°F:
        - Without rounding: P(38 ≤ X ≤ 39) ≈ 19%
        - With rounding: P(37.5 ≤ X < 39.5) ≈ 38% (much higher!)

        Forecast 56.6°F, std 2.0, bracket 57-58°F:
        - Without rounding: P(57 ≤ X ≤ 58) ≈ 15%
        - With rounding: P(56.5 ≤ X < 58.5) ≈ 34% (56.6 rounds to 57, IN bracket)
    """
    # Get continuous range that settles into this bracket
    cont_low, cont_high = get_settlement_range(bracket_low, bracket_high)

    # Ensure std has a reasonable floor
    std = max(forecast_std, 1.0)

    # Calculate probability using appropriate distribution
    if abs(skew) >= 0.1:
        # Use skew-normal
        if cont_high == np.inf:
            # "≥X" bracket
            prob = 1 - skewnorm.cdf(cont_low, a=skew, loc=forecast_mean, scale=std)
        elif cont_low == -np.inf:
            # "≤X" bracket
            prob = skewnorm.cdf(cont_high, a=skew, loc=forecast_mean, scale=std)
        else:
            # Range bracket
            prob = skewnorm.cdf(cont_high, a=skew, loc=forecast_mean, scale=std) - \
                   skewnorm.cdf(cont_low, a=skew, loc=forecast_mean, scale=std)
    else:
        # Use normal distribution
        if cont_high == np.inf:
            prob = 1 - stats.norm.cdf(cont_low, loc=forecast_mean, scale=std)
        elif cont_low == -np.inf:
            prob = stats.norm.cdf(cont_high, loc=forecast_mean, scale=std)
        else:
            prob = stats.norm.cdf(cont_high, loc=forecast_mean, scale=std) - \
                   stats.norm.cdf(cont_low, loc=forecast_mean, scale=std)

    return float(np.clip(prob, 0.001, 0.999))


def get_determined_bracket(
    observed_high: float,
    brackets: list,
) -> Optional[str]:
    """
    For same-day markets with known outcome, find which bracket won.

    Args:
        observed_high: NWS observed high temperature (already rounded integer)
        brackets: List of bracket objects with temp_low, temp_high, ticker

    Returns:
        Ticker of the winning bracket, or None if not found
    """
    rounded_high = kalshi_round(observed_high)

    for bracket in brackets:
        temp_low = bracket.temp_low
        temp_high = bracket.temp_high

        if temp_low is None and temp_high is not None:
            # "≤X" bracket
            if rounded_high <= temp_high:
                return bracket.ticker
        elif temp_high is None and temp_low is not None:
            # "≥X" bracket
            if rounded_high >= temp_low:
                return bracket.ticker
        elif temp_low is not None and temp_high is not None:
            # Range bracket
            if temp_low <= rounded_high <= temp_high:
                return bracket.ticker

    return None


def is_bracket_relevant(
    forecast_mean: float,
    forecast_std: float,
    bracket_low: Optional[float],
    bracket_high: Optional[float],
    std_threshold: float = 2.5,
) -> bool:
    """
    Check if a bracket is within reasonable range of forecast.

    Filters out brackets that are highly unlikely given the forecast.
    For example, don't show 61-62°F bracket when forecast is 60.1°F ± 2°F.

    Args:
        forecast_mean: Forecast temperature
        forecast_std: Forecast std dev
        bracket_low: Lower bound (None for "≤X")
        bracket_high: Upper bound (None for "≥X")
        std_threshold: Number of std devs to consider relevant (default 2.5)

    Returns:
        True if bracket is within forecast range
    """
    # Calculate forecast range
    low_bound = forecast_mean - std_threshold * forecast_std
    high_bound = forecast_mean + std_threshold * forecast_std

    # Check bracket overlap with forecast range
    if bracket_low is None and bracket_high is not None:
        # "≤X" bracket - relevant if upper bound is within range
        return bracket_high >= low_bound - 1

    elif bracket_high is None and bracket_low is not None:
        # "≥X" bracket - relevant if lower bound is within range
        return bracket_low <= high_bound + 1

    elif bracket_low is not None and bracket_high is not None:
        # Range bracket - check overlap
        return not (bracket_high < low_bound or bracket_low > high_bound)

    return False


def validate_edge_with_rounding(
    forecast_mean: float,
    forecast_std: float,
    bracket_low: Optional[float],
    bracket_high: Optional[float],
    market_prob: float,
    min_edge: float = 0.05,
    skew: float = 0.0,
) -> dict:
    """
    Validate if an edge is real after accounting for Kalshi rounding.

    Many "edges" disappear when you properly account for rounding.

    Args:
        forecast_mean: Forecast temperature
        forecast_std: Forecast std dev
        bracket_low: Lower bound
        bracket_high: Upper bound
        market_prob: Market's implied probability (0-1)
        min_edge: Minimum edge to consider tradeable
        skew: Distribution skewness

    Returns:
        Dict with:
        - our_prob: Our probability with rounding
        - edge: Actual edge (our_prob - market_prob)
        - is_real: Whether edge exceeds minimum
        - rounded_forecast: What forecast rounds to
        - forecast_in_bracket: Whether rounded forecast is in this bracket
    """
    # Calculate probability with proper rounding
    our_prob = get_bracket_probability_with_rounding(
        forecast_mean, forecast_std, bracket_low, bracket_high, skew
    )

    # Calculate edge
    edge = our_prob - market_prob

    # Check if rounded forecast falls in this bracket
    rounded_forecast = kalshi_round(forecast_mean)

    if bracket_low is None and bracket_high is not None:
        forecast_in_bracket = rounded_forecast <= bracket_high
    elif bracket_high is None and bracket_low is not None:
        forecast_in_bracket = rounded_forecast >= bracket_low
    elif bracket_low is not None and bracket_high is not None:
        forecast_in_bracket = bracket_low <= rounded_forecast <= bracket_high
    else:
        forecast_in_bracket = False

    return {
        "our_prob": our_prob,
        "market_prob": market_prob,
        "edge": edge,
        "is_real": abs(edge) >= min_edge,
        "rounded_forecast": rounded_forecast,
        "forecast_in_bracket": forecast_in_bracket,
        "bracket_desc": format_bracket(bracket_low, bracket_high),
    }


def format_bracket(bracket_low: Optional[float], bracket_high: Optional[float]) -> str:
    """Format bracket bounds as human-readable string."""
    if bracket_low is None and bracket_high is not None:
        return f"≤{bracket_high:.0f}°F"
    elif bracket_high is None and bracket_low is not None:
        return f"≥{bracket_low:.0f}°F"
    elif bracket_low is not None and bracket_high is not None:
        return f"{bracket_low:.0f}-{bracket_high:.0f}°F"
    return "Unknown"


def get_best_bracket_for_forecast(
    forecast_mean: float,
    brackets: list,
) -> Optional[str]:
    """
    Find the bracket most likely to win given a forecast.

    Uses rounded forecast to determine the "center" bracket.

    Args:
        forecast_mean: Forecast temperature
        brackets: List of bracket objects

    Returns:
        Ticker of the bracket containing the rounded forecast
    """
    rounded = kalshi_round(forecast_mean)

    for bracket in brackets:
        temp_low = bracket.temp_low
        temp_high = bracket.temp_high

        if temp_low is None and temp_high is not None:
            if rounded <= temp_high:
                return bracket.ticker
        elif temp_high is None and temp_low is not None:
            if rounded >= temp_low:
                return bracket.ticker
        elif temp_low is not None and temp_high is not None:
            if temp_low <= rounded <= temp_high:
                return bracket.ticker

    return None
