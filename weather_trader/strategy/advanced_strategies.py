"""
Advanced Trading Strategies

Implements strategies used by successful weather traders:

1. TEMPERATURE LADDERING (Neobrother style)
   - Multi-bracket positioning across temperature range
   - Small bets, many positions
   - One winner covers all losers

2. TAIL BRACKET HUNTING (Hans323 / Reddit style)
   - Find 10x+ opportunities at 2-10¢
   - "Buy YES under 10-15¢ when 3+ models agree"
   - "Buy NO above 40-50¢ when range clearly wrong"

3. MODEL CONSENSUS DETECTION
   - "Simple rule: if 3+ models agree and odds are under 10-15¢,
     that's usually not gambling"

Key insight from Reddit weather traders:
"The pattern is almost always the same:
 buy YES under ~10–15c when models line up
 or buy NO above ~40–50c when a range is clearly wrong"
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import numpy as np
from scipy import stats

from ..models.ensemble import EnsembleForecast


class StrategyType(Enum):
    """Type of trading strategy."""
    CORE = "core"  # Standard edge-based trading
    LADDER = "ladder"  # Multi-bracket laddering
    TAIL = "tail"  # Tail bracket hunting


@dataclass
class LadderPosition:
    """A single position in a temperature ladder."""
    temp_bracket: str  # e.g., "52-54°F"
    temp_low: Optional[float]
    temp_high: Optional[float]
    target_price: float  # Target entry price (cents)
    max_price: float  # Maximum price willing to pay
    allocation_pct: float  # % of ladder allocation for this rung
    our_probability: float  # Our calculated probability
    potential_return: float  # Potential return multiple if wins


@dataclass
class TailOpportunity:
    """A tail bracket trading opportunity."""
    ticker: str
    bracket_desc: str
    temp_low: Optional[float]
    temp_high: Optional[float]
    side: str  # "YES" or "NO"
    market_price: float  # Current market price (0-1)
    our_probability: float  # Our calculated probability
    edge: float  # our_prob - market_prob (for YES)
    potential_return: float  # Return multiple if wins
    models_agreeing: int  # Number of models that agree
    consensus_ratio: float  # 0-1, how strongly models agree
    confidence_score: float  # Combined confidence metric
    recommendation: str  # Human-readable recommendation


@dataclass
class LadderStrategy:
    """
    Complete ladder strategy for a city/date.

    Based on Neobrother's approach: simultaneously bet across multiple
    temperature brackets, weighted by probability.

    Example: For forecast 52°F ± 2°F std:
    - 48°F: 2% allocation at 3¢
    - 50°F: 10% allocation at 8¢
    - 52°F: 30% allocation at 15¢  <- Peak
    - 54°F: 30% allocation at 15¢
    - 56°F: 10% allocation at 8¢
    - 58°F: 2% allocation at 3¢

    One winning position covers all losses.
    """
    city: str
    target_date: datetime
    forecast: EnsembleForecast
    total_allocation: float  # Total $ to allocate to ladder
    positions: list[LadderPosition] = field(default_factory=list)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def expected_value(self) -> float:
        """Calculate expected value of entire ladder."""
        ev = 0
        for pos in self.positions:
            # EV = prob * win - (1-prob) * cost
            cost = pos.target_price * pos.allocation_pct * self.total_allocation
            win = pos.allocation_pct * self.total_allocation  # $1 per contract
            ev += pos.our_probability * win - (1 - pos.our_probability) * cost
        return ev


def generate_temperature_ladder(
    forecast: EnsembleForecast,
    available_brackets: list[dict],
    total_allocation: float,
    max_rungs: int = 7,
    min_probability: float = 0.02,
    max_entry_price: float = 0.20,
) -> LadderStrategy:
    """
    Generate a temperature ladder strategy.

    This implements Neobrother's approach: spread bets across multiple
    adjacent temperature brackets, with more allocation near the forecast mean.

    Args:
        forecast: Ensemble forecast with mean and std
        available_brackets: List of available market brackets
        total_allocation: Total $ to allocate to this ladder
        max_rungs: Maximum number of ladder positions
        min_probability: Minimum probability to include in ladder
        max_entry_price: Maximum price willing to pay per bracket

    Returns:
        LadderStrategy with positioned rungs
    """
    positions = []

    # Sort brackets by temperature midpoint
    sorted_brackets = sorted(
        available_brackets,
        key=lambda b: (b.get("temp_low", 0) or 0) + (b.get("temp_high", 100) or 100)
    )

    # Calculate probability for each bracket
    bracket_probs = []
    for bracket in sorted_brackets:
        temp_low = bracket.get("temp_low")
        temp_high = bracket.get("temp_high")

        if temp_low is None and temp_high is None:
            continue

        # Calculate our probability using forecast distribution
        if temp_low is not None and temp_high is not None:
            prob = forecast.get_probability_in_range(temp_low, temp_high)
        elif temp_high is not None:
            prob = forecast.get_probability_below(temp_high + 0.5)
        else:
            prob = forecast.get_probability_above(temp_low - 0.5)

        market_price = bracket.get("yes_price", 0.5)

        # Only include if probability is significant and price is cheap enough
        if prob >= min_probability and market_price <= max_entry_price:
            bracket_probs.append({
                "bracket": bracket,
                "probability": prob,
                "market_price": market_price,
                "edge": prob - market_price,
            })

    # Sort by probability (highest first) and take top rungs
    bracket_probs.sort(key=lambda x: x["probability"], reverse=True)
    top_brackets = bracket_probs[:max_rungs]

    # Allocate proportionally to probability
    total_prob = sum(b["probability"] for b in top_brackets)

    for bp in top_brackets:
        bracket = bp["bracket"]
        prob = bp["probability"]
        market_price = bp["market_price"]

        # Allocation weighted by probability
        allocation_pct = prob / total_prob if total_prob > 0 else 1 / len(top_brackets)

        # Target price: slightly above current for fill, but cap at our probability
        target_price = min(market_price + 0.02, prob * 0.8)  # Don't pay more than 80% of fair value
        max_price = min(market_price + 0.05, prob * 0.9)

        # Potential return
        potential_return = (1 / target_price) - 1 if target_price > 0 else 0

        temp_low = bracket.get("temp_low")
        temp_high = bracket.get("temp_high")
        bracket_desc = bracket.get("outcome_desc", f"{temp_low}-{temp_high}°F")

        positions.append(LadderPosition(
            temp_bracket=bracket_desc,
            temp_low=temp_low,
            temp_high=temp_high,
            target_price=target_price,
            max_price=max_price,
            allocation_pct=allocation_pct,
            our_probability=prob,
            potential_return=potential_return,
        ))

    return LadderStrategy(
        city=forecast.city,
        target_date=forecast.date,
        forecast=forecast,
        total_allocation=total_allocation,
        positions=positions,
    )


def find_tail_opportunities(
    markets: list[dict],
    forecasts: dict[str, EnsembleForecast],
    min_potential_return: float = 5.0,  # Minimum 5x return
    max_yes_price: float = 0.15,  # Max 15¢ for YES
    min_no_price: float = 0.40,  # Min 40¢ for NO (we're selling)
    min_edge: float = 0.03,  # Minimum 3% edge
    min_consensus_models: int = 3,  # At least 3 models must agree
) -> list[TailOpportunity]:
    """
    Find tail bracket opportunities with high return potential.

    Implements the Reddit trader rule:
    "Buy YES under 10-15¢ when models line up,
     or buy NO above 40-50¢ when range clearly wrong"

    Args:
        markets: List of available markets with brackets
        forecasts: Dict of city -> EnsembleForecast
        min_potential_return: Minimum return multiple (e.g., 5 = 5x)
        max_yes_price: Maximum YES price to consider (default 15¢)
        min_no_price: Minimum NO price for "clearly wrong" (default 40¢)
        min_edge: Minimum edge required
        min_consensus_models: Minimum models that must agree

    Returns:
        List of TailOpportunity sorted by confidence score
    """
    opportunities = []

    for market in markets:
        city_key = market.get("city_key", "").lower()
        forecast = forecasts.get(city_key)

        if not forecast:
            continue

        # Check model consensus
        models_agreeing = int(forecast.consensus_ratio * forecast.model_count)
        if models_agreeing < min_consensus_models:
            continue

        brackets = market.get("brackets", [])
        if not brackets:
            # Try alternate structure
            brackets = [market] if "yes_price" in market else []

        for bracket in brackets:
            yes_price = bracket.get("yes_price", 0.5)
            temp_low = bracket.get("temp_low")
            temp_high = bracket.get("temp_high")
            ticker = bracket.get("ticker", bracket.get("condition_id", ""))
            bracket_desc = bracket.get("outcome_desc", f"{temp_low}-{temp_high}°F")

            # Calculate our probability
            if temp_low is not None and temp_high is not None:
                our_prob = forecast.get_probability_in_range(temp_low, temp_high)
            elif temp_high is not None:
                our_prob = forecast.get_probability_below(temp_high + 0.5)
            elif temp_low is not None:
                our_prob = forecast.get_probability_above(temp_low - 0.5)
            else:
                continue

            # Check for YES tail opportunity (cheap YES, models say likely)
            if yes_price <= max_yes_price:
                edge = our_prob - yes_price
                potential_return = (1 / yes_price) - 1 if yes_price > 0 else 0

                if edge >= min_edge and potential_return >= min_potential_return:
                    # Calculate confidence score
                    # Higher edge + higher consensus + lower price = better
                    confidence = (
                        edge * 2 +  # Edge contribution
                        forecast.consensus_ratio * 0.5 +  # Consensus contribution
                        (1 - yes_price) * 0.3  # Cheapness contribution
                    )

                    opportunities.append(TailOpportunity(
                        ticker=ticker,
                        bracket_desc=bracket_desc,
                        temp_low=temp_low,
                        temp_high=temp_high,
                        side="YES",
                        market_price=yes_price,
                        our_probability=our_prob,
                        edge=edge,
                        potential_return=potential_return,
                        models_agreeing=models_agreeing,
                        consensus_ratio=forecast.consensus_ratio,
                        confidence_score=confidence,
                        recommendation=f"BUY YES at {yes_price*100:.0f}¢ - {models_agreeing} models agree, {potential_return:.0f}x potential return",
                    ))

            # Check for NO tail opportunity (expensive YES, models say unlikely)
            if yes_price >= min_no_price:
                no_price = 1 - yes_price
                our_no_prob = 1 - our_prob
                edge = our_no_prob - no_price
                potential_return = (1 / no_price) - 1 if no_price > 0 else 0

                if edge >= min_edge and potential_return >= min_potential_return:
                    confidence = (
                        edge * 2 +
                        forecast.consensus_ratio * 0.5 +
                        yes_price * 0.3  # Higher YES price = more room for NO
                    )

                    opportunities.append(TailOpportunity(
                        ticker=ticker,
                        bracket_desc=bracket_desc,
                        temp_low=temp_low,
                        temp_high=temp_high,
                        side="NO",
                        market_price=no_price,
                        our_probability=our_no_prob,
                        edge=edge,
                        potential_return=potential_return,
                        models_agreeing=models_agreeing,
                        consensus_ratio=forecast.consensus_ratio,
                        confidence_score=confidence,
                        recommendation=f"BUY NO at {no_price*100:.0f}¢ - Bracket clearly wrong, {potential_return:.0f}x potential return",
                    ))

    # Sort by confidence score (highest first)
    opportunities.sort(key=lambda x: x.confidence_score, reverse=True)

    return opportunities


def calculate_bankroll_allocation(
    bankroll: float,
    core_pct: float = 0.60,  # 60% to core strategy
    ladder_pct: float = 0.25,  # 25% to laddering
    tail_pct: float = 0.15,  # 15% to tail hunting
) -> dict:
    """
    Calculate bankroll allocation across strategies.

    Default allocation based on recommendation:
    - 60% Core: High-probability, edge-based trading
    - 25% Ladder: Multi-bracket temperature laddering
    - 15% Tail: Low-probability, high-return hunting

    Args:
        bankroll: Total bankroll
        core_pct: Percentage for core strategy
        ladder_pct: Percentage for ladder strategy
        tail_pct: Percentage for tail strategy

    Returns:
        Dict with allocations for each strategy
    """
    assert abs(core_pct + ladder_pct + tail_pct - 1.0) < 0.01, "Allocations must sum to 1"

    return {
        "core": {
            "allocation": bankroll * core_pct,
            "percentage": core_pct,
            "description": "High-probability edge trades",
        },
        "ladder": {
            "allocation": bankroll * ladder_pct,
            "percentage": ladder_pct,
            "description": "Multi-bracket temperature laddering",
        },
        "tail": {
            "allocation": bankroll * tail_pct,
            "percentage": tail_pct,
            "description": "Low-probability tail hunting",
        },
    }


def score_model_consensus(forecast: EnsembleForecast, threshold_temp: float) -> dict:
    """
    Score how strongly models agree on a temperature outcome.

    The Reddit rule: "If 3+ models agree and odds are under 10-15¢,
    that's usually not gambling"

    Args:
        forecast: Ensemble forecast
        threshold_temp: Temperature threshold to check

    Returns:
        Dict with consensus analysis
    """
    if not forecast.model_forecasts:
        return {"has_consensus": False, "reason": "No model data"}

    model_highs = [m.forecast_high for m in forecast.model_forecasts]

    # Count models within 2°F of threshold
    close_to_threshold = sum(1 for h in model_highs if abs(h - threshold_temp) <= 2)

    # Count models above/below threshold
    above = sum(1 for h in model_highs if h >= threshold_temp)
    below = sum(1 for h in model_highs if h < threshold_temp)

    total = len(model_highs)

    # Strong consensus if 3+ models agree on direction
    has_consensus = above >= 3 or below >= 3
    consensus_direction = "ABOVE" if above > below else "BELOW"
    consensus_strength = max(above, below) / total if total > 0 else 0

    return {
        "has_consensus": has_consensus,
        "models_above": above,
        "models_below": below,
        "models_close": close_to_threshold,
        "total_models": total,
        "consensus_direction": consensus_direction,
        "consensus_strength": consensus_strength,
        "consensus_ratio": forecast.consensus_ratio,
        "recommendation": f"{max(above, below)}/{total} models agree temp will be {consensus_direction} {threshold_temp}°F" if has_consensus else "No strong consensus",
    }
