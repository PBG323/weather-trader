"""
Position Sizing with Kelly Criterion

The Kelly Criterion determines optimal bet sizing based on:
- Edge (probability advantage)
- Odds

We use fractional Kelly (typically 0.25x) to reduce variance.

Risk management includes:
- Maximum position size per market
- Daily loss limits
- Portfolio-level constraints
"""

from dataclasses import dataclass
from typing import Optional

from ..config import config
from .expected_value import TradeSignal


@dataclass
class PositionRecommendation:
    """Recommended position size for a trade."""
    signal: TradeSignal

    # Position sizing
    kelly_fraction: float          # Full Kelly fraction
    adjusted_kelly: float          # After applying multiplier
    recommended_size: float        # Dollar amount to risk
    recommended_shares: float      # Number of shares at current price

    # Risk metrics
    max_loss: float               # Maximum possible loss
    expected_profit: float        # Expected profit based on edge
    risk_reward_ratio: float      # Expected profit / max loss

    # Constraints applied
    position_capped: bool         # Was position reduced due to max size?
    bankroll_limited: bool        # Was position reduced due to bankroll?

    @property
    def should_trade(self) -> bool:
        """Check if position size warrants a trade."""
        return self.recommended_size >= 1.0  # Minimum $1 trade


class PositionSizer:
    """
    Calculates position sizes using Kelly Criterion with risk management.
    """

    def __init__(
        self,
        bankroll: float,
        kelly_multiplier: Optional[float] = None,
        max_position_percent: Optional[float] = None,
        daily_loss_limit_percent: Optional[float] = None,
    ):
        """
        Initialize position sizer.

        Args:
            bankroll: Total trading capital (USD)
            kelly_multiplier: Fraction of Kelly to use (default from config)
            max_position_percent: Max % of bankroll per position (default from config)
            daily_loss_limit_percent: Stop trading if down this % (default from config)
        """
        self.bankroll = bankroll
        self.kelly_multiplier = kelly_multiplier or config.trading.kelly_fraction
        self.max_position_percent = max_position_percent or config.trading.max_position_percent
        self.daily_loss_limit_percent = daily_loss_limit_percent or config.trading.daily_loss_limit_percent

        # Track daily P&L
        self.daily_pnl = 0.0
        self.positions_today = []

    def calculate_kelly(self, win_prob: float, win_amount: float, lose_amount: float) -> float:
        """
        Calculate Kelly fraction for a bet.

        Kelly formula: f* = (p * b - q) / b
        where:
            f* = fraction of bankroll to bet
            p = probability of winning
            q = probability of losing (1 - p)
            b = odds (win_amount / lose_amount)

        Args:
            win_prob: Probability of winning
            win_amount: Amount won if successful (per dollar risked)
            lose_amount: Amount lost if unsuccessful (per dollar risked)

        Returns:
            Kelly fraction (can be negative if -EV)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        if lose_amount <= 0:
            return 0.0

        q = 1 - win_prob
        b = win_amount / lose_amount

        kelly = (win_prob * b - q) / b

        return max(0, kelly)  # Never recommend negative sizing

    def size_position(self, signal: TradeSignal) -> PositionRecommendation:
        """
        Calculate recommended position size for a trade signal.

        Args:
            signal: TradeSignal from EV calculator

        Returns:
            PositionRecommendation with sizing details
        """
        # Determine win/lose amounts based on the side we're taking
        if signal.side == "YES":
            entry_price = signal.bracket.yes_price
            # If we buy YES at price P:
            # - Win: receive $1, profit = 1 - P
            # - Lose: receive $0, loss = P
            win_amount = 1 - entry_price
            lose_amount = entry_price
            win_prob = signal.forecast_probability
        else:
            # If we buy NO (effectively selling YES):
            entry_price = signal.bracket.no_price
            win_amount = 1 - entry_price
            lose_amount = entry_price
            win_prob = 1 - signal.forecast_probability

        # Calculate Kelly fraction
        kelly = self.calculate_kelly(win_prob, win_amount, lose_amount)

        # Apply Kelly multiplier (fractional Kelly)
        adjusted_kelly = kelly * self.kelly_multiplier

        # Calculate dollar amount
        raw_size = self.bankroll * adjusted_kelly

        # Apply maximum position constraint
        max_position = self.bankroll * (self.max_position_percent / 100)
        position_capped = raw_size > max_position
        sized_amount = min(raw_size, max_position)

        # Check daily loss limit
        daily_limit = self.bankroll * (self.daily_loss_limit_percent / 100)
        remaining_risk = daily_limit + self.daily_pnl  # How much more we can lose

        bankroll_limited = sized_amount > remaining_risk
        if remaining_risk <= 0:
            sized_amount = 0  # Stop trading for the day
        else:
            sized_amount = min(sized_amount, remaining_risk)

        # Calculate shares
        shares = sized_amount / entry_price if entry_price > 0 else 0

        # Calculate risk metrics
        max_loss = sized_amount
        expected_profit = sized_amount * signal.expected_value
        risk_reward = expected_profit / max_loss if max_loss > 0 else 0

        return PositionRecommendation(
            signal=signal,
            kelly_fraction=kelly,
            adjusted_kelly=adjusted_kelly,
            recommended_size=sized_amount,
            recommended_shares=shares,
            max_loss=max_loss,
            expected_profit=expected_profit,
            risk_reward_ratio=risk_reward,
            position_capped=position_capped,
            bankroll_limited=bankroll_limited,
        )

    def update_bankroll(self, pnl: float):
        """
        Update bankroll after a trade settles.

        Args:
            pnl: Profit or loss from the trade
        """
        self.bankroll += pnl
        self.daily_pnl += pnl

    def reset_daily_tracking(self):
        """Reset daily P&L tracking (call at start of each day)."""
        self.daily_pnl = 0.0
        self.positions_today = []

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if we should continue trading today.

        Returns:
            Tuple of (can_trade, reason)
        """
        daily_limit = self.bankroll * (self.daily_loss_limit_percent / 100)

        if -self.daily_pnl >= daily_limit:
            return False, f"Daily loss limit reached: ${-self.daily_pnl:.2f}"

        if self.bankroll <= 0:
            return False, "Bankroll depleted"

        return True, "OK"

    def get_portfolio_summary(self) -> dict:
        """
        Get summary of current portfolio state.

        Returns:
            Dictionary with portfolio metrics
        """
        daily_limit = self.bankroll * (self.daily_loss_limit_percent / 100)

        return {
            "bankroll": self.bankroll,
            "daily_pnl": self.daily_pnl,
            "daily_limit": daily_limit,
            "remaining_risk_budget": daily_limit + self.daily_pnl,
            "positions_today": len(self.positions_today),
            "can_trade": self.can_trade()[0],
        }

    def recommend_batch(
        self,
        signals: list[TradeSignal],
        max_positions: int = 5
    ) -> list[PositionRecommendation]:
        """
        Recommend positions for multiple signals.

        Allocates capital across best opportunities while respecting
        portfolio-level constraints.

        Args:
            signals: List of TradeSignal objects (should be sorted by edge)
            max_positions: Maximum number of positions to take

        Returns:
            List of PositionRecommendation objects
        """
        recommendations = []
        remaining_capital = self.bankroll

        for signal in signals[:max_positions]:
            if not signal.is_tradeable:
                continue

            # Create temporary sizer with remaining capital
            temp_sizer = PositionSizer(
                bankroll=remaining_capital,
                kelly_multiplier=self.kelly_multiplier,
                max_position_percent=self.max_position_percent,
                daily_loss_limit_percent=100,  # No daily limit for batch
            )

            rec = temp_sizer.size_position(signal)

            if rec.should_trade:
                recommendations.append(rec)
                remaining_capital -= rec.recommended_size

            if remaining_capital <= 0:
                break

        return recommendations
