"""
Position Manager with Smart Exit Logic

Manages positions with edge-based profit taking rather than arbitrary targets.
Core principle: Hold while edge is expanding, exit when edge is exhausted or reversed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from collections import deque
import numpy as np

from .config import TradingConfig, default_config


class PositionStatus(Enum):
    """Position lifecycle states."""
    PENDING = "pending"          # Order placed, not filled
    OPEN = "open"                # Position is active
    CLOSING = "closing"          # Exit order placed
    CLOSED = "closed"            # Position fully closed


class ExitReason(Enum):
    """Reasons for closing a position."""
    EDGE_EXHAUSTED = "edge_exhausted"       # Captured most of available edge
    EDGE_REVERSED = "edge_reversed"         # Edge turned negative
    MOMENTUM_SHIFT = "momentum_shift"       # Edge trending down while profitable
    TIME_DECAY = "time_decay"               # Approaching settlement
    STOP_LOSS = "stop_loss"                 # Hit maximum loss threshold
    TRAILING_STOP = "trailing_stop"         # Gave back too much profit
    MANUAL = "manual"                       # User-initiated close
    SETTLEMENT = "settlement"               # Market settled
    RISK_LIMIT = "risk_limit"               # Portfolio risk limit triggered


@dataclass
class EdgeSnapshot:
    """Point-in-time edge measurement."""
    timestamp: datetime
    current_edge: float           # Current forecast prob - market price
    market_price: float           # Current market price for our side
    forecast_prob: float          # Our model's probability
    unrealized_pnl: float         # Current P/L if closed now


@dataclass
class Position:
    """
    Represents an open trading position with edge tracking.

    Tracks the evolution of edge over time to make smart exit decisions.
    """
    # Identity
    position_id: str
    market_id: str
    condition_id: str
    city: str
    outcome_description: str      # e.g., "20-21°F"
    settlement_date: datetime

    # Entry details
    side: str                     # "YES" or "NO"
    entry_price: float            # Price paid per share
    shares: float                 # Number of shares
    entry_time: datetime
    entry_edge: float             # Edge at time of entry
    entry_forecast_prob: float    # Model probability at entry

    # Current state
    status: PositionStatus = PositionStatus.OPEN
    current_price: float = 0.0
    current_forecast_prob: float = 0.0

    # Tracking
    edge_history: deque = field(default_factory=lambda: deque(maxlen=20))
    peak_unrealized_pnl: float = 0.0
    peak_edge: float = 0.0

    # Exit details (populated on close)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    realized_pnl: Optional[float] = None

    def __post_init__(self):
        """Initialize tracking fields."""
        if not self.edge_history:
            self.edge_history = deque(maxlen=20)
        self.peak_edge = self.entry_edge

    @property
    def cost_basis(self) -> float:
        """
        Total cost of position (what we actually paid).

        Note: entry_price stores YES price for both YES and NO positions.
        - YES: paid entry_price per share
        - NO: paid (1 - entry_price) per share
        """
        if self.side == "YES":
            return self.entry_price * self.shares
        else:
            return (1 - self.entry_price) * self.shares

    @property
    def current_value(self) -> float:
        """
        Current market value of position.

        Note: current_price stores YES price for both YES and NO positions.
        - YES: worth current_price per share
        - NO: worth (1 - current_price) per share
        """
        if self.side == "YES":
            return self.current_price * self.shares
        else:
            return (1 - self.current_price) * self.shares

    @property
    def unrealized_pnl(self) -> float:
        """
        Current unrealized profit/loss.

        This is simply current_value - cost_basis.
        The side-specific logic is handled in cost_basis and current_value.
        """
        return self.current_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P/L as percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        # Use absolute value of entry for percentage calculation
        return self.unrealized_pnl / self.cost_basis

    @property
    def current_edge(self) -> float:
        """
        Current edge (our probability - market price).

        For YES: edge = forecast_prob - YES_price (positive = underpriced)
        For NO: edge = (1 - forecast_prob) - (1 - YES_price) = YES_price - forecast_prob
                (positive = YES is overpriced, so NO is underpriced)

        Note: current_price stores YES price for both YES and NO positions.
        """
        if self.side == "YES":
            return self.current_forecast_prob - self.current_price
        else:
            # NO edge: YES_price - forecast_prob (we profit when YES is overpriced)
            return self.current_price - self.current_forecast_prob

    @property
    def edge_captured_pct(self) -> float:
        """
        Percentage of original edge that has been captured.

        If entry_edge was 0.10 and current_edge is 0.03, we've captured 70%.
        """
        if self.entry_edge <= 0:
            return 1.0  # Edge was never positive

        edge_reduction = self.entry_edge - self.current_edge
        return edge_reduction / self.entry_edge

    @property
    def hours_until_settlement(self) -> float:
        """Hours remaining until market settlement."""
        delta = self.settlement_date - datetime.now()
        return max(0, delta.total_seconds() / 3600)

    def record_edge_snapshot(self, market_price: float, forecast_prob: float) -> None:
        """
        Record current edge state for trend analysis.

        Call this periodically (e.g., every market update) to build edge history.
        """
        self.current_price = market_price
        self.current_forecast_prob = forecast_prob

        snapshot = EdgeSnapshot(
            timestamp=datetime.now(),
            current_edge=self.current_edge,
            market_price=market_price,
            forecast_prob=forecast_prob,
            unrealized_pnl=self.unrealized_pnl
        )
        self.edge_history.append(snapshot)

        # Track peaks
        if self.unrealized_pnl > self.peak_unrealized_pnl:
            self.peak_unrealized_pnl = self.unrealized_pnl
        if self.current_edge > self.peak_edge:
            self.peak_edge = self.current_edge

    def get_edge_trend(self, lookback: int = 5) -> float:
        """
        Calculate the trend of edge over recent history.

        Returns:
            Negative value = edge is shrinking (bearish for position)
            Positive value = edge is expanding (bullish for position)
            Zero = stable edge
        """
        if len(self.edge_history) < 2:
            return 0.0

        recent = list(self.edge_history)[-lookback:]
        if len(recent) < 2:
            return 0.0

        edges = [s.current_edge for s in recent]

        # Simple linear regression slope
        x = np.arange(len(edges))
        slope = np.polyfit(x, edges, 1)[0]

        return slope

    def get_consecutive_negative_trend_count(self) -> int:
        """
        Count consecutive periods where edge decreased.

        Used to detect sustained momentum against the position.
        """
        if len(self.edge_history) < 2:
            return 0

        count = 0
        history = list(self.edge_history)

        for i in range(len(history) - 1, 0, -1):
            if history[i].current_edge < history[i-1].current_edge:
                count += 1
            else:
                break

        return count


class PositionManager:
    """
    Manages all open positions with smart exit logic.

    Exit decisions are based on edge dynamics, not arbitrary profit targets:
    - HOLD if edge is stable or expanding
    - SELL if edge is exhausted (captured most of available edge)
    - SELL if edge reversed (turned negative)
    - SELL if momentum shifting against position while profitable
    - SELL on stop loss / trailing stop
    - SELL on time decay approaching settlement
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or default_config
        self.positions: dict[str, Position] = {}
        self._position_counter = 0

    def create_position(
        self,
        market_id: str,
        condition_id: str,
        city: str,
        outcome_description: str,
        settlement_date: datetime,
        side: str,
        entry_price: float,
        shares: float,
        forecast_prob: float
    ) -> Position:
        """
        Create and register a new position.

        Args:
            market_id: Polymarket market identifier
            condition_id: Specific condition/outcome ID
            city: City for this weather market
            outcome_description: Human-readable outcome (e.g., "20-21°F")
            settlement_date: When market settles
            side: "YES" or "NO"
            entry_price: Price per share
            shares: Number of shares purchased
            forecast_prob: Model's probability at entry

        Returns:
            The created Position object
        """
        self._position_counter += 1
        position_id = f"POS-{self._position_counter:06d}"

        # Calculate entry edge
        # Note: entry_price is always YES price for consistency
        if side == "YES":
            # YES edge: our_prob - YES_price (positive = YES underpriced)
            entry_edge = forecast_prob - entry_price
        else:
            # NO edge: YES_price - forecast_prob (positive = YES overpriced, NO underpriced)
            entry_edge = entry_price - forecast_prob

        position = Position(
            position_id=position_id,
            market_id=market_id,
            condition_id=condition_id,
            city=city,
            outcome_description=outcome_description,
            settlement_date=settlement_date,
            side=side,
            entry_price=entry_price,
            shares=shares,
            entry_time=datetime.now(),
            entry_edge=entry_edge,
            entry_forecast_prob=forecast_prob,
            current_price=entry_price,
            current_forecast_prob=forecast_prob
        )

        # Record initial edge snapshot
        position.record_edge_snapshot(entry_price, forecast_prob)

        self.positions[position_id] = position
        return position

    def update_position(
        self,
        position_id: str,
        market_price: float,
        forecast_prob: float
    ) -> None:
        """
        Update position with latest market data and forecast.

        Should be called on each market update / forecast refresh.
        """
        if position_id not in self.positions:
            raise ValueError(f"Unknown position: {position_id}")

        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            return

        position.record_edge_snapshot(market_price, forecast_prob)

    def should_exit_position(self, position: Position) -> tuple[bool, Optional[ExitReason]]:
        """
        Determine if a position should be closed.

        This is the core smart exit logic. Returns (should_exit, reason).

        Exit triggers (in priority order):
        1. Edge reversed (turned negative) - immediate exit
        2. Stop loss hit - risk management
        3. Trailing stop hit - protect profits
        4. Time decay - approaching settlement
        5. Edge exhausted - captured most available edge
        6. Momentum shift - edge trending down while profitable
        """
        if position.status != PositionStatus.OPEN:
            return False, None

        # 1. EDGE REVERSED - Exit immediately if edge turned negative
        if position.current_edge < 0:
            return True, ExitReason.EDGE_REVERSED

        # 2. STOP LOSS - Exit if loss exceeds threshold
        if position.unrealized_pnl_pct < -self.config.stop_loss_pct:
            return True, ExitReason.STOP_LOSS

        # 3. TRAILING STOP - Exit if gave back too much profit
        if position.peak_unrealized_pnl > 0 and position.unrealized_pnl > 0:
            drawdown_from_peak = (position.peak_unrealized_pnl - position.unrealized_pnl) / position.peak_unrealized_pnl
            if drawdown_from_peak > self.config.trailing_stop_pct:
                return True, ExitReason.TRAILING_STOP

        # 4. TIME DECAY - Exit approaching settlement
        hours_left = position.hours_until_settlement

        # Force close if very close to settlement
        if hours_left < self.config.hours_before_settlement_close:
            return True, ExitReason.TIME_DECAY

        # Consider closing profitable positions as settlement approaches
        if hours_left < self.config.hours_before_settlement_warning:
            if position.unrealized_pnl > 0:
                return True, ExitReason.TIME_DECAY

        # 5. EDGE EXHAUSTED - Exit if captured most of available edge
        edge_captured = position.edge_captured_pct
        if edge_captured >= self.config.edge_exhaustion_threshold:
            # Only exit if remaining edge is small
            if position.current_edge < self.config.min_edge_to_hold:
                return True, ExitReason.EDGE_EXHAUSTED

        # 6. MOMENTUM SHIFT - Exit if edge consistently declining while profitable
        if position.unrealized_pnl > 0:
            negative_trend_count = position.get_consecutive_negative_trend_count()
            if negative_trend_count >= self.config.negative_trend_threshold:
                # Edge is consistently shrinking
                edge_trend = position.get_edge_trend()
                if edge_trend < -0.005:  # Significant negative trend
                    return True, ExitReason.MOMENTUM_SHIFT

        # No exit trigger - continue holding
        return False, None

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: ExitReason
    ) -> Position:
        """
        Close a position and record the exit details.

        Args:
            position_id: Position to close
            exit_price: Price received on exit
            reason: Why the position was closed

        Returns:
            The closed Position with finalized P/L
        """
        if position_id not in self.positions:
            raise ValueError(f"Unknown position: {position_id}")

        position = self.positions[position_id]

        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        position.status = PositionStatus.CLOSED

        # Calculate realized P/L
        exit_value = exit_price * position.shares
        position.realized_pnl = exit_value - position.cost_basis

        return position

    def get_open_positions(self) -> list[Position]:
        """Get all currently open positions."""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def get_positions_by_city(self, city: str) -> list[Position]:
        """Get all open positions for a specific city."""
        return [
            p for p in self.positions.values()
            if p.status == PositionStatus.OPEN and p.city.lower() == city.lower()
        ]

    def get_positions_by_market(self, market_id: str) -> list[Position]:
        """Get all positions for a specific market."""
        return [
            p for p in self.positions.values()
            if p.market_id == market_id
        ]

    def get_total_exposure(self) -> float:
        """Get total capital currently deployed across all positions."""
        return sum(p.cost_basis for p in self.get_open_positions())

    def get_city_exposure(self, city: str) -> float:
        """Get total capital deployed in a specific city."""
        return sum(p.cost_basis for p in self.get_positions_by_city(city))

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P/L across all open positions."""
        return sum(p.unrealized_pnl for p in self.get_open_positions())

    def evaluate_all_positions(self) -> list[tuple[Position, bool, Optional[ExitReason]]]:
        """
        Evaluate all open positions for exit signals.

        Returns:
            List of (position, should_exit, exit_reason) tuples
        """
        results = []
        for position in self.get_open_positions():
            should_exit, reason = self.should_exit_position(position)
            results.append((position, should_exit, reason))
        return results

    def get_position_summary(self, position: Position) -> dict:
        """Get a summary dict of position state for display/logging."""
        return {
            "position_id": position.position_id,
            "city": position.city,
            "outcome": position.outcome_description,
            "side": position.side,
            "entry_price": round(position.entry_price, 4),
            "current_price": round(position.current_price, 4),
            "shares": round(position.shares, 2),
            "cost_basis": round(position.cost_basis, 2),
            "unrealized_pnl": round(position.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(position.unrealized_pnl_pct * 100, 1),
            "entry_edge": round(position.entry_edge * 100, 1),
            "current_edge": round(position.current_edge * 100, 1),
            "edge_captured_pct": round(position.edge_captured_pct * 100, 1),
            "edge_trend": round(position.get_edge_trend() * 100, 2),
            "hours_until_settlement": round(position.hours_until_settlement, 1),
            "status": position.status.value,
        }
