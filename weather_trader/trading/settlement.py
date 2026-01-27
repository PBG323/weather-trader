"""
Settlement Handler

Handles market settlements and position resolution when markets close.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from enum import Enum
import logging

from .position_manager import PositionManager, Position, PositionStatus, ExitReason
from .pnl_tracker import PnLTracker

logger = logging.getLogger(__name__)


class SettlementOutcome(Enum):
    """Outcome of a settled market."""
    YES = "yes"     # YES won
    NO = "no"       # NO won
    VOID = "void"   # Market voided


@dataclass
class SettlementResult:
    """Result of settling a position."""
    position_id: str
    market_id: str
    city: str
    outcome_description: str

    our_side: str               # What we were holding
    winning_side: str           # What actually won

    entry_price: float
    settlement_price: float     # 1.0 if we won, 0.0 if we lost
    shares: float

    realized_pnl: float
    won: bool

    settled_at: datetime = field(default_factory=datetime.now)


class SettlementHandler:
    """
    Handles market settlements for weather markets.

    When a market settles:
    1. Fetch the actual outcome from Weather Underground
    2. Determine winning side
    3. Settle all positions in that market
    4. Record P/L
    """

    def __init__(
        self,
        position_manager: PositionManager = None,
        pnl_tracker: PnLTracker = None
    ):
        self.position_manager = position_manager
        self.pnl_tracker = pnl_tracker

        self.settlement_history: list[SettlementResult] = []

    def settle_position(
        self,
        position: Position,
        winning_side: str
    ) -> SettlementResult:
        """
        Settle a single position based on market outcome.

        Args:
            position: Position to settle
            winning_side: "YES" or "NO" - which side won

        Returns:
            SettlementResult with P/L details
        """
        if position.status != PositionStatus.OPEN:
            raise ValueError(f"Position {position.position_id} is not open")

        # Determine if we won
        won = position.side == winning_side
        settlement_price = 1.0 if won else 0.0

        # Calculate P/L
        # We paid entry_price per share
        # If we won, we receive 1.0 per share
        # If we lost, we receive 0.0 per share
        payout = settlement_price * position.shares
        cost = position.entry_price * position.shares
        realized_pnl = payout - cost

        result = SettlementResult(
            position_id=position.position_id,
            market_id=position.market_id,
            city=position.city,
            outcome_description=position.outcome_description,
            our_side=position.side,
            winning_side=winning_side,
            entry_price=position.entry_price,
            settlement_price=settlement_price,
            shares=position.shares,
            realized_pnl=realized_pnl,
            won=won
        )

        # Close the position
        if self.position_manager:
            self.position_manager.close_position(
                position_id=position.position_id,
                exit_price=settlement_price,
                reason=ExitReason.SETTLEMENT
            )

        # Record trade
        if self.pnl_tracker and self.position_manager:
            closed_position = self.position_manager.positions[position.position_id]
            self.pnl_tracker.record_trade(closed_position)

        self.settlement_history.append(result)

        logger.info(
            f"Settled {position.position_id}: "
            f"{'WON' if won else 'LOST'} "
            f"(${realized_pnl:+.2f})"
        )

        return result

    def settle_market(
        self,
        market_id: str,
        winning_outcome_id: str,
        outcome_side_map: dict[str, str]
    ) -> list[SettlementResult]:
        """
        Settle all positions in a market.

        Args:
            market_id: The market that settled
            winning_outcome_id: ID of the winning outcome
            outcome_side_map: Map of outcome_id -> "YES"/"NO" side that wins

        Returns:
            List of SettlementResults for all settled positions
        """
        if not self.position_manager:
            return []

        results = []
        positions = self.position_manager.get_positions_by_market(market_id)

        for position in positions:
            if position.status != PositionStatus.OPEN:
                continue

            # Determine winning side for this position's outcome
            # For temperature markets, typically the condition that matches
            # the actual temperature resolves YES
            winning_side = outcome_side_map.get(position.condition_id, "NO")

            result = self.settle_position(position, winning_side)
            results.append(result)

        return results

    def determine_winning_outcomes(
        self,
        actual_temperature: float,
        outcomes: list[dict]
    ) -> dict[str, str]:
        """
        Determine which outcomes won based on actual temperature.

        Args:
            actual_temperature: The actual temperature recorded
            outcomes: List of outcome dicts with 'condition_id', 'temp_min', 'temp_max'

        Returns:
            Dict of condition_id -> winning side ("YES" or "NO")
        """
        result = {}

        for outcome in outcomes:
            condition_id = outcome["condition_id"]
            temp_min = outcome.get("temp_min")
            temp_max = outcome.get("temp_max")

            # Check if actual temp falls in this range
            in_range = True
            if temp_min is not None and actual_temperature < temp_min:
                in_range = False
            if temp_max is not None and actual_temperature > temp_max:
                in_range = False

            # If actual temp is in this range, YES wins; otherwise NO wins
            result[condition_id] = "YES" if in_range else "NO"

        return result

    def get_pending_settlements(self) -> list[Position]:
        """
        Get positions that should be settled (market has closed).

        Returns positions where settlement_date has passed.
        """
        if not self.position_manager:
            return []

        now = datetime.now()
        pending = []

        for position in self.position_manager.get_open_positions():
            if position.settlement_date <= now:
                pending.append(position)

        return pending

    def get_settlement_summary(self) -> dict:
        """Get summary of settlement history."""
        if not self.settlement_history:
            return {
                "total_settlements": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            }

        wins = [s for s in self.settlement_history if s.won]
        losses = [s for s in self.settlement_history if not s.won]

        return {
            "total_settlements": len(self.settlement_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.settlement_history),
            "total_pnl": sum(s.realized_pnl for s in self.settlement_history),
            "average_win": sum(s.realized_pnl for s in wins) / len(wins) if wins else 0.0,
            "average_loss": sum(s.realized_pnl for s in losses) / len(losses) if losses else 0.0,
        }

    def get_result_summary(self, result: SettlementResult) -> dict:
        """Get display-friendly summary of a settlement result."""
        return {
            "position_id": result.position_id,
            "city": result.city,
            "outcome": result.outcome_description,
            "our_side": result.our_side,
            "winning_side": result.winning_side,
            "entry_price": round(result.entry_price, 4),
            "settlement_price": result.settlement_price,
            "shares": round(result.shares, 2),
            "pnl": round(result.realized_pnl, 2),
            "result": "WON" if result.won else "LOST",
            "settled_at": result.settled_at.strftime("%Y-%m-%d %H:%M"),
        }
