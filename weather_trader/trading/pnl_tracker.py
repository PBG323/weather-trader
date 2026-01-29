"""
P&L Tracker

Tracks trading performance, maintains trade history, and calculates metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional
from collections import defaultdict
import json
import os

from .position_manager import Position, ExitReason


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    trade_id: str
    position_id: str
    market_id: str
    city: str
    outcome_description: str
    side: str                   # YES or NO

    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    shares: float

    entry_edge: float
    exit_reason: ExitReason

    realized_pnl: float
    realized_pnl_pct: float
    hold_duration_hours: float

    # Additional context
    forecast_prob_at_entry: float
    forecast_prob_at_exit: float


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics."""
    # Overall
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    average_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0          # gross profit / gross loss
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Trade characteristics
    average_hold_hours: float = 0.0
    average_entry_edge: float = 0.0

    # By exit reason
    exits_by_reason: dict = field(default_factory=dict)

    # By city
    pnl_by_city: dict = field(default_factory=dict)


class PnLTracker:
    """
    Tracks all trading activity and calculates performance metrics.

    Features:
    - Trade history logging
    - Real-time P&L tracking
    - Performance metric calculation
    - Persistence to disk
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.getcwd(), "data", "trades")
        os.makedirs(self.data_dir, exist_ok=True)

        self.trades: list[TradeRecord] = []
        self._trade_counter = 0

        # Real-time tracking
        self._equity_curve: list[tuple[datetime, float]] = []
        self._peak_equity = 0.0
        self._current_drawdown = 0.0

        # Load existing trades
        self._load_trades()

    def record_trade(self, position: Position) -> TradeRecord:
        """
        Record a completed trade from a closed position.

        Args:
            position: The closed position to record

        Returns:
            The created TradeRecord
        """
        if position.exit_time is None or position.exit_price is None:
            raise ValueError("Position must be closed to record trade")

        self._trade_counter += 1
        trade_id = f"TRD-{self._trade_counter:06d}"

        hold_duration = (position.exit_time - position.entry_time).total_seconds() / 3600

        trade = TradeRecord(
            trade_id=trade_id,
            position_id=position.position_id,
            market_id=position.market_id,
            city=position.city,
            outcome_description=position.outcome_description,
            side=position.side,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=position.exit_time,
            exit_price=position.exit_price,
            shares=position.shares,
            entry_edge=position.entry_edge,
            exit_reason=position.exit_reason or ExitReason.MANUAL,
            realized_pnl=position.realized_pnl or 0.0,
            realized_pnl_pct=self._calculate_pnl_pct(position),
            hold_duration_hours=hold_duration,
            forecast_prob_at_entry=position.entry_forecast_prob,
            forecast_prob_at_exit=position.current_forecast_prob
        )

        self.trades.append(trade)
        self._save_trade(trade)

        return trade

    def _calculate_pnl_pct(self, position) -> float:
        """Calculate P/L percentage correctly for both YES and NO positions.

        For YES: bought at entry_price, sold at exit_price
            Cost basis = entry_price, Return = (exit - entry) / entry
        For NO: bought at (1 - entry_price), sold at (1 - exit_price)
            Cost basis = (1 - entry_price), Return = ((1-exit) - (1-entry)) / (1-entry)
            Simplified: (entry - exit) / (1 - entry)
        """
        entry = position.entry_price
        exit_p = position.exit_price

        if position.side == "YES":
            if entry > 0:
                return (exit_p - entry) / entry
            return 0.0
        else:  # NO position
            cost_basis = 1 - entry  # What we paid per share
            if cost_basis > 0:
                exit_value = 1 - exit_p  # What we receive per share
                return (exit_value - cost_basis) / cost_basis
            return 0.0

    def update_equity(self, current_equity: float) -> None:
        """
        Update equity curve tracking.

        Call this periodically to track equity over time.
        """
        now = datetime.now()
        self._equity_curve.append((now, current_equity))

        # Update peak and drawdown
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._current_drawdown = 0.0
        else:
            self._current_drawdown = self._peak_equity - current_equity

    def get_metrics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        city: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a time period.

        Args:
            start_date: Start of period (default: all time)
            end_date: End of period (default: today)
            city: Filter to specific city (default: all cities)

        Returns:
            PerformanceMetrics with calculated values
        """
        # Filter trades
        filtered = self.trades

        if start_date:
            filtered = [t for t in filtered if t.entry_time.date() >= start_date]
        if end_date:
            filtered = [t for t in filtered if t.exit_time.date() <= end_date]
        if city:
            filtered = [t for t in filtered if t.city.lower() == city.lower()]

        if not filtered:
            return PerformanceMetrics()

        # Calculate metrics
        metrics = PerformanceMetrics()
        metrics.total_trades = len(filtered)

        wins = [t for t in filtered if t.realized_pnl > 0]
        losses = [t for t in filtered if t.realized_pnl < 0]

        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0.0

        # P&L calculations
        metrics.total_pnl = sum(t.realized_pnl for t in filtered)
        metrics.total_profit = sum(t.realized_pnl for t in wins)
        metrics.total_loss = abs(sum(t.realized_pnl for t in losses))

        metrics.average_pnl = metrics.total_pnl / metrics.total_trades
        metrics.average_win = metrics.total_profit / len(wins) if wins else 0.0
        metrics.average_loss = metrics.total_loss / len(losses) if losses else 0.0

        # Profit factor
        metrics.profit_factor = (
            metrics.total_profit / metrics.total_loss
            if metrics.total_loss > 0 else float('inf')
        )

        # Trade characteristics
        metrics.average_hold_hours = sum(t.hold_duration_hours for t in filtered) / len(filtered)
        metrics.average_entry_edge = sum(t.entry_edge for t in filtered) / len(filtered)

        # Exits by reason
        exits_by_reason = defaultdict(int)
        for trade in filtered:
            exits_by_reason[trade.exit_reason.value] += 1
        metrics.exits_by_reason = dict(exits_by_reason)

        # P&L by city
        pnl_by_city = defaultdict(float)
        for trade in filtered:
            pnl_by_city[trade.city] += trade.realized_pnl
        metrics.pnl_by_city = dict(pnl_by_city)

        # Calculate drawdown from equity curve
        if self._equity_curve:
            peak = 0.0
            max_dd = 0.0
            for _, equity in self._equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            metrics.max_drawdown_pct = max_dd

        return metrics

    def get_daily_pnl(self, days: int = 30) -> list[tuple[date, float]]:
        """
        Get daily P&L for the last N days.

        Returns list of (date, pnl) tuples.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        daily_pnl = defaultdict(float)

        for trade in self.trades:
            trade_date = trade.exit_time.date()
            if start_date <= trade_date <= end_date:
                daily_pnl[trade_date] += trade.realized_pnl

        # Fill in missing days with 0
        result = []
        current = start_date
        while current <= end_date:
            result.append((current, daily_pnl.get(current, 0.0)))
            current += timedelta(days=1)

        return result

    def get_trades_for_date(self, trade_date: date) -> list[TradeRecord]:
        """Get all trades closed on a specific date."""
        return [t for t in self.trades if t.exit_time.date() == trade_date]

    def get_recent_trades(self, limit: int = 20) -> list[TradeRecord]:
        """Get most recent trades."""
        return sorted(self.trades, key=lambda t: t.exit_time, reverse=True)[:limit]

    def get_trade_summary(self, trade: TradeRecord) -> dict:
        """Get summary dict of a trade for display."""
        return {
            "trade_id": trade.trade_id,
            "city": trade.city,
            "outcome": trade.outcome_description,
            "side": trade.side,
            "entry_time": trade.entry_time.strftime("%Y-%m-%d %H:%M"),
            "exit_time": trade.exit_time.strftime("%Y-%m-%d %H:%M"),
            "entry_price": round(trade.entry_price, 4),
            "exit_price": round(trade.exit_price, 4),
            "shares": round(trade.shares, 2),
            "pnl": round(trade.realized_pnl, 2),
            "pnl_pct": round(trade.realized_pnl_pct * 100, 1),
            "hold_hours": round(trade.hold_duration_hours, 1),
            "exit_reason": trade.exit_reason.value,
            "entry_edge": round(trade.entry_edge * 100, 1),
        }

    def _save_trade(self, trade: TradeRecord) -> None:
        """Save a trade to disk."""
        trade_dict = {
            "trade_id": trade.trade_id,
            "position_id": trade.position_id,
            "market_id": trade.market_id,
            "city": trade.city,
            "outcome_description": trade.outcome_description,
            "side": trade.side,
            "entry_time": trade.entry_time.isoformat(),
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time.isoformat(),
            "exit_price": trade.exit_price,
            "shares": trade.shares,
            "entry_edge": trade.entry_edge,
            "exit_reason": trade.exit_reason.value,
            "realized_pnl": trade.realized_pnl,
            "realized_pnl_pct": trade.realized_pnl_pct,
            "hold_duration_hours": trade.hold_duration_hours,
            "forecast_prob_at_entry": trade.forecast_prob_at_entry,
            "forecast_prob_at_exit": trade.forecast_prob_at_exit,
        }

        # Append to daily file
        file_date = trade.exit_time.strftime("%Y-%m-%d")
        filepath = os.path.join(self.data_dir, f"trades_{file_date}.jsonl")

        with open(filepath, "a") as f:
            f.write(json.dumps(trade_dict) + "\n")

    def _load_trades(self) -> None:
        """Load existing trades from disk."""
        if not os.path.exists(self.data_dir):
            return

        for filename in os.listdir(self.data_dir):
            if filename.startswith("trades_") and filename.endswith(".jsonl"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            trade = TradeRecord(
                                trade_id=data["trade_id"],
                                position_id=data["position_id"],
                                market_id=data["market_id"],
                                city=data["city"],
                                outcome_description=data["outcome_description"],
                                side=data["side"],
                                entry_time=datetime.fromisoformat(data["entry_time"]),
                                entry_price=data["entry_price"],
                                exit_time=datetime.fromisoformat(data["exit_time"]),
                                exit_price=data["exit_price"],
                                shares=data["shares"],
                                entry_edge=data["entry_edge"],
                                exit_reason=ExitReason(data["exit_reason"]),
                                realized_pnl=data["realized_pnl"],
                                realized_pnl_pct=data["realized_pnl_pct"],
                                hold_duration_hours=data["hold_duration_hours"],
                                forecast_prob_at_entry=data["forecast_prob_at_entry"],
                                forecast_prob_at_exit=data["forecast_prob_at_exit"],
                            )
                            self.trades.append(trade)
                            self._trade_counter = max(
                                self._trade_counter,
                                int(trade.trade_id.split("-")[1])
                            )
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error loading trade from {filepath}: {e}")
                            continue

    def export_trades_csv(self, filepath: str) -> None:
        """Export all trades to CSV format."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id", "city", "outcome", "side",
                "entry_time", "exit_time",
                "entry_price", "exit_price", "shares",
                "pnl", "pnl_pct", "hold_hours",
                "entry_edge", "exit_reason"
            ])

            for trade in sorted(self.trades, key=lambda t: t.entry_time):
                writer.writerow([
                    trade.trade_id,
                    trade.city,
                    trade.outcome_description,
                    trade.side,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    round(trade.entry_price, 4),
                    round(trade.exit_price, 4),
                    round(trade.shares, 2),
                    round(trade.realized_pnl, 2),
                    round(trade.realized_pnl_pct * 100, 2),
                    round(trade.hold_duration_hours, 2),
                    round(trade.entry_edge * 100, 2),
                    trade.exit_reason.value
                ])
