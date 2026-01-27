"""
Risk Manager

Portfolio-level risk controls and position sizing.
Enforces trading limits to prevent catastrophic losses.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional

from .config import TradingConfig, default_config
from .position_manager import PositionManager, Position


class RiskCheck(Enum):
    """Types of risk checks."""
    MAX_POSITIONS = "max_positions"
    MAX_POSITION_SIZE = "max_position_size"
    MAX_CITY_CONCENTRATION = "max_city_concentration"
    DAILY_TRADE_LIMIT = "daily_trade_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    MIN_EDGE = "min_edge"
    MIN_CONFIDENCE = "min_confidence"
    TRADE_INTERVAL = "trade_interval"


@dataclass
class RiskViolation:
    """Details of a risk limit violation."""
    check: RiskCheck
    message: str
    current_value: float
    limit_value: float
    severity: str = "warning"  # "warning" or "critical"


@dataclass
class DailyStats:
    """Daily trading statistics for risk tracking."""
    date: date
    trades_count: int = 0
    realized_pnl: float = 0.0
    starting_equity: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0

    @property
    def daily_pnl(self) -> float:
        """Today's total P/L."""
        return self.current_equity - self.starting_equity

    @property
    def daily_pnl_pct(self) -> float:
        """Today's P/L as percentage."""
        if self.starting_equity == 0:
            return 0.0
        return self.daily_pnl / self.starting_equity

    @property
    def drawdown_from_peak(self) -> float:
        """Current drawdown from today's peak."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity


class RiskManager:
    """
    Enforces portfolio-level risk limits.

    Checks performed before any trade:
    - Position count limits
    - Position size limits
    - City concentration limits
    - Daily trade count limits
    - Daily loss limits
    - Maximum drawdown
    - Minimum edge requirements
    - Trade interval (cooldown)
    """

    def __init__(
        self,
        config: TradingConfig = None,
        position_manager: PositionManager = None,
        initial_bankroll: float = 10000.0
    ):
        self.config = config or default_config
        self.position_manager = position_manager
        self.bankroll = initial_bankroll
        self.peak_equity = initial_bankroll

        # Daily tracking
        self._daily_stats = DailyStats(
            date=date.today(),
            starting_equity=initial_bankroll,
            peak_equity=initial_bankroll,
            current_equity=initial_bankroll
        )

        # Trade timing
        self._last_trade_time: dict[str, datetime] = {}  # market_id -> last trade time

        # Track if trading is halted
        self._trading_halted = False
        self._halt_reason: Optional[str] = None

    def _ensure_daily_stats_current(self) -> None:
        """Reset daily stats if it's a new day."""
        today = date.today()
        if self._daily_stats.date != today:
            # New day - reset stats
            self._daily_stats = DailyStats(
                date=today,
                starting_equity=self.get_current_equity(),
                peak_equity=self.get_current_equity(),
                current_equity=self.get_current_equity()
            )
            # Also reset halt if it was due to daily limits
            if self._halt_reason in ["daily_loss_limit", "daily_trade_limit"]:
                self._trading_halted = False
                self._halt_reason = None

    def get_current_equity(self) -> float:
        """Get current total equity (bankroll + unrealized P/L)."""
        unrealized = 0.0
        if self.position_manager:
            unrealized = self.position_manager.get_total_unrealized_pnl()
        return self.bankroll + unrealized

    def update_equity(self) -> None:
        """Update equity tracking after market prices change."""
        self._ensure_daily_stats_current()
        current = self.get_current_equity()
        self._daily_stats.current_equity = current

        # Update peaks
        if current > self._daily_stats.peak_equity:
            self._daily_stats.peak_equity = current
        if current > self.peak_equity:
            self.peak_equity = current

    def record_trade(self, market_id: str, realized_pnl: float = 0.0) -> None:
        """Record a completed trade for risk tracking."""
        self._ensure_daily_stats_current()
        self._daily_stats.trades_count += 1
        self._daily_stats.realized_pnl += realized_pnl
        self._last_trade_time[market_id] = datetime.now()
        self.bankroll += realized_pnl
        self.update_equity()

    def check_can_trade(
        self,
        market_id: str,
        city: str,
        position_size: float,
        edge: float,
        confidence: float
    ) -> tuple[bool, list[RiskViolation]]:
        """
        Comprehensive check if a new trade is allowed.

        Args:
            market_id: Market to trade
            city: City for the market
            position_size: Proposed position size in dollars
            edge: Expected edge (forecast prob - market price)
            confidence: Model confidence level

        Returns:
            (can_trade, list of violations)
        """
        self._ensure_daily_stats_current()
        violations: list[RiskViolation] = []

        # 0. Check if trading is halted
        if self._trading_halted:
            violations.append(RiskViolation(
                check=RiskCheck.DAILY_LOSS_LIMIT,
                message=f"Trading halted: {self._halt_reason}",
                current_value=0,
                limit_value=0,
                severity="critical"
            ))
            return False, violations

        # 1. Check position count
        if self.position_manager:
            open_count = len(self.position_manager.get_open_positions())
            if open_count >= self.config.max_open_positions:
                violations.append(RiskViolation(
                    check=RiskCheck.MAX_POSITIONS,
                    message=f"Max open positions reached ({open_count}/{self.config.max_open_positions})",
                    current_value=open_count,
                    limit_value=self.config.max_open_positions,
                    severity="critical"
                ))

        # 2. Check position size
        max_size = self.bankroll * self.config.max_position_pct
        if position_size > max_size:
            violations.append(RiskViolation(
                check=RiskCheck.MAX_POSITION_SIZE,
                message=f"Position size ${position_size:.2f} exceeds max ${max_size:.2f}",
                current_value=position_size,
                limit_value=max_size,
                severity="critical"
            ))

        # 3. Check city concentration
        if self.position_manager:
            city_exposure = self.position_manager.get_city_exposure(city)
            max_city_exposure = self.bankroll * self.config.max_city_concentration
            new_city_exposure = city_exposure + position_size
            if new_city_exposure > max_city_exposure:
                violations.append(RiskViolation(
                    check=RiskCheck.MAX_CITY_CONCENTRATION,
                    message=f"{city} exposure ${new_city_exposure:.2f} exceeds max ${max_city_exposure:.2f}",
                    current_value=new_city_exposure,
                    limit_value=max_city_exposure,
                    severity="critical"
                ))

        # 4. Check daily trade count
        if self._daily_stats.trades_count >= self.config.max_daily_trades:
            violations.append(RiskViolation(
                check=RiskCheck.DAILY_TRADE_LIMIT,
                message=f"Daily trade limit reached ({self._daily_stats.trades_count}/{self.config.max_daily_trades})",
                current_value=self._daily_stats.trades_count,
                limit_value=self.config.max_daily_trades,
                severity="critical"
            ))

        # 5. Check daily loss limit
        daily_loss_pct = abs(min(0, self._daily_stats.daily_pnl_pct))
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            self._trading_halted = True
            self._halt_reason = "daily_loss_limit"
            violations.append(RiskViolation(
                check=RiskCheck.DAILY_LOSS_LIMIT,
                message=f"Daily loss limit hit ({daily_loss_pct*100:.1f}% >= {self.config.max_daily_loss_pct*100:.1f}%)",
                current_value=daily_loss_pct,
                limit_value=self.config.max_daily_loss_pct,
                severity="critical"
            ))

        # 6. Check max drawdown from peak
        drawdown = (self.peak_equity - self.get_current_equity()) / self.peak_equity
        if drawdown >= self.config.max_drawdown_pct:
            self._trading_halted = True
            self._halt_reason = "max_drawdown"
            violations.append(RiskViolation(
                check=RiskCheck.MAX_DRAWDOWN,
                message=f"Max drawdown hit ({drawdown*100:.1f}% >= {self.config.max_drawdown_pct*100:.1f}%)",
                current_value=drawdown,
                limit_value=self.config.max_drawdown_pct,
                severity="critical"
            ))

        # 7. Check minimum edge
        if edge < self.config.min_edge_to_enter:
            violations.append(RiskViolation(
                check=RiskCheck.MIN_EDGE,
                message=f"Edge {edge*100:.1f}% below minimum {self.config.min_edge_to_enter*100:.1f}%",
                current_value=edge,
                limit_value=self.config.min_edge_to_enter,
                severity="warning"
            ))

        # 8. Check minimum confidence
        if confidence < self.config.min_confidence_to_enter:
            violations.append(RiskViolation(
                check=RiskCheck.MIN_CONFIDENCE,
                message=f"Confidence {confidence*100:.1f}% below minimum {self.config.min_confidence_to_enter*100:.1f}%",
                current_value=confidence,
                limit_value=self.config.min_confidence_to_enter,
                severity="warning"
            ))

        # 9. Check trade interval (cooldown)
        if market_id in self._last_trade_time:
            elapsed = (datetime.now() - self._last_trade_time[market_id]).total_seconds()
            if elapsed < self.config.min_trade_interval:
                violations.append(RiskViolation(
                    check=RiskCheck.TRADE_INTERVAL,
                    message=f"Trade cooldown: {self.config.min_trade_interval - elapsed:.0f}s remaining",
                    current_value=elapsed,
                    limit_value=self.config.min_trade_interval,
                    severity="warning"
                ))

        # Determine if trade is allowed
        critical_violations = [v for v in violations if v.severity == "critical"]
        can_trade = len(critical_violations) == 0

        return can_trade, violations

    def calculate_position_size(
        self,
        edge: float,
        win_probability: float,
        price: float
    ) -> float:
        """
        Calculate optimal position size using fractional Kelly criterion.

        Kelly formula: f* = (bp - q) / b
        where:
            b = odds received (win/lose ratio)
            p = probability of winning
            q = probability of losing (1 - p)

        For binary markets at price p, if we buy YES:
            b = (1 - price) / price  (we risk 'price', win '1 - price')

        Args:
            edge: Expected edge (our prob - market price)
            win_probability: Our model's probability of YES
            price: Current market price

        Returns:
            Recommended position size in dollars
        """
        if edge <= 0 or win_probability <= 0 or price <= 0 or price >= 1:
            return 0.0

        # For YES position: odds = (1 - price) / price
        odds = (1 - price) / price
        q = 1 - win_probability

        # Kelly fraction
        kelly = (odds * win_probability - q) / odds

        # Apply fractional Kelly and bankroll
        if kelly <= 0:
            return 0.0

        position_size = self.bankroll * kelly * self.config.kelly_fraction

        # Apply position size limits
        max_size = self.bankroll * self.config.max_position_pct
        position_size = min(position_size, max_size)

        # Apply minimum position size
        if position_size < self.config.min_position_size:
            return 0.0

        return position_size

    def get_risk_summary(self) -> dict:
        """Get summary of current risk state."""
        self._ensure_daily_stats_current()

        open_positions = 0
        total_exposure = 0.0
        if self.position_manager:
            open_positions = len(self.position_manager.get_open_positions())
            total_exposure = self.position_manager.get_total_exposure()

        return {
            "bankroll": round(self.bankroll, 2),
            "current_equity": round(self.get_current_equity(), 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round((self.peak_equity - self.get_current_equity()) / self.peak_equity * 100, 2),
            "open_positions": open_positions,
            "max_positions": self.config.max_open_positions,
            "total_exposure": round(total_exposure, 2),
            "daily_trades": self._daily_stats.trades_count,
            "max_daily_trades": self.config.max_daily_trades,
            "daily_pnl": round(self._daily_stats.daily_pnl, 2),
            "daily_pnl_pct": round(self._daily_stats.daily_pnl_pct * 100, 2),
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
        }

    def is_trading_allowed(self) -> bool:
        """Quick check if trading is currently allowed."""
        self._ensure_daily_stats_current()
        return not self._trading_halted

    def reset_halt(self) -> None:
        """Manually reset trading halt (use with caution)."""
        self._trading_halted = False
        self._halt_reason = None
