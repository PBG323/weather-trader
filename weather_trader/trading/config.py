"""
Trading Configuration

All configurable parameters for the trading engine.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class TradingConfig:
    """
    Trading engine configuration parameters.

    These can be adjusted based on risk tolerance and market conditions.
    """

    # ========================
    # ENTRY PARAMETERS
    # ========================

    # Minimum edge required to enter a position
    min_edge_to_enter: float = field(
        default_factory=lambda: float(os.getenv("MIN_EDGE_TO_ENTER", "0.05"))
    )

    # Minimum model confidence to enter
    min_confidence_to_enter: float = field(
        default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.65"))
    )

    # Kelly fraction for position sizing (0.25 = quarter Kelly)
    kelly_fraction: float = field(
        default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25"))
    )

    # ========================
    # EXIT PARAMETERS - SMART PROFIT TAKING
    # ========================

    # Sell when this % of original edge has been captured
    # e.g., 0.75 = sell when 75% of edge opportunity is gone
    edge_exhaustion_threshold: float = field(
        default_factory=lambda: float(os.getenv("EDGE_EXHAUSTION_THRESHOLD", "0.75"))
    )

    # Minimum edge to continue holding (below this, consider exit)
    min_edge_to_hold: float = field(
        default_factory=lambda: float(os.getenv("MIN_EDGE_TO_HOLD", "0.02"))
    )

    # Number of price points to track for edge trend
    edge_history_length: int = field(
        default_factory=lambda: int(os.getenv("EDGE_HISTORY_LENGTH", "10"))
    )

    # Sell if edge trend is negative for this many consecutive checks
    negative_trend_threshold: int = field(
        default_factory=lambda: int(os.getenv("NEGATIVE_TREND_THRESHOLD", "3"))
    )

    # ========================
    # EXIT PARAMETERS - STOP LOSS
    # ========================

    # Maximum loss before forced exit (as % of position)
    stop_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("STOP_LOSS_PCT", "0.30"))
    )

    # Trailing stop: lock in gains after position is profitable
    # e.g., 0.50 = if up 20%, stop triggers if falls back to 10%
    trailing_stop_pct: float = field(
        default_factory=lambda: float(os.getenv("TRAILING_STOP_PCT", "0.50"))
    )

    # ========================
    # EXIT PARAMETERS - TIME BASED
    # ========================

    # Hours before settlement to consider closing profitable positions
    hours_before_settlement_warning: int = field(
        default_factory=lambda: int(os.getenv("HOURS_BEFORE_SETTLEMENT_WARNING", "6"))
    )

    # Hours before settlement to force close all positions
    hours_before_settlement_close: int = field(
        default_factory=lambda: int(os.getenv("HOURS_BEFORE_SETTLEMENT_CLOSE", "2"))
    )

    # ========================
    # PORTFOLIO RISK LIMITS
    # ========================

    # Maximum open positions at any time
    max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    )

    # Maximum position size as % of bankroll
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.05"))
    )

    # Maximum exposure to any single city as % of bankroll
    max_city_concentration: float = field(
        default_factory=lambda: float(os.getenv("MAX_CITY_CONCENTRATION", "0.20"))
    )

    # Maximum trades per day
    max_daily_trades: int = field(
        default_factory=lambda: int(os.getenv("MAX_DAILY_TRADES", "50"))
    )

    # Stop trading if daily loss exceeds this % of bankroll
    max_daily_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "0.10"))
    )

    # Maximum drawdown from peak equity before halting
    max_drawdown_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_PCT", "0.20"))
    )

    # ========================
    # EXECUTION PARAMETERS
    # ========================

    # Minimum time between trades on same market (seconds)
    min_trade_interval: int = field(
        default_factory=lambda: int(os.getenv("MIN_TRADE_INTERVAL", "300"))
    )

    # Slippage buffer for market orders (added to spread)
    slippage_buffer: float = field(
        default_factory=lambda: float(os.getenv("SLIPPAGE_BUFFER", "0.01"))
    )

    # Minimum position size in dollars
    min_position_size: float = field(
        default_factory=lambda: float(os.getenv("MIN_POSITION_SIZE", "1.0"))
    )

    def __post_init__(self):
        """Validate configuration values."""
        assert 0 < self.min_edge_to_enter < 1, "min_edge_to_enter must be between 0 and 1"
        assert 0 < self.kelly_fraction <= 1, "kelly_fraction must be between 0 and 1"
        assert 0 < self.stop_loss_pct < 1, "stop_loss_pct must be between 0 and 1"
        assert 0 < self.max_position_pct < 1, "max_position_pct must be between 0 and 1"
        assert self.max_open_positions > 0, "max_open_positions must be positive"


# Global default configuration
default_config = TradingConfig()
