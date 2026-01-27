"""
Trading Engine Module

Professional-grade position management, risk controls, and execution.
"""

from .position_manager import (
    Position,
    PositionStatus,
    ExitReason,
    PositionManager,
)
from .risk_manager import (
    RiskManager,
    RiskCheck,
    RiskViolation,
)
from .execution_engine import (
    ExecutionEngine,
    AutoTrader,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
)
from .pnl_tracker import (
    PnLTracker,
    PerformanceMetrics,
    TradeRecord,
)
from .settlement import (
    SettlementHandler,
    SettlementResult,
    SettlementOutcome,
)
from .config import TradingConfig, default_config

__all__ = [
    # Position Manager
    "Position",
    "PositionStatus",
    "ExitReason",
    "PositionManager",
    # Risk Manager
    "RiskManager",
    "RiskCheck",
    "RiskViolation",
    # Execution Engine
    "ExecutionEngine",
    "AutoTrader",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    # P&L Tracker
    "PnLTracker",
    "PerformanceMetrics",
    "TradeRecord",
    # Settlement
    "SettlementHandler",
    "SettlementResult",
    "SettlementOutcome",
    # Config
    "TradingConfig",
    "default_config",
]
