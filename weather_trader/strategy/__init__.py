"""
Trading strategy components.

Includes:
- Expected value calculation
- Position sizing (Kelly criterion)
- Trade execution logic
- Advanced strategies (laddering, tail hunting)
- Forecast change alerts
"""

from .expected_value import ExpectedValueCalculator, TradeSignal
from .position_sizing import PositionSizer, PositionRecommendation
from .executor import TradeExecutor, ExecutionResult
from .advanced_strategies import (
    LadderStrategy,
    LadderPosition,
    TailOpportunity,
    StrategyType,
    generate_temperature_ladder,
    find_tail_opportunities,
    calculate_bankroll_allocation,
    score_model_consensus,
)
from .forecast_alerts import (
    ForecastAlert,
    ForecastAlertMonitor,
    AlertType,
    AlertPriority,
    check_edge_window,
)

__all__ = [
    # Core
    "ExpectedValueCalculator",
    "TradeSignal",
    "PositionSizer",
    "PositionRecommendation",
    "TradeExecutor",
    "ExecutionResult",
    # Advanced strategies
    "LadderStrategy",
    "LadderPosition",
    "TailOpportunity",
    "StrategyType",
    "generate_temperature_ladder",
    "find_tail_opportunities",
    "calculate_bankroll_allocation",
    "score_model_consensus",
    # Forecast alerts
    "ForecastAlert",
    "ForecastAlertMonitor",
    "AlertType",
    "AlertPriority",
    "check_edge_window",
]
