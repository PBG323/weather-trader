"""
Logging Configuration

Uses loguru for structured, colorful logging with:
- Console output with colors
- File rotation
- JSON format for structured logging
"""

import sys
from pathlib import Path
from loguru import logger

from ..config import config


def setup_logging(
    log_dir: str = "logs",
    log_level: str = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure logging for the application.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation: Log file rotation size
        retention: How long to keep old logs
    """
    log_level = log_level or config.log_level

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # General log file
    logger.add(
        log_path / "weather_trader.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="gz",
    )

    # Trade-specific log file (JSON format for analysis)
    logger.add(
        log_path / "trades.json",
        level="INFO",
        format="{message}",
        filter=lambda record: record["extra"].get("trade_log", False),
        rotation="1 day",
        retention="90 days",
        serialize=True,
    )

    # Error log file
    logger.add(
        log_path / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        rotation=rotation,
        retention=retention,
        compression="gz",
    )

    logger.info(f"Logging initialized at level {log_level}")


def get_logger(name: str = "weather_trader"):
    """
    Get a named logger instance.

    Args:
        name: Logger name for context

    Returns:
        Logger instance bound to the name
    """
    return logger.bind(name=name)


class TradeLogger:
    """
    Specialized logger for trade events.

    Logs trades in a structured format suitable for analysis.
    """

    def __init__(self):
        self.logger = logger.bind(trade_log=True)

    def log_signal(
        self,
        city: str,
        threshold: float,
        edge: float,
        confidence: float,
        side: str,
        market_prob: float,
        forecast_prob: float,
    ):
        """Log a trade signal."""
        self.logger.info({
            "event": "signal",
            "city": city,
            "threshold": threshold,
            "edge": edge,
            "confidence": confidence,
            "side": side,
            "market_prob": market_prob,
            "forecast_prob": forecast_prob,
        })

    def log_execution(
        self,
        city: str,
        side: str,
        size: float,
        price: float,
        order_id: str,
        success: bool,
        error: str = None,
    ):
        """Log a trade execution."""
        self.logger.info({
            "event": "execution",
            "city": city,
            "side": side,
            "size": size,
            "price": price,
            "order_id": order_id,
            "success": success,
            "error": error,
        })

    def log_settlement(
        self,
        city: str,
        market_id: str,
        outcome: str,
        pnl: float,
    ):
        """Log a market settlement."""
        self.logger.info({
            "event": "settlement",
            "city": city,
            "market_id": market_id,
            "outcome": outcome,
            "pnl": pnl,
        })

    def log_forecast_accuracy(
        self,
        city: str,
        date: str,
        forecast_high: float,
        forecast_low: float,
        actual_high: float,
        actual_low: float,
    ):
        """Log forecast accuracy for tracking."""
        self.logger.info({
            "event": "forecast_accuracy",
            "city": city,
            "date": date,
            "forecast_high": forecast_high,
            "forecast_low": forecast_low,
            "actual_high": actual_high,
            "actual_low": actual_low,
            "high_error": abs(forecast_high - actual_high),
            "low_error": abs(forecast_low - actual_low),
        })


# Global trade logger instance
trade_logger = TradeLogger()
