"""
Monitoring module for logging and alerts.

Provides:
- Structured logging with loguru
- Discord/Telegram alert notifications
- Performance tracking
"""

from .logger import setup_logging, get_logger
from .alerts import AlertManager, Alert, AlertLevel

__all__ = ["setup_logging", "get_logger", "AlertManager", "Alert", "AlertLevel"]
