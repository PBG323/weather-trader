"""
Alert System

Sends notifications for important events via:
- Discord webhooks
- Telegram bots

Alert levels:
- INFO: Trade executions, daily summaries
- WARNING: Low confidence trades, API issues
- ERROR: Execution failures, system errors
- CRITICAL: Complete system failures, loss limits hit
"""

import httpx
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

from ..config import config


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert to be sent."""
    level: AlertLevel
    title: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_discord_embed(self) -> dict:
        """Format as Discord embed."""
        colors = {
            AlertLevel.INFO: 0x3498db,      # Blue
            AlertLevel.WARNING: 0xf39c12,   # Orange
            AlertLevel.ERROR: 0xe74c3c,     # Red
            AlertLevel.CRITICAL: 0x9b59b6,  # Purple
        }

        embed = {
            "title": f"{self._level_emoji()} {self.title}",
            "description": self.message,
            "color": colors.get(self.level, 0x95a5a6),
            "timestamp": self.timestamp.isoformat(),
            "footer": {"text": "Weather Trader"},
        }

        if self.details:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in self.details.items()
            ]

        return embed

    def to_telegram_message(self) -> str:
        """Format as Telegram message."""
        emoji = self._level_emoji()
        msg = f"{emoji} *{self.title}*\n\n{self.message}"

        if self.details:
            msg += "\n\n*Details:*"
            for k, v in self.details.items():
                msg += f"\n• {k}: `{v}`"

        msg += f"\n\n_{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"
        return msg

    def _level_emoji(self) -> str:
        """Get emoji for alert level."""
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        return emojis.get(self.level, "📢")


class AlertManager:
    """
    Manages sending alerts to configured channels.
    """

    def __init__(
        self,
        discord_webhook: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ):
        """
        Initialize alert manager.

        Args:
            discord_webhook: Discord webhook URL
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID to send to
        """
        self.discord_webhook = discord_webhook or config.alerts.discord_webhook_url
        self.telegram_token = telegram_token or config.alerts.telegram_bot_token
        self.telegram_chat_id = telegram_chat_id or config.alerts.telegram_chat_id

        self.client = httpx.AsyncClient(timeout=10.0)

        # Track sent alerts to avoid duplicates
        self._recent_alerts: list[str] = []
        self._max_recent = 100

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def is_configured(self) -> bool:
        """Check if any alert channel is configured."""
        return bool(self.discord_webhook) or bool(self.telegram_token and self.telegram_chat_id)

    async def send(self, alert: Alert, dedupe: bool = True) -> bool:
        """
        Send an alert to all configured channels.

        Args:
            alert: Alert to send
            dedupe: Skip if similar alert sent recently

        Returns:
            True if alert was sent successfully to at least one channel
        """
        # Deduplication
        alert_key = f"{alert.level.value}:{alert.title}"
        if dedupe and alert_key in self._recent_alerts:
            return False

        self._recent_alerts.append(alert_key)
        if len(self._recent_alerts) > self._max_recent:
            self._recent_alerts.pop(0)

        success = False

        # Send to Discord
        if self.discord_webhook:
            try:
                discord_success = await self._send_discord(alert)
                success = success or discord_success
            except Exception as e:
                print(f"Discord alert failed: {e}")

        # Send to Telegram
        if self.telegram_token and self.telegram_chat_id:
            try:
                telegram_success = await self._send_telegram(alert)
                success = success or telegram_success
            except Exception as e:
                print(f"Telegram alert failed: {e}")

        return success

    async def _send_discord(self, alert: Alert) -> bool:
        """Send alert to Discord webhook."""
        payload = {
            "embeds": [alert.to_discord_embed()]
        }

        response = await self.client.post(
            self.discord_webhook,
            json=payload,
        )

        return response.status_code in [200, 204]

    async def _send_telegram(self, alert: Alert) -> bool:
        """Send alert to Telegram."""
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"

        payload = {
            "chat_id": self.telegram_chat_id,
            "text": alert.to_telegram_message(),
            "parse_mode": "Markdown",
        }

        response = await self.client.post(url, json=payload)

        return response.status_code == 200

    # Convenience methods for common alerts

    async def trade_executed(
        self,
        city: str,
        side: str,
        size: float,
        price: float,
        edge: float,
    ):
        """Send alert for executed trade."""
        alert = Alert(
            level=AlertLevel.INFO,
            title=f"Trade Executed: {city}",
            message=f"Bought {side} at ${price:.3f}",
            details={
                "Size": f"${size:.2f}",
                "Edge": f"{edge*100:.1f}%",
                "Side": side,
            },
        )
        await self.send(alert)

    async def daily_summary(
        self,
        pnl: float,
        trades: int,
        win_rate: float,
    ):
        """Send daily summary alert."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING

        alert = Alert(
            level=level,
            title="Daily Summary",
            message=f"P&L: ${pnl:+.2f}",
            details={
                "Trades": trades,
                "Win Rate": f"{win_rate*100:.1f}%",
            },
        )
        await self.send(alert, dedupe=False)

    async def loss_limit_hit(self, daily_loss: float, limit: float):
        """Send alert when loss limit is reached."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="Loss Limit Reached",
            message="Trading halted for the day",
            details={
                "Daily Loss": f"${daily_loss:.2f}",
                "Limit": f"${limit:.2f}",
            },
        )
        await self.send(alert, dedupe=False)

    async def api_error(self, api_name: str, error: str):
        """Send alert for API errors."""
        alert = Alert(
            level=AlertLevel.ERROR,
            title=f"API Error: {api_name}",
            message=error,
        )
        await self.send(alert)

    async def forecast_ready(self, city: str, high: float, low: float, confidence: float):
        """Send alert when forecast is ready."""
        alert = Alert(
            level=AlertLevel.INFO,
            title=f"Forecast Ready: {city}",
            message=f"High: {high:.0f}°F, Low: {low:.0f}°F",
            details={
                "Confidence": f"{confidence*100:.0f}%",
            },
        )
        await self.send(alert)
