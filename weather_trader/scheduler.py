"""
Task Scheduler

Schedules trading cycles at regular intervals:
- Hourly checks during active hours
- Increased frequency near market close
- Daily summary and reset
"""

import asyncio
from datetime import datetime, time, timedelta
from typing import Optional, Callable
import signal
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .main import WeatherTrader
from .monitoring import get_logger, AlertManager, Alert, AlertLevel

logger = get_logger("scheduler")


class TradingScheduler:
    """
    Manages scheduled trading tasks.
    """

    def __init__(
        self,
        trader: WeatherTrader,
        alerts: Optional[AlertManager] = None,
    ):
        """
        Initialize scheduler.

        Args:
            trader: WeatherTrader instance
            alerts: AlertManager for notifications
        """
        self.trader = trader
        self.alerts = alerts
        self.scheduler = AsyncIOScheduler()

        # Track state
        self._running = False
        self._cycles_today = 0
        self._last_cycle_time: Optional[datetime] = None

    def setup_jobs(self):
        """Configure scheduled jobs."""
        # Regular trading cycle every hour during market hours
        # Weather markets are most active during US daytime hours
        self.scheduler.add_job(
            self._run_trading_cycle,
            CronTrigger(hour="8-22", minute="0"),  # 8 AM - 10 PM every hour
            id="hourly_trading",
            name="Hourly Trading Cycle",
        )

        # More frequent checks near typical market settlement times
        # Most weather markets settle around 8-10 AM local time
        self.scheduler.add_job(
            self._run_trading_cycle,
            CronTrigger(hour="7-10", minute="*/15"),  # Every 15 min 7-10 AM
            id="settlement_window_trading",
            name="Settlement Window Trading",
        )

        # Daily summary at end of day
        self.scheduler.add_job(
            self._send_daily_summary,
            CronTrigger(hour="23", minute="0"),  # 11 PM
            id="daily_summary",
            name="Daily Summary",
        )

        # Reset daily tracking at midnight
        self.scheduler.add_job(
            self._reset_daily_tracking,
            CronTrigger(hour="0", minute="0"),
            id="daily_reset",
            name="Daily Reset",
        )

        # Health check every 5 minutes
        self.scheduler.add_job(
            self._health_check,
            IntervalTrigger(minutes=5),
            id="health_check",
            name="Health Check",
        )

        logger.info("Scheduled jobs configured")

    async def _run_trading_cycle(self):
        """Execute a trading cycle."""
        if not self._running:
            return

        try:
            logger.info("Running scheduled trading cycle")
            summary = await self.trader.run_cycle()
            self._cycles_today += 1
            self._last_cycle_time = datetime.now()

            logger.info(f"Cycle complete. Cycles today: {self._cycles_today}")

        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            if self.alerts:
                await self.alerts.api_error("TradingCycle", str(e))

    async def _send_daily_summary(self):
        """Send end-of-day summary."""
        try:
            if self.alerts:
                journal_summary = self.trader._executor.get_journal_summary()

                await self.alerts.daily_summary(
                    pnl=self.trader._position_sizer.daily_pnl,
                    trades=journal_summary.get("total_trades", 0),
                    win_rate=journal_summary.get("success_rate", 0),
                )

            logger.info("Daily summary sent")

        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

    async def _reset_daily_tracking(self):
        """Reset daily P&L and tracking."""
        try:
            self.trader._position_sizer.reset_daily_tracking()
            self._cycles_today = 0
            logger.info("Daily tracking reset")

        except Exception as e:
            logger.error(f"Failed to reset daily tracking: {e}")

    async def _health_check(self):
        """Perform health check."""
        try:
            # Check wallet balance
            if self.trader._poly_auth and self.trader._poly_auth.is_configured:
                balance = self.trader._poly_auth.get_usdc_balance()
                if balance < 10:
                    logger.warning(f"Low USDC balance: ${balance:.2f}")
                    if self.alerts:
                        await self.alerts.send(
                            Alert(
                                level=AlertLevel.WARNING,
                                title="Low Balance",
                                message=f"USDC balance is ${balance:.2f}",
                            )
                        )

            # Log health status
            status = {
                "running": self._running,
                "cycles_today": self._cycles_today,
                "last_cycle": self._last_cycle_time.isoformat() if self._last_cycle_time else None,
            }
            logger.debug(f"Health check: {status}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self.setup_jobs()
        self.scheduler.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        self.scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")

    async def run_forever(self):
        """Run the scheduler until interrupted."""
        self.start()

        # Set up signal handlers
        loop = asyncio.get_event_loop()

        def shutdown():
            logger.info("Shutdown signal received")
            self.stop()
            loop.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown)

        logger.info("Running scheduler... Press Ctrl+C to stop")

        # Keep running
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()


async def run_scheduler(dry_run: bool = True, bankroll: Optional[float] = None):
    """
    Main entry point for running the scheduler.

    Args:
        dry_run: If True, don't execute real trades
        bankroll: Starting bankroll
    """
    async with WeatherTrader(dry_run=dry_run, bankroll=bankroll) as trader:
        alerts = AlertManager()

        scheduler = TradingScheduler(trader, alerts)

        # Run initial cycle immediately
        logger.info("Running initial trading cycle...")
        await trader.run_cycle()

        # Start scheduled operation
        await scheduler.run_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather Trader Scheduler")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Run without executing real trades")
    parser.add_argument("--live", action="store_true",
                        help="Execute real trades")
    parser.add_argument("--bankroll", type=float, default=None,
                        help="Starting bankroll")

    args = parser.parse_args()

    dry_run = not args.live

    asyncio.run(run_scheduler(dry_run=dry_run, bankroll=args.bankroll))
