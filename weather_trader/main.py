"""
Main Trading Loop

Orchestrates the complete trading workflow:
1. Fetch forecasts from multiple APIs
2. Build ensemble predictions
3. Discover weather markets
4. Calculate expected value
5. Size positions
6. Execute trades
7. Monitor and report
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .config import config, get_city_config, get_all_cities, CityConfig

# Kalshi markets operate in Eastern time
EST = ZoneInfo("America/New_York")
from .apis import OpenMeteoClient, TomorrowIOClient, NWSClient
from .apis.open_meteo import WeatherModel
from .models import BiasCorrector, EnsembleForecaster, EnsembleForecast
from .models.ensemble import ModelForecast
from .kalshi import KalshiAuth, KalshiClient, KalshiMarketFinder
from .strategy import ExpectedValueCalculator, PositionSizer, TradeExecutor
from .monitoring import setup_logging, get_logger, AlertManager


logger = get_logger("main")


class WeatherTrader:
    """
    Main trading system that coordinates all components.
    """

    def __init__(
        self,
        dry_run: bool = True,
        bankroll: Optional[float] = None,
    ):
        """
        Initialize the trading system.

        Args:
            dry_run: If True, don't execute real trades
            bankroll: Starting bankroll (fetched from wallet if None)
        """
        self.dry_run = dry_run
        self._bankroll = bankroll

        # Components (initialized lazily)
        self._open_meteo: Optional[OpenMeteoClient] = None
        self._tomorrow_io: Optional[TomorrowIOClient] = None
        self._nws: Optional[NWSClient] = None
        self._bias_corrector: Optional[BiasCorrector] = None
        self._ensemble: Optional[EnsembleForecaster] = None
        self._market_finder: Optional[KalshiMarketFinder] = None
        self._kalshi_auth: Optional[KalshiAuth] = None
        self._kalshi_client: Optional[KalshiClient] = None
        self._ev_calculator: Optional[ExpectedValueCalculator] = None
        self._position_sizer: Optional[PositionSizer] = None
        self._executor: Optional[TradeExecutor] = None
        self._alerts: Optional[AlertManager] = None

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Weather Trader...")

        # Setup logging
        setup_logging()

        # Initialize API clients
        self._open_meteo = OpenMeteoClient()
        self._nws = NWSClient()

        # Tomorrow.io requires API key
        if config.api.tomorrow_io_api_key:
            try:
                self._tomorrow_io = TomorrowIOClient()
            except ValueError:
                logger.warning("Tomorrow.io API key not configured, skipping")

        # Initialize models
        self._bias_corrector = BiasCorrector()
        self._ensemble = EnsembleForecaster(self._bias_corrector)

        # Initialize Kalshi
        self._kalshi_auth = KalshiAuth()
        self._kalshi_client = KalshiClient(self._kalshi_auth)
        self._market_finder = KalshiMarketFinder()

        # Get bankroll
        if self._bankroll is None:
            if self._kalshi_auth.is_configured:
                async with self._kalshi_client:
                    self._bankroll = await self._kalshi_client.get_balance()
            else:
                self._bankroll = 1000.0  # Default for dry run
                logger.warning(f"Kalshi credentials not configured, using default bankroll: ${self._bankroll}")

        # Initialize strategy components
        self._ev_calculator = ExpectedValueCalculator()
        self._position_sizer = PositionSizer(self._bankroll)
        self._executor = TradeExecutor(
            self._kalshi_client,
            dry_run=self.dry_run,
        )

        # Initialize alerts
        self._alerts = AlertManager()

        logger.info(f"Initialized with bankroll: ${self._bankroll:.2f}")
        logger.info(f"Dry run mode: {self.dry_run}")

    async def close(self):
        """Clean up resources."""
        if self._open_meteo:
            await self._open_meteo.close()
        if self._tomorrow_io:
            await self._tomorrow_io.close()
        if self._nws:
            await self._nws.close()
        if self._kalshi_client and self._kalshi_client._client:
            await self._kalshi_client.__aexit__(None, None, None)
        if self._market_finder and self._market_finder._client:
            await self._market_finder.__aexit__(None, None, None)
        if self._alerts:
            await self._alerts.close()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def fetch_forecasts(
        self,
        target_date: Optional[date] = None
    ) -> dict[str, EnsembleForecast]:
        """
        Fetch forecasts for all cities.

        Args:
            target_date: Date to forecast (default: tomorrow)

        Returns:
            Dictionary mapping city keys to EnsembleForecast objects
        """
        if target_date is None:
            # Use EST since Kalshi markets operate in Eastern time
            today_est = datetime.now(EST).date()
            target_date = today_est + timedelta(days=1)

        logger.info(f"Fetching forecasts for {target_date}")

        forecasts = {}

        for city_key in get_all_cities():
            city_config = get_city_config(city_key)

            try:
                ensemble = await self._fetch_city_forecast(city_config, target_date)
                forecasts[city_key] = ensemble
                logger.info(
                    f"{city_config.name}: High {ensemble.high_mean:.1f}°F "
                    f"(±{ensemble.high_std:.1f}), Confidence {ensemble.confidence:.0%}"
                )
            except Exception as e:
                logger.error(f"Failed to fetch forecast for {city_config.name}: {e}")

        return forecasts

    async def _fetch_city_forecast(
        self,
        city_config: CityConfig,
        target_date: date
    ) -> EnsembleForecast:
        """Fetch and combine forecasts for a single city."""
        model_forecasts = []

        # Fetch from Open-Meteo (multiple models)
        try:
            om_forecasts = await self._open_meteo.get_ensemble_forecast(city_config)
            for model_name, forecast_list in om_forecasts.items():
                for f in forecast_list:
                    if f.timestamp.date() == target_date:
                        model_forecasts.append(ModelForecast(
                            model_name=model_name,
                            forecast_high=f.temperature_high,
                            forecast_low=f.temperature_low,
                        ))
                        break
        except Exception as e:
            logger.warning(f"Open-Meteo fetch failed for {city_config.name}: {e}")

        # Fetch from Tomorrow.io if available
        if self._tomorrow_io:
            try:
                tomorrow_data = await self._tomorrow_io.get_forecast_with_confidence(
                    city_config, datetime.combine(target_date, datetime.min.time())
                )
                model_forecasts.append(ModelForecast(
                    model_name="tomorrow",
                    forecast_high=tomorrow_data["high_mean"],
                    forecast_low=tomorrow_data["low_mean"],
                    weight=tomorrow_data["confidence"],
                ))
            except Exception as e:
                logger.warning(f"Tomorrow.io fetch failed for {city_config.name}: {e}")

        if not model_forecasts:
            raise ValueError(f"No forecasts available for {city_config.name}")

        # Build ensemble
        return self._ensemble.create_ensemble(
            city_config,
            model_forecasts,
            target_date,
            apply_bias_correction=True,
        )

    async def find_markets(
        self,
        target_date: Optional[date] = None
    ) -> list:
        """
        Find active weather markets.

        Args:
            target_date: Filter markets for this date

        Returns:
            List of WeatherMarket objects
        """
        logger.info("Discovering weather markets...")

        markets = await self._market_finder.find_weather_markets(active_only=True)

        # Filter by date if specified
        if target_date:
            markets = [m for m in markets if m.target_date == target_date]

        logger.info(f"Found {len(markets)} active weather markets")
        return markets

    async def analyze_opportunities(
        self,
        forecasts: dict[str, EnsembleForecast],
        markets: list
    ) -> list:
        """
        Analyze markets for trading opportunities.

        Args:
            forecasts: Dictionary of city forecasts
            markets: List of active markets

        Returns:
            List of TradeSignal objects
        """
        signals = self._ev_calculator.analyze_markets(markets, forecasts)
        tradeable = self._ev_calculator.get_tradeable_signals(signals)

        summary = self._ev_calculator.summarize_opportunities(signals)
        logger.info(
            f"Analysis complete: {summary['tradeable_markets']}/{summary['total_markets']} "
            f"tradeable, avg edge {summary['avg_edge']:.1%}"
        )

        return tradeable

    async def execute_trades(self, signals: list) -> list:
        """
        Execute trades for given signals.

        Args:
            signals: List of tradeable TradeSignal objects

        Returns:
            List of ExecutionResult objects
        """
        # Check if we can trade
        can_trade, reason = self._position_sizer.can_trade()
        if not can_trade:
            logger.warning(f"Cannot trade: {reason}")
            if self._alerts:
                await self._alerts.loss_limit_hit(
                    -self._position_sizer.daily_pnl,
                    self._position_sizer.bankroll * (config.trading.daily_loss_limit_percent / 100)
                )
            return []

        # Size positions
        recommendations = self._position_sizer.recommend_batch(signals)

        # Filter to tradeable sizes
        recommendations = [r for r in recommendations if r.should_trade]

        if not recommendations:
            logger.info("No positions meet minimum size requirements")
            return []

        logger.info(f"Executing {len(recommendations)} trades...")

        # Execute
        results = await self._executor.execute_batch(recommendations)

        # Log and alert
        for result in results:
            if result.success:
                logger.info(
                    f"Executed: {result.recommendation.signal.market.city} "
                    f"{result.recommendation.signal.side} "
                    f"${result.executed_size:.2f} @ {result.executed_price:.3f}"
                )
                if self._alerts:
                    await self._alerts.trade_executed(
                        result.recommendation.signal.market.city,
                        result.recommendation.signal.side,
                        result.executed_size,
                        result.executed_price,
                        result.recommendation.signal.edge,
                    )
            else:
                logger.error(f"Trade failed: {result.error_message}")

        return results

    async def run_cycle(self) -> dict:
        """
        Run a complete trading cycle.

        Returns:
            Summary of the cycle
        """
        logger.info("=" * 50)
        logger.info("Starting trading cycle")
        logger.info("=" * 50)

        # Use EST since Kalshi markets operate in Eastern time
        today_est = datetime.now(EST).date()
        target_date = today_est + timedelta(days=1)

        # 1. Fetch forecasts
        forecasts = await self.fetch_forecasts(target_date)

        # 2. Find markets
        markets = await self.find_markets(target_date)

        if not markets:
            logger.info("No active markets found for tomorrow")
            return {"trades": 0, "markets": 0}

        # 3. Analyze opportunities
        signals = await self.analyze_opportunities(forecasts, markets)

        if not signals:
            logger.info("No tradeable opportunities found")
            return {"trades": 0, "markets": len(markets), "signals": 0}

        # 4. Execute trades
        results = await self.execute_trades(signals)

        # 5. Summary
        successful = [r for r in results if r.success]
        total_volume = sum(r.executed_size for r in successful)

        summary = {
            "target_date": str(target_date),
            "markets_found": len(markets),
            "signals_found": len(signals),
            "trades_attempted": len(results),
            "trades_successful": len(successful),
            "total_volume": total_volume,
            "dry_run": self.dry_run,
        }

        logger.info(f"Cycle complete: {summary}")

        return summary


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Weather Trader for Kalshi")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Run without executing real trades")
    parser.add_argument("--live", action="store_true",
                        help="Execute real trades (overrides --dry-run)")
    parser.add_argument("--bankroll", type=float, default=None,
                        help="Starting bankroll (default: fetch from wallet)")

    args = parser.parse_args()

    dry_run = not args.live

    async with WeatherTrader(dry_run=dry_run, bankroll=args.bankroll) as trader:
        await trader.run_cycle()


if __name__ == "__main__":
    asyncio.run(main())
