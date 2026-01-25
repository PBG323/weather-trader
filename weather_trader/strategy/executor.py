"""
Trade Executor

Orchestrates the execution of trades based on position recommendations.

Responsibilities:
- Order placement and confirmation
- Slippage management
- Trade logging and journaling
- Position tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import asyncio

from ..polymarket.client import PolymarketClient, OrderResult
from ..polymarket.markets import WeatherMarket
from .position_sizing import PositionRecommendation
from .expected_value import TradeSignal


@dataclass
class ExecutionResult:
    """Result of a trade execution."""
    recommendation: PositionRecommendation
    order_result: OrderResult

    # Execution details
    intended_size: float
    executed_size: float
    intended_price: float
    executed_price: float
    slippage: float

    # Timestamps
    signal_time: datetime
    execution_time: datetime

    # Status
    success: bool
    partial_fill: bool
    error_message: Optional[str] = None

    @property
    def fill_rate(self) -> float:
        """Percentage of intended order that was filled."""
        if self.intended_size == 0:
            return 0
        return self.executed_size / self.intended_size

    def to_journal_entry(self) -> dict:
        """Convert to journal entry format for logging."""
        return {
            "timestamp": self.execution_time.isoformat(),
            "market": self.recommendation.signal.market.market_slug,
            "city": self.recommendation.signal.market.city,
            "threshold": self.recommendation.signal.threshold,
            "side": self.recommendation.signal.side,
            "edge": self.recommendation.signal.edge,
            "confidence": self.recommendation.signal.confidence,
            "intended_size": self.intended_size,
            "executed_size": self.executed_size,
            "price": self.executed_price,
            "slippage": self.slippage,
            "success": self.success,
            "order_id": self.order_result.order_id if self.order_result else None,
        }


@dataclass
class TradeJournal:
    """Journal of all trades for analysis."""
    entries: list[ExecutionResult] = field(default_factory=list)

    def add(self, result: ExecutionResult):
        """Add an execution result to the journal."""
        self.entries.append(result)

    def get_summary(self) -> dict:
        """Get summary statistics of all trades."""
        if not self.entries:
            return {
                "total_trades": 0,
                "success_rate": 0,
                "total_volume": 0,
                "avg_edge": 0,
                "avg_slippage": 0,
            }

        successful = [e for e in self.entries if e.success]

        return {
            "total_trades": len(self.entries),
            "successful_trades": len(successful),
            "success_rate": len(successful) / len(self.entries),
            "total_volume": sum(e.executed_size for e in successful),
            "avg_edge": sum(e.recommendation.signal.edge for e in self.entries) / len(self.entries),
            "avg_slippage": sum(e.slippage for e in successful) / len(successful) if successful else 0,
            "by_city": self._group_by_city(),
        }

    def _group_by_city(self) -> dict:
        """Group trades by city."""
        by_city = {}
        for entry in self.entries:
            city = entry.recommendation.signal.market.city
            if city not in by_city:
                by_city[city] = {"count": 0, "volume": 0}
            by_city[city]["count"] += 1
            by_city[city]["volume"] += entry.executed_size
        return by_city


class TradeExecutor:
    """
    Executes trades based on position recommendations.
    """

    def __init__(
        self,
        client: PolymarketClient,
        max_slippage: float = 0.02,  # 2% max slippage
        use_limit_orders: bool = True,
        dry_run: bool = False,
    ):
        """
        Initialize trade executor.

        Args:
            client: PolymarketClient for order execution
            max_slippage: Maximum acceptable slippage
            use_limit_orders: Use limit orders (vs market)
            dry_run: If True, don't actually execute trades
        """
        self.client = client
        self.max_slippage = max_slippage
        self.use_limit_orders = use_limit_orders
        self.dry_run = dry_run

        self.journal = TradeJournal()

    async def execute(self, recommendation: PositionRecommendation) -> ExecutionResult:
        """
        Execute a single trade recommendation.

        Args:
            recommendation: PositionRecommendation to execute

        Returns:
            ExecutionResult with execution details
        """
        signal = recommendation.signal
        signal_time = datetime.now()

        # Determine order parameters
        market = signal.market
        side = f"BUY_{signal.side}"
        size = recommendation.recommended_shares

        # Get current price for slippage calculation
        current_prices = await self.client.get_market_price(market)

        if signal.side == "YES":
            current_price = current_prices["yes_ask"]
            intended_price = market.yes_price
        else:
            current_price = current_prices["no_ask"]
            intended_price = market.no_price

        # Check slippage
        slippage = (current_price - intended_price) / intended_price if intended_price > 0 else 0

        if slippage > self.max_slippage:
            return ExecutionResult(
                recommendation=recommendation,
                order_result=OrderResult(
                    success=False,
                    order_id=None,
                    filled_size=0,
                    filled_price=0,
                    message=f"Slippage too high: {slippage:.2%}",
                    timestamp=datetime.now(),
                ),
                intended_size=recommendation.recommended_size,
                executed_size=0,
                intended_price=intended_price,
                executed_price=0,
                slippage=slippage,
                signal_time=signal_time,
                execution_time=datetime.now(),
                success=False,
                partial_fill=False,
                error_message=f"Slippage {slippage:.2%} exceeds max {self.max_slippage:.2%}",
            )

        # Execute trade
        if self.dry_run:
            order_result = OrderResult(
                success=True,
                order_id=f"dry_run_{int(signal_time.timestamp())}",
                filled_size=size,
                filled_price=current_price,
                message="Dry run - no actual trade",
                timestamp=datetime.now(),
            )
        else:
            # Calculate limit price with small offset for better fills
            limit_price = current_price * 1.005 if self.use_limit_orders else None

            order_result = await self.client.place_limit_order(
                market=market,
                side=side,
                size=size,
                price=limit_price,
            )

        # Build execution result
        executed_price = order_result.filled_price if order_result.success else 0
        actual_slippage = (executed_price - intended_price) / intended_price if intended_price > 0 and executed_price > 0 else 0

        result = ExecutionResult(
            recommendation=recommendation,
            order_result=order_result,
            intended_size=recommendation.recommended_size,
            executed_size=order_result.filled_size * executed_price,
            intended_price=intended_price,
            executed_price=executed_price,
            slippage=actual_slippage,
            signal_time=signal_time,
            execution_time=datetime.now(),
            success=order_result.success,
            partial_fill=order_result.filled_size < size and order_result.filled_size > 0,
            error_message=None if order_result.success else order_result.message,
        )

        # Add to journal
        self.journal.add(result)

        return result

    async def execute_batch(
        self,
        recommendations: list[PositionRecommendation],
        parallel: bool = False,
        delay_seconds: float = 0.5,
    ) -> list[ExecutionResult]:
        """
        Execute multiple trade recommendations.

        Args:
            recommendations: List of PositionRecommendation objects
            parallel: Execute in parallel (may cause issues with rate limits)
            delay_seconds: Delay between sequential executions

        Returns:
            List of ExecutionResult objects
        """
        results = []

        if parallel:
            # Execute all at once
            tasks = [self.execute(rec) for rec in recommendations]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to failed results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = ExecutionResult(
                        recommendation=recommendations[i],
                        order_result=OrderResult(
                            success=False,
                            order_id=None,
                            filled_size=0,
                            filled_price=0,
                            message=str(result),
                            timestamp=datetime.now(),
                        ),
                        intended_size=recommendations[i].recommended_size,
                        executed_size=0,
                        intended_price=0,
                        executed_price=0,
                        slippage=0,
                        signal_time=datetime.now(),
                        execution_time=datetime.now(),
                        success=False,
                        partial_fill=False,
                        error_message=str(result),
                    )
        else:
            # Execute sequentially with delay
            for rec in recommendations:
                result = await self.execute(rec)
                results.append(result)
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

        return results

    def get_journal_summary(self) -> dict:
        """Get summary of all executed trades."""
        return self.journal.get_summary()

    async def verify_positions(self) -> dict:
        """
        Verify current positions match expected state.

        Returns:
            Dictionary with position verification results
        """
        positions = await self.client.get_positions()

        expected_from_journal = {}
        for entry in self.journal.entries:
            if entry.success:
                market_id = entry.recommendation.signal.market.condition_id
                side = entry.recommendation.signal.side

                if market_id not in expected_from_journal:
                    expected_from_journal[market_id] = {"yes": 0, "no": 0}

                if side == "YES":
                    expected_from_journal[market_id]["yes"] += entry.executed_size
                else:
                    expected_from_journal[market_id]["no"] += entry.executed_size

        return {
            "expected": expected_from_journal,
            "actual": {p.market.condition_id: {"yes": p.yes_shares, "no": p.no_shares}
                       for p in positions},
        }
