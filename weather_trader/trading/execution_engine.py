"""
Execution Engine

Handles order creation, submission, and tracking for Kalshi trades.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import logging
import asyncio

from .config import TradingConfig, default_config
from .position_manager import PositionManager, Position, PositionStatus, ExitReason
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported."""
    MARKET = "market"       # Execute at current price
    LIMIT = "limit"         # Execute at specified price or better
    IOC = "ioc"             # Immediate-or-cancel


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "pending"         # Order created, not submitted
    SUBMITTED = "submitted"     # Sent to exchange
    PARTIAL = "partial"         # Partially filled
    FILLED = "filled"           # Fully filled
    CANCELLED = "cancelled"     # Cancelled
    REJECTED = "rejected"       # Rejected by exchange
    EXPIRED = "expired"         # Limit order expired


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    market_id: str
    condition_id: str
    ticker: str                  # Kalshi market ticker for the outcome
    city: str
    outcome_description: str

    order_type: OrderType
    side: OrderSide             # BUY or SELL
    outcome_side: str           # YES or NO (which outcome we're trading)

    price: float                # For limit orders
    size: float                 # Dollar amount
    shares: float               # Number of shares (size / price)

    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    filled_shares: float = 0.0
    filled_price: float = 0.0   # Average fill price
    filled_value: float = 0.0   # Total value filled

    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None

    # Linked position (for exits)
    position_id: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]

    @property
    def remaining_shares(self) -> float:
        """Shares remaining to be filled."""
        return max(0, self.shares - self.filled_shares)


class ExecutionEngine:
    """
    Manages order execution for the trading system.

    Handles:
    - Order creation and validation
    - Order submission to Kalshi
    - Order status tracking
    - Fill processing
    - Position creation from fills
    """

    def __init__(
        self,
        config: TradingConfig = None,
        position_manager: PositionManager = None,
        risk_manager: RiskManager = None,
        kalshi_client=None  # Will be the actual client
    ):
        self.config = config or default_config
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.kalshi_client = kalshi_client

        self.orders: dict[str, Order] = {}
        self._order_counter = 0

        # Callbacks for order events
        self._on_fill_callbacks: list[Callable[[Order], None]] = []
        self._on_reject_callbacks: list[Callable[[Order], None]] = []

    def create_entry_order(
        self,
        market_id: str,
        condition_id: str,
        ticker: str,
        city: str,
        outcome_description: str,
        outcome_side: str,
        price: float,
        size: float,
        forecast_prob: float,
        confidence: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> tuple[Optional[Order], Optional[str]]:
        """
        Create an order to enter a new position.

        Args:
            market_id: Market ID (event_ticker)
            condition_id: Condition ID for the outcome
            ticker: Kalshi market ticker for the outcome
            city: City for the weather market
            outcome_description: Human-readable outcome
            outcome_side: "YES" or "NO"
            price: Order price
            size: Dollar amount to invest
            forecast_prob: Model's probability forecast
            confidence: Model confidence level
            order_type: Type of order

        Returns:
            (Order if created, error message if failed)
        """
        # Calculate edge
        if outcome_side == "YES":
            edge = forecast_prob - price
        else:
            edge = (1 - forecast_prob) - price

        # Risk check
        if self.risk_manager:
            can_trade, violations = self.risk_manager.check_can_trade(
                market_id=market_id,
                city=city,
                position_size=size,
                edge=edge,
                confidence=confidence
            )
            if not can_trade:
                error_msgs = [v.message for v in violations if v.severity == "critical"]
                return None, "; ".join(error_msgs)

        # Calculate shares
        shares = size / price

        # Create order
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        order = Order(
            order_id=order_id,
            market_id=market_id,
            condition_id=condition_id,
            ticker=ticker,
            city=city,
            outcome_description=outcome_description,
            order_type=order_type,
            side=OrderSide.BUY,
            outcome_side=outcome_side,
            price=price,
            size=size,
            shares=shares
        )

        self.orders[order_id] = order
        logger.info(f"Created entry order {order_id}: {outcome_side} {outcome_description} @ {price} for ${size}")

        return order, None

    def create_exit_order(
        self,
        position: Position,
        exit_price: float,
        exit_reason: ExitReason,
        order_type: OrderType = OrderType.LIMIT
    ) -> tuple[Optional[Order], Optional[str]]:
        """
        Create an order to exit an existing position.

        Args:
            position: Position to close
            exit_price: Price to sell at
            exit_reason: Why we're exiting
            order_type: Type of order

        Returns:
            (Order if created, error message if failed)
        """
        if position.status != PositionStatus.OPEN:
            return None, f"Position {position.position_id} is not open"

        # Create order
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        # For exits, we sell what we bought
        order = Order(
            order_id=order_id,
            market_id=position.market_id,
            condition_id=position.condition_id,
            ticker="",  # Will need to look this up
            city=position.city,
            outcome_description=position.outcome_description,
            order_type=order_type,
            side=OrderSide.SELL,
            outcome_side=position.side,
            price=exit_price,
            size=exit_price * position.shares,
            shares=position.shares,
            position_id=position.position_id
        )

        self.orders[order_id] = order

        # Mark position as closing
        position.status = PositionStatus.CLOSING

        logger.info(
            f"Created exit order {order_id} for position {position.position_id}: "
            f"SELL {position.shares:.2f} shares @ {exit_price} ({exit_reason.value})"
        )

        return order, None

    async def submit_order(self, order: Order) -> bool:
        """
        Submit an order to Kalshi.

        Returns True if submission was successful.
        """
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Cannot submit order {order.order_id} in status {order.status}")
            return False

        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED

        # If no client, simulate success for testing
        if self.kalshi_client is None:
            logger.warning("No Kalshi client configured - simulating order submission")
            await self._simulate_fill(order)
            return True

        try:
            result = await self._submit_kalshi_order(order)

            if result:
                logger.info(f"Order {order.order_id} submitted successfully")
                return True
            else:
                order.status = OrderStatus.REJECTED
                logger.error(f"Order {order.order_id} rejected")
                return False

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Order {order.order_id} failed: {e}")
            return False

    async def _submit_kalshi_order(self, order: Order) -> bool:
        """Submit an order to Kalshi REST API."""
        try:
            action = "buy" if order.side == OrderSide.BUY else "sell"
            side = "yes" if order.outcome_side == "YES" else "no"
            price_cents = max(1, min(99, int(round(order.price * 100))))
            count = max(1, int(round(order.shares)))
            order_type = "market" if order.order_type == OrderType.MARKET else "limit"

            result = await self.kalshi_client.place_order(
                ticker=order.ticker,
                action=action,
                side=side,
                count=count,
                price_cents=price_cents,
                order_type=order_type,
            )

            if result.success:
                order.exchange_order_id = result.order_id
                order.filled_shares = result.filled_count if result.filled_count else count
                order.filled_price = result.filled_price if result.filled_price else order.price
                order.filled_value = order.filled_shares * order.filled_price
                order.filled_at = datetime.now()
                order.status = OrderStatus.FILLED
                self._process_fill(order)
                logger.info(f"Kalshi order {order.order_id} filled, exchange ID: {result.order_id}")
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.message
                logger.warning(f"Kalshi order {order.order_id} rejected: {result.message}")

            return result.success

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Kalshi order {order.order_id} failed: {e}")
            return False

    async def _simulate_fill(self, order: Order) -> None:
        """Simulate order fill for testing."""
        await asyncio.sleep(0.1)  # Simulate network delay

        # Fill at order price
        order.filled_shares = order.shares
        order.filled_price = order.price
        order.filled_value = order.shares * order.price
        order.filled_at = datetime.now()
        order.status = OrderStatus.FILLED

        self._process_fill(order)

    def _process_fill(self, order: Order) -> None:
        """Process a filled order - create/update positions."""
        if order.side == OrderSide.BUY:
            # Entry - create position
            if self.position_manager:
                position = self.position_manager.create_position(
                    market_id=order.market_id,
                    condition_id=order.condition_id,
                    city=order.city,
                    outcome_description=order.outcome_description,
                    settlement_date=datetime.now(),  # Will need actual settlement date
                    side=order.outcome_side,
                    entry_price=order.filled_price,
                    shares=order.filled_shares,
                    forecast_prob=order.filled_price  # Will need actual forecast
                )
                logger.info(f"Created position {position.position_id} from order {order.order_id}")

        else:
            # Exit - close position
            if self.position_manager and order.position_id:
                position = self.position_manager.positions.get(order.position_id)
                if position:
                    # Find the exit reason from the order context
                    exit_reason = ExitReason.MANUAL  # Default
                    self.position_manager.close_position(
                        position_id=order.position_id,
                        exit_price=order.filled_price,
                        reason=exit_reason
                    )
                    logger.info(f"Closed position {order.position_id} from order {order.order_id}")

        # Record trade for risk tracking
        if self.risk_manager:
            pnl = 0.0
            if order.side == OrderSide.SELL and order.position_id:
                position = self.position_manager.positions.get(order.position_id)
                if position:
                    pnl = position.realized_pnl or 0.0
            self.risk_manager.record_trade(order.market_id, pnl)

        # Trigger callbacks
        for callback in self._on_fill_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Returns True if cancellation was successful.
        """
        if order_id not in self.orders:
            logger.warning(f"Unknown order: {order_id}")
            return False

        order = self.orders[order_id]
        if order.is_complete:
            logger.warning(f"Cannot cancel completed order {order_id}")
            return False

        # If position was marked as closing, revert it
        if order.position_id and self.position_manager:
            position = self.position_manager.positions.get(order.position_id)
            if position and position.status == PositionStatus.CLOSING:
                position.status = PositionStatus.OPEN

        order.status = OrderStatus.CANCELLED
        logger.info(f"Cancelled order {order_id}")
        return True

    def get_open_orders(self) -> list[Order]:
        """Get all orders that are not in a terminal state."""
        return [o for o in self.orders.values() if not o.is_complete]

    def get_orders_for_market(self, market_id: str) -> list[Order]:
        """Get all orders for a specific market."""
        return [o for o in self.orders.values() if o.market_id == market_id]

    def on_fill(self, callback: Callable[[Order], None]) -> None:
        """Register a callback for order fills."""
        self._on_fill_callbacks.append(callback)

    def on_reject(self, callback: Callable[[Order], None]) -> None:
        """Register a callback for order rejections."""
        self._on_reject_callbacks.append(callback)

    def get_order_summary(self, order: Order) -> dict:
        """Get summary dict of order for display."""
        return {
            "order_id": order.order_id,
            "city": order.city,
            "outcome": order.outcome_description,
            "side": f"{order.side.value} {order.outcome_side}",
            "type": order.order_type.value,
            "price": round(order.price, 4),
            "size": round(order.size, 2),
            "shares": round(order.shares, 2),
            "status": order.status.value,
            "filled_shares": round(order.filled_shares, 2),
            "filled_price": round(order.filled_price, 4) if order.filled_price else None,
            "created_at": order.created_at.isoformat(),
            "error": order.error_message,
        }


class AutoTrader:
    """
    Automated trading coordinator.

    Orchestrates the flow:
    1. Receive trading signals
    2. Check risk limits
    3. Calculate position sizes
    4. Submit entry orders
    5. Monitor positions for exit signals
    6. Submit exit orders
    """

    def __init__(
        self,
        config: TradingConfig = None,
        position_manager: PositionManager = None,
        risk_manager: RiskManager = None,
        execution_engine: ExecutionEngine = None
    ):
        self.config = config or default_config
        self.position_manager = position_manager or PositionManager(config)
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine

        if not self.risk_manager:
            self.risk_manager = RiskManager(config, self.position_manager)

        if not self.execution_engine:
            self.execution_engine = ExecutionEngine(
                config,
                self.position_manager,
                self.risk_manager
            )

        self._enabled = False
        self._pending_exits: dict[str, ExitReason] = {}

    @property
    def enabled(self) -> bool:
        """Check if auto-trading is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable auto-trading."""
        self._enabled = True
        logger.info("Auto-trading enabled")

    def disable(self) -> None:
        """Disable auto-trading."""
        self._enabled = False
        logger.info("Auto-trading disabled")

    async def process_signal(
        self,
        market_id: str,
        condition_id: str,
        ticker: str,
        city: str,
        outcome_description: str,
        outcome_side: str,
        market_price: float,
        forecast_prob: float,
        confidence: float,
        settlement_date: datetime
    ) -> Optional[Order]:
        """
        Process a trading signal and potentially enter a position.

        Returns the created order if a trade was made, None otherwise.
        """
        if not self._enabled:
            return None

        # Calculate edge
        if outcome_side == "YES":
            edge = forecast_prob - market_price
        else:
            edge = (1 - forecast_prob) - market_price

        # Check minimum edge
        if edge < self.config.min_edge_to_enter:
            return None

        # Check confidence
        if confidence < self.config.min_confidence_to_enter:
            return None

        # Check if we already have a position in this market
        existing = self.position_manager.get_positions_by_market(market_id)
        if existing:
            # Already have a position - don't double up
            return None

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            edge=edge,
            win_probability=forecast_prob if outcome_side == "YES" else (1 - forecast_prob),
            price=market_price
        )

        if position_size < self.config.min_position_size:
            return None

        # Create and submit order
        order, error = self.execution_engine.create_entry_order(
            market_id=market_id,
            condition_id=condition_id,
            ticker=ticker,
            city=city,
            outcome_description=outcome_description,
            outcome_side=outcome_side,
            price=market_price,
            size=position_size,
            forecast_prob=forecast_prob,
            confidence=confidence
        )

        if error:
            logger.warning(f"Could not create order: {error}")
            return None

        # Submit order
        success = await self.execution_engine.submit_order(order)
        if not success:
            logger.warning(f"Order submission failed: {order.error_message}")
            return None

        return order

    async def check_exits(self) -> list[Order]:
        """
        Check all open positions for exit signals and submit exit orders.

        Returns list of exit orders created.
        """
        if not self._enabled:
            return []

        exit_orders = []
        evaluations = self.position_manager.evaluate_all_positions()

        for position, should_exit, exit_reason in evaluations:
            if should_exit and exit_reason:
                # Create exit order
                order, error = self.execution_engine.create_exit_order(
                    position=position,
                    exit_price=position.current_price,
                    exit_reason=exit_reason
                )

                if error:
                    logger.warning(f"Could not create exit order for {position.position_id}: {error}")
                    continue

                # Submit exit order
                success = await self.execution_engine.submit_order(order)
                if success:
                    exit_orders.append(order)
                    logger.info(
                        f"Auto-exit: {position.position_id} due to {exit_reason.value} "
                        f"(P/L: ${position.unrealized_pnl:.2f})"
                    )

        return exit_orders

    async def update_positions(
        self,
        market_updates: dict[str, tuple[float, float]]
    ) -> None:
        """
        Update all positions with latest market data.

        Args:
            market_updates: Dict of market_id -> (market_price, forecast_prob)
        """
        for position in self.position_manager.get_open_positions():
            if position.market_id in market_updates:
                price, prob = market_updates[position.market_id]
                self.position_manager.update_position(
                    position.position_id,
                    market_price=price,
                    forecast_prob=prob
                )

        # Update risk manager equity tracking
        self.risk_manager.update_equity()

    def get_status(self) -> dict:
        """Get auto-trader status summary."""
        return {
            "enabled": self._enabled,
            "open_positions": len(self.position_manager.get_open_positions()),
            "pending_orders": len(self.execution_engine.get_open_orders()),
            "risk_summary": self.risk_manager.get_risk_summary(),
        }
