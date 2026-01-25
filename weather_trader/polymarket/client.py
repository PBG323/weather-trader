"""
Polymarket CLOB Client

Handles order execution on Polymarket's Central Limit Order Book.

Key concepts:
- Polymarket uses a CLOB model (not AMM)
- Orders are placed on Polygon network
- Settlement is in USDC
- 0% trading fees
"""

import time
import httpx
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
from enum import Enum

from ..config import config
from .auth import PolymarketAuth
from .markets import WeatherMarket


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    LIMIT = "GTC"  # Good til cancelled
    FOK = "FOK"    # Fill or kill
    IOC = "IOC"    # Immediate or cancel


@dataclass
class Order:
    """An order to be placed."""
    market: WeatherMarket
    side: OrderSide
    token_id: str  # YES or NO token ID
    size: float    # Amount in shares
    price: float   # Price per share (0-1)
    order_type: OrderType = OrderType.LIMIT


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: Optional[str]
    filled_size: float
    filled_price: float
    message: str
    timestamp: datetime


@dataclass
class Position:
    """Current position in a market."""
    market: WeatherMarket
    yes_shares: float
    no_shares: float
    average_entry_price: float
    unrealized_pnl: float


class PolymarketClient:
    """
    Client for trading on Polymarket CLOB.
    """

    def __init__(self, auth: Optional[PolymarketAuth] = None):
        self.auth = auth or PolymarketAuth()
        self.clob_url = config.api.polymarket_clob_url
        self.client = httpx.AsyncClient(timeout=30.0)

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    async def get_orderbook(self, token_id: str) -> dict:
        """
        Get orderbook for a token.

        Args:
            token_id: The token ID (YES or NO)

        Returns:
            Dictionary with bids, asks, and spread
        """
        self._rate_limit()

        endpoint = f"{self.clob_url}/book"
        params = {"token_id": token_id}

        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        bids = sorted(data.get("bids", []), key=lambda x: -float(x["price"]))
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))

        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1

        return {
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": (best_bid + best_ask) / 2,
            "spread": best_ask - best_bid,
        }

    async def get_market_price(self, market: WeatherMarket) -> dict:
        """
        Get current market prices for both YES and NO tokens.

        Args:
            market: The WeatherMarket to get prices for

        Returns:
            Dictionary with prices for both sides
        """
        yes_book = await self.get_orderbook(market.yes_token_id)

        return {
            "yes_bid": yes_book["best_bid"],
            "yes_ask": yes_book["best_ask"],
            "yes_mid": yes_book["mid_price"],
            "no_bid": 1 - yes_book["best_ask"],
            "no_ask": 1 - yes_book["best_bid"],
            "spread": yes_book["spread"],
        }

    async def place_order(self, order: Order) -> OrderResult:
        """
        Place an order on Polymarket.

        Note: This is a simplified implementation. The actual py-clob-client
        library should be used for production, as it handles:
        - Proper signature generation
        - Nonce management
        - Order encoding

        Args:
            order: The Order to place

        Returns:
            OrderResult with fill information
        """
        if not self.auth.is_configured:
            return OrderResult(
                success=False,
                order_id=None,
                filled_size=0,
                filled_price=0,
                message="Wallet not configured",
                timestamp=datetime.now(),
            )

        # Validate order
        if order.size <= 0:
            return OrderResult(
                success=False,
                order_id=None,
                filled_size=0,
                filled_price=0,
                message="Order size must be positive",
                timestamp=datetime.now(),
            )

        if not 0 < order.price < 1:
            return OrderResult(
                success=False,
                order_id=None,
                filled_size=0,
                filled_price=0,
                message="Price must be between 0 and 1",
                timestamp=datetime.now(),
            )

        self._rate_limit()

        # In production, use py-clob-client for proper order signing
        # This is a placeholder showing the order structure
        order_payload = {
            "tokenID": order.token_id,
            "price": str(order.price),
            "size": str(order.size),
            "side": order.side.value,
            "feeRateBps": "0",
            "nonce": str(int(time.time() * 1000)),
            "expiration": "0",  # Never expires for GTC
            "taker": self.auth.address,
        }

        try:
            # The actual implementation would use py-clob-client
            # For now, we'll return a simulated result
            endpoint = f"{self.clob_url}/order"

            # Note: In production, you'd sign the order with the wallet
            # headers = {"Authorization": f"Bearer {signed_auth}"}

            # Simulated response for development
            return OrderResult(
                success=True,
                order_id=f"sim_{int(time.time())}",
                filled_size=order.size,
                filled_price=order.price,
                message="Order placed (simulated)",
                timestamp=datetime.now(),
            )

        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                filled_size=0,
                filled_price=0,
                message=f"Order failed: {str(e)}",
                timestamp=datetime.now(),
            )

    async def place_limit_order(
        self,
        market: WeatherMarket,
        side: Literal["BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"],
        size: float,
        price: Optional[float] = None
    ) -> OrderResult:
        """
        Place a limit order with smart pricing.

        Args:
            market: The WeatherMarket to trade
            side: Order side (BUY_YES, BUY_NO, SELL_YES, SELL_NO)
            size: Number of shares
            price: Limit price (if None, uses best bid/ask)

        Returns:
            OrderResult
        """
        # Determine token and order side
        if side in ["BUY_YES", "SELL_YES"]:
            token_id = market.yes_token_id
        else:
            token_id = market.no_token_id

        order_side = OrderSide.BUY if side.startswith("BUY") else OrderSide.SELL

        # Get current prices if price not specified
        if price is None:
            prices = await self.get_market_price(market)
            if side == "BUY_YES":
                price = prices["yes_ask"] * 0.99  # Slightly under ask
            elif side == "BUY_NO":
                price = prices["no_ask"] * 0.99
            elif side == "SELL_YES":
                price = prices["yes_bid"] * 1.01  # Slightly over bid
            else:  # SELL_NO
                price = prices["no_bid"] * 1.01

        order = Order(
            market=market,
            side=order_side,
            token_id=token_id,
            size=size,
            price=price,
            order_type=OrderType.LIMIT,
        )

        return await self.place_order(order)

    async def get_positions(self) -> list[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        if not self.auth.is_configured:
            return []

        # In production, query the CLOB API for positions
        # This is a placeholder
        return []

    async def get_open_orders(self) -> list[dict]:
        """
        Get all open orders.

        Returns:
            List of open order dictionaries
        """
        if not self.auth.is_configured:
            return []

        self._rate_limit()

        try:
            endpoint = f"{self.clob_url}/orders"
            params = {"maker": self.auth.address}

            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return []

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: The order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if not self.auth.is_configured:
            return False

        self._rate_limit()

        try:
            endpoint = f"{self.clob_url}/order/{order_id}"

            # In production, sign the cancellation request
            response = await self.client.delete(endpoint)
            return response.status_code == 200

        except Exception:
            return False

    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        open_orders = await self.get_open_orders()
        cancelled = 0

        for order in open_orders:
            order_id = order.get("id")
            if order_id and await self.cancel_order(order_id):
                cancelled += 1

        return cancelled
