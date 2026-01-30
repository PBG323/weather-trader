"""
Kalshi order execution client.

Handles authenticated REST API calls for trading: balance, orders, positions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx

from .auth import KalshiAuth
from .markets import TemperatureBracket
from ..config import config


@dataclass
class OrderResult:
    """Result of an order placement on Kalshi."""
    success: bool
    order_id: str = ""
    filled_count: int = 0
    filled_price_cents: int = 0
    remaining_count: int = 0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def filled_price(self) -> float:
        """Filled price as 0-1 probability (strategy layer compat)."""
        return self.filled_price_cents / 100.0

    @property
    def filled_size(self) -> float:
        """Dollar value of filled contracts (each contract is $1 notional)."""
        return float(self.filled_count)


class KalshiClient:
    """Authenticated client for Kalshi trading API."""

    def __init__(self, auth: Optional[KalshiAuth] = None, base_url: str = ""):
        self._auth = auth or KalshiAuth()
        self._base_url = base_url or config.api.kalshi_api_base_url
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    def _ensure_client(self):
        """Ensure HTTP client is initialized (for use outside async context manager)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=30.0,
            )

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated request to the Kalshi API.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API path (e.g., /trade-api/v2/portfolio/balance)
            **kwargs: Additional arguments to httpx (json, params, etc.)

        Returns:
            JSON response as dict.
        """
        self._ensure_client()

        # Build the full path for signature (relative to domain)
        parsed = urlparse(self._base_url)
        base_path = parsed.path.rstrip("/")
        full_path = f"{base_path}/{path.lstrip('/')}" if not path.startswith(base_path) else path

        headers = self._auth.get_auth_headers(method, full_path)
        resp = await self._client.request(method, path, headers=headers, **kwargs)
        resp.raise_for_status()

        if resp.status_code == 204:
            return {}
        return resp.json()

    async def get_balance(self) -> float:
        """Get account balance in dollars.

        Kalshi returns balance in cents; we convert to dollars.
        """
        data = await self._request("GET", "/portfolio/balance")
        balance_cents = data.get("balance", 0)
        return balance_cents / 100.0

    async def get_market_price(self, bracket: TemperatureBracket) -> dict:
        """Get current prices for a bracket as 0-1 floats.

        Returns:
            Dict with yes_bid, yes_ask, yes_mid, no_bid, no_ask, spread as 0-1 floats.
        """
        data = await self._request("GET", f"/markets/{bracket.ticker}")
        market = data if "ticker" in data else data.get("market", {})

        yes_bid = (market.get("yes_bid", 0) or 0) / 100.0
        yes_ask = (market.get("yes_ask", 0) or 0) / 100.0
        yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else yes_bid or yes_ask

        return {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_mid": yes_mid,
            "no_bid": 1 - yes_ask,
            "no_ask": 1 - yes_bid,
            "spread": yes_ask - yes_bid,
        }

    async def place_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price_cents: int,
        order_type: str = "limit",
        time_in_force: str = "immediate_or_cancel",
    ) -> OrderResult:
        """Place an order on Kalshi.

        Args:
            ticker: Market ticker (e.g., "KXHIGHNY-26JAN28-T52")
            action: "buy" or "sell"
            side: "yes" or "no"
            count: Number of contracts
            price_cents: Limit price in cents (1-99)
            order_type: "limit" or "market"
            time_in_force: "immediate_or_cancel" (IOC), "good_till_canceled" (GTC),
                          or "fill_or_kill" (FOK). Default is IOC for instant feedback.

        Returns:
            OrderResult with fill details.
        """
        # Validate price is within valid range (1-99 cents)
        if order_type == "limit":
            if not (1 <= price_cents <= 99):
                return OrderResult(
                    success=False,
                    message=f"Invalid price: {price_cents} cents. Must be between 1 and 99.",
                )

        # Validate count is positive
        if count <= 0:
            return OrderResult(
                success=False,
                message=f"Invalid count: {count}. Must be positive.",
            )

        body = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        if order_type == "limit":
            # Kalshi expects yes_price only (not both yes_price and no_price)
            body["yes_price"] = price_cents if side == "yes" else (100 - price_cents)

        try:
            data = await self._request("POST", "/portfolio/orders", json=body)
            order = data.get("order", data)

            return OrderResult(
                success=True,
                order_id=order.get("order_id", ""),
                filled_count=order.get("filled_count", 0),
                filled_price_cents=order.get("avg_price", price_cents) or price_cents,
                remaining_count=order.get("remaining_count", 0),
                message="Order placed successfully",
            )
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_json = e.response.json()
                error_body = error_json.get("message", "") or error_json.get("error", "") or str(error_json)
            except Exception:
                error_body = e.response.text or str(e)
            return OrderResult(
                success=False,
                message=f"Order failed: {error_body}",
            )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Order error: {str(e)}",
            )

    async def place_limit_order(
        self,
        bracket: TemperatureBracket,
        side_str: str,
        count: int,
        price_cents: int,
    ) -> OrderResult:
        """High-level limit order: translates side_str to action+side.

        Args:
            bracket: The TemperatureBracket to trade.
            side_str: "YES" or "NO" (which outcome to buy).
            count: Number of contracts.
            price_cents: Limit price in cents.

        Returns:
            OrderResult with fill details.
        """
        # Buying YES → action=buy, side=yes
        # Buying NO  → action=buy, side=no
        action = "buy"
        side = "yes" if side_str.upper() == "YES" else "no"

        return await self.place_order(
            ticker=bracket.ticker,
            action=action,
            side=side,
            count=count,
            price_cents=price_cents,
        )

    async def get_positions(self) -> list[dict]:
        """Get all open positions."""
        data = await self._request("GET", "/portfolio/positions")
        positions = data.get("market_positions", [])
        return positions

    async def get_fills(self, ticker: str = None, limit: int = 100) -> list[dict]:
        """Get trade fills (executed orders) with actual prices.

        Args:
            ticker: Optional ticker to filter fills
            limit: Max number of fills to return

        Returns:
            List of fill dicts with 'ticker', 'price', 'count', 'side', etc.
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        data = await self._request("GET", "/portfolio/fills", params=params)
        return data.get("fills", [])

    async def get_average_entry_price(self, ticker: str) -> float:
        """Calculate average entry price for a position from fills.

        Returns price in cents (1-99).
        """
        fills = await self.get_fills(ticker=ticker)
        if not fills:
            return 50  # Default

        # Calculate weighted average of buy fills
        # Kalshi fields: ticker, side (yes/no), action (buy/sell),
        # yes_price, no_price, count
        total_cost = 0
        total_count = 0
        for fill in fills:
            # Filter to buys for this ticker
            action = fill.get("action", "").lower()
            if action != "buy":
                continue

            count = fill.get("count", 0)
            if count <= 0:
                continue

            # Get price - try multiple field names
            side = fill.get("side", "yes").lower()
            if side == "yes":
                price = fill.get("yes_price") or fill.get("price", 50)
            else:
                price = fill.get("no_price") or fill.get("price", 50)

            total_cost += price * count
            total_count += count

        if total_count > 0:
            return total_cost / total_count
        return 50  # Default if no buy fills found

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            await self._request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except Exception:
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        try:
            data = await self._request("DELETE", "/portfolio/orders")
            return data.get("cancelled_count", 0)
        except Exception:
            return 0
