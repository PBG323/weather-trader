"""
Weather Trader Dashboard - Full Featured

A comprehensive Streamlit-based UI for the weather trading system.
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import scipy.stats as stats
import time
import random
from zoneinfo import ZoneInfo

# EST timezone
EST = ZoneInfo("America/New_York")

def get_est_now() -> datetime:
    """Get current time in EST."""
    return datetime.now(EST)

def today_est() -> date:
    """Get today's date in Eastern time (Kalshi market timezone)."""
    return datetime.now(EST).date()

def format_est_time(dt: datetime = None) -> str:
    """Format datetime in EST."""
    if dt is None:
        dt = get_est_now()
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=EST)
    return dt.astimezone(EST).strftime('%H:%M:%S EST')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather_trader.config import get_all_cities, get_city_config, config
from weather_trader.apis import OpenMeteoClient, TomorrowIOClient, NWSClient
from weather_trader.models import EnsembleForecaster, BiasCorrector
from weather_trader.models.ensemble import ModelForecast
from weather_trader.kalshi import KalshiMarketFinder, KalshiAuth, KalshiClient, SameDayTradingChecker
from weather_trader.strategy import ExpectedValueCalculator
from weather_trader.trading import (
    TradingConfig, PositionManager, RiskManager, ExecutionEngine,
    AutoTrader, PnLTracker, Position, ExitReason, PositionStatus
)

# Page config
st.set_page_config(
    page_title="Weather Trader",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin: 10px 0; }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .neutral { color: #ffc107; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .alert-box { padding: 10px; border-radius: 5px; margin: 5px 0; }
    .alert-info { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .alert-success { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .alert-warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .alert-danger { background-color: #ffebee; border-left: 4px solid #f44336; }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #00c853;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .auto-trade-active {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []  # Active positions not yet settled
if 'closed_positions' not in st.session_state:
    st.session_state.closed_positions = []  # Settled/sold positions
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'price_history' not in st.session_state:
    st.session_state.price_history = {}
if 'forecast_history' not in st.session_state:
    st.session_state.forecast_history = []
if 'pnl' not in st.session_state:
    st.session_state.pnl = 0.0
if 'realized_pnl' not in st.session_state:
    st.session_state.realized_pnl = 0.0
if 'total_trades' not in st.session_state:
    st.session_state.total_trades = 0
if 'winning_trades' not in st.session_state:
    st.session_state.winning_trades = 0
if 'auto_trade_enabled' not in st.session_state:
    st.session_state.auto_trade_enabled = False
if 'last_auto_trade_check' not in st.session_state:
    st.session_state.last_auto_trade_check = None
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True

# Trading Engine Components
if 'trading_config' not in st.session_state:
    st.session_state.trading_config = TradingConfig()
if 'position_manager' not in st.session_state:
    st.session_state.position_manager = PositionManager(st.session_state.trading_config)
if 'pnl_tracker' not in st.session_state:
    st.session_state.pnl_tracker = PnLTracker()
if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = RiskManager(
        config=st.session_state.trading_config,
        position_manager=st.session_state.position_manager,
        initial_bankroll=1000.0
    )
if 'auto_trader' not in st.session_state:
    st.session_state.auto_trader = AutoTrader(
        config=st.session_state.trading_config,
        position_manager=st.session_state.position_manager,
        risk_manager=st.session_state.risk_manager
    )

# Kalshi live trading client (initialized lazily when live mode is selected)
if 'kalshi_client' not in st.session_state:
    st.session_state.kalshi_client = None
if 'live_wallet_validated' not in st.session_state:
    st.session_state.live_wallet_validated = False
if 'positions_synced' not in st.session_state:
    st.session_state.positions_synced = False
if 'pending_orders' not in st.session_state:
    st.session_state.pending_orders = []  # Track orders waiting to fill


def check_pending_orders():
    """Check status of pending orders on Kalshi and update tracking."""
    from weather_trader.kalshi import KalshiAuth, KalshiClient

    auth = KalshiAuth()
    if not auth.is_configured:
        return {"filled": 0, "pending": 0, "cancelled": 0, "pending_orders": []}

    async def _fetch_orders():
        client = KalshiClient(auth)
        data = await client._request("GET", "/portfolio/orders")
        return data.get("orders", [])

    try:
        orders = run_async(_fetch_orders())
    except Exception as e:
        add_alert(f"Error checking orders: {e}", "warning")
        return {"filled": 0, "pending": 0, "cancelled": 0, "pending_orders": []}

    # Filter to weather orders
    weather_orders = [o for o in orders if "KXHIGH" in o.get("ticker", "")]

    filled_count = 0
    pending_count = 0
    cancelled_count = 0

    for order in weather_orders:
        status = order.get("status", "")
        ticker = order.get("ticker", "")
        remaining = order.get("remaining_count", 0)
        filled = order.get("filled_count", 0) or order.get("fill_count", 0)

        if status == "executed" and filled > 0:
            filled_count += 1
        elif status in ("open", "pending") and remaining > 0:
            pending_count += 1
        elif status == "cancelled":
            cancelled_count += 1

    # Update session state
    st.session_state.pending_orders = [
        o for o in weather_orders
        if o.get("status") in ("open", "pending") and o.get("remaining_count", 0) > 0
    ]

    return {
        "filled": filled_count,
        "pending": pending_count,
        "cancelled": cancelled_count,
        "pending_orders": st.session_state.pending_orders
    }


def get_live_kalshi_data():
    """Fetch comprehensive live data from Kalshi."""
    from weather_trader.kalshi import KalshiAuth, KalshiClient

    auth = KalshiAuth()
    if not auth.is_configured:
        return None

    async def _fetch_all():
        client = KalshiClient(auth)

        # Get balance
        balance = await client.get_balance()

        # Get positions with current prices
        positions = await client.get_positions()

        # Get pending orders
        orders_data = await client._request("GET", "/portfolio/orders")
        orders = orders_data.get("orders", [])

        # Calculate totals
        weather_positions = [p for p in positions if "KXHIGH" in p.get("ticker", "")]
        total_position_value = sum(
            abs(p.get("position", 0)) * (p.get("market_price", 50) / 100.0)
            for p in weather_positions
        )

        pending_orders = [
            o for o in orders
            if o.get("status") in ("open", "pending")
            and o.get("remaining_count", 0) > 0
            and "KXHIGH" in o.get("ticker", "")
        ]

        return {
            "balance": balance,
            "positions": weather_positions,
            "position_count": len(weather_positions),
            "total_position_value": total_position_value,
            "pending_orders": pending_orders,
            "pending_count": len(pending_orders),
        }

    try:
        return run_async(_fetch_all())
    except Exception as e:
        add_alert(f"Error fetching Kalshi data: {e}", "warning")
        return None


def update_position_prices_from_kalshi():
    """Update all position prices from live Kalshi data."""
    from weather_trader.kalshi import KalshiAuth, KalshiClient
    import asyncio

    auth = KalshiAuth()
    if not auth.is_configured:
        return 0

    async def _update_prices():
        client = KalshiClient(auth)
        updated = 0

        # Get list of tickers to update (avoid modifying list during iteration)
        positions_to_update = [
            (i, pos.get("ticker") or pos.get("condition_id", ""))
            for i, pos in enumerate(st.session_state.open_positions)
            if (pos.get("ticker") or pos.get("condition_id", "")) and not (pos.get("ticker") or pos.get("condition_id", "")).startswith("demo_")
        ]

        for idx, ticker in positions_to_update:
            if idx >= len(st.session_state.open_positions):
                continue  # Position was removed

            pos = st.session_state.open_positions[idx]

            try:
                market_data = await client._request("GET", f"/markets/{ticker}")
                market = market_data if "ticker" in market_data else market_data.get("market", {})

                # Debug: log market data fields
                yes_bid = (market.get("yes_bid", 0) or 0) / 100.0
                yes_ask = (market.get("yes_ask", 0) or 0) / 100.0
                last_price = (market.get("last_price", 0) or 0) / 100.0
                previous_yes_bid = (market.get("previous_yes_bid", 0) or 0) / 100.0

                add_alert(f"DEBUG {ticker}: bid={yes_bid:.2f} ask={yes_ask:.2f} last={last_price:.2f}", "info")

                # Use last_price if available (matches Kalshi display), else yes_bid
                current_price = last_price if last_price > 0 else yes_bid if yes_bid > 0 else yes_ask if yes_ask > 0 else pos.get("current_price", 0.5)

                pos["current_price"] = current_price
                pos["yes_bid"] = yes_bid
                pos["yes_ask"] = yes_ask
                pos["spread"] = yes_ask - yes_bid

                # Update P/L using current price
                entry = pos.get("entry_price", current_price)
                shares = pos.get("shares", 1)
                side = pos.get("side", "YES")
                if side == "YES":
                    pos["unrealized_pnl"] = (current_price - entry) * shares
                else:
                    pos["unrealized_pnl"] = (entry - current_price) * shares

                updated += 1

                # Rate limit: small delay between API calls
                await asyncio.sleep(0.2)

            except Exception as e:
                # Log but continue with other positions
                add_alert(f"Price update error for {ticker}: {e}", "warning")

        return updated

    try:
        return run_async(_update_prices())
    except Exception as e:
        return 0


def check_settled_markets():
    """Check if any positions are in settled markets and close them."""
    from weather_trader.kalshi import KalshiAuth, KalshiClient
    import asyncio

    auth = KalshiAuth()
    if not auth.is_configured:
        return 0

    async def _check_settlements():
        client = KalshiClient(auth)
        settled_positions = []

        # Build list of positions to check (copy to avoid modification during iteration)
        positions_to_check = [
            (pos["id"], pos.get("ticker") or pos.get("condition_id", ""))
            for pos in st.session_state.open_positions
            if (pos.get("ticker") or pos.get("condition_id", "")) and not (pos.get("ticker") or pos.get("condition_id", "")).startswith("demo_")
        ]

        for pos_id, ticker in positions_to_check:
            try:
                market_data = await client._request("GET", f"/markets/{ticker}")
                market = market_data if "ticker" in market_data else market_data.get("market", {})

                status = market.get("status", "")
                if status in ("settled", "closed", "finalized"):
                    result = market.get("result", "")
                    settlement_price = 1.0 if result == "yes" else 0.0 if result == "no" else 0.5
                    settled_positions.append((pos_id, settlement_price))

                # Rate limit
                await asyncio.sleep(0.2)

            except Exception as e:
                pass

        return settled_positions

    try:
        settled_positions = run_async(_check_settlements())
        # Close positions outside of async context
        for pos_id, settlement_price in settled_positions:
            close_position_smart(pos_id, settlement_price, ExitReason.SETTLEMENT)
        return len(settled_positions)
    except Exception as e:
        return 0


def cancel_stale_orders(max_age_minutes: int = 5):
    """Cancel orders that have been pending too long (default 5 minutes)."""
    from weather_trader.kalshi import KalshiAuth, KalshiClient
    from datetime import datetime, timedelta, timezone
    import asyncio

    auth = KalshiAuth()
    if not auth.is_configured:
        return 0

    async def _cancel_stale():
        client = KalshiClient(auth)
        data = await client._request("GET", "/portfolio/orders")
        orders = data.get("orders", [])

        cancelled = 0
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)

        for order in orders:
            if order.get("status") not in ("open", "pending"):
                continue
            if order.get("remaining_count", 0) == 0:
                continue
            if "KXHIGH" not in order.get("ticker", ""):
                continue

            # Check age
            created = order.get("created_time", "")
            if created:
                try:
                    # Handle ISO format with Z suffix
                    if created.endswith("Z"):
                        order_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    else:
                        order_time = datetime.fromisoformat(created)

                    # Ensure timezone aware comparison
                    if order_time.tzinfo is None:
                        order_time = order_time.replace(tzinfo=timezone.utc)

                    if order_time < cutoff:
                        order_id = order.get("order_id", "")
                        if order_id:
                            success = await client.cancel_order(order_id)
                            if success:
                                cancelled += 1
                            # Rate limit between cancels
                            await asyncio.sleep(0.2)
                except Exception as e:
                    # Continue with other orders
                    pass

        return cancelled

    try:
        return run_async(_cancel_stale())
    except Exception as e:
        add_alert(f"Error cancelling orders: {e}", "warning")
        return 0


def sync_positions_from_kalshi():
    """Sync open positions from Kalshi to dashboard state."""
    from weather_trader.kalshi import KalshiAuth, KalshiClient
    from datetime import datetime
    import re

    auth = KalshiAuth()
    if not auth.is_configured:
        add_alert("Cannot sync: Kalshi not configured", "warning")
        return 0

    async def _fetch_positions():
        client = KalshiClient(auth)
        return await client.get_positions()

    try:
        positions = run_async(_fetch_positions())
    except Exception as e:
        add_alert(f"Sync failed: {e}", "error")
        return 0

    # Filter to weather positions with non-zero holdings
    weather_positions = [
        p for p in positions
        if "KXHIGH" in p.get("ticker", "") and p.get("position", 0) != 0
    ]

    synced = 0
    # Check both condition_id and ticker fields to avoid duplicates
    existing_tickers = set()
    for pos in st.session_state.open_positions:
        if pos.get("condition_id"):
            existing_tickers.add(pos.get("condition_id"))
        if pos.get("ticker"):
            existing_tickers.add(pos.get("ticker"))

    for kp in weather_positions:
        ticker = kp.get("ticker", "")
        if not ticker or ticker in existing_tickers:
            continue  # Already tracked or invalid

        position_count = kp.get("position", 0)
        avg_price = kp.get("average_price_paid", 50) or 50

        # Determine side: positive = YES, negative = NO
        if position_count > 0:
            side = "YES"
            shares = position_count
        else:
            side = "NO"
            shares = abs(position_count)

        # Parse city from ticker (e.g., KXHIGHNY-26JAN28-B23.5 -> nyc)
        city_map = {"NY": "nyc", "CHI": "chicago", "MIA": "miami", "AUS": "austin", "LA": "la", "DEN": "denver", "PHL": "philadelphia"}
        city = "unknown"
        for code, city_key in city_map.items():
            if f"KXHIGH{code}" in ticker:
                city = city_key
                break

        # Parse date from ticker
        date_match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})', ticker)
        target_date = None
        if date_match:
            months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                      "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
            try:
                from datetime import date as dt_date
                year = 2000 + int(date_match.group(1))
                month = months.get(date_match.group(2), 1)
                day = int(date_match.group(3))
                target_date = dt_date(year, month, day)
            except (ValueError, TypeError) as e:
                print(f"[Sync] Date parsing error for ticker {ticker}: {e}")
                pass

        # Parse temperature bracket from ticker
        # Format: KXHIGHNY-26JAN28-B52 (below 52) or -T52 (52 or higher) or -B45-50 (range 45-50)
        temp_low = None
        temp_high = None
        bracket_part = ticker.split("-")[-1] if "-" in ticker else ""
        bracket_match = re.match(r'([BT])(\d+(?:\.\d+)?)', bracket_part)
        if bracket_match:
            bracket_type = bracket_match.group(1)
            temp_val = float(bracket_match.group(2))
            if bracket_type == "B":
                # Below X: temp_high = X-1 (or X-0.5 for inclusive)
                temp_high = temp_val - 1
                temp_low = None  # "or below"
            elif bracket_type == "T":
                # X or higher: temp_low = X
                temp_low = temp_val
                temp_high = None  # "or above"
        else:
            # Try range format like "45-50"
            range_match = re.match(r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', bracket_part)
            if range_match:
                temp_low = float(range_match.group(1))
                temp_high = float(range_match.group(2))

        entry_price = avg_price / 100.0

        legacy_position = {
            "id": f"sync_{ticker}_{datetime.now().timestamp()}",
            "open_time": datetime.now(),
            "city": city,
            "outcome": ticker.split("-")[-1] if "-" in ticker else ticker,
            "side": side,
            "size": shares * entry_price,
            "entry_price": entry_price,
            "current_price": entry_price,
            "edge_at_entry": 0.0,  # Will be calculated from current forecast
            "forecast_prob": entry_price,  # Initial estimate, will be recalculated
            "market_prob_at_entry": entry_price,
            "status": "OPEN",
            "mode": "LIVE (synced)",
            "is_demo": False,
            "condition_id": ticker,
            "ticker": ticker,
            "target_date": target_date,
            "unrealized_pnl": 0.0,
            "position_obj": None,
            "exchange_order_id": None,
            "shares": shares,
            "temp_low": temp_low,  # For edge calculation
            "temp_high": temp_high,  # For edge calculation
            "peak_pnl": 0.0,  # Initialize peak P/L for trailing stop
        }

        st.session_state.open_positions.append(legacy_position)
        synced += 1

    st.session_state.positions_synced = True
    return synced


def reconcile_positions_with_kalshi():
    """
    Full reconciliation of positions between dashboard and Kalshi.

    This function:
    1. Fetches ALL positions from Kalshi
    2. Removes dashboard positions that no longer exist in Kalshi
    3. Adds new positions from Kalshi that aren't in dashboard
    4. Updates share counts and prices for existing positions

    Returns: tuple (added, removed, updated)
    """
    from weather_trader.kalshi import KalshiAuth, KalshiClient
    from datetime import datetime
    import re

    auth = KalshiAuth()
    if not auth.is_configured:
        add_alert("Cannot reconcile: Kalshi not configured", "warning")
        return 0, 0, 0

    async def _fetch_positions():
        client = KalshiClient(auth)
        return await client.get_positions()

    try:
        kalshi_positions = run_async(_fetch_positions())
    except Exception as e:
        add_alert(f"Reconciliation failed: {e}", "error")
        return 0, 0, 0

    # Filter to weather positions with non-zero holdings
    kalshi_weather = {
        p.get("ticker", ""): p for p in kalshi_positions
        if "KXHIGH" in p.get("ticker", "") and p.get("position", 0) != 0
    }

    added = 0
    removed = 0
    updated = 0

    # Get dashboard position tickers
    dashboard_tickers = {}
    for pos in st.session_state.open_positions:
        ticker = pos.get("ticker") or pos.get("condition_id", "")
        if ticker and "KXHIGH" in ticker:
            dashboard_tickers[ticker] = pos

    # 1. REMOVE positions from dashboard that are no longer in Kalshi
    positions_to_remove = []
    for ticker, pos in dashboard_tickers.items():
        if ticker not in kalshi_weather:
            # Position was closed in Kalshi - mark for removal
            positions_to_remove.append(pos["id"])
            add_alert(f"Position {ticker} closed in Kalshi - removing from dashboard", "info")

    for pos_id in positions_to_remove:
        for i, pos in enumerate(st.session_state.open_positions):
            if pos["id"] == pos_id:
                # Move to closed positions with zero P/L (already settled in Kalshi)
                closed = {
                    **pos,
                    "close_time": datetime.now(),
                    "close_price": pos.get("current_price", pos.get("entry_price", 0.5)),
                    "status": "CLOSED",
                    "exit_reason": "Closed in Kalshi",
                    "realized_pnl": 0,  # Already realized in Kalshi
                }
                st.session_state.closed_positions.insert(0, closed)
                st.session_state.open_positions.pop(i)
                removed += 1
                break

    # 2. UPDATE existing positions with latest Kalshi data
    # Also check for old-format bracket descriptions that need fixing
    positions_to_refresh = []

    for ticker, kalshi_pos in kalshi_weather.items():
        if ticker in dashboard_tickers:
            dash_pos = dashboard_tickers[ticker]
            kalshi_shares = abs(kalshi_pos.get("position", 0))
            dash_shares = dash_pos.get("shares", 0)

            # Update if share count changed (handles partial fills)
            if kalshi_shares != dash_shares:
                dash_pos["shares"] = kalshi_shares

                # Bug #8 fix: Recalculate entry price from Kalshi data for partial fills
                # market_exposure is the total cost in cents, so avg price = exposure / shares
                market_exposure = kalshi_pos.get("market_exposure", 0) or 0
                if market_exposure > 0 and kalshi_shares > 0:
                    new_entry_price = (market_exposure / kalshi_shares) / 100.0
                    dash_pos["entry_price"] = new_entry_price
                    dash_pos["size"] = kalshi_shares * new_entry_price
                    add_alert(f"Updated {ticker}: {dash_shares}->{kalshi_shares} shares @ ${new_entry_price:.3f}", "info")
                else:
                    dash_pos["size"] = kalshi_shares * dash_pos.get("entry_price", 0.5)
                    add_alert(f"Updated {ticker}: shares {dash_shares} -> {kalshi_shares}", "info")
                updated += 1

            # Update market price if available
            market_price = kalshi_pos.get("market_price")
            if market_price:
                dash_pos["current_price"] = market_price / 100.0

            # Check if bracket description is in old format (T##, B##.#, etc.)
            outcome_desc = dash_pos.get("outcome_desc", "")
            if re.match(r'^[TB]\d', outcome_desc):
                # Old format detected - mark for refresh
                positions_to_refresh.append(dash_pos["id"])

    # Remove positions with old format so they get re-added with correct descriptions
    for pos_id in positions_to_refresh:
        for i, pos in enumerate(st.session_state.open_positions):
            if pos["id"] == pos_id:
                ticker = pos.get("ticker") or pos.get("condition_id", "")
                add_alert(f"Refreshing {ticker} with correct bracket description", "info")
                st.session_state.open_positions.pop(i)
                # Remove from dashboard_tickers so it gets re-added in step 3
                if ticker in dashboard_tickers:
                    del dashboard_tickers[ticker]
                break

    # 3. ADD new positions from Kalshi that aren't in dashboard
    # Need to fetch market data to get correct bracket info
    async def _fetch_market_data(client, ticker):
        try:
            data = await client._request("GET", f"/markets/{ticker}")
            return data.get("market", data)
        except Exception as e:
            print(f"[Sync] Failed to fetch market data for {ticker}: {e}")
            return None

    async def _add_new_positions():
        nonlocal added
        client = KalshiClient(auth)

        for ticker, kalshi_pos in kalshi_weather.items():
            if ticker in dashboard_tickers:
                continue

            position_count = kalshi_pos.get("position", 0)

            # Calculate entry price from Kalshi position data
            # Key field: market_exposure (cost in cents) / position (shares)
            market_exposure = kalshi_pos.get("market_exposure", 0) or 0

            if market_exposure > 0 and abs(position_count) > 0:
                # market_exposure is total cost in cents, divide by shares
                avg_price = market_exposure / abs(position_count)
            else:
                # Fallback: try to fetch from fills
                try:
                    avg_price = await client.get_average_entry_price(ticker)
                except Exception as e:
                    print(f"[Sync] Failed to get entry price for {ticker}: {e}")
                    avg_price = 50  # Default to 50 cents

            # Sanity check - ensure avg_price is reasonable (1-99 cents)
            if avg_price is None or avg_price < 1 or avg_price > 99:
                avg_price = 50

            if position_count > 0:
                side = "YES"
                shares = position_count
            else:
                side = "NO"
                shares = abs(position_count)

            # Parse city from ticker
            city_map = {"NY": "nyc", "CHI": "chicago", "MIA": "miami", "AUS": "austin",
                       "LA": "la", "DEN": "denver", "PHL": "philadelphia"}
            city = "unknown"
            for code, city_key in city_map.items():
                if f"KXHIGH{code}" in ticker:
                    city = city_key
                    break

            # Parse date from ticker
            date_match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})', ticker)
            target_date = None
            if date_match:
                months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
                try:
                    from datetime import date as dt_date
                    year = 2000 + int(date_match.group(1))
                    month = months.get(date_match.group(2), 1)
                    day = int(date_match.group(3))
                    target_date = dt_date(year, month, day)
                except (ValueError, TypeError) as e:
                    print(f"[Sync] Date parsing error for {ticker}: {e}")
                    pass

            # FETCH ACTUAL MARKET DATA for correct bracket info
            temp_low = None
            temp_high = None
            outcome_desc = ticker.split("-")[-1] if "-" in ticker else ticker
            market_price_cents = kalshi_pos.get("market_price", avg_price)

            market_data = await _fetch_market_data(client, ticker)
            if market_data:
                floor_strike = market_data.get("floor_strike")
                cap_strike = market_data.get("cap_strike")
                title = market_data.get("title", "")
                subtitle = market_data.get("subtitle", "")

                # Determine bracket type from floor/cap strikes
                if floor_strike is not None and cap_strike is not None:
                    # Range bracket (e.g., 63-64)
                    temp_low = floor_strike
                    temp_high = cap_strike
                    outcome_desc = f"{int(floor_strike)}-{int(cap_strike)}F"
                elif cap_strike is not None and floor_strike is None:
                    # Below bracket (e.g., <45 means 44 or below)
                    temp_high = cap_strike - 1
                    temp_low = None
                    outcome_desc = f"<={int(temp_high)}F"
                elif floor_strike is not None and cap_strike is None:
                    # Above bracket (e.g., >70 means 71 or above)
                    temp_low = floor_strike + 1
                    temp_high = None
                    outcome_desc = f">={int(temp_low)}F"

                # Get live price from market data - use yes_bid (sell price) to match Kalshi display
                yes_bid = market_data.get("yes_bid", 0) or 0
                yes_ask = market_data.get("yes_ask", 0) or 0
                # Use yes_bid as current price (what you can sell at)
                market_price_cents = yes_bid if yes_bid > 0 else yes_ask

            entry_price = avg_price / 100.0 if avg_price else 0.5
            current_price = market_price_cents / 100.0 if market_price_cents else entry_price

            legacy_position = {
                "id": f"sync_{ticker}_{datetime.now().timestamp()}",
                "open_time": datetime.now(),
                "city": city,
                "outcome": outcome_desc,
                "side": side,
                "size": shares * entry_price,
                "entry_price": entry_price,
                "current_price": current_price,
                "edge_at_entry": 0.0,
                "forecast_prob": entry_price,
                "market_prob_at_entry": entry_price,
                "status": "OPEN",
                "mode": "LIVE (synced)",
                "is_demo": False,
                "condition_id": ticker,
                "ticker": ticker,
                "target_date": target_date,
                "unrealized_pnl": 0.0,
                "position_obj": None,
                "exchange_order_id": None,
                "shares": shares,
                "temp_low": temp_low,
                "temp_high": temp_high,
                "peak_pnl": 0.0,
            }

            st.session_state.open_positions.append(legacy_position)
            added += 1
            cost = shares * entry_price
            add_alert(f"Added: {city.upper()} {outcome_desc} | {shares} {side} @ ${entry_price:.2f} = ${cost:.2f}", "success")

    # Run the async add function
    if any(ticker not in dashboard_tickers for ticker in kalshi_weather):
        try:
            run_async(_add_new_positions())
        except Exception as e:
            add_alert(f"Error adding positions: {e}", "warning")

    st.session_state.positions_synced = True
    return added, removed, updated


# ========================
# PROBABILITY CLIPPING & CONTINUITY CORRECTION
# ========================
# No weather outcome should be treated as 0% or 100% certain.
# Weather has fat tails - freak events happen. Clipping prevents
# the model from being catastrophically overconfident on extremes,
# which would cause outsized Kelly positions on tail bets.
PROB_FLOOR = 0.01    # 1% minimum - never say "impossible"
PROB_CEILING = 0.99  # 99% maximum - never say "guaranteed"

def clip_probability(p: float) -> float:
    """Clip probability to [PROB_FLOOR, PROB_CEILING].

    Prevents tail overconfidence: a raw CDF of 0.001% becomes 1%.
    This is standard practice in prediction markets and sports betting
    to guard against model error in the distribution tails.
    """
    return max(PROB_FLOOR, min(PROB_CEILING, p))


def calc_outcome_probability(
    temp_low: float,
    temp_high: float,
    forecast_mean: float,
    forecast_std: float,
) -> float:
    """Calculate probability for a market outcome with continuity correction.

    Weather markets settle on INTEGER temperatures (e.g., "8°C" means the
    station reported 8). When modeling discrete integer outcomes with a
    continuous normal distribution, we must apply continuity correction:

        "8°C" (single temp) → P(7.5 ≤ T < 8.5)
        "20-21°F" (range)   → P(19.5 ≤ T < 21.5)
        "≤6°C" (or below)   → P(T < 6.5)
        "≥12°C" (or higher) → P(T ≥ 11.5)

    Without this, single-degree outcomes like "8°C" compute CDF(8)-CDF(8)=0,
    which is mathematically correct for a point but wrong for the market's
    intent (an integer reading of 8).

    Returns clipped probability, or None if inputs are invalid.
    """
    CONTINUITY = 0.5

    if temp_low is None and temp_high is not None:
        # "≤X" - P(temp rounds to X or below)
        raw_prob = stats.norm.cdf(temp_high + CONTINUITY, loc=forecast_mean, scale=forecast_std)
    elif temp_high is None and temp_low is not None:
        # "≥X" - P(temp rounds to X or above)
        raw_prob = 1 - stats.norm.cdf(temp_low - CONTINUITY, loc=forecast_mean, scale=forecast_std)
    elif temp_low is not None and temp_high is not None:
        # Range or single temp - P(low ≤ reading ≤ high)
        raw_prob = stats.norm.cdf(temp_high + CONTINUITY, loc=forecast_mean, scale=forecast_std) - \
                   stats.norm.cdf(temp_low - CONTINUITY, loc=forecast_mean, scale=forecast_std)
    else:
        return None

    return clip_probability(raw_prob)


def run_async(coro):
    """Helper to run async functions in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def add_alert(message: str, level: str = "info"):
    """Add an alert to the alerts panel."""
    st.session_state.alerts.insert(0, {
        "time": get_est_now(),
        "message": message,
        "level": level
    })
    st.session_state.alerts = st.session_state.alerts[:50]


def record_price(market_id: str, price: float):
    """Record a price point for historical tracking."""
    if market_id not in st.session_state.price_history:
        st.session_state.price_history[market_id] = []
    st.session_state.price_history[market_id].append({
        "time": get_est_now(),
        "price": price
    })


def _simulate_price_movement(base_price: float, condition_id: str) -> float:
    """
    Bug #13 fix: Simulate realistic price movements for demo mode.

    Uses mean reversion + random walk to create realistic price dynamics.
    Prices tend to drift toward fair value but with noise.
    """
    # Get previous price if exists, otherwise use base
    price_history = st.session_state.get("price_history", {})
    prev_price = price_history.get(condition_id, base_price)

    # Mean reversion factor (pull toward fair value)
    mean_reversion = 0.1 * (base_price - prev_price)

    # Random walk component (market noise)
    random_walk = random.gauss(0, 0.02)  # 2% std dev

    # Momentum component (slight trend continuation)
    momentum = 0.3 * (prev_price - price_history.get(f"{condition_id}_prev", prev_price))

    # Calculate new price
    new_price = prev_price + mean_reversion + random_walk + momentum

    # Clamp to valid range
    new_price = max(0.01, min(0.99, new_price))

    # Store for next iteration
    if "price_history" not in st.session_state:
        st.session_state.price_history = {}
    st.session_state.price_history[f"{condition_id}_prev"] = prev_price
    st.session_state.price_history[condition_id] = new_price

    return new_price


def generate_demo_markets(forecasts):
    """Generate demo markets based on real forecasts for testing."""
    markets = []
    target_date = today_est() + timedelta(days=1)

    for city_key in get_all_cities():
        if city_key not in forecasts:
            continue
        fc = forecasts[city_key]
        city_config = get_city_config(city_key)
        forecast_high = fc["high_mean"]
        forecast_std = max(fc["high_std"], 2.0)

        # Create temperature ranges around the forecast
        base_temp = round(forecast_high / 2) * 2  # Round to nearest 2
        ranges = [
            (None, base_temp - 4),           # "≤X" (low end)
            (base_temp - 4, base_temp - 2),  # Range below
            (base_temp - 2, base_temp),      # Range around
            (base_temp, base_temp + 2),      # Range around
            (base_temp + 2, base_temp + 4),  # Range above
            (base_temp + 4, None),           # "≥X" (high end)
        ]

        for temp_low, temp_high in ranges:
            # Calculate fair probability for this range
            if temp_low is None:
                fair_prob = stats.norm.cdf(temp_high, loc=forecast_high, scale=forecast_std)
                outcome_desc = f"{temp_high}°F or below"
            elif temp_high is None:
                fair_prob = 1 - stats.norm.cdf(temp_low, loc=forecast_high, scale=forecast_std)
                outcome_desc = f"{temp_low}°F or higher"
            else:
                prob_high = stats.norm.cdf(temp_high, loc=forecast_high, scale=forecast_std)
                prob_low = stats.norm.cdf(temp_low, loc=forecast_high, scale=forecast_std)
                fair_prob = prob_high - prob_low
                outcome_desc = f"{temp_low}-{temp_high}°F"

            condition_id = f"demo_{city_key}_{target_date}_{temp_low}_{temp_high}"

            # Bug #13 fix: Use dynamic price simulation instead of static noise
            market_prob = _simulate_price_movement(fair_prob, condition_id)

            markets.append({
                "city": city_key,
                "city_name": fc["city"],
                "question": f"Highest temperature in {fc['city']} on {target_date}?",
                "outcome_desc": outcome_desc,
                "temp_low": temp_low,
                "temp_high": temp_high,
                "temp_midpoint": (temp_low if temp_low else temp_high - 1) if temp_high is None else
                                 (temp_high if temp_high else temp_low + 1) if temp_low is None else
                                 (temp_low + temp_high) / 2,
                "yes_price": market_prob,
                "no_price": 1 - market_prob,
                "volume": random.randint(1000, 20000),
                "liquidity": random.randint(100, 2000),
                "target_date": target_date,
                "temp_unit": "F",
                "condition_id": condition_id,
                "ticker": condition_id,
                "resolution_source": f"Demo - {city_config.station_name}",
                "is_demo": True,
                "is_same_day": False,  # Demo markets are always for tomorrow
                "city_config": city_config,
                "fair_value": fair_prob,  # Store fair value for reference
            })

            record_price(condition_id, market_prob)

    return markets


@st.cache_data(ttl=600)
def fetch_multi_day_forecasts():
    """Fetch 5-day forecasts for all cities."""
    return run_async(_fetch_multi_day())


async def _fetch_multi_day():
    """Fetch multi-day forecasts."""
    forecasts = {}

    async with OpenMeteoClient() as client:
        for city_key in get_all_cities():
            city_config = get_city_config(city_key)
            try:
                daily = await client.get_daily_forecast(city_config, days=7)
                forecasts[city_key] = {
                    "city": city_config.name,
                    "daily": [
                        {
                            "date": f.timestamp.date(),
                            "high": f.temperature_high,
                            "low": f.temperature_low,
                            "model": f.model
                        }
                        for f in daily
                    ]
                }
            except Exception as e:
                pass

    return forecasts


@st.cache_data(ttl=600)
@st.cache_data(ttl=600)  # Cache for 10 minutes to avoid rate limiting
def fetch_model_comparison():
    """Fetch forecasts from different models for comparison."""
    return run_async(_fetch_model_comparison())


async def _fetch_model_comparison():
    """Fetch from Open-Meteo models for comparison.

    Note: Tomorrow.io data is NOT fetched here to avoid rate-limit conflicts.
    Instead, Tomorrow.io data is merged from fetch_forecasts_with_models()
    in the Model Comparison tab display code.
    """
    from weather_trader.apis.open_meteo import WeatherModel

    comparisons = {}
    today = today_est()
    tomorrow = today_est() + timedelta(days=1)

    # Models to compare - include HRRR for US cities
    all_models = [WeatherModel.ECMWF, WeatherModel.GFS, WeatherModel.HRRR, WeatherModel.BEST_MATCH]

    async with OpenMeteoClient() as om_client:
        for city_key in get_all_cities():
            city_config = get_city_config(city_key)
            city_data = {
                "city": city_config.name,
                "today": {"date": today, "models": {}},
                "tomorrow": {"date": tomorrow, "models": {}},
            }

            for model in all_models:
                # Skip HRRR for non-US cities
                if model == WeatherModel.HRRR and city_config.country != "US":
                    continue

                try:
                    forecasts = await om_client.get_daily_forecast(city_config, model, days=3)
                    for f in forecasts:
                        if f.timestamp.date() == today:
                            city_data["today"]["models"][model.value] = {
                                "high": f.temperature_high,
                                "low": f.temperature_low
                            }
                        elif f.timestamp.date() == tomorrow:
                            city_data["tomorrow"]["models"][model.value] = {
                                "high": f.temperature_high,
                                "low": f.temperature_low
                            }
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"Model comparison error for {city_key}/{model.value}: {e}")
                    # Continue with other models even if one fails

            comparisons[city_key] = city_data

    return comparisons


@st.cache_data(ttl=300)  # 5 minutes - increased to reduce API calls
def fetch_real_markets():
    """Fetch real Kalshi markets."""
    return run_async(_fetch_real_markets())


async def _fetch_real_markets():
    """Fetch weather markets from Kalshi, including same-day markets with uncertainty."""
    markets = []

    async with KalshiMarketFinder() as finder:
        try:
            # Include same-day markets - they will be filtered by uncertainty checker
            found = await finder.find_weather_markets(
                active_only=True,
                days_ahead=3,
                include_same_day=True  # Allow same-day markets through
            )
            add_alert(f"Kalshi API: Found {len(found)} market events", "info")

            for market in found:
                is_same_day = market.is_same_day
                add_alert(f"Processing {market.city}: {len(market.brackets)} brackets (same_day={is_same_day})", "info")
                for bracket in market.brackets:
                    markets.append({
                        "city": market.city,
                        "city_name": market.city_config.name if market.city_config else market.city,
                        "question": market.question,
                        "outcome_desc": bracket.description,
                        "temp_low": bracket.temp_low,
                        "temp_high": bracket.temp_high,
                        "temp_midpoint": bracket.midpoint,
                        "yes_price": bracket.yes_price,
                        "no_price": 1 - bracket.yes_price,
                        "volume": bracket.volume,
                        "liquidity": bracket.open_interest,
                        "target_date": market.target_date,
                        "temp_unit": "F",
                        "condition_id": bracket.ticker,
                        "ticker": bracket.ticker,
                        "resolution_source": market.resolution_source,
                        "is_demo": False,
                        "is_same_day": is_same_day,  # Flag for uncertainty check
                        "event_ticker": market.event_ticker,
                        "total_market_volume": market.total_volume,
                        "city_config": market.city_config,  # For uncertainty checker
                    })
                    record_price(bracket.ticker, bracket.yes_price)
        except Exception as e:
            import traceback
            add_alert(f"Error fetching markets: {str(e)}", "danger")
            add_alert(f"Traceback: {traceback.format_exc()[:200]}", "warning")

    add_alert(f"Total outcomes fetched: {len(markets)}", "info")
    return markets


@st.cache_data(ttl=300)  # 5 min cache (reduced from 10)
def fetch_forecasts_with_models():
    """Fetch forecasts with individual model data."""
    try:
        result = run_async(_fetch_forecasts_with_models())
        if not result:
            add_alert("Warning: No forecasts returned from API - may be rate limited", "warning")
        return result
    except Exception as e:
        add_alert(f"Forecast fetch error: {e}", "error")
        return {}


# Tomorrow.io API rate limiting - Module-level variables
# (Must be module-level, NOT session state, to work inside @st.cache_data functions)
_tomorrow_io_last_call: Optional[datetime] = None
_tomorrow_io_calls_today: int = 0
_tomorrow_io_call_date: date = today_est()


def can_call_tomorrow_io() -> bool:
    """Check if we can make a Tomorrow.io API call (rate limiting).

    Strategy: 500 calls/day, each cache miss uses ~5 calls (one per city).
    With 10-minute cache TTL = max 6 cache misses/hour = 30 calls/hour.
    Over 16 hours = 480 calls, within the 500 limit.

    No per-call cooldown needed since @st.cache_data(ttl=600) already
    prevents calls more than once per 10 minutes.
    """
    global _tomorrow_io_calls_today, _tomorrow_io_call_date

    # Reset counter if it's a new day
    if _tomorrow_io_call_date != today_est():
        _tomorrow_io_calls_today = 0
        _tomorrow_io_call_date = today_est()

    # 500 calls/day, use 480 (leave 20 buffer for manual refreshes)
    if _tomorrow_io_calls_today >= 480:
        return False

    return True


def record_tomorrow_io_call(count: int = 1):
    """Record Tomorrow.io API calls. Count = number of city calls made."""
    global _tomorrow_io_last_call, _tomorrow_io_calls_today
    _tomorrow_io_last_call = datetime.now()
    _tomorrow_io_calls_today += count


def get_tomorrow_io_usage() -> dict:
    """Get Tomorrow.io API usage stats for display."""
    global _tomorrow_io_calls_today, _tomorrow_io_call_date
    if _tomorrow_io_call_date != today_est():
        _tomorrow_io_calls_today = 0
        _tomorrow_io_call_date = today_est()
    return {
        "calls_today": _tomorrow_io_calls_today,
        "daily_limit": 500,
        "budget": 480,
        "last_call": _tomorrow_io_last_call,
        "can_call": can_call_tomorrow_io(),
    }


async def _fetch_forecasts_with_models():
    """Fetch forecasts for all cities with model breakdown, including Tomorrow.io.

    Generates forecasts for today through +3 days to match the market date range
    from find_weather_markets(days_ahead=3). Stores date-specific entries
    (e.g. forecasts["nyc_2026-01-28"]) for signal calculation, plus a backward-compat
    city-level entry (forecasts["nyc"]) pointing to tomorrow's forecast for visualization.
    """
    forecasts = {}
    target_dates = [today_est() + timedelta(days=d) for d in range(4)]  # today..+3
    tomorrow = today_est() + timedelta(days=1)

    # Check if Tomorrow.io is available
    use_tomorrow_io = (
        config.api.tomorrow_io_api_key and
        "your_" not in config.api.tomorrow_io_api_key.lower() and
        can_call_tomorrow_io()
    )

    async with OpenMeteoClient() as om_client:
        bias_corrector = BiasCorrector()
        ensemble = EnsembleForecaster(bias_corrector)

        # Optionally create Tomorrow.io client
        tm_client = None
        tm_call_count = 0
        if use_tomorrow_io:
            tm_client = TomorrowIOClient()
            await tm_client.__aenter__()

        try:
            for city_key in get_all_cities():
                city_config = get_city_config(city_key)
                try:
                    om_forecasts = await om_client.get_ensemble_forecast(city_config, days=5)

                    # Fetch Tomorrow.io once per city (outside date loop)
                    tm_forecasts_list = None
                    if tm_client:
                        try:
                            tm_forecasts_list = await tm_client.get_daily_forecast(city_config, days=5)
                            tm_call_count += 1
                        except Exception as e:
                            print(f"Tomorrow.io error for {city_key}: {e}")

                    # Lazy-loaded fallback for basic forecast (fetched at most once per city)
                    basic_forecasts = None
                    basic_forecasts_fetched = False

                    for target_date in target_dates:
                        model_forecasts = []
                        model_details = []

                        # Add Open-Meteo models (exact date match only)
                        for model_name, forecast_list in om_forecasts.items():
                            for f in forecast_list:
                                if f.timestamp.date() == target_date:
                                    if f.temperature_high is not None and f.temperature_low is not None:
                                        model_forecasts.append(ModelForecast(
                                            model_name=model_name,
                                            forecast_high=f.temperature_high,
                                            forecast_low=f.temperature_low,
                                        ))
                                        model_details.append({
                                            "model": model_name,
                                            "high": f.temperature_high,
                                            "low": f.temperature_low
                                        })
                                    else:
                                        print(f"{city_key}/{model_name}: date matched but temps are None")
                                    break

                        # Fallback: if ensemble returned nothing for this date, try basic forecast
                        if not model_forecasts:
                            if not basic_forecasts_fetched:
                                basic_forecasts_fetched = True
                                print(f"{city_key}: ensemble empty for {target_date}, trying basic forecast")
                                try:
                                    from weather_trader.apis.open_meteo import WeatherModel
                                    basic_forecasts = await om_client.get_daily_forecast(
                                        city_config, WeatherModel.BEST_MATCH, days=5)
                                except Exception as fallback_err:
                                    print(f"{city_key}: basic forecast fallback failed: {fallback_err}")
                                    basic_forecasts = []

                            if basic_forecasts:
                                for f in basic_forecasts:
                                    if f.timestamp.date() == target_date and f.temperature_high is not None and f.temperature_low is not None:
                                        model_forecasts = [ModelForecast(
                                            model_name="best_match",
                                            forecast_high=f.temperature_high,
                                            forecast_low=f.temperature_low,
                                        )]
                                        model_details = [{
                                            "model": "best_match",
                                            "high": f.temperature_high,
                                            "low": f.temperature_low
                                        }]
                                        break

                        # Add Tomorrow.io for this date
                        if tm_forecasts_list:
                            for f in tm_forecasts_list:
                                if f.timestamp.date() == target_date:
                                    if f.temperature_high is not None and f.temperature_low is not None:
                                        model_forecasts.append(ModelForecast(
                                            model_name="tomorrow.io",
                                            forecast_high=f.temperature_high,
                                            forecast_low=f.temperature_low,
                                        ))
                                        model_details.append({
                                            "model": "tomorrow.io",
                                            "high": f.temperature_high,
                                            "low": f.temperature_low
                                        })
                                    break

                        if model_forecasts:
                            ens = ensemble.create_ensemble(
                                city_config, model_forecasts, target_date,
                                apply_bias_correction=False
                            )
                            entry = {
                                "city": city_config.name,
                                "high_mean": float(ens.high_mean),
                                "high_std": float(ens.high_std),
                                "low_mean": float(ens.low_mean),
                                "low_std": float(ens.low_std),
                                "confidence": float(ens.confidence),
                                "model_count": ens.model_count,
                                "date": target_date,
                                "models": model_details,
                                "high_ci_lower": float(ens.high_ci_lower),
                                "high_ci_upper": float(ens.high_ci_upper),
                            }

                            # Date-specific key for signal calculation
                            forecasts[f"{city_key}_{target_date.isoformat()}"] = entry

                            # Backward-compat city key = tomorrow's forecast
                            if target_date == tomorrow:
                                forecasts[city_key] = entry

                            print(f"[Forecast] {city_key} ({target_date}): high={ens.high_mean:.1f}F std={ens.high_std:.1f} ({ens.model_count} models)")
                        else:
                            print(f"WARNING: No forecast data for {city_key} on {target_date}")

                except Exception as e:
                    print(f"ERROR fetching forecast for {city_key}: {e}")
        finally:
            if tm_client:
                await tm_client.__aexit__(None, None, None)
                if tm_call_count > 0:
                    record_tomorrow_io_call(tm_call_count)
                    print(f"Tomorrow.io: {tm_call_count} city calls (#{_tomorrow_io_calls_today} today)")

    city_count = len([k for k in forecasts if '_' not in k or not k.split('_')[-1][:4].isdigit()])
    date_count = len([k for k in forecasts if '_' in k and k.split('_')[-1][:4].isdigit()])
    print(f"[Forecasts] Loaded {city_count} cities, {date_count} date-specific entries: {list(forecasts.keys())}")
    return forecasts


def check_same_day_uncertainty(markets, ensemble_forecasts=None):
    """
    Check same-day markets for genuine remaining uncertainty.

    Returns a tuple of:
    - uncertainty_map: dict mapping ticker -> SameDayUncertainty assessment
    - same_day_forecasts: dict mapping city_key -> blended forecast for today

    The same_day_forecasts blends NWS observations with ensemble forecasts
    based on time of day:
    - Early morning (midnight-8am): 80% ensemble, 20% NWS
    - Morning (8am-noon): 50% ensemble, 50% NWS
    - Afternoon (noon-6pm): 20% ensemble, 80% NWS
    - Evening (6pm+): 100% NWS (observations are ground truth)
    """
    return run_async(_check_same_day_uncertainty_async(markets, ensemble_forecasts))


async def _check_same_day_uncertainty_async(markets, ensemble_forecasts=None):
    """Async implementation of same-day uncertainty checking.

    Returns:
        Tuple of (uncertainty_map, same_day_forecasts)
        - uncertainty_map: ticker -> SameDayUncertainty
        - same_day_forecasts: city_key -> dict with blended forecast for today

    Blending weights by time of day (EST):
    - 00:00-08:00: 80% ensemble, 20% NWS (day hasn't started)
    - 08:00-12:00: 50% ensemble, 50% NWS (warming phase)
    - 12:00-18:00: 20% ensemble, 80% NWS (peak time, observations reliable)
    - 18:00-24:00: 0% ensemble, 100% NWS (day over, observations are truth)
    """
    from weather_trader.kalshi.markets import TemperatureBracket

    uncertainty_map = {}
    same_day_forecasts = {}
    checker = SameDayTradingChecker()

    # Calculate time-based weights for blending
    now_est = get_est_now()
    hour = now_est.hour

    if hour < 8:
        # Early morning: trust ensemble more (day hasn't started)
        ensemble_weight = 0.80
        nws_weight = 0.20
        blend_reason = "early morning (ensemble-heavy)"
    elif hour < 12:
        # Morning: blend evenly (warming phase)
        ensemble_weight = 0.50
        nws_weight = 0.50
        blend_reason = "morning (balanced blend)"
    elif hour < 18:
        # Afternoon: trust NWS more (near/past peak)
        ensemble_weight = 0.20
        nws_weight = 0.80
        blend_reason = "afternoon (NWS-heavy)"
    else:
        # Evening: NWS only (day essentially over)
        ensemble_weight = 0.0
        nws_weight = 1.0
        blend_reason = "evening (NWS only)"

    # Group same-day markets by city to batch the NWS calls
    # CRITICAL: Compute is_same_day dynamically from target_date, not cached flag
    # This ensures consistency with calculate_signals() which also computes dynamically
    same_day_by_city = {}
    for market in markets:
        target_date = market.get("target_date", today_est())
        if hasattr(target_date, 'date'):
            target_date = target_date.date()
        is_same_day = target_date == today_est()

        if is_same_day:
            city_key = market["city"].lower()
            if city_key not in same_day_by_city:
                same_day_by_city[city_key] = []
            same_day_by_city[city_key].append(market)

    if not same_day_by_city:
        return uncertainty_map, same_day_forecasts

    # Process each city
    for city_key, city_markets in same_day_by_city.items():
        try:
            city_config = city_markets[0].get("city_config")
            if not city_config:
                city_config = get_city_config(city_key)

            # Fetch current conditions once per city
            current_high, remaining_high = await checker.get_current_conditions(city_config)

            # BUILD BLENDED FORECAST FOR SAME-DAY TRADING
            # Combines NWS observations with ensemble forecast based on time of day

            # Get NWS-based expected high
            if current_high is not None:
                if remaining_high is not None:
                    nws_expected_high = max(current_high, remaining_high)
                    nws_uncertainty = abs(remaining_high - current_high)
                    nws_std = max(1.5, min(3.0, nws_uncertainty / 2))
                else:
                    nws_expected_high = current_high
                    nws_std = 1.0
            else:
                nws_expected_high = None
                nws_std = 3.0

            # Get ensemble forecast for today
            today_key = f"{city_key}_{today_est().isoformat()}"
            ensemble_fc = None
            if ensemble_forecasts:
                ensemble_fc = ensemble_forecasts.get(today_key)

            ensemble_high = ensemble_fc.get("high_mean") if ensemble_fc else None
            ensemble_std = ensemble_fc.get("high_std", 3.0) if ensemble_fc else 3.0

            # Blend forecasts based on time-of-day weights
            if nws_expected_high is not None and ensemble_high is not None:
                # Both available - blend them
                blended_high = (ensemble_weight * ensemble_high) + (nws_weight * nws_expected_high)
                blended_std = (ensemble_weight * ensemble_std) + (nws_weight * nws_std)
                source = f"blended ({ensemble_weight:.0%} ensemble + {nws_weight:.0%} NWS)"
                model_count = (ensemble_fc.get("model_count", 1) if ensemble_fc else 0) + 1
            elif nws_expected_high is not None:
                # Only NWS available
                blended_high = nws_expected_high
                blended_std = nws_std
                source = "NWS_only"
                model_count = 1
            elif ensemble_high is not None:
                # Only ensemble available
                blended_high = ensemble_high
                blended_std = ensemble_std
                source = "ensemble_only"
                model_count = ensemble_fc.get("model_count", 1) if ensemble_fc else 1
            else:
                # Neither available - skip
                continue

            # Confidence based on blend quality
            if nws_expected_high is not None and ensemble_high is not None:
                # Both sources agree closely = higher confidence
                agreement = abs(ensemble_high - nws_expected_high)
                if agreement < 2:
                    confidence = 0.90
                elif agreement < 4:
                    confidence = 0.80
                else:
                    confidence = 0.70
            else:
                confidence = 0.75

            same_day_forecasts[city_key] = {
                "city": city_config.name,
                "high_mean": blended_high,
                "high_std": blended_std,
                "low_mean": blended_high - 10,
                "low_std": 3.0,
                "confidence": confidence,
                "model_count": model_count,
                "date": today_est(),
                "source": source,
                "current_high": current_high,
                "remaining_high": remaining_high,
                "ensemble_high": ensemble_high,
                "nws_high": nws_expected_high,
                "blend_weights": {"ensemble": ensemble_weight, "nws": nws_weight},
            }
            add_alert(
                f"Same-day {city_key}: {blended_high:.1f}°F ({blend_reason}) "
                f"[NWS: {nws_expected_high}, Ensemble: {ensemble_high}]",
                "info"
            )

            for market in city_markets:
                ticker = market.get("ticker", market.get("condition_id", ""))

                # Create a TemperatureBracket for the checker
                bracket = TemperatureBracket(
                    ticker=ticker,
                    event_ticker=market.get("event_ticker", ""),
                    description=market.get("outcome_desc", ""),
                    temp_low=market.get("temp_low"),
                    temp_high=market.get("temp_high"),
                    yes_price_cents=int(market.get("yes_price", 0.5) * 100),
                )

                # Check uncertainty for this bracket
                uncertainty = await checker.check_bracket_uncertainty(
                    bracket, city_config, current_high, remaining_high
                )
                uncertainty_map[ticker] = uncertainty

                # Log for debugging
                if uncertainty.has_uncertainty:
                    add_alert(
                        f"Same-day {city_key}: {market.get('outcome_desc')} has uncertainty "
                        f"(current: {current_high}°F, {uncertainty.reason})",
                        "info"
                    )
                else:
                    add_alert(
                        f"Same-day {city_key}: {market.get('outcome_desc')} outcome determined - {uncertainty.reason}",
                        "warning"
                    )

        except Exception as e:
            add_alert(f"Error checking same-day uncertainty for {city_key}: {e}", "warning")

    return uncertainty_map, same_day_forecasts


def calculate_signals(forecasts, markets, show_all_outcomes=False):
    """Calculate trading signals for multi-outcome markets.

    Args:
        forecasts: Weather forecast data
        markets: Market data from Kalshi
        show_all_outcomes: If True, return signals for ALL outcomes, not just the best

    Same-day markets are now supported with uncertainty-based filtering:
    - Markets where the outcome is already determined are skipped
    - Markets with genuine remaining uncertainty can be traded
    - Confidence is adjusted based on the uncertainty level
    - Same-day forecasts use NWS observations (more accurate than weather models)
    """
    signals = []

    # Pre-check same-day market uncertainty
    # This fetches NWS observations and blends with ensemble based on time of day
    same_day_uncertainty, same_day_forecasts = check_same_day_uncertainty(markets, forecasts)

    # Group markets by city and date
    market_groups = {}
    for market in markets:
        city_key = market["city"].lower()
        target_date = market.get("target_date", today_est())
        # Normalize to date object (in case it's a datetime)
        if hasattr(target_date, 'date'):
            target_date = target_date.date()
        group_key = f"{city_key}_{target_date.isoformat()}"
        if group_key not in market_groups:
            market_groups[group_key] = []
        market_groups[group_key].append(market)

    for group_key, group_markets in market_groups.items():
        if not group_markets:
            continue

        city_key = group_markets[0]["city"].lower()
        target_date = group_markets[0].get("target_date", today_est())
        # Normalize to date object (in case it's a datetime)
        if hasattr(target_date, 'date'):
            target_date = target_date.date()
        is_same_day = target_date == today_est()

        # For same-day markets, we now allow trading IF there's genuine uncertainty
        # Skip only past dates (before today)
        if target_date < today_est():
            continue

        date_key = f"{city_key}_{target_date.isoformat()}"

        # FOR SAME-DAY MARKETS: Use NWS observations-based forecast (more accurate)
        if is_same_day and city_key in same_day_forecasts:
            fc = same_day_forecasts[city_key]
            print(f"[Signal] Using blended same-day forecast for {city_key}: {fc['high_mean']:.1f}°F (std: {fc['high_std']:.1f})")
        elif is_same_day:
            # Same-day market but NWS forecast not available - try date-specific ensemble
            fc = forecasts.get(date_key)
            if fc:
                print(f"[Signal] Using date-specific ensemble for same-day {city_key}: {fc['high_mean']:.1f}°F (date: {fc.get('date')})")
            else:
                # No today forecast - try city key but verify date
                fc = forecasts.get(city_key)
                if fc and fc.get("date") != target_date:
                    print(f"[Signal] SKIPPING same-day {city_key} - no today forecast, city_key has {fc.get('date')}")
                    continue
                elif fc:
                    print(f"[Signal] Using city-key forecast for same-day {city_key}: {fc['high_mean']:.1f}°F")
        else:
            # Future market - use date-specific forecast
            fc = forecasts.get(date_key)
            if not fc:
                # Try city-only key but VERIFY the date matches
                fc = forecasts.get(city_key)
                if fc and fc.get("date") != target_date:
                    # Wrong date - do NOT use this forecast
                    print(f"[Signal] SKIPPING {city_key} {target_date} - forecast date mismatch (have {fc.get('date')})")
                    continue
        if not fc:
            continue
        forecast_temp_f = fc["high_mean"]  # Forecast is always in Fahrenheit
        forecast_std_f = max(fc["high_std"], 2.0)
        city_config = get_city_config(city_key)

        # All Kalshi markets are Fahrenheit
        forecast_temp_market = forecast_temp_f
        forecast_std_market = forecast_std_f

        best_signal = None
        best_edge = 0

        for market in group_markets:
            ticker = market.get("ticker", market.get("condition_id", ""))

            # SAME-DAY UNCERTAINTY CHECK
            # For same-day markets, only trade if there's genuine remaining uncertainty
            uncertainty_adjustment = 1.0  # Default: full confidence
            if is_same_day:
                uncertainty = same_day_uncertainty.get(ticker)
                if uncertainty is None:
                    # No uncertainty data - skip same-day market to be safe
                    continue
                if not uncertainty.has_uncertainty:
                    # Outcome is determined - skip this bracket
                    continue
                if uncertainty.outcome_determined:
                    # Outcome is known - skip
                    continue
                # Apply probability adjustment based on remaining uncertainty
                uncertainty_adjustment = uncertainty.probability_adjustment
                if uncertainty_adjustment < 0.3:
                    # Too little uncertainty remaining - not worth the risk
                    continue

            temp_low = market.get("temp_low")
            temp_high = market.get("temp_high")
            market_prob = market["yes_price"]

            # Calculate our probability with continuity correction + clipping
            if temp_low is None and temp_high is not None:
                range_type = "or_below"
            elif temp_high is None and temp_low is not None:
                range_type = "or_higher"
            elif temp_low is not None and temp_high is not None:
                range_type = "range"
            else:
                continue

            our_prob = calc_outcome_probability(
                temp_low, temp_high, forecast_temp_market, forecast_std_market
            )
            if our_prob is None:
                continue

            edge = our_prob - market_prob

            # For same-day markets, require higher edge to compensate for reduced forecast value
            if is_same_day:
                # Require 25% more edge for same-day trades
                effective_edge_threshold = 0.0625  # 6.25% instead of 5%
                if abs(edge) < effective_edge_threshold:
                    # Edge too small for same-day risk
                    continue

            # Determine signal type
            if edge > 0.10:
                signal_type = "STRONG BUY YES"
                signal_color = "#00c853"
            elif edge > 0.05:
                signal_type = "BUY YES"
                signal_color = "#4caf50"
            elif edge < -0.10:
                signal_type = "STRONG BUY NO"
                signal_color = "#f44336"
            elif edge < -0.05:
                signal_type = "BUY NO"
                signal_color = "#ff5722"
            else:
                signal_type = "PASS"
                signal_color = "#9e9e9e"

            # Adjust confidence for same-day trades based on remaining uncertainty
            base_confidence = fc["confidence"]
            if is_same_day:
                adjusted_confidence = base_confidence * uncertainty_adjustment
            else:
                adjusted_confidence = base_confidence

            signal_data = {
                "city": fc["city"],
                "city_key": city_key,
                "outcome": market.get("outcome_desc", f"{temp_low}-{temp_high}"),
                "temp_low": temp_low,
                "temp_high": temp_high,
                "temp_unit": "F",
                "range_type": range_type,
                "our_prob": our_prob,
                "market_prob": market_prob,
                "edge": edge,
                "confidence": adjusted_confidence,
                "base_confidence": base_confidence,
                "forecast_high_f": forecast_temp_f,
                "forecast_high_market": forecast_temp_market,
                "forecast_std": forecast_std_market,
                "condition_id": market["condition_id"],
                "ticker": market.get("ticker", ""),
                "is_demo": market.get("is_demo", False),
                "is_same_day": is_same_day,
                "uncertainty_adjustment": uncertainty_adjustment if is_same_day else 1.0,
                "target_date": market.get("target_date"),
                "resolution_source": market.get("resolution_source", ""),
                "volume": market.get("volume", 0),
                "signal": signal_type,
                "signal_color": signal_color,
                "signal_strength": abs(edge),
            }

            if show_all_outcomes:
                # Add all outcomes
                signals.append(signal_data)
            else:
                # Track best opportunity
                if abs(edge) > abs(best_edge):
                    best_edge = edge
                    best_signal = signal_data

        # If not showing all, add only the best signal
        if not show_all_outcomes and best_signal:
            signals.append(best_signal)
            if abs(best_signal["edge"]) > 0.10 and best_signal["confidence"] > 0.7:
                add_alert(f"🎯 Strong signal: {best_signal['city']} {best_signal['outcome']} - {best_signal['signal']} ({best_signal['edge']:+.1%} edge)", "success")

    return signals


def execute_trade(signal, size, is_live=False):
    """Execute a trade using the trading engine."""
    outcome = signal.get("outcome", "")
    side = "YES" if signal["edge"] > 0 else "NO"

    # IMPORTANT: Always store YES price for consistency
    # For YES positions: we pay market_prob per share
    # For NO positions: we pay (1 - market_prob) per share, but store YES price for tracking
    yes_price = signal["market_prob"]
    actual_cost_per_share = yes_price if side == "YES" else (1 - yes_price)
    shares = size / actual_cost_per_share

    # Store YES price as entry_price for consistent price tracking
    entry_price = yes_price

    # Determine settlement date
    target_date = signal.get("target_date", today_est() + timedelta(days=1))
    if isinstance(target_date, date) and not isinstance(target_date, datetime):
        settlement_date = datetime.combine(target_date, datetime.max.time())
    else:
        settlement_date = target_date

    # Create position using the PositionManager
    position = st.session_state.position_manager.create_position(
        market_id=signal.get("condition_id", f"demo_{signal['city']}"),
        condition_id=signal.get("condition_id", ""),
        city=signal["city"],
        outcome_description=outcome,
        settlement_date=settlement_date,
        side=side,
        entry_price=entry_price,
        shares=shares,
        forecast_prob=signal["our_prob"]
    )

    # Submit real order to Kalshi if live trading
    exchange_order_id = None
    is_demo_market = signal.get("is_demo", False)
    if is_live and st.session_state.kalshi_client is not None and not is_demo_market:
        try:
            ticker = signal.get("ticker", "") or signal.get("condition_id", "")
            if not ticker or ticker.startswith("demo_"):
                add_alert("Cannot submit order: invalid ticker (demo market?)", "warning")
            else:
                kalshi_side = "yes" if side == "YES" else "no"
                # Add offset to improve fill rate (pay slightly more to beat the spread)
                # Default: 2 cents more aggressive than fair value
                fill_offset = st.session_state.get("fill_offset_cents", 2)
                base_price = int(round(actual_cost_per_share * 100))
                price_cents = max(1, min(99, base_price + fill_offset))
                count = max(1, int(round(shares)))

                async def _place_order():
                    async with st.session_state.kalshi_client as client:
                        return await client.place_order(
                            ticker=ticker,
                            action="buy",
                            side=kalshi_side,
                            count=count,
                            price_cents=price_cents,
                        )

                result = run_async(_place_order())

                if result.success:
                    exchange_order_id = result.order_id
                    add_alert(f"Live order submitted: {exchange_order_id} ({ticker})", "success")
                else:
                    add_alert(f"Live order failed: {result.message}", "warning")

        except Exception as e:
            add_alert(f"Live order failed (position still tracked): {e}", "warning")

    # Also add to legacy open_positions for UI compatibility
    legacy_position = {
        "id": position.position_id,
        "open_time": position.entry_time,
        "city": signal["city"],
        "outcome": outcome,
        "side": side,
        "size": size,
        "shares": shares,  # Store shares for accurate tracking
        "entry_price": entry_price,
        "current_price": entry_price,
        "edge_at_entry": signal["edge"],
        "forecast_prob": signal["our_prob"],
        "market_prob_at_entry": signal["market_prob"],
        "status": "OPEN",
        "mode": "LIVE" if is_live else "SIMULATED",
        "is_demo": signal.get("is_demo", False),
        "condition_id": signal.get("condition_id", ""),
        "ticker": signal.get("ticker", "") or signal.get("condition_id", ""),
        "target_date": target_date,
        "unrealized_pnl": 0.0,
        "peak_pnl": 0.0,  # Initialize for trailing stop
        "position_obj": position,  # Link to actual Position object
        "exchange_order_id": exchange_order_id,
        "temp_low": signal.get("temp_low"),  # For edge recalculation
        "temp_high": signal.get("temp_high"),  # For edge recalculation
    }

    st.session_state.open_positions.insert(0, legacy_position)

    # Also add to trade history for record keeping
    trade = {**legacy_position, "time": position.entry_time, "price": entry_price}
    st.session_state.trade_history.insert(0, trade)
    st.session_state.total_trades += 1

    # Record trade in risk manager
    st.session_state.risk_manager.record_trade(signal.get("condition_id", ""), 0.0)

    mode = "LIVE" if is_live else "SIMULATED"
    add_alert(f"📈 [{mode}] Opened: {signal['city']} {outcome} {side} ${size:.2f} @ {entry_price:.2f} (Edge: {signal['edge']:+.1%})",
              "success" if is_live else "info")
    return legacy_position


def check_smart_exits(forecasts, markets):
    """
    Check all open positions for smart exit signals.

    Uses the trading engine's edge-based exit logic:
    - Edge exhausted (captured 75%+ of entry edge)
    - Edge reversed (turned negative)
    - Momentum shift (edge trending down while profitable)
    - Stop loss / trailing stop
    - Time decay (approaching settlement)
    """
    if not st.session_state.open_positions:
        return

    # Build market price lookup
    market_prices = {}
    for m in markets:
        key = m.get("condition_id", "")
        if key:
            market_prices[key] = m.get("yes_price", 0.5)

    # Get forecast probabilities
    positions_to_close = []

    # Edge exit threshold from session state or default
    min_edge_exit = st.session_state.get("min_edge_exit_pct", 0.05)

    for legacy_pos in st.session_state.open_positions:
        position = legacy_pos.get("position_obj")
        cid = legacy_pos.get("condition_id", "") or legacy_pos.get("ticker", "")
        city_key = legacy_pos.get("city", "").lower()

        # Get current market price
        current_price = market_prices.get(cid, legacy_pos.get("current_price", 0.5))
        legacy_pos["current_price"] = current_price

        # Get current forecast probability - recalculate from latest forecast data
        target_date = legacy_pos.get("target_date")
        if target_date and hasattr(target_date, 'isoformat'):
            date_key = f"{city_key}_{target_date.isoformat()}"
            fc = forecasts.get(date_key) or forecasts.get(city_key, {})
        else:
            fc = forecasts.get(city_key, {})

        forecast_prob = legacy_pos.get("forecast_prob", 0.5)
        if fc:
            # Recalculate probability using current forecast and position's temp bracket
            temp_low = legacy_pos.get("temp_low")
            temp_high = legacy_pos.get("temp_high")
            if temp_low is not None or temp_high is not None:
                forecast_mean = fc.get("high_mean", 50)
                forecast_std = max(fc.get("high_std", 3.0), 2.0)
                recalc_prob = calc_outcome_probability(temp_low, temp_high, forecast_mean, forecast_std)
                if recalc_prob is not None:
                    forecast_prob = recalc_prob
                    legacy_pos["forecast_prob"] = forecast_prob  # Update stored value

        # Calculate current edge
        side = legacy_pos.get("side", "YES")
        if side == "YES":
            current_edge = forecast_prob - current_price
        else:
            current_edge = (1 - forecast_prob) - (1 - current_price)

        legacy_pos["current_edge"] = current_edge

        # Calculate unrealized P/L
        # Bug #7 fix: Handle NO positions correctly
        # For synced positions from Kalshi, entry_price is stored as the actual cost paid:
        # - YES: entry_price = YES price paid
        # - NO: entry_price = NO price paid (= 1 - YES price at entry)
        entry_price = legacy_pos.get("entry_price", current_price)
        shares = legacy_pos.get("shares", 1)
        if side == "YES":
            # YES: bought at entry_price, current value is current_price
            unrealized_pnl = (current_price - entry_price) * shares
        else:
            # NO: bought at entry_price (NO cost), current value is (1 - current_price)
            # For synced positions, entry_price is the NO price we paid
            current_no_value = 1 - current_price
            unrealized_pnl = (current_no_value - entry_price) * shares
        legacy_pos["unrealized_pnl"] = unrealized_pnl

        # Track peak P/L for trailing stop
        peak_pnl = legacy_pos.get("peak_pnl", unrealized_pnl)
        if unrealized_pnl > peak_pnl:
            legacy_pos["peak_pnl"] = unrealized_pnl
            peak_pnl = unrealized_pnl

        should_exit = False
        exit_reason = None

        # For positions with Position objects, use full exit logic
        if position and position.status == PositionStatus.OPEN:
            st.session_state.position_manager.update_position(
                position.position_id,
                market_price=current_price,
                forecast_prob=forecast_prob
            )
            should_exit, exit_reason = st.session_state.position_manager.should_exit_position(position)

        # For synced positions (no Position object), use simplified exit logic
        else:
            entry_price = legacy_pos.get("entry_price", 0.5)
            size = legacy_pos.get("size", 1)
            # Only apply edge-based exits if we have temp bracket data to calculate edge
            has_temp_data = legacy_pos.get("temp_low") is not None or legacy_pos.get("temp_high") is not None

            # 1. EDGE TOO LOW - Exit if edge below threshold (requires temp data)
            if has_temp_data and abs(current_edge) < min_edge_exit:
                should_exit = True
                exit_reason = ExitReason.EDGE_EXHAUSTED

            # 2. EDGE REVERSED - Exit if edge turned negative (requires temp data)
            elif has_temp_data and current_edge < 0:
                should_exit = True
                exit_reason = ExitReason.EDGE_REVERSED

            # 3. STOP LOSS - Exit if loss > 30% (always applies)
            elif unrealized_pnl < 0:
                pnl_pct = unrealized_pnl / size if size > 0 else 0
                if pnl_pct < -0.30:
                    should_exit = True
                    exit_reason = ExitReason.STOP_LOSS

            # 4. TRAILING STOP - Exit if gave back 20%+ of peak profit (always applies)
            elif peak_pnl > 0 and unrealized_pnl > 0:
                drawdown = (peak_pnl - unrealized_pnl) / peak_pnl
                if drawdown > 0.20:
                    should_exit = True
                    exit_reason = ExitReason.TRAILING_STOP

        if should_exit and exit_reason:
            positions_to_close.append((legacy_pos, position, exit_reason, current_price))

    # Close positions that triggered exit signals
    for legacy_pos, position, exit_reason, exit_price in positions_to_close:
        close_position_smart(legacy_pos["id"], exit_price, exit_reason)


def close_position(position_id: str, current_price: float):
    """Close an open position manually and realize P/L."""
    return close_position_smart(position_id, current_price, ExitReason.MANUAL)


def close_position_smart(position_id: str, current_price: float, exit_reason: ExitReason):
    """Close an open position with specified exit reason and realize P/L."""
    for i, pos in enumerate(st.session_state.open_positions):
        if pos["id"] == position_id:
            # Submit SELL order to Kalshi if live trading
            ticker = pos.get("ticker") or pos.get("condition_id", "")
            is_live_position = pos.get("mode", "").startswith("LIVE")

            if is_live_position and ticker and not ticker.startswith("demo_") and st.session_state.kalshi_client is not None:
                try:
                    # Determine sell parameters
                    shares = pos.get("shares") or int(pos.get("size", 1) / max(0.01, pos.get("entry_price", 0.5)))
                    original_side = pos.get("side", "YES")

                    # To close: sell what we bought
                    # If we bought YES, we sell YES
                    # If we bought NO, we sell NO
                    kalshi_side = "yes" if original_side == "YES" else "no"

                    # Use current market price with fill offset for faster exit
                    fill_offset = st.session_state.get("fill_offset_cents", 2)
                    # For selling, we go BELOW market to fill faster (accept less)
                    price_cents = max(1, min(99, int(current_price * 100) - fill_offset))

                    async def _sell_order():
                        async with st.session_state.kalshi_client as client:
                            return await client.place_order(
                                ticker=ticker,
                                action="sell",
                                side=kalshi_side,
                                count=shares,
                                price_cents=price_cents,
                            )

                    result = run_async(_sell_order())

                    if result.success:
                        add_alert(f"SELL order submitted: {ticker} x{shares} @ {price_cents}c ({exit_reason.value})", "success")
                    else:
                        add_alert(f"SELL order issue: {result.message}", "warning")

                except Exception as e:
                    add_alert(f"SELL order failed: {e}", "error")

            # Get Position object for accurate P/L calculation
            position = pos.get("position_obj")

            # Calculate P/L using Position object if available (more accurate)
            if position:
                # Update position with final price before calculating P/L
                position.current_price = current_price
                pnl = position.unrealized_pnl
            else:
                # Fallback legacy calculation
                # Note: entry_price and current_price are both YES prices
                if pos["side"] == "YES":
                    pnl = pos["size"] * (current_price - pos["entry_price"])
                else:
                    # For NO: profit = (entry_YES - current_YES) * shares
                    # shares = size / (1 - entry_YES)
                    no_price_at_entry = 1 - pos["entry_price"]
                    shares = pos["size"] / no_price_at_entry if no_price_at_entry > 0 else 0
                    pnl = (pos["entry_price"] - current_price) * shares
            if position:
                st.session_state.position_manager.close_position(
                    position_id=position.position_id,
                    exit_price=current_price,
                    reason=exit_reason
                )
                # Record in P&L tracker
                st.session_state.pnl_tracker.record_trade(position)

            # Move to closed positions
            closed = {**pos}
            closed["close_time"] = datetime.now()
            closed["exit_price"] = current_price
            closed["realized_pnl"] = pnl
            closed["status"] = "CLOSED"
            closed["exit_reason"] = exit_reason.value

            st.session_state.closed_positions.insert(0, closed)
            st.session_state.open_positions.pop(i)
            st.session_state.realized_pnl += pnl

            # Update risk manager
            st.session_state.risk_manager.record_trade(pos.get("condition_id", ""), pnl)

            if pnl > 0:
                st.session_state.winning_trades += 1

            # Format alert message based on exit reason
            reason_emoji = {
                ExitReason.EDGE_EXHAUSTED: "📊",
                ExitReason.EDGE_REVERSED: "↩️",
                ExitReason.MOMENTUM_SHIFT: "📉",
                ExitReason.TIME_DECAY: "⏰",
                ExitReason.STOP_LOSS: "🛑",
                ExitReason.TRAILING_STOP: "📍",
                ExitReason.MANUAL: "👤",
                ExitReason.SETTLEMENT: "🏁",
            }.get(exit_reason, "💰")

            add_alert(
                f"{reason_emoji} Closed ({exit_reason.value}): {pos['city']} {pos['outcome']} @ {current_price:.2f} | P/L: ${pnl:+.2f}",
                "success" if pnl > 0 else "warning"
            )
            return pnl
    return 0


def update_position_prices(markets, forecasts=None):
    """Update current prices for all open positions based on market data."""
    market_prices = {}
    for m in markets:
        key = m.get("condition_id", "")
        if key:
            market_prices[key] = m.get("yes_price", 0.5)

    for pos in st.session_state.open_positions:
        cid = pos.get("condition_id", "")
        if cid in market_prices:
            new_price = market_prices[cid]
            pos["current_price"] = new_price

            # Update Position object with edge snapshot and get accurate P/L
            position = pos.get("position_obj")
            if position:
                forecast_prob = pos.get("forecast_prob", 0.5)
                position.record_edge_snapshot(new_price, forecast_prob)
                # Use Position object's P/L calculation (handles YES/NO correctly)
                pos["unrealized_pnl"] = position.unrealized_pnl
            else:
                # Fallback legacy calculation - use shares not size
                shares = pos.get("shares", 1)
                entry_price = pos.get("entry_price", 0.5)
                if pos["side"] == "YES":
                    pos["unrealized_pnl"] = (new_price - entry_price) * shares
                else:
                    pos["unrealized_pnl"] = (entry_price - new_price) * shares

                # Calculate edge for synced positions (no position_obj)
                if forecasts:
                    city_key = pos.get("city", "").lower()
                    target_date = pos.get("target_date")
                    temp_low = pos.get("temp_low")
                    temp_high = pos.get("temp_high")

                    # Try to find matching forecast
                    fc = None
                    if target_date:
                        date_key = f"{city_key}_{target_date.isoformat() if hasattr(target_date, 'isoformat') else target_date}"
                        fc = forecasts.get(date_key) or forecasts.get(city_key)
                    else:
                        fc = forecasts.get(city_key)

                    if fc and (temp_low is not None or temp_high is not None):
                        forecast_mean = fc.get("high_mean", 50)
                        forecast_std = max(fc.get("high_std", 3), 2.0)
                        forecast_prob = calc_outcome_probability(temp_low, temp_high, forecast_mean, forecast_std)

                        if forecast_prob is not None:
                            pos["forecast_prob"] = forecast_prob
                            if pos["side"] == "YES":
                                pos["current_edge"] = forecast_prob - new_price
                            else:
                                pos["current_edge"] = (1 - forecast_prob) - (1 - new_price)

    # Update risk manager equity
    st.session_state.risk_manager.update_equity()


def auto_trade_check(signals, bankroll, kelly_fraction, max_position, min_edge, is_live, forecasts=None, markets=None):
    """Check signals and execute trades automatically, including smart exits."""
    if not st.session_state.auto_trade_enabled:
        return

    # PHASE 1: Check for smart exits on existing positions
    if forecasts and markets:
        check_smart_exits(forecasts, markets)

    # PHASE 2: Check risk limits before new trades
    if not st.session_state.risk_manager.is_trading_allowed():
        risk_summary = st.session_state.risk_manager.get_risk_summary()
        add_alert(f"⛔ Trading halted: {risk_summary.get('halt_reason', 'risk limit')}", "danger")
        st.session_state.last_auto_trade_check = get_est_now()
        return

    # PHASE 3: Execute new trades on signals
    trades_made = 0
    replacements_made = 0
    for signal in signals:
        if signal["signal"] == "PASS":
            continue

        if abs(signal["edge"]) < min_edge:
            continue

        if signal["confidence"] < st.session_state.trading_config.min_confidence_to_enter:
            continue

        # Use risk manager to calculate position size
        position_size = st.session_state.risk_manager.calculate_position_size(
            edge=abs(signal["edge"]),
            win_probability=signal["our_prob"] if signal["edge"] > 0 else (1 - signal["our_prob"]),
            price=signal["market_prob"] if signal["edge"] > 0 else (1 - signal["market_prob"])
        )

        # Apply user's max position override
        max_size = bankroll * max_position / 100
        position_size = min(position_size, max_size)

        if position_size < st.session_state.trading_config.min_position_size:
            continue

        # Check if we already have a position in this market
        market_id = signal.get("condition_id", f"demo_{signal['city']}")
        existing_positions = st.session_state.position_manager.get_positions_by_market(market_id)
        if existing_positions:
            continue

        # Risk check before trade
        can_trade, violations = st.session_state.risk_manager.check_can_trade(
            market_id=market_id,
            city=signal["city"],
            position_size=position_size,
            edge=abs(signal["edge"]),
            confidence=signal["confidence"]
        )

        if not can_trade:
            # POSITION REPLACEMENT: If blocked only by max positions, try replacing weakest
            max_pos_blocked = any(
                v.check.value == "max_positions" for v in violations if v.severity == "critical"
            )
            only_max_pos_blocked = max_pos_blocked and sum(
                1 for v in violations if v.severity == "critical"
            ) == 1

            if only_max_pos_blocked:
                new_edge = abs(signal["edge"])
                min_replace_advantage = st.session_state.trading_config.min_edge_advantage_to_replace

                # Find the open position with the weakest current edge
                weakest_pos = None
                weakest_edge = float('inf')
                weakest_legacy = None

                for legacy_pos in st.session_state.open_positions:
                    pos_obj = legacy_pos.get("position_obj")
                    if pos_obj and pos_obj.status == PositionStatus.OPEN:
                        pos_edge = pos_obj.current_edge
                        if pos_edge < weakest_edge:
                            weakest_edge = pos_edge
                            weakest_pos = pos_obj
                            weakest_legacy = legacy_pos

                # Replace if new signal has meaningfully better edge
                if weakest_pos and (new_edge - weakest_edge) >= min_replace_advantage:
                    add_alert(
                        f"🔄 Replacing weakest position: {weakest_legacy['city']} {weakest_legacy['outcome']} "
                        f"(edge {weakest_edge:+.1%}) with {signal['city']} {signal.get('outcome','')} "
                        f"(edge {new_edge:+.1%})",
                        "info"
                    )
                    close_position_smart(
                        weakest_pos.position_id,
                        weakest_legacy["current_price"],
                        ExitReason.RISK_LIMIT
                    )
                    execute_trade(signal, position_size, is_live)
                    trades_made += 1
                    replacements_made += 1
                    continue

            critical = [v for v in violations if v.severity == "critical"]
            if critical:
                add_alert(f"⚠️ Trade blocked: {critical[0].message}", "warning")
            continue

        execute_trade(signal, position_size, is_live)
        trades_made += 1

    if trades_made > 0:
        msg = f"🤖 Auto-trader executed {trades_made} new trade(s)"
        if replacements_made > 0:
            msg += f" ({replacements_made} replaced weaker positions)"
        add_alert(msg, "success")

    st.session_state.last_auto_trade_check = get_est_now()


def main():
    # Sidebar
    st.sidebar.title("🌤️ Weather Trader")
    st.sidebar.markdown("---")

    # Mode toggle
    mode = st.sidebar.radio(
        "Trading Mode",
        ["🔒 Dry Run", "🔴 Live Trading"],
        index=0,
        help="Dry Run simulates trades without real money"
    )
    is_live = mode == "🔴 Live Trading"

    if is_live:
        # Validate credentials and initialize KalshiClient for live trading
        auth = KalshiAuth()
        if not auth.is_configured:
            st.sidebar.error("⚠️ Kalshi credentials not configured! Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env")
            st.sidebar.warning("Falling back to Dry Run mode.")
            is_live = False
        else:
            if st.session_state.kalshi_client is None:
                try:
                    valid, msg = auth.validate_for_trading()
                    if not valid:
                        raise ValueError(msg)
                    client = KalshiClient(auth=auth)
                    st.session_state.kalshi_client = client
                    st.session_state.live_wallet_validated = True
                    st.sidebar.success(f"Kalshi connected: {auth.key_id[:8]}...")
                except Exception as e:
                    st.sidebar.error(f"⚠️ Live trading init failed: {e}")
                    st.sidebar.warning("Falling back to Dry Run mode.")
                    is_live = False
            else:
                st.sidebar.success(f"Kalshi connected: {auth.key_id[:8]}...")

            if is_live:
                st.sidebar.error("⚠️ LIVE TRADING ENABLED - Real orders will be submitted")

                # Auto-sync positions on first live connection (use full reconciliation)
                if not st.session_state.positions_synced:
                    added, removed, updated = reconcile_positions_with_kalshi()
                    total_synced = added + updated
                    if total_synced > 0 or removed > 0:
                        add_alert(f"Synced positions: +{added} added, ~{updated} updated, -{removed} removed", "success")
                    st.session_state.positions_synced = True

                # Manual sync button - full reconciliation
                if st.sidebar.button("🔄 Sync Positions from Kalshi"):
                    added, removed, updated = reconcile_positions_with_kalshi()
                    if added > 0 or removed > 0 or updated > 0:
                        add_alert(f"Kalshi sync: +{added} added, -{removed} removed, ~{updated} updated", "success")
                    else:
                        add_alert("Positions already in sync with Kalshi", "info")
                    st.rerun()

                # Clear cache button - forces fresh API calls
                if st.sidebar.button("🗑️ Clear Cache & Refresh"):
                    st.cache_data.clear()
                    # Bug #9 fix: Reset ALL API states, not just Open-Meteo
                    from weather_trader.apis import reset_all_api_state
                    reset_all_api_state()
                    # Also clear session state forecast data that might be stale
                    if 'forecast_history' in st.session_state:
                        st.session_state.forecast_history = []
                    add_alert("Cache cleared & all API states reset - fetching fresh data", "info")
                    st.rerun()

                # Live Kalshi stats
                st.sidebar.markdown("---")
                st.sidebar.markdown("**💰 Kalshi Account**")
                if 'kalshi_balance' not in st.session_state:
                    # Fetch initial balance
                    kalshi_data = get_live_kalshi_data()
                    if kalshi_data:
                        st.session_state.kalshi_balance = kalshi_data["balance"]
                        st.session_state.kalshi_position_count = kalshi_data["position_count"]

                if 'kalshi_balance' in st.session_state:
                    st.sidebar.metric("Balance", f"${st.session_state.kalshi_balance:.2f}")
                    open_count = len([p for p in st.session_state.open_positions if p.get("mode", "").startswith("LIVE")])
                    st.sidebar.caption(f"{open_count} tracked positions")

                # Pending orders section
                st.sidebar.markdown("---")
                st.sidebar.markdown("**📋 Pending Orders**")
                if st.sidebar.button("🔍 Check Order Status"):
                    result = check_pending_orders()
                    if result:
                        add_alert(f"Orders: {result['filled']} filled, {result['pending']} pending", "info")
                    st.rerun()

                pending_count = len(st.session_state.pending_orders)
                if pending_count > 0:
                    st.sidebar.warning(f"⏳ {pending_count} orders waiting to fill")
                    if st.sidebar.button("❌ Cancel Stale Orders (>5min)"):
                        cancelled = cancel_stale_orders(5)
                        if cancelled > 0:
                            add_alert(f"Cancelled {cancelled} stale orders", "info")
                        else:
                            add_alert("No stale orders to cancel", "info")
                        st.rerun()
                else:
                    st.sidebar.caption("No pending orders")
                st.sidebar.caption("Orders >5min auto-cancelled on refresh")

    # Demo mode toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Source")
    # Remember demo mode state (default True only on first load)
    if 'demo_mode_set' not in st.session_state:
        st.session_state.demo_mode_set = True
        st.session_state.demo_mode = True
    use_demo = st.sidebar.checkbox(
        "Demo Mode (Simulated Markets)",
        value=st.session_state.demo_mode,
        help="Use simulated markets for testing. Disable when real Kalshi weather markets are available."
    )
    st.session_state.demo_mode = use_demo

    if use_demo:
        st.sidebar.info("Using simulated markets for demo")
        if is_live:
            st.sidebar.warning("⚠️ Demo + Live = orders won't submit (fake tickers)")
    else:
        st.sidebar.success("🔍 Using real Kalshi markets")

    # Auto-trade toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Auto-Trading")

    auto_trade = st.sidebar.checkbox(
        "Enable Auto-Trading",
        value=st.session_state.auto_trade_enabled,
        help="Automatically execute trades when signals meet criteria"
    )
    st.session_state.auto_trade_enabled = auto_trade

    if auto_trade:
        st.sidebar.markdown('<div class="auto-trade-active">', unsafe_allow_html=True)
        st.sidebar.markdown("🟢 **AUTO-TRADE ACTIVE**")
        if st.session_state.last_auto_trade_check:
            st.sidebar.caption(f"Last check: {format_est_time(st.session_state.last_auto_trade_check)}")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Auto-refresh for auto-trading
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (1 min)",
        value=auto_trade,
        help="Automatically refresh data for auto-trading"
    )

    st.sidebar.markdown("---")

    # Fill aggressiveness setting
    st.sidebar.markdown("### 📊 Order Settings")
    fill_offset = st.sidebar.slider(
        "Fill Offset (cents)",
        min_value=0,
        max_value=10,
        value=2,
        help="Add cents to limit price to improve fill rate. 0=exact price, 5=aggressive"
    )
    st.session_state.fill_offset_cents = fill_offset
    if fill_offset == 0:
        st.sidebar.caption("Orders at exact fair value (may not fill)")
    elif fill_offset <= 2:
        st.sidebar.caption("Conservative: small premium for fills")
    else:
        st.sidebar.caption("Aggressive: pays more for faster fills")

    # Exit settings
    st.sidebar.markdown("### 🚪 Exit Settings")
    min_edge_exit = st.sidebar.slider(
        "Min Edge to Hold (%)",
        min_value=1,
        max_value=15,
        value=5,
        help="Exit position when edge falls below this percentage"
    )
    st.session_state.min_edge_exit_pct = min_edge_exit / 100.0
    st.sidebar.caption(f"Exit when edge < {min_edge_exit}%")

    st.sidebar.markdown("---")

    # Bankroll setting - default to Kalshi balance if available in live mode
    default_bankroll = 1000.0
    if is_live and 'kalshi_balance' in st.session_state:
        default_bankroll = st.session_state.kalshi_balance

    # Use session state to remember bankroll setting
    if 'user_bankroll' not in st.session_state:
        st.session_state.user_bankroll = default_bankroll

    bankroll = st.sidebar.number_input(
        "Bankroll ($)",
        min_value=10.0,
        max_value=100000.0,
        value=st.session_state.user_bankroll,
        step=100.0,
        help="In live mode, this syncs with your Kalshi balance"
    )
    st.session_state.user_bankroll = bankroll

    # Sync with Kalshi balance button in live mode
    if is_live and 'kalshi_balance' in st.session_state:
        if st.sidebar.button("🔄 Sync Bankroll with Kalshi"):
            st.session_state.user_bankroll = st.session_state.kalshi_balance
            bankroll = st.session_state.kalshi_balance
            st.rerun()

    # Sync bankroll with risk manager
    if st.session_state.risk_manager.bankroll != bankroll:
        st.session_state.risk_manager.bankroll = bankroll
        # Reset peak equity to new bankroll so changing bankroll doesn't trigger false drawdown
        st.session_state.risk_manager.peak_equity = bankroll
        st.session_state.risk_manager.reset_halt()

    # Risk settings
    st.sidebar.markdown("### ⚙️ Risk Settings")
    kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)
    max_position = st.sidebar.slider("Max Position (%)", 1, 20, 5)
    min_edge = st.sidebar.slider("Min Edge (%)", 1, 20, 5) / 100

    st.sidebar.markdown("---")

    # Quick stats - sync with actual position data
    st.sidebar.markdown("### 📊 Session Stats")

    # Calculate actual P/L from positions
    total_unrealized = sum(p.get("unrealized_pnl", 0) for p in st.session_state.open_positions)
    total_pnl = st.session_state.realized_pnl + total_unrealized

    # Update the session state pnl to stay in sync
    st.session_state.pnl = total_pnl

    st.sidebar.metric("Realized P/L", f"${st.session_state.realized_pnl:+.2f}")
    st.sidebar.metric("Unrealized P/L", f"${total_unrealized:+.2f}")
    st.sidebar.metric("Total P/L", f"${total_pnl:+.2f}")
    win_rate = (st.session_state.winning_trades / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0
    st.sidebar.metric("Win Rate", f"{win_rate:.1f}%")
    st.sidebar.metric("Total Trades", st.session_state.total_trades)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        add_alert("Data refreshed", "info")
        st.rerun()

    # Main content
    st.title("🌤️ Weather Trading Dashboard")

    # Status bar
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if is_live:
            st.markdown("**Mode:** 🔴 LIVE")
        else:
            st.markdown("**Mode:** 🔒 DRY RUN")
    with col2:
        st.markdown(f"**Bankroll:** ${bankroll:,.2f}")
    with col3:
        if auto_trade:
            st.markdown('**Auto-Trade:** <span class="live-indicator"></span> ON', unsafe_allow_html=True)
        else:
            st.markdown("**Auto-Trade:** OFF")
    with col4:
        st.markdown(f"**Data:** {'Demo' if use_demo else 'Live'}")
    with col5:
        st.markdown(f"**Updated:** {format_est_time()}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "🌡️ Forecasts",
        "📈 Markets",
        "🔬 Model Comparison",
        "📅 Multi-Day",
        "💰 Trades & P/L",
        "📉 Price Charts",
        "🔔 Alerts"
    ])

    # Fetch data
    with st.spinner("Fetching data..."):
        forecasts = fetch_forecasts_with_models()
        multi_day = fetch_multi_day_forecasts()
        model_comparison = fetch_model_comparison()

        # Get markets (real or demo)
        if use_demo:
            markets = generate_demo_markets(forecasts) if forecasts else []
        else:
            markets = fetch_real_markets()

    # Calculate signals (best only for trading, all for analysis)
    signals = calculate_signals(forecasts, markets, show_all_outcomes=False) if markets and forecasts else []
    all_signals = calculate_signals(forecasts, markets, show_all_outcomes=True) if markets and forecasts else []

    # Update position prices and check for smart exits
    if markets:
        update_position_prices(markets, forecasts)

    # Auto-trade check (only use best signals)
    if auto_trade and signals:
        auto_trade_check(signals, bankroll, kelly_fraction, max_position, min_edge, is_live, forecasts, markets)

    # =====================
    # TAB 1: Overview
    # =====================
    with tab1:
        st.header("Trading Overview")

        # Market status banner
        if use_demo:
            st.info("📊 **Demo Mode Active** - Using simulated markets based on real weather forecasts. Toggle off 'Demo Mode' in sidebar when real Kalshi weather markets are available.")
        else:
            if markets:
                st.success(f"✅ **Live Markets Found** - {len(markets)} active weather markets on Kalshi")
            else:
                st.warning("⚠️ **No Live Markets** - No weather markets currently active on Kalshi. Enable 'Demo Mode' to test the system.")

        # Tomorrow.io status
        if config.api.tomorrow_io_api_key and "your_" not in config.api.tomorrow_io_api_key.lower():
            usage = get_tomorrow_io_usage()
            st.caption(f"📡 Tomorrow.io: {usage['calls_today']}/{usage['daily_limit']} calls today")

        # Auto-trade status with smart exit info
        if auto_trade:
            st.markdown("""
            <div class="auto-trade-active">
                <strong>🤖 Smart Auto-Trading Enabled</strong><br>
                <strong>Entry Criteria:</strong>
                <ul>
                    <li>Edge exceeds minimum threshold</li>
                    <li>Confidence above minimum (configurable)</li>
                    <li>Risk limits not exceeded</li>
                </ul>
                <strong>Smart Exit Triggers:</strong>
                <ul>
                    <li>📊 <strong>Edge Exhausted</strong>: Captured 75%+ of entry edge</li>
                    <li>↩️ <strong>Edge Reversed</strong>: Edge turned negative</li>
                    <li>📉 <strong>Momentum Shift</strong>: Edge trending down while profitable</li>
                    <li>🛑 <strong>Stop Loss</strong>: Loss exceeds 30% of position</li>
                    <li>📍 <strong>Trailing Stop</strong>: Gave back 30%+ of peak profit</li>
                    <li>⏰ <strong>Time Decay</strong>: Approaching market settlement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Show risk summary - sync equity with actual P/L
            risk_summary = st.session_state.risk_manager.get_risk_summary()

            # Calculate actual equity from bankroll + P/L to ensure it ties out
            actual_unrealized = sum(p.get("unrealized_pnl", 0) for p in st.session_state.open_positions)
            actual_equity = bankroll + st.session_state.realized_pnl + actual_unrealized
            actual_total_pnl = st.session_state.realized_pnl + actual_unrealized

            st.markdown("---")
            st.subheader("📊 Risk Dashboard")

            risk_cols = st.columns(5)
            with risk_cols[0]:
                st.metric("Bankroll", f"${bankroll:,.2f}")
            with risk_cols[1]:
                pnl_pct = (actual_total_pnl / bankroll * 100) if bankroll > 0 else 0
                st.metric("Equity", f"${actual_equity:,.2f}",
                         f"{pnl_pct:+.1f}% P/L")
            with risk_cols[2]:
                # Drawdown from peak equity
                peak = max(bankroll, actual_equity)
                drawdown_pct = ((peak - actual_equity) / peak * 100) if peak > 0 else 0
                dd_color = "🔴" if drawdown_pct > 10 else "🟡" if drawdown_pct > 5 else "🟢"
                st.metric("Drawdown", f"{dd_color} {drawdown_pct:.1f}%")
            with risk_cols[3]:
                st.metric("Open Positions", f"{len(st.session_state.open_positions)}/{risk_summary['max_positions']}")
            with risk_cols[4]:
                daily_trades_display = "Unlimited" if risk_summary['max_daily_trades'] >= 999999 else f"{risk_summary['daily_trades']}/{risk_summary['max_daily_trades']}"
                st.metric("Daily Trades", f"{risk_summary['daily_trades']} ({daily_trades_display})")

            if risk_summary['trading_halted']:
                st.error(f"⛔ **TRADING HALTED**: {risk_summary['halt_reason']}")
                if st.button("🔓 Reset Trading Halt"):
                    st.session_state.risk_manager.reset_halt()
                    st.rerun()

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Cities Tracked", len(forecasts))
        with col2:
            st.metric("Active Markets", len(markets))
        with col3:
            tradeable = len([s for s in signals if s["signal"] != "PASS"])
            st.metric("Trade Signals", tradeable)
        with col4:
            avg_conf = sum(f["confidence"] for f in forecasts.values()) / len(forecasts) if forecasts else 0
            st.metric("Avg Confidence", f"{avg_conf:.0%}")
        with col5:
            strong = len([s for s in signals if "STRONG" in s["signal"]])
            st.metric("Strong Signals", strong)

        st.markdown("---")

        # Signals table
        if signals:
            st.subheader("🎯 Trading Signals")

            # Explanation of columns
            with st.expander("📖 How to Read Trading Signals", expanded=False):
                st.markdown("""
                | Column | What It Means | How It's Calculated |
                |--------|---------------|---------------------|
                | **City** | The city/market being analyzed | From Kalshi market data |
                | **Outcome** | Temperature range we're betting on | The specific bucket (e.g., "20-21°F") |
                | **Forecast** | Our predicted high temperature | Ensemble average from ECMWF, GFS, etc. |
                | **Our Prob** | Our probability this outcome wins | `P(temp falls in this range)` using normal distribution |
                | **Market** | Kalshi's implied probability | Current YES price (e.g., $0.42 = 42%) |
                | **Edge** | Our advantage over the market | `Our Prob - Market Prob` |
                | **Action** | Trade recommendation | Based on edge size and confidence |

                ### Edge Interpretation:
                - **+10% or more**: STRONG BUY YES (market significantly undervalues)
                - **+5% to +10%**: BUY YES (market undervalues)
                - **-5% to +5%**: PASS (no meaningful edge)
                - **-5% to -10%**: BUY NO (market overvalues YES)
                - **-10% or less**: STRONG BUY NO (market significantly overvalues)

                ### Example:
                - Forecast: 21°F for NYC tomorrow
                - Outcome: "20-21°F" range
                - Our Prob: 35% (our model says 35% chance temp is 20-21°F)
                - Market: 25% (Kalshi prices YES at $0.25)
                - Edge: +10% (we think it's worth 35¢, market sells for 25¢)
                - Action: BUY YES at $0.25, expecting to win 35% of the time
                """)

            df_signals = pd.DataFrame(signals)
            df_signals = df_signals.sort_values("signal_strength", ascending=False)

            # Column headers
            header_cols = st.columns([2, 1.5, 1.2, 1, 1, 1, 2])
            header_cols[0].markdown("**City**")
            header_cols[1].markdown("**Outcome**")
            header_cols[2].markdown("**Forecast**")
            header_cols[3].markdown("**Our Prob**")
            header_cols[4].markdown("**Market**")
            header_cols[5].markdown("**Edge**")
            header_cols[6].markdown("**Action**")
            st.divider()

            for _, row in df_signals.iterrows():
                with st.container():
                    cols = st.columns([2, 1.5, 1.2, 1, 1, 1, 2])

                    # City
                    city_label = f"**{row['city']}**"
                    if row.get('is_demo'):
                        city_label += " 🎮"
                    cols[0].markdown(city_label)

                    # Outcome (temperature range)
                    outcome = row.get('outcome', f"{row.get('temp_low', '?')}-{row.get('temp_high', '?')}")
                    cols[1].markdown(outcome)

                    # Forecast (our predicted temp in market's unit)
                    unit = row.get('temp_unit', 'F')
                    fcst = row.get('forecast_high_market', row.get('forecast_high_f', 0))
                    cols[2].markdown(f"{fcst:.1f}°{unit}")

                    # Our Probability
                    cols[3].markdown(f"{row['our_prob']:.0%}")

                    # Market Probability
                    cols[4].markdown(f"{row['market_prob']:.0%}")

                    # Edge (with color)
                    edge_color = "green" if row['edge'] > 0 else "red"
                    cols[5].markdown(f"**:{edge_color}[{row['edge']:+.1%}]**")

                    # Action button
                    if row['signal'] != "PASS":
                        button_label = f"{'🤖 ' if auto_trade else ''}{row['signal']}"
                        button_key = f"trade_{row['city']}_{row.get('condition_id', '')}"
                        if cols[6].button(button_label, key=button_key):
                            kelly = max(0, abs(row['edge']) / (1 - row['market_prob'])) if row['market_prob'] < 1 else 0
                            position = min(bankroll * kelly * kelly_fraction, bankroll * max_position / 100)
                            if position > 1:
                                execute_trade(row, position, is_live)
                                st.rerun()
                    else:
                        cols[6].markdown("*No edge*")

            st.markdown("---")

            # Edge visualization
            st.subheader("Edge Distribution")
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[s["city"] for s in signals],
                y=[s["edge"] * 100 for s in signals],
                marker_color=[s["signal_color"] for s in signals],
                text=[f"{s['edge']:+.1%}" for s in signals],
                textposition='outside'
            ))

            fig.update_layout(
                title="Edge by Market (Our Probability - Market Probability)",
                yaxis_title="Edge (%)",
                xaxis_title="City",
                showlegend=False,
                height=400
            )
            fig.add_hline(y=min_edge*100, line_dash="dash", line_color="green", annotation_text="Min Edge")
            fig.add_hline(y=-min_edge*100, line_dash="dash", line_color="red")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signals available. Enable Demo Mode or wait for real markets.")

    # =====================
    # TAB 2: Forecasts
    # =====================
    with tab2:
        st.header("🌡️ Weather Forecasts")
        today = today_est()
        tomorrow = today_est() + timedelta(days=1)
        st.caption(f"Today: {today.strftime('%A, %B %d')} | Tomorrow: {tomorrow.strftime('%A, %B %d')}")

        if forecasts:
            # Show both today's and tomorrow's forecasts for each city
            for city_key in get_all_cities():
                city_config = get_city_config(city_key)

                # Get forecasts for today and tomorrow
                today_key = f"{city_key}_{today.isoformat()}"
                tomorrow_key = f"{city_key}_{tomorrow.isoformat()}"
                fc_today = forecasts.get(today_key)
                fc_tomorrow = forecasts.get(tomorrow_key) or forecasts.get(city_key)

                if not fc_today and not fc_tomorrow:
                    continue

                st.markdown(f"### {city_config.name}")
                st.caption(f"📍 {city_config.station_name}")

                col1, col2 = st.columns(2)

                # TODAY's forecast
                with col1:
                    st.markdown(f"**Today** ({today.strftime('%b %d')})")
                    if fc_today:
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.metric("High", f"{fc_today['high_mean']:.1f}°F", f"±{fc_today['high_std']:.1f}°")
                        with subcol2:
                            st.metric("Low", f"{fc_today['low_mean']:.1f}°F")
                        with subcol3:
                            conf_pct = fc_today['confidence'] * 100
                            conf_color = "🟢" if conf_pct >= 80 else "🟡" if conf_pct >= 60 else "🔴"
                            st.metric("Conf", f"{conf_color} {conf_pct:.0f}%", f"{fc_today['model_count']} models")
                    else:
                        st.info("No forecast available for today")

                # TOMORROW's forecast
                with col2:
                    st.markdown(f"**Tomorrow** ({tomorrow.strftime('%b %d')})")
                    if fc_tomorrow:
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.metric("High", f"{fc_tomorrow['high_mean']:.1f}°F", f"±{fc_tomorrow['high_std']:.1f}°")
                        with subcol2:
                            st.metric("Low", f"{fc_tomorrow['low_mean']:.1f}°F")
                        with subcol3:
                            conf_pct = fc_tomorrow['confidence'] * 100
                            conf_color = "🟢" if conf_pct >= 80 else "🟡" if conf_pct >= 60 else "🔴"
                            st.metric("Conf", f"{conf_color} {conf_pct:.0f}%", f"{fc_tomorrow['model_count']} models")
                    else:
                        st.info("No forecast available for tomorrow")

                st.divider()

            # Temperature comparison chart - show each city in its market unit
            st.subheader("📊 Temperature Comparison")

            # Build data — all cities are Fahrenheit
            temp_data = []
            for city_key in get_all_cities():
                if city_key not in forecasts:
                    continue
                fc = forecasts[city_key]

                temp_data.append({
                    "City": fc['city'],
                    "High": fc["high_mean"],
                    "Low": fc["low_mean"],
                    "High_err": fc["high_std"],
                    "Low_err": fc["low_std"],
                })

            df_temps = pd.DataFrame(temp_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='High',
                x=df_temps['City'],
                y=df_temps['High'],
                error_y=dict(type='data', array=df_temps['High_err'], color='rgba(255,87,34,0.5)'),
                marker_color='#ff5722',
                text=[f"{h:.1f}" for h in df_temps['High']],
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                name='Low',
                x=df_temps['City'],
                y=df_temps['Low'],
                error_y=dict(type='data', array=df_temps['Low_err'], color='rgba(33,150,243,0.5)'),
                marker_color='#2196f3',
                text=[f"{l:.1f}" for l in df_temps['Low']],
                textposition='outside'
            ))
            fig.update_layout(
                barmode='group',
                yaxis_title="Temperature (°F)",
                height=350,
                margin=dict(t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # =====================
    # TAB 3: Markets
    # =====================
    with tab3:
        st.header("📈 Kalshi Weather Markets")

        if use_demo:
            st.info("🎮 **Demo Markets** - These are simulated markets based on real forecasts. Disable 'Demo Mode' in sidebar to search for real Kalshi markets.")
        else:
            st.info("🔍 **Live Markets** - Fetching from Kalshi API. Markets are grouped by city/date.")

        if markets:
            # Group markets by city and date for display
            market_groups = {}
            for market in markets:
                city = market.get("city_name", market["city"])
                target_date = market.get("target_date", today_est())
                group_key = f"{city} - {target_date}"
                if group_key not in market_groups:
                    market_groups[group_key] = {
                        "city_key": market["city"],
                        "city_name": city,
                        "target_date": target_date,
                        "question": market.get("question", ""),
                        "resolution_source": market.get("resolution_source", ""),
                        "is_demo": market.get("is_demo", False),
                        "outcomes": []
                    }
                market_groups[group_key]["outcomes"].append(market)

            st.success(f"Found {len(market_groups)} markets with {len(markets)} total outcomes")

            for group_key, group in market_groups.items():
                demo_badge = " 🎮" if group["is_demo"] else ""
                with st.expander(f"**{group_key}**{demo_badge}", expanded=True):
                    st.markdown(f"**Question:** {group['question']}")
                    st.markdown(f"**Resolution:** {group['resolution_source']}")
                    st.markdown("---")

                    # Show all outcomes as a table with calculation details
                    outcome_data = []
                    city_key = group["city_key"].lower()
                    target_date = group.get("target_date", today_est())

                    # CRITICAL: Use date-specific key for forecast lookup
                    # For same-day markets, we need today's forecast, not tomorrow's
                    if hasattr(target_date, 'date'):
                        target_date = target_date.date()
                    date_key = f"{city_key}_{target_date.isoformat()}"
                    fc = forecasts.get(date_key)

                    # Fallback to city_key only if it matches the target date
                    if not fc:
                        fc = forecasts.get(city_key)
                        if fc and fc.get("date") != target_date:
                            # Wrong date - don't use tomorrow's forecast for today's market
                            print(f"[Markets] Skipping wrong-date forecast for {city_key}: have {fc.get('date')}, need {target_date}")
                            fc = None

                    if not fc:
                        print(f"[Markets] No forecast for '{date_key}' - available keys: {[k for k in forecasts.keys() if city_key in k]}")

                    # Get forecast in market's unit
                    market_unit = group["outcomes"][0].get("temp_unit", "F") if group["outcomes"] else "F"
                    forecast_high_f = fc.get("high_mean", 70) if fc else 70
                    forecast_std_f = max(fc.get("high_std", 3), 2) if fc else 3

                    if market_unit == "C":
                        forecast_high = (forecast_high_f - 32) * 5 / 9
                        forecast_std = forecast_std_f * 5 / 9
                    else:
                        forecast_high = forecast_high_f
                        forecast_std = forecast_std_f

                    for o in sorted(group["outcomes"], key=lambda x: x.get("temp_midpoint", 0)):
                        outcome_desc = o.get("outcome_desc", "")
                        yes_price = o.get("yes_price", 0)
                        volume = o.get("volume", 0)
                        temp_low = o.get("temp_low")
                        temp_high = o.get("temp_high")

                        # Calculate our probability with formula explanation
                        our_prob = None
                        formula = ""
                        calc_type = ""

                        if fc:
                            if temp_low is None and temp_high is not None:
                                calc_type = "≤ (or below)"
                                formula = f"P(T≤{temp_high:.0f}) = CDF({temp_high:.0f}+0.5)"
                            elif temp_high is None and temp_low is not None:
                                calc_type = "≥ (or higher)"
                                formula = f"P(T≥{temp_low:.0f}) = 1-CDF({temp_low:.0f}-0.5)"
                            elif temp_low is not None and temp_high is not None:
                                if temp_low == temp_high:
                                    calc_type = f"= {temp_low:.0f}° (±0.5)"
                                else:
                                    calc_type = "range"
                                formula = f"P({temp_low:.0f}-0.5≤T≤{temp_high:.0f}+0.5)"

                            our_prob = calc_outcome_probability(
                                temp_low, temp_high, forecast_high, forecast_std
                            )

                        edge = (our_prob - yes_price) if our_prob is not None else None

                        # Determine if this is likely correct or problematic
                        status = ""
                        if edge is not None:
                            if abs(edge) > 0.30:
                                status = "⚠️"  # Very large edge might indicate issue
                            elif edge > 0.05:
                                status = "📈"  # Buy YES opportunity
                            elif edge < -0.05:
                                status = "📉"  # Buy NO opportunity

                        outcome_data.append({
                            "Outcome": outcome_desc,
                            "Type": calc_type,
                            "Market": f"{yes_price:.1%}",
                            "Ours": f"{our_prob:.1%}" if our_prob is not None else "N/A",
                            "Edge": f"{edge:+.1%}" if edge is not None else "N/A",
                            "Signal": status,
                            "Volume": f"${volume:,.0f}"
                        })

                    df_outcomes = pd.DataFrame(outcome_data)
                    st.dataframe(df_outcomes, use_container_width=True, hide_index=True)

                    # Show forecast context
                    if fc:
                        fc_date = fc.get("date", "unknown")
                        fc_source = fc.get("source", "ensemble")
                        is_same_day_market = target_date == today_est()
                        day_label = "TODAY" if is_same_day_market else f"{target_date}"
                        st.caption(f"📊 Forecast for {day_label}: {forecast_high:.1f}°{market_unit} (±{forecast_std:.1f}°) | Source: {fc_source} | Date: {fc_date}")
                    else:
                        st.warning(f"⚠️ No forecast available for {target_date} - calculations may be incorrect")

                    # Calculation verification
                    with st.expander("🔍 Verify Calculations"):
                        st.markdown(f"""
                        **Forecast (in market unit):** {forecast_high:.2f}°{market_unit}
                        **Standard Deviation:** {forecast_std:.2f}°

                        **How probabilities are calculated:**
                        Markets settle on integer temperatures. We apply **continuity correction** (±0.5) to model discrete readings with a continuous normal distribution:
                        - **"X or below"**: `P(T < X+0.5) = CDF(X+0.5)`
                        - **"X or higher"**: `P(T ≥ X-0.5) = 1 - CDF(X-0.5)`
                        - **"X°" (single)**: `P(X-0.5 ≤ T < X+0.5) = CDF(X+0.5) - CDF(X-0.5)`
                        - **"X-Y range"**: `P(X-0.5 ≤ T < Y+0.5) = CDF(Y+0.5) - CDF(X-0.5)`
                        - **Clipping**: All probabilities clipped to [{PROB_FLOOR:.0%}, {PROB_CEILING:.0%}]

                        **Example for this market:**
                        - Forecast = {forecast_high:.1f}°{market_unit}, std = {forecast_std:.1f}°
                        - "≤{forecast_high-2:.0f}" → {calc_outcome_probability(None, forecast_high-2, forecast_high, forecast_std):.1%}
                        - "={forecast_high:.0f}" → {calc_outcome_probability(forecast_high, forecast_high, forecast_high, forecast_std):.1%}
                        - "≥{forecast_high+2:.0f}" → {calc_outcome_probability(forecast_high+2, None, forecast_high, forecast_std):.1%}
                        """)
        else:
            st.warning("No markets found")
            st.info("Enable Demo Mode in the sidebar to test with simulated markets, or check if Kalshi has active weather markets.")

    # =====================
    # TAB 4: Model Comparison
    # =====================
    with tab4:
        st.header("🔬 Model Comparison")
        st.caption("Comparing forecasts from ECMWF, GFS, HRRR, Tomorrow.io, and Best Match models")

        if model_comparison:
            for city_key, data in model_comparison.items():
                city_config = get_city_config(city_key)
                unit_label = "°F"

                st.subheader(f"{data['city']} ({unit_label})")

                # Display both TODAY and TOMORROW forecasts
                col1, col2 = st.columns(2)

                for col, day_key, day_label in [(col1, "today", "Today"), (col2, "tomorrow", "Tomorrow")]:
                    with col:
                        day_data = data.get(day_key, {})
                        day_date = day_data.get("date", today_est())
                        models_data = day_data.get("models", {})

                        st.markdown(f"**{day_label}** ({day_date.strftime('%b %d')})")

                        if models_data:
                            # Merge Tomorrow.io data from forecasts (fetched separately)
                            date_key = f"{city_key}_{day_date.isoformat()}"
                            fc = forecasts.get(date_key) or (forecasts.get(city_key) if day_key == "tomorrow" else None)
                            if fc and "tomorrow" not in [m.lower() for m in models_data.keys()]:
                                fc_models = fc.get("models", [])
                                for m in fc_models:
                                    if m.get("model") == "tomorrow":
                                        models_data["tomorrow.io"] = {
                                            "high": m["high"],
                                            "low": m["low"]
                                        }
                                        break

                            models = list(models_data.keys())
                            highs = [models_data[m]["high"] for m in models]
                            lows = [models_data[m]["low"] for m in models]

                            fig = go.Figure()
                            fig.add_trace(go.Bar(name='High', x=models, y=highs, marker_color='#ff5722',
                                                text=[f"{h:.1f}" for h in highs], textposition='outside'))
                            fig.add_trace(go.Bar(name='Low', x=models, y=lows, marker_color='#2196f3',
                                                text=[f"{l:.1f}" for l in lows], textposition='outside'))

                            # Add ensemble line if available
                            if fc:
                                ensemble_high = fc.get("high_mean")
                                if ensemble_high:
                                    fig.add_hline(y=ensemble_high, line_dash="dash",
                                                line_color="red", annotation_text=f"Ensemble: {ensemble_high:.1f}")

                            fig.update_layout(barmode='group', height=280, showlegend=False,
                                            yaxis_title=f"Temp ({unit_label})", xaxis_title="",
                                            margin=dict(l=40, r=20, t=20, b=40))
                            st.plotly_chart(fig, use_container_width=True, key=f"model_{city_key}_{day_key}")

                            if len(highs) > 1:
                                spread = max(highs) - min(highs)
                                agreement = "High" if spread < 3 else ("Medium" if spread < 6 else "Low")
                                st.caption(f"Agreement: {agreement} (spread: {spread:.1f}°F)")
                        else:
                            st.info("No model data available")

                st.markdown("---")

        # Tomorrow.io API Usage Counter
        st.markdown("---")
        if config.api.tomorrow_io_api_key and "your_" not in config.api.tomorrow_io_api_key.lower():
            usage = get_tomorrow_io_usage()
            calls = usage["calls_today"]
            limit = usage["daily_limit"]
            budget = usage["budget"]
            pct_used = (calls / limit * 100) if limit > 0 else 0

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("API Calls Today", f"{calls} / {limit}")
            with col_b:
                status_color = "🟢" if calls < budget * 0.5 else "🟡" if calls < budget else "🔴"
                st.metric("Status", f"{status_color} {pct_used:.0f}% used")
            with col_c:
                remaining = max(0, budget - calls)
                st.metric("Remaining (Budget)", f"{remaining}")
            with col_d:
                if usage["last_call"]:
                    last_str = usage["last_call"].strftime("%H:%M:%S")
                    st.metric("Last Call", last_str)
                else:
                    st.metric("Last Call", "None yet")

            # Show whether Tomorrow.io data is present in the forecasts
            has_tomorrow = any(
                any(m.get("model") == "tomorrow.io" for m in fc.get("models", []))
                for fc in forecasts.values()
            ) if forecasts else False
            if has_tomorrow:
                st.success("Tomorrow.io data is included in the model comparison above.")
            else:
                if not usage["can_call"]:
                    st.warning(f"Tomorrow.io daily budget reached ({calls}/{budget}). Resets at midnight.")
                else:
                    st.info("Tomorrow.io data not yet loaded. Hit 'Refresh All Data' to fetch.")
        else:
            st.caption("Tomorrow.io API key not configured. Add TOMORROW_IO_API_KEY to .env for additional model data.")

    # =====================
    # TAB 5: Multi-Day
    # =====================
    with tab5:
        st.header("📅 7-Day Forecast")

        if multi_day:
            city_select = st.selectbox("Select City", list(multi_day.keys()),
                                      format_func=lambda x: multi_day[x]["city"])

            if city_select:
                unit_label = "°F"

                data = multi_day[city_select]
                df = pd.DataFrame(data["daily"])

                df['high_display'] = df['high']
                df['low_display'] = df['low']

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['date'], y=df['high_display'], name='High',
                                        line=dict(color='#ff5722', width=3), mode='lines+markers'))
                fig.add_trace(go.Scatter(x=df['date'], y=df['low_display'], name='Low',
                                        line=dict(color='#2196f3', width=3), mode='lines+markers'))
                fig.add_trace(go.Scatter(
                    x=list(df['date']) + list(df['date'])[::-1],
                    y=list(df['high_display']) + list(df['low_display'])[::-1],
                    fill='toself', fillcolor='rgba(255, 87, 34, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'), showlegend=False
                ))
                fig.update_layout(title=f"{data['city']} - 7 Day Forecast",
                                xaxis_title="Date", yaxis_title=f"Temperature ({unit_label})",
                                height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                # Display table with correct units
                display_df = df[['date', 'high_display', 'low_display', 'model']].copy()
                display_df.columns = ['Date', f'High ({unit_label})', f'Low ({unit_label})', 'Model']
                st.dataframe(display_df.style.format({
                    f'High ({unit_label})': "{:.1f}",
                    f'Low ({unit_label})': "{:.1f}"
                }), use_container_width=True)

    # =====================
    # TAB 6: Trades & P/L
    # =====================
    with tab6:
        st.header("💰 Trades & P/L")

        # Note: Position prices are already updated in the main flow

        # Summary metrics
        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in st.session_state.open_positions)
        total_pnl = st.session_state.realized_pnl + total_unrealized

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Open Positions", len(st.session_state.open_positions))
        with col2:
            color = "normal" if total_unrealized >= 0 else "inverse"
            st.metric("Unrealized P/L", f"${total_unrealized:+.2f}", delta_color=color)
        with col3:
            st.metric("Realized P/L", f"${st.session_state.realized_pnl:+.2f}")
        with col4:
            st.metric("Total P/L", f"${total_pnl:+.2f}")

        st.divider()

        # Open Positions Section with Edge Tracking
        st.subheader("📊 Open Positions")

        if st.session_state.open_positions:
            # Header row
            header_cols = st.columns([2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, 1])
            header_cols[0].markdown("**Position**")
            header_cols[1].markdown("**Side**")
            header_cols[2].markdown("**Qty/Cost**")
            header_cols[3].markdown("**Entry**")
            header_cols[4].markdown("**Current**")
            header_cols[5].markdown("**Edge**")
            header_cols[6].markdown("**P/L**")
            header_cols[7].markdown("**Status**")
            header_cols[8].markdown("**Action**")
            st.divider()

            for pos in st.session_state.open_positions:
                with st.container():
                    cols = st.columns([2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, 1])

                    # Escape > to prevent markdown blockquote interpretation
                    outcome_display = pos['outcome'].replace(">", "\\>")
                    cols[0].markdown(f"**{pos['city']}**\n{outcome_display}")
                    cols[1].markdown(f"**{pos['side']}**")
                    shares = pos.get('shares', int(pos['size'] / max(pos['entry_price'], 0.01)))
                    cost = shares * pos['entry_price']
                    cols[2].markdown(f"{shares}\n${cost:.2f}")
                    cols[3].markdown(f"{pos['entry_price']:.2f}")
                    cols[4].markdown(f"{pos['current_price']:.2f}")

                    # Show current edge
                    position = pos.get("position_obj")
                    if position:
                        current_edge = position.current_edge
                        edge_captured = position.edge_captured_pct
                        edge_trend = position.get_edge_trend()

                        edge_color = "green" if current_edge > 0 else "red"
                        trend_arrow = "↗️" if edge_trend > 0.005 else "↘️" if edge_trend < -0.005 else "→"
                        cols[5].markdown(f":{edge_color}[{current_edge:+.1%}] {trend_arrow}")
                    else:
                        # For synced positions, use current_edge if available, else edge_at_entry
                        current_edge = pos.get('current_edge', pos.get('edge_at_entry', 0))
                        edge_color = "green" if current_edge > 0 else "red"
                        cols[5].markdown(f":{edge_color}[{current_edge:+.1%}]")

                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_color = "green" if pnl >= 0 else "red"
                    cols[6].markdown(f"**:{pnl_color}[${pnl:+.2f}]**")

                    # Show edge exhaustion status
                    if position:
                        if edge_captured >= 0.75:
                            cols[7].markdown("⚠️ Edge exhausted")
                        elif current_edge < 0:
                            cols[7].markdown("🔴 Edge reversed")
                        elif edge_trend < -0.005:
                            cols[7].markdown("📉 Momentum ↓")
                        else:
                            cols[7].markdown("🟢 Holding")
                    else:
                        cols[7].markdown("—")

                    if cols[8].button("🔴 SELL", key=f"sell_{pos['id']}"):
                        close_position(pos['id'], pos['current_price'])
                        st.rerun()

                    st.divider()
        else:
            st.info("No open positions. Trades will appear here when executed.")

        st.divider()

        # Position Calculator - Shows actual signals with auto-calculated positions
        st.subheader("🧮 Position Calculator")

        if signals:
            st.markdown("**Active trading signals with calculated positions:**")

            for sig in sorted(signals, key=lambda x: abs(x.get("edge", 0)), reverse=True):
                if sig["signal"] == "PASS":
                    continue

                edge = sig["edge"]
                our_prob = sig["our_prob"]
                market_prob = sig["market_prob"]

                kelly = max(0, abs(edge) / (1 - market_prob)) if market_prob < 1 else 0
                adj_kelly = kelly * kelly_fraction
                position = min(bankroll * adj_kelly, bankroll * max_position / 100)

                unit = sig.get("temp_unit", "F")
                fcst = sig.get("forecast_high_market", sig.get("forecast_high_f", 0))

                with st.container():
                    st.markdown(f"### {sig['city']} — {sig['outcome']}")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Forecast", f"{fcst:.1f}°{unit}")
                    col2.metric("Our Prob", f"{our_prob:.1%}")
                    col3.metric("Market", f"{market_prob:.1%}")

                    edge_delta = "normal" if edge > 0 else "inverse"
                    col4.metric("Edge", f"{edge:+.1%}", delta_color=edge_delta)

                    col1b, col2b, col3b, col4b = st.columns(4)
                    col1b.metric("Kelly %", f"{kelly:.1%}")
                    col2b.metric("Adj Kelly", f"{adj_kelly:.1%}")
                    col3b.metric("Position", f"${position:.2f}")

                    if position >= 1 and abs(edge) >= min_edge:
                        col4b.success(f"✅ {sig['signal']}")
                    elif abs(edge) >= min_edge:
                        col4b.warning("⚠️ Size too small")
                    else:
                        col4b.info(f"Edge < {min_edge:.0%}")

                    st.divider()
        else:
            st.info("No signals available. Enable markets to see position calculations.")

        # Explanation of settings
        with st.expander("📖 Understanding the Settings"):
            st.markdown("""
            ### Sidebar Settings Explained

            **Bankroll ($)**
            Your total trading capital. All position sizes are calculated as percentages of this amount.

            **Kelly Fraction (0.1 - 1.0)**
            The Kelly Criterion calculates the mathematically optimal bet size to maximize long-term growth.
            However, full Kelly is very aggressive and can lead to large drawdowns.
            - **0.25 (Quarter Kelly)**: Conservative, recommended for beginners. Smoother returns.
            - **0.5 (Half Kelly)**: Balanced approach.
            - **1.0 (Full Kelly)**: Maximum growth but very volatile. High risk of ruin.

            **Max Position (%)**
            Hard cap on any single trade as a percentage of bankroll. Even if Kelly suggests more,
            the position won't exceed this. Prevents overexposure to any single market.

            **Min Edge (%)**
            Only trade when your edge exceeds this threshold. Accounts for:
            - Model uncertainty
            - Execution costs
            - Market microstructure

            ### The Formula
            ```
            Edge = Your Probability - Market Price
            Kelly % = Edge / (1 - Market Price)
            Position = min(Bankroll × Kelly × Fraction, Bankroll × Max%)
            ```

            ### Example
            - Your forecast: 70% chance of YES
            - Market price: 55¢
            - Edge: 70% - 55% = **+15%**
            - Kelly: 15% / 45% = **33%** of bankroll
            - With 0.25 fraction: 33% × 0.25 = **8.3%**
            - With $1000 bankroll and 5% max: **$50** position
            """)

        st.divider()

        # Trade History
        st.subheader("📜 Trade History")

        # Show closed positions with exit reasons
        if st.session_state.closed_positions:
            st.markdown("**Closed Positions:**")
            for pos in st.session_state.closed_positions[:10]:
                pnl = pos.get('realized_pnl', 0)
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                exit_reason = pos.get('exit_reason', 'manual')

                # Format exit reason
                reason_display = {
                    'edge_exhausted': '📊 Edge Captured',
                    'edge_reversed': '↩️ Edge Reversed',
                    'momentum_shift': '📉 Momentum',
                    'time_decay': '⏰ Time',
                    'stop_loss': '🛑 Stop Loss',
                    'trailing_stop': '📍 Trail Stop',
                    'manual': '👤 Manual',
                    'settlement': '🏁 Settled',
                }.get(exit_reason, exit_reason)

                st.markdown(
                    f"{pnl_emoji} {pos['city']} {pos['outcome']} | "
                    f"Entry: {pos['entry_price']:.2f} → Exit: {pos['exit_price']:.2f} | "
                    f"**P/L: ${pnl:+.2f}** | {reason_display}"
                )

        if st.session_state.trade_history:
            with st.expander("View All Trade Records"):
                df_hist = pd.DataFrame(st.session_state.trade_history)
                display_cols = ['time', 'city', 'outcome', 'side', 'size', 'entry_price', 'status']
                available_cols = [c for c in display_cols if c in df_hist.columns]
                st.dataframe(df_hist[available_cols], use_container_width=True)

            if st.button("🗑️ Clear All History"):
                st.session_state.trade_history = []
                st.session_state.open_positions = []
                st.session_state.closed_positions = []
                st.session_state.total_trades = 0
                st.session_state.winning_trades = 0
                st.session_state.pnl = 0
                st.session_state.realized_pnl = 0
                st.rerun()
        else:
            st.info("No trades recorded yet")

    # =====================
    # TAB 7: Price Charts
    # =====================
    with tab7:
        st.header("📉 Price History")

        if st.session_state.price_history:
            for market_id, prices in st.session_state.price_history.items():
                if len(prices) > 1:
                    df = pd.DataFrame(prices)
                    fig = px.line(df, x='time', y='price', title=f"Market: {market_id[:30]}...")
                    fig.update_layout(yaxis_tickformat='.2f', height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price history accumulates with each refresh")

            # Demo
            demo_times = pd.date_range(end=get_est_now(), periods=20, freq='30min')
            demo_prices = 0.5 + np.cumsum(np.random.randn(20) * 0.02)
            demo_prices = np.clip(demo_prices, 0.1, 0.9)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=demo_times, y=demo_prices, mode='lines', line=dict(color='#2196f3')))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(title="Demo: Price Over Time", yaxis_title="Price ($)", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # =====================
    # TAB 8: Alerts
    # =====================
    with tab8:
        st.header("🔔 Alerts")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Alerts"):
                st.session_state.alerts = []
                st.rerun()

        if st.session_state.alerts:
            for alert in st.session_state.alerts[:30]:
                level_styles = {
                    "info": {"bg": "#1e3a5f", "border": "#3b82f6", "text": "#e0e7ff"},
                    "success": {"bg": "#14532d", "border": "#22c55e", "text": "#dcfce7"},
                    "warning": {"bg": "#713f12", "border": "#f59e0b", "text": "#fef3c7"},
                    "danger": {"bg": "#7f1d1d", "border": "#ef4444", "text": "#fee2e2"}
                }
                level_icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "danger": "🚨"}

                style = level_styles.get(alert['level'], {"bg": "#374151", "border": "#6b7280", "text": "#f3f4f6"})

                # Format time in EST
                alert_time = alert['time']
                if alert_time.tzinfo is None:
                    alert_time = alert_time.replace(tzinfo=EST)
                time_str = alert_time.strftime('%H:%M:%S EST')

                st.markdown(f"""
                <div style="background-color: {style['bg']};
                            border-left: 4px solid {style['border']};
                            color: {style['text']};
                            padding: 12px 15px;
                            border-radius: 5px;
                            margin: 8px 0;
                            font-size: 14px;">
                    {level_icons.get(alert['level'], '📢')}
                    <strong>{time_str}</strong> — {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alerts yet")

    # Footer
    st.markdown("---")
    market_status = "Demo Markets" if use_demo else ("Live Markets" if markets else "No Markets")
    open_pos_count = len(st.session_state.open_positions)
    st.markdown(
        f"*Weather Trader v0.4.2 (Smart Trading Engine)* | "
        f"*{market_status}* | "
        f"*{len(forecasts)} cities • {len(signals)} signals • {open_pos_count} positions* | "
        f"*Auto-Trade: {'🟢 ON' if auto_trade else 'OFF'}* | "
        f"*{format_est_time()}*"
    )

    # Auto-refresh with Kalshi sync
    if auto_refresh:
        time.sleep(60)

        # Auto-sync with Kalshi on each refresh (if live trading)
        if is_live and st.session_state.kalshi_client is not None:
            try:
                # 1. Update live position prices from Kalshi
                update_position_prices_from_kalshi()

                # 2. Check and update pending order status
                check_pending_orders()

                # 3. Cancel orders pending > 5 minutes
                cancelled = cancel_stale_orders(5)
                if cancelled > 0:
                    add_alert(f"Auto-cancelled {cancelled} stale orders (>5min)", "info")

                # 4. Check for settled markets and close positions
                settled = check_settled_markets()
                if settled > 0:
                    add_alert(f"Auto-closed {settled} settled positions", "success")

                # 5. Full reconciliation with Kalshi (add, remove, update)
                added, removed, updated = reconcile_positions_with_kalshi()
                if added > 0 or removed > 0 or updated > 0:
                    add_alert(f"Kalshi sync: +{added} added, -{removed} removed, ~{updated} updated", "info")

                # 6. Check smart exits on all positions (if auto-trade enabled)
                if st.session_state.auto_trade_enabled and forecasts and markets:
                    check_smart_exits(forecasts, markets)

                # 7. Get latest balance
                kalshi_data = get_live_kalshi_data()
                if kalshi_data:
                    st.session_state.kalshi_balance = kalshi_data["balance"]
                    st.session_state.kalshi_position_count = kalshi_data["position_count"]

            except Exception as e:
                add_alert(f"Auto-sync error: {e}", "warning")

        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
