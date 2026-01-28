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
from weather_trader.kalshi import KalshiMarketFinder, KalshiAuth, KalshiClient
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

            # Add market inefficiency
            market_noise = random.uniform(-0.10, 0.10)
            market_prob = max(0.01, min(0.99, fair_prob + market_noise))

            condition_id = f"demo_{city_key}_{target_date}_{temp_low}_{temp_high}"

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
                "resolution_source": f"Demo - {city_config.station_name}",
                "is_demo": True,
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
    target_date = today_est() + timedelta(days=1)

    async with OpenMeteoClient() as om_client:
        for city_key in get_all_cities():
            city_config = get_city_config(city_key)
            city_data = {"city": city_config.name, "models": {}}

            for model in [WeatherModel.ECMWF, WeatherModel.GFS, WeatherModel.BEST_MATCH]:
                try:
                    forecasts = await om_client.get_daily_forecast(city_config, model, days=3)
                    for f in forecasts:
                        if f.timestamp.date() == target_date:
                            city_data["models"][model.value] = {
                                "high": f.temperature_high,
                                "low": f.temperature_low
                            }
                            break
                except Exception:
                    pass

            comparisons[city_key] = city_data

    return comparisons


@st.cache_data(ttl=60)
def fetch_real_markets():
    """Fetch real Kalshi markets."""
    return run_async(_fetch_real_markets())


async def _fetch_real_markets():
    """Fetch weather markets from Kalshi."""
    markets = []

    async with KalshiMarketFinder() as finder:
        try:
            found = await finder.find_weather_markets(active_only=True, days_ahead=3)
            add_alert(f"Kalshi API: Found {len(found)} market events", "info")

            for market in found:
                add_alert(f"Processing {market.city}: {len(market.brackets)} brackets", "info")
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
                        "event_ticker": market.event_ticker,
                        "total_market_volume": market.total_volume,
                    })
                    record_price(bracket.ticker, bracket.yes_price)
        except Exception as e:
            import traceback
            add_alert(f"Error fetching markets: {str(e)}", "danger")
            add_alert(f"Traceback: {traceback.format_exc()[:200]}", "warning")

    add_alert(f"Total outcomes fetched: {len(markets)}", "info")
    return markets


@st.cache_data(ttl=600)
def fetch_forecasts_with_models():
    """Fetch forecasts with individual model data."""
    return run_async(_fetch_forecasts_with_models())


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


def calculate_signals(forecasts, markets, show_all_outcomes=False):
    """Calculate trading signals for multi-outcome markets.

    Args:
        forecasts: Weather forecast data
        markets: Market data from Kalshi
        show_all_outcomes: If True, return signals for ALL outcomes, not just the best
    """
    signals = []

    # Group markets by city and date
    market_groups = {}
    for market in markets:
        city_key = market["city"].lower()
        target_date = market.get("target_date", today_est())
        group_key = f"{city_key}_{target_date}"
        if group_key not in market_groups:
            market_groups[group_key] = []
        market_groups[group_key].append(market)

    for group_key, group_markets in market_groups.items():
        if not group_markets:
            continue

        city_key = group_markets[0]["city"].lower()
        target_date = group_markets[0].get("target_date", today_est())
        date_key = f"{city_key}_{target_date.isoformat()}"

        fc = forecasts.get(date_key) or forecasts.get(city_key)
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
                "confidence": fc["confidence"],
                "forecast_high_f": forecast_temp_f,
                "forecast_high_market": forecast_temp_market,
                "forecast_std": forecast_std_market,
                "condition_id": market["condition_id"],
                "ticker": market.get("ticker", ""),
                "is_demo": market.get("is_demo", False),
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
                price_cents = max(1, min(99, int(round(actual_cost_per_share * 100))))
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
        "entry_price": entry_price,
        "current_price": entry_price,
        "edge_at_entry": signal["edge"],
        "forecast_prob": signal["our_prob"],
        "market_prob_at_entry": signal["market_prob"],
        "status": "OPEN",
        "mode": "LIVE" if is_live else "SIMULATED",
        "is_demo": signal.get("is_demo", False),
        "condition_id": signal.get("condition_id", ""),
        "target_date": target_date,
        "unrealized_pnl": 0.0,
        "position_obj": position,  # Link to actual Position object
        "exchange_order_id": exchange_order_id,
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

    for legacy_pos in st.session_state.open_positions:
        position = legacy_pos.get("position_obj")
        if not position or position.status != PositionStatus.OPEN:
            continue

        cid = legacy_pos.get("condition_id", "")
        city_key = legacy_pos.get("city", "").lower()

        # Get current market price
        current_price = market_prices.get(cid, legacy_pos["current_price"])

        # Get current forecast probability
        target_date = legacy_pos.get("target_date")
        if target_date and hasattr(target_date, 'isoformat'):
            date_key = f"{city_key}_{target_date.isoformat()}"
            fc = forecasts.get(date_key) or forecasts.get(city_key, {})
        else:
            fc = forecasts.get(city_key, {})
        forecast_prob = legacy_pos.get("forecast_prob", 0.5)

        if fc:
            # Recalculate probability based on current forecast
            # (simplified - using stored probability for now)
            forecast_prob = legacy_pos.get("forecast_prob", 0.5)

        # Update position with current data
        st.session_state.position_manager.update_position(
            position.position_id,
            market_price=current_price,
            forecast_prob=forecast_prob
        )

        # Check for exit signals
        should_exit, exit_reason = st.session_state.position_manager.should_exit_position(position)

        if should_exit:
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
                # Fallback legacy calculation
                if pos["side"] == "YES":
                    pos["unrealized_pnl"] = pos["size"] * (new_price - pos["entry_price"])
                else:
                    # For NO: use proper shares calculation
                    no_price_at_entry = 1 - pos["entry_price"]
                    shares = pos["size"] / no_price_at_entry if no_price_at_entry > 0 else 0
                    pos["unrealized_pnl"] = (pos["entry_price"] - new_price) * shares

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

    # Demo mode toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Source")
    use_demo = st.sidebar.checkbox(
        "Demo Mode (Simulated Markets)",
        value=True,
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

    # Bankroll setting
    bankroll = st.sidebar.number_input(
        "Bankroll ($)",
        min_value=10.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0
    )
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
        target_date = today_est() + timedelta(days=1)
        st.caption(f"Tomorrow: {target_date.strftime('%A, %B %d, %Y')}")

        if forecasts:
            # Clean card-based layout (only city-level keys, skip date-specific ones)
            for city_key in get_all_cities():
                if city_key not in forecasts:
                    continue
                fc = forecasts[city_key]
                city_config = get_city_config(city_key)
                temp_unit = city_config.temp_unit

                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])

                    with col1:
                        st.markdown(f"### {fc['city']}")
                        st.caption(f"📍 {city_config.station_name}")

                    high_f = fc['high_mean']
                    low_f = fc['low_mean']

                    with col2:
                        st.metric("High", f"{high_f:.1f}°F", f"±{fc['high_std']:.1f}°")

                    with col3:
                        st.metric("Low", f"{low_f:.1f}°F", f"±{fc['low_std']:.1f}°")

                    with col4:
                        st.metric("90% Range", f"{fc['high_ci_lower']:.1f}-{fc['high_ci_upper']:.1f}°F")

                    with col5:
                        conf_pct = fc['confidence'] * 100
                        conf_color = "🟢" if conf_pct >= 80 else "🟡" if conf_pct >= 60 else "🔴"
                        st.metric(
                            "Confidence",
                            f"{conf_color} {conf_pct:.0f}%",
                            f"{fc['model_count']} models"
                        )

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
                    fc = forecasts.get(city_key)

                    if not fc:
                        print(f"[Markets] No forecast for '{city_key}' - available keys: {list(forecasts.keys())}")

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
                        st.caption(f"📊 Forecast: {forecast_high:.1f}°{market_unit} (±{forecast_std:.1f}°) | Mean={forecast_high:.1f}, StdDev={forecast_std:.1f}")

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

        if model_comparison:
            for city_key, data in model_comparison.items():
                city_config = get_city_config(city_key)
                unit_label = "°F"

                st.subheader(f"{data['city']} ({unit_label})")

                if data["models"]:
                    # Merge Tomorrow.io data from forecasts (fetched separately)
                    if city_key in forecasts and "tomorrow.io" not in data["models"]:
                        fc_models = forecasts[city_key].get("models", [])
                        for m in fc_models:
                            if m.get("model") == "tomorrow.io":
                                data["models"]["tomorrow.io"] = {
                                    "high": m["high"],
                                    "low": m["low"]
                                }
                                break

                    models = list(data["models"].keys())
                    highs = [data["models"][m]["high"] for m in models]
                    lows = [data["models"][m]["low"] for m in models]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='High', x=models, y=highs, marker_color='#ff5722',
                                        text=[f"{h:.1f}{unit_label}" for h in highs], textposition='outside'))
                    fig.add_trace(go.Bar(name='Low', x=models, y=lows, marker_color='#2196f3',
                                        text=[f"{l:.1f}{unit_label}" for l in lows], textposition='outside'))

                    if city_key in forecasts:
                        ensemble_high = forecasts[city_key]["high_mean"]
                        fig.add_hline(y=ensemble_high, line_dash="dash",
                                    line_color="red", annotation_text="Ensemble High")

                    fig.update_layout(barmode='group', height=300,
                                    yaxis_title=f"Temperature ({unit_label})", xaxis_title="Model")
                    st.plotly_chart(fig, use_container_width=True)

                    if len(highs) > 1:
                        spread = max(highs) - min(highs)
                        agreement = "High" if spread < 3 else ("Medium" if spread < 6 else "Low")
                        st.metric("Model Agreement", agreement, f"Spread: {spread:.1f}{unit_label}")

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
            header_cols[2].markdown("**Size**")
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

                    cols[0].markdown(f"**{pos['city']}**\n{pos['outcome']}")
                    cols[1].markdown(f"**{pos['side']}**")
                    cols[2].markdown(f"${pos['size']:.2f}")
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
                        entry_edge = pos.get('edge_at_entry', 0)
                        cols[5].markdown(f"{entry_edge:+.1%}")

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

    # Auto-refresh
    if auto_refresh:
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
