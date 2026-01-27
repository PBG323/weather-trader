"""
Weather Trader Dashboard - Full Featured

A comprehensive Streamlit-based UI for the weather trading system.
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import scipy.stats as stats
import time
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather_trader.config import get_all_cities, get_city_config, config
from weather_trader.apis import OpenMeteoClient, TomorrowIOClient, NWSClient
from weather_trader.models import EnsembleForecaster, BiasCorrector
from weather_trader.models.ensemble import ModelForecast
from weather_trader.polymarket import WeatherMarketFinder, PolymarketAuth
from weather_trader.strategy import ExpectedValueCalculator

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
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'price_history' not in st.session_state:
    st.session_state.price_history = {}
if 'forecast_history' not in st.session_state:
    st.session_state.forecast_history = []
if 'pnl' not in st.session_state:
    st.session_state.pnl = 0.0
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
        "time": datetime.now(),
        "message": message,
        "level": level
    })
    st.session_state.alerts = st.session_state.alerts[:50]


def record_price(market_id: str, price: float):
    """Record a price point for historical tracking."""
    if market_id not in st.session_state.price_history:
        st.session_state.price_history[market_id] = []
    st.session_state.price_history[market_id].append({
        "time": datetime.now(),
        "price": price
    })


def generate_demo_markets(forecasts):
    """Generate demo markets based on real forecasts for testing."""
    markets = []
    target_date = date.today() + timedelta(days=1)

    for city_key, fc in forecasts.items():
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

            condition_id = f"demo_{city_key}_{temp_low}_{temp_high}"

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
                "temp_unit": city_config.temp_unit,
                "condition_id": condition_id,
                "resolution_source": f"Demo - {city_config.station_name}",
                "is_demo": True,
            })

            record_price(condition_id, market_prob)

    return markets


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
def fetch_model_comparison():
    """Fetch forecasts from different models for comparison."""
    return run_async(_fetch_model_comparison())


async def _fetch_model_comparison():
    """Fetch from multiple models."""
    from weather_trader.apis.open_meteo import WeatherModel

    comparisons = {}
    target_date = date.today() + timedelta(days=1)

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
                except:
                    pass

            if config.api.tomorrow_io_api_key and "your_" not in config.api.tomorrow_io_api_key.lower():
                try:
                    async with TomorrowIOClient() as tm_client:
                        tm_forecasts = await tm_client.get_daily_forecast(city_config, days=3)
                        for f in tm_forecasts:
                            if f.timestamp.date() == target_date:
                                city_data["models"]["tomorrow.io"] = {
                                    "high": f.temperature_high,
                                    "low": f.temperature_low
                                }
                                break
                except:
                    pass

            comparisons[city_key] = city_data

    return comparisons


@st.cache_data(ttl=60)
def fetch_real_markets():
    """Fetch real Polymarket markets."""
    return run_async(_fetch_real_markets())


async def _fetch_real_markets():
    """Fetch weather markets from Polymarket."""
    markets = []

    async with WeatherMarketFinder() as finder:
        try:
            found = await finder.find_weather_markets(active_only=True, days_ahead=3)
            for market in found:
                # Each market has multiple temperature outcomes
                for outcome in market.outcomes:
                    markets.append({
                        "city": market.city,
                        "city_name": market.city_config.name if market.city_config else market.city,
                        "question": market.question,
                        "outcome_desc": outcome.description,
                        "temp_low": outcome.temp_low,
                        "temp_high": outcome.temp_high,
                        "temp_midpoint": outcome.midpoint,
                        "yes_price": outcome.yes_price,
                        "no_price": 1 - outcome.yes_price,
                        "volume": outcome.volume,
                        "liquidity": outcome.liquidity,
                        "target_date": market.target_date,
                        "temp_unit": market.temp_unit,
                        "condition_id": outcome.condition_id,
                        "outcome_id": outcome.outcome_id,
                        "yes_token_id": outcome.yes_token_id,
                        "resolution_source": market.resolution_source,
                        "is_demo": False,
                        "event_slug": market.event_slug,
                        "total_market_volume": market.total_volume,
                    })
                    record_price(outcome.condition_id, outcome.yes_price)
        except Exception as e:
            add_alert(f"Error fetching markets: {str(e)}", "warning")

    return markets


@st.cache_data(ttl=300)
def fetch_forecasts_with_models():
    """Fetch forecasts with individual model data."""
    return run_async(_fetch_forecasts_with_models())


async def _fetch_forecasts_with_models():
    """Fetch forecasts for all cities with model breakdown."""
    forecasts = {}
    target_date = date.today() + timedelta(days=1)

    async with OpenMeteoClient() as client:
        bias_corrector = BiasCorrector()
        ensemble = EnsembleForecaster(bias_corrector)

        for city_key in get_all_cities():
            city_config = get_city_config(city_key)
            try:
                om_forecasts = await client.get_ensemble_forecast(city_config, days=3)

                model_forecasts = []
                model_details = []

                for model_name, forecast_list in om_forecasts.items():
                    for f in forecast_list:
                        if f.timestamp.date() == target_date:
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
                            break

                if model_forecasts:
                    ens = ensemble.create_ensemble(
                        city_config, model_forecasts, target_date,
                        apply_bias_correction=False
                    )
                    forecasts[city_key] = {
                        "city": city_config.name,
                        "high_mean": ens.high_mean,
                        "high_std": ens.high_std,
                        "low_mean": ens.low_mean,
                        "low_std": ens.low_std,
                        "confidence": ens.confidence,
                        "model_count": ens.model_count,
                        "date": target_date,
                        "models": model_details,
                        "high_ci_lower": ens.high_ci_lower,
                        "high_ci_upper": ens.high_ci_upper,
                    }

            except Exception as e:
                pass

    return forecasts


def calculate_signals(forecasts, markets):
    """Calculate trading signals for multi-outcome markets."""
    signals = []
    seen_markets = set()  # Track which city/date combos we've processed

    # Group markets by city and date
    market_groups = {}
    for market in markets:
        city_key = market["city"].lower()
        target_date = market.get("target_date", date.today())
        group_key = f"{city_key}_{target_date}"
        if group_key not in market_groups:
            market_groups[group_key] = []
        market_groups[group_key].append(market)

    for group_key, group_markets in market_groups.items():
        if not group_markets:
            continue

        city_key = group_markets[0]["city"].lower()
        if city_key not in forecasts:
            continue

        fc = forecasts[city_key]
        forecast_temp = fc["high_mean"]
        forecast_std = max(fc["high_std"], 2.0)
        temp_unit = group_markets[0].get("temp_unit", "F")
        city_config = get_city_config(city_key)

        # Convert forecast to Celsius if needed
        if temp_unit == "C" and city_config.temp_unit == "C":
            # Forecast is in F, market is in C - convert forecast
            forecast_temp_market = (forecast_temp - 32) * 5/9
            forecast_std_market = forecast_std * 5/9
        else:
            forecast_temp_market = forecast_temp
            forecast_std_market = forecast_std

        # Find the best opportunity across all outcomes
        best_signal = None
        best_edge = 0

        for market in group_markets:
            temp_low = market.get("temp_low")
            temp_high = market.get("temp_high")
            market_prob = market["yes_price"]

            # Calculate our probability for this outcome
            if temp_low is None and temp_high is not None:
                # "≤X" outcome: P(temp <= X)
                our_prob = stats.norm.cdf(temp_high, loc=forecast_temp_market, scale=forecast_std_market)
            elif temp_high is None and temp_low is not None:
                # "≥X" outcome: P(temp >= X)
                our_prob = 1 - stats.norm.cdf(temp_low, loc=forecast_temp_market, scale=forecast_std_market)
            elif temp_low is not None and temp_high is not None:
                # "X-Y" range: P(X <= temp <= Y)
                prob_high = stats.norm.cdf(temp_high, loc=forecast_temp_market, scale=forecast_std_market)
                prob_low = stats.norm.cdf(temp_low, loc=forecast_temp_market, scale=forecast_std_market)
                our_prob = prob_high - prob_low
            else:
                continue

            edge = our_prob - market_prob

            # Keep track of best opportunity
            if abs(edge) > abs(best_edge):
                best_edge = edge
                best_signal = {
                    "city": fc["city"],
                    "city_key": city_key,
                    "outcome": market.get("outcome_desc", f"{temp_low}-{temp_high}"),
                    "temp_low": temp_low,
                    "temp_high": temp_high,
                    "temp_unit": temp_unit,
                    "our_prob": our_prob,
                    "market_prob": market_prob,
                    "edge": edge,
                    "confidence": fc["confidence"],
                    "forecast_high": forecast_temp,
                    "forecast_std": fc["high_std"],
                    "condition_id": market["condition_id"],
                    "yes_token_id": market.get("yes_token_id", ""),
                    "is_demo": market.get("is_demo", False),
                    "target_date": market.get("target_date"),
                    "resolution_source": market.get("resolution_source", ""),
                    "volume": market.get("volume", 0),
                }

        if best_signal:
            # Determine signal type
            edge = best_signal["edge"]
            if edge > 0.10:
                signal = "STRONG BUY YES"
                signal_color = "#00c853"
            elif edge > 0.05:
                signal = "BUY YES"
                signal_color = "#4caf50"
            elif edge < -0.10:
                signal = "STRONG BUY NO"
                signal_color = "#f44336"
            elif edge < -0.05:
                signal = "BUY NO"
                signal_color = "#ff5722"
            else:
                signal = "PASS"
                signal_color = "#9e9e9e"

            best_signal["signal"] = signal
            best_signal["signal_color"] = signal_color
            best_signal["signal_strength"] = abs(edge)

            signals.append(best_signal)

            if abs(edge) > 0.10 and best_signal["confidence"] > 0.7:
                add_alert(f"🎯 Strong signal: {best_signal['city']} {best_signal['outcome']} - {signal} ({edge:+.1%} edge)", "success")

    return signals


def execute_trade(signal, size, is_live=False):
    """Execute a trade (simulated or real)."""
    outcome = signal.get("outcome", "")
    trade = {
        "time": datetime.now(),
        "city": signal["city"],
        "outcome": outcome,
        "side": "YES" if signal["edge"] > 0 else "NO",
        "size": size,
        "price": signal["market_prob"] if signal["edge"] > 0 else (1 - signal["market_prob"]),
        "edge": signal["edge"],
        "forecast_prob": signal["our_prob"],
        "market_prob": signal["market_prob"],
        "status": "SIMULATED" if not is_live else "LIVE",
        "is_demo": signal.get("is_demo", False),
        "condition_id": signal.get("condition_id", ""),
        "target_date": signal.get("target_date"),
        "pnl": 0,
    }
    st.session_state.trade_history.insert(0, trade)
    st.session_state.total_trades += 1

    mode = "LIVE" if is_live else "SIMULATED"
    add_alert(f"📈 [{mode}] Trade: {signal['city']} {outcome} {trade['side']} ${size:.2f} @ {trade['price']:.2f}",
              "success" if is_live else "info")
    return trade


def auto_trade_check(signals, bankroll, kelly_fraction, max_position, min_edge, is_live):
    """Check signals and execute trades automatically."""
    if not st.session_state.auto_trade_enabled:
        return

    trades_made = 0
    for signal in signals:
        if signal["signal"] == "PASS":
            continue

        if abs(signal["edge"]) < min_edge:
            continue

        if signal["confidence"] < 0.65:
            continue

        # Calculate position size
        kelly = max(0, abs(signal["edge"]) / (1 - signal["market_prob"])) if signal["market_prob"] < 1 else 0
        position = min(bankroll * kelly * kelly_fraction, bankroll * max_position / 100)

        if position < 1:
            continue

        # Check if we already traded this market recently
        recent_trades = [t for t in st.session_state.trade_history
                        if t["city"] == signal["city"]
                        and (datetime.now() - t["time"]).seconds < 3600]
        if recent_trades:
            continue

        execute_trade(signal, position, is_live)
        trades_made += 1

    if trades_made > 0:
        add_alert(f"🤖 Auto-trader executed {trades_made} trade(s)", "success")

    st.session_state.last_auto_trade_check = datetime.now()


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
        st.sidebar.error("⚠️ LIVE TRADING ENABLED")

    # Demo mode toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Source")
    use_demo = st.sidebar.checkbox(
        "Demo Mode (Simulated Markets)",
        value=True,
        help="Use simulated markets for testing. Disable when real Polymarket weather markets are available."
    )
    st.session_state.demo_mode = use_demo

    if use_demo:
        st.sidebar.info("Using simulated markets for demo")
    else:
        st.sidebar.warning("Looking for real Polymarket markets")

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
            st.sidebar.caption(f"Last check: {st.session_state.last_auto_trade_check.strftime('%H:%M:%S')}")
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

    # Risk settings
    st.sidebar.markdown("### ⚙️ Risk Settings")
    kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)
    max_position = st.sidebar.slider("Max Position (%)", 1, 20, 5)
    min_edge = st.sidebar.slider("Min Edge (%)", 1, 20, 5) / 100

    st.sidebar.markdown("---")

    # Quick stats
    st.sidebar.markdown("### 📊 Session Stats")
    st.sidebar.metric("P&L", f"${st.session_state.pnl:+.2f}")
    win_rate = (st.session_state.winning_trades / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0
    st.sidebar.metric("Win Rate", f"{win_rate:.0f}%")
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
        st.markdown(f"**Updated:** {datetime.now().strftime('%H:%M:%S')}")

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

    signals = calculate_signals(forecasts, markets) if markets and forecasts else []

    # Auto-trade check
    if auto_trade and signals:
        auto_trade_check(signals, bankroll, kelly_fraction, max_position, min_edge, is_live)

    # =====================
    # TAB 1: Overview
    # =====================
    with tab1:
        st.header("Trading Overview")

        # Market status banner
        if use_demo:
            st.info("📊 **Demo Mode Active** - Using simulated markets based on real weather forecasts. Toggle off 'Demo Mode' in sidebar when real Polymarket weather markets are available.")
        else:
            if markets:
                st.success(f"✅ **Live Markets Found** - {len(markets)} active weather markets on Polymarket")
            else:
                st.warning("⚠️ **No Live Markets** - No weather markets currently active on Polymarket. Enable 'Demo Mode' to test the system.")

        # Auto-trade status
        if auto_trade:
            st.markdown("""
            <div class="auto-trade-active">
                <strong>🤖 Auto-Trading Enabled</strong><br>
                The bot will automatically execute trades when:
                <ul>
                    <li>Edge exceeds minimum threshold</li>
                    <li>Confidence is above 65%</li>
                    <li>Position size meets minimum requirements</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

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

            df_signals = pd.DataFrame(signals)
            df_signals = df_signals.sort_values("signal_strength", ascending=False)

            for _, row in df_signals.iterrows():
                with st.container():
                    cols = st.columns([2, 1.5, 1, 1, 1, 1, 2])

                    city_label = f"**{row['city']}**"
                    if row.get('is_demo'):
                        city_label += " 🎮"
                    cols[0].markdown(city_label)
                    temp_unit = row.get('temp_unit', 'F')
                    outcome = row.get('outcome', f"{row.get('temp_low', '?')}-{row.get('temp_high', '?')}")
                    cols[1].markdown(f"Range: {outcome}")
                    cols[2].markdown(f"Fcst: {row['forecast_high']:.1f}°F")
                    cols[3].markdown(f"Our: {row['our_prob']:.0%}")
                    cols[4].markdown(f"Mkt: {row['market_prob']:.0%}")

                    edge_color = "green" if row['edge'] > 0 else "red"
                    cols[5].markdown(f"Edge: **:{edge_color}[{row['edge']:+.1%}]**")

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
        st.markdown(f"**Forecast Date:** {date.today() + timedelta(days=1)}")

        if forecasts:
            cols = st.columns(min(len(forecasts), 3))

            for i, (city_key, fc) in enumerate(forecasts.items()):
                with cols[i % 3]:
                    st.markdown(f"### {fc['city']}")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fc['high_mean'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "High °F"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ff5722"},
                            'steps': [
                                {'range': [0, 32], 'color': "#e3f2fd"},
                                {'range': [32, 50], 'color': "#bbdefb"},
                                {'range': [50, 70], 'color': "#fff9c4"},
                                {'range': [70, 85], 'color': "#ffcc80"},
                                {'range': [85, 100], 'color': "#ffab91"},
                            ],
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                    st.metric("Low", f"{fc['low_mean']:.1f}°F", f"±{fc['low_std']:.1f}°F", delta_color="off")
                    st.progress(fc['confidence'], text=f"Confidence: {fc['confidence']:.0%}")
                    st.caption(f"{fc['model_count']} models • 90% CI: {fc['high_ci_lower']:.0f}-{fc['high_ci_upper']:.0f}°F")
                    st.markdown("---")

            # Temperature comparison
            st.subheader("Temperature Comparison")
            df_temps = pd.DataFrame([
                {"City": fc["city"], "High": fc["high_mean"], "Low": fc["low_mean"],
                 "High_err": fc["high_std"], "Low_err": fc["low_std"]}
                for fc in forecasts.values()
            ])

            fig = go.Figure()
            fig.add_trace(go.Bar(name='High', x=df_temps['City'], y=df_temps['High'],
                                error_y=dict(type='data', array=df_temps['High_err']),
                                marker_color='#ff5722'))
            fig.add_trace(go.Bar(name='Low', x=df_temps['City'], y=df_temps['Low'],
                                error_y=dict(type='data', array=df_temps['Low_err']),
                                marker_color='#2196f3'))
            fig.update_layout(barmode='group', title="Forecasted Temperatures with Uncertainty",
                            yaxis_title="Temperature (°F)", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # =====================
    # TAB 3: Markets
    # =====================
    with tab3:
        st.header("📈 Polymarket Weather Markets")

        if use_demo:
            st.info("🎮 **Demo Markets** - These are simulated markets based on real forecasts. Disable 'Demo Mode' in sidebar to search for real Polymarket markets.")
        else:
            st.info("🔍 **Live Markets** - Fetching from Polymarket API. Markets are grouped by city/date.")

        if markets:
            # Group markets by city and date for display
            market_groups = {}
            for market in markets:
                city = market.get("city_name", market["city"])
                target_date = market.get("target_date", date.today())
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

                    # Show all outcomes as a table
                    outcome_data = []
                    city_key = group["city_key"].lower()
                    fc = forecasts.get(city_key, {})

                    for o in sorted(group["outcomes"], key=lambda x: x.get("temp_midpoint", 0)):
                        outcome_desc = o.get("outcome_desc", "")
                        yes_price = o.get("yes_price", 0)
                        volume = o.get("volume", 0)

                        # Calculate our probability
                        our_prob = None
                        if fc:
                            temp_low = o.get("temp_low")
                            temp_high = o.get("temp_high")
                            forecast_high = fc.get("high_mean", 70)
                            forecast_std = max(fc.get("high_std", 3), 2)

                            # Convert if needed (F to C for Toronto/London)
                            temp_unit = o.get("temp_unit", "F")
                            if temp_unit == "C":
                                forecast_high = (forecast_high - 32) * 5/9
                                forecast_std = forecast_std * 5/9

                            if temp_low is None and temp_high is not None:
                                our_prob = stats.norm.cdf(temp_high, loc=forecast_high, scale=forecast_std)
                            elif temp_high is None and temp_low is not None:
                                our_prob = 1 - stats.norm.cdf(temp_low, loc=forecast_high, scale=forecast_std)
                            elif temp_low is not None and temp_high is not None:
                                our_prob = stats.norm.cdf(temp_high, loc=forecast_high, scale=forecast_std) - \
                                          stats.norm.cdf(temp_low, loc=forecast_high, scale=forecast_std)

                        edge = (our_prob - yes_price) if our_prob else None

                        outcome_data.append({
                            "Outcome": outcome_desc,
                            "Market": f"{yes_price:.1%}",
                            "Ours": f"{our_prob:.1%}" if our_prob else "N/A",
                            "Edge": f"{edge:+.1%}" if edge else "N/A",
                            "Volume": f"${volume:,.0f}"
                        })

                    df_outcomes = pd.DataFrame(outcome_data)
                    st.dataframe(df_outcomes, use_container_width=True, hide_index=True)

                    # Show forecast context
                    if fc:
                        st.caption(f"📊 Our forecast: {fc.get('high_mean', 0):.1f}°F (±{fc.get('high_std', 0):.1f}°F)")
        else:
            st.warning("No markets found")
            st.info("Enable Demo Mode in the sidebar to test with simulated markets, or check if Polymarket has active weather markets.")

    # =====================
    # TAB 4: Model Comparison
    # =====================
    with tab4:
        st.header("🔬 Model Comparison")

        if model_comparison:
            for city_key, data in model_comparison.items():
                st.subheader(data["city"])

                if data["models"]:
                    models = list(data["models"].keys())
                    highs = [data["models"][m]["high"] for m in models]
                    lows = [data["models"][m]["low"] for m in models]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='High', x=models, y=highs, marker_color='#ff5722'))
                    fig.add_trace(go.Bar(name='Low', x=models, y=lows, marker_color='#2196f3'))

                    if city_key in forecasts:
                        fig.add_hline(y=forecasts[city_key]["high_mean"], line_dash="dash",
                                    line_color="red", annotation_text="Ensemble High")

                    fig.update_layout(barmode='group', height=300,
                                    yaxis_title="Temperature (°F)", xaxis_title="Model")
                    st.plotly_chart(fig, use_container_width=True)

                    if len(highs) > 1:
                        spread = max(highs) - min(highs)
                        agreement = "High" if spread < 3 else ("Medium" if spread < 6 else "Low")
                        st.metric("Model Agreement", agreement, f"Spread: {spread:.1f}°F")

                st.markdown("---")

    # =====================
    # TAB 5: Multi-Day
    # =====================
    with tab5:
        st.header("📅 7-Day Forecast")

        if multi_day:
            city_select = st.selectbox("Select City", list(multi_day.keys()),
                                      format_func=lambda x: multi_day[x]["city"])

            if city_select:
                data = multi_day[city_select]
                df = pd.DataFrame(data["daily"])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['date'], y=df['high'], name='High',
                                        line=dict(color='#ff5722', width=3), mode='lines+markers'))
                fig.add_trace(go.Scatter(x=df['date'], y=df['low'], name='Low',
                                        line=dict(color='#2196f3', width=3), mode='lines+markers'))
                fig.add_trace(go.Scatter(
                    x=list(df['date']) + list(df['date'])[::-1],
                    y=list(df['high']) + list(df['low'])[::-1],
                    fill='toself', fillcolor='rgba(255, 87, 34, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'), showlegend=False
                ))
                fig.update_layout(title=f"{data['city']} - 7 Day Forecast",
                                xaxis_title="Date", yaxis_title="Temperature (°F)",
                                height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df.style.format({"high": "{:.1f}°F", "low": "{:.1f}°F"}),
                           use_container_width=True)

    # =====================
    # TAB 6: Trades & P/L
    # =====================
    with tab6:
        st.header("💰 Trades & P/L")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Position Calculator")
            calc_prob = st.slider("Your Probability (%)", 1, 99, 60) / 100
            calc_market = st.slider("Market Price (%)", 1, 99, 50) / 100

            edge = calc_prob - calc_market
            kelly = max(0, edge / (1 - calc_market)) if calc_market < 1 else 0
            adj_kelly = kelly * kelly_fraction
            position = min(bankroll * adj_kelly, bankroll * max_position / 100)

            st.metric("Edge", f"{edge:+.1%}")
            st.metric("Kelly Suggests", f"{kelly:.1%} of bankroll")
            st.metric("Position Size", f"${position:.2f}")

            if edge > min_edge:
                st.success(f"✅ Trade recommended: ${position:.2f}")
            else:
                st.warning(f"⚠️ Edge below minimum ({min_edge:.0%})")

        with col2:
            st.subheader("P&L Tracking")

            if st.session_state.trade_history:
                cumulative_pnl = []
                running = 0
                for t in st.session_state.trade_history[::-1]:
                    outcome = random.random() < (0.5 + t['edge'])
                    pnl = t['size'] * (1 - t['price']) if outcome else -t['size'] * t['price']
                    running += pnl
                    cumulative_pnl.append(running)

                st.session_state.pnl = cumulative_pnl[-1] if cumulative_pnl else 0

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=cumulative_pnl, mode='lines+markers',
                                        line=dict(color='green' if cumulative_pnl[-1] > 0 else 'red')))
                fig.update_layout(title="Cumulative P&L (Simulated)", yaxis_title="$", height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trades yet")

        st.markdown("---")
        st.subheader("Trade History")

        if st.session_state.trade_history:
            df_hist = pd.DataFrame(st.session_state.trade_history)
            st.dataframe(df_hist[['time', 'city', 'side', 'size', 'price', 'edge', 'status']]
                        .style.format({
                            'size': '${:.2f}',
                            'price': '{:.2f}',
                            'edge': '{:+.1%}'
                        }), use_container_width=True)

            if st.button("Clear Trade History"):
                st.session_state.trade_history = []
                st.session_state.total_trades = 0
                st.session_state.winning_trades = 0
                st.session_state.pnl = 0
                st.rerun()
        else:
            st.info("No trades recorded")

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
            demo_times = pd.date_range(end=datetime.now(), periods=20, freq='30min')
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
            for alert in st.session_state.alerts[:20]:
                level_colors = {"info": "#e3f2fd", "success": "#e8f5e9", "warning": "#fff3e0", "danger": "#ffebee"}
                level_icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "danger": "🚨"}

                st.markdown(f"""
                <div style="background-color: {level_colors.get(alert['level'], '#f5f5f5')};
                            padding: 10px; border-radius: 5px; margin: 5px 0;">
                    {level_icons.get(alert['level'], '📢')}
                    <strong>{alert['time'].strftime('%H:%M:%S')}</strong> - {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alerts yet")

    # Footer
    st.markdown("---")
    market_status = "Demo Markets" if use_demo else ("Live Markets" if markets else "No Markets")
    st.markdown(
        f"*Weather Trader v0.3.0* | "
        f"*{market_status}* | "
        f"*{len(forecasts)} cities • {len(signals)} signals* | "
        f"*Auto-Trade: {'ON' if auto_trade else 'OFF'}*"
    )

    # Auto-refresh
    if auto_refresh:
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
