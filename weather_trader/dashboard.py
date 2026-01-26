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
    # Keep only last 50 alerts
    st.session_state.alerts = st.session_state.alerts[:50]


def record_price(market_id: str, price: float):
    """Record a price point for historical tracking."""
    if market_id not in st.session_state.price_history:
        st.session_state.price_history[market_id] = []
    st.session_state.price_history[market_id].append({
        "time": datetime.now(),
        "price": price
    })


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
                # Get 7-day forecast
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
                st.warning(f"Failed to fetch {city_config.name}: {e}")

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

            # Fetch from different models
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

            # Try Tomorrow.io if configured
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
                except Exception as e:
                    pass

            comparisons[city_key] = city_data

    return comparisons


@st.cache_data(ttl=60)
def fetch_markets_cached():
    """Fetch Polymarket markets with caching."""
    return run_async(_fetch_markets())


async def _fetch_markets():
    """Fetch weather markets from Polymarket."""
    markets = []

    async with WeatherMarketFinder() as finder:
        try:
            found = await finder.find_weather_markets(active_only=True)
            for m in found:
                markets.append({
                    "city": m.city,
                    "question": m.question,
                    "threshold": m.threshold,
                    "yes_price": m.yes_price,
                    "no_price": m.no_price,
                    "volume": m.volume,
                    "target_date": m.target_date,
                    "market_type": m.market_type.value,
                    "condition_id": m.condition_id,
                })
                # Record price for history
                record_price(m.condition_id, m.yes_price)
        except Exception as e:
            pass

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

                    # Record forecast for accuracy tracking
                    st.session_state.forecast_history.append({
                        "time": datetime.now(),
                        "city": city_key,
                        "date": target_date,
                        "forecast_high": ens.high_mean,
                        "forecast_low": ens.low_mean,
                    })

            except Exception as e:
                st.warning(f"Failed to fetch {city_config.name}: {e}")

    return forecasts


def calculate_signals(forecasts, markets):
    """Calculate trading signals."""
    signals = []

    for market in markets:
        city_key = market["city"].lower()
        if city_key not in forecasts:
            continue

        fc = forecasts[city_key]
        threshold = market["threshold"]
        market_prob = market["yes_price"]

        # Calculate our probability using normal distribution
        our_prob = 1 - stats.norm.cdf(threshold, loc=fc["high_mean"], scale=max(fc["high_std"], 2.0))

        edge = our_prob - market_prob

        # Determine signal
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

        signals.append({
            "city": fc["city"],
            "threshold": threshold,
            "our_prob": our_prob,
            "market_prob": market_prob,
            "edge": edge,
            "confidence": fc["confidence"],
            "forecast_high": fc["high_mean"],
            "forecast_std": fc["high_std"],
            "signal": signal,
            "signal_color": signal_color,
            "signal_strength": abs(edge),
            "condition_id": market["condition_id"],
        })

        # Add alert for strong signals
        if abs(edge) > 0.10 and fc["confidence"] > 0.7:
            add_alert(f"🎯 Strong signal: {fc['city']} - {signal} ({edge:+.1%} edge)", "success")

    return signals


def simulate_trade(signal, size, bankroll):
    """Simulate a trade execution."""
    trade = {
        "time": datetime.now(),
        "city": signal["city"],
        "side": "YES" if signal["edge"] > 0 else "NO",
        "size": size,
        "price": signal["market_prob"] if signal["edge"] > 0 else (1 - signal["market_prob"]),
        "edge": signal["edge"],
        "forecast_prob": signal["our_prob"],
        "market_prob": signal["market_prob"],
        "status": "OPEN",
        "pnl": 0,
    }
    st.session_state.trade_history.insert(0, trade)
    st.session_state.total_trades += 1
    add_alert(f"📈 Trade executed: {signal['city']} {trade['side']} ${size:.2f} @ {trade['price']:.2f}", "info")
    return trade


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

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        st.sidebar.info("Data refreshes every 5 minutes")

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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Mode:** {'🔴 LIVE' if is_live else '🔒 DRY RUN'}")
    with col2:
        st.markdown(f"**Bankroll:** ${bankroll:,.2f}")
    with col3:
        st.markdown(f"**P&L:** ${st.session_state.pnl:+.2f}")
    with col4:
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
        markets = fetch_markets_cached()
        multi_day = fetch_multi_day_forecasts()
        model_comparison = fetch_model_comparison()

    signals = calculate_signals(forecasts, markets) if markets else []

    # =====================
    # TAB 1: Overview
    # =====================
    with tab1:
        st.header("Trading Overview")

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

            # Display as formatted table
            for _, row in df_signals.iterrows():
                with st.container():
                    cols = st.columns([2, 1, 1, 1, 1, 1, 2])
                    cols[0].markdown(f"**{row['city']}**")
                    cols[1].markdown(f"Threshold: {row['threshold']:.0f}°F")
                    cols[2].markdown(f"Forecast: {row['forecast_high']:.1f}°F")
                    cols[3].markdown(f"Our: {row['our_prob']:.0%}")
                    cols[4].markdown(f"Market: {row['market_prob']:.0%}")
                    cols[5].markdown(f"Edge: **{row['edge']:+.1%}**")

                    if row['signal'] != "PASS":
                        if cols[6].button(f"{row['signal']}", key=f"trade_{row['city']}"):
                            # Calculate position size
                            kelly = max(0, (row['our_prob'] - row['market_prob']) / (1 - row['market_prob']))
                            position = min(bankroll * kelly * kelly_fraction, bankroll * max_position / 100)
                            if position > 1:
                                simulate_trade(row, position, bankroll)
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
            fig.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="Min Edge")
            fig.add_hline(y=-5, line_dash="dash", line_color="red")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active markets found. Markets may be closed or unavailable.")

    # =====================
    # TAB 2: Forecasts
    # =====================
    with tab2:
        st.header("🌡️ Weather Forecasts")
        st.markdown(f"**Forecast Date:** {date.today() + timedelta(days=1)}")

        if forecasts:
            # Forecast cards in a grid
            cols = st.columns(min(len(forecasts), 3))

            for i, (city_key, fc) in enumerate(forecasts.items()):
                with cols[i % 3]:
                    st.markdown(f"### {fc['city']}")

                    # Temperature gauge
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
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': fc['high_mean']
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                    st.metric("Low", f"{fc['low_mean']:.1f}°F", f"±{fc['low_std']:.1f}°F", delta_color="off")

                    # Confidence bar
                    conf_color = "green" if fc['confidence'] > 0.7 else ("orange" if fc['confidence'] > 0.5 else "red")
                    st.progress(fc['confidence'], text=f"Confidence: {fc['confidence']:.0%}")
                    st.caption(f"{fc['model_count']} models • 90% CI: {fc['high_ci_lower']:.0f}-{fc['high_ci_upper']:.0f}°F")

                    st.markdown("---")

            # Temperature comparison chart
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

        if markets:
            st.success(f"Found {len(markets)} active markets")

            for market in markets:
                with st.expander(f"**{market['city'].upper()}** - Over {market['threshold']}°F", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("YES Price", f"${market['yes_price']:.2f}",
                                help="Price to buy YES (probability market assigns)")
                    with col2:
                        st.metric("NO Price", f"${market['no_price']:.2f}")
                    with col3:
                        st.metric("Volume", f"${market['volume']:,.0f}")
                    with col4:
                        # Compare with forecast
                        city_key = market["city"].lower()
                        if city_key in forecasts:
                            our_prob = 1 - stats.norm.cdf(market["threshold"],
                                                          loc=forecasts[city_key]["high_mean"],
                                                          scale=max(forecasts[city_key]["high_std"], 2))
                            edge = our_prob - market["yes_price"]
                            st.metric("Our Edge", f"{edge:+.1%}",
                                    delta=f"We say {our_prob:.0%}")

                    st.markdown(f"**Question:** {market['question']}")
                    st.markdown(f"**Target Date:** {market['target_date']} | **Type:** {market['market_type']}")
        else:
            st.warning("No active weather markets found on Polymarket")
            st.info("""
            **Why no markets?**
            - Weather markets may not be active right now
            - Markets might use different question formats
            - Try refreshing later

            The system will automatically detect markets when they become available.
            """)

    # =====================
    # TAB 4: Model Comparison
    # =====================
    with tab4:
        st.header("🔬 Model Comparison")
        st.markdown("Compare forecasts from different weather models")

        if model_comparison:
            for city_key, data in model_comparison.items():
                st.subheader(data["city"])

                if data["models"]:
                    # Create comparison chart
                    models = list(data["models"].keys())
                    highs = [data["models"][m]["high"] for m in models]
                    lows = [data["models"][m]["low"] for m in models]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='High', x=models, y=highs, marker_color='#ff5722'))
                    fig.add_trace(go.Bar(name='Low', x=models, y=lows, marker_color='#2196f3'))

                    # Add ensemble line
                    if city_key in forecasts:
                        fig.add_hline(y=forecasts[city_key]["high_mean"], line_dash="dash",
                                    line_color="red", annotation_text="Ensemble High")

                    fig.update_layout(barmode='group', height=300,
                                    yaxis_title="Temperature (°F)",
                                    xaxis_title="Model")
                    st.plotly_chart(fig, use_container_width=True)

                    # Model agreement metric
                    if len(highs) > 1:
                        spread = max(highs) - min(highs)
                        agreement = "High" if spread < 3 else ("Medium" if spread < 6 else "Low")
                        st.metric("Model Agreement", agreement, f"Spread: {spread:.1f}°F")
                else:
                    st.info("No model data available")

                st.markdown("---")

    # =====================
    # TAB 5: Multi-Day Forecasts
    # =====================
    with tab5:
        st.header("📅 5-Day Forecast")

        if multi_day:
            city_select = st.selectbox("Select City", list(multi_day.keys()),
                                      format_func=lambda x: multi_day[x]["city"])

            if city_select:
                data = multi_day[city_select]
                df = pd.DataFrame(data["daily"])

                # Line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['date'], y=df['high'], name='High',
                                        line=dict(color='#ff5722', width=3),
                                        mode='lines+markers'))
                fig.add_trace(go.Scatter(x=df['date'], y=df['low'], name='Low',
                                        line=dict(color='#2196f3', width=3),
                                        mode='lines+markers'))

                # Fill between
                fig.add_trace(go.Scatter(
                    x=list(df['date']) + list(df['date'])[::-1],
                    y=list(df['high']) + list(df['low'])[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 87, 34, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False
                ))

                fig.update_layout(
                    title=f"{data['city']} - 7 Day Forecast",
                    xaxis_title="Date",
                    yaxis_title="Temperature (°F)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Data table
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
                # P&L over time
                df_trades = pd.DataFrame(st.session_state.trade_history)

                # Simple P&L simulation (random for demo)
                cumulative_pnl = []
                running = 0
                for t in st.session_state.trade_history[::-1]:
                    # Simulate outcome based on edge
                    outcome = np.random.random() < (0.5 + t['edge'])
                    pnl = t['size'] * (1 - t['price']) if outcome else -t['size'] * t['price']
                    running += pnl
                    cumulative_pnl.append(running)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=cumulative_pnl, mode='lines+markers',
                                        line=dict(color='green' if cumulative_pnl[-1] > 0 else 'red')))
                fig.update_layout(title="Cumulative P&L", yaxis_title="$", height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trades yet. Execute trades from the Overview tab.")

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
        else:
            st.info("No trades recorded this session")

    # =====================
    # TAB 7: Price Charts
    # =====================
    with tab7:
        st.header("📉 Price History")

        if st.session_state.price_history:
            for market_id, prices in st.session_state.price_history.items():
                if len(prices) > 1:
                    df = pd.DataFrame(prices)
                    fig = px.line(df, x='time', y='price', title=f"Market: {market_id[:20]}...")
                    fig.update_layout(yaxis_tickformat='.0%', height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price history will accumulate as you refresh data. Each refresh records current prices.")

            # Demo chart
            st.subheader("Demo: What price tracking looks like")
            demo_times = pd.date_range(end=datetime.now(), periods=20, freq='30min')
            demo_prices = 0.5 + np.cumsum(np.random.randn(20) * 0.02)
            demo_prices = np.clip(demo_prices, 0.1, 0.9)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=demo_times, y=demo_prices, mode='lines',
                                    line=dict(color='#2196f3')))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(title="Demo: YES Price Over Time",
                            yaxis_title="Price ($)", yaxis_tickformat='.2f',
                            height=300)
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
                level_colors = {
                    "info": "#e3f2fd",
                    "success": "#e8f5e9",
                    "warning": "#fff3e0",
                    "danger": "#ffebee"
                }
                level_icons = {
                    "info": "ℹ️",
                    "success": "✅",
                    "warning": "⚠️",
                    "danger": "🚨"
                }

                st.markdown(f"""
                <div style="background-color: {level_colors.get(alert['level'], '#f5f5f5')};
                            padding: 10px; border-radius: 5px; margin: 5px 0;">
                    {level_icons.get(alert['level'], '📢')}
                    <strong>{alert['time'].strftime('%H:%M:%S')}</strong> - {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alerts yet. Alerts appear when:")
            st.markdown("""
            - Strong trading signals are detected (>10% edge)
            - Trades are executed
            - Data is refreshed
            - Errors occur
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        f"*Weather Trader v0.2.0* | "
        f"*Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}* | "
        f"*{len(forecasts)} cities • {len(markets)} markets • {len(signals)} signals*"
    )

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(300)  # 5 minutes
        st.rerun()


if __name__ == "__main__":
    main()
