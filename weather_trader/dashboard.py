"""
Weather Trader Dashboard

A Streamlit-based UI for monitoring and controlling the trading system.
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go

from .config import get_all_cities, get_city_config, config
from .apis import OpenMeteoClient
from .models import EnsembleForecaster, BiasCorrector
from .models.ensemble import ModelForecast
from .polymarket import WeatherMarketFinder, PolymarketAuth
from .strategy import ExpectedValueCalculator

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
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .neutral { color: #ffc107; }
</style>
""", unsafe_allow_html=True)


def run_async(coro):
    """Helper to run async functions in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_forecasts_cached():
    """Fetch forecasts with caching."""
    return run_async(_fetch_all_forecasts())


async def _fetch_all_forecasts():
    """Fetch forecasts for all cities."""
    forecasts = {}
    target_date = date.today() + timedelta(days=1)

    async with OpenMeteoClient() as client:
        bias_corrector = BiasCorrector()
        ensemble = EnsembleForecaster(bias_corrector)

        for city_key in get_all_cities():
            city_config = get_city_config(city_key)
            try:
                # Fetch from multiple models
                om_forecasts = await client.get_ensemble_forecast(city_config, days=3)

                model_forecasts = []
                for model_name, forecast_list in om_forecasts.items():
                    for f in forecast_list:
                        if f.timestamp.date() == target_date:
                            model_forecasts.append(ModelForecast(
                                model_name=model_name,
                                forecast_high=f.temperature_high,
                                forecast_low=f.temperature_low,
                            ))
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
                    }
            except Exception as e:
                st.warning(f"Failed to fetch {city_config.name}: {e}")

    return forecasts


@st.cache_data(ttl=60)  # Cache for 1 minute
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
        except Exception as e:
            st.error(f"Failed to fetch markets: {e}")

    return markets


def calculate_signals(forecasts, markets):
    """Calculate trading signals."""
    signals = []
    ev_calc = ExpectedValueCalculator()

    for market in markets:
        city_key = market["city"].lower()
        if city_key not in forecasts:
            continue

        fc = forecasts[city_key]
        threshold = market["threshold"]
        market_prob = market["yes_price"]

        # Calculate our probability
        # Simple normal distribution approximation
        import scipy.stats as stats
        our_prob = 1 - stats.norm.cdf(threshold, loc=fc["high_mean"], scale=max(fc["high_std"], 2.0))

        edge = our_prob - market_prob

        signals.append({
            "city": fc["city"],
            "threshold": threshold,
            "our_prob": our_prob,
            "market_prob": market_prob,
            "edge": edge,
            "confidence": fc["confidence"],
            "forecast_high": fc["high_mean"],
            "signal": "BUY YES" if edge > 0.05 else ("BUY NO" if edge < -0.05 else "PASS"),
            "signal_strength": abs(edge),
        })

    return signals


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
        st.sidebar.warning("⚠️ Live trading enabled!")

    # Bankroll setting
    bankroll = st.sidebar.number_input(
        "Bankroll ($)",
        min_value=10.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0
    )

    # Risk settings
    st.sidebar.markdown("### Risk Settings")
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Fraction of Kelly criterion to use (0.25 = quarter Kelly)"
    )

    max_position = st.sidebar.slider(
        "Max Position (%)",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum position size as % of bankroll"
    )

    min_edge = st.sidebar.slider(
        "Min Edge (%)",
        min_value=1,
        max_value=20,
        value=5,
        help="Minimum edge required to trade"
    )

    st.sidebar.markdown("---")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Main content
    st.title("Weather Trading Dashboard")
    st.markdown(f"**Mode:** {'🔴 LIVE' if is_live else '🔒 DRY RUN'} | **Bankroll:** ${bankroll:,.2f} | **Date:** {date.today()}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🌡️ Forecasts", "📈 Markets", "💰 Trades"])

    # Fetch data
    with st.spinner("Fetching forecasts..."):
        forecasts = fetch_forecasts_cached()

    with st.spinner("Fetching markets..."):
        markets = fetch_markets_cached()

    # Calculate signals
    signals = calculate_signals(forecasts, markets) if markets else []

    # TAB 1: Overview
    with tab1:
        st.header("Trading Overview")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Cities Tracked",
                len(forecasts),
                help="Number of cities with active forecasts"
            )

        with col2:
            st.metric(
                "Active Markets",
                len(markets),
                help="Weather markets on Polymarket"
            )

        with col3:
            tradeable = len([s for s in signals if s["signal"] != "PASS"])
            st.metric(
                "Trade Signals",
                tradeable,
                help="Markets with sufficient edge"
            )

        with col4:
            avg_confidence = sum(f["confidence"] for f in forecasts.values()) / len(forecasts) if forecasts else 0
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.0%}",
                help="Average forecast confidence"
            )

        st.markdown("---")

        # Signals table
        if signals:
            st.subheader("Trading Signals")

            df_signals = pd.DataFrame(signals)
            df_signals = df_signals.sort_values("signal_strength", ascending=False)

            # Color code the signals
            def color_signal(val):
                if val == "BUY YES":
                    return "background-color: #c8e6c9"
                elif val == "BUY NO":
                    return "background-color: #ffcdd2"
                return ""

            def color_edge(val):
                if val > 0.1:
                    return "color: #00c853; font-weight: bold"
                elif val > 0.05:
                    return "color: #00c853"
                elif val < -0.1:
                    return "color: #ff1744; font-weight: bold"
                elif val < -0.05:
                    return "color: #ff1744"
                return "color: #9e9e9e"

            styled_df = df_signals[["city", "threshold", "forecast_high", "our_prob", "market_prob", "edge", "confidence", "signal"]].style\
                .format({
                    "threshold": "{:.0f}°F",
                    "forecast_high": "{:.1f}°F",
                    "our_prob": "{:.1%}",
                    "market_prob": "{:.1%}",
                    "edge": "{:+.1%}",
                    "confidence": "{:.0%}",
                })\
                .applymap(color_signal, subset=["signal"])\
                .applymap(color_edge, subset=["edge"])

            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No active markets found. Check the Markets tab for details.")

        # Edge distribution chart
        if signals:
            st.subheader("Edge Distribution")
            fig = px.bar(
                df_signals,
                x="city",
                y="edge",
                color="signal",
                color_discrete_map={"BUY YES": "#00c853", "BUY NO": "#ff1744", "PASS": "#9e9e9e"},
                title="Edge by Market (Our Probability - Market Probability)"
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Forecasts
    with tab2:
        st.header("Weather Forecasts")
        st.markdown(f"**Forecast Date:** {date.today() + timedelta(days=1)}")

        if forecasts:
            # Forecast cards
            cols = st.columns(len(forecasts))

            for i, (city_key, fc) in enumerate(forecasts.items()):
                with cols[i]:
                    st.markdown(f"### {fc['city']}")
                    st.metric(
                        "High",
                        f"{fc['high_mean']:.1f}°F",
                        delta=f"±{fc['high_std']:.1f}°F",
                        delta_color="off"
                    )
                    st.metric(
                        "Low",
                        f"{fc['low_mean']:.1f}°F",
                        delta=f"±{fc['low_std']:.1f}°F",
                        delta_color="off"
                    )

                    # Confidence bar
                    conf = fc['confidence']
                    conf_color = "#00c853" if conf > 0.7 else ("#ffc107" if conf > 0.5 else "#ff1744")
                    st.markdown(f"**Confidence:** {conf:.0%}")
                    st.progress(conf)
                    st.caption(f"{fc['model_count']} models")

            st.markdown("---")

            # Temperature comparison chart
            st.subheader("Temperature Comparison")

            df_fc = pd.DataFrame([
                {"City": fc["city"], "Temperature": fc["high_mean"], "Type": "High", "Std": fc["high_std"]}
                for fc in forecasts.values()
            ] + [
                {"City": fc["city"], "Temperature": fc["low_mean"], "Type": "Low", "Std": fc["low_std"]}
                for fc in forecasts.values()
            ])

            fig = px.bar(
                df_fc,
                x="City",
                y="Temperature",
                color="Type",
                barmode="group",
                error_y="Std",
                title="Forecasted Temperatures with Uncertainty"
            )
            fig.update_layout(yaxis_title="Temperature (°F)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No forecasts available. Click Refresh to fetch.")

    # TAB 3: Markets
    with tab3:
        st.header("Polymarket Weather Markets")

        if markets:
            df_markets = pd.DataFrame(markets)

            # Market overview
            st.subheader(f"Found {len(markets)} Active Markets")

            for market in markets:
                with st.expander(f"**{market['city'].upper()}** - {market['question'][:80]}..."):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("YES Price", f"${market['yes_price']:.2f}")
                    with col2:
                        st.metric("NO Price", f"${market['no_price']:.2f}")
                    with col3:
                        st.metric("Volume", f"${market['volume']:,.0f}")

                    st.markdown(f"**Threshold:** {market['threshold']}°F")
                    st.markdown(f"**Target Date:** {market['target_date']}")
                    st.markdown(f"**Type:** {market['market_type']}")

                    # Compare with our forecast
                    city_key = market["city"].lower()
                    if city_key in forecasts:
                        fc = forecasts[city_key]
                        st.markdown("---")
                        st.markdown("**Our Forecast:**")
                        st.markdown(f"- High: {fc['high_mean']:.1f}°F (±{fc['high_std']:.1f})")
                        st.markdown(f"- Confidence: {fc['confidence']:.0%}")
        else:
            st.info("No active weather markets found on Polymarket.")
            st.markdown("""
            This could mean:
            - No temperature markets are currently active
            - Markets use different question formats
            - API rate limiting

            Try refreshing in a few minutes.
            """)

            # Show sample of what we're looking for
            st.subheader("Market Search Terms")
            st.code("temperature, weather, degrees, °F")

    # TAB 4: Trades
    with tab4:
        st.header("Trade Management")

        # Position calculator
        st.subheader("Position Calculator")

        col1, col2 = st.columns(2)

        with col1:
            calc_edge = st.number_input("Edge (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0) / 100
            calc_prob = st.number_input("Win Probability (%)", min_value=1.0, max_value=99.0, value=60.0, step=1.0) / 100

        with col2:
            # Kelly calculation
            if calc_prob > 0 and calc_prob < 1:
                q = 1 - calc_prob
                b = (1 - calc_prob + calc_edge) / calc_prob  # Simplified odds
                kelly = max(0, (calc_prob * b - q) / b)
                adjusted_kelly = kelly * kelly_fraction
                position_size = min(bankroll * adjusted_kelly, bankroll * max_position / 100)

                st.metric("Full Kelly", f"{kelly:.1%}")
                st.metric("Adjusted Kelly", f"{adjusted_kelly:.1%}")
                st.metric("Position Size", f"${position_size:.2f}")

        st.markdown("---")

        # Trade history (placeholder - would connect to actual trade journal)
        st.subheader("Trade History")

        # Demo data
        demo_trades = pd.DataFrame([
            {"Time": "2026-01-26 14:30", "City": "NYC", "Side": "YES", "Size": 47.50, "Price": 0.52, "Edge": 0.12, "Status": "Filled"},
            {"Time": "2026-01-26 14:15", "City": "Atlanta", "Side": "NO", "Size": 35.00, "Price": 0.45, "Edge": 0.08, "Status": "Filled"},
            {"Time": "2026-01-26 13:45", "City": "Seattle", "Side": "YES", "Size": 52.00, "Price": 0.61, "Edge": 0.15, "Status": "Filled"},
        ])

        if is_live:
            st.dataframe(demo_trades, use_container_width=True)
        else:
            st.info("Trade history will appear here when running in Live mode.")
            st.markdown("**Demo trades (not real):**")
            st.dataframe(demo_trades, use_container_width=True)

        st.markdown("---")

        # Manual trade execution
        st.subheader("Execute Trade")

        col1, col2, col3 = st.columns(3)

        with col1:
            trade_city = st.selectbox("City", get_all_cities())
            trade_side = st.radio("Side", ["YES", "NO"])

        with col2:
            trade_size = st.number_input("Size ($)", min_value=1.0, max_value=bankroll, value=50.0)
            trade_price = st.number_input("Limit Price", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Execute Trade", type="primary", disabled=not is_live):
                if is_live:
                    st.warning("Trade execution not yet connected to Polymarket API")
                else:
                    st.info("Switch to Live mode to execute trades")

    # Footer
    st.markdown("---")
    st.markdown(
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}* | "
        f"*Weather Trader v0.1.0* | "
        f"[Documentation](./FORPERRY.md)"
    )


if __name__ == "__main__":
    main()
