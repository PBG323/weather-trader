#!/usr/bin/env python3
"""
Comprehensive validation of trading signals pipeline.
Run: python scripts/validate_signals.py
"""
import asyncio
import sys
sys.path.insert(0, '/home/user/weather-trader')

from datetime import datetime, date
from zoneinfo import ZoneInfo

from weather_trader.kalshi.markets import KalshiMarketFinder
from weather_trader.apis.nws import NWSClient
from weather_trader.apis.aviation_weather import AviationWeatherClient
from weather_trader.config import get_city_config, get_all_cities

EST = ZoneInfo("America/New_York")


async def validate_metar_fetch():
    """Test METAR fetch functionality."""
    print("\n" + "="*60)
    print("METAR FETCH TEST")
    print("="*60)

    try:
        async with AviationWeatherClient() as client:
            result = await client.get_all_city_temperatures()
            if result:
                print("✅ METAR fetch working!")
                for city, data in result.items():
                    print(f"  {city.upper()}: {data['temperature_f']:.1f}°F ({data['station']}, {data['age_minutes']}min old)")
            else:
                print("❌ METAR fetch returned empty")
    except Exception as e:
        print(f"❌ METAR fetch failed: {e}")
        import traceback
        traceback.print_exc()


async def validate_nws_observations():
    """Test NWS observations (now using METAR as primary)."""
    print("\n" + "="*60)
    print("NWS/METAR OBSERVATIONS TEST")
    print("="*60)

    us_cities = ['nyc', 'chicago', 'denver', 'miami', 'austin', 'philadelphia']

    async with NWSClient() as nws:
        for city_key in us_cities:
            city_config = get_city_config(city_key)
            if not city_config:
                continue

            try:
                current_high = await nws.get_current_day_high(city_config)
                if current_high:
                    print(f"✅ {city_config.name}: Today's high = {current_high:.1f}°F")
                else:
                    print(f"⚠️  {city_config.name}: No current high available")
            except Exception as e:
                print(f"❌ {city_config.name}: Error - {e}")


async def validate_kalshi_markets():
    """Validate Kalshi market data and brackets."""
    print("\n" + "="*60)
    print("KALSHI MARKET BRACKETS")
    print("="*60)

    today = datetime.now(EST).date()
    tomorrow = today

    async with KalshiMarketFinder() as finder:
        try:
            markets = await finder.find_weather_markets(
                active_only=True,
                days_ahead=3,
                include_same_day=True
            )

            print(f"Found {len(markets)} market events\n")

            for market in markets:
                target = market.target_date
                is_today = target == today
                day_label = "TODAY" if is_today else target.strftime("%m/%d")

                print(f"\n{market.city.upper()} - {day_label}")
                print(f"  Event: {market.event_ticker}")
                print(f"  Brackets:")

                for bracket in market.brackets[:5]:  # Show first 5 brackets
                    temp_low = bracket.temp_low
                    temp_high = bracket.temp_high

                    # Determine bracket type
                    if temp_low is None and temp_high is not None:
                        bracket_desc = f"≤{temp_high:.0f}°F"
                        bracket_type = "or_below"
                    elif temp_high is None and temp_low is not None:
                        bracket_desc = f"≥{temp_low:.0f}°F"
                        bracket_type = "or_above"
                    elif temp_low is not None and temp_high is not None:
                        bracket_desc = f"{temp_low:.0f}-{temp_high:.0f}°F"
                        bracket_type = "range"
                    else:
                        bracket_desc = "UNKNOWN"
                        bracket_type = "unknown"

                    yes_pct = bracket.yes_price * 100
                    vol = bracket.volume or 0

                    print(f"    {bracket_desc:12} | YES: {yes_pct:5.1f}% | Vol: {vol:5d} | {bracket.ticker}")

                    # Validate floor_strike/cap_strike parsing
                    if bracket_type == "range" and (temp_high - temp_low) > 5:
                        print(f"      ⚠️  WIDE RANGE: {temp_high - temp_low}°F spread - check parsing!")

        except Exception as e:
            print(f"❌ Error fetching Kalshi markets: {e}")
            import traceback
            traceback.print_exc()


async def validate_signal_calculation():
    """Test signal calculation with sample data."""
    print("\n" + "="*60)
    print("SIGNAL CALCULATION TEST")
    print("="*60)

    from scipy import stats

    # Test probability calculations for various brackets
    test_cases = [
        # (forecast_mean, forecast_std, temp_low, temp_high, expected_approx)
        (75.0, 2.0, 74.0, 75.0, 0.38),  # Forecast centered in bracket
        (75.0, 2.0, None, 73.0, 0.16),  # Below bracket
        (75.0, 2.0, 77.0, None, 0.16),  # Above bracket
        (75.0, 2.0, 70.0, 72.0, 0.07),  # Far below
        (75.0, 2.0, 78.0, 80.0, 0.07),  # Far above
    ]

    print("\nProbability Calculation Tests:")
    print(f"{'Forecast':^12} | {'Bracket':^12} | {'Our Prob':^10} | {'Expected':^10} | {'Status':^8}")
    print("-" * 60)

    for mean, std, low, high, expected in test_cases:
        # Calculate probability
        CONTINUITY = 0.5
        if low is None and high is not None:
            prob = stats.norm.cdf(high + CONTINUITY, loc=mean, scale=std)
            bracket_str = f"≤{high:.0f}°F"
        elif high is None and low is not None:
            prob = 1 - stats.norm.cdf(low - CONTINUITY, loc=mean, scale=std)
            bracket_str = f"≥{low:.0f}°F"
        else:
            prob = (stats.norm.cdf(high + CONTINUITY, loc=mean, scale=std) -
                   stats.norm.cdf(low - CONTINUITY, loc=mean, scale=std))
            bracket_str = f"{low:.0f}-{high:.0f}°F"

        status = "✅" if abs(prob - expected) < 0.1 else "⚠️"
        print(f"{mean:^12.1f} | {bracket_str:^12} | {prob:^10.2%} | {expected:^10.2%} | {status:^8}")


async def main():
    print("="*60)
    print("TRADING SIGNALS VALIDATION")
    print(f"Time: {datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("="*60)

    await validate_metar_fetch()
    await validate_nws_observations()
    await validate_kalshi_markets()
    await validate_signal_calculation()

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
