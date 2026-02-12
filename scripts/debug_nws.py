#!/usr/bin/env python3
"""
Debug script to compare NWS data with dashboard same-day forecasts.
Run: python scripts/debug_nws.py
"""
import asyncio
import sys
sys.path.insert(0, '/home/user/weather-trader')

from datetime import datetime
from zoneinfo import ZoneInfo

from weather_trader.apis.nws import NWSClient
from weather_trader.config import get_city_config, get_all_cities

EST = ZoneInfo("America/New_York")

async def debug_city(city_key: str):
    """Debug NWS data for a single city."""
    city_config = get_city_config(city_key)
    if not city_config:
        print(f"  Unknown city: {city_key}")
        return

    print(f"\n{'='*60}")
    print(f"CITY: {city_config.name} (Station: {city_config.nws_station_id})")
    print(f"{'='*60}")

    async with NWSClient() as nws:
        # Get current observed high
        current_high = await nws.get_current_day_high(city_config)
        print(f"\nCurrent Observed High Today: {current_high:.1f}°F" if current_high else "\nCurrent Observed High Today: N/A")

        # Get remaining forecast
        remaining = await nws.get_remaining_day_forecast(city_config)
        if remaining:
            rem_high = remaining.get('remaining_high')
            hours = remaining.get('hours_remaining', 0)
            print(f"Remaining Forecast High: {rem_high:.1f}°F ({hours} hours left)" if rem_high else f"Remaining Forecast High: Day over")
        else:
            print("Remaining Forecast: N/A")

        # Calculate expected final high
        if current_high and remaining and remaining.get('remaining_high'):
            expected_high = max(current_high, remaining['remaining_high'])
            print(f"Expected Final High: {expected_high:.1f}°F")
        elif current_high:
            print(f"Expected Final High: {current_high:.1f}°F (no more rise expected)")

        # Get NWS forecast for comparison
        print("\nNWS Point Forecast:")
        try:
            forecast_periods = await nws.get_forecast(city_config)
            for period in forecast_periods[:4]:  # First 4 periods
                name = period.get('name', 'Unknown')
                temp = period.get('temperature', 'N/A')
                unit = period.get('temperatureUnit', 'F')
                short = period.get('shortForecast', '')
                print(f"  {name}: {temp}°{unit} - {short}")
        except Exception as e:
            print(f"  Error fetching forecast: {e}")

        # Get recent observations
        print("\nRecent Observations (last 6 hours):")
        try:
            observations = await nws.get_station_observations(city_config, hours=6)
            for obs in observations[:6]:
                time_str = obs.timestamp.strftime("%H:%M")
                print(f"  {time_str}: {obs.temperature:.1f}°F - {obs.description}")
        except Exception as e:
            print(f"  Error fetching observations: {e}")


async def main():
    print("=" * 60)
    print("NWS DATA DIAGNOSTIC")
    print(f"Time: {datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 60)

    # Get all US cities (NWS only works for US)
    us_cities = ['nyc', 'chicago', 'denver', 'miami', 'austin', 'philadelphia']

    for city in us_cities:
        try:
            await debug_city(city)
        except Exception as e:
            print(f"\nError processing {city}: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
