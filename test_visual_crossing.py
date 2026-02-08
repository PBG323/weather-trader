"""Test Visual Crossing API integration."""
import asyncio
import sys
sys.path.insert(0, 'C:/Users/perry/weather_trader')

from dotenv import load_dotenv
load_dotenv()

from weather_trader.apis.visual_crossing import VisualCrossingClient
from weather_trader.config import get_city_config

async def test_visual_crossing():
    """Test Visual Crossing API for all cities."""
    print("=" * 60)
    print("VISUAL CROSSING API TEST")
    print("=" * 60)

    client = VisualCrossingClient()

    # Check if configured
    if not client.is_configured():
        print("ERROR: Visual Crossing API key not configured!")
        print("Add VISUAL_CROSSING_API_KEY to your .env file")
        return False

    print(f"API Key configured: Yes")
    print(f"Remaining calls today: {client.get_remaining_calls()}")
    print()

    # Test cities
    test_cities = ["nyc", "miami", "chicago"]

    async with client:
        for city_key in test_cities:
            city_config = get_city_config(city_key)
            print(f"Testing {city_config.name}...")

            try:
                # Get forecast
                forecasts = await client.get_forecast(city_config, days=3)

                if forecasts:
                    print(f"  Forecasts received: {len(forecasts)} days")
                    for f in forecasts:
                        print(f"    {f.date}: High {f.temperature_high:.0f}F, Low {f.temperature_low:.0f}F - {f.conditions}")
                else:
                    print("  No forecasts returned")

                # Get current conditions
                current = await client.get_current_conditions(city_config)
                if current:
                    print(f"  Current: {current.get('temperature')}F - {current.get('conditions')}")

                print()

            except Exception as e:
                print(f"  ERROR: {e}")
                return False

    print(f"Remaining calls after test: {client.get_remaining_calls()}")
    print()
    print("SUCCESS: Visual Crossing API is working correctly!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_visual_crossing())
    sys.exit(0 if success else 1)
