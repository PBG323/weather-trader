"""Test Kalshi API connection."""
import asyncio
from weather_trader.kalshi import KalshiAuth, KalshiClient, KalshiMarketFinder

# Step 1: Validate credentials
print("=== Step 1: Validate API Credentials ===")
auth = KalshiAuth()
print(f"Auth configured: {auth.is_configured}")

client = KalshiClient(auth)
balance = asyncio.run(client.get_balance())
print(f"Account balance: ${balance:.2f}")

# Step 2: Verify market discovery
print("\n=== Step 2: Verify Market Discovery ===")
finder = KalshiMarketFinder()
markets = asyncio.run(finder.find_weather_markets(days_ahead=2))
print(f"Found {len(markets)} markets")
for m in markets[:5]:
    print(f"  {m.event_ticker} - {m.city} - {len(m.brackets)} brackets")

print("\n=== Tests Complete ===")
