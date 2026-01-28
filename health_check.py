"""Health check - verify system status and sync with Kalshi."""
import asyncio
from datetime import datetime
from weather_trader.kalshi import KalshiAuth, KalshiClient, KalshiMarketFinder
from weather_trader.config import get_all_cities

async def main():
    print(f"=== Weather Trader Health Check ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Check API connection
    print("1. API Connection")
    auth = KalshiAuth()
    if not auth.is_configured:
        print("   ❌ Kalshi credentials not configured!")
        return
    print(f"   ✅ Auth configured (Key: {auth.key_id[:8]}...)")

    # 2. Check balance
    print("\n2. Account Balance")
    client = KalshiClient(auth)
    try:
        balance = await client.get_balance()
        print(f"   ✅ Balance: ${balance:.2f}")
    except Exception as e:
        print(f"   ❌ Error getting balance: {e}")
        return

    # 3. Check market discovery
    print("\n3. Market Discovery")
    finder = KalshiMarketFinder()
    try:
        markets = await finder.find_weather_markets(days_ahead=3)
        print(f"   ✅ Found {len(markets)} weather markets")
        for m in markets[:5]:
            print(f"      - {m.event_ticker} ({m.city}) - {len(m.brackets)} brackets")
    except Exception as e:
        print(f"   ❌ Error fetching markets: {e}")

    # 4. Check open positions
    print("\n4. Open Positions (Kalshi)")
    try:
        positions = await client.get_positions()
        weather_positions = [p for p in positions if "KXHIGH" in p.get("ticker", "")]
        if weather_positions:
            print(f"   Found {len(weather_positions)} weather positions:")
            for p in weather_positions:
                ticker = p.get("ticker", "")
                count = p.get("position", 0)
                avg_price = p.get("average_price_paid", 0)
                print(f"      - {ticker}: {count} @ {avg_price}c")
        else:
            print("   No weather positions")

        other_positions = [p for p in positions if "KXHIGH" not in p.get("ticker", "")]
        if other_positions:
            print(f"   Also {len(other_positions)} non-weather positions")
    except Exception as e:
        print(f"   ❌ Error fetching positions: {e}")

    # 5. Check open orders
    print("\n5. Pending Orders (Kalshi)")
    try:
        data = await client._request("GET", "/portfolio/orders")
        orders = data.get("orders", [])
        pending = [o for o in orders if o.get("status") not in ("executed", "cancelled")]
        weather_pending = [o for o in pending if "KXHIGH" in o.get("ticker", "")]
        if weather_pending:
            print(f"   Found {len(weather_pending)} pending weather orders:")
            for o in weather_pending:
                print(f"      - {o.get('ticker')} {o.get('side')} {o.get('remaining_count')} @ {o.get('yes_price')}c")
        else:
            print("   No pending weather orders")
    except Exception as e:
        print(f"   ❌ Error fetching orders: {e}")

    # 6. API rate limit check
    print("\n6. Tomorrow.io API Usage")
    try:
        from weather_trader.dashboard import get_tomorrow_io_usage
        calls, date_str = get_tomorrow_io_usage()
        remaining = 500 - calls
        print(f"   Calls today: {calls}/500 (remaining: {remaining})")
    except Exception as e:
        print(f"   ⚠️ Could not check (dashboard not running): {e}")

    print("\n=== Health Check Complete ===")

asyncio.run(main())
