"""Check open orders and positions on Kalshi."""
import asyncio
from weather_trader.kalshi import KalshiAuth, KalshiClient

async def main():
    auth = KalshiAuth()
    client = KalshiClient(auth)

    # Check balance
    balance = await client.get_balance()
    print(f"Account balance: ${balance:.2f}")

    # Check open orders
    print("\n=== Open Orders ===")
    try:
        data = await client._request("GET", "/portfolio/orders")
        orders = data.get("orders", [])
        if orders:
            for o in orders:
                print(f"  {o.get('ticker')} | {o.get('side')} | {o.get('remaining_count')} @ {o.get('yes_price')}c | Status: {o.get('status')}")
        else:
            print("  No open orders")
    except Exception as e:
        print(f"  Error fetching orders: {e}")

    # Check positions
    print("\n=== Open Positions ===")
    try:
        positions = await client.get_positions()
        if positions:
            for p in positions:
                ticker = p.get("ticker", "")
                yes_count = p.get("position", 0)
                avg_price = p.get("average_price_paid", 0)
                print(f"  {ticker} | {yes_count} contracts @ avg {avg_price}c")
        else:
            print("  No open positions")
    except Exception as e:
        print(f"  Error fetching positions: {e}")

asyncio.run(main())
