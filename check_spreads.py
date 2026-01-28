"""Check pending orders vs current market prices."""
import asyncio
from weather_trader.kalshi import KalshiAuth, KalshiClient

async def main():
    auth = KalshiAuth()
    client = KalshiClient(auth)

    # Get pending orders
    data = await client._request("GET", "/portfolio/orders")
    orders = data.get("orders", [])
    pending = [o for o in orders if o.get("status") not in ("executed", "cancelled") and o.get("remaining_count", 0) > 0]

    if not pending:
        print("No pending orders")
        return

    print(f"{'Ticker':<35} {'Side':<5} {'Your Limit':<12} {'Market Bid':<12} {'Market Ask':<12} {'Gap'}")
    print("-" * 95)

    for order in pending:
        ticker = order.get("ticker", "")
        side = order.get("side", "")
        your_price = order.get("yes_price", 0)
        remaining = order.get("remaining_count", 0)

        if remaining == 0:
            continue

        # Get current market price
        try:
            market_data = await client._request("GET", f"/markets/{ticker}")
            market = market_data if "ticker" in market_data else market_data.get("market", {})
            yes_bid = market.get("yes_bid", 0) or 0
            yes_ask = market.get("yes_ask", 0) or 0

            if side == "yes":
                # Buying YES: need to beat the ask
                gap = yes_ask - your_price if yes_ask else "N/A"
                gap_str = f"+{gap}c to fill" if isinstance(gap, int) and gap > 0 else "✓ should fill" if gap <= 0 else gap
            else:
                # Buying NO: your yes_price should be >= bid (you're selling YES)
                gap = your_price - yes_bid if yes_bid else "N/A"
                gap_str = f"-{gap}c to fill" if isinstance(gap, int) and gap > 0 else "✓ should fill" if gap <= 0 else gap

            print(f"{ticker:<35} {side:<5} {your_price:<12} {yes_bid:<12} {yes_ask:<12} {gap_str}")

        except Exception as e:
            print(f"{ticker:<35} {side:<5} {your_price:<12} Error: {e}")

        await asyncio.sleep(0.3)  # Rate limit

    print("\n" + "=" * 95)
    print("To fill faster, your limit should be closer to the ask (for YES) or bid (for NO)")

asyncio.run(main())
