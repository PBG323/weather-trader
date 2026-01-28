"""Debug order placement - test with a real weather market."""
import asyncio
from weather_trader.kalshi import KalshiAuth, KalshiClient, KalshiMarketFinder

async def main():
    auth = KalshiAuth()
    print(f"Auth configured: {auth.is_configured}")

    # Step 1: Find real markets
    print("\n=== Finding Real Weather Markets ===")
    finder = KalshiMarketFinder()
    markets = await finder.find_weather_markets(days_ahead=2)

    if not markets:
        print("ERROR: No markets found!")
        return

    print(f"Found {len(markets)} markets")

    # Pick first market with brackets
    market = markets[0]
    print(f"\nUsing market: {market.event_ticker}")
    print(f"City: {market.city}")
    print(f"Date: {market.target_date}")
    print(f"Brackets: {len(market.brackets)}")

    for b in market.brackets:
        print(f"  {b.ticker}: {b.description} @ {b.yes_price_cents}c")

    # Step 2: Get current price for first bracket
    bracket = market.brackets[0]
    print(f"\n=== Testing Order on {bracket.ticker} ===")

    client = KalshiClient(auth)

    try:
        price_data = await client.get_market_price(bracket)
        print(f"Current prices: yes_bid={price_data['yes_bid']:.2f}, yes_ask={price_data['yes_ask']:.2f}")
    except Exception as e:
        print(f"Error getting price: {e}")

    # Step 3: Try to place a $1 test order (1 contract at low price that won't fill)
    print("\n=== Placing Test Order ===")
    print(f"Ticker: {bracket.ticker}")
    print(f"Action: buy")
    print(f"Side: yes")
    print(f"Count: 1")
    print(f"Price: 5 cents (intentionally low so it won't fill)")

    confirm = input("\nPlace this test order? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    result = await client.place_order(
        ticker=bracket.ticker,
        action="buy",
        side="yes",
        count=1,
        price_cents=5,  # Very low price, won't fill
    )

    print(f"\nOrder result:")
    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")
    print(f"  Message: {result.message}")
    print(f"  Filled: {result.filled_count}")
    print(f"  Remaining: {result.remaining_count}")

    if result.success and result.order_id:
        print(f"\n=== Cancelling Test Order ===")
        cancelled = await client.cancel_order(result.order_id)
        print(f"Cancelled: {cancelled}")

asyncio.run(main())
