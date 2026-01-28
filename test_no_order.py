"""Test NO order placement on Kalshi."""
import asyncio
from weather_trader.kalshi import KalshiAuth, KalshiClient, KalshiMarketFinder

async def main():
    auth = KalshiAuth()
    client = KalshiClient(auth)

    # Find a market
    print("=== Finding Markets ===")
    finder = KalshiMarketFinder()
    markets = await finder.find_weather_markets(city_filter="nyc", days_ahead=2)

    if not markets:
        print("No markets found")
        return

    market = markets[0]
    print(f"Market: {market.event_ticker}")

    # Find a bracket where NO is cheap (high yes_price)
    # Sort by yes_price descending to find where NO is cheapest
    sorted_brackets = sorted(market.brackets, key=lambda b: b.yes_price_cents, reverse=True)
    bracket = sorted_brackets[0]

    no_price = 100 - bracket.yes_price_cents
    print(f"\nBracket: {bracket.ticker}")
    print(f"Description: {bracket.description}")
    print(f"YES price: {bracket.yes_price_cents}c")
    print(f"NO price: {no_price}c")

    # Place NO order at 5 cents (very low, might not fill unless NO is already cheap)
    test_price = 5
    print(f"\n=== Placing NO Order ===")
    print(f"Ticker: {bracket.ticker}")
    print(f"Side: NO")
    print(f"Count: 1")
    print(f"Price: {test_price} cents")
    print(f"(This means yes_price sent to API will be: {100 - test_price})")

    confirm = input("\nPlace this NO order? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    result = await client.place_order(
        ticker=bracket.ticker,
        action="buy",
        side="no",
        count=1,
        price_cents=test_price,
    )

    print(f"\nOrder result:")
    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")
    print(f"  Message: {result.message}")
    print(f"  Filled: {result.filled_count}")
    print(f"  Remaining: {result.remaining_count}")

    if result.success and result.order_id and result.remaining_count > 0:
        print(f"\nOrder pending - cancelling...")
        cancelled = await client.cancel_order(result.order_id)
        print(f"Cancelled: {cancelled}")
    elif result.success and result.filled_count > 0:
        print(f"\nOrder filled! You now own 1 NO contract on {bracket.ticker}")

asyncio.run(main())
