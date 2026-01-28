"""Debug order placement - show exact API error."""
import asyncio
import httpx
from weather_trader.kalshi import KalshiAuth, KalshiMarketFinder
from weather_trader.config import config

async def main():
    auth = KalshiAuth()
    base_url = config.api.kalshi_api_base_url

    # Get a real market ticker
    print("=== Finding Markets ===")
    finder = KalshiMarketFinder()
    markets = await finder.find_weather_markets(city_filter="nyc", days_ahead=2)

    if not markets:
        print("No markets found")
        return

    market = markets[0]
    bracket = market.brackets[0]
    print(f"Market: {market.event_ticker}")
    print(f"Bracket: {bracket.ticker}")
    print(f"Current yes price: {bracket.yes_price_cents}c")

    # Try order with detailed error output
    print("\n=== Attempting Order ===")

    order_body = {
        "ticker": bracket.ticker,
        "action": "buy",
        "side": "yes",
        "count": 1,
        "type": "limit",
        "yes_price": 5,  # 5 cents
    }

    print(f"Request body: {order_body}")

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        path = "/portfolio/orders"
        full_path = f"/trade-api/v2{path}"
        headers = auth.get_auth_headers("POST", full_path)

        print(f"URL: {base_url}{path}")
        print(f"Headers: KALSHI-ACCESS-KEY={headers.get('KALSHI-ACCESS-KEY', '')[:20]}...")

        try:
            resp = await client.post(path, json=order_body, headers=headers)
            print(f"\nResponse status: {resp.status_code}")
            print(f"Response body: {resp.text}")

            if resp.status_code == 200 or resp.status_code == 201:
                data = resp.json()
                order_id = data.get("order", {}).get("order_id", "")
                print(f"\nSUCCESS! Order ID: {order_id}")

                # Cancel it
                if order_id:
                    print("Cancelling order...")
                    cancel_resp = await client.delete(
                        f"/portfolio/orders/{order_id}",
                        headers=auth.get_auth_headers("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
                    )
                    print(f"Cancel response: {cancel_resp.status_code}")
        except Exception as e:
            print(f"Exception: {e}")

asyncio.run(main())
