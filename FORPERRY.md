# Weather Trader: A Deep Dive for Perry

Hey Perry! This document is your comprehensive guide to understanding the weather trading system we built. I've written it the way I'd explain it to you over coffee—no boring textbook vibes, just real talk about how this thing works, why we made certain decisions, and what you can learn from building it.

---

## Table of Contents

1. [The Big Picture: What Are We Actually Building?](#the-big-picture)
2. [The Mental Model: How to Think About This System](#the-mental-model)
3. [Architecture: The Blueprint](#architecture)
4. [The Codebase Tour: A Walk Through the Neighborhood](#codebase-tour)
5. [Technologies: Our Toolbox](#technologies)
6. [The Secret Sauce: Where the Edge Comes From](#the-secret-sauce)
7. [Lessons Learned: The Hard-Won Wisdom](#lessons-learned)
8. [How Good Engineers Think](#how-good-engineers-think)
9. [Pitfalls to Avoid](#pitfalls-to-avoid)
10. [Running the System](#running-the-system)

---

<a name="the-big-picture"></a>
## 1. The Big Picture: What Are We Actually Building?

Imagine you're a sports bettor, but instead of betting on football games, you're betting on the weather. Specifically, you're betting on questions like:

> "Will New York City's high temperature be over 50°F tomorrow?"

Kalshi is a prediction market where people trade on these outcomes. If you think YES, you buy YES contracts. If the temperature ends up being 52°F, your YES contracts pay out $1 each. If it's 48°F, they're worth $0.

**Here's the key insight:** Most traders on Kalshi are just looking at weather.com and guessing. But we have access to the same sophisticated weather models that meteorologists use—ECMWF (the European model that's consistently the world's best), GFS, HRRR, and more.

**Even better:** We know a secret that most traders don't. Kalshi settles these markets based on *specific weather station readings*—like Central Park for NYC. Weather apps give you a general city forecast, but stations can read differently. A forecast might say "NYC high: 50°F" but Central Park specifically might hit 52°F because it's in a park (urban heat island effect works differently there).

We exploit this gap.

---

<a name="the-mental-model"></a>
## 2. The Mental Model: How to Think About This System

Think of our system as a **factory with an assembly line**:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GATHER    │ ──▶ │   REFINE    │ ──▶ │   DECIDE    │ ──▶ │   EXECUTE   │
│  Raw Data   │     │  Forecasts  │     │   Trades    │     │   Orders    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     APIs              Models            Strategy            Kalshi
```

**Stage 1 - GATHER:** We pull forecasts from multiple weather APIs. It's like asking five different meteorologists for their opinion instead of just one.

**Stage 2 - REFINE:** We don't just average the forecasts. We apply "bias correction"—adjusting each forecast based on how that model has historically performed for that specific weather station. Then we combine them into an ensemble with uncertainty estimates.

**Stage 3 - DECIDE:** We compare our probability estimate to the market price. If we think there's a 75% chance of "over 50°F" but the market is pricing it at 55%, we have a 20% edge. We then figure out how much to bet using the Kelly Criterion (more on this later).

**Stage 4 - EXECUTE:** We place orders on Kalshi and track everything.

---

<a name="architecture"></a>
## 3. Architecture: The Blueprint

Here's how the pieces fit together:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Open-Meteo  │  │ Tomorrow.io  │  │     NWS      │                  │
│  │ (ECMWF/GFS/  │  │ (Proprietary │  │  (Official   │                  │
│  │    HRRR)     │  │   models)    │  │   stations)  │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
└─────────┼─────────────────┼─────────────────┼───────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                                │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Bias Correction (ML)                         │    │
│  │   "ECMWF tends to forecast 2°F too low for Central Park"       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Ensemble Forecaster                          │    │
│  │   Combines models → Weighted average + Confidence intervals     │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          TRADING LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ Expected Value  │  │ Position Sizing │  │    Executor     │         │
│  │   Calculator    │─▶│ (Kelly Criterion│─▶│  (Order Mgmt)   │         │
│  │ "20% edge found"│  │  "Bet $47.50")  │  │ "Order filled"  │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          KALSHI API                                     │
│              (REST API with RSA-PSS Authentication)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Separation of Concerns:** Each layer does ONE thing well. The API layer doesn't care about trading. The trading layer doesn't care about weather models. This makes the code easier to test, debug, and modify.

**Pluggability:** Want to add a new weather API? Just create a new file in `apis/`. Want to change position sizing? Only touch `strategy/position_sizing.py`. Nothing else breaks.

**Fail-Safe Design:** If Tomorrow.io goes down, we still have Open-Meteo. If one trade fails, others continue. The system degrades gracefully instead of crashing entirely.

---

<a name="codebase-tour"></a>
## 4. The Codebase Tour: A Walk Through the Neighborhood

Let's walk through the code like we're touring a city, neighborhood by neighborhood.

### 📁 `config.py` — The City Hall

This is where all the important city records live. Every configuration, every API endpoint, every weather station mapping.

```python
CITY_CONFIGS: dict[str, CityConfig] = {
    "nyc": CityConfig(
        name="New York City",
        nws_station_id="KNYC",        # THIS IS CRUCIAL
        station_name="Central Park",
        latitude=40.7789,
        longitude=-73.9692,
        timezone="America/New_York",
        country="US"
    ),
    # ... more cities
}
```

**Why station IDs matter:** See that `KNYC`? That's the ICAO code for the Central Park weather station. This is what Kalshi uses to settle contracts via the NWS Daily Climate Report. If we forecast for "New York City" generically, we might be off. We need to forecast for *that exact station*.

**Lesson:** Configuration should be centralized. When you need to change something (like adding a new city), you change it in ONE place, not hunt through 20 files.

---

### 📁 `apis/` — The Foreign Embassies

Each file here is like an embassy—a connection to an external service that speaks its own language. Our job is to translate everything into a common format.

#### `open_meteo.py` — The Free University

Open-Meteo is amazing: free, fast, and gives us access to world-class models (ECMWF, GFS, HRRR).

```python
async def get_daily_forecast(
    self,
    city_config: CityConfig,
    model: WeatherModel = WeatherModel.BEST_MATCH,
    days: int = 7
) -> list[ForecastPoint]:
```

**The `async` keyword:** Notice this is an `async` function. Weather APIs take time to respond (maybe 200ms each). If we called three APIs sequentially, that's 600ms. With async, we can call all three *simultaneously* and get results in ~200ms total. This matters when you're trading and speed = money.

#### `tomorrow_io.py` — The Premium Consultant

Tomorrow.io costs money but offers proprietary models and hyperlocal data. We use it as one voice in our ensemble.

#### `nws.py` — The Source of Truth

This is special. The National Weather Service provides the *actual readings* that Kalshi uses to settle markets. We use this data to:
1. Train our bias correction models
2. Verify our forecasts after the fact
3. Understand the ground truth

**Analogy:** If weather APIs are students taking a test, NWS is the answer key. We train our models by seeing how each "student" historically performed against the answer key.

---

### 📁 `models/` — The Research Lab

This is where the magic happens. Raw forecasts go in, refined predictions come out.

#### `bias_correction.py` — The Calibration Machine

Imagine you have a bathroom scale that always reads 3 pounds heavy. Once you know this, you subtract 3 from every reading. That's bias correction.

```python
def correct_forecast(
    self,
    city_config: CityConfig,
    forecast_high: float,
    forecast_low: float,
    forecast_date: date,
    model_source: str = "unknown"
) -> CorrectedForecast:
```

We train a machine learning model (Gradient Boosting) that learns patterns like:
- "ECMWF undershoots Central Park highs by 1.8°F in winter"
- "GFS overshoots Seattle lows when there's an onshore flow"

**Why this matters:** A 2°F systematic error can mean the difference between a winning and losing trade. If the market is 50°F and our raw forecast says 51°F, we might pass. But if we know ECMWF undershoots by 2°F, our corrected forecast is 53°F—a much stronger signal.

#### `ensemble.py` — The Wisdom of Crowds

One forecaster can be wrong. Five forecasters are usually more accurate than any single one. This is the [wisdom of crowds](https://en.wikipedia.org/wiki/Wisdom_of_the_crowd) principle.

```python
# Different models get different weights based on historical accuracy
DEFAULT_WEIGHTS = {
    "ecmwf": 1.5,    # Best overall
    "gfs": 1.0,      # Good baseline
    "hrrr": 1.2,     # Best for short-term US
    "tomorrow": 1.3, # Proprietary edge
}
```

We don't just average—we compute a *weighted* average where better models count more. We also calculate uncertainty:

```python
def get_probability_above(self, threshold: float, for_high: bool = True) -> float:
    """
    If our forecast is 53°F with std dev of 3°F, what's the probability
    the actual temperature exceeds 50°F?

    Answer: Calculate using normal distribution CDF
    """
```

**Key insight:** The *spread* of forecasts tells us something. If all five models say 52-53°F, we're confident. If they range from 48-58°F, there's genuine uncertainty, and we should bet smaller.

---

### 📁 `kalshi/` — The Trading Floor

#### `auth.py` — The Vault

Handles API authentication. Kalshi uses RSA key-pair signing — you sign each request with your private key, and Kalshi verifies with your public key.

```python
def create_wallet() -> WalletInfo:
    """
    WARNING: Store the private key securely! Loss of private key
    means loss of all funds.
    """
    private_key = "0x" + secrets.token_hex(32)
```

**Security lesson:** Private keys are like the master password to your bank account. We NEVER log them, NEVER commit them to git, NEVER send them over the network unencrypted. They live in `.env` files that are `.gitignore`d.

#### `markets.py` — The Market Scanner

This scans Kalshi for weather-related markets and parses events like:

> "Will NYC high temperature be over 50°F on January 15?"

Into structured data:

```python
WeatherMarket(
    city="nyc",
    threshold=50.0,
    market_type=MarketType.HIGH_OVER,
    target_date=date(2024, 1, 15),
    yes_price=0.55,  # Market thinks 55% chance
    # ...
)
```

**The parsing challenge:** Market data comes from Kalshi's REST API with structured fields (series_ticker, floor_strike, cap_strike), making parsing much more reliable than scraping natural-language questions.

#### `client.py` — The Broker

Places actual orders on the exchange.

```python
async def place_limit_order(
    self,
    market: WeatherMarket,
    side: Literal["BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"],
    size: float,
    price: Optional[float] = None
) -> OrderResult:
```

**Limit orders vs Market orders:** We use limit orders (specify exact price) rather than market orders (take whatever's available). This protects us from slippage—buying at a worse price than expected because we moved the market.

---

### 📁 `strategy/` — The Brain Trust

#### `expected_value.py` — The Edge Calculator

This is where we determine if a trade is worth making.

```python
edge = forecast_probability - market_probability
```

If we think there's a 75% chance of "over 50°F" but the market prices it at 55%, our edge is 20%. That's a fantastic trade.

**But edge isn't everything:** We also check confidence. If our models disagree wildly, we might have positive expected value but low confidence. We pass on those trades.

```python
def _determine_signal(self, edge: float, confidence: float) -> SignalStrength:
    if confidence < self.min_confidence:
        return SignalStrength.NEUTRAL  # Too uncertain, don't trade

    if edge >= self.strong_edge:
        return SignalStrength.STRONG_BUY_YES
```

#### `position_sizing.py` — The Risk Manager

This might be the most important file. Great traders don't just find good trades—they size them correctly.

We use the **Kelly Criterion**, a formula from information theory that tells you the optimal bet size to maximize long-term growth:

```python
def calculate_kelly(self, win_prob: float, win_amount: float, lose_amount: float) -> float:
    """
    Kelly formula: f* = (p * b - q) / b

    If you have a 60% chance of winning 1:1 odds:
    f* = (0.6 * 1 - 0.4) / 1 = 0.2 = 20% of bankroll
    """
```

**But full Kelly is scary:** If you bet full Kelly and you're wrong about your edge, you can blow up fast. So we use "fractional Kelly"—typically 25% of what Kelly recommends.

```python
self.kelly_multiplier = 0.25  # Bet 1/4 of Kelly
```

**Analogy:** Full Kelly is driving at the speed limit on a mountain road. Fractional Kelly is driving 25% under the limit. You'll get there a bit slower, but you won't fly off a cliff if you misjudge a turn.

We also have hard limits:
- **Max position:** 5% of bankroll per trade (diversification)
- **Daily loss limit:** Stop trading if down 10% (prevents tilt/death spiral)

#### `executor.py` — The Trader's Hands

Actually places the trades and keeps a journal of everything.

```python
@dataclass
class ExecutionResult:
    recommendation: PositionRecommendation
    order_result: OrderResult
    intended_size: float
    executed_size: float
    slippage: float  # How much worse was our fill than expected?
```

**Slippage protection:** If the market moved against us too much before we could execute, we cancel the trade. No point buying at a 5% worse price—that eats our entire edge.

```python
if slippage > self.max_slippage:
    return ExecutionResult(success=False, error="Slippage too high")
```

---

### 📁 `monitoring/` — The Control Room

#### `logger.py` — The Historian

Uses `loguru` for beautiful, structured logging.

```python
# Regular logs for debugging
logger.info(f"Executing trade: {city} {side} ${size:.2f}")

# Trade-specific logs (JSON format for analysis)
trade_logger.log_execution(city="NYC", side="YES", size=50.0, ...)
```

We keep two log files:
1. `weather_trader.log` — Human-readable, for debugging
2. `trades.json` — Machine-readable, for backtesting and analysis

**Lesson:** Log everything. When something goes wrong at 3 AM, you'll thank yourself for having detailed logs.

#### `alerts.py` — The Alarm System

Sends notifications to Discord or Telegram when important things happen:

```python
async def trade_executed(self, city, side, size, price, edge):
    """Perry gets a ping: '🎯 Trade Executed: NYC - Bought YES at $0.52'"""

async def loss_limit_hit(self, daily_loss, limit):
    """Perry gets URGENT ping: '🚨 Loss limit reached, trading halted'"""
```

**Why this matters:** You don't want to babysit an algo 24/7. Let it run, and it'll tell you when something needs attention.

---

### 📁 `main.py` and `scheduler.py` — The Control Center

`main.py` orchestrates a single trading cycle:

```python
async def run_cycle(self):
    # 1. Get forecasts for all cities
    forecasts = await self.fetch_forecasts(target_date)

    # 2. Find active markets
    markets = await self.find_markets(target_date)

    # 3. Analyze for opportunities
    signals = await self.analyze_opportunities(forecasts, markets)

    # 4. Execute trades
    results = await self.execute_trades(signals)
```

`scheduler.py` runs these cycles on a schedule:

```python
# Every hour during market hours
scheduler.add_job(run_cycle, CronTrigger(hour="8-22", minute="0"))

# Every 15 minutes near settlement (when prices move most)
scheduler.add_job(run_cycle, CronTrigger(hour="7-10", minute="*/15"))
```

---

<a name="technologies"></a>
## 5. Technologies: Our Toolbox

### Python 3.10+
Why Python? It's the lingua franca of data science and has excellent async support. For trading systems, we need both (data processing + concurrent API calls).

### httpx (not requests)
`httpx` is like `requests` but with native async support. When you're calling 5 APIs, being able to do it concurrently cuts latency by 80%.

### asyncio
Python's async framework. Lets us write code that looks sequential but executes concurrently:

```python
# These all run at the same time!
results = await asyncio.gather(
    fetch_open_meteo(),
    fetch_tomorrow_io(),
    fetch_nws_data(),
)
```

### scikit-learn
For our bias correction ML models. We started with Ridge regression (simple, fast) but graduated to Gradient Boosting (captures non-linear patterns).

### pandas + numpy
The workhorses of data manipulation. We use pandas for tabular data (training datasets) and numpy for numerical operations (probability calculations).

### loguru
Better logging than the standard library. Colors, rotation, structured output—all with minimal configuration.

### APScheduler
For running trading cycles on a schedule. Supports cron-style triggers and handles timezone-aware scheduling.

### cryptography
For RSA-PSS signing of Kalshi API requests. Each request is authenticated by signing a timestamp+method+path string with your private key.

---

<a name="the-secret-sauce"></a>
## 6. The Secret Sauce: Where the Edge Comes From

Let me be real with you, Perry. This system has edge, but it's not magic. Here's where the edge actually comes from:

### 1. Station-Specific Forecasting
Most people forecast for "New York City." We forecast for Central Park specifically. This is a 1-2°F difference that creates systematic mispricing.

### 2. Ensemble Methods
One weather model can be wrong. Five models are usually more accurate. We're using the same ensemble approach that the best meteorologists use.

### 3. Bias Correction
We've learned each model's quirks for each station. ECMWF runs cold for Seattle in spring? We adjust. This is alpha that comes from doing the work.

### 4. Proper Position Sizing
Most retail traders bet too big or too small. Kelly Criterion is mathematically optimal for long-term growth. Using fractional Kelly gives us the upside with a safety margin.

### 5. Discipline
The system doesn't trade unless there's sufficient edge AND confidence. It's okay to skip trades. Overtrading is how most algos die.

### Where We DON'T Have Edge
- **Speed:** High-frequency traders will beat us on execution
- **Size:** We can't move big money without impacting prices
- **Inside information:** Professional weather traders might have better models

Our edge is in *diligence*—doing the boring work of calibrating models, tracking station-specific biases, and being disciplined about position sizing.

---

<a name="lessons-learned"></a>
## 7. Lessons Learned: The Hard-Won Wisdom

### Bug #1: The Timezone Trap

**What happened:** Forecasts for "January 15" were being matched to markets for "January 14" because one was in UTC and the other in local time.

**The fix:** Be explicit about timezones everywhere:

```python
# Bad
timestamp = datetime.fromisoformat(date_str)

# Good
timestamp = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
```

**Lesson:** Timezones are one of the two hard problems in computer science (the other being cache invalidation and off-by-one errors). Always use timezone-aware datetimes. Always.

### Bug #2: The Missing Import

**What happened:** `scheduler.py` used `Alert` and `AlertLevel` but didn't import them. The code looked fine but crashed at runtime.

**The fix:**
```python
from .monitoring import get_logger, AlertManager, Alert, AlertLevel
```

**Lesson:** Python's dynamic typing means import errors are runtime errors, not compile-time errors. This is why we write tests. Running `pytest` would have caught this immediately.

### Bug #3: The Rate Limit Surprise

**What happened:** Tomorrow.io started returning 429 errors (rate limited) because we were hammering their API during testing.

**The fix:** Added rate limiting in the client:

```python
self._last_request_time = 0
self._min_request_interval = 0.1  # 100ms between requests

def _rate_limit(self):
    elapsed = time.time() - self._last_request_time
    if elapsed < self._min_request_interval:
        time.sleep(self._min_request_interval - elapsed)
```

**Lesson:** Always respect API rate limits. Better yet, add rate limiting proactively before you get banned.

### Bug #4: The Negative Position Size

**What happened:** Kelly criterion returned negative values when there was negative edge, and we almost placed a negative-size order.

**The fix:**
```python
return max(0, kelly)  # Never recommend negative sizing
```

**Lesson:** Validate all inputs and outputs. Even mathematically "correct" formulas can produce garbage values with garbage inputs.

---

<a name="how-good-engineers-think"></a>
## 8. How Good Engineers Think

### They Think in Layers

Good engineers don't build one giant thing. They build layers that talk to each other through clean interfaces. Our system has:
- Data layer (APIs)
- Processing layer (Models)
- Decision layer (Strategy)
- Execution layer (Kalshi)

If one layer changes, others don't care. This is called **separation of concerns**.

### They Plan for Failure

Everything that can fail, will fail. Good engineers ask:
- What if the API is down? → Use fallback APIs
- What if the network is slow? → Add timeouts
- What if we get bad data? → Validate before using
- What if we lose money? → Daily loss limits

### They Write for Readers, Not Computers

Code is read 10x more than it's written. Good engineers:
- Use descriptive names (`forecast_probability` not `fp`)
- Write docstrings explaining *why*, not just *what*
- Keep functions short and focused

### They Test the Scary Parts

You can't test everything, so focus on:
- Edge calculations (this is literally your money)
- Position sizing (wrong sizing = blowup)
- Data parsing (garbage in = garbage out)

### They Keep Logs Like Detectives

When something goes wrong at 2 AM, logs are your only witnesses. Log:
- Every trade decision and why
- Every API call and response time
- Every error, with full context

---

<a name="pitfalls-to-avoid"></a>
## 9. Pitfalls to Avoid

### 🚫 Don't Trade Real Money Until...

1. You've paper traded for at least 2 weeks
2. You've verified your forecasts against actual temperatures
3. You've seen your system handle edge cases (API downtime, market close, etc.)
4. You understand every line of code that touches your money

### 🚫 Don't Trust Your Backtest

Backtests always look better than reality because:
- You can't backtest execution slippage perfectly
- Markets were different in the past
- You might be overfitting to historical data

Always assume real performance will be 30-50% worse than backtest.

### 🚫 Don't Ignore Correlation

If you're trading NYC, Atlanta, and Seattle simultaneously, remember they're not independent. A cold front affecting the entire East Coast could make all your positions lose together.

This is why we cap position sizes per trade AND track total exposure.

### 🚫 Don't Bet on Low-Liquidity Markets

If a market has $500 in liquidity and you try to buy $200, you'll move the price against yourself. Stick to markets with at least 10x your position size in liquidity.

### 🚫 Don't Forget About Fees

Kalshi has low trading fees, and:
- You pay spreads (bid-ask difference)
- You pay slippage on larger orders

Factor these into your edge calculations.

---

<a name="running-the-system"></a>
## 10. Running the System

### First Time Setup

```bash
# Navigate to project
cd weather_trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Create your config
cp .env.example .env
```

### Configure `.env`

```bash
# Required for Tomorrow.io (optional but recommended)
TOMORROW_IO_API_KEY=your_key_here

# Required for real trading (not for dry run)
KALSHI_KEY_ID=your_kalshi_api_key_id
KALSHI_PRIVATE_KEY_PATH=path/to/kalshi_private_key.pem

# Optional: alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Run in Dry Mode (No Real Trades)

```bash
# Single cycle
python -m weather_trader.main --dry-run

# Scheduled operation
python -m weather_trader.scheduler --dry-run
```

### Run with Real Money (CAREFUL!)

```bash
python -m weather_trader.main --live --bankroll 100
```

### Run Tests

```bash
pytest weather_trader/tests/ -v
```

---

## Final Thoughts

Perry, you now have a complete algorithmic trading system. It's not a magic money printer—no system is. But it's built on sound principles:

1. **Multiple data sources** beat single sources
2. **Calibration to ground truth** creates edge
3. **Proper position sizing** prevents blowups
4. **Discipline and automation** remove emotion

The code is clean, tested, and documented. The architecture is modular and extensible. The strategy is grounded in math, not hope.

But the most valuable thing isn't the code—it's what you've learned building it:
- How to design systems that don't fall over
- How to think about risk and uncertainty
- How to build something that runs while you sleep

That's what separates engineers from coders.

Now go make it rain. ☔📈

---

*Last updated: January 2025*
*Questions? Issues? You know where to find me.*
