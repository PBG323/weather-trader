# Opus 4.6 Recommendations: Analysis & Implementation Plan

## Overview

Six recommendations were identified for improving the weather trading system. This plan organizes them by priority and provides implementation details.

---

## Code Analysis Findings

### Data Availability (CONFIRMED)
From `kalshi/markets.py` and `dashboard.py`:
- **yes_bid, yes_ask**: Available from Kalshi API (lines 611-614)
- **volume**: Available (market.volume)
- **open_interest**: Available (market.open_interest)

### Current Implementation Locations
| Component | File | Lines |
|-----------|------|-------|
| Edge calculation | dashboard.py | 2460 |
| Probability calc (normal dist) | ensemble.py | 87-88, 128-129 |
| Std floor | ensemble.py | 85, 126 (`max(std, 2.0)`) |
| MIN_FORECAST_UNCERTAINTY | ensemble.py | 285 (1.5°F) |
| Confidence calculation | ensemble.py | 341-358 |

---

## Priority Matrix

| # | Issue | Priority | Effort | Impact | Status |
|---|-------|----------|--------|--------|--------|
| 1 | Spread-Adjusted Edge | HIGH | Low | High | Pending |
| 2 | Skew-Normal Distribution | HIGH | Medium | Medium-High | Pending |
| 3 | Forecast Horizon Decay | MEDIUM | Low | Medium | Pending |
| 4 | Volume-Based Filtering | MEDIUM | Low | Medium | Pending |
| 5 | Minimum Std Floor Adjustment | LOW | Low | Low | Pending |
| 6 | Backtest Framework | LOWER | High | High (long-term) | Pending |

---

## Phase 1: Quick Wins (Low Effort, High Impact)

### 1.1 Spread-Adjusted Edge

**Current Problem:**
- Edge calculation at `dashboard.py:2460`: `edge = our_prob - market_prob`
- Uses `yes_price` directly (which is yes_bid)
- Doesn't account for bid-ask spread
- A 5% edge could be consumed by spread costs

**Data Available:**
- `market.get("yes_bid")` - what you can sell at
- `market.get("yes_ask")` - what you can buy at
- Already extracted in dashboard.py lines 611-614

**Implementation Location:** `dashboard.py` around line 2434

```python
# BEFORE current code at line 2434:
market_prob = market["yes_price"]

# REPLACE WITH:
yes_bid = market.get("yes_bid", 0) / 100.0 if market.get("yes_bid") else 0
yes_ask = market.get("yes_ask", 0) / 100.0 if market.get("yes_ask") else 0

# Use mid-price for probability comparison
if yes_bid > 0 and yes_ask > 0:
    market_prob = (yes_bid + yes_ask) / 2
    spread = yes_ask - yes_bid
else:
    market_prob = market["yes_price"]  # Fallback
    spread = 0.02  # Assume 2% spread if unknown

# Later, after edge calculation at line 2460:
raw_edge = our_prob - market_prob
effective_edge = abs(raw_edge) - (spread / 2)

# Modify signal thresholds to use effective_edge
```

**Display Changes:**
- Show both "Raw Edge" and "Effective Edge" in signals table
- Add spread column to market display

---

### 1.2 Forecast Horizon Confidence Decay

**Current Problem:**
- Confidence at `ensemble.py:358`: `confidence = min(0.95, model_agreement * 0.9 * station_reliability)`
- No horizon decay applied
- 7-day forecast treated same as 1-day forecast

**Current Forecast Structure:**
- `EnsembleForecast.date` - already tracks forecast date
- Can calculate horizon as `(forecast_date - date.today()).days`

**Implementation Location:** `ensemble.py` around line 358

```python
# After calculating model_agreement and station_reliability (line 356)

# Add horizon decay
horizon_days = (forecast_date - date.today()).days
# ~8% decay per day, floor at 50%
horizon_factor = max(0.5, 1.0 - 0.08 * horizon_days)

# Final confidence with horizon decay
confidence = min(0.95, model_agreement * 0.9 * station_reliability * horizon_factor)
```

**Alternative:** Apply in dashboard.py signal generation where target_date is known

**Decay Schedule:**
| Days Out | Horizon Factor | If base = 85% |
|----------|----------------|---------------|
| 0 (today) | 1.00 | 85% |
| 1 (tomorrow) | 0.92 | 78% |
| 2 | 0.84 | 71% |
| 3 | 0.76 | 65% |
| 7 | 0.50 | 42% |

**Note:** Current trading focuses on same-day and next-day, so impact is moderate.

---

### 1.3 Volume-Based Filtering

**Current Problem:**
- Low-volume brackets may have stale/manipulated prices
- Trading illiquid markets increases slippage risk

**Data Available (CONFIRMED):**
- `market.get("volume", 0)` - Daily volume
- `market.get("open_interest", 0)` - Open interest
- Already stored in `TemperatureBracket` dataclass (`kalshi/markets.py:40-41`)

**Implementation Location:** `dashboard.py` signal generation loop, around line 2410

```python
# After getting market data, before processing:
volume = market.get("volume", 0)
open_interest = market.get("open_interest", 0)  # Mapped from "liquidity" in some places

# Liquidity thresholds
MIN_VOLUME = 100  # Start conservative for weather markets
MIN_OPEN_INTEREST = 50

if volume < MIN_VOLUME and open_interest < MIN_OPEN_INTEREST:
    # Log skip for transparency
    print(f"[Signal] SKIP {ticker}: low liquidity (vol={volume}, OI={open_interest})")
    continue
```

**Display Changes:**
- Add volume/OI columns to signals table
- Show liquidity warning icon for low-liquidity signals

**Note:** Weather markets are newer and may have lower typical volumes.
Should analyze actual volume data before setting thresholds.

---

## Phase 2: Distribution Improvements (Medium Effort)

### 2.1 Skew-Normal Distribution

**Current Implementation:**
- `ensemble.py:87-88`: `stats.norm.cdf(threshold, loc=mean, scale=std)`
- `dashboard.py:1305`: `stats.norm.cdf(...)` in `calc_outcome_probability()`
- Pure normal distribution assumption

**The Problem:**
- Temperature distributions are often skewed:
  - Cold fronts: sharp drops (left skew)
  - Warm spells: long tails (right skew)
- Normal distribution underestimates tail probabilities by 5-15%

**Implementation Locations:**
1. `ensemble.py` - Add skew estimation to `create_ensemble()`
2. `EnsembleForecast` - Add `high_skew`, `low_skew` fields
3. `get_probability_above/below/in_range` - Use skewnorm
4. `dashboard.py:calc_outcome_probability()` - Accept skew parameter

```python
# In ensemble.py create_ensemble(), after calculating highs array:
from scipy.stats import skewnorm

def estimate_skew(values: np.ndarray) -> float:
    """Estimate skew from model spread using Pearson's coefficient."""
    if len(values) < 3:
        return 0.0

    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)

    if std < 0.5:
        return 0.0  # Tight consensus, normal is fine

    # Pearson skewness: 3 * (mean - median) / std
    skew = 3 * (mean - median) / std
    return np.clip(skew, -2.0, 2.0)

# In probability calculations:
if abs(skew) < 0.1:
    prob = stats.norm.cdf(threshold, loc=mean, scale=std)
else:
    prob = skewnorm.cdf(threshold, a=skew, loc=mean, scale=std)
```

**Phased Approach:**
1. First, log skew values to understand typical ranges
2. Then implement skew-normal calculation
3. Compare predictions to actuals

**Computational Note:** skewnorm.cdf is ~3x slower than norm.cdf, but still <1ms per call.

---

### 2.2 Minimum Std Floor Adjustment

**Current Implementation:**
- `ensemble.py:85, 126`: `std = max(std, 2.0)` in probability methods
- `ensemble.py:285`: `MIN_FORECAST_UNCERTAINTY = 1.5` in combined uncertainty

**The Problem:**
- Floor of 2.0°F artificially widens distribution
- When models strongly agree (e.g., summer high pressure), this:
  - Reduces probability for center brackets
  - Reduces calculated edge
  - May cause missed trades

**Consensus is already tracked at line 343-344:**
```python
consensus_count = np.sum(np.abs(highs - median_high) <= 3.0)
consensus_ratio = consensus_count / len(highs)
```

**Implementation Location:** `ensemble.py` lines 85, 126

```python
# Replace simple floor with consensus-aware floor:
def get_std_floor(consensus_ratio: float, raw_std: float) -> float:
    """Apply minimum std with consensus adjustment."""
    if consensus_ratio > 0.9 and raw_std < 1.0:
        return 1.0  # Very strong consensus
    elif consensus_ratio > 0.8:
        return 1.5  # Strong consensus
    else:
        return 2.0  # Normal case

# In get_probability_above (line 85):
# Would need to pass consensus_ratio to methods, or store in dataclass
```

**Challenge:** Probability methods don't have access to consensus_ratio.
**Solution:** Add `consensus_ratio` field to `EnsembleForecast` dataclass.

**Risk Assessment:** LOW - if models strongly agree and are wrong, smaller std just means the error is more concentrated. Historical validation needed.

---

## Phase 3: Major Infrastructure (High Effort)

### 3.1 Backtest Framework

**Purpose:**
- Validate strategy changes before live deployment
- Calculate realistic Sharpe ratio
- Identify seasonal patterns
- Measure forecast accuracy over time

**Components Needed:**

1. **Historical Data Collection**
   - Past forecasts (ensemble predictions)
   - Actual temperatures (settlements)
   - Historical market prices
   - Trade history

2. **Backtest Engine**
   ```python
   class BacktestEngine:
       def __init__(self, start_date, end_date):
           self.start_date = start_date
           self.end_date = end_date
           self.trades = []
           self.pnl = []

       def run(self, strategy):
           for date in date_range(self.start_date, self.end_date):
               # Load historical data for this date
               forecasts = load_historical_forecasts(date)
               markets = load_historical_markets(date)
               actuals = load_actual_temperatures(date)

               # Run strategy
               signals = strategy.generate_signals(forecasts, markets)

               # Simulate trades
               for signal in signals:
                   self.simulate_trade(signal, actuals)

           return self.calculate_metrics()
   ```

3. **Metrics to Calculate**
   - Win rate
   - Average edge vs realized edge
   - Sharpe ratio
   - Max drawdown
   - Profit factor
   - Forecast accuracy (RMSE, MAE)

**Data Sources:**
- Tomorrow.io historical (API limits?)
- NWS Climate Data Online
- Kalshi historical prices (if available)
- Our own logged forecasts/trades

**Effort Estimate:** 2-3 weeks for basic framework

---

## Implementation Order

### Phase 1: Quick Wins (This Session)
**All data is available - can implement immediately**

| Step | Task | File | Lines |
|------|------|------|-------|
| 1a | Add spread calculation to signal loop | dashboard.py | ~2434 |
| 1b | Calculate effective_edge = raw_edge - spread/2 | dashboard.py | ~2460 |
| 1c | Add spread/effective_edge to signal display | dashboard.py | signals table |
| 2 | Add volume/OI filter with logging | dashboard.py | ~2410 |
| 3 | Add horizon_factor to confidence calc | ensemble.py | ~358 |

### Phase 2: Distribution Improvements (Next Session)

| Step | Task | File | Lines |
|------|------|------|-------|
| 4a | Add skew estimation function | ensemble.py | new function |
| 4b | Store skew in EnsembleForecast | ensemble.py | dataclass |
| 4c | Use skewnorm in probability calcs | ensemble.py | 87, 128 |
| 5 | Add consensus_ratio to dataclass | ensemble.py | dataclass |
| 6 | Make std floor consensus-aware | ensemble.py | 85, 126 |

### Phase 3: Backtest Framework (Future)

| Step | Task | Effort |
|------|------|--------|
| 7a | Design historical data schema | 2 hours |
| 7b | Create data collection scripts | 4 hours |
| 7c | Build backtest engine | 8 hours |
| 7d | Create reporting/visualization | 4 hours |

---

## Implementation Checklist

### Phase 1 COMPLETE
- [x] Data availability confirmed (bid/ask/volume/OI all present)
- [x] Exact file locations identified
- [x] Implementation code drafted
- [x] Implement spread-adjusted edge (dashboard.py)
- [x] Implement volume filtering (dashboard.py)
- [x] Implement horizon decay (ensemble.py)

### Phase 2 COMPLETE
- [x] Add skew estimation function (ensemble.py)
- [x] Add high_skew, low_skew fields to EnsembleForecast
- [x] Add consensus_ratio field to EnsembleForecast
- [x] Use skewnorm for probability calculations
- [x] Implement consensus-aware std floor

### Phase 3 Planning Needed
- [ ] Identify historical data sources
- [ ] Design data retention policy
- [ ] Define backtest metrics

---

## Summary of Changes Made

### dashboard.py
1. **Spread-Adjusted Edge**: Uses mid-price for market comparison, calculates effective_edge after spread cost
2. **Volume Filtering**: Skips markets with volume < 50 AND open_interest < 25
3. **Signal data**: Added raw_edge, effective_edge, spread, yes_bid, yes_ask, open_interest fields

### ensemble.py
1. **Skew Estimation**: Added `estimate_skew()` function using Pearson's coefficient
2. **EnsembleForecast**: Added high_skew, low_skew, consensus_ratio fields
3. **Probability Methods**: Use skewnorm when |skew| >= 0.1, otherwise normal
4. **Horizon Decay**: Confidence decays 8%/day, floor at 50%
5. **Consensus-Aware Std Floor**:
   - consensus > 90%: floor = 1.0°F
   - consensus > 80%: floor = 1.5°F
   - otherwise: floor = 2.0°F

---

*Created: 2026-02-06*
*Last updated: 2026-02-06*
*Phases 1 & 2 implemented: 2026-02-06*
