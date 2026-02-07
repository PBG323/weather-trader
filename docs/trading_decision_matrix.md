# Trading Decision Matrix

## Confidence-Adjusted Probability System

### How It Works

**Key Formula:**
- `confidence_factor = 1.0 - (confidence - 0.5)` clamped to [0.5, 1.2]
- `adjusted_std = forecast_std × confidence_factor` (floor at 1.5°F)

| Confidence | Factor | If base std = 4°F → adjusted |
|------------|--------|------------------------------|
| 90% | 0.6 | 2.4°F (tighter) |
| 85% | 0.65 | 2.6°F |
| 80% | 0.7 | 2.8°F |
| 75% | 0.75 | 3.0°F |
| 70% | 0.8 | 3.2°F |

---

## Trading Scenarios

### Scenario 1: High Confidence + Forecast Centered in Bracket
**Setup:** Denver 64-66°F, forecast 65.1°F, 87% confidence, price 42¢

**Calculation:**
- adjusted_std = 4.0 × 0.63 = 2.5°F
- P(63.5 ≤ T ≤ 66.5) with mean=65.1, std=2.5 = ~73%
- Edge = 73% - 42% = +31%

**Signal:** CONVICTION YES (≥80% conf, ≤45¢, forecast inside bracket)
**Action:** Auto-trade executes YES buy

---

### Scenario 2: High Confidence + Forecast Just Outside Bracket
**Setup:** Chicago 38-40°F, forecast 37.2°F, 85% confidence, price 35¢

**Calculation:**
- adjusted_std = 3.5 × 0.65 = 2.3°F
- Forecast 37.2°F is ~0.8°F below bracket (38-40)
- P(37.5 ≤ T ≤ 40.5) with mean=37.2, std=2.3 = ~42%
- Edge = 42% - 35% = +7%

**Signal:** BUY YES (7% edge > 5% threshold)
**Action:** Not a conviction trade (forecast NOT inside bracket), but still BUY YES
**Note:** Tighter distribution due to high confidence means less probability in bracket

---

### Scenario 3: High Confidence Tail Bet - FLAGGED
**Setup:** Denver ≤62°F, forecast 64.1°F, 90% confidence, price 28¢

**Calculation:**
- adjusted_std = 4.5 × 0.6 = 2.7°F
- Forecast 64.1°F is 2.1°F ABOVE the bracket ceiling (62°F)
- P(T ≤ 62.5) with mean=64.1, std=2.7 = ~27%
- Edge = 27% - 28% = -1% (essentially no edge)

**Detection:** `is_tail_bet_against_forecast = True`
- Why: ≥80% conf + forecast > temp_high + 1 → betting against our forecast

**Signal:** PASS (near zero edge)
**Action:** SKIPPED in auto-trade due to tail bet warning
**Reason:** We're 90% confident temp will be ~64°F, so betting it'll be ≤62°F is contradictory

---

### Scenario 4: High Confidence Strengthens BUY YES Edge
**Setup:** Miami 78-80°F, forecast 79.3°F, 92% confidence, price 55¢

**Calculation:**
- adjusted_std = 3.5 × 0.58 = 2.0°F (tighter due to high confidence)
- P(77.5 ≤ T ≤ 80.5) with mean=79.3, std=2.0 = ~76%
- Edge = 76% - 55% = +21%

**Signal:** STRONG BUY YES (edge > 10%)
**Action:** Auto-trade executes YES buy with full size

---

### Scenario 5: High Confidence BUY NO - Market Overpriced YES
**Setup:** NYC 42-44°F, forecast 47.2°F, 88% confidence, price 62¢

**Calculation:**
- adjusted_std = 4.0 × 0.62 = 2.5°F
- Forecast 47.2°F is ~3°F ABOVE bracket
- P(41.5 ≤ T ≤ 44.5) with mean=47.2, std=2.5 = ~14%
- Edge = 14% - 62% = -48%

**Signal:** STRONG BUY NO (edge < -10%)
**Action:** Auto-trade executes NO buy
**Note:** High confidence REDUCES probability for brackets far from forecast

---

### Scenario 6: Low Confidence Widens Distribution
**Setup:** LA 72-74°F, forecast 73.1°F, 68% confidence, price 45¢

**Calculation:**
- adjusted_std = 4.5 × 0.82 = 3.7°F (wider due to low confidence)
- P(71.5 ≤ T ≤ 74.5) with mean=73.1, std=3.7 = ~33%
- Edge = 33% - 45% = -12%

**Signal:** STRONG BUY NO
**Action:** Auto-trade executes NO buy
**Note:** Low confidence = more uncertainty = probability spread wider = less concentration in any bracket

---

### Scenario 7: Same-Day with Uncertainty Adjustment
**Setup:** Chicago 38-40°F same-day, forecast 39.0°F, 85% base confidence, current temp 41°F at 11 AM, price 48¢

**Adjustments:**
- Base confidence: 85%
- Uncertainty adjustment factor: 0.9 (still uncertain)
- Adjusted confidence: 85% × 0.9 = 76.5%
- confidence_factor = 1.0 - (0.765 - 0.5) = 0.735
- adjusted_std = 3.5 × 0.735 = 2.6°F

**Calculation:**
- P(37.5 ≤ T ≤ 40.5) with mean=39.0, std=2.6 = ~54%
- Edge = 54% - 48% = +6%

**Signal:** BUY YES (6% > 5%)
**Additional check:** Same-day requires 6.25% edge minimum
**Action:** SKIPPED (6% < 6.25% same-day threshold)

---

### Scenario 8: Conviction Trade Just Misses Criteria
**Setup:** Denver 62-64°F, forecast 63.2°F, 78% confidence, price 43¢

**Conviction Check:**
- Is forecast inside bracket? ✓ (63.2 between 62-64)
- Is confidence ≥ 80%? ✗ (78% < 80%)
- Is price ≤ 45¢? ✓ (43¢)
- Is this a range bracket? ✓

**Result:** NOT a conviction trade (fails confidence threshold)

**Standard Calculation:**
- adjusted_std = 4.5 × 0.72 = 3.2°F
- P(61.5 ≤ T ≤ 64.5) with mean=63.2, std=3.2 = ~38%
- Edge = 38% - 43% = -5%

**Signal:** BUY NO (edge ≤ -5%)
**Action:** Auto-trade executes NO buy

---

## Summary Decision Matrix

| Scenario | Confidence | Edge | Conviction? | Tail Bet? | Final Action |
|----------|------------|------|-------------|-----------|--------------|
| Centered high conf | 87% | +31% | ✓ YES | No | **CONVICTION YES** |
| Just outside | 85% | +7% | No | No | **BUY YES** |
| Tail bet | 90% | -1% | No | ⚠️ YES | **SKIP (warning)** |
| Strong YES | 92% | +21% | No | No | **STRONG BUY YES** |
| Strong NO | 88% | -48% | No | No | **STRONG BUY NO** |
| Low conf | 68% | -12% | No | No | **STRONG BUY NO** |
| Same-day | 76% adj | +6% | No | No | **SKIP (threshold)** |
| Near miss | 78% | -5% | No | No | **BUY NO** |

---

## Key Thresholds (Configurable)

| Parameter | Value | Location |
|-----------|-------|----------|
| Minimum confidence to trade | 75% | `TradingDefaults.MIN_CONFIDENCE` |
| Conviction trade min confidence | 80% | `check_conviction_trade()` |
| Conviction trade max price | 45¢ | `check_conviction_trade()` |
| Standard edge threshold | 5% | Signal generation |
| Same-day edge threshold | 6.25% | Signal generation |
| Tail bet detection threshold | 1°F outside bracket | Signal generation |

---

## System Behavior Summary

The confidence-adjusted probability system:

1. **Tightens** probabilities around forecast when confidence is high
2. **Widens** probabilities when confidence is low
3. **Detects and warns** on tail bets against high-confidence forecasts
4. **Enables conviction trades** for centered forecasts with high confidence and low prices

---

*Last updated: 2026-02-06*
