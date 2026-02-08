"""
Trade History Tracker

Logs every trade with entry conditions and tracks settlement results.
Provides analysis to learn from historical performance and adjust strategy.

Data stored in: data/trade_history.json
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")
HISTORY_FILE = Path(__file__).parent.parent / "data" / "trade_history.json"


@dataclass
class TradeEntry:
    """Record of a single trade."""
    # Identification
    trade_id: str
    timestamp: str  # ISO format
    ticker: str
    city: str
    target_date: str  # The date the market settles on

    # Bracket info
    bracket_desc: str  # e.g., "64-66°F" or "≤62°F"
    temp_low: Optional[float]
    temp_high: Optional[float]

    # Entry conditions
    side: str  # "YES" or "NO"
    entry_price: float  # 0-1
    size_contracts: int
    size_dollars: float

    # Forecast at entry
    forecast_mean: float
    forecast_std: float
    forecast_confidence: float
    our_probability: float
    raw_edge: float
    effective_edge: float
    spread: float

    # Signal info
    signal_type: str  # "BUY YES", "STRONG BUY NO", "CONVICTION YES", etc.
    is_same_day: bool
    is_conviction: bool

    # Settlement (filled in later)
    settled: bool = False
    actual_temp: Optional[float] = None
    settlement_result: Optional[str] = None  # "WIN" or "LOSS"
    settlement_price: Optional[float] = None  # 1.0 for win, 0.0 for loss
    pnl: Optional[float] = None

    # Analysis flags (filled in after settlement)
    forecast_error: Optional[float] = None  # actual - forecast
    was_correct_side: Optional[bool] = None  # Did we bet the right direction?


def load_trade_history() -> List[Dict]:
    """Load trade history from disk."""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading trade history: {e}")
    return []


def save_trade_history(history: List[Dict]) -> bool:
    """Save trade history to disk."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving trade history: {e}")
        return False


def generate_trade_id() -> str:
    """Generate unique trade ID."""
    now = datetime.now(EST)
    return f"T{now.strftime('%Y%m%d%H%M%S%f')}"


def log_trade(
    ticker: str,
    city: str,
    target_date: str,
    bracket_desc: str,
    temp_low: Optional[float],
    temp_high: Optional[float],
    side: str,
    entry_price: float,
    size_contracts: int,
    size_dollars: float,
    forecast_mean: float,
    forecast_std: float,
    forecast_confidence: float,
    our_probability: float,
    raw_edge: float,
    effective_edge: float,
    spread: float,
    signal_type: str,
    is_same_day: bool,
    is_conviction: bool = False,
) -> str:
    """
    Log a new trade entry.

    Returns:
        trade_id for tracking
    """
    trade_id = generate_trade_id()

    entry = TradeEntry(
        trade_id=trade_id,
        timestamp=datetime.now(EST).isoformat(),
        ticker=ticker,
        city=city,
        target_date=target_date,
        bracket_desc=bracket_desc,
        temp_low=temp_low,
        temp_high=temp_high,
        side=side,
        entry_price=entry_price,
        size_contracts=size_contracts,
        size_dollars=size_dollars,
        forecast_mean=forecast_mean,
        forecast_std=forecast_std,
        forecast_confidence=forecast_confidence,
        our_probability=our_probability,
        raw_edge=raw_edge,
        effective_edge=effective_edge,
        spread=spread,
        signal_type=signal_type,
        is_same_day=is_same_day,
        is_conviction=is_conviction,
    )

    history = load_trade_history()
    history.append(asdict(entry))
    save_trade_history(history)

    print(f"[TradeHistory] Logged trade {trade_id}: {city} {bracket_desc} {side} @ {entry_price:.2f}")
    return trade_id


def update_settlement(
    ticker: str,
    actual_temp: float,
    won: bool,
) -> bool:
    """
    Update trade(s) with settlement result.

    Args:
        ticker: Market ticker
        actual_temp: Actual temperature that settled
        won: Whether our position won

    Returns:
        True if trade was found and updated
    """
    history = load_trade_history()
    updated = False

    for trade in history:
        if trade["ticker"] == ticker and not trade.get("settled", False):
            trade["settled"] = True
            trade["actual_temp"] = actual_temp
            trade["settlement_result"] = "WIN" if won else "LOSS"
            trade["settlement_price"] = 1.0 if won else 0.0

            # Calculate P&L
            entry_price = trade["entry_price"]
            size = trade["size_dollars"]
            if trade["side"] == "YES":
                if won:
                    trade["pnl"] = size * (1.0 - entry_price) / entry_price
                else:
                    trade["pnl"] = -size
            else:  # NO side
                if won:
                    trade["pnl"] = size * entry_price / (1.0 - entry_price)
                else:
                    trade["pnl"] = -size

            # Calculate forecast error
            trade["forecast_error"] = actual_temp - trade["forecast_mean"]

            # Was our side correct?
            # For YES bets: we win if actual temp is in the bracket
            # For NO bets: we win if actual temp is outside the bracket
            trade["was_correct_side"] = won

            updated = True
            print(f"[TradeHistory] Updated {trade['trade_id']}: {trade['settlement_result']} (actual: {actual_temp}°F, P&L: ${trade['pnl']:.2f})")

    if updated:
        save_trade_history(history)

    return updated


def get_unsettled_trades() -> List[Dict]:
    """Get all trades awaiting settlement."""
    history = load_trade_history()
    return [t for t in history if not t.get("settled", False)]


def get_trades_for_date(target_date: str) -> List[Dict]:
    """Get all trades for a specific target date."""
    history = load_trade_history()
    return [t for t in history if t.get("target_date") == target_date]


def analyze_performance() -> Dict[str, Any]:
    """
    Analyze historical trade performance.

    Returns comprehensive statistics for strategy tuning.
    """
    history = load_trade_history()
    settled = [t for t in history if t.get("settled", False)]

    if not settled:
        return {"error": "No settled trades to analyze"}

    # Overall stats
    total_trades = len(settled)
    wins = [t for t in settled if t["settlement_result"] == "WIN"]
    losses = [t for t in settled if t["settlement_result"] == "LOSS"]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0

    total_pnl = sum(t.get("pnl", 0) for t in settled)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    # By confidence bucket
    confidence_buckets = {
        "90%+": {"trades": [], "wins": 0, "total": 0},
        "80-90%": {"trades": [], "wins": 0, "total": 0},
        "75-80%": {"trades": [], "wins": 0, "total": 0},
        "<75%": {"trades": [], "wins": 0, "total": 0},
    }

    for t in settled:
        conf = t.get("forecast_confidence", 0)
        if conf >= 0.90:
            bucket = "90%+"
        elif conf >= 0.80:
            bucket = "80-90%"
        elif conf >= 0.75:
            bucket = "75-80%"
        else:
            bucket = "<75%"

        confidence_buckets[bucket]["trades"].append(t)
        confidence_buckets[bucket]["total"] += 1
        if t["settlement_result"] == "WIN":
            confidence_buckets[bucket]["wins"] += 1

    confidence_analysis = {}
    for bucket, data in confidence_buckets.items():
        if data["total"] > 0:
            confidence_analysis[bucket] = {
                "count": data["total"],
                "win_rate": data["wins"] / data["total"],
                "avg_pnl": sum(t.get("pnl", 0) for t in data["trades"]) / data["total"],
            }

    # By edge bucket
    edge_buckets = {
        "15%+": {"trades": [], "wins": 0, "total": 0},
        "10-15%": {"trades": [], "wins": 0, "total": 0},
        "5-10%": {"trades": [], "wins": 0, "total": 0},
        "<5%": {"trades": [], "wins": 0, "total": 0},
    }

    for t in settled:
        edge = abs(t.get("effective_edge", 0))
        if edge >= 0.15:
            bucket = "15%+"
        elif edge >= 0.10:
            bucket = "10-15%"
        elif edge >= 0.05:
            bucket = "5-10%"
        else:
            bucket = "<5%"

        edge_buckets[bucket]["trades"].append(t)
        edge_buckets[bucket]["total"] += 1
        if t["settlement_result"] == "WIN":
            edge_buckets[bucket]["wins"] += 1

    edge_analysis = {}
    for bucket, data in edge_buckets.items():
        if data["total"] > 0:
            edge_analysis[bucket] = {
                "count": data["total"],
                "win_rate": data["wins"] / data["total"],
                "avg_pnl": sum(t.get("pnl", 0) for t in data["trades"]) / data["total"],
            }

    # By city
    city_stats = {}
    for t in settled:
        city = t.get("city", "Unknown")
        if city not in city_stats:
            city_stats[city] = {"wins": 0, "losses": 0, "pnl": 0, "forecast_errors": []}

        if t["settlement_result"] == "WIN":
            city_stats[city]["wins"] += 1
        else:
            city_stats[city]["losses"] += 1
        city_stats[city]["pnl"] += t.get("pnl", 0)
        if t.get("forecast_error") is not None:
            city_stats[city]["forecast_errors"].append(t["forecast_error"])

    city_analysis = {}
    for city, data in city_stats.items():
        total = data["wins"] + data["losses"]
        errors = data["forecast_errors"]
        city_analysis[city] = {
            "total_trades": total,
            "win_rate": data["wins"] / total if total > 0 else 0,
            "total_pnl": data["pnl"],
            "avg_forecast_error": sum(errors) / len(errors) if errors else 0,
            "forecast_bias": "warm" if sum(errors) / len(errors) > 0.5 else "cold" if sum(errors) / len(errors) < -0.5 else "neutral" if errors else "unknown",
        }

    # By signal type
    signal_stats = {}
    for t in settled:
        sig = t.get("signal_type", "Unknown")
        if sig not in signal_stats:
            signal_stats[sig] = {"wins": 0, "losses": 0, "pnl": 0}
        if t["settlement_result"] == "WIN":
            signal_stats[sig]["wins"] += 1
        else:
            signal_stats[sig]["losses"] += 1
        signal_stats[sig]["pnl"] += t.get("pnl", 0)

    signal_analysis = {}
    for sig, data in signal_stats.items():
        total = data["wins"] + data["losses"]
        signal_analysis[sig] = {
            "count": total,
            "win_rate": data["wins"] / total if total > 0 else 0,
            "total_pnl": data["pnl"],
        }

    # Same-day vs next-day
    same_day = [t for t in settled if t.get("is_same_day", False)]
    next_day = [t for t in settled if not t.get("is_same_day", False)]

    same_day_analysis = {
        "count": len(same_day),
        "win_rate": len([t for t in same_day if t["settlement_result"] == "WIN"]) / len(same_day) if same_day else 0,
        "total_pnl": sum(t.get("pnl", 0) for t in same_day),
    }

    next_day_analysis = {
        "count": len(next_day),
        "win_rate": len([t for t in next_day if t["settlement_result"] == "WIN"]) / len(next_day) if next_day else 0,
        "total_pnl": sum(t.get("pnl", 0) for t in next_day),
    }

    # Conviction trades
    conviction = [t for t in settled if t.get("is_conviction", False)]
    conviction_analysis = {
        "count": len(conviction),
        "win_rate": len([t for t in conviction if t["settlement_result"] == "WIN"]) / len(conviction) if conviction else 0,
        "total_pnl": sum(t.get("pnl", 0) for t in conviction),
    }

    # Forecast accuracy
    forecast_errors = [t["forecast_error"] for t in settled if t.get("forecast_error") is not None]
    if forecast_errors:
        mae = sum(abs(e) for e in forecast_errors) / len(forecast_errors)
        bias = sum(forecast_errors) / len(forecast_errors)
        rmse = (sum(e**2 for e in forecast_errors) / len(forecast_errors)) ** 0.5
    else:
        mae = bias = rmse = None

    return {
        "summary": {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
        },
        "by_confidence": confidence_analysis,
        "by_edge": edge_analysis,
        "by_city": city_analysis,
        "by_signal_type": signal_analysis,
        "same_day_vs_next_day": {
            "same_day": same_day_analysis,
            "next_day": next_day_analysis,
        },
        "conviction_trades": conviction_analysis,
        "forecast_accuracy": {
            "mean_absolute_error": mae,
            "bias": bias,
            "bias_direction": "warm" if bias and bias > 0.5 else "cold" if bias and bias < -0.5 else "neutral",
            "rmse": rmse,
        },
        "recommendations": generate_recommendations(settled),
    }


def generate_recommendations(settled_trades: List[Dict]) -> List[str]:
    """Generate actionable recommendations based on trade history."""
    recommendations = []

    if len(settled_trades) < 10:
        recommendations.append("Need more trades (10+) for meaningful analysis")
        return recommendations

    # Check overall win rate
    wins = len([t for t in settled_trades if t["settlement_result"] == "WIN"])
    win_rate = wins / len(settled_trades)

    if win_rate < 0.45:
        recommendations.append(f"⚠️ Win rate is low ({win_rate:.1%}). Consider raising minimum edge threshold.")
    elif win_rate > 0.65:
        recommendations.append(f"✅ Strong win rate ({win_rate:.1%}). Strategy is working well.")

    # Check by confidence level
    high_conf = [t for t in settled_trades if t.get("forecast_confidence", 0) >= 0.85]
    if len(high_conf) >= 5:
        high_conf_wins = len([t for t in high_conf if t["settlement_result"] == "WIN"])
        high_conf_wr = high_conf_wins / len(high_conf)
        if high_conf_wr < 0.50:
            recommendations.append(f"⚠️ High-confidence trades ({high_conf_wr:.1%} WR) underperforming. Model may be overconfident.")

    # Check forecast bias
    errors = [t["forecast_error"] for t in settled_trades if t.get("forecast_error") is not None]
    if errors:
        avg_error = sum(errors) / len(errors)
        if avg_error > 1.0:
            recommendations.append(f"⚠️ Forecasts are {avg_error:.1f}°F too cold on average. Consider warm bias adjustment.")
        elif avg_error < -1.0:
            recommendations.append(f"⚠️ Forecasts are {abs(avg_error):.1f}°F too warm on average. Consider cold bias adjustment.")

    # Check by city
    city_stats = {}
    for t in settled_trades:
        city = t.get("city", "Unknown")
        if city not in city_stats:
            city_stats[city] = {"wins": 0, "total": 0, "errors": []}
        city_stats[city]["total"] += 1
        if t["settlement_result"] == "WIN":
            city_stats[city]["wins"] += 1
        if t.get("forecast_error") is not None:
            city_stats[city]["errors"].append(t["forecast_error"])

    for city, stats in city_stats.items():
        if stats["total"] >= 5:
            wr = stats["wins"] / stats["total"]
            if wr < 0.35:
                recommendations.append(f"⚠️ {city} has poor win rate ({wr:.1%}). Consider excluding or adjusting bias.")
            if stats["errors"]:
                city_bias = sum(stats["errors"]) / len(stats["errors"])
                if abs(city_bias) > 1.5:
                    direction = "cold" if city_bias > 0 else "warm"
                    recommendations.append(f"📊 {city} forecast bias: {abs(city_bias):.1f}°F too {direction}.")

    # Check same-day performance
    same_day = [t for t in settled_trades if t.get("is_same_day", False)]
    if len(same_day) >= 5:
        sd_wins = len([t for t in same_day if t["settlement_result"] == "WIN"])
        sd_wr = sd_wins / len(same_day)
        if sd_wr < 0.40:
            recommendations.append(f"⚠️ Same-day trades underperforming ({sd_wr:.1%} WR). Consider tightening criteria.")

    # Check conviction trades
    conviction = [t for t in settled_trades if t.get("is_conviction", False)]
    if len(conviction) >= 3:
        conv_wins = len([t for t in conviction if t["settlement_result"] == "WIN"])
        conv_wr = conv_wins / len(conviction)
        if conv_wr < 0.50:
            recommendations.append(f"⚠️ Conviction trades underperforming ({conv_wr:.1%} WR). Review criteria.")
        elif conv_wr > 0.70:
            recommendations.append(f"✅ Conviction trades performing well ({conv_wr:.1%} WR).")

    if not recommendations:
        recommendations.append("✅ No immediate issues detected. Continue monitoring.")

    return recommendations


def get_recent_trades(limit: int = 20) -> List[Dict]:
    """Get most recent trades."""
    history = load_trade_history()
    return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]


def export_to_csv(filepath: str = None) -> str:
    """Export trade history to CSV for external analysis."""
    import csv

    if filepath is None:
        filepath = str(HISTORY_FILE.parent / "trade_history.csv")

    history = load_trade_history()
    if not history:
        return "No trades to export"

    fieldnames = list(history[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return filepath


def analyze_trade_outcome(trade: Dict, actual_temp: float) -> Dict:
    """
    Attribute trade outcome to specific factors.

    Analyzes WHY a trade won or lost to improve future decisions.

    Args:
        trade: Trade record from history
        actual_temp: Actual temperature that settled

    Returns:
        Dict with attribution analysis
    """
    forecast_temp = trade.get("forecast_mean", 0)
    forecast_error = actual_temp - forecast_temp

    our_prob = trade.get("our_probability", 0.5)
    market_prob = trade.get("entry_price", 0.5)  # Entry price approximates market prob

    # Determine if outcome fell in the bracket
    temp_low = trade.get("temp_low")
    temp_high = trade.get("temp_high")

    if temp_low is None and temp_high is not None:
        # "or_below" bracket
        actual_in_bracket = actual_temp <= temp_high
    elif temp_high is None and temp_low is not None:
        # "or_higher" bracket
        actual_in_bracket = actual_temp >= temp_low
    elif temp_low is not None and temp_high is not None:
        # Range bracket
        actual_in_bracket = temp_low <= actual_temp <= temp_high
    else:
        actual_in_bracket = None

    # Calculate actual probability (binary for settlement)
    actual_prob = 1.0 if actual_in_bracket else 0.0

    # Was our probability estimate better than market?
    our_error = abs(our_prob - actual_prob)
    market_error = abs(market_prob - actual_prob)
    edge_was_real = our_error < market_error

    # Spread cost as percentage of edge
    spread = trade.get("spread", 0.02)
    edge = trade.get("effective_edge", 0.05)
    spread_cost_ratio = (spread / 2) / edge if edge > 0 else 0

    # Identify which model was closest (if models data available)
    models = trade.get("models", [])
    best_model = None
    best_model_error = float("inf")

    for model in models:
        model_high = model.get("high", 0)
        error = abs(model_high - actual_temp)
        if error < best_model_error:
            best_model_error = error
            best_model = model.get("model", "unknown")

    return {
        "forecast_error_f": round(forecast_error, 1),
        "was_forecast_correct": abs(forecast_error) < 2.0,
        "actual_in_bracket": actual_in_bracket,
        "edge_was_real": edge_was_real,
        "our_prob_error": round(our_error, 3),
        "market_prob_error": round(market_error, 3),
        "spread_cost_ratio": round(spread_cost_ratio, 2),
        "model_closest": best_model,
        "model_closest_error": round(best_model_error, 1) if best_model else None,
        "confidence_at_entry": trade.get("forecast_confidence", 0),
        "lesson": _derive_lesson(forecast_error, edge_was_real, spread_cost_ratio)
    }


def _derive_lesson(forecast_error: float, edge_was_real: bool, spread_cost_ratio: float) -> str:
    """Generate actionable lesson from trade outcome."""
    lessons = []

    if abs(forecast_error) > 4:
        lessons.append("Large forecast error - review model weights")
    elif abs(forecast_error) > 2:
        lessons.append("Moderate forecast error - within expectations")
    else:
        lessons.append("Accurate forecast")

    if not edge_was_real:
        lessons.append("Market was more accurate than our forecast")
    else:
        lessons.append("Our edge was real")

    if spread_cost_ratio > 0.5:
        lessons.append("Spread cost ate >50% of edge - need larger edges")
    elif spread_cost_ratio > 0.3:
        lessons.append("Spread cost significant - consider spread in sizing")

    return "; ".join(lessons)


def get_model_performance(history: List[Dict] = None) -> Dict:
    """
    Analyze which models are most accurate.

    Returns ranking of models by forecast accuracy.
    """
    if history is None:
        history = load_trade_history()

    model_errors = {}  # model_name -> list of errors

    for trade in history:
        actual = trade.get("actual_temp")
        if actual is None:
            continue

        models = trade.get("models", [])
        for model in models:
            name = model.get("model", "unknown")
            high = model.get("high", 0)
            error = abs(high - actual)

            if name not in model_errors:
                model_errors[name] = []
            model_errors[name].append(error)

    # Calculate stats for each model
    results = {}
    for name, errors in model_errors.items():
        if len(errors) >= 3:  # Need minimum sample
            results[name] = {
                "count": len(errors),
                "mae": round(sum(errors) / len(errors), 2),
                "max_error": round(max(errors), 1),
                "min_error": round(min(errors), 1),
            }

    # Sort by MAE (lower is better)
    return dict(sorted(results.items(), key=lambda x: x[1]["mae"]))
