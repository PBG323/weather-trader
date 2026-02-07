"""
Scheduler State Manager

Handles scheduler state management for dashboard-integrated scheduling:
- Enable/disable state
- Cycle tracking (cycles today, API calls estimate)
- Activity logging
- Disk persistence to data/scheduler_state.json
- Daily reset at midnight
"""

import json
from datetime import datetime, date, time
from pathlib import Path
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

# EST timezone (Kalshi markets)
EST = ZoneInfo("America/New_York")

# State file location
STATE_FILE = Path(__file__).parent.parent / "data" / "scheduler_state.json"

# Import trading schedule from scheduler_optimized
TRADING_SCHEDULE = {
    # Early morning - Settlement window & new forecasts
    "early_morning": {
        "hours": [5, 6, 7],
        "minute_interval": 30,
        "description": "Settlement + morning forecasts",
    },
    # Mid-morning - Same-day trading active
    "mid_morning": {
        "hours": [8, 9, 10, 11],
        "minute_interval": 20,
        "description": "Same-day trading peak",
    },
    # Afternoon - Same-day final trades
    "afternoon": {
        "hours": [12, 13, 14],
        "minute_interval": 15,
        "description": "Peak temperature window",
    },
    # Late afternoon - Next-day positioning
    "late_afternoon": {
        "hours": [15, 16, 17],
        "minute_interval": 30,
        "description": "Next-day positioning",
    },
    # Evening - Key forecast updates
    "evening": {
        "hours": [18, 19, 20, 21],
        "minute_interval": 30,
        "description": "Evening forecast updates",
    },
    # Night - Overnight models
    "night": {
        "hours": [22, 23],
        "minute_interval": 30,
        "description": "Overnight model runs",
    },
    # Late night - Best next-day data
    "late_night": {
        "hours": [0, 1, 2],
        "minute_interval": 60,
        "description": "GFS/ECMWF model completion",
    },
    # Early early morning - Light monitoring
    "pre_dawn": {
        "hours": [3, 4],
        "minute_interval": 60,
        "description": "Pre-dawn monitoring",
    },
}


def get_default_state() -> Dict[str, Any]:
    """Return default scheduler state."""
    return {
        "enabled": False,
        "cycles_today": 0,
        "api_calls_today": 0,
        "last_cycle_time": None,
        "current_window": None,
        "activity_log": [],  # Last 50 entries
        "trades_today": [],
        "daily_reset_date": str(date.today()),
    }


def load_scheduler_state() -> Dict[str, Any]:
    """Load scheduler state from disk, returning default if not found."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            # Check for daily reset
            today = str(datetime.now(EST).date())
            if state.get("daily_reset_date") != today:
                state = reset_daily_counters(state)
                save_scheduler_state(state)

            return state
    except Exception as e:
        print(f"Error loading scheduler state: {e}")

    return get_default_state()


def save_scheduler_state(state: Dict[str, Any]) -> bool:
    """Save scheduler state to disk."""
    try:
        # Ensure data directory exists
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving scheduler state: {e}")
        return False


def get_current_window() -> tuple[str, Dict[str, Any]]:
    """
    Get the current trading window based on EST time.

    Returns:
        Tuple of (window_name, window_config) or ("unknown", {}) if outside windows
    """
    now = datetime.now(EST)
    current_hour = now.hour

    for window_name, config in TRADING_SCHEDULE.items():
        if current_hour in config["hours"]:
            return window_name, config

    return "unknown", {"description": "Outside trading windows", "minute_interval": 60}


def calculate_daily_cycles() -> Dict[str, Any]:
    """Calculate total expected cycles per day and by window."""
    cycles_by_window = {}
    total_cycles = 0

    for window_name, config in TRADING_SCHEDULE.items():
        hours = len(config["hours"])
        cycles_per_hour = 60 // config["minute_interval"]
        window_cycles = hours * cycles_per_hour
        cycles_by_window[window_name] = {
            "cycles": window_cycles,
            "description": config["description"]
        }
        total_cycles += window_cycles

    return {
        "total_cycles": total_cycles,
        "estimated_api_calls": total_cycles * 7,  # 7 cities
        "by_window": cycles_by_window
    }


def should_run_cycle(state: Dict[str, Any], dashboard_interval_seconds: int = 240) -> bool:
    """
    Determine if a scheduler cycle should run based on the current time window.

    The dashboard already handles the auto-refresh interval, but we use window
    information to track and log activity appropriately.

    Args:
        state: Current scheduler state
        dashboard_interval_seconds: Dashboard's refresh interval (default 240 = 4 min)

    Returns:
        True if we're in a valid trading window and scheduler is enabled
    """
    if not state.get("enabled", False):
        return False

    window_name, window_config = get_current_window()

    # Always allow cycles when scheduler is enabled - the dashboard controls timing
    # We just track which window we're in
    return window_name != "unknown"


def log_cycle(
    state: Dict[str, Any],
    signals_count: int,
    trades_count: int,
    details: str = "",
    trade_info: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Log a scheduler cycle and update counters.

    Args:
        state: Current scheduler state
        signals_count: Number of signals generated
        trades_count: Number of trades executed
        details: Optional additional details
        trade_info: Optional list of trade dictionaries

    Returns:
        Updated state dict
    """
    now = datetime.now(EST)
    window_name, window_config = get_current_window()

    # Update counters
    state["cycles_today"] = state.get("cycles_today", 0) + 1
    state["api_calls_today"] = state.get("api_calls_today", 0) + 7  # ~7 cities
    state["last_cycle_time"] = now.isoformat()
    state["current_window"] = window_name

    # Create log entry
    log_entry = {
        "time": now.strftime("%H:%M:%S"),
        "timestamp": now.isoformat(),
        "window": window_name,
        "signals": signals_count,
        "trades": trades_count,
        "details": details,
        "status": "trade" if trades_count > 0 else ("signal" if signals_count > 0 else "no_action")
    }

    # Add to activity log (keep last 50)
    activity_log = state.get("activity_log", [])
    activity_log.insert(0, log_entry)
    state["activity_log"] = activity_log[:50]

    # Track trades today
    if trade_info:
        trades_today = state.get("trades_today", [])
        for trade in trade_info:
            trade["logged_time"] = now.isoformat()
            trades_today.append(trade)
        state["trades_today"] = trades_today

    return state


def reset_daily_counters(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reset daily counters at midnight.

    Args:
        state: Current scheduler state

    Returns:
        Updated state with reset counters
    """
    today = str(datetime.now(EST).date())

    # Store yesterday's stats in log before reset
    if state.get("cycles_today", 0) > 0:
        yesterday_summary = {
            "time": "00:00:00",
            "timestamp": datetime.now(EST).isoformat(),
            "window": "daily_reset",
            "signals": 0,
            "trades": 0,
            "details": f"Day ended: {state.get('cycles_today', 0)} cycles, {state.get('api_calls_today', 0)} API calls",
            "status": "reset"
        }
        activity_log = state.get("activity_log", [])
        activity_log.insert(0, yesterday_summary)
        state["activity_log"] = activity_log[:50]

    # Reset counters
    state["cycles_today"] = 0
    state["api_calls_today"] = 0
    state["trades_today"] = []
    state["daily_reset_date"] = today

    return state


def get_schedule_summary() -> List[Dict[str, Any]]:
    """
    Get a summary of all trading windows for display.

    Returns:
        List of window info dicts sorted by start hour
    """
    summary = []

    for window_name, config in TRADING_SCHEDULE.items():
        hours = sorted(config["hours"])
        start_hour = hours[0]
        end_hour = hours[-1]

        summary.append({
            "name": window_name,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "hours_display": f"{start_hour:02d}:00 - {end_hour:02d}:59",
            "interval_minutes": config["minute_interval"],
            "description": config["description"],
            "cycles_per_hour": 60 // config["minute_interval"],
            "total_cycles": len(hours) * (60 // config["minute_interval"])
        })

    # Sort by start hour
    summary.sort(key=lambda x: (x["start_hour"] if x["start_hour"] >= 5 else x["start_hour"] + 24))

    return summary


def format_time_until_next_cycle(state: Dict[str, Any]) -> str:
    """Estimate time until next cycle based on dashboard refresh interval."""
    last_cycle = state.get("last_cycle_time")
    if not last_cycle:
        return "Starting..."

    try:
        last = datetime.fromisoformat(last_cycle)
        if last.tzinfo is None:
            last = last.replace(tzinfo=EST)

        now = datetime.now(EST)
        elapsed = (now - last).total_seconds()

        # Dashboard refreshes every 4 minutes normally
        next_in = max(0, 240 - elapsed)

        if next_in < 60:
            return f"~{int(next_in)}s"
        else:
            return f"~{int(next_in / 60)}m"
    except Exception:
        return "Unknown"
