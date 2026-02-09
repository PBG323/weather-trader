"""
Forecast Change Alerts

Monitors weather model updates and generates trading alerts when:
1. Forecasts shift significantly (opportunity to front-run market)
2. Model consensus changes (new agreement or disagreement)
3. Aviation observations conflict with forecasts (same-day edge)

Key insight from X post:
"Polymarket odds sometimes feel hours behind what the actual forecast already says.
That means someone with faster data could be buying YES at 10-15 cents when the
real probability is already way higher."

Model update schedule (UTC):
- GFS: 00z, 06z, 12z, 18z (every 6 hours)
- ECMWF: 00z, 12z (every 12 hours)
- HRRR: Every hour
- METAR: Every 1-3 hours (real observations)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
from zoneinfo import ZoneInfo
import json
import os

from ..models.ensemble import EnsembleForecast, ModelForecast


class AlertType(Enum):
    """Type of forecast alert."""
    MAJOR_SHIFT = "major_shift"  # Forecast moved 3°F+
    MODERATE_SHIFT = "moderate_shift"  # Forecast moved 2-3°F
    CONSENSUS_CHANGE = "consensus_change"  # Models now agree/disagree
    OBSERVATION_CONFLICT = "observation_conflict"  # METAR vs forecast mismatch
    MODEL_UPDATE = "model_update"  # New model run available
    EDGE_WINDOW = "edge_window"  # Time-sensitive opportunity


class AlertPriority(Enum):
    """Priority level for alerts."""
    CRITICAL = "critical"  # Act immediately
    HIGH = "high"  # Review within 15 minutes
    MEDIUM = "medium"  # Review within 1 hour
    LOW = "low"  # Informational


@dataclass
class ForecastAlert:
    """A forecast change alert."""
    alert_type: AlertType
    priority: AlertPriority
    city: str
    target_date: datetime
    message: str
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(ZoneInfo("UTC")))

    @property
    def age_minutes(self) -> int:
        """How old is this alert in minutes."""
        now = datetime.now(ZoneInfo("UTC"))
        return int((now - self.timestamp).total_seconds() / 60)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage/display."""
        return {
            "alert_type": self.alert_type.value,
            "priority": self.priority.value,
            "city": self.city,
            "target_date": self.target_date.isoformat() if isinstance(self.target_date, datetime) else str(self.target_date),
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "age_minutes": self.age_minutes,
        }


# Model update times (UTC hours)
MODEL_UPDATE_SCHEDULE = {
    "gfs": [0, 6, 12, 18],  # Every 6 hours
    "ecmwf": [0, 12],  # Every 12 hours
    "hrrr": list(range(24)),  # Every hour
    "nws": [6, 18],  # Twice daily
}

# How long after run time the data typically becomes available (minutes)
MODEL_AVAILABILITY_DELAY = {
    "gfs": 210,  # ~3.5 hours after run time
    "ecmwf": 360,  # ~6 hours after run time
    "hrrr": 60,  # ~1 hour after run time
    "nws": 30,  # ~30 minutes after update
}


class ForecastAlertMonitor:
    """
    Monitors forecast changes and generates alerts.

    Stores previous forecasts and compares against new ones
    to detect significant changes.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize alert monitor.

        Args:
            storage_path: Path to store previous forecasts (for persistence)
        """
        self.storage_path = storage_path or "weather_trader/data/forecast_history.json"
        self._previous_forecasts: dict[str, dict] = {}
        self._alerts: list[ForecastAlert] = []
        self._load_history()

    def _load_history(self):
        """Load previous forecast history from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self._previous_forecasts = json.load(f)
            except Exception as e:
                print(f"[ForecastAlerts] Failed to load history: {e}")
                self._previous_forecasts = {}

    def _save_history(self):
        """Save forecast history to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(self._previous_forecasts, f)
        except Exception as e:
            print(f"[ForecastAlerts] Failed to save history: {e}")

    def _get_forecast_key(self, city: str, target_date) -> str:
        """Generate unique key for a city/date forecast."""
        date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
        return f"{city.lower()}_{date_str}"

    def check_forecast_change(
        self,
        current: EnsembleForecast,
        major_shift_threshold: float = 3.0,
        moderate_shift_threshold: float = 2.0,
    ) -> list[ForecastAlert]:
        """
        Check if forecast has changed significantly since last check.

        Args:
            current: Current ensemble forecast
            major_shift_threshold: Degrees F change for major alert (default 3°F)
            moderate_shift_threshold: Degrees F change for moderate alert (default 2°F)

        Returns:
            List of alerts generated
        """
        alerts = []
        key = self._get_forecast_key(current.city, current.date)

        previous = self._previous_forecasts.get(key)

        if previous:
            prev_high = previous.get("high_mean", current.high_mean)
            prev_std = previous.get("high_std", current.high_std)
            prev_consensus = previous.get("consensus_ratio", current.consensus_ratio)

            # Check for temperature shift
            high_change = current.high_mean - prev_high

            if abs(high_change) >= major_shift_threshold:
                direction = "WARMER" if high_change > 0 else "COLDER"
                alerts.append(ForecastAlert(
                    alert_type=AlertType.MAJOR_SHIFT,
                    priority=AlertPriority.CRITICAL,
                    city=current.city,
                    target_date=current.date,
                    message=f"MAJOR SHIFT: {current.city} forecast moved {abs(high_change):.1f}°F {direction}",
                    details={
                        "previous_high": prev_high,
                        "current_high": current.high_mean,
                        "change": high_change,
                        "direction": direction,
                        "action": f"Review {current.city} markets - prices may not have adjusted yet",
                    },
                ))
            elif abs(high_change) >= moderate_shift_threshold:
                direction = "warmer" if high_change > 0 else "colder"
                alerts.append(ForecastAlert(
                    alert_type=AlertType.MODERATE_SHIFT,
                    priority=AlertPriority.HIGH,
                    city=current.city,
                    target_date=current.date,
                    message=f"Forecast shift: {current.city} moved {abs(high_change):.1f}°F {direction}",
                    details={
                        "previous_high": prev_high,
                        "current_high": current.high_mean,
                        "change": high_change,
                    },
                ))

            # Check for consensus change
            consensus_change = current.consensus_ratio - prev_consensus

            if abs(consensus_change) >= 0.2:  # 20% consensus shift
                if current.consensus_ratio > prev_consensus:
                    alerts.append(ForecastAlert(
                        alert_type=AlertType.CONSENSUS_CHANGE,
                        priority=AlertPriority.HIGH,
                        city=current.city,
                        target_date=current.date,
                        message=f"Models now AGREE: {current.city} consensus up to {current.consensus_ratio:.0%}",
                        details={
                            "previous_consensus": prev_consensus,
                            "current_consensus": current.consensus_ratio,
                            "action": "Confidence increased - edge opportunities may be stronger",
                        },
                    ))
                else:
                    alerts.append(ForecastAlert(
                        alert_type=AlertType.CONSENSUS_CHANGE,
                        priority=AlertPriority.MEDIUM,
                        city=current.city,
                        target_date=current.date,
                        message=f"Models now DISAGREE: {current.city} consensus down to {current.consensus_ratio:.0%}",
                        details={
                            "previous_consensus": prev_consensus,
                            "current_consensus": current.consensus_ratio,
                            "action": "Reduce position sizing due to uncertainty",
                        },
                    ))

        # Store current forecast for future comparison
        self._previous_forecasts[key] = {
            "high_mean": current.high_mean,
            "high_std": current.high_std,
            "consensus_ratio": current.consensus_ratio,
            "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
        }
        self._save_history()

        # Add to alert history
        self._alerts.extend(alerts)

        return alerts

    def check_observation_conflict(
        self,
        city: str,
        observed_temp: float,
        forecast: EnsembleForecast,
        conflict_threshold: float = 3.0,
    ) -> Optional[ForecastAlert]:
        """
        Check if current observation conflicts with forecast.

        This is the "pilot edge" - when real observations don't match forecasts.

        Args:
            city: City key
            observed_temp: Current observed temperature (F)
            forecast: Current forecast
            conflict_threshold: Degrees F difference to flag as conflict

        Returns:
            ForecastAlert if conflict detected, None otherwise
        """
        # For same-day forecasts, compare observation to forecast high
        diff = observed_temp - forecast.high_mean

        if abs(diff) >= conflict_threshold:
            direction = "above" if diff > 0 else "below"

            return ForecastAlert(
                alert_type=AlertType.OBSERVATION_CONFLICT,
                priority=AlertPriority.CRITICAL,
                city=city,
                target_date=forecast.date,
                message=f"OBSERVATION CONFLICT: {city} currently {observed_temp:.0f}°F, {abs(diff):.0f}°F {direction} forecast",
                details={
                    "observed_temp": observed_temp,
                    "forecast_high": forecast.high_mean,
                    "difference": diff,
                    "action": f"Current temp is {abs(diff):.0f}°F {direction} forecast high - check if market has priced this in",
                },
            )

        # Also alert if current temp already exceeds forecast high
        if observed_temp >= forecast.high_mean:
            return ForecastAlert(
                alert_type=AlertType.OBSERVATION_CONFLICT,
                priority=AlertPriority.HIGH,
                city=city,
                target_date=forecast.date,
                message=f"HIGH EXCEEDED: {city} already at {observed_temp:.0f}°F (forecast high: {forecast.high_mean:.0f}°F)",
                details={
                    "observed_temp": observed_temp,
                    "forecast_high": forecast.high_mean,
                    "action": "Temperature brackets above current observation have increased probability",
                },
            )

        return None

    def get_model_update_window(self) -> dict:
        """
        Check if any models have recently updated or are about to update.

        Returns:
            Dict with model update status and recommended actions
        """
        now = datetime.now(ZoneInfo("UTC"))
        current_hour = now.hour

        updates = {}

        for model, run_hours in MODEL_UPDATE_SCHEDULE.items():
            delay = MODEL_AVAILABILITY_DELAY.get(model, 120)

            # Find the most recent run
            recent_runs = [h for h in run_hours if h <= current_hour]
            if not recent_runs:
                recent_runs = [run_hours[-1]]  # Yesterday's last run

            last_run_hour = max(recent_runs)
            availability_hour = last_run_hour + delay // 60
            availability_minute = delay % 60

            # Time since data became available
            if availability_hour <= current_hour:
                minutes_since_available = (current_hour - availability_hour) * 60 + now.minute - availability_minute
            else:
                minutes_since_available = -((availability_hour - current_hour) * 60 - now.minute + availability_minute)

            # Find next run
            future_runs = [h for h in run_hours if h > current_hour]
            next_run_hour = min(future_runs) if future_runs else run_hours[0]  # Tomorrow's first run
            hours_until_next = next_run_hour - current_hour if next_run_hour > current_hour else (24 - current_hour + next_run_hour)

            updates[model] = {
                "last_run_utc": f"{last_run_hour:02d}:00",
                "minutes_since_available": minutes_since_available,
                "is_fresh": 0 <= minutes_since_available <= 60,  # Less than 1 hour old
                "next_run_utc": f"{next_run_hour:02d}:00",
                "hours_until_next": hours_until_next,
            }

        # Generate recommendations
        fresh_models = [m for m, u in updates.items() if u["is_fresh"]]
        if fresh_models:
            updates["_recommendation"] = f"Fresh data from: {', '.join(fresh_models)}. Good time to check for mispricing."
        else:
            upcoming = min(updates.items(), key=lambda x: x[1].get("hours_until_next", 99) if x[0] != "_recommendation" else 99)
            if upcoming[0] != "_recommendation":
                updates["_recommendation"] = f"Next update: {upcoming[0].upper()} in {upcoming[1]['hours_until_next']}h"

        return updates

    def get_recent_alerts(self, max_age_minutes: int = 60) -> list[ForecastAlert]:
        """Get alerts from the last N minutes."""
        return [a for a in self._alerts if a.age_minutes <= max_age_minutes]

    def get_critical_alerts(self) -> list[ForecastAlert]:
        """Get all critical priority alerts."""
        return [a for a in self._alerts if a.priority == AlertPriority.CRITICAL]

    def clear_old_alerts(self, max_age_minutes: int = 120):
        """Remove alerts older than N minutes."""
        self._alerts = [a for a in self._alerts if a.age_minutes <= max_age_minutes]


def check_edge_window(
    forecast: EnsembleForecast,
    market_price: float,
    model_just_updated: bool = False,
) -> Optional[ForecastAlert]:
    """
    Check if there's a time-sensitive edge window.

    Edge windows occur when:
    1. A model just updated with new data
    2. Market price hasn't adjusted yet
    3. Our calculated edge is significant

    Args:
        forecast: Current forecast
        market_price: Current market price for a bracket
        model_just_updated: Whether a key model just updated

    Returns:
        ForecastAlert if edge window detected
    """
    if not model_just_updated:
        return None

    # Calculate basic edge (simplified - actual would use specific bracket)
    our_prob = 0.5  # Placeholder - would calculate for specific bracket
    edge = our_prob - market_price

    if abs(edge) >= 0.10:  # 10%+ edge
        return ForecastAlert(
            alert_type=AlertType.EDGE_WINDOW,
            priority=AlertPriority.CRITICAL,
            city=forecast.city,
            target_date=forecast.date,
            message=f"EDGE WINDOW: Model updated, {abs(edge)*100:.0f}% edge detected",
            details={
                "edge": edge,
                "market_price": market_price,
                "our_probability": our_prob,
                "action": "Market may not have priced in new model data. Act quickly.",
            },
        )

    return None
