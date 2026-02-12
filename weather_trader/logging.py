"""
Forecast logging for accuracy tracking.

Logs forecasts to enable historical analysis of model accuracy.
"""

from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional
import json
from pathlib import Path


@dataclass
class ForecastRecord:
    """Record of a forecast for accuracy tracking."""
    forecast_made_date: date
    target_date: date
    city: str
    station_id: str
    ecmwf_high: Optional[float] = None
    gfs_high: Optional[float] = None
    tomorrow_io_high: Optional[float] = None
    nws_forecast_high: Optional[float] = None
    ensemble_high_mean: float = 0.0
    ensemble_high_std: float = 0.0
    ensemble_confidence: float = 0.0
    station_bias_applied: float = 0.0
    model_count: int = 0
    actual_high: Optional[float] = None  # Filled in after settlement

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['forecast_made_date'] = str(self.forecast_made_date)
        d['target_date'] = str(self.target_date)
        return d


class ForecastLogger:
    """Simple file-based forecast logger."""

    def __init__(self, log_dir: str = "forecast_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def log_forecast(self, record: ForecastRecord) -> None:
        """Log a forecast record to JSON file."""
        log_file = self.log_dir / f"forecasts_{record.forecast_made_date}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')

    def get_forecasts_for_date(self, target_date: date) -> list[ForecastRecord]:
        """Retrieve all forecasts for a specific target date."""
        records = []

        for log_file in self.log_dir.glob("forecasts_*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('target_date') == str(target_date):
                        records.append(ForecastRecord(**{
                            k: v for k, v in data.items()
                            if k in ForecastRecord.__dataclass_fields__
                        }))

        return records
