"""
Bias Correction Models

Weather APIs forecast for grid points, not specific weather stations.
Polymarket settles on specific NWS station readings, creating systematic bias.

This module trains station-specific bias correction models to adjust
API forecasts to match expected station readings.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle

from ..config import CityConfig, CITY_CONFIGS


@dataclass
class BiasModel:
    """Trained bias correction model for a specific station."""
    station_id: str
    city: str
    model_type: str  # 'linear' or 'gbm'
    high_model: object  # Trained model for daily high
    low_model: object   # Trained model for daily low
    high_scaler: StandardScaler
    low_scaler: StandardScaler
    training_samples: int
    training_date: datetime
    mae_high: float  # Mean absolute error on validation
    mae_low: float
    features: list[str]

    def to_dict(self) -> dict:
        """Serialize model metadata (not the model itself)."""
        return {
            "station_id": self.station_id,
            "city": self.city,
            "model_type": self.model_type,
            "training_samples": self.training_samples,
            "training_date": self.training_date.isoformat(),
            "mae_high": self.mae_high,
            "mae_low": self.mae_low,
            "features": self.features,
        }


@dataclass
class CorrectedForecast:
    """A forecast after bias correction."""
    date: date
    original_high: float
    original_low: float
    corrected_high: float
    corrected_low: float
    correction_high: float  # corrected - original
    correction_low: float
    confidence: float  # Based on model accuracy


class BiasCorrector:
    """
    Trains and applies bias correction models for weather forecasts.

    The key insight is that weather APIs forecast for grid points,
    while Polymarket settles on specific station readings. The difference
    is systematic and predictable.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("weather_trader/data")
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_models: dict[str, BiasModel] = {}

    def _extract_features(
        self,
        forecast_high: float,
        forecast_low: float,
        forecast_date: date,
        model_source: str
    ) -> dict:
        """
        Extract features for bias correction model.

        Features capture:
        - The forecast values themselves
        - Seasonal patterns
        - Day of week effects
        - Model-specific biases
        """
        # Day of year for seasonality (circular encoding)
        day_of_year = forecast_date.timetuple().tm_yday
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)

        # Temperature features
        temp_range = forecast_high - forecast_low
        temp_avg = (forecast_high + forecast_low) / 2

        return {
            "forecast_high": forecast_high,
            "forecast_low": forecast_low,
            "temp_range": temp_range,
            "temp_avg": temp_avg,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "day_of_week": forecast_date.weekday(),
            "month": forecast_date.month,
            "is_ecmwf": 1 if model_source == "ecmwf" else 0,
            "is_gfs": 1 if model_source == "gfs" else 0,
            "is_tomorrow": 1 if model_source == "tomorrow" else 0,
        }

    def train_model(
        self,
        city_config: CityConfig,
        training_data: pd.DataFrame,
        model_type: str = "gbm"
    ) -> BiasModel:
        """
        Train a bias correction model for a specific station.

        Args:
            city_config: City configuration with station details
            training_data: DataFrame with columns:
                - date: Forecast date
                - forecast_high: Predicted high temperature
                - forecast_low: Predicted low temperature
                - actual_high: Actual station high
                - actual_low: Actual station low
                - model_source: Which weather model made the forecast
            model_type: 'linear' for Ridge regression, 'gbm' for Gradient Boosting

        Returns:
            Trained BiasModel
        """
        if len(training_data) < 30:
            raise ValueError(f"Need at least 30 samples for training, got {len(training_data)}")

        # Extract features
        feature_records = []
        for _, row in training_data.iterrows():
            features = self._extract_features(
                row["forecast_high"],
                row["forecast_low"],
                row["date"],
                row.get("model_source", "unknown")
            )
            feature_records.append(features)

        X = pd.DataFrame(feature_records)
        feature_names = list(X.columns)

        # Target is the actual station readings
        y_high = training_data["actual_high"].values
        y_low = training_data["actual_low"].values

        # Scale features
        scaler_high = StandardScaler()
        scaler_low = StandardScaler()
        X_scaled_high = scaler_high.fit_transform(X)
        X_scaled_low = scaler_low.fit_transform(X)

        # Train models
        if model_type == "linear":
            model_high = Ridge(alpha=1.0)
            model_low = Ridge(alpha=1.0)
        else:
            model_high = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model_low = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )

        model_high.fit(X_scaled_high, y_high)
        model_low.fit(X_scaled_low, y_low)

        # Evaluate with cross-validation
        cv_scores_high = cross_val_score(
            model_high, X_scaled_high, y_high,
            cv=5, scoring='neg_mean_absolute_error'
        )
        cv_scores_low = cross_val_score(
            model_low, X_scaled_low, y_low,
            cv=5, scoring='neg_mean_absolute_error'
        )

        mae_high = -cv_scores_high.mean()
        mae_low = -cv_scores_low.mean()

        bias_model = BiasModel(
            station_id=city_config.nws_station_id,
            city=city_config.name,
            model_type=model_type,
            high_model=model_high,
            low_model=model_low,
            high_scaler=scaler_high,
            low_scaler=scaler_low,
            training_samples=len(training_data),
            training_date=datetime.now(),
            mae_high=mae_high,
            mae_low=mae_low,
            features=feature_names,
        )

        # Cache the model
        self.loaded_models[city_config.nws_station_id] = bias_model

        return bias_model

    def save_model(self, model: BiasModel) -> str:
        """
        Save a trained model to disk.

        Returns:
            Path to saved model file
        """
        model_path = self.models_dir / f"{model.station_id}_bias.pkl"
        metadata_path = self.models_dir / f"{model.station_id}_bias_meta.json"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        with open(metadata_path, "w") as f:
            json.dump(model.to_dict(), f, indent=2)

        return str(model_path)

    def load_model(self, station_id: str) -> Optional[BiasModel]:
        """
        Load a trained model from disk.

        Args:
            station_id: NWS station ID

        Returns:
            BiasModel or None if not found
        """
        if station_id in self.loaded_models:
            return self.loaded_models[station_id]

        model_path = self.models_dir / f"{station_id}_bias.pkl"

        if not model_path.exists():
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        self.loaded_models[station_id] = model
        return model

    def correct_forecast(
        self,
        city_config: CityConfig,
        forecast_high: float,
        forecast_low: float,
        forecast_date: date,
        model_source: str = "unknown"
    ) -> CorrectedForecast:
        """
        Apply bias correction to a forecast.

        Args:
            city_config: City configuration
            forecast_high: Uncorrected forecast high
            forecast_low: Uncorrected forecast low
            forecast_date: Date of forecast
            model_source: Which weather model produced the forecast

        Returns:
            CorrectedForecast with adjusted values
        """
        bias_model = self.load_model(city_config.nws_station_id)

        if bias_model is None:
            # No model trained - return uncorrected with low confidence
            return CorrectedForecast(
                date=forecast_date,
                original_high=forecast_high,
                original_low=forecast_low,
                corrected_high=forecast_high,
                corrected_low=forecast_low,
                correction_high=0,
                correction_low=0,
                confidence=0.5,
            )

        # Extract features
        features = self._extract_features(
            forecast_high, forecast_low, forecast_date, model_source
        )
        X = pd.DataFrame([features])[bias_model.features]

        # Scale and predict
        X_high = bias_model.high_scaler.transform(X)
        X_low = bias_model.low_scaler.transform(X)

        corrected_high = float(bias_model.high_model.predict(X_high)[0])
        corrected_low = float(bias_model.low_model.predict(X_low)[0])

        # Ensure high >= low
        if corrected_high < corrected_low:
            corrected_high, corrected_low = corrected_low, corrected_high

        # Calculate confidence based on model accuracy
        # Lower MAE = higher confidence
        avg_mae = (bias_model.mae_high + bias_model.mae_low) / 2
        confidence = max(0.5, min(0.95, 1 - avg_mae / 10))

        return CorrectedForecast(
            date=forecast_date,
            original_high=forecast_high,
            original_low=forecast_low,
            corrected_high=corrected_high,
            corrected_low=corrected_low,
            correction_high=corrected_high - forecast_high,
            correction_low=corrected_low - forecast_low,
            confidence=confidence,
        )

    def get_model_stats(self, station_id: str) -> Optional[dict]:
        """Get statistics about a trained model."""
        model = self.load_model(station_id)
        if model is None:
            return None

        return model.to_dict()


def create_training_dataset(
    forecasts: list[dict],
    actuals: list[dict]
) -> pd.DataFrame:
    """
    Create a training dataset from forecast and actual data.

    Args:
        forecasts: List of dicts with date, forecast_high, forecast_low, model_source
        actuals: List of dicts with date, actual_high, actual_low

    Returns:
        DataFrame ready for training
    """
    # Convert to DataFrames
    df_forecasts = pd.DataFrame(forecasts)
    df_actuals = pd.DataFrame(actuals)

    # Merge on date
    df = df_forecasts.merge(df_actuals, on="date", how="inner")

    return df
