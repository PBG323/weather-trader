"""
Tests for forecast models (bias correction and ensemble).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

from weather_trader.config import get_city_config
from weather_trader.models.bias_correction import BiasCorrector, create_training_dataset
from weather_trader.models.ensemble import (
    EnsembleForecaster, ModelForecast, EnsembleForecast, DEFAULT_WEIGHTS
)


class TestBiasCorrector:
    """Tests for bias correction model."""

    @pytest.fixture
    def corrector(self, tmp_path):
        return BiasCorrector(data_dir=str(tmp_path))

    @pytest.fixture
    def training_data(self):
        """Generate synthetic training data."""
        np.random.seed(42)
        n_samples = 100

        # Simulate forecasts with systematic bias
        forecast_highs = np.random.normal(50, 10, n_samples)
        forecast_lows = forecast_highs - np.random.uniform(10, 20, n_samples)

        # Actual temps have a bias (station reads slightly higher)
        actual_highs = forecast_highs + 2 + np.random.normal(0, 1, n_samples)
        actual_lows = forecast_lows + 1 + np.random.normal(0, 1, n_samples)

        dates = [date(2024, 1, 1) + pd.Timedelta(days=i) for i in range(n_samples)]

        return pd.DataFrame({
            "date": dates,
            "forecast_high": forecast_highs,
            "forecast_low": forecast_lows,
            "actual_high": actual_highs,
            "actual_low": actual_lows,
            "model_source": ["ecmwf"] * n_samples,
        })

    def test_train_model(self, corrector, training_data):
        """Test training a bias correction model."""
        nyc_config = get_city_config("nyc")

        model = corrector.train_model(nyc_config, training_data, model_type="linear")

        assert model is not None
        assert model.station_id == "KNYC"
        assert model.training_samples == len(training_data)
        assert model.mae_high > 0
        assert model.mae_low > 0

    def test_correct_forecast(self, corrector, training_data):
        """Test applying bias correction to a forecast."""
        nyc_config = get_city_config("nyc")

        # Train model first
        model = corrector.train_model(nyc_config, training_data, model_type="linear")

        # Correct a forecast
        result = corrector.correct_forecast(
            nyc_config,
            forecast_high=50.0,
            forecast_low=35.0,
            forecast_date=date(2024, 6, 15),
            model_source="ecmwf"
        )

        assert result is not None
        assert result.original_high == 50.0
        assert result.original_low == 35.0
        # Should apply positive correction (station reads higher)
        assert result.corrected_high > result.original_high - 5  # Allow some variance
        assert result.confidence > 0.5

    def test_no_model_returns_uncorrected(self, corrector):
        """Test that missing model returns uncorrected forecast."""
        seattle_config = get_city_config("seattle")  # No model trained

        result = corrector.correct_forecast(
            seattle_config,
            forecast_high=60.0,
            forecast_low=45.0,
            forecast_date=date(2024, 6, 15),
        )

        assert result.corrected_high == result.original_high
        assert result.corrected_low == result.original_low
        assert result.confidence == 0.5  # Low confidence without model

    def test_save_and_load_model(self, corrector, training_data, tmp_path):
        """Test model persistence."""
        nyc_config = get_city_config("nyc")

        # Train and save
        model = corrector.train_model(nyc_config, training_data)
        path = corrector.save_model(model)

        # Create new corrector and load
        new_corrector = BiasCorrector(data_dir=str(tmp_path))
        loaded = new_corrector.load_model("KNYC")

        assert loaded is not None
        assert loaded.station_id == model.station_id
        assert loaded.mae_high == model.mae_high


class TestEnsembleForecaster:
    """Tests for ensemble forecast model."""

    @pytest.fixture
    def forecaster(self):
        return EnsembleForecaster(bias_corrector=None)

    @pytest.fixture
    def sample_forecasts(self):
        return [
            ModelForecast(model_name="ecmwf", forecast_high=52.0, forecast_low=38.0),
            ModelForecast(model_name="gfs", forecast_high=54.0, forecast_low=40.0),
            ModelForecast(model_name="tomorrow", forecast_high=53.0, forecast_low=39.0),
        ]

    def test_create_ensemble(self, forecaster, sample_forecasts):
        """Test creating an ensemble forecast."""
        nyc_config = get_city_config("nyc")
        target = date(2024, 1, 15)

        ensemble = forecaster.create_ensemble(
            nyc_config,
            sample_forecasts,
            target,
            apply_bias_correction=False
        )

        assert ensemble is not None
        assert ensemble.date == target
        assert ensemble.city == "New York City"
        assert ensemble.model_count == 3

        # Check weighted average is reasonable
        assert 52 < ensemble.high_mean < 54
        assert 38 < ensemble.low_mean < 40

        # Check uncertainty measures exist
        assert ensemble.high_std > 0
        assert ensemble.low_std > 0
        assert ensemble.confidence > 0

    def test_probability_calculations(self, forecaster, sample_forecasts):
        """Test probability calculation methods."""
        nyc_config = get_city_config("nyc")
        target = date(2024, 1, 15)

        ensemble = forecaster.create_ensemble(
            nyc_config,
            sample_forecasts,
            target,
            apply_bias_correction=False
        )

        # Test probability above threshold
        prob_above_50 = ensemble.get_probability_above(50.0, for_high=True)
        prob_above_60 = ensemble.get_probability_above(60.0, for_high=True)

        assert 0 < prob_above_50 < 1
        assert prob_above_50 > prob_above_60  # Higher threshold = lower probability

        # Test probability below threshold
        prob_below_50 = ensemble.get_probability_below(50.0, for_high=True)
        assert abs(prob_above_50 + prob_below_50 - 1.0) < 0.01  # Should sum to 1

        # Test range probability
        prob_in_range = ensemble.get_probability_in_range(50.0, 55.0, for_high=True)
        assert 0 < prob_in_range < 1

    def test_model_weights_applied(self, forecaster):
        """Test that model weights affect ensemble."""
        nyc_config = get_city_config("nyc")
        target = date(2024, 1, 15)

        # ECMWF has higher weight than GFS
        forecasts = [
            ModelForecast(model_name="ecmwf", forecast_high=50.0, forecast_low=35.0),
            ModelForecast(model_name="gfs", forecast_high=60.0, forecast_low=45.0),
        ]

        ensemble = forecaster.create_ensemble(
            nyc_config,
            forecasts,
            target,
            apply_bias_correction=False
        )

        # Result should be closer to ECMWF due to higher weight
        ecmwf_weight = DEFAULT_WEIGHTS["ecmwf"]
        gfs_weight = DEFAULT_WEIGHTS["gfs"]
        expected = (50 * ecmwf_weight + 60 * gfs_weight) / (ecmwf_weight + gfs_weight)

        assert abs(ensemble.high_mean - expected) < 1.0

    def test_forecast_for_market(self, forecaster, sample_forecasts):
        """Test market-specific probability generation."""
        nyc_config = get_city_config("nyc")
        target = date(2024, 1, 15)

        ensemble = forecaster.create_ensemble(
            nyc_config,
            sample_forecasts,
            target,
            apply_bias_correction=False
        )

        # Test over market
        result = forecaster.forecast_for_market(
            ensemble,
            market_threshold=50.0,
            is_over_market=True,
            for_high=True
        )

        assert "probability" in result
        assert 0 < result["probability"] < 1
        assert result["threshold"] == 50.0
        assert result["is_over"] is True

    def test_calculate_edge(self, forecaster, sample_forecasts):
        """Test edge calculation vs market price."""
        nyc_config = get_city_config("nyc")
        target = date(2024, 1, 15)

        ensemble = forecaster.create_ensemble(
            nyc_config,
            sample_forecasts,
            target,
            apply_bias_correction=False
        )

        # Forecast high is ~53, so "over 50" should have high probability
        # If market prices it at 0.5, we should have positive edge
        result = forecaster.calculate_edge(
            ensemble,
            market_threshold=50.0,
            market_price=0.5,
            is_over_market=True,
            for_high=True
        )

        assert "edge" in result
        assert result["edge"] > 0  # Our forecast > market price
        assert result["direction"] == "BUY_YES"

        # Test opposite scenario
        result_under = forecaster.calculate_edge(
            ensemble,
            market_threshold=60.0,
            market_price=0.5,
            is_over_market=True,
            for_high=True
        )

        # Forecast is ~53, so "over 60" should have low probability
        # Market at 0.5 means we should buy NO
        assert result_under["edge"] < 0
        assert result_under["direction"] == "BUY_NO"


class TestCreateTrainingDataset:
    """Tests for training dataset creation."""

    def test_merge_forecasts_and_actuals(self):
        """Test merging forecast and actual data."""
        forecasts = [
            {"date": date(2024, 1, 1), "forecast_high": 50, "forecast_low": 35, "model_source": "ecmwf"},
            {"date": date(2024, 1, 2), "forecast_high": 52, "forecast_low": 37, "model_source": "ecmwf"},
        ]

        actuals = [
            {"date": date(2024, 1, 1), "actual_high": 51, "actual_low": 36},
            {"date": date(2024, 1, 2), "actual_high": 53, "actual_low": 38},
        ]

        df = create_training_dataset(forecasts, actuals)

        assert len(df) == 2
        assert "forecast_high" in df.columns
        assert "actual_high" in df.columns
        assert df.iloc[0]["forecast_high"] == 50
        assert df.iloc[0]["actual_high"] == 51
