"""
Tests for trading strategy components.
"""

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, AsyncMock

from weather_trader.config import get_city_config
from weather_trader.models.ensemble import EnsembleForecast, ModelForecast
from weather_trader.polymarket.markets import WeatherMarket, MarketType
from weather_trader.strategy.expected_value import (
    ExpectedValueCalculator, TradeSignal, SignalStrength
)
from weather_trader.strategy.position_sizing import (
    PositionSizer, PositionRecommendation
)
from weather_trader.strategy.executor import TradeExecutor, ExecutionResult


class TestExpectedValueCalculator:
    """Tests for expected value calculation."""

    @pytest.fixture
    def calculator(self):
        return ExpectedValueCalculator(
            min_edge=0.05,
            min_confidence=0.70,
            strong_edge=0.15
        )

    @pytest.fixture
    def sample_market(self):
        return WeatherMarket(
            condition_id="test_123",
            question_id="q_123",
            market_slug="nyc-temp-jan-15",
            city="nyc",
            city_config=get_city_config("nyc"),
            target_date=date(2024, 1, 15),
            threshold=50.0,
            market_type=MarketType.HIGH_OVER,
            yes_token_id="yes_token",
            no_token_id="no_token",
            yes_price=0.50,
            no_price=0.50,
            volume=10000,
            liquidity=5000,
            is_active=True,
            end_date=datetime(2024, 1, 16),
            question="Will NYC high temperature be over 50°F on Jan 15?"
        )

    @pytest.fixture
    def sample_forecast(self):
        return EnsembleForecast(
            date=date(2024, 1, 15),
            city="New York City",
            high_mean=55.0,  # Well above 50, should be bullish
            high_median=55.0,
            low_mean=40.0,
            low_median=40.0,
            high_std=3.0,
            low_std=3.0,
            high_ci_lower=50.0,
            high_ci_upper=60.0,
            low_ci_lower=35.0,
            low_ci_upper=45.0,
            model_count=3,
            model_forecasts=[],
            confidence=0.85
        )

    def test_calculate_positive_edge(self, calculator, sample_market, sample_forecast):
        """Test calculating positive edge (forecast > market)."""
        signal = calculator.calculate_ev(sample_market, sample_forecast)

        # Forecast of 55 should give high probability of "over 50"
        assert signal.forecast_probability > 0.7
        assert signal.edge > 0  # Positive edge vs 0.50 market price
        assert signal.side == "YES"
        assert signal.is_tradeable

    def test_calculate_negative_edge(self, calculator, sample_market, sample_forecast):
        """Test calculating negative edge (forecast < market)."""
        # Adjust market price to be higher than our forecast probability
        sample_market.yes_price = 0.95

        signal = calculator.calculate_ev(sample_market, sample_forecast)

        assert signal.edge < 0  # Negative edge
        assert signal.side == "NO"

    def test_neutral_signal_low_edge(self, calculator, sample_market, sample_forecast):
        """Test that small edge results in neutral signal."""
        # Set market close to forecast probability
        sample_market.yes_price = 0.82  # Close to actual probability

        signal = calculator.calculate_ev(sample_market, sample_forecast)

        # Edge should be small
        assert abs(signal.edge) < 0.05
        assert signal.signal == SignalStrength.NEUTRAL
        assert not signal.is_tradeable

    def test_neutral_signal_low_confidence(self, calculator, sample_market, sample_forecast):
        """Test that low confidence results in neutral signal."""
        sample_forecast.confidence = 0.5  # Below threshold

        signal = calculator.calculate_ev(sample_market, sample_forecast)

        assert signal.signal == SignalStrength.NEUTRAL

    def test_strong_signal_detection(self, calculator, sample_market, sample_forecast):
        """Test detection of strong signals."""
        # Set market way off from forecast
        sample_market.yes_price = 0.30  # Market says 30%, we say ~85%

        signal = calculator.calculate_ev(sample_market, sample_forecast)

        assert signal.signal == SignalStrength.STRONG_BUY_YES
        assert signal.edge > 0.15

    def test_analyze_multiple_markets(self, calculator, sample_market, sample_forecast):
        """Test analyzing multiple markets at once."""
        markets = [sample_market]
        forecasts = {"nyc": sample_forecast}

        signals = calculator.analyze_markets(markets, forecasts)

        assert len(signals) == 1
        assert signals[0].market == sample_market


class TestPositionSizer:
    """Tests for position sizing."""

    @pytest.fixture
    def sizer(self):
        return PositionSizer(
            bankroll=10000.0,
            kelly_multiplier=0.25,
            max_position_percent=5.0,
            daily_loss_limit_percent=10.0
        )

    @pytest.fixture
    def sample_signal(self):
        market = WeatherMarket(
            condition_id="test_123",
            question_id="q_123",
            market_slug="nyc-temp",
            city="nyc",
            city_config=get_city_config("nyc"),
            target_date=date(2024, 1, 15),
            threshold=50.0,
            market_type=MarketType.HIGH_OVER,
            yes_token_id="yes_token",
            no_token_id="no_token",
            yes_price=0.50,
            no_price=0.50,
            volume=10000,
            liquidity=5000,
            is_active=True,
            end_date=datetime(2024, 1, 16),
            question="Test question"
        )

        return TradeSignal(
            market=market,
            forecast=MagicMock(confidence=0.85),
            forecast_probability=0.70,  # 70% forecast
            market_probability=0.50,    # 50% market
            edge=0.20,                  # 20% edge
            expected_value=0.20,
            signal=SignalStrength.STRONG_BUY_YES,
            side="YES",
            confidence=0.85,
            model_agreement=0.90,
            threshold=50.0,
            is_high_temp=True,
            is_over_market=True,
        )

    def test_kelly_calculation(self, sizer):
        """Test Kelly criterion calculation."""
        # 60% win probability, 1:1 odds
        kelly = sizer.calculate_kelly(win_prob=0.6, win_amount=1.0, lose_amount=1.0)

        # Kelly = (p * b - q) / b = (0.6 * 1 - 0.4) / 1 = 0.2
        assert abs(kelly - 0.2) < 0.01

    def test_kelly_with_edge(self, sizer):
        """Test Kelly with different odds."""
        # 55% probability with 2:1 payoff
        kelly = sizer.calculate_kelly(win_prob=0.55, win_amount=2.0, lose_amount=1.0)

        # Kelly = (0.55 * 2 - 0.45) / 2 = 0.325
        assert abs(kelly - 0.325) < 0.01

    def test_position_sizing(self, sizer, sample_signal):
        """Test position sizing for a signal."""
        rec = sizer.size_position(sample_signal)

        assert rec is not None
        assert rec.kelly_fraction > 0
        assert rec.adjusted_kelly < rec.kelly_fraction  # Fractional Kelly
        assert rec.recommended_size > 0
        assert rec.recommended_size <= 500  # Max 5% of $10k

    def test_position_capping(self, sizer, sample_signal):
        """Test that positions are capped at max size."""
        # Create signal with very high edge to trigger capping
        sample_signal.edge = 0.50
        sample_signal.forecast_probability = 0.90
        sample_signal.expected_value = 0.50

        rec = sizer.size_position(sample_signal)

        # Should be capped at 5% of bankroll
        assert rec.recommended_size <= 500
        assert rec.position_capped

    def test_daily_loss_limit(self, sizer, sample_signal):
        """Test daily loss limit enforcement."""
        # Simulate losses
        sizer.daily_pnl = -800  # $800 loss, limit is $1000

        rec = sizer.size_position(sample_signal)

        # Should have reduced size due to approaching limit
        assert rec.recommended_size <= 200  # Only $200 risk budget left

    def test_cannot_trade_at_limit(self, sizer):
        """Test that trading stops at loss limit."""
        sizer.daily_pnl = -1000  # At limit

        can_trade, reason = sizer.can_trade()

        assert not can_trade
        assert "loss limit" in reason.lower()

    def test_batch_recommendations(self, sizer, sample_signal):
        """Test batch position recommendations."""
        signals = [sample_signal] * 5

        recs = sizer.recommend_batch(signals, max_positions=3)

        assert len(recs) <= 3
        # Total allocated should not exceed bankroll
        total_allocated = sum(r.recommended_size for r in recs)
        assert total_allocated <= 10000


class TestTradeExecutor:
    """Tests for trade execution."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.get_market_price = AsyncMock(return_value={
            "yes_bid": 0.49,
            "yes_ask": 0.51,
            "yes_mid": 0.50,
            "no_bid": 0.49,
            "no_ask": 0.51,
            "spread": 0.02,
        })
        client.place_limit_order = AsyncMock(return_value=MagicMock(
            success=True,
            order_id="test_order_123",
            filled_size=100,
            filled_price=0.50,
            message="Filled",
            timestamp=datetime.now(),
        ))
        return client

    @pytest.fixture
    def executor(self, mock_client):
        return TradeExecutor(
            client=mock_client,
            max_slippage=0.02,
            dry_run=True
        )

    @pytest.fixture
    def sample_recommendation(self):
        market = WeatherMarket(
            condition_id="test_123",
            question_id="q_123",
            market_slug="nyc-temp",
            city="nyc",
            city_config=get_city_config("nyc"),
            target_date=date(2024, 1, 15),
            threshold=50.0,
            market_type=MarketType.HIGH_OVER,
            yes_token_id="yes_token",
            no_token_id="no_token",
            yes_price=0.50,
            no_price=0.50,
            volume=10000,
            liquidity=5000,
            is_active=True,
            end_date=datetime(2024, 1, 16),
            question="Test"
        )

        signal = TradeSignal(
            market=market,
            forecast=MagicMock(),
            forecast_probability=0.70,
            market_probability=0.50,
            edge=0.20,
            expected_value=0.20,
            signal=SignalStrength.STRONG_BUY_YES,
            side="YES",
            confidence=0.85,
            model_agreement=0.90,
            threshold=50.0,
            is_high_temp=True,
            is_over_market=True,
        )

        return PositionRecommendation(
            signal=signal,
            kelly_fraction=0.20,
            adjusted_kelly=0.05,
            recommended_size=100.0,
            recommended_shares=200.0,
            max_loss=100.0,
            expected_profit=20.0,
            risk_reward_ratio=0.20,
            position_capped=False,
            bankroll_limited=False,
        )

    @pytest.mark.asyncio
    async def test_execute_trade(self, executor, sample_recommendation):
        """Test executing a single trade."""
        result = await executor.execute(sample_recommendation)

        assert result.success
        assert result.executed_size > 0
        assert result.order_result.order_id is not None

    @pytest.mark.asyncio
    async def test_slippage_rejection(self, executor, sample_recommendation, mock_client):
        """Test that high slippage trades are rejected."""
        # Simulate high slippage
        mock_client.get_market_price = AsyncMock(return_value={
            "yes_bid": 0.45,
            "yes_ask": 0.60,  # 20% higher than expected
            "yes_mid": 0.525,
            "no_bid": 0.40,
            "no_ask": 0.55,
            "spread": 0.15,
        })

        result = await executor.execute(sample_recommendation)

        assert not result.success
        assert "slippage" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_batch_execution(self, executor, sample_recommendation):
        """Test executing multiple trades."""
        recommendations = [sample_recommendation] * 3

        results = await executor.execute_batch(recommendations, parallel=False)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_journal_tracking(self, executor, sample_recommendation):
        """Test that trades are journaled."""
        # Simulate adding a result to journal
        result = ExecutionResult(
            recommendation=sample_recommendation,
            order_result=MagicMock(
                success=True,
                order_id="test_123",
                filled_size=100,
                filled_price=0.50,
            ),
            intended_size=100,
            executed_size=100,
            intended_price=0.50,
            executed_price=0.50,
            slippage=0.0,
            signal_time=datetime.now(),
            execution_time=datetime.now(),
            success=True,
            partial_fill=False,
        )

        executor.journal.add(result)

        summary = executor.get_journal_summary()
        assert summary["total_trades"] == 1
        assert summary["success_rate"] == 1.0
