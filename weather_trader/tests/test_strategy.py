"""
Tests for trading strategy components.
"""

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, AsyncMock

from weather_trader.config import get_city_config
from weather_trader.models.ensemble import EnsembleForecast, ModelForecast
from weather_trader.kalshi.markets import WeatherMarket, TemperatureBracket
from weather_trader.strategy.expected_value import (
    ExpectedValueCalculator, TradeSignal, SignalStrength
)
from weather_trader.strategy.position_sizing import (
    PositionSizer, PositionRecommendation
)
from weather_trader.strategy.executor import TradeExecutor, ExecutionResult


def _make_market(yes_price=0.50, city="nyc", target=None):
    """Helper to create a WeatherMarket with brackets for testing.

    Default bracket is 52-58°F, centered near the sample forecast mean of 55°F.
    With mean=55, std=3: P(52 ≤ T ≤ 58) ≈ 0.68.
    """
    if target is None:
        target = date(2024, 1, 15)
    city_config = get_city_config(city)
    bracket = TemperatureBracket(
        ticker="KXHIGHNY-24JAN15-T52",
        event_ticker="KXHIGHNY-24JAN15",
        description="52-58°F",
        temp_low=52.0,
        temp_high=58.0,
        yes_price_cents=int(yes_price * 100),
        no_price_cents=int((1 - yes_price) * 100),
        volume=10000,
        open_interest=5000,
    )
    return WeatherMarket(
        event_ticker="KXHIGHNY-24JAN15",
        series_ticker="KXHIGHNY",
        city=city,
        city_config=city_config,
        target_date=target,
        brackets=[bracket],
        is_active=True,
        question="Will NYC high temperature be 52-58°F on Jan 15?",
    )


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
        return _make_market(yes_price=0.50)

    @pytest.fixture
    def sample_forecast(self):
        return EnsembleForecast(
            date=date(2024, 1, 15),
            city="New York City",
            high_mean=55.0,
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
        """Test calculating positive edge (forecast prob > market price).

        Bracket 52-58°F with forecast mean=55, std=3 → P ≈ 0.68.
        Market price 50 cents → edge ≈ +0.18.
        """
        bracket = sample_market.brackets[0]
        signal = calculator.calculate_ev(sample_market, bracket, sample_forecast)

        assert signal.forecast_probability > 0.6
        assert signal.edge > 0
        assert signal.side == "YES"
        assert signal.is_tradeable

    def test_calculate_negative_edge(self, calculator, sample_market, sample_forecast):
        """Test calculating negative edge (forecast prob < market price)."""
        sample_market.brackets[0].yes_price_cents = 95
        bracket = sample_market.brackets[0]

        signal = calculator.calculate_ev(sample_market, bracket, sample_forecast)

        assert signal.edge < 0
        assert signal.side == "NO"

    def test_neutral_signal_low_edge(self, calculator, sample_market, sample_forecast):
        """Test that small edge results in neutral signal."""
        # Set market price close to forecast probability (~0.68) for small edge
        sample_market.brackets[0].yes_price_cents = 67
        bracket = sample_market.brackets[0]

        signal = calculator.calculate_ev(sample_market, bracket, sample_forecast)

        assert abs(signal.edge) < 0.05
        assert signal.signal == SignalStrength.NEUTRAL
        assert not signal.is_tradeable

    def test_neutral_signal_low_confidence(self, calculator, sample_market, sample_forecast):
        """Test that low confidence results in neutral signal."""
        sample_forecast.confidence = 0.5
        bracket = sample_market.brackets[0]

        signal = calculator.calculate_ev(sample_market, bracket, sample_forecast)

        assert signal.signal == SignalStrength.NEUTRAL

    def test_strong_signal_detection(self, calculator, sample_market, sample_forecast):
        """Test detection of strong signals."""
        sample_market.brackets[0].yes_price_cents = 30
        bracket = sample_market.brackets[0]

        signal = calculator.calculate_ev(sample_market, bracket, sample_forecast)

        assert signal.signal == SignalStrength.STRONG_BUY_YES
        assert signal.edge > 0.15

    def test_analyze_multiple_markets(self, calculator, sample_market, sample_forecast):
        """Test analyzing multiple markets at once."""
        markets = [sample_market]
        forecasts = {"nyc": sample_forecast}

        signals = calculator.analyze_markets(markets, forecasts)

        assert len(signals) == 1
        assert signals[0].market == sample_market
        assert signals[0].bracket == sample_market.brackets[0]


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
        market = _make_market(yes_price=0.50)

        return TradeSignal(
            market=market,
            bracket=market.brackets[0],
            forecast=MagicMock(confidence=0.85),
            forecast_probability=0.70,
            market_probability=0.50,
            edge=0.20,
            expected_value=0.20,
            signal=SignalStrength.STRONG_BUY_YES,
            side="YES",
            confidence=0.85,
            model_agreement=0.90,
            threshold=55.0,
            is_high_temp=True,
            is_over_market=True,
        )

    def test_kelly_calculation(self, sizer):
        """Test Kelly criterion calculation."""
        kelly = sizer.calculate_kelly(win_prob=0.6, win_amount=1.0, lose_amount=1.0)
        assert abs(kelly - 0.2) < 0.01

    def test_kelly_with_edge(self, sizer):
        """Test Kelly with different odds."""
        kelly = sizer.calculate_kelly(win_prob=0.55, win_amount=2.0, lose_amount=1.0)
        assert abs(kelly - 0.325) < 0.01

    def test_position_sizing(self, sizer, sample_signal):
        """Test position sizing for a signal."""
        rec = sizer.size_position(sample_signal)

        assert rec is not None
        assert rec.kelly_fraction > 0
        assert rec.adjusted_kelly < rec.kelly_fraction
        assert rec.recommended_size > 0
        assert rec.recommended_size <= 500

    def test_position_capping(self, sizer, sample_signal):
        """Test that positions are capped at max size."""
        sample_signal.edge = 0.50
        sample_signal.forecast_probability = 0.90
        sample_signal.expected_value = 0.50

        rec = sizer.size_position(sample_signal)

        assert rec.recommended_size <= 500
        assert rec.position_capped

    def test_daily_loss_limit(self, sizer, sample_signal):
        """Test daily loss limit enforcement."""
        sizer.daily_pnl = -800

        rec = sizer.size_position(sample_signal)

        assert rec.recommended_size <= 200

    def test_cannot_trade_at_limit(self, sizer):
        """Test that trading stops at loss limit."""
        sizer.daily_pnl = -1000

        can_trade, reason = sizer.can_trade()

        assert not can_trade
        assert "loss limit" in reason.lower()

    def test_batch_recommendations(self, sizer, sample_signal):
        """Test batch position recommendations."""
        signals = [sample_signal] * 5

        recs = sizer.recommend_batch(signals, max_positions=3)

        assert len(recs) <= 3
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
        market = _make_market(yes_price=0.50)

        signal = TradeSignal(
            market=market,
            bracket=market.brackets[0],
            forecast=MagicMock(),
            forecast_probability=0.70,
            market_probability=0.50,
            edge=0.20,
            expected_value=0.20,
            signal=SignalStrength.STRONG_BUY_YES,
            side="YES",
            confidence=0.85,
            model_agreement=0.90,
            threshold=55.0,
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
        mock_client.get_market_price = AsyncMock(return_value={
            "yes_bid": 0.45,
            "yes_ask": 0.60,
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
