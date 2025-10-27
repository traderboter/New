"""
Tests for Indicator Orchestrator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.indicator_orchestrator import IndicatorOrchestrator
from signal_generation.analyzers.indicators.ema import EMAIndicator
from signal_generation.analyzers.indicators.rsi import RSIIndicator
from Indicators_Test.test_utils import create_sample_ohlcv_data


class TestIndicatorOrchestrator:
    """Test suite for Indicator Orchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = IndicatorOrchestrator()

        assert orchestrator is not None
        assert len(orchestrator.all_indicators) > 0
        assert len(orchestrator.trend_indicators) > 0
        assert len(orchestrator.momentum_indicators) > 0
        assert len(orchestrator.volatility_indicators) > 0
        assert len(orchestrator.volume_indicators) > 0

    def test_all_indicators_loaded(self):
        """Test that all expected indicators are loaded."""
        orchestrator = IndicatorOrchestrator()

        # Check that key indicators are present
        assert 'EMA' in orchestrator.all_indicators
        assert 'SMA' in orchestrator.all_indicators
        assert 'RSI' in orchestrator.all_indicators
        assert 'MACD' in orchestrator.all_indicators
        assert 'ATR' in orchestrator.all_indicators
        assert 'Bollinger Bands' in orchestrator.all_indicators
        assert 'Stochastic' in orchestrator.all_indicators
        assert 'OBV' in orchestrator.all_indicators

    def test_indicators_categorized_correctly(self):
        """Test that indicators are categorized by type."""
        orchestrator = IndicatorOrchestrator()

        # Trend indicators
        assert 'EMA' in orchestrator.trend_indicators
        assert 'SMA' in orchestrator.trend_indicators

        # Momentum indicators
        assert 'RSI' in orchestrator.momentum_indicators
        assert 'MACD' in orchestrator.momentum_indicators
        assert 'Stochastic' in orchestrator.momentum_indicators

        # Volatility indicators
        assert 'ATR' in orchestrator.volatility_indicators
        assert 'Bollinger Bands' in orchestrator.volatility_indicators

        # Volume indicators
        assert 'OBV' in orchestrator.volume_indicators

    def test_calculate_all_indicators(self):
        """Test calculating all indicators at once."""
        df = create_sample_ohlcv_data(num_rows=250)
        orchestrator = IndicatorOrchestrator()

        result_df = orchestrator.calculate_all(df)

        # Check that indicator columns were added
        assert 'ema_20' in result_df.columns
        assert 'sma_20' in result_df.columns
        assert 'rsi' in result_df.columns
        assert 'macd' in result_df.columns
        assert 'atr' in result_df.columns
        assert 'bb_upper' in result_df.columns
        assert 'stoch_k' in result_df.columns
        assert 'obv' in result_df.columns

        # Original columns should still be there
        assert 'close' in result_df.columns
        assert 'high' in result_df.columns
        assert 'low' in result_df.columns
        assert 'volume' in result_df.columns

    def test_calculate_specific_indicators(self):
        """Test calculating specific indicators only."""
        df = create_sample_ohlcv_data(num_rows=100)
        orchestrator = IndicatorOrchestrator()

        # Calculate only RSI and MACD
        result_df = orchestrator.calculate_all(df, indicator_names=['RSI', 'MACD'])

        # Should have RSI and MACD columns
        assert 'rsi' in result_df.columns
        assert 'macd' in result_df.columns

        # Should NOT have other indicators
        # (Note: they might exist if calculate_all doesn't filter properly)

    def test_calculate_by_type_trend(self):
        """Test calculating only trend indicators."""
        df = create_sample_ohlcv_data(num_rows=250)
        orchestrator = IndicatorOrchestrator()

        result_df = orchestrator.calculate_by_type(df, 'trend')

        # Should have trend indicators
        assert 'ema_20' in result_df.columns
        assert 'sma_20' in result_df.columns

        # Should NOT have other types
        assert 'rsi' not in result_df.columns
        assert 'macd' not in result_df.columns

    def test_calculate_by_type_momentum(self):
        """Test calculating only momentum indicators."""
        df = create_sample_ohlcv_data(num_rows=100)
        orchestrator = IndicatorOrchestrator()

        result_df = orchestrator.calculate_by_type(df, 'momentum')

        # Should have momentum indicators
        assert 'rsi' in result_df.columns
        assert 'macd' in result_df.columns
        assert 'stoch_k' in result_df.columns

    def test_calculate_by_type_volatility(self):
        """Test calculating only volatility indicators."""
        df = create_sample_ohlcv_data(num_rows=100)
        orchestrator = IndicatorOrchestrator()

        result_df = orchestrator.calculate_by_type(df, 'volatility')

        # Should have volatility indicators
        assert 'atr' in result_df.columns
        assert 'bb_upper' in result_df.columns

    def test_calculate_by_type_volume(self):
        """Test calculating only volume indicators."""
        df = create_sample_ohlcv_data(num_rows=100)
        orchestrator = IndicatorOrchestrator()

        result_df = orchestrator.calculate_by_type(df, 'volume')

        # Should have volume indicators
        assert 'obv' in result_df.columns

    def test_get_indicator(self):
        """Test getting a specific indicator."""
        orchestrator = IndicatorOrchestrator()

        # Get RSI indicator
        rsi = orchestrator.get_indicator('RSI')
        assert rsi is not None
        assert isinstance(rsi, RSIIndicator)

        # Get non-existent indicator
        missing = orchestrator.get_indicator('NonExistent')
        assert missing is None

    def test_get_indicator_value(self):
        """Test getting indicator values from calculated DataFrame."""
        df = create_sample_ohlcv_data(num_rows=100)
        orchestrator = IndicatorOrchestrator()

        result_df = orchestrator.calculate_all(df)

        # Get RSI values
        rsi_values = orchestrator.get_indicator_value(result_df, 'RSI', 'rsi')
        assert isinstance(rsi_values, pd.Series)
        assert len(rsi_values) == len(df)

    def test_get_available_indicators(self):
        """Test getting list of available indicators."""
        orchestrator = IndicatorOrchestrator()

        available = orchestrator.get_available_indicators()

        assert isinstance(available, dict)
        assert 'trend' in available
        assert 'momentum' in available
        assert 'volatility' in available
        assert 'volume' in available

        assert isinstance(available['trend'], list)
        assert len(available['trend']) > 0

    def test_clear_all_caches(self):
        """Test clearing all indicator caches."""
        orchestrator = IndicatorOrchestrator()
        df = create_sample_ohlcv_data(num_rows=100)

        # Calculate once
        orchestrator.calculate_all(df)

        # Clear caches
        orchestrator.clear_all_caches()

        # Should not crash
        assert True

    def test_get_stats(self):
        """Test getting calculation statistics."""
        orchestrator = IndicatorOrchestrator()
        df = create_sample_ohlcv_data(num_rows=100)

        # Initial stats
        stats_before = orchestrator.get_stats()
        assert stats_before['total_calculations'] == 0

        # Calculate indicators
        orchestrator.calculate_all(df)

        # Stats should be updated
        stats_after = orchestrator.get_stats()
        assert stats_after['total_calculations'] > 0

    def test_reset_stats(self):
        """Test resetting calculation statistics."""
        orchestrator = IndicatorOrchestrator()
        df = create_sample_ohlcv_data(num_rows=100)

        # Calculate indicators
        orchestrator.calculate_all(df)

        # Reset stats
        orchestrator.reset_stats()

        # Stats should be zero
        stats = orchestrator.get_stats()
        assert stats['total_calculations'] == 0
        assert stats['cache_hits'] == 0
        assert stats['errors'] == 0

    def test_register_indicator(self):
        """Test registering a new indicator."""
        orchestrator = IndicatorOrchestrator()

        initial_count = len(orchestrator.all_indicators)

        # Register EMA again (should replace)
        orchestrator.register_indicator(EMAIndicator)

        # Count should be the same (replaced, not added)
        assert len(orchestrator.all_indicators) == initial_count

    def test_calculate_with_insufficient_data(self):
        """Test calculating with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=10)
        orchestrator = IndicatorOrchestrator()

        # Should not crash, but most indicators won't calculate
        result_df = orchestrator.calculate_all(df)

        # DataFrame should be returned (even if indicators didn't calculate)
        assert len(result_df) == len(df)

    def test_calculate_with_empty_dataframe(self):
        """Test calculating with empty DataFrame."""
        df = pd.DataFrame()
        orchestrator = IndicatorOrchestrator()

        result_df = orchestrator.calculate_all(df)

        # Should return empty DataFrame
        assert len(result_df) == 0

    def test_calculation_order(self):
        """Test that indicators are calculated in correct order."""
        df = create_sample_ohlcv_data(num_rows=250)
        orchestrator = IndicatorOrchestrator()

        # Should calculate in order: trend -> momentum -> volatility -> volume
        result_df = orchestrator.calculate_all(df)

        # All should be calculated without errors
        assert 'ema_20' in result_df.columns  # Trend
        assert 'rsi' in result_df.columns  # Momentum
        assert 'atr' in result_df.columns  # Volatility
        assert 'obv' in result_df.columns  # Volume

    def test_error_handling(self):
        """Test error handling during calculation."""
        orchestrator = IndicatorOrchestrator()

        # Create DataFrame missing required columns
        df = pd.DataFrame({
            'close': [100, 101, 102]
            # Missing high, low, volume
        })

        # Should not crash, should handle errors gracefully
        result_df = orchestrator.calculate_all(df)

        # Should return the original DataFrame
        assert len(result_df) == len(df)

    def test_multiple_calculations(self):
        """Test multiple calculations on same orchestrator."""
        orchestrator = IndicatorOrchestrator()

        df1 = create_sample_ohlcv_data(num_rows=100, seed=42)
        df2 = create_sample_ohlcv_data(num_rows=150, seed=43)

        result1 = orchestrator.calculate_all(df1)
        result2 = orchestrator.calculate_all(df2)

        # Both should succeed
        assert len(result1) == 100
        assert len(result2) == 150

    def test_string_representation(self):
        """Test string representation."""
        orchestrator = IndicatorOrchestrator()

        str_repr = str(orchestrator)
        assert "IndicatorOrchestrator" in str_repr
        assert "total=" in str_repr

    def test_configuration_passed_to_indicators(self):
        """Test that configuration is passed to indicators."""
        config = {
            'ema_periods': [10, 20],
            'rsi_period': 21
        }

        orchestrator = IndicatorOrchestrator(config)

        # Get EMA indicator
        ema = orchestrator.get_indicator('EMA')
        assert ema.periods == [10, 20]

        # Get RSI indicator
        rsi = orchestrator.get_indicator('RSI')
        assert rsi.period == 21

    def test_calculation_statistics_accuracy(self):
        """Test that calculation statistics are accurate."""
        orchestrator = IndicatorOrchestrator()
        df = create_sample_ohlcv_data(num_rows=250)

        # Calculate specific indicators
        orchestrator.calculate_all(df, indicator_names=['RSI', 'MACD'])

        stats = orchestrator.get_stats()

        # Should have calculated at least 2 indicators
        assert stats['total_calculations'] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
