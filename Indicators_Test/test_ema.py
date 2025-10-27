"""
Tests for EMA (Exponential Moving Average) Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.ema import EMAIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_trending_data,
    create_flat_data,
    assert_column_exists,
    assert_no_inf
)


class TestEMAIndicator:
    """Test suite for EMA indicator."""

    def test_initialization_default(self):
        """Test EMA initialization with default config."""
        ema = EMAIndicator()
        assert ema.name == "EMA"
        assert ema.indicator_type == "trend"
        assert ema.periods == [20, 50, 200]
        assert ema.required_columns == ['close']
        assert ema.output_columns == ['ema_20', 'ema_50', 'ema_200']

    def test_initialization_custom(self):
        """Test EMA initialization with custom config."""
        config = {'ema_periods': [12, 26]}
        ema = EMAIndicator(config)
        assert ema.periods == [12, 26]
        assert ema.output_columns == ['ema_12', 'ema_26']

    def test_calculate_basic(self):
        """Test basic EMA calculation."""
        df = create_sample_ohlcv_data(num_rows=250)
        config = {'ema_periods': [20, 50]}
        ema = EMAIndicator(config)

        result_df = ema.calculate(df)

        # Check output columns exist
        assert_column_exists(result_df, 'ema_20')
        assert_column_exists(result_df, 'ema_50')

        # Check no infinite values
        assert_no_inf(result_df, ['ema_20', 'ema_50'])

        # Check first 19 values are NaN for 20-period EMA
        assert result_df['ema_20'].iloc[:19].isna().all()
        assert not result_df['ema_20'].iloc[19:].isna().all()

        # Check first 49 values are NaN for 50-period EMA
        assert result_df['ema_50'].iloc[:49].isna().all()
        assert not result_df['ema_50'].iloc[49:].isna().all()

    def test_ema_calculation_correctness(self):
        """Test EMA calculation correctness with known values."""
        # Create simple sequential data
        close_prices = list(range(10, 30))  # 10, 11, 12, ..., 29
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 20
        })

        config = {'ema_periods': [5]}
        ema = EMAIndicator(config)
        result_df = ema.calculate(df)

        # First EMA value should equal SMA
        # Index 4 (5th row): SMA = (10+11+12+13+14)/5 = 12
        assert abs(result_df['ema_5'].iloc[4] - 12.0) < 1e-6

        # Check that EMA values exist and are reasonable
        ema_values = result_df['ema_5'].dropna()
        assert len(ema_values) == 16  # 20 - 4 (first 4 are NaN)
        assert ema_values.min() >= 12.0  # Should be at least the first value
        assert ema_values.max() <= 29.0  # Should not exceed the last close price

    def test_ema_responds_faster_than_sma(self):
        """Test that EMA responds faster to price changes than SMA."""
        from signal_generation.analyzers.indicators.sma import SMAIndicator

        # Create data with a sudden price jump
        close_prices = [100] * 50 + [110] * 50
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 100
        })

        config = {'ema_periods': [20]}
        ema = EMAIndicator(config)
        result_df_ema = ema.calculate(df)

        config_sma = {'sma_periods': [20]}
        sma = SMAIndicator(config_sma)
        result_df_sma = sma.calculate(df)

        # After price jump (at index 60), EMA should be higher than SMA
        # because it gives more weight to recent prices
        idx = 60
        assert result_df_ema['ema_20'].iloc[idx] > result_df_sma['sma_20'].iloc[idx]

    def test_flat_data(self):
        """Test EMA with flat price data."""
        df = create_flat_data(num_rows=100, price=100.0)
        config = {'ema_periods': [20]}
        ema = EMAIndicator(config)

        result_df = ema.calculate(df)

        # EMA of flat data should equal the constant price
        ema_values = result_df['ema_20'].dropna()
        assert np.allclose(ema_values, 100.0, rtol=1e-9)

    def test_trending_data(self):
        """Test EMA with trending data."""
        df = create_trending_data(num_rows=100, trend='up')
        config = {'ema_periods': [10, 20]}
        ema = EMAIndicator(config)

        result_df = ema.calculate(df)

        # In uptrend, shorter EMA should generally be above longer EMA
        short_ema = result_df['ema_10'].iloc[-50:]
        long_ema = result_df['ema_20'].iloc[-50:]

        comparison = short_ema > long_ema
        assert comparison.sum() > len(comparison) * 0.7  # At least 70%

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=10)
        config = {'ema_periods': [20]}
        ema = EMAIndicator(config)

        # Should return original df without calculation
        result_df = ema.calculate_safe(df)
        assert 'ema_20' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5]
            # Missing 'close'
        })
        ema = EMAIndicator()

        result_df = ema.calculate_safe(df)
        assert 'ema_20' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        ema = EMAIndicator()

        result_df = ema.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'ema_periods': [20]}
        ema = EMAIndicator(config)

        result_df = ema.calculate(df)

        # Get single column
        values = ema.get_values(result_df, 'ema_20')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

        # Get all columns
        all_values = ema.get_values(result_df)
        assert isinstance(all_values, dict)
        assert 'ema_20' in all_values

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'ema_periods': [20]}
        ema = EMAIndicator(config)

        result_df = ema.calculate(df)

        latest = ema.get_latest_value(result_df, 'ema_20')
        assert isinstance(latest, float)
        assert not np.isnan(latest)

    def test_multiple_periods(self):
        """Test EMA with multiple periods."""
        df = create_sample_ohlcv_data(num_rows=300)
        config = {'ema_periods': [12, 26, 50, 100, 200]}
        ema = EMAIndicator(config)

        result_df = ema.calculate(df)

        # All columns should exist
        for period in [12, 26, 50, 100, 200]:
            assert_column_exists(result_df, f'ema_{period}')

        # Longer periods should have more initial NaN values
        assert result_df['ema_12'].isna().sum() < result_df['ema_200'].isna().sum()

    def test_ema_smoothness(self):
        """Test that EMA values are smooth (no sudden jumps)."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'ema_periods': [20]}
        ema = EMAIndicator(config)

        result_df = ema.calculate(df)

        # Calculate the maximum percentage change in EMA
        ema_values = result_df['ema_20'].dropna()
        pct_change = ema_values.pct_change().abs()

        # EMA should be relatively smooth - max change should be reasonable
        # (less than the price volatility)
        close_pct_change = df['close'].pct_change().abs()
        assert pct_change.max() <= close_pct_change.max()

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'ema_periods': [20], 'cache_enabled': True}
        ema = EMAIndicator(config)

        # First calculation
        result_df1 = ema.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = ema.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache
        ema.clear_cache()
        result_df3 = ema.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_string_representation(self):
        """Test string representations."""
        ema = EMAIndicator()

        str_repr = str(ema)
        assert "EMA" in str_repr
        assert "trend" in str_repr

        repr_str = repr(ema)
        assert "EMAIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
