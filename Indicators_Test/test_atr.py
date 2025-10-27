"""
Tests for ATR (Average True Range) Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.atr import ATRIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_flat_data,
    create_spike_data,
    assert_column_exists,
    assert_no_inf
)


class TestATRIndicator:
    """Test suite for ATR indicator."""

    def test_initialization_default(self):
        """Test ATR initialization with default config."""
        atr = ATRIndicator()
        assert atr.name == "ATR"
        assert atr.indicator_type == "volatility"
        assert atr.period == 14
        assert atr.required_columns == ['high', 'low', 'close']
        assert atr.output_columns == ['atr']

    def test_initialization_custom(self):
        """Test ATR initialization with custom config."""
        config = {'atr_period': 21}
        atr = ATRIndicator(config)
        assert atr.period == 21

    def test_calculate_basic(self):
        """Test basic ATR calculation."""
        df = create_sample_ohlcv_data(num_rows=100)
        atr = ATRIndicator()

        result_df = atr.calculate(df)

        # Check output column exists
        assert_column_exists(result_df, 'atr')

        # Check no infinite values
        assert_no_inf(result_df, ['atr'])

        # ATR should be positive (volatility measure)
        atr_values = result_df['atr'].dropna()
        assert (atr_values >= 0).all()

    def test_atr_always_positive(self):
        """Test that ATR values are always non-negative."""
        df = create_sample_ohlcv_data(num_rows=100)
        atr = ATRIndicator()

        result_df = atr.calculate(df)

        atr_values = result_df['atr'].dropna()
        assert (atr_values >= 0).all()

    def test_flat_data(self):
        """Test ATR with flat price data."""
        df = create_flat_data(num_rows=100, price=100.0)
        atr = ATRIndicator()

        result_df = atr.calculate(df)

        # With flat prices (no volatility), ATR should be 0
        atr_values = result_df['atr'].dropna()
        assert np.allclose(atr_values, 0, atol=1e-10)

    def test_high_volatility(self):
        """Test ATR increases with volatility."""
        # Low volatility data
        df_low_vol = create_flat_data(num_rows=100, price=100.0)

        # High volatility data
        df_high_vol = create_spike_data(num_rows=100, spike_position=50)

        atr = ATRIndicator()

        result_low = atr.calculate(df_low_vol)
        result_high = atr.calculate(df_high_vol)

        # High volatility should have higher ATR
        atr_low = result_low['atr'].iloc[-10:].mean()
        atr_high = result_high['atr'].iloc[-10:].mean()

        assert atr_high > atr_low

    def test_true_range_calculation(self):
        """Test True Range calculation (basis of ATR)."""
        # Create simple known data
        df = pd.DataFrame({
            'open': [10, 12, 11, 13],
            'high': [11, 13, 12, 14],
            'low': [9, 11, 10, 12],
            'close': [10.5, 12.5, 11.5, 13.5],
            'volume': [1000, 1000, 1000, 1000]
        })

        atr = ATRIndicator({'atr_period': 3})
        result_df = atr.calculate(df)

        # ATR should be calculated and non-negative
        atr_values = result_df['atr'].dropna()
        assert len(atr_values) > 0
        assert (atr_values >= 0).all()

    def test_wilder_smoothing(self):
        """Test that Wilder's smoothing is applied."""
        df = create_sample_ohlcv_data(num_rows=100, seed=42)
        atr = ATRIndicator({'atr_period': 14})

        result_df = atr.calculate(df)

        # ATR should be smooth (not jumpy)
        atr_values = result_df['atr'].dropna()

        # Calculate consecutive differences
        diffs = atr_values.diff().abs()

        # Differences should be relatively small (smoothed)
        # No single change should be more than 50% of mean ATR
        mean_atr = atr_values.mean()
        max_diff = diffs.max()

        assert max_diff < mean_atr * 0.5

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=10)
        atr = ATRIndicator()  # Needs 15 periods (14 + 1)

        result_df = atr.calculate_safe(df)
        assert 'atr' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1.5, 2.5, 3.5]
            # Missing 'high' and 'low'
        })
        atr = ATRIndicator()

        result_df = atr.calculate_safe(df)
        assert 'atr' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        atr = ATRIndicator()

        result_df = atr.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        atr = ATRIndicator()

        result_df = atr.calculate(df)

        values = atr.get_values(result_df, 'atr')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        atr = ATRIndicator()

        result_df = atr.calculate(df)

        latest = atr.get_latest_value(result_df, 'atr')
        assert isinstance(latest, float)
        assert not np.isnan(latest)
        assert latest >= 0

    def test_different_periods(self):
        """Test ATR with different periods."""
        df = create_sample_ohlcv_data(num_rows=100)

        atr_7 = ATRIndicator({'atr_period': 7})
        atr_21 = ATRIndicator({'atr_period': 21})

        result_7 = atr_7.calculate(df)
        result_21 = atr_21.calculate(df)

        # Both should produce valid ATR values
        assert result_7['atr'].dropna().between(0, float('inf')).all()
        assert result_21['atr'].dropna().between(0, float('inf')).all()

        # 7-period should have fewer NaN values
        assert result_7['atr'].isna().sum() < result_21['atr'].isna().sum()

    def test_atr_as_volatility_measure(self):
        """Test that ATR responds to volatility changes."""
        # Create data with changing volatility
        # First half: low volatility, second half: high volatility
        np.random.seed(42)
        close_low_vol = 100 + np.random.randn(50) * 0.5
        close_high_vol = 100 + np.random.randn(50) * 5.0
        close = np.concatenate([close_low_vol, close_high_vol])

        high = close + np.abs(np.random.randn(100)) * 0.5
        low = close - np.abs(np.random.randn(100)) * 0.5

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': [1000] * 100
        })

        atr = ATRIndicator({'atr_period': 14})
        result_df = atr.calculate(df)

        # ATR in high volatility period should be higher
        atr_low_vol = result_df['atr'].iloc[30:45].mean()
        atr_high_vol = result_df['atr'].iloc[70:85].mean()

        assert atr_high_vol > atr_low_vol

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'atr_period': 14, 'cache_enabled': True}
        atr = ATRIndicator(config)

        # First calculation
        result_df1 = atr.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = atr.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache
        atr.clear_cache()
        result_df3 = atr.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_string_representation(self):
        """Test string representations."""
        atr = ATRIndicator()

        str_repr = str(atr)
        assert "ATR" in str_repr
        assert "volatility" in str_repr

        repr_str = repr(atr)
        assert "ATRIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
