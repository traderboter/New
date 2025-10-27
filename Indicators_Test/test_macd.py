"""
Tests for MACD (Moving Average Convergence Divergence) Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.macd import MACDIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_trending_data,
    create_flat_data,
    assert_column_exists,
    assert_no_inf
)


class TestMACDIndicator:
    """Test suite for MACD indicator."""

    def test_initialization_default(self):
        """Test MACD initialization with default config."""
        macd = MACDIndicator()
        assert macd.name == "MACD"
        assert macd.indicator_type == "momentum"
        assert macd.fast_period == 12
        assert macd.slow_period == 26
        assert macd.signal_period == 9
        assert macd.required_columns == ['close']
        assert macd.output_columns == ['macd', 'macd_signal', 'macd_hist']

    def test_initialization_custom(self):
        """Test MACD initialization with custom config."""
        config = {
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5
        }
        macd = MACDIndicator(config)
        assert macd.fast_period == 8
        assert macd.slow_period == 21
        assert macd.signal_period == 5

    def test_calculate_basic(self):
        """Test basic MACD calculation."""
        df = create_sample_ohlcv_data(num_rows=100)
        macd = MACDIndicator()

        result_df = macd.calculate(df)

        # Check all output columns exist
        assert_column_exists(result_df, 'macd')
        assert_column_exists(result_df, 'macd_signal')
        assert_column_exists(result_df, 'macd_hist')

        # Check no infinite values
        assert_no_inf(result_df, ['macd', 'macd_signal', 'macd_hist'])

    def test_macd_components_relationship(self):
        """Test relationship between MACD components."""
        df = create_sample_ohlcv_data(num_rows=100)
        macd = MACDIndicator()

        result_df = macd.calculate(df)

        # MACD Histogram = MACD Line - Signal Line
        expected_hist = result_df['macd'] - result_df['macd_signal']

        # Check that histogram matches the formula (allowing for small floating point errors)
        diff = (result_df['macd_hist'] - expected_hist).abs()
        assert diff.max() < 1e-10

    def test_macd_uptrend(self):
        """Test MACD in uptrend."""
        df = create_trending_data(num_rows=100, trend='up')
        macd = MACDIndicator()

        result_df = macd.calculate(df)

        # In uptrend, MACD should generally be positive (MACD line > Signal line)
        macd_positive_count = (result_df['macd_hist'].iloc[-20:] > 0).sum()
        assert macd_positive_count > 10  # Most recent values should be positive

    def test_macd_downtrend(self):
        """Test MACD in downtrend."""
        df = create_trending_data(num_rows=100, trend='down')
        macd = MACDIndicator()

        result_df = macd.calculate(df)

        # In downtrend, MACD histogram should generally be negative
        macd_negative_count = (result_df['macd_hist'].iloc[-20:] < 0).sum()
        assert macd_negative_count > 10

    def test_flat_data(self):
        """Test MACD with flat price data."""
        df = create_flat_data(num_rows=100, price=100.0)
        macd = MACDIndicator()

        result_df = macd.calculate(df)

        # With flat prices, MACD should be 0 (or very close to 0)
        macd_values = result_df['macd'].dropna()
        macd_signal_values = result_df['macd_signal'].dropna()
        macd_hist_values = result_df['macd_hist'].dropna()

        assert np.allclose(macd_values, 0, atol=1e-10)
        assert np.allclose(macd_signal_values, 0, atol=1e-10)
        assert np.allclose(macd_hist_values, 0, atol=1e-10)

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=20)
        macd = MACDIndicator()  # Needs 26 + 9 = 35 periods

        result_df = macd.calculate_safe(df)
        assert 'macd' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5]
            # Missing 'close'
        })
        macd = MACDIndicator()

        result_df = macd.calculate_safe(df)
        assert 'macd' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        macd = MACDIndicator()

        result_df = macd.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        macd = MACDIndicator()

        result_df = macd.calculate(df)

        # Get single column
        values = macd.get_values(result_df, 'macd')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

        # Get all columns
        all_values = macd.get_values(result_df)
        assert isinstance(all_values, dict)
        assert 'macd' in all_values
        assert 'macd_signal' in all_values
        assert 'macd_hist' in all_values

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        macd = MACDIndicator()

        result_df = macd.calculate(df)

        # Get latest single value
        latest = macd.get_latest_value(result_df, 'macd')
        assert isinstance(latest, float)
        assert not np.isnan(latest)

        # Get all latest values
        all_latest = macd.get_latest_value(result_df)
        assert isinstance(all_latest, dict)
        assert 'macd' in all_latest
        assert 'macd_signal' in all_latest
        assert 'macd_hist' in all_latest

    def test_macd_crossover(self):
        """Test MACD crossover detection."""
        # Create data where MACD will cross signal line
        close_prices = [100] * 30 + list(range(100, 130))  # Flat then uptrend
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * len(close_prices)
        })

        macd = MACDIndicator()
        result_df = macd.calculate(df)

        # Check for crossover (histogram changes sign)
        hist = result_df['macd_hist'].dropna()
        sign_changes = ((hist > 0) != (hist.shift(1) > 0)).sum()

        # Should have at least one crossover
        assert sign_changes >= 1

    def test_different_periods(self):
        """Test MACD with different period configurations."""
        df = create_sample_ohlcv_data(num_rows=100)

        config1 = {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9}
        config2 = {'macd_fast': 8, 'macd_slow': 17, 'macd_signal': 9}

        macd1 = MACDIndicator(config1)
        macd2 = MACDIndicator(config2)

        result1 = macd1.calculate(df)
        result2 = macd2.calculate(df)

        # Both should produce valid results
        assert not result1['macd'].dropna().empty
        assert not result2['macd'].dropna().empty

        # Different configurations should produce different values
        # (at least some values should differ)
        assert not np.allclose(
            result1['macd'].dropna(),
            result2['macd'].dropna(),
            rtol=1e-5
        )

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'cache_enabled': True}
        macd = MACDIndicator(config)

        # First calculation
        result_df1 = macd.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = macd.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache
        macd.clear_cache()
        result_df3 = macd.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_string_representation(self):
        """Test string representations."""
        macd = MACDIndicator()

        str_repr = str(macd)
        assert "MACD" in str_repr
        assert "momentum" in str_repr

        repr_str = repr(macd)
        assert "MACDIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
