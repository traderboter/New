"""
Tests for SMA (Simple Moving Average) Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.sma import SMAIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_trending_data,
    create_flat_data,
    assert_column_exists,
    assert_no_inf
)


class TestSMAIndicator:
    """Test suite for SMA indicator."""

    def test_initialization_default(self):
        """Test SMA initialization with default config."""
        sma = SMAIndicator()
        assert sma.name == "SMA"
        assert sma.indicator_type == "trend"
        assert sma.periods == [20, 50, 200]
        assert sma.required_columns == ['close']
        assert sma.output_columns == ['sma_20', 'sma_50', 'sma_200']

    def test_initialization_custom(self):
        """Test SMA initialization with custom config."""
        config = {'sma_periods': [10, 30]}
        sma = SMAIndicator(config)
        assert sma.periods == [10, 30]
        assert sma.output_columns == ['sma_10', 'sma_30']

    def test_calculate_basic(self):
        """Test basic SMA calculation."""
        df = create_sample_ohlcv_data(num_rows=250)
        config = {'sma_periods': [20, 50]}
        sma = SMAIndicator(config)

        result_df = sma.calculate(df)

        # Check output columns exist
        assert_column_exists(result_df, 'sma_20')
        assert_column_exists(result_df, 'sma_50')

        # Check no infinite values
        assert_no_inf(result_df, ['sma_20', 'sma_50'])

        # Check first 19 values are NaN for 20-period SMA
        assert result_df['sma_20'].iloc[:19].isna().all()
        assert not result_df['sma_20'].iloc[19:].isna().all()

        # Check first 49 values are NaN for 50-period SMA
        assert result_df['sma_50'].iloc[:49].isna().all()
        assert not result_df['sma_50'].iloc[49:].isna().all()

    def test_sma_calculation_correctness(self):
        """Test SMA calculation correctness with known values."""
        # Create simple data
        close_prices = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 10
        })

        config = {'sma_periods': [3, 5]}
        sma = SMAIndicator(config)
        result_df = sma.calculate(df)

        # Manual calculation for 3-period SMA
        # Index 2: (10+12+14)/3 = 12
        # Index 3: (12+14+16)/3 = 14
        # Index 4: (14+16+18)/3 = 16
        assert abs(result_df['sma_3'].iloc[2] - 12.0) < 1e-6
        assert abs(result_df['sma_3'].iloc[3] - 14.0) < 1e-6
        assert abs(result_df['sma_3'].iloc[4] - 16.0) < 1e-6

        # Manual calculation for 5-period SMA
        # Index 4: (10+12+14+16+18)/5 = 14
        assert abs(result_df['sma_5'].iloc[4] - 14.0) < 1e-6

    def test_flat_data(self):
        """Test SMA with flat price data."""
        df = create_flat_data(num_rows=100, price=100.0)
        config = {'sma_periods': [20]}
        sma = SMAIndicator(config)

        result_df = sma.calculate(df)

        # SMA of flat data should equal the constant price
        sma_values = result_df['sma_20'].dropna()
        assert np.allclose(sma_values, 100.0, rtol=1e-9)

    def test_trending_data(self):
        """Test SMA with trending data."""
        df = create_trending_data(num_rows=100, trend='up')
        config = {'sma_periods': [10, 20]}
        sma = SMAIndicator(config)

        result_df = sma.calculate(df)

        # In uptrend, shorter SMA should be above longer SMA (after initial period)
        # Check last 50 values
        short_sma = result_df['sma_10'].iloc[-50:]
        long_sma = result_df['sma_20'].iloc[-50:]

        # Most values should show short > long in uptrend
        comparison = short_sma > long_sma
        assert comparison.sum() > len(comparison) * 0.8  # At least 80% should be true

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=10)
        config = {'sma_periods': [20]}
        sma = SMAIndicator(config)

        # Should return original df without calculation
        result_df = sma.calculate_safe(df)
        assert 'sma_20' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5]
            # Missing 'close'
        })
        sma = SMAIndicator()

        # Should return original df
        result_df = sma.calculate_safe(df)
        assert 'sma_20' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        sma = SMAIndicator()

        result_df = sma.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'sma_periods': [20]}
        sma = SMAIndicator(config)

        result_df = sma.calculate(df)

        # Get single column
        values = sma.get_values(result_df, 'sma_20')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

        # Get all columns
        all_values = sma.get_values(result_df)
        assert isinstance(all_values, dict)
        assert 'sma_20' in all_values

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'sma_periods': [20]}
        sma = SMAIndicator(config)

        result_df = sma.calculate(df)

        # Get latest value
        latest = sma.get_latest_value(result_df, 'sma_20')
        assert isinstance(latest, float)
        assert not np.isnan(latest)

        # Get all latest values
        all_latest = sma.get_latest_value(result_df)
        assert isinstance(all_latest, dict)
        assert 'sma_20' in all_latest

    def test_multiple_periods(self):
        """Test SMA with multiple periods."""
        df = create_sample_ohlcv_data(num_rows=300)
        config = {'sma_periods': [10, 20, 50, 100, 200]}
        sma = SMAIndicator(config)

        result_df = sma.calculate(df)

        # All columns should exist
        for period in [10, 20, 50, 100, 200]:
            assert_column_exists(result_df, f'sma_{period}')

        # Longer periods should have more initial NaN values
        assert result_df['sma_10'].isna().sum() < result_df['sma_200'].isna().sum()

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'sma_periods': [20], 'cache_enabled': True}
        sma = SMAIndicator(config)

        # First calculation
        result_df1 = sma.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = sma.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache and recalculate
        sma.clear_cache()
        result_df3 = sma.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_string_representation(self):
        """Test string representations."""
        sma = SMAIndicator()

        str_repr = str(sma)
        assert "SMA" in str_repr
        assert "trend" in str_repr

        repr_str = repr(sma)
        assert "SMAIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
