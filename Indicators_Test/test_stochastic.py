"""
Tests for Stochastic Oscillator Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.stochastic import StochasticIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_trending_data,
    create_flat_data,
    assert_column_exists,
    assert_no_inf,
    assert_in_range
)


class TestStochasticIndicator:
    """Test suite for Stochastic Oscillator indicator."""

    def test_initialization_default(self):
        """Test Stochastic initialization with default config."""
        stoch = StochasticIndicator()
        assert stoch.name == "Stochastic"
        assert stoch.indicator_type == "momentum"
        assert stoch.k_period == 14
        assert stoch.d_period == 3
        assert stoch.smooth_k == 3
        assert stoch.required_columns == ['high', 'low', 'close']
        assert stoch.output_columns == ['stoch_k', 'stoch_d']

    def test_initialization_custom(self):
        """Test Stochastic initialization with custom config."""
        config = {'stoch_k': 21, 'stoch_d': 5, 'stoch_smooth': 5}
        stoch = StochasticIndicator(config)
        assert stoch.k_period == 21
        assert stoch.d_period == 5
        assert stoch.smooth_k == 5

    def test_calculate_basic(self):
        """Test basic Stochastic calculation."""
        df = create_sample_ohlcv_data(num_rows=100)
        stoch = StochasticIndicator()

        result_df = stoch.calculate(df)

        # Check output columns exist
        assert_column_exists(result_df, 'stoch_k')
        assert_column_exists(result_df, 'stoch_d')

        # Check no infinite values
        assert_no_inf(result_df, ['stoch_k', 'stoch_d'])

        # Check values are in range [0, 100]
        assert_in_range(result_df, 'stoch_k', 0, 100, allow_nan=True)
        assert_in_range(result_df, 'stoch_d', 0, 100, allow_nan=True)

    def test_stochastic_range(self):
        """Test that Stochastic values are always between 0 and 100."""
        df = create_sample_ohlcv_data(num_rows=200)
        stoch = StochasticIndicator()

        result_df = stoch.calculate(df)

        k_values = result_df['stoch_k'].dropna()
        d_values = result_df['stoch_d'].dropna()

        assert (k_values >= 0).all() and (k_values <= 100).all()
        assert (d_values >= 0).all() and (d_values <= 100).all()

    def test_stochastic_overbought(self):
        """Test Stochastic in overbought condition."""
        # Create strong uptrend where close is at high
        close_prices = list(range(100, 150))
        df = pd.DataFrame({
            'open': [x - 0.5 for x in close_prices],
            'high': close_prices,  # Close = High
            'low': [x - 2 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 50
        })

        stoch = StochasticIndicator()
        result_df = stoch.calculate(df)

        # When close = high, stochastic %K should be 100
        k_latest = result_df['stoch_k'].iloc[-10:].mean()
        assert k_latest > 80  # Should be in overbought region

    def test_stochastic_oversold(self):
        """Test Stochastic in oversold condition."""
        # Create strong downtrend where close is at low
        close_prices = list(range(150, 100, -1))
        df = pd.DataFrame({
            'open': [x + 0.5 for x in close_prices],
            'high': [x + 2 for x in close_prices],
            'low': close_prices,  # Close = Low
            'close': close_prices,
            'volume': [1000] * 50
        })

        stoch = StochasticIndicator()
        result_df = stoch.calculate(df)

        # When close = low, stochastic %K should be 0
        k_latest = result_df['stoch_k'].iloc[-10:].mean()
        assert k_latest < 20  # Should be in oversold region

    def test_flat_data(self):
        """Test Stochastic with flat price data."""
        df = create_flat_data(num_rows=100, price=100.0)
        stoch = StochasticIndicator()

        result_df = stoch.calculate(df)

        # With flat prices, stochastic should be around 50 (neutral)
        # or could be NaN due to zero range
        k_values = result_df['stoch_k'].dropna()

        if len(k_values) > 0:
            # If values exist, they should be around 50
            assert np.allclose(k_values, 50, atol=1)

    def test_d_is_sma_of_k(self):
        """Test that %D is the moving average of %K."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3}
        stoch = StochasticIndicator(config)

        result_df = stoch.calculate(df)

        # Calculate %D manually as SMA of %K
        manual_d = result_df['stoch_k'].rolling(window=3).mean()

        # Compare with calculated %D
        diff = (result_df['stoch_d'] - manual_d).abs()
        assert diff.max() < 1e-10

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=10)
        stoch = StochasticIndicator()  # Needs k_period + smooth_k + d_period

        result_df = stoch.calculate_safe(df)
        assert 'stoch_k' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1.5, 2.5, 3.5]
            # Missing 'high' and 'low'
        })
        stoch = StochasticIndicator()

        result_df = stoch.calculate_safe(df)
        assert 'stoch_k' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        stoch = StochasticIndicator()

        result_df = stoch.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        stoch = StochasticIndicator()

        result_df = stoch.calculate(df)

        # Get single column
        values = stoch.get_values(result_df, 'stoch_k')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

        # Get all columns
        all_values = stoch.get_values(result_df)
        assert isinstance(all_values, dict)
        assert 'stoch_k' in all_values
        assert 'stoch_d' in all_values

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        stoch = StochasticIndicator()

        result_df = stoch.calculate(df)

        # Get latest single value
        latest = stoch.get_latest_value(result_df, 'stoch_k')
        assert isinstance(latest, float)
        assert not np.isnan(latest)
        assert 0 <= latest <= 100

        # Get all latest values
        all_latest = stoch.get_latest_value(result_df)
        assert isinstance(all_latest, dict)
        assert 'stoch_k' in all_latest
        assert 'stoch_d' in all_latest

    def test_different_periods(self):
        """Test Stochastic with different periods."""
        df = create_sample_ohlcv_data(num_rows=100)

        stoch_14 = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})
        stoch_21 = StochasticIndicator({'stoch_k': 21, 'stoch_d': 3, 'stoch_smooth': 3})

        result_14 = stoch_14.calculate(df)
        result_21 = stoch_21.calculate(df)

        # Both should produce valid values
        assert result_14['stoch_k'].dropna().between(0, 100).all()
        assert result_21['stoch_k'].dropna().between(0, 100).all()

        # Different periods should produce different values
        assert not np.allclose(
            result_14['stoch_k'].dropna().iloc[-20:],
            result_21['stoch_k'].dropna().iloc[-20:],
            rtol=1e-3
        )

    def test_stochastic_crossover(self):
        """Test Stochastic %K and %D crossover."""
        df = create_sample_ohlcv_data(num_rows=100, seed=42)
        stoch = StochasticIndicator()

        result_df = stoch.calculate(df)

        # Check for crossovers (where %K crosses %D)
        valid_data = result_df.dropna(subset=['stoch_k', 'stoch_d'])

        k_above_d = valid_data['stoch_k'] > valid_data['stoch_d']
        crossovers = (k_above_d != k_above_d.shift(1)).sum()

        # Should have at least some crossovers in random data
        assert crossovers > 0

    def test_k_more_sensitive_than_d(self):
        """Test that %K is more sensitive than %D."""
        df = create_sample_ohlcv_data(num_rows=100, seed=42)
        stoch = StochasticIndicator()

        result_df = stoch.calculate(df)

        # %K should be more volatile than %D (higher std dev)
        k_std = result_df['stoch_k'].dropna().std()
        d_std = result_df['stoch_d'].dropna().std()

        assert k_std >= d_std

    def test_safe_division(self):
        """Test that safe division handles zero range correctly."""
        # Create data with zero range (high = low)
        df = pd.DataFrame({
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })

        stoch = StochasticIndicator()
        result_df = stoch.calculate(df)

        # Should not crash, and values should be valid (50 or NaN)
        k_values = result_df['stoch_k'].dropna()

        if len(k_values) > 0:
            # Values should be neutral (around 50) when range is zero
            assert k_values.between(40, 60).all()

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'cache_enabled': True}
        stoch = StochasticIndicator(config)

        # First calculation
        result_df1 = stoch.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = stoch.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache
        stoch.clear_cache()
        result_df3 = stoch.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_string_representation(self):
        """Test string representations."""
        stoch = StochasticIndicator()

        str_repr = str(stoch)
        assert "Stochastic" in str_repr
        assert "momentum" in str_repr

        repr_str = repr(stoch)
        assert "StochasticIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
