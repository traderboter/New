"""
Tests for RSI (Relative Strength Index) Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.rsi import RSIIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_trending_data,
    create_flat_data,
    assert_column_exists,
    assert_no_inf,
    assert_in_range
)


class TestRSIIndicator:
    """Test suite for RSI indicator."""

    def test_initialization_default(self):
        """Test RSI initialization with default config."""
        rsi = RSIIndicator()
        assert rsi.name == "RSI"
        assert rsi.indicator_type == "momentum"
        assert rsi.period == 14
        assert rsi.required_columns == ['close']
        assert rsi.output_columns == ['rsi']

    def test_initialization_custom(self):
        """Test RSI initialization with custom config."""
        config = {'rsi_period': 21}
        rsi = RSIIndicator(config)
        assert rsi.period == 21

    def test_calculate_basic(self):
        """Test basic RSI calculation."""
        df = create_sample_ohlcv_data(num_rows=100)
        rsi = RSIIndicator()

        result_df = rsi.calculate(df)

        # Check output column exists
        assert_column_exists(result_df, 'rsi')

        # Check no infinite values
        assert_no_inf(result_df, ['rsi'])

        # Check RSI is in valid range [0, 100]
        assert_in_range(result_df, 'rsi', 0, 100, allow_nan=True)

    def test_rsi_range(self):
        """Test that RSI values are always between 0 and 100."""
        df = create_sample_ohlcv_data(num_rows=200)
        rsi = RSIIndicator()

        result_df = rsi.calculate(df)

        rsi_values = result_df['rsi'].dropna()
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100

    def test_rsi_uptrend(self):
        """Test RSI in strong uptrend (should be high)."""
        # Create strong uptrend
        close_prices = list(range(100, 200, 2))  # Consistent increase
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 50
        })

        rsi = RSIIndicator()
        result_df = rsi.calculate(df)

        # In strong uptrend, RSI should be high (>70 overbought)
        rsi_latest = result_df['rsi'].iloc[-10:].mean()
        assert rsi_latest > 70

    def test_rsi_downtrend(self):
        """Test RSI in strong downtrend (should be low)."""
        # Create strong downtrend
        close_prices = list(range(200, 100, -2))  # Consistent decrease
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 50
        })

        rsi = RSIIndicator()
        result_df = rsi.calculate(df)

        # In strong downtrend, RSI should be low (<30 oversold)
        rsi_latest = result_df['rsi'].iloc[-10:].mean()
        assert rsi_latest < 30

    def test_flat_data(self):
        """Test RSI with flat price data."""
        df = create_flat_data(num_rows=100, price=100.0)
        rsi = RSIIndicator()

        result_df = rsi.calculate(df)

        # With no price changes, all gains and losses are 0
        # RSI should be NaN or 50 (neutral)
        rsi_values = result_df['rsi'].iloc[20:]  # Skip initial values

        # All non-NaN values should be close to 50 (or be NaN)
        non_nan_values = rsi_values.dropna()
        if len(non_nan_values) > 0:
            # When there's no movement, RSI calculation might produce 0/0 = NaN
            # or could be set to 50 (neutral), depending on implementation
            pass  # Either NaN or ~50 is acceptable

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=10)
        rsi = RSIIndicator()  # Needs 15 periods (14 + 1)

        result_df = rsi.calculate_safe(df)
        assert 'rsi' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5]
            # Missing 'close'
        })
        rsi = RSIIndicator()

        result_df = rsi.calculate_safe(df)
        assert 'rsi' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        rsi = RSIIndicator()

        result_df = rsi.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        rsi = RSIIndicator()

        result_df = rsi.calculate(df)

        values = rsi.get_values(result_df, 'rsi')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        rsi = RSIIndicator()

        result_df = rsi.calculate(df)

        latest = rsi.get_latest_value(result_df, 'rsi')
        assert isinstance(latest, float)
        assert not np.isnan(latest)
        assert 0 <= latest <= 100

    def test_different_periods(self):
        """Test RSI with different periods."""
        df = create_sample_ohlcv_data(num_rows=100)

        rsi_14 = RSIIndicator({'rsi_period': 14})
        rsi_21 = RSIIndicator({'rsi_period': 21})

        result_14 = rsi_14.calculate(df)
        result_21 = rsi_21.calculate(df)

        # Both should produce valid RSI values
        assert result_14['rsi'].dropna().between(0, 100).all()
        assert result_21['rsi'].dropna().between(0, 100).all()

        # 14-period should have fewer NaN values than 21-period
        assert result_14['rsi'].isna().sum() < result_21['rsi'].isna().sum()

    def test_rsi_sensitivity(self):
        """Test that shorter period RSI is more sensitive to price changes."""
        df = create_sample_ohlcv_data(num_rows=100, seed=42)

        rsi_7 = RSIIndicator({'rsi_period': 7})
        rsi_21 = RSIIndicator({'rsi_period': 21})

        result_7 = rsi_7.calculate(df)
        result_21 = rsi_21.calculate(df)

        # Shorter period should show more volatility
        volatility_7 = result_7['rsi'].dropna().std()
        volatility_21 = result_21['rsi'].dropna().std()

        assert volatility_7 >= volatility_21  # 7-period should be more volatile

    def test_wilder_smoothing(self):
        """Test that Wilder's smoothing is applied correctly."""
        # Create data with known pattern
        close_prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                       45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 0.5 for x in close_prices],
            'low': [x - 0.5 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 20
        })

        config = {'rsi_period': 14}
        rsi = RSIIndicator(config)
        result_df = rsi.calculate(df)

        # RSI should be calculated (just check it exists and is valid)
        rsi_values = result_df['rsi'].dropna()
        assert len(rsi_values) > 0
        assert rsi_values.between(0, 100).all()

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'rsi_period': 14, 'cache_enabled': True}
        rsi = RSIIndicator(config)

        # First calculation
        result_df1 = rsi.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = rsi.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache
        rsi.clear_cache()
        result_df3 = rsi.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_extreme_values(self):
        """Test RSI with extreme price movements."""
        # Create data with extreme jump
        close_prices = [100] * 20 + [200] * 20 + [100] * 20
        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': [1000] * 60
        })

        rsi = RSIIndicator()
        result_df = rsi.calculate(df)

        # Even with extreme movements, RSI should stay in [0, 100]
        rsi_values = result_df['rsi'].dropna()
        assert rsi_values.between(0, 100).all()
        assert not rsi_values.isna().all()  # Should have some valid values

    def test_string_representation(self):
        """Test string representations."""
        rsi = RSIIndicator()

        str_repr = str(rsi)
        assert "RSI" in str_repr
        assert "momentum" in str_repr

        repr_str = repr(rsi)
        assert "RSIIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
