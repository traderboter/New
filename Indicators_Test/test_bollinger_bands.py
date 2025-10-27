"""
Tests for Bollinger Bands Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.bollinger_bands import BollingerBandsIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_trending_data,
    create_flat_data,
    create_spike_data,
    assert_column_exists,
    assert_no_inf
)


class TestBollingerBandsIndicator:
    """Test suite for Bollinger Bands indicator."""

    def test_initialization_default(self):
        """Test Bollinger Bands initialization with default config."""
        bb = BollingerBandsIndicator()
        assert bb.name == "Bollinger Bands"
        assert bb.indicator_type == "volatility"
        assert bb.period == 20
        assert bb.std_multiplier == 2.0
        assert bb.required_columns == ['close']
        assert bb.output_columns == ['bb_upper', 'bb_middle', 'bb_lower']

    def test_initialization_custom(self):
        """Test Bollinger Bands initialization with custom config."""
        config = {'bb_period': 10, 'bb_std': 2.5}
        bb = BollingerBandsIndicator(config)
        assert bb.period == 10
        assert bb.std_multiplier == 2.5

    def test_calculate_basic(self):
        """Test basic Bollinger Bands calculation."""
        df = create_sample_ohlcv_data(num_rows=100)
        bb = BollingerBandsIndicator()

        result_df = bb.calculate(df)

        # Check all output columns exist
        assert_column_exists(result_df, 'bb_upper')
        assert_column_exists(result_df, 'bb_middle')
        assert_column_exists(result_df, 'bb_lower')

        # Check no infinite values
        assert_no_inf(result_df, ['bb_upper', 'bb_middle', 'bb_lower'])

    def test_bands_relationship(self):
        """Test relationship between Bollinger Bands (upper > middle > lower)."""
        df = create_sample_ohlcv_data(num_rows=100)
        bb = BollingerBandsIndicator()

        result_df = bb.calculate(df)

        # Remove NaN values
        valid_data = result_df.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])

        # Upper band should always be >= middle band
        assert (valid_data['bb_upper'] >= valid_data['bb_middle']).all()

        # Middle band should always be >= lower band
        assert (valid_data['bb_middle'] >= valid_data['bb_lower']).all()

    def test_middle_band_is_sma(self):
        """Test that middle band equals SMA."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'bb_period': 20}
        bb = BollingerBandsIndicator(config)

        result_df = bb.calculate(df)

        # Calculate SMA manually
        sma_manual = df['close'].rolling(window=20).mean()

        # Middle band should equal SMA
        diff = (result_df['bb_middle'] - sma_manual).abs()
        assert diff.max() < 1e-10

    def test_band_width_calculation(self):
        """Test Bollinger Bands width calculation."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'bb_period': 20, 'bb_std': 2.0}
        bb = BollingerBandsIndicator(config)

        result_df = bb.calculate(df)

        # Calculate standard deviation manually
        std_manual = df['close'].rolling(window=20).std(ddof=0)  # Population std

        # Upper band = middle + (std * multiplier)
        expected_upper = result_df['bb_middle'] + (std_manual * 2.0)
        diff_upper = (result_df['bb_upper'] - expected_upper).abs()
        assert diff_upper.max() < 1e-10

        # Lower band = middle - (std * multiplier)
        expected_lower = result_df['bb_middle'] - (std_manual * 2.0)
        diff_lower = (result_df['bb_lower'] - expected_lower).abs()
        assert diff_lower.max() < 1e-10

    def test_flat_data(self):
        """Test Bollinger Bands with flat price data."""
        df = create_flat_data(num_rows=100, price=100.0)
        bb = BollingerBandsIndicator()

        result_df = bb.calculate(df)

        # With flat prices, std = 0, so all bands should be equal
        valid_data = result_df.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])

        assert np.allclose(valid_data['bb_upper'], 100.0, rtol=1e-9)
        assert np.allclose(valid_data['bb_middle'], 100.0, rtol=1e-9)
        assert np.allclose(valid_data['bb_lower'], 100.0, rtol=1e-9)

    def test_high_volatility(self):
        """Test Bollinger Bands with high volatility data."""
        df = create_spike_data(num_rows=100, spike_position=50)
        bb = BollingerBandsIndicator()

        result_df = bb.calculate(df)

        # Band width around spike should be larger
        band_width = result_df['bb_upper'] - result_df['bb_lower']

        # Width around spike (position 50-60) should be larger than before spike (30-40)
        width_before = band_width.iloc[30:40].mean()
        width_during = band_width.iloc[50:60].mean()

        assert width_during > width_before

    def test_trending_data(self):
        """Test Bollinger Bands with trending data."""
        df = create_trending_data(num_rows=100, trend='up')
        bb = BollingerBandsIndicator()

        result_df = bb.calculate(df)

        # In a trend, price often rides the upper or lower band
        # Check that close price is sometimes near the bands
        valid_data = result_df.iloc[-50:].dropna()

        # Calculate distance to bands
        dist_to_upper = (valid_data['bb_upper'] - valid_data['close']).abs()
        dist_to_lower = (valid_data['close'] - valid_data['bb_lower']).abs()
        band_width = valid_data['bb_upper'] - valid_data['bb_lower']

        # Some prices should be close to bands (within 20% of band width)
        close_to_upper = (dist_to_upper < band_width * 0.2).any()
        close_to_lower = (dist_to_lower < band_width * 0.2).any()

        assert close_to_upper or close_to_lower

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = create_sample_ohlcv_data(num_rows=10)
        bb = BollingerBandsIndicator()  # Needs 20 periods

        result_df = bb.calculate_safe(df)
        assert 'bb_upper' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5]
            # Missing 'close'
        })
        bb = BollingerBandsIndicator()

        result_df = bb.calculate_safe(df)
        assert 'bb_upper' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        bb = BollingerBandsIndicator()

        result_df = bb.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        bb = BollingerBandsIndicator()

        result_df = bb.calculate(df)

        # Get single column
        values = bb.get_values(result_df, 'bb_upper')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

        # Get all columns
        all_values = bb.get_values(result_df)
        assert isinstance(all_values, dict)
        assert 'bb_upper' in all_values
        assert 'bb_middle' in all_values
        assert 'bb_lower' in all_values

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        bb = BollingerBandsIndicator()

        result_df = bb.calculate(df)

        # Get latest single value
        latest = bb.get_latest_value(result_df, 'bb_upper')
        assert isinstance(latest, float)
        assert not np.isnan(latest)

        # Get all latest values
        all_latest = bb.get_latest_value(result_df)
        assert isinstance(all_latest, dict)
        assert 'bb_upper' in all_latest
        assert 'bb_middle' in all_latest
        assert 'bb_lower' in all_latest

    def test_different_multipliers(self):
        """Test Bollinger Bands with different standard deviation multipliers."""
        df = create_sample_ohlcv_data(num_rows=100)

        bb_2std = BollingerBandsIndicator({'bb_std': 2.0})
        bb_3std = BollingerBandsIndicator({'bb_std': 3.0})

        result_2std = bb_2std.calculate(df)
        result_3std = bb_3std.calculate(df)

        # 3-std bands should be wider than 2-std bands
        width_2std = (result_2std['bb_upper'] - result_2std['bb_lower']).mean()
        width_3std = (result_3std['bb_upper'] - result_3std['bb_lower']).mean()

        assert width_3std > width_2std

    def test_different_periods(self):
        """Test Bollinger Bands with different periods."""
        df = create_sample_ohlcv_data(num_rows=100)

        bb_10 = BollingerBandsIndicator({'bb_period': 10})
        bb_30 = BollingerBandsIndicator({'bb_period': 30})

        result_10 = bb_10.calculate(df)
        result_30 = bb_30.calculate(df)

        # 10-period should have fewer NaN values
        assert result_10['bb_middle'].isna().sum() < result_30['bb_middle'].isna().sum()

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'cache_enabled': True}
        bb = BollingerBandsIndicator(config)

        # First calculation
        result_df1 = bb.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = bb.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache
        bb.clear_cache()
        result_df3 = bb.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_string_representation(self):
        """Test string representations."""
        bb = BollingerBandsIndicator()

        str_repr = str(bb)
        assert "Bollinger Bands" in str_repr
        assert "volatility" in str_repr

        repr_str = repr(bb)
        assert "BollingerBandsIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
