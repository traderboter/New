"""
Tests for Bollinger Bands Indicator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.bollinger_bands import BollingerBandsIndicator


class TestBollingerBandsIndicator:
    """Test suite for Bollinger Bands indicator."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        indicator = BollingerBandsIndicator()

        assert indicator.name == "Bollinger Bands"
        assert indicator.indicator_type == "volatility"
        assert indicator.required_columns == ['close']
        assert indicator.period == 20
        assert indicator.std_multiplier == 2.0
        assert set(indicator.output_columns) == {'bb_upper', 'bb_middle', 'bb_lower'}

    def test_initialization_custom_params(self, config_custom):
        """Test initialization with custom parameters."""
        indicator = BollingerBandsIndicator(config_custom)

        assert indicator.period == 10
        assert indicator.std_multiplier == 1.5

    def test_calculate_basic(self, sample_ohlcv_data):
        """Test basic Bollinger Bands calculation."""
        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'bb_upper' in result_df.columns
        assert 'bb_middle' in result_df.columns
        assert 'bb_lower' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)

    def test_band_relationships(self, sample_ohlcv_data):
        """Test relationships between bands."""
        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        upper = result_df['bb_upper']
        middle = result_df['bb_middle']
        lower = result_df['bb_lower']

        # Remove NaN values
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())

        # Upper should always be >= Middle >= Lower
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

        # Bands should be symmetric around middle
        upper_distance = upper[valid_idx] - middle[valid_idx]
        lower_distance = middle[valid_idx] - lower[valid_idx]
        assert np.allclose(upper_distance, lower_distance, rtol=1e-10)

    def test_middle_band_is_sma(self, sample_ohlcv_data):
        """Test that middle band is SMA."""
        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        # Calculate SMA manually
        expected_sma = sample_ohlcv_data['close'].rolling(window=20).mean()

        assert np.allclose(
            result_df['bb_middle'].dropna(),
            expected_sma.dropna(),
            rtol=1e-10
        )

    def test_band_calculation_formula(self, sample_ohlcv_data):
        """Test band calculation formula."""
        period = 20
        std_mult = 2.0

        indicator = BollingerBandsIndicator({'bb_period': period, 'bb_std': std_mult})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        # Calculate manually
        sma = sample_ohlcv_data['close'].rolling(window=period).mean()
        std = sample_ohlcv_data['close'].rolling(window=period).std(ddof=0)

        expected_upper = sma + (std * std_mult)
        expected_lower = sma - (std * std_mult)

        assert np.allclose(
            result_df['bb_upper'].dropna(),
            expected_upper.dropna(),
            rtol=1e-10
        )
        assert np.allclose(
            result_df['bb_lower'].dropna(),
            expected_lower.dropna(),
            rtol=1e-10
        )

    def test_high_volatility_widens_bands(self):
        """Test that high volatility widens the bands."""
        # Create high volatility data
        np.random.seed(42)
        high_vol_close = 100 + np.random.normal(0, 10, 50)

        df = pd.DataFrame({'close': high_vol_close})

        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(df)

        # Calculate band width
        band_width = result_df['bb_upper'] - result_df['bb_lower']
        avg_width = band_width.dropna().mean()

        # With high volatility (std=10), bands should be relatively wide
        assert avg_width > 20  # 2 * 2 * ~5 (rough estimate)

    def test_low_volatility_narrows_bands(self):
        """Test that low volatility narrows the bands."""
        # Create low volatility data
        np.random.seed(42)
        low_vol_close = 100 + np.random.normal(0, 1, 50)

        df = pd.DataFrame({'close': low_vol_close})

        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(df)

        # Calculate band width
        band_width = result_df['bb_upper'] - result_df['bb_lower']
        avg_width = band_width.dropna().mean()

        # With low volatility (std=1), bands should be narrow
        assert avg_width < 10

    def test_flat_price(self, flat_price_data):
        """Test Bollinger Bands with flat price."""
        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(flat_price_data)

        upper = result_df['bb_upper'].dropna()
        middle = result_df['bb_middle'].dropna()
        lower = result_df['bb_lower'].dropna()

        # With no volatility, all bands should equal the price
        assert np.allclose(upper, 100.0, rtol=1e-10)
        assert np.allclose(middle, 100.0, rtol=1e-10)
        assert np.allclose(lower, 100.0, rtol=1e-10)

    def test_price_at_upper_band(self):
        """Test when price reaches upper band."""
        # Create data with upward spike
        close = [100] * 30 + [120] + [100] * 19

        df = pd.DataFrame({'close': close})

        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(df)

        # After spike, price should be near or above upper band
        spike_idx = 30
        price_at_spike = result_df['close'].iloc[spike_idx]
        upper_at_spike = result_df['bb_upper'].iloc[spike_idx]

        # Price spike should reach toward upper band
        assert price_at_spike >= upper_at_spike * 0.9  # Within 10%

    def test_different_std_multipliers(self, sample_ohlcv_data):
        """Test different standard deviation multipliers."""
        indicator_1 = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 1.0})
        indicator_2 = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})
        indicator_3 = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 3.0})

        result_1 = indicator_1.calculate_safe(sample_ohlcv_data)
        result_2 = indicator_2.calculate_safe(sample_ohlcv_data)
        result_3 = indicator_3.calculate_safe(sample_ohlcv_data)

        # Width should increase with multiplier
        width_1 = (result_1['bb_upper'] - result_1['bb_lower']).dropna().mean()
        width_2 = (result_2['bb_upper'] - result_2['bb_lower']).dropna().mean()
        width_3 = (result_3['bb_upper'] - result_3['bb_lower']).dropna().mean()

        assert width_1 < width_2 < width_3

    def test_min_periods(self):
        """Test minimum periods requirement."""
        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        assert indicator._get_min_periods() == 20

    def test_insufficient_data(self, small_ohlcv_data):
        """Test with insufficient data."""
        config = {'bb_period': 50, 'bb_std': 2.0}
        indicator = BollingerBandsIndicator(config)

        result_df = indicator.calculate_safe(small_ohlcv_data)

        # Should fail validation
        assert 'bb_upper' not in result_df.columns

    def test_get_latest_values(self, sample_ohlcv_data):
        """Test getting latest Bollinger Bands values."""
        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(sample_ohlcv_data)
        latest = indicator.get_latest_value(result_df)

        assert isinstance(latest, dict)
        assert 'bb_upper' in latest
        assert 'bb_middle' in latest
        assert 'bb_lower' in latest

        # Check relationships
        assert latest['bb_upper'] >= latest['bb_middle']
        assert latest['bb_middle'] >= latest['bb_lower']


class TestBollingerBandsEdgeCases:
    """Test edge cases for Bollinger Bands."""

    def test_very_short_period(self, sample_ohlcv_data):
        """Test with very short period."""
        indicator = BollingerBandsIndicator({'bb_period': 5, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'bb_upper' in result_df.columns
        assert 'bb_middle' in result_df.columns
        assert 'bb_lower' in result_df.columns

    def test_squeeze_pattern(self):
        """Test Bollinger Bands squeeze pattern."""
        # Create data: low volatility followed by expansion
        close = [100 + np.random.normal(0, 0.1) for _ in range(30)]  # Low vol
        close += [100 + np.random.normal(0, 5) for _ in range(30)]  # High vol

        df = pd.DataFrame({'close': close})

        indicator = BollingerBandsIndicator({'bb_period': 20, 'bb_std': 2.0})

        result_df = indicator.calculate_safe(df)

        # Calculate bandwidth
        bandwidth = result_df['bb_upper'] - result_df['bb_lower']

        # Bandwidth should be smaller in first half
        first_half = bandwidth.iloc[20:30].mean()
        second_half = bandwidth.iloc[40:50].mean()

        assert second_half > first_half
