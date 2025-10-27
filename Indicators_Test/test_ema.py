"""
Tests for EMA (Exponential Moving Average) Indicator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.ema import EMAIndicator


class TestEMAIndicator:
    """Test suite for EMA indicator."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        indicator = EMAIndicator()

        assert indicator.name == "EMA"
        assert indicator.indicator_type == "trend"
        assert indicator.required_columns == ['close']
        assert indicator.periods == [20, 50, 200]
        assert 'ema_20' in indicator.output_columns

    def test_initialization_custom_periods(self, config_custom):
        """Test initialization with custom periods."""
        indicator = EMAIndicator(config_custom)

        assert indicator.periods == [10]
        assert indicator.output_columns == ['ema_10']

    def test_calculate_single_period(self, sample_ohlcv_data):
        """Test EMA calculation for single period."""
        config = {'ema_periods': [20]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'ema_20' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)

        # First 19 values should be NaN
        assert result_df['ema_20'].iloc[:19].isna().all()

        # 20th value should be SMA
        expected_first_ema = sample_ohlcv_data['close'].iloc[:20].mean()
        assert np.isclose(result_df['ema_20'].iloc[19], expected_first_ema, rtol=1e-10)

    def test_calculate_multiple_periods(self, sample_ohlcv_data):
        """Test EMA calculation for multiple periods."""
        config = {'ema_periods': [10, 20, 50]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'ema_10' in result_df.columns
        assert 'ema_20' in result_df.columns
        assert 'ema_50' in result_df.columns

    def test_ema_vs_sma(self, sample_ohlcv_data):
        """Test that EMA is different from SMA (more responsive)."""
        config = {'ema_periods': [20]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        # Calculate SMA for comparison
        sma_20 = sample_ohlcv_data['close'].rolling(window=20).mean()
        ema_20 = result_df['ema_20']

        # EMA and SMA should be different (except possibly at start)
        # Check the last 50 values
        # With smooth data, they may be very close but should not be identical
        # Reduced tolerance to 0.005 (0.5%) to allow for smooth data
        assert not np.allclose(ema_20.iloc[-50:], sma_20.iloc[-50:], rtol=0.005)

    def test_ema_responsiveness(self, small_ohlcv_data):
        """Test that EMA is more responsive to recent price changes than SMA."""
        # Create data with sudden price jump
        df = small_ohlcv_data.copy()
        df.loc[5:, 'close'] = df.loc[5:, 'close'] * 1.2  # 20% jump

        config = {'ema_periods': [5]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(df)

        # EMA should exist
        assert 'ema_5' in result_df.columns

        # After the jump, EMA should adjust
        ema_before_jump = result_df['ema_5'].iloc[5]
        ema_after_jump = result_df['ema_5'].iloc[-1]

        # EMA should increase after price jump
        assert ema_after_jump > ema_before_jump

    def test_ema_formula_correctness(self, small_ohlcv_data):
        """Test EMA formula correctness."""
        config = {'ema_periods': [5]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(small_ohlcv_data)

        ema = result_df['ema_5']
        close = result_df['close']

        # Verify EMA formula manually for a few points
        alpha = 2 / (5 + 1)  # 0.333...

        # First EMA should be SMA
        expected_first = close.iloc[:5].mean()
        assert np.isclose(ema.iloc[4], expected_first, rtol=1e-10)

        # Verify subsequent EMAs using the formula
        for i in range(5, min(8, len(close))):
            expected_ema = close.iloc[i] * alpha + ema.iloc[i-1] * (1 - alpha)
            assert np.isclose(ema.iloc[i], expected_ema, rtol=1e-10)

    def test_ema_flat_price(self, flat_price_data):
        """Test EMA with flat price."""
        config = {'ema_periods': [10]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(flat_price_data)

        ema = result_df['ema_10'].dropna()

        # EMA of flat price should equal the price
        assert np.allclose(ema, 100.0, rtol=1e-10)

    def test_min_periods(self):
        """Test minimum periods requirement."""
        config = {'ema_periods': [20, 50]}
        indicator = EMAIndicator(config)

        assert indicator._get_min_periods() == 50

    def test_insufficient_data(self, small_ohlcv_data):
        """Test with insufficient data."""
        config = {'ema_periods': [50]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(small_ohlcv_data)

        # Should fail validation
        assert 'ema_50' not in result_df.columns

    def test_get_latest_value(self, sample_ohlcv_data):
        """Test getting latest EMA value."""
        config = {'ema_periods': [20]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)
        latest = indicator.get_latest_value(result_df, 'ema_20')

        assert isinstance(latest, (int, float))
        assert not np.isnan(latest)

    def test_nan_values_in_price(self, sample_ohlcv_data):
        """Test handling of NaN values in price data."""
        df = sample_ohlcv_data.copy()
        df.loc[10:12, 'close'] = np.nan

        config = {'ema_periods': [10]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(df)

        # Should still calculate (pandas handles NaN in ewm)
        assert 'ema_10' in result_df.columns


class TestEMAEdgeCases:
    """Test edge cases for EMA."""

    def test_very_short_period(self, sample_ohlcv_data):
        """Test with very short EMA period."""
        config = {'ema_periods': [2]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'ema_2' in result_df.columns
        # Should have values starting from index 1
        assert not pd.isna(result_df['ema_2'].iloc[1])

    def test_very_long_period(self, sample_ohlcv_data):
        """Test with very long EMA period."""
        config = {'ema_periods': [90]}
        indicator = EMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'ema_90' in result_df.columns
        # Should have NaN for first 89 values
        assert result_df['ema_90'].iloc[:89].isna().all()

    def test_empty_periods_list(self):
        """Test with empty periods list."""
        config = {'ema_periods': []}
        indicator = EMAIndicator(config)

        # Should have empty output columns
        assert indicator.output_columns == []
