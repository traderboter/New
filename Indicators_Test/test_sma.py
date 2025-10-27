"""
Tests for SMA (Simple Moving Average) Indicator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.sma import SMAIndicator


class TestSMAIndicator:
    """Test suite for SMA indicator."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        indicator = SMAIndicator()

        assert indicator.name == "SMA"
        assert indicator.indicator_type == "trend"
        assert indicator.required_columns == ['close']
        assert indicator.periods == [20, 50, 200]
        assert 'sma_20' in indicator.output_columns
        assert 'sma_50' in indicator.output_columns
        assert 'sma_200' in indicator.output_columns

    def test_initialization_custom_periods(self, config_custom):
        """Test initialization with custom periods."""
        indicator = SMAIndicator(config_custom)

        assert indicator.periods == [10]
        assert indicator.output_columns == ['sma_10']

    def test_calculate_single_period(self, sample_ohlcv_data):
        """Test SMA calculation for single period."""
        config = {'sma_periods': [20]}
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'sma_20' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)

        # Verify calculation manually for a few points
        expected_sma_20 = sample_ohlcv_data['close'].rolling(window=20).mean()
        assert np.allclose(
            result_df['sma_20'].dropna(),
            expected_sma_20.dropna(),
            rtol=1e-10
        )

    def test_calculate_multiple_periods(self, sample_ohlcv_data):
        """Test SMA calculation for multiple periods."""
        config = {'sma_periods': [10, 20, 50]}
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'sma_10' in result_df.columns
        assert 'sma_20' in result_df.columns
        assert 'sma_50' in result_df.columns

        # All SMAs should have correct length
        assert len(result_df) == len(sample_ohlcv_data)

        # Verify each SMA
        for period in [10, 20, 50]:
            col = f'sma_{period}'
            expected = sample_ohlcv_data['close'].rolling(window=period).mean()
            assert np.allclose(
                result_df[col].dropna(),
                expected.dropna(),
                rtol=1e-10
            )

    def test_sma_properties(self, sample_ohlcv_data):
        """Test SMA properties and characteristics."""
        config = {'sma_periods': [10]}
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        sma = result_df['sma_10']

        # First 9 values should be NaN (need 10 periods)
        assert sma.iloc[:9].isna().all()

        # 10th value should be valid
        assert not pd.isna(sma.iloc[9])

        # SMA should be smoother than close price (less variance)
        sma_valid = sma.dropna()
        close_valid = result_df['close'].iloc[9:]

        assert sma_valid.std() < close_valid.std()

    def test_sma_flat_price(self, flat_price_data):
        """Test SMA with flat price (no movement)."""
        config = {'sma_periods': [10]}
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(flat_price_data)

        sma = result_df['sma_10'].dropna()

        # SMA of flat price should be equal to the price
        assert np.allclose(sma, 100.0, rtol=1e-10)

    def test_min_periods(self):
        """Test minimum periods requirement."""
        config = {'sma_periods': [20, 50]}
        indicator = SMAIndicator(config)

        # Min periods should be the maximum period
        assert indicator._get_min_periods() == 50

    def test_insufficient_data(self, small_ohlcv_data):
        """Test with insufficient data."""
        config = {'sma_periods': [50]}  # Need 50 periods but only have 10
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(small_ohlcv_data)

        # Should return original DataFrame (validation failed)
        assert 'sma_50' not in result_df.columns

    def test_get_latest_value(self, sample_ohlcv_data):
        """Test getting latest SMA value."""
        config = {'sma_periods': [20]}
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)
        latest = indicator.get_latest_value(result_df, 'sma_20')

        # Should be a valid number
        assert isinstance(latest, (int, float))
        assert not np.isnan(latest)

        # Should match manual calculation
        expected = sample_ohlcv_data['close'].iloc[-20:].mean()
        assert np.isclose(latest, expected, rtol=1e-10)

    def test_edge_case_single_row(self):
        """Test with single row of data."""
        df = pd.DataFrame({
            'close': [100]
        })

        config = {'sma_periods': [5]}
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(df)

        # Should fail validation (insufficient data)
        assert 'sma_5' not in result_df.columns

    def test_nan_handling(self, sample_ohlcv_data):
        """Test handling of NaN values in price data."""
        df = sample_ohlcv_data.copy()
        # Insert some NaN values
        df.loc[10:15, 'close'] = np.nan

        config = {'sma_periods': [10]}
        indicator = SMAIndicator(config)

        result_df = indicator.calculate_safe(df)

        # SMA should handle NaN appropriately (pandas rolling handles this)
        assert 'sma_10' in result_df.columns
