"""
Tests for MACD (Moving Average Convergence Divergence) Indicator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.macd import MACDIndicator


class TestMACDIndicator:
    """Test suite for MACD indicator."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        indicator = MACDIndicator()

        assert indicator.name == "MACD"
        assert indicator.indicator_type == "momentum"
        assert indicator.required_columns == ['close']
        assert indicator.fast_period == 12
        assert indicator.slow_period == 26
        assert indicator.signal_period == 9
        assert set(indicator.output_columns) == {'macd', 'macd_signal', 'macd_hist'}

    def test_initialization_custom_periods(self, config_custom):
        """Test initialization with custom periods."""
        indicator = MACDIndicator(config_custom)

        assert indicator.fast_period == 5
        assert indicator.slow_period == 13
        assert indicator.signal_period == 5

    def test_calculate_basic(self, sample_ohlcv_data):
        """Test basic MACD calculation."""
        indicator = MACDIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'macd' in result_df.columns
        assert 'macd_signal' in result_df.columns
        assert 'macd_hist' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)

    def test_macd_components_relationship(self, sample_ohlcv_data):
        """Test relationship between MACD components."""
        indicator = MACDIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        macd = result_df['macd']
        signal = result_df['macd_signal']
        hist = result_df['macd_hist']

        # Histogram should equal MACD - Signal
        valid_idx = ~(macd.isna() | signal.isna() | hist.isna())
        assert np.allclose(
            hist[valid_idx],
            macd[valid_idx] - signal[valid_idx],
            rtol=1e-10
        )

    def test_macd_calculation_correctness(self, sample_ohlcv_data):
        """Test MACD calculation correctness."""
        indicator = MACDIndicator({'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        # Manually calculate MACD components
        ema_fast = sample_ohlcv_data['close'].ewm(span=12, adjust=False).mean()
        ema_slow = sample_ohlcv_data['close'].ewm(span=26, adjust=False).mean()
        expected_macd = ema_fast - ema_slow

        # Compare calculated MACD
        assert np.allclose(
            result_df['macd'].dropna(),
            expected_macd.dropna(),
            rtol=1e-10
        )

    def test_signal_line_calculation(self, sample_ohlcv_data):
        """Test signal line calculation."""
        indicator = MACDIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        macd = result_df['macd']
        signal = result_df['macd_signal']

        # Signal should be EMA of MACD
        expected_signal = macd.ewm(span=9, adjust=False).mean()

        assert np.allclose(
            signal.dropna(),
            expected_signal.dropna(),
            rtol=1e-10
        )

    def test_macd_uptrend(self):
        """Test MACD in uptrend."""
        # Create uptrend data
        df = pd.DataFrame({
            'close': np.arange(100, 200)
        })

        indicator = MACDIndicator({'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9})

        result_df = indicator.calculate_safe(df)

        # In uptrend, MACD should generally be positive
        macd_values = result_df['macd'].iloc[-20:]
        assert macd_values.mean() > 0

    def test_macd_downtrend(self):
        """Test MACD in downtrend."""
        # Create downtrend data
        df = pd.DataFrame({
            'close': np.arange(200, 100, -1)
        })

        indicator = MACDIndicator({'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9})

        result_df = indicator.calculate_safe(df)

        # In downtrend, MACD should generally be negative
        macd_values = result_df['macd'].iloc[-20:]
        assert macd_values.mean() < 0

    def test_macd_crossover(self):
        """Test MACD crossover detection."""
        # Create data that will produce crossover
        close_prices = [100] * 30 + list(np.arange(100, 150))  # Flat then uptrend
        df = pd.DataFrame({'close': close_prices})

        indicator = MACDIndicator({'macd_fast': 5, 'macd_slow': 10, 'macd_signal': 3})

        result_df = indicator.calculate_safe(df)

        hist = result_df['macd_hist'].dropna()

        # Histogram should change sign (crossover)
        # Check if there's any sign change
        sign_changes = np.diff(np.sign(hist))
        assert len(sign_changes[sign_changes != 0]) > 0

    def test_min_periods(self):
        """Test minimum periods requirement."""
        indicator = MACDIndicator({'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9})

        # Need slow_period + signal_period
        assert indicator._get_min_periods() == 35

    def test_insufficient_data(self, small_ohlcv_data):
        """Test with insufficient data."""
        config = {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9}
        indicator = MACDIndicator(config)

        result_df = indicator.calculate_safe(small_ohlcv_data)

        # Should fail validation
        assert 'macd' not in result_df.columns

    def test_get_latest_values(self, sample_ohlcv_data):
        """Test getting latest MACD values."""
        indicator = MACDIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)
        latest = indicator.get_latest_value(result_df)

        assert isinstance(latest, dict)
        assert 'macd' in latest
        assert 'macd_signal' in latest
        assert 'macd_hist' in latest

        # All should be valid numbers
        assert not np.isnan(latest['macd'])
        assert not np.isnan(latest['macd_signal'])
        assert not np.isnan(latest['macd_hist'])

    def test_flat_price(self, flat_price_data):
        """Test MACD with flat price."""
        indicator = MACDIndicator({'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9})

        result_df = indicator.calculate_safe(flat_price_data)

        # With flat price, MACD should be 0 (or very close)
        macd = result_df['macd'].dropna()
        assert np.allclose(macd, 0, atol=1e-10)


class TestMACDEdgeCases:
    """Test edge cases for MACD."""

    def test_very_short_periods(self, sample_ohlcv_data):
        """Test with very short periods."""
        config = {'macd_fast': 3, 'macd_slow': 6, 'macd_signal': 2}
        indicator = MACDIndicator(config)

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'macd' in result_df.columns
        assert 'macd_signal' in result_df.columns
        assert 'macd_hist' in result_df.columns

    def test_histogram_zero_crossings(self):
        """Test detecting histogram zero crossings."""
        # Create oscillating price
        close_prices = [100 + 10 * np.sin(i * 0.2) for i in range(100)]
        df = pd.DataFrame({'close': close_prices})

        indicator = MACDIndicator({'macd_fast': 5, 'macd_slow': 10, 'macd_signal': 3})

        result_df = indicator.calculate_safe(df)

        hist = result_df['macd_hist'].dropna()

        # Should have multiple zero crossings in oscillating data
        zero_crossings = np.where(np.diff(np.sign(hist)))[0]
        assert len(zero_crossings) > 0

    def test_strong_momentum(self):
        """Test MACD with strong momentum."""
        # Create strong trending data
        close_prices = [100 + i**1.5 for i in range(100)]
        df = pd.DataFrame({'close': close_prices})

        indicator = MACDIndicator({'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9})

        result_df = indicator.calculate_safe(df)

        # MACD histogram should be positive and increasing
        hist = result_df['macd_hist'].iloc[-20:]
        assert hist.mean() > 0
