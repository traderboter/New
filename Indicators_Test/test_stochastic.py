"""
Tests for Stochastic Oscillator Indicator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.stochastic import StochasticIndicator


class TestStochasticIndicator:
    """Test suite for Stochastic indicator."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        indicator = StochasticIndicator()

        assert indicator.name == "Stochastic"
        assert indicator.indicator_type == "momentum"
        assert indicator.required_columns == ['high', 'low', 'close']
        assert indicator.k_period == 14
        assert indicator.d_period == 3
        assert indicator.smooth_k == 3
        assert set(indicator.output_columns) == {'stoch_k', 'stoch_d'}

    def test_initialization_custom_periods(self, config_custom):
        """Test initialization with custom periods."""
        indicator = StochasticIndicator(config_custom)

        assert indicator.k_period == 7
        assert indicator.d_period == 2
        assert indicator.smooth_k == 2

    def test_calculate_basic(self, sample_ohlcv_data):
        """Test basic Stochastic calculation."""
        indicator = StochasticIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'stoch_k' in result_df.columns
        assert 'stoch_d' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)

    def test_stochastic_range(self, sample_ohlcv_data):
        """Test that Stochastic values are in range [0, 100]."""
        indicator = StochasticIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        k_values = result_df['stoch_k'].dropna()
        d_values = result_df['stoch_d'].dropna()

        # All values should be in [0, 100]
        assert (k_values >= 0).all()
        assert (k_values <= 100).all()
        assert (d_values >= 0).all()
        assert (d_values <= 100).all()

    def test_d_is_smoothed_k(self, sample_ohlcv_data):
        """Test that %D is smoothed version of %K."""
        indicator = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        k = result_df['stoch_k']
        d = result_df['stoch_d']

        # %D should be moving average of %K
        expected_d = k.rolling(window=3).mean()

        assert np.allclose(
            d.dropna(),
            expected_d.dropna(),
            rtol=1e-10
        )

    def test_stochastic_at_high(self):
        """Test Stochastic when price is at high."""
        # Create data where close is always at high
        df = pd.DataFrame({
            'high': [110, 115, 120, 125, 130],
            'low': [100, 105, 110, 115, 120],
            'close': [110, 115, 120, 125, 130]  # Always at high
        })

        # Need more data for stochastic
        for i in range(20):
            df = pd.concat([df, pd.DataFrame({
                'high': [130 + i],
                'low': [120 + i],
                'close': [130 + i]
            })], ignore_index=True)

        indicator = StochasticIndicator({'stoch_k': 5, 'stoch_d': 3, 'stoch_smooth': 1})

        result_df = indicator.calculate_safe(df)

        # When close is at high, %K should be 100
        k_values = result_df['stoch_k'].iloc[-5:]
        assert np.allclose(k_values.dropna(), 100, rtol=0.01)

    def test_stochastic_at_low(self):
        """Test Stochastic when price is at low."""
        # Create data where close is always at low
        df = pd.DataFrame({
            'high': [110, 115, 120, 125, 130],
            'low': [100, 105, 110, 115, 120],
            'close': [100, 105, 110, 115, 120]  # Always at low
        })

        # Need more data
        for i in range(20):
            df = pd.concat([df, pd.DataFrame({
                'high': [130 + i],
                'low': [120 + i],
                'close': [120 + i]
            })], ignore_index=True)

        indicator = StochasticIndicator({'stoch_k': 5, 'stoch_d': 3, 'stoch_smooth': 1})

        result_df = indicator.calculate_safe(df)

        # When close is at low, %K should be 0
        k_values = result_df['stoch_k'].iloc[-5:]
        assert np.allclose(k_values.dropna(), 0, rtol=0.01)

    def test_stochastic_mid_range(self):
        """Test Stochastic when price is in middle of range."""
        # Create data where close is always in middle
        df = pd.DataFrame({
            'high': [110] * 30,
            'low': [100] * 30,
            'close': [105] * 30  # Middle of range
        })

        indicator = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})

        result_df = indicator.calculate_safe(df)

        # Should be around 50
        k_values = result_df['stoch_k'].dropna()
        assert 45 < k_values.mean() < 55

    def test_flat_price(self, flat_price_data):
        """Test Stochastic with flat price."""
        indicator = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})

        result_df = indicator.calculate_safe(flat_price_data)

        # With flat price (high=low=close), result should be neutral (50)
        k = result_df['stoch_k'].dropna()
        d = result_df['stoch_d'].dropna()

        # Should be around 50 (neutral)
        assert np.allclose(k, 50, rtol=0.01)
        assert np.allclose(d, 50, rtol=0.01)

    def test_min_periods(self):
        """Test minimum periods requirement."""
        indicator = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})

        # Need k_period + smooth_k + d_period
        assert indicator._get_min_periods() == 20

    def test_insufficient_data(self, small_ohlcv_data):
        """Test with insufficient data."""
        config = {'stoch_k': 50, 'stoch_d': 3, 'stoch_smooth': 3}
        indicator = StochasticIndicator(config)

        result_df = indicator.calculate_safe(small_ohlcv_data)

        # Should fail validation
        assert 'stoch_k' not in result_df.columns

    def test_get_latest_values(self, sample_ohlcv_data):
        """Test getting latest Stochastic values."""
        indicator = StochasticIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)
        latest = indicator.get_latest_value(result_df)

        assert isinstance(latest, dict)
        assert 'stoch_k' in latest
        assert 'stoch_d' in latest

        # Should be in valid range
        assert 0 <= latest['stoch_k'] <= 100
        assert 0 <= latest['stoch_d'] <= 100


class TestStochasticEdgeCases:
    """Test edge cases for Stochastic."""

    def test_overbought_condition(self):
        """Test overbought condition detection."""
        # Create strong uptrend
        df = pd.DataFrame({
            'high': np.arange(110, 210),
            'low': np.arange(100, 200),
            'close': np.arange(105, 205)
        })

        indicator = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})

        result_df = indicator.calculate_safe(df)

        # In strong uptrend, Stochastic should show overbought (>80)
        k_values = result_df['stoch_k'].iloc[-10:]
        assert k_values.mean() > 80

    def test_oversold_condition(self):
        """Test oversold condition detection."""
        # Create strong downtrend
        df = pd.DataFrame({
            'high': np.arange(200, 100, -1),
            'low': np.arange(190, 90, -1),
            'close': np.arange(195, 95, -1)
        })

        indicator = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})

        result_df = indicator.calculate_safe(df)

        # In strong downtrend, Stochastic should show oversold (<20)
        k_values = result_df['stoch_k'].iloc[-10:]
        assert k_values.mean() < 20

    def test_zero_range(self):
        """Test with zero range (high = low)."""
        df = pd.DataFrame({
            'high': [100] * 30,
            'low': [100] * 30,
            'close': [100] * 30
        })

        indicator = StochasticIndicator({'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3})

        result_df = indicator.calculate_safe(df)

        # Should handle gracefully with safe_divide
        k = result_df['stoch_k'].dropna()
        assert len(k) > 0
        # Should be neutral (50) when range is 0
        assert np.allclose(k, 50, rtol=0.01)
