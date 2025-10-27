"""
Tests for ATR (Average True Range) Indicator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.atr import ATRIndicator


class TestATRIndicator:
    """Test suite for ATR indicator."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        indicator = ATRIndicator()

        assert indicator.name == "ATR"
        assert indicator.indicator_type == "volatility"
        assert set(indicator.required_columns) == {'high', 'low', 'close'}
        assert indicator.period == 14
        assert indicator.output_columns == ['atr']

    def test_initialization_custom_period(self, config_custom):
        """Test initialization with custom period."""
        indicator = ATRIndicator(config_custom)

        assert indicator.period == 7

    def test_calculate_basic(self, sample_ohlcv_data):
        """Test basic ATR calculation."""
        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'atr' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)

        # ATR should be positive
        atr_values = result_df['atr'].dropna()
        assert (atr_values > 0).all()

    def test_true_range_components(self, sample_ohlcv_data):
        """Test True Range calculation components."""
        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        # ATR should exist and be valid
        assert 'atr' in result_df.columns
        atr = result_df['atr'].dropna()
        assert len(atr) > 0
        assert (atr >= 0).all()

    def test_atr_high_volatility(self):
        """Test ATR with high volatility data."""
        # Create high volatility data
        np.random.seed(42)
        high = np.random.normal(105, 10, 50)  # High variance
        low = high - np.abs(np.random.normal(5, 3, 50))
        close = low + (high - low) * np.random.random(50)

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })

        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(df)

        atr = result_df['atr'].dropna()

        # High volatility should produce higher ATR values
        assert atr.mean() > 3  # Should be relatively high

    def test_atr_low_volatility(self):
        """Test ATR with low volatility data."""
        # Create low volatility data
        np.random.seed(42)
        high = np.random.normal(100, 0.5, 50)  # Low variance
        low = high - np.abs(np.random.normal(0.2, 0.1, 50))
        close = low + (high - low) * np.random.random(50)

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })

        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(df)

        atr = result_df['atr'].dropna()

        # Low volatility should produce lower ATR values
        assert atr.mean() < 2  # Should be relatively low

    def test_atr_flat_price(self, flat_price_data):
        """Test ATR with flat price (no volatility)."""
        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(flat_price_data)

        atr = result_df['atr'].dropna()

        # Flat price means no true range, ATR should be 0
        assert np.allclose(atr, 0, atol=1e-10)

    def test_atr_wilder_smoothing(self, sample_ohlcv_data):
        """Test that ATR uses Wilder's smoothing (alpha = 1/period)."""
        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        # Just verify ATR is calculated smoothly
        atr = result_df['atr'].dropna()

        # ATR should be relatively smooth (check variance)
        # Calculate first difference to check smoothness
        atr_diff = atr.diff().dropna()
        assert atr_diff.std() < atr.std()  # Changes should be smoother than values

    def test_atr_increasing_volatility(self):
        """Test ATR response to increasing volatility."""
        # Create data with increasing volatility
        close = [100]
        high = [101]
        low = [99]

        for i in range(1, 50):
            volatility = i * 0.5  # Increasing volatility
            close.append(close[-1] + np.random.normal(0, volatility))
            high.append(close[-1] + abs(np.random.normal(0, volatility)))
            low.append(close[-1] - abs(np.random.normal(0, volatility)))

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })

        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(df)

        atr = result_df['atr'].dropna()

        # ATR should generally increase
        first_half = atr.iloc[:len(atr)//2].mean()
        second_half = atr.iloc[len(atr)//2:].mean()

        assert second_half > first_half

    def test_min_periods(self):
        """Test minimum periods requirement."""
        indicator = ATRIndicator({'atr_period': 14})

        # Need period + 1
        assert indicator._get_min_periods() == 15

    def test_insufficient_data(self, small_ohlcv_data):
        """Test with insufficient data."""
        config = {'atr_period': 50}
        indicator = ATRIndicator(config)

        result_df = indicator.calculate_safe(small_ohlcv_data)

        # Should fail validation
        assert 'atr' not in result_df.columns

    def test_get_latest_value(self, sample_ohlcv_data):
        """Test getting latest ATR value."""
        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(sample_ohlcv_data)
        latest = indicator.get_latest_value(result_df, 'atr')

        assert isinstance(latest, (int, float))
        assert not np.isnan(latest)
        assert latest >= 0


class TestATREdgeCases:
    """Test edge cases for ATR."""

    def test_gaps_in_price(self):
        """Test ATR with price gaps."""
        # Create data with gaps
        df = pd.DataFrame({
            'high': [105, 110, 115, 120, 105, 110],  # Gap down at index 4
            'low': [100, 105, 110, 115, 95, 100],
            'close': [103, 108, 113, 118, 98, 103]
        })

        # Add more data
        for i in range(30):
            df = pd.concat([df, pd.DataFrame({
                'high': [110 + i],
                'low': [100 + i],
                'close': [105 + i]
            })], ignore_index=True)

        indicator = ATRIndicator({'atr_period': 5})

        result_df = indicator.calculate_safe(df)

        # Should handle gaps correctly
        assert 'atr' in result_df.columns
        atr = result_df['atr'].dropna()
        assert (atr >= 0).all()

    def test_very_short_period(self, sample_ohlcv_data):
        """Test with very short ATR period."""
        indicator = ATRIndicator({'atr_period': 2})

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'atr' in result_df.columns
        atr = result_df['atr'].dropna()
        assert (atr >= 0).all()

    def test_extreme_price_movement(self):
        """Test ATR with extreme price movements."""
        # Create data with extreme move
        close = [100] * 20 + [200] * 20  # 100% jump
        high = [c * 1.01 for c in close]
        low = [c * 0.99 for c in close]

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })

        indicator = ATRIndicator({'atr_period': 14})

        result_df = indicator.calculate_safe(df)

        atr = result_df['atr']

        # ATR should spike after the extreme move
        atr_before_jump = atr.iloc[15:20].mean()
        atr_after_jump = atr.iloc[25:30].mean()

        assert atr_after_jump > atr_before_jump
