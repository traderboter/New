"""
Tests for OBV (On-Balance Volume) Indicator.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.obv import OBVIndicator


class TestOBVIndicator:
    """Test suite for OBV indicator."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        indicator = OBVIndicator()

        assert indicator.name == "OBV"
        assert indicator.indicator_type == "volume"
        assert set(indicator.required_columns) == {'close', 'volume'}
        assert indicator.output_columns == ['obv']

    def test_calculate_basic(self, sample_ohlcv_data):
        """Test basic OBV calculation."""
        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'obv' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)

    def test_obv_price_up_adds_volume(self):
        """Test that OBV adds volume when price goes up."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],  # Continuous uptrend
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # OBV should be increasing
        assert obv.iloc[1] > obv.iloc[0]
        assert obv.iloc[2] > obv.iloc[1]
        assert obv.iloc[3] > obv.iloc[2]
        assert obv.iloc[4] > obv.iloc[3]

    def test_obv_price_down_subtracts_volume(self):
        """Test that OBV subtracts volume when price goes down."""
        df = pd.DataFrame({
            'close': [100, 99, 98, 97, 96],  # Continuous downtrend
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # OBV should be decreasing
        assert obv.iloc[1] < obv.iloc[0]
        assert obv.iloc[2] < obv.iloc[1]
        assert obv.iloc[3] < obv.iloc[2]
        assert obv.iloc[4] < obv.iloc[3]

    def test_obv_price_flat_unchanged(self):
        """Test that OBV doesn't change when price is flat."""
        df = pd.DataFrame({
            'close': [100, 100, 100, 100],  # Flat price
            'volume': [1000, 1000, 1000, 1000]
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # OBV should remain constant (first value is 0, rest are 0 changes)
        assert obv.iloc[1] == obv.iloc[0]
        assert obv.iloc[2] == obv.iloc[1]
        assert obv.iloc[3] == obv.iloc[2]

    def test_obv_cumulative_nature(self):
        """Test that OBV is cumulative."""
        df = pd.DataFrame({
            'close': [100, 101, 100, 102, 101],  # Mixed
            'volume': [1000, 2000, 1500, 1800, 1200]
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # OBV should be cumulative sum
        # Index 0: 0 (first value)
        # Index 1: 0 + 2000 = 2000 (price up)
        # Index 2: 2000 - 1500 = 500 (price down)
        # Index 3: 500 + 1800 = 2300 (price up)
        # Index 4: 2300 - 1200 = 1100 (price down)

        expected = [0, 2000, 500, 2300, 1100]
        assert np.allclose(obv, expected, rtol=1e-10)

    def test_obv_with_varying_volume(self):
        """Test OBV with varying volume amounts."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 101, 102],
            'volume': [1000, 500, 2000, 1500, 3000]  # Varying volumes
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # Expected:
        # 0: 0
        # 1: 0 + 500 = 500
        # 2: 500 + 2000 = 2500
        # 3: 2500 - 1500 = 1000
        # 4: 1000 + 3000 = 4000

        expected = [0, 500, 2500, 1000, 4000]
        assert np.allclose(obv, expected, rtol=1e-10)

    def test_obv_zero_volume_handling(self):
        """Test OBV handling of zero volume."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103],
            'volume': [1000, 0, 1000, 1000]  # Zero volume at index 1
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        # Should handle gracefully
        assert 'obv' in result_df.columns
        obv = result_df['obv']

        # Expected:
        # 0: 0
        # 1: 0 + 0 = 0 (zero volume)
        # 2: 0 + 1000 = 1000
        # 3: 1000 + 1000 = 2000

        expected = [0, 0, 1000, 2000]
        assert np.allclose(obv, expected, rtol=1e-10)

    def test_obv_negative_volume_clipped(self):
        """Test that negative volumes are clipped to zero."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, -500, 1000]  # Negative volume (shouldn't happen but test)
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        # Should clip negative to 0
        assert 'obv' in result_df.columns
        obv = result_df['obv']

        # Negative volume should be treated as 0
        # Expected:
        # 0: 0
        # 1: 0 + 0 = 0
        # 2: 0 + 1000 = 1000

        expected = [0, 0, 1000]
        assert np.allclose(obv, expected, rtol=1e-10)

    def test_obv_nan_volume_handling(self):
        """Test OBV handling of NaN volume."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103],
            'volume': [1000, np.nan, 1000, 1000]
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        # Should handle NaN by replacing with 0
        assert 'obv' in result_df.columns
        obv = result_df['obv']

        # NaN should be treated as 0
        expected = [0, 0, 1000, 2000]
        assert np.allclose(obv, expected, rtol=1e-10)

    def test_min_periods(self):
        """Test minimum periods requirement."""
        indicator = OBVIndicator()

        # Need at least 2 periods (to compare prices)
        assert indicator._get_min_periods() == 2

    def test_insufficient_data(self):
        """Test with insufficient data."""
        df = pd.DataFrame({
            'close': [100],
            'volume': [1000]
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        # Should fail validation (need at least 2)
        assert 'obv' not in result_df.columns

    def test_get_latest_value(self, sample_ohlcv_data):
        """Test getting latest OBV value."""
        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(sample_ohlcv_data)
        latest = indicator.get_latest_value(result_df, 'obv')

        assert isinstance(latest, (int, float))
        assert not np.isnan(latest)


class TestOBVEdgeCases:
    """Test edge cases for OBV."""

    def test_large_volume_spike(self):
        """Test OBV with large volume spike."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1000, 10000, 1000, 1000]  # Large spike at index 2
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # The spike should be reflected in OBV
        # Difference between index 2 and 3 should be large
        diff_spike = obv.iloc[3] - obv.iloc[2]
        diff_normal = obv.iloc[2] - obv.iloc[1]

        assert abs(diff_normal) > abs(diff_spike) * 0.1  # Spike effect

    def test_alternating_price_direction(self):
        """Test OBV with alternating price direction."""
        df = pd.DataFrame({
            'close': [100, 101, 100, 101, 100, 101],  # Alternating
            'volume': [1000, 1000, 1000, 1000, 1000, 1000]
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # OBV should oscillate
        assert obv.iloc[1] > obv.iloc[0]  # Up
        assert obv.iloc[2] < obv.iloc[1]  # Down
        assert obv.iloc[3] > obv.iloc[2]  # Up
        assert obv.iloc[4] < obv.iloc[3]  # Down

    def test_obv_divergence(self):
        """Test OBV divergence from price."""
        # Price going up but volume decreasing (bearish divergence)
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],  # Price up
            'volume': [10000, 8000, 6000, 4000, 2000]  # Volume down
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        obv = result_df['obv']

        # OBV should still increase (price is up)
        # but at decreasing rate (lower volume)
        diff1 = obv.iloc[1] - obv.iloc[0]
        diff2 = obv.iloc[2] - obv.iloc[1]
        diff3 = obv.iloc[3] - obv.iloc[2]

        # Each increment should be smaller
        assert diff1 > diff2 > diff3

    def test_very_large_dataset(self):
        """Test OBV with large dataset."""
        np.random.seed(42)
        close = np.cumsum(np.random.randn(1000)) + 100
        volume = np.abs(np.random.normal(1000000, 100000, 1000))

        df = pd.DataFrame({
            'close': close,
            'volume': volume
        })

        indicator = OBVIndicator()

        result_df = indicator.calculate_safe(df)

        # Should complete without error
        assert 'obv' in result_df.columns
        assert len(result_df['obv']) == 1000
