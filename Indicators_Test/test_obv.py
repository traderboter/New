"""
Tests for OBV (On-Balance Volume) Indicator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.obv import OBVIndicator
from Indicators_Test.test_utils import (
    create_sample_ohlcv_data,
    create_trending_data,
    assert_column_exists,
    assert_no_inf
)


class TestOBVIndicator:
    """Test suite for OBV indicator."""

    def test_initialization_default(self):
        """Test OBV initialization with default config."""
        obv = OBVIndicator()
        assert obv.name == "OBV"
        assert obv.indicator_type == "volume"
        assert obv.required_columns == ['close', 'volume']
        assert obv.output_columns == ['obv']

    def test_calculate_basic(self):
        """Test basic OBV calculation."""
        df = create_sample_ohlcv_data(num_rows=100)
        obv = OBVIndicator()

        result_df = obv.calculate(df)

        # Check output column exists
        assert_column_exists(result_df, 'obv')

        # Check no infinite values
        assert_no_inf(result_df, ['obv'])

    def test_obv_price_up_volume_added(self):
        """Test OBV adds volume when price goes up."""
        # Create data where price consistently goes up
        close_prices = [100, 101, 102, 103, 104]
        volumes = [1000, 1000, 1000, 1000, 1000]

        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': volumes
        })

        obv = OBVIndicator()
        result_df = obv.calculate(df)

        # OBV should be cumulative and increasing
        obv_values = result_df['obv'].values

        # First value is 0 (no previous price to compare)
        assert obv_values[0] == 0

        # Each subsequent value should increase by volume
        assert obv_values[1] == 1000  # Price up from 100 to 101
        assert obv_values[2] == 2000  # Price up from 101 to 102
        assert obv_values[3] == 3000  # Price up from 102 to 103
        assert obv_values[4] == 4000  # Price up from 103 to 104

    def test_obv_price_down_volume_subtracted(self):
        """Test OBV subtracts volume when price goes down."""
        # Create data where price consistently goes down
        close_prices = [104, 103, 102, 101, 100]
        volumes = [1000, 1000, 1000, 1000, 1000]

        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': volumes
        })

        obv = OBVIndicator()
        result_df = obv.calculate(df)

        obv_values = result_df['obv'].values

        # First value is 0
        assert obv_values[0] == 0

        # Each subsequent value should decrease by volume
        assert obv_values[1] == -1000  # Price down from 104 to 103
        assert obv_values[2] == -2000  # Price down from 103 to 102
        assert obv_values[3] == -3000  # Price down from 102 to 101
        assert obv_values[4] == -4000  # Price down from 101 to 100

    def test_obv_price_unchanged(self):
        """Test OBV unchanged when price doesn't change."""
        # Create data where price stays the same
        close_prices = [100, 100, 100, 100, 100]
        volumes = [1000, 1000, 1000, 1000, 1000]

        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': volumes
        })

        obv = OBVIndicator()
        result_df = obv.calculate(df)

        obv_values = result_df['obv'].values

        # All OBV values should be 0 (no change)
        assert np.allclose(obv_values, 0)

    def test_obv_mixed_price_movement(self):
        """Test OBV with mixed price movements."""
        # Up, Up, Down, Up, Down
        close_prices = [100, 102, 104, 103, 105, 104]
        volumes = [1000, 2000, 1500, 1000, 2500, 1200]

        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': volumes
        })

        obv = OBVIndicator()
        result_df = obv.calculate(df)

        obv_values = result_df['obv'].values

        # Calculate expected OBV manually
        expected = [
            0,              # First value
            2000,           # +2000 (price up)
            3500,           # +1500 (price up)
            2500,           # -1000 (price down)
            5000,           # +2500 (price up)
            3800            # -1200 (price down)
        ]

        assert np.allclose(obv_values, expected)

    def test_obv_with_zero_volume(self):
        """Test OBV handles zero volume correctly."""
        close_prices = [100, 101, 102, 103]
        volumes = [1000, 0, 1000, 1000]  # Zero volume in second period

        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': volumes
        })

        obv = OBVIndicator()
        result_df = obv.calculate(df)

        # Should not crash
        assert_column_exists(result_df, 'obv')
        assert_no_inf(result_df, ['obv'])

    def test_obv_with_nan_volume(self):
        """Test OBV handles NaN volume correctly."""
        close_prices = [100, 101, 102, 103]
        volumes = [1000, np.nan, 1000, 1000]

        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': volumes
        })

        obv = OBVIndicator()
        result_df = obv.calculate(df)

        # Should handle NaN gracefully
        assert_column_exists(result_df, 'obv')
        assert_no_inf(result_df, ['obv'])

    def test_obv_uptrend_positive(self):
        """Test OBV is generally positive in uptrend."""
        df = create_trending_data(num_rows=100, trend='up')
        obv = OBVIndicator()

        result_df = obv.calculate(df)

        # In uptrend, OBV should generally increase
        obv_final = result_df['obv'].iloc[-1]
        obv_start = result_df['obv'].iloc[10]  # Skip first few

        assert obv_final > obv_start

    def test_obv_downtrend_negative(self):
        """Test OBV is generally negative in downtrend."""
        df = create_trending_data(num_rows=100, trend='down')
        obv = OBVIndicator()

        result_df = obv.calculate(df)

        # In downtrend, OBV should generally decrease
        obv_final = result_df['obv'].iloc[-1]
        obv_start = result_df['obv'].iloc[10]

        assert obv_final < obv_start

    def test_calculate_safe_with_insufficient_data(self):
        """Test calculate_safe with insufficient data."""
        df = pd.DataFrame({
            'close': [100],
            'volume': [1000]
        })
        obv = OBVIndicator()  # Needs at least 2 periods

        result_df = obv.calculate_safe(df)
        assert 'obv' not in result_df.columns

    def test_calculate_safe_with_missing_columns(self):
        """Test calculate_safe with missing required columns."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5]
            # Missing 'close' and 'volume'
        })
        obv = OBVIndicator()

        result_df = obv.calculate_safe(df)
        assert 'obv' not in result_df.columns

    def test_calculate_safe_with_empty_dataframe(self):
        """Test calculate_safe with empty DataFrame."""
        df = pd.DataFrame()
        obv = OBVIndicator()

        result_df = obv.calculate_safe(df)
        assert len(result_df) == 0

    def test_get_values(self):
        """Test get_values method."""
        df = create_sample_ohlcv_data(num_rows=100)
        obv = OBVIndicator()

        result_df = obv.calculate(df)

        values = obv.get_values(result_df, 'obv')
        assert isinstance(values, pd.Series)
        assert len(values) == len(df)

    def test_get_latest_value(self):
        """Test get_latest_value method."""
        df = create_sample_ohlcv_data(num_rows=100)
        obv = OBVIndicator()

        result_df = obv.calculate(df)

        latest = obv.get_latest_value(result_df, 'obv')
        assert isinstance(latest, float)
        assert not np.isnan(latest)

    def test_obv_cumulative_nature(self):
        """Test that OBV is cumulative."""
        df = create_sample_ohlcv_data(num_rows=50, seed=42)
        obv = OBVIndicator()

        result_df = obv.calculate(df)

        # OBV should be cumulative sum of signed volume
        # Verify by manually calculating
        price_direction = np.sign(df['close'].diff())
        volume = df['volume'].fillna(0).clip(lower=0)
        signed_volume = volume * price_direction
        signed_volume = signed_volume.replace([np.inf, -np.inf], 0).fillna(0)
        expected_obv = signed_volume.cumsum()

        # Compare
        diff = (result_df['obv'] - expected_obv).abs()
        assert diff.max() < 1e-10

    def test_obv_with_negative_volume(self):
        """Test OBV handles negative volume correctly."""
        close_prices = [100, 101, 102]
        volumes = [1000, -500, 1000]  # Negative volume (should be clipped to 0)

        df = pd.DataFrame({
            'open': close_prices,
            'high': [x + 1 for x in close_prices],
            'low': [x - 1 for x in close_prices],
            'close': close_prices,
            'volume': volumes
        })

        obv = OBVIndicator()
        result_df = obv.calculate(df)

        # Should handle negative volume by clipping to 0
        assert_column_exists(result_df, 'obv')
        assert_no_inf(result_df, ['obv'])

    def test_caching(self):
        """Test that caching works correctly."""
        df = create_sample_ohlcv_data(num_rows=100)
        config = {'cache_enabled': True}
        obv = OBVIndicator(config)

        # First calculation
        result_df1 = obv.calculate_safe(df)

        # Second calculation (should use cache)
        result_df2 = obv.calculate_safe(df)

        # Results should be identical
        pd.testing.assert_frame_equal(result_df1, result_df2)

        # Clear cache
        obv.clear_cache()
        result_df3 = obv.calculate_safe(df)

        # Should still be the same
        pd.testing.assert_frame_equal(result_df1, result_df3)

    def test_string_representation(self):
        """Test string representations."""
        obv = OBVIndicator()

        str_repr = str(obv)
        assert "OBV" in str_repr
        assert "volume" in str_repr

        repr_str = repr(obv)
        assert "OBVIndicator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
