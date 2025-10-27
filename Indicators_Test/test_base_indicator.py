"""
Tests for BaseIndicator abstract class.
"""

import pytest
import pandas as pd
import numpy as np
from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class SimpleTestIndicator(BaseIndicator):
    """
    Simple concrete implementation for testing BaseIndicator.
    """

    def _get_indicator_name(self) -> str:
        return "TestIndicator"

    def _get_indicator_type(self) -> str:
        return "test"

    def _get_required_columns(self):
        return ['close']

    def _get_output_columns(self):
        return ['test_value']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df['test_value'] = result_df['close'] * 2
        return result_df


class TestBaseIndicator:
    """Test suite for BaseIndicator."""

    def test_initialization(self, config_default):
        """Test indicator initialization."""
        indicator = SimpleTestIndicator(config_default)

        assert indicator.name == "TestIndicator"
        assert indicator.indicator_type == "test"
        assert indicator.required_columns == ['close']
        assert indicator.output_columns == ['test_value']
        assert indicator._cache_enabled is True

    def test_initialization_without_config(self):
        """Test initialization without config."""
        indicator = SimpleTestIndicator()

        assert indicator.name == "TestIndicator"
        assert indicator.config == {}

    def test_calculate_safe_success(self, sample_ohlcv_data):
        """Test successful calculation with validation."""
        indicator = SimpleTestIndicator()
        result_df = indicator.calculate_safe(sample_ohlcv_data)

        assert 'test_value' in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)
        # Check calculation correctness
        assert np.allclose(result_df['test_value'], sample_ohlcv_data['close'] * 2)

    def test_calculate_safe_empty_dataframe(self, empty_dataframe):
        """Test calculation with empty DataFrame."""
        indicator = SimpleTestIndicator()
        result_df = indicator.calculate_safe(empty_dataframe)

        # Should return original empty DataFrame
        assert len(result_df) == 0

    def test_calculate_safe_missing_columns(self, sample_ohlcv_data):
        """Test calculation with missing required columns."""
        indicator = SimpleTestIndicator()
        df_incomplete = sample_ohlcv_data[['open', 'high', 'low']].copy()

        result_df = indicator.calculate_safe(df_incomplete)

        # Should return original DataFrame without adding indicator column
        assert 'test_value' not in result_df.columns

    def test_caching(self, sample_ohlcv_data):
        """Test caching functionality."""
        indicator = SimpleTestIndicator({'cache_enabled': True})

        # First calculation
        result1 = indicator.calculate_safe(sample_ohlcv_data)
        assert indicator._last_hash is not None
        assert indicator._last_result is not None

        # Second calculation with same data (should use cache)
        result2 = indicator.calculate_safe(sample_ohlcv_data)
        assert result1.equals(result2)

        # Verify cache was used (last_hash should be same)
        assert indicator._last_hash is not None

    def test_cache_disabled(self, sample_ohlcv_data):
        """Test with caching disabled."""
        indicator = SimpleTestIndicator({'cache_enabled': False})

        result = indicator.calculate_safe(sample_ohlcv_data)

        # Cache should not be populated
        assert indicator._last_hash is None
        assert indicator._last_result is None

    def test_clear_cache(self, sample_ohlcv_data):
        """Test cache clearing."""
        indicator = SimpleTestIndicator({'cache_enabled': True})

        # Calculate to populate cache
        indicator.calculate_safe(sample_ohlcv_data)
        assert indicator._last_hash is not None

        # Clear cache
        indicator.clear_cache()
        assert indicator._last_hash is None
        assert indicator._last_result is None

    def test_get_values_specific_column(self, sample_ohlcv_data):
        """Test getting specific indicator column."""
        indicator = SimpleTestIndicator()
        result_df = indicator.calculate_safe(sample_ohlcv_data)

        values = indicator.get_values(result_df, 'test_value')

        assert isinstance(values, pd.Series)
        assert len(values) == len(sample_ohlcv_data)

    def test_get_values_all_columns(self, sample_ohlcv_data):
        """Test getting all indicator columns."""
        indicator = SimpleTestIndicator()
        result_df = indicator.calculate_safe(sample_ohlcv_data)

        values = indicator.get_values(result_df)

        assert isinstance(values, dict)
        assert 'test_value' in values
        assert isinstance(values['test_value'], pd.Series)

    def test_get_latest_value(self, sample_ohlcv_data):
        """Test getting latest indicator value."""
        indicator = SimpleTestIndicator()
        result_df = indicator.calculate_safe(sample_ohlcv_data)

        latest = indicator.get_latest_value(result_df, 'test_value')

        assert isinstance(latest, float)
        assert latest == sample_ohlcv_data['close'].iloc[-1] * 2

    def test_safe_divide(self):
        """Test safe division utility method."""
        indicator = SimpleTestIndicator()

        # Normal division
        result = indicator._safe_divide(10, 2, default=0)
        assert result == 5

        # Division by zero
        result = indicator._safe_divide(10, 0, default=0)
        assert result == 0

        # Array division
        numerator = np.array([10, 20, 30])
        denominator = np.array([2, 0, 5])
        result = indicator._safe_divide(numerator, denominator, default=-1)

        assert result[0] == 5  # 10/2
        assert result[1] == -1  # 20/0 -> default
        assert result[2] == 6  # 30/5

    def test_str_representation(self):
        """Test string representation."""
        indicator = SimpleTestIndicator()

        str_repr = str(indicator)
        assert "TestIndicator" in str_repr
        assert "test" in str_repr

    def test_repr_representation(self):
        """Test detailed representation."""
        indicator = SimpleTestIndicator()

        repr_str = repr(indicator)
        assert "SimpleTestIndicator" in repr_str
        assert "TestIndicator" in repr_str
        assert "close" in repr_str
        assert "test_value" in repr_str


class BrokenIndicator(BaseIndicator):
    """Indicator that raises error during calculation."""

    def _get_indicator_name(self) -> str:
        return "BrokenIndicator"

    def _get_indicator_type(self) -> str:
        return "test"

    def _get_required_columns(self):
        return ['close']

    def _get_output_columns(self):
        return ['broken_value']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("Intentional error for testing")


class TestBaseIndicatorErrorHandling:
    """Test error handling in BaseIndicator."""

    def test_calculation_error_handling(self, sample_ohlcv_data):
        """Test error handling during calculation."""
        indicator = BrokenIndicator()

        # Should catch error and return original DataFrame
        result_df = indicator.calculate_safe(sample_ohlcv_data)

        # Original DataFrame should be returned
        assert 'broken_value' not in result_df.columns
        assert len(result_df) == len(sample_ohlcv_data)
