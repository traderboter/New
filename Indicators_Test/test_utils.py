"""
Test Utilities - Helper functions for indicator testing
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def create_sample_ohlcv_data(num_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create sample OHLCV (Open, High, Low, Close, Volume) data for testing.

    Args:
        num_rows: Number of rows to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV columns
    """
    np.random.seed(seed)

    # Generate base close prices with trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, num_rows)
    volatility = np.random.randn(num_rows) * 2
    close = base_price + trend + volatility
    close = np.maximum(close, 1)  # Ensure positive prices

    # Generate OHLC from close
    high = close + np.abs(np.random.randn(num_rows)) * 1.5
    low = close - np.abs(np.random.randn(num_rows)) * 1.5
    open_price = close + np.random.randn(num_rows) * 1

    # Ensure high >= low and price constraints
    high = np.maximum(high, low + 0.01)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Generate volume
    volume = np.random.randint(1000, 10000, size=num_rows)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


def create_trending_data(num_rows: int = 100, trend: str = 'up') -> pd.DataFrame:
    """
    Create data with specific trend pattern.

    Args:
        num_rows: Number of rows
        trend: 'up', 'down', or 'sideways'

    Returns:
        DataFrame with OHLCV data
    """
    if trend == 'up':
        close = np.linspace(100, 150, num_rows)
    elif trend == 'down':
        close = np.linspace(150, 100, num_rows)
    else:  # sideways
        close = np.ones(num_rows) * 125

    # Add small noise
    close = close + np.random.randn(num_rows) * 0.5

    high = close + np.abs(np.random.randn(num_rows)) * 0.5
    low = close - np.abs(np.random.randn(num_rows)) * 0.5
    open_price = close + np.random.randn(num_rows) * 0.3
    volume = np.random.randint(1000, 5000, size=num_rows)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


def create_flat_data(num_rows: int = 100, price: float = 100.0) -> pd.DataFrame:
    """
    Create completely flat price data (for edge case testing).

    Args:
        num_rows: Number of rows
        price: Constant price

    Returns:
        DataFrame with flat OHLCV data
    """
    df = pd.DataFrame({
        'open': [price] * num_rows,
        'high': [price] * num_rows,
        'low': [price] * num_rows,
        'close': [price] * num_rows,
        'volume': [1000] * num_rows
    })

    return df


def create_spike_data(num_rows: int = 100, spike_position: int = 50) -> pd.DataFrame:
    """
    Create data with a price spike (for testing volatility indicators).

    Args:
        num_rows: Number of rows
        spike_position: Position of the spike

    Returns:
        DataFrame with spike
    """
    close = np.ones(num_rows) * 100
    close[spike_position] = 150  # Spike up

    high = close + 1
    low = close - 1
    open_price = close.copy()
    volume = np.ones(num_rows) * 1000

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


def assert_column_exists(df: pd.DataFrame, column: str):
    """Assert that a column exists in DataFrame."""
    assert column in df.columns, f"Column '{column}' not found in DataFrame"


def assert_no_inf(df: pd.DataFrame, columns: List[str]):
    """Assert that specified columns contain no infinite values."""
    for col in columns:
        assert not np.isinf(df[col]).any(), f"Column '{col}' contains infinite values"


def assert_in_range(df: pd.DataFrame, column: str, min_val: float, max_val: float, allow_nan: bool = True):
    """
    Assert that column values are within specified range.

    Args:
        df: DataFrame
        column: Column name
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_nan: Whether to allow NaN values
    """
    values = df[column].dropna() if allow_nan else df[column]
    assert values.min() >= min_val, f"Column '{column}' has values below {min_val}"
    assert values.max() <= max_val, f"Column '{column}' has values above {max_val}"


def assert_values_match_expected(actual: pd.Series, expected: pd.Series, tolerance: float = 1e-6):
    """
    Assert that two series match within tolerance.

    Args:
        actual: Actual values
        expected: Expected values
        tolerance: Allowed difference
    """
    # Compare non-NaN values
    mask = ~(actual.isna() | expected.isna())
    if mask.any():
        diff = np.abs(actual[mask] - expected[mask])
        max_diff = diff.max()
        assert max_diff <= tolerance, f"Maximum difference {max_diff} exceeds tolerance {tolerance}"

    # Check NaN positions match
    assert (actual.isna() == expected.isna()).all(), "NaN positions don't match"
