"""
Pytest configuration and fixtures for indicator tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """
    Generate sample OHLCV data for testing.
    Returns a DataFrame with 100 rows of realistic price data.
    """
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

    # Generate realistic price data with trend
    base_price = 50000
    trend = np.linspace(0, 2000, 100)  # Upward trend
    noise = np.random.normal(0, 500, 100)

    close = base_price + trend + noise

    # Generate OHLCV from close
    high = close + np.abs(np.random.normal(100, 50, 100))
    low = close - np.abs(np.random.normal(100, 50, 100))
    open_price = close + np.random.normal(0, 100, 100)
    volume = np.abs(np.random.normal(1000000, 200000, 100))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


@pytest.fixture
def small_ohlcv_data():
    """
    Generate small OHLCV dataset for edge case testing.
    Returns a DataFrame with only 10 rows.
    """
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')

    close = [100, 102, 101, 105, 103, 107, 106, 110, 108, 112]

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [99, 100, 102, 101, 105, 103, 107, 106, 110, 108],
        'high': [103, 104, 103, 107, 106, 109, 108, 112, 111, 114],
        'low': [98, 99, 100, 100, 102, 102, 105, 105, 107, 107],
        'close': close,
        'volume': [1000, 1200, 900, 1500, 1100, 1300, 1000, 1400, 1200, 1600]
    })

    return df


@pytest.fixture
def flat_price_data():
    """
    Generate flat price data (no movement) for edge case testing.
    """
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [100] * 50,
        'high': [100] * 50,
        'low': [100] * 50,
        'close': [100] * 50,
        'volume': [1000] * 50
    })

    return df


@pytest.fixture
def empty_dataframe():
    """
    Generate empty DataFrame for error handling tests.
    """
    return pd.DataFrame()


@pytest.fixture
def config_default():
    """
    Default configuration for indicators.
    """
    return {
        'cache_enabled': True,
        # EMA/SMA
        'ema_periods': [20, 50],
        'sma_periods': [20, 50],
        # RSI
        'rsi_period': 14,
        # MACD
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        # Stochastic
        'stoch_k': 14,
        'stoch_d': 3,
        'stoch_smooth': 3,
        # ATR
        'atr_period': 14,
        # Bollinger Bands
        'bb_period': 20,
        'bb_std': 2.0
    }


@pytest.fixture
def config_custom():
    """
    Custom configuration for testing different parameters.
    """
    return {
        'cache_enabled': False,
        'ema_periods': [10],
        'sma_periods': [10],
        'rsi_period': 7,
        'macd_fast': 5,
        'macd_slow': 13,
        'macd_signal': 5,
        'stoch_k': 7,
        'stoch_d': 2,
        'stoch_smooth': 2,
        'atr_period': 7,
        'bb_period': 10,
        'bb_std': 1.5
    }
