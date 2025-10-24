"""
MACD (Moving Average Convergence Divergence) Indicator

Calculates MACD, Signal line, and Histogram.
MACD is a trend-following momentum indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class MACDIndicator(BaseIndicator):
    """
    MACD (Moving Average Convergence Divergence) indicator calculator.

    MACD shows the relationship between two moving averages.
    Components:
    - MACD Line: (12-period EMA - 26-period EMA)
    - Signal Line: 9-period EMA of MACD Line
    - Histogram: MACD Line - Signal Line
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Get parameters from config
        self.fast_period = config.get('macd_fast', 12) if config else 12
        self.slow_period = config.get('macd_slow', 26) if config else 26
        self.signal_period = config.get('macd_signal', 9) if config else 9

    def _get_indicator_name(self) -> str:
        return "MACD"

    def _get_indicator_type(self) -> str:
        return "momentum"

    def _get_required_columns(self) -> List[str]:
        return ['close']

    def _get_output_columns(self) -> List[str]:
        return ['macd', 'macd_signal', 'macd_hist']

    def _get_min_periods(self) -> int:
        return self.slow_period + self.signal_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD, Signal, and Histogram.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with MACD columns added
        """
        result_df = df.copy()

        # Calculate fast and slow EMAs
        ema_fast = result_df['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = result_df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        result_df['macd'] = ema_fast - ema_slow

        # Calculate Signal line
        result_df['macd_signal'] = result_df['macd'].ewm(span=self.signal_period, adjust=False).mean()

        # Calculate Histogram
        result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']

        return result_df
