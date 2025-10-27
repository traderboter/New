"""
EMA (Exponential Moving Average) Indicator

Calculates Exponential Moving Average for different periods.
EMA gives more weight to recent prices.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class EMAIndicator(BaseIndicator):
    """
    EMA (Exponential Moving Average) indicator calculator.

    Calculates EMA for multiple periods (20, 50, 200 by default).
    EMA is a trend-following indicator that gives more weight to recent prices.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Get periods from config BEFORE calling super().__init__
        # This is needed because _get_output_columns() will be called during parent init
        self.periods = config.get('ema_periods', [20, 50, 200]) if config else [20, 50, 200]

        super().__init__(config)

    def _get_indicator_name(self) -> str:
        return "EMA"

    def _get_indicator_type(self) -> str:
        return "trend"

    def _get_required_columns(self) -> List[str]:
        return ['close']

    def _get_output_columns(self) -> List[str]:
        return [f'ema_{period}' for period in self.periods]

    def _get_min_periods(self) -> int:
        return max(self.periods) if self.periods else 200

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA for all configured periods.

        EMA is calculated the same way as TA-Lib:
        1. First EMA value = SMA of first N periods
        2. Subsequent values: EMA = price * alpha + prev_EMA * (1-alpha)
           where alpha = 2 / (period + 1)

        Optimized version using numpy arrays for fast computation.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with EMA columns added
        """
        result_df = df.copy()

        for period in self.periods:
            col_name = f'ema_{period}'

            # Convert to numpy array for fast computation
            close_values = result_df['close'].values
            n = len(close_values)

            # Initialize EMA array
            ema_values = np.empty(n)
            ema_values[:period-1] = np.nan

            # First EMA value = SMA of first N periods
            ema_values[period-1] = np.mean(close_values[:period])

            # Calculate remaining EMA values using vectorized operations
            # This is much faster than using iloc in a loop
            alpha = 2.0 / (period + 1)
            for i in range(period, n):
                ema_values[i] = alpha * close_values[i] + (1 - alpha) * ema_values[i-1]

            result_df[col_name] = ema_values

        return result_df
