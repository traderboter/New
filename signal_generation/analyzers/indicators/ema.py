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

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with EMA columns added
        """
        result_df = df.copy()

        for period in self.periods:
            col_name = f'ema_{period}'

            # Calculate SMA for the first period values
            sma_initial = result_df['close'].rolling(window=period).mean()

            # Calculate EMA using pandas ewm
            # adjust=False means we use recursive formula: EMA[i] = alpha * price[i] + (1-alpha) * EMA[i-1]
            ema = result_df['close'].ewm(span=period, adjust=False).mean()

            # Replace the initial values (before first complete window) with NaN
            # and the first complete window value with SMA
            ema_corrected = ema.copy()
            ema_corrected.iloc[:period-1] = np.nan  # Set early values to NaN
            ema_corrected.iloc[period-1] = sma_initial.iloc[period-1]  # First EMA = SMA

            # Now recalculate EMA from the SMA starting point
            # We need to manually apply the EMA formula after the SMA initialization
            alpha = 2 / (period + 1)
            for i in range(period, len(result_df)):
                ema_corrected.iloc[i] = (result_df['close'].iloc[i] * alpha +
                                        ema_corrected.iloc[i-1] * (1 - alpha))

            result_df[col_name] = ema_corrected

        return result_df
