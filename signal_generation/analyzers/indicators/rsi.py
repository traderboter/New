"""
RSI (Relative Strength Index) Indicator

Calculates RSI momentum oscillator.
RSI measures the magnitude of recent price changes to evaluate overbought/oversold conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class RSIIndicator(BaseIndicator):
    """
    RSI (Relative Strength Index) indicator calculator.

    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100.
    - Above 70: Overbought
    - Below 30: Oversold
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Get period from config (default: 14)
        self.period = config.get('rsi_period', 14) if config else 14

    def _get_indicator_name(self) -> str:
        return "RSI"

    def _get_indicator_type(self) -> str:
        return "momentum"

    def _get_required_columns(self) -> List[str]:
        return ['close']

    def _get_output_columns(self) -> List[str]:
        return ['rsi']

    def _get_min_periods(self) -> int:
        return self.period + 1

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI using Wilder's smoothing method (EMA).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with RSI column added
        """
        result_df = df.copy()

        # Calculate price changes
        delta = result_df['close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss using EMA (Wilder's method)
        # Wilder's smoothing is equivalent to EMA with alpha = 1/period
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()

        # Calculate RS (Relative Strength) with safe division
        rs = self._safe_divide(avg_gain, avg_loss, 0)

        # Calculate RSI
        result_df['rsi'] = 100 - (100 / (1 + rs))

        # Ensure RSI is within valid range [0, 100]
        # Set invalid values to NaN
        result_df['rsi'] = result_df['rsi'].where(
            result_df['rsi'].between(0, 100, inclusive='both'),
            np.nan
        )

        return result_df
