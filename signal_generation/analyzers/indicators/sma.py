"""
SMA (Simple Moving Average) Indicator

Calculates Simple Moving Average for different periods.
SMA is the average price over a specific number of periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class SMAIndicator(BaseIndicator):
    """
    SMA (Simple Moving Average) indicator calculator.

    Calculates SMA for multiple periods (20, 50, 200 by default).
    SMA is a trend-following indicator that averages prices over a period.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Get periods from config BEFORE calling super().__init__
        # This is needed because _get_output_columns() will be called during parent init
        self.periods = config.get('sma_periods', [20, 50, 200]) if config else [20, 50, 200]

        super().__init__(config)

    def _get_indicator_name(self) -> str:
        return "SMA"

    def _get_indicator_type(self) -> str:
        return "trend"

    def _get_required_columns(self) -> List[str]:
        return ['close']

    def _get_output_columns(self) -> List[str]:
        return [f'sma_{period}' for period in self.periods]

    def _get_min_periods(self) -> int:
        return max(self.periods) if self.periods else 200

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMA for all configured periods.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with SMA columns added
        """
        result_df = df.copy()

        for period in self.periods:
            col_name = f'sma_{period}'
            result_df[col_name] = result_df['close'].rolling(window=period).mean()

        return result_df
