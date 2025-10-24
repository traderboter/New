"""
OBV (On-Balance Volume) Indicator

Calculates OBV volume indicator.
OBV relates volume to price change to predict price movements.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class OBVIndicator(BaseIndicator):
    """
    OBV (On-Balance Volume) indicator calculator.

    OBV is a momentum indicator that uses volume flow to predict changes in price.
    - When close > previous close: Add volume to OBV
    - When close < previous close: Subtract volume from OBV
    - When close = previous close: OBV unchanged
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def _get_indicator_name(self) -> str:
        return "OBV"

    def _get_indicator_type(self) -> str:
        return "volume"

    def _get_required_columns(self) -> List[str]:
        return ['close', 'volume']

    def _get_output_columns(self) -> List[str]:
        return ['obv']

    def _get_min_periods(self) -> int:
        return 2  # Need at least 2 periods to compare close prices

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OBV.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with OBV column added
        """
        result_df = df.copy()

        # Calculate price direction
        price_direction = np.sign(result_df['close'].diff())

        # OBV is cumulative sum of (volume * direction)
        result_df['obv'] = (result_df['volume'] * price_direction).cumsum()

        return result_df
