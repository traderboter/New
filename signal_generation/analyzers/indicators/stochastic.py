"""
Stochastic Oscillator Indicator

Calculates Stochastic %K and %D lines.
Stochastic compares a closing price to its price range over a period.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from signal_generation.analyzers.indicators.base_indicator import BaseIndicator


class StochasticIndicator(BaseIndicator):
    """
    Stochastic Oscillator indicator calculator.

    Stochastic shows the location of the close relative to the high-low range over a period.
    Components:
    - %K: Fast stochastic (raw value)
    - %D: Slow stochastic (moving average of %K)

    Values range from 0 to 100:
    - Above 80: Overbought
    - Below 20: Oversold
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Get parameters from config
        self.k_period = config.get('stoch_k', 14) if config else 14
        self.d_period = config.get('stoch_d', 3) if config else 3
        self.smooth_k = config.get('stoch_smooth', 3) if config else 3

    def _get_indicator_name(self) -> str:
        return "Stochastic"

    def _get_indicator_type(self) -> str:
        return "momentum"

    def _get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def _get_output_columns(self) -> List[str]:
        return ['stoch_k', 'stoch_d']

    def _get_min_periods(self) -> int:
        return self.k_period + self.smooth_k + self.d_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic %K and %D.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with Stochastic columns added
        """
        result_df = df.copy()

        # Calculate lowest low and highest high over k_period
        low_min = result_df['low'].rolling(window=self.k_period).min()
        high_max = result_df['high'].rolling(window=self.k_period).max()

        # Calculate the range (high - low)
        range_hl = high_max - low_min

        # Calculate raw %K with safe division
        # When range is 0 (flat price), use 50 as neutral value
        raw_k = 100 * self._safe_divide(
            result_df['close'] - low_min,
            range_hl,
            0.5  # 0.5 * 100 = 50 (neutral value)
        )

        # Ensure raw_k is within valid range [0, 100]
        raw_k = raw_k.clip(0, 100)

        # Smooth %K
        result_df['stoch_k'] = raw_k.rolling(window=self.smooth_k).mean()

        # Calculate %D (moving average of %K)
        result_df['stoch_d'] = result_df['stoch_k'].rolling(window=self.d_period).mean()

        return result_df
