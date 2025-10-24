"""
Dark Cloud Cover Pattern Detector

Detects Dark Cloud Cover candlestick pattern using TALib.
Dark Cloud Cover is a bearish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class DarkCloudCoverPattern(BasePattern):
    """
    Dark Cloud Cover candlestick pattern detector.

    Characteristics:
    - Bearish reversal pattern (2 candles)
    - First candle: Bullish
    - Second candle: Bearish, opens above previous high, closes below midpoint of first

    Strength: 2/3 (Medium)
    """

    def _get_pattern_name(self) -> str:
        return "Dark Cloud Cover"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "bearish"

    def _get_base_strength(self) -> int:
        return 2

    def detect(
        self,
        df: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> bool:
        """Detect Dark Cloud Cover pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        if len(df) < 2:
            return False

        try:
            result = talib.CDLDARKCLOUDCOVER(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Dark Cloud Cover detection."""
        if len(df) < 2:
            return super()._get_detection_details(df)

        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]

        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])
        prev_full_range = prev_candle['high'] - prev_candle['low']
        curr_full_range = curr_candle['high'] - curr_candle['low']

        # How far into previous candle's body
        # Use safe division: minimum threshold is 30% of candle's full range
        safe_prev_body = max(prev_body, prev_full_range * 0.3) if prev_full_range > 0 else 0.0001
        prev_midpoint = (prev_candle['open'] + prev_candle['close']) / 2
        penetration = (prev_midpoint - curr_candle['close']) / safe_prev_body if safe_prev_body > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.70 + (penetration / 5), 0.95),
            'metadata': {
                'prev_body': float(prev_body),
                'curr_body': float(curr_body),
                'prev_full_range': float(prev_full_range),
                'curr_full_range': float(curr_full_range),
                'penetration_ratio': float(penetration)
            }
        }
