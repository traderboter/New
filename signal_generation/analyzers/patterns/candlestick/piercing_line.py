"""
Piercing Line Pattern Detector

Detects Piercing Line candlestick pattern using TALib.
Piercing Line is a bullish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class PiercingLinePattern(BasePattern):
    """
    Piercing Line candlestick pattern detector.

    Characteristics:
    - Bullish reversal pattern (2 candles)
    - First candle: Bearish
    - Second candle: Bullish, opens below previous low, closes above midpoint of first

    Strength: 2/3 (Medium)
    """

    def _get_pattern_name(self) -> str:
        return "Piercing Line"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "bullish"

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
        """Detect Piercing Line pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        if len(df) < 2:
            return False

        try:
            result = talib.CDLPIERCING(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Piercing Line detection."""
        if len(df) < 2:
            return super()._get_detection_details(df)

        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]

        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])

        # How far into previous candle's body
        prev_midpoint = (prev_candle['open'] + prev_candle['close']) / 2
        penetration = (curr_candle['close'] - prev_midpoint) / prev_body if prev_body > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.70 + (penetration / 5), 0.95),
            'metadata': {
                'prev_body': float(prev_body),
                'curr_body': float(curr_body),
                'penetration_ratio': float(penetration)
            }
        }
