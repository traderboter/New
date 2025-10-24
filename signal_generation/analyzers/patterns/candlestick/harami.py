"""
Harami Pattern Detector

Detects Harami candlestick pattern using TALib.
Harami is a reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class HaramiPattern(BasePattern):
    """
    Harami candlestick pattern detector.

    Characteristics:
    - Reversal pattern (2 candles)
    - First candle: Large body
    - Second candle: Small body completely within first candle's body
    - Can be bullish or bearish

    Strength: 2/3 (Medium)
    """

    def _get_pattern_name(self) -> str:
        return "Harami"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "reversal"  # Can be bullish or bearish

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
        """Detect Harami pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        if len(df) < 2:
            return False

        try:
            result = talib.CDLHARAMI(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_actual_direction(
        self,
        df: pd.DataFrame,
        detection_details: Dict[str, Any]
    ) -> str:
        """Determine actual direction (bullish or bearish)."""
        try:
            result = talib.CDLHARAMI(
                df['open'].values,
                df['high'].values,
                df['low'].values,
                df['close'].values
            )

            # Positive = bullish, negative = bearish
            return 'bullish' if result[-1] > 0 else 'bearish'

        except Exception:
            return 'bullish'

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Harami detection."""
        if len(df) < 2:
            return super()._get_detection_details(df)

        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]

        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])

        # Second body should be much smaller
        size_ratio = curr_body / prev_body if prev_body > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.70 - (size_ratio * 2), 0.90),  # Smaller second candle = higher confidence
            'metadata': {
                'prev_body': float(prev_body),
                'curr_body': float(curr_body),
                'size_ratio': float(size_ratio)
            }
        }
