"""
Hanging Man Pattern Detector

Detects Hanging Man candlestick pattern using TALib.
Hanging Man is a bearish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class HangingManPattern(BasePattern):
    """
    Hanging Man candlestick pattern detector.

    Characteristics:
    - Bearish reversal pattern
    - Small body at top of candle
    - Long lower shadow (at least 2x body)
    - Little to no upper shadow
    - Best when appears after uptrend

    Strength: 2/3 (Medium)
    """

    def _get_pattern_name(self) -> str:
        return "Hanging Man"

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
        """Detect Hanging Man pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        try:
            result = talib.CDLHANGINGMAN(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Hanging Man detection."""
        last_candle = df.iloc[-1]

        body_size = abs(last_candle['close'] - last_candle['open'])
        lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])

        shadow_ratio = lower_shadow / body_size if body_size > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.70 + (shadow_ratio / 10), 0.95),
            'metadata': {
                'body_size': float(body_size),
                'lower_shadow': float(lower_shadow),
                'upper_shadow': float(upper_shadow),
                'shadow_ratio': float(shadow_ratio)
            }
        }
