"""
Engulfing Pattern Detector

Detects Bullish and Bearish Engulfing candlestick patterns using TALib.
Engulfing patterns are strong reversal patterns.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class EngulfingPattern(BasePattern):
    """
    Engulfing candlestick pattern detector.

    Characteristics:
    - Can be bullish or bearish
    - Second candle completely engulfs first candle's body
    - Strong reversal signal
    - Direction determined by engulfing candle color

    Strength: 3/3 (Strong)
    """

    def _get_pattern_name(self) -> str:
        return "Engulfing"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "both"  # Can be bullish or bearish

    def _get_base_strength(self) -> int:
        return 3  # Strong pattern

    def detect(
        self,
        df: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> bool:
        """Detect Engulfing pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        try:
            # Use TALib to detect
            result = talib.CDLENGULFING(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            # Check last candle
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
            # Use TALib result to determine direction
            result = talib.CDLENGULFING(
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
        """Get additional details about Engulfing detection."""
        if len(df) < 2:
            return super()._get_detection_details(df)

        # Get last two candles
        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]

        # Calculate body sizes
        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])

        # Engulfing ratio (how much bigger is current body)
        engulfing_ratio = curr_body / prev_body if prev_body > 0 else 1

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.75 + (engulfing_ratio / 10), 0.95),
            'metadata': {
                'prev_body_size': float(prev_body),
                'curr_body_size': float(curr_body),
                'engulfing_ratio': float(engulfing_ratio)
            }
        }
