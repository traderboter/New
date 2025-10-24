"""
Shooting Star Pattern Detector

Detects Shooting Star candlestick pattern using TALib.
Shooting Star is a bearish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class ShootingStarPattern(BasePattern):
    """
    Shooting Star candlestick pattern detector.

    Characteristics:
    - Bearish reversal pattern
    - Small body at bottom of candle
    - Long upper shadow (at least 2x body)
    - Little to no lower shadow
    - Best when appears after uptrend

    Strength: 2/3 (Medium)
    """

    def _get_pattern_name(self) -> str:
        return "Shooting Star"

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
        """Detect Shooting Star pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        try:
            result = talib.CDLSHOOTINGSTAR(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Shooting Star detection."""
        last_candle = df.iloc[-1]

        body_size = abs(last_candle['close'] - last_candle['open'])
        lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        full_range = last_candle['high'] - last_candle['low']

        # Use full_range instead of body_size to avoid zero/very small body issues
        shadow_ratio = upper_shadow / full_range if full_range > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.70 + (shadow_ratio / 10), 0.95),
            'metadata': {
                'body_size': float(body_size),
                'lower_shadow': float(lower_shadow),
                'upper_shadow': float(upper_shadow),
                'full_range': float(full_range),
                'shadow_ratio': float(shadow_ratio)
            }
        }
