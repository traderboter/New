"""
Inverted Hammer Pattern Detector

Detects Inverted Hammer candlestick pattern using TALib.
Inverted Hammer is a bullish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class InvertedHammerPattern(BasePattern):
    """
    Inverted Hammer candlestick pattern detector.

    Characteristics:
    - Bullish reversal pattern
    - Small body at bottom of candle
    - Long upper shadow (at least 2x body)
    - Little to no lower shadow
    - Best when appears after downtrend

    Strength: 2/3 (Medium)
    """

    def _get_pattern_name(self) -> str:
        return "Inverted Hammer"

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
        """Detect Inverted Hammer pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        try:
            # Use TALib to detect
            result = talib.CDLINVERTEDHAMMER(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            # Check last candle
            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Inverted Hammer detection."""
        # Validate dataframe
        if len(df) == 0:
            return super()._get_detection_details(df)

        try:
            last_candle = df.iloc[-1]

            # Calculate shadow ratios
            body_size = abs(last_candle['close'] - last_candle['open'])
            lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
            upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])

            shadow_ratio = upper_shadow / body_size if body_size > 0 else 0

            return {
                'location': 'current',
                'candles_ago': 0,
                'confidence': min(0.7 + (shadow_ratio / 10), 0.95),
                'metadata': {
                    'body_size': float(body_size),
                    'lower_shadow': float(lower_shadow),
                    'upper_shadow': float(upper_shadow),
                    'shadow_ratio': float(shadow_ratio)
                }
            }
        except Exception:
            return super()._get_detection_details(df)
