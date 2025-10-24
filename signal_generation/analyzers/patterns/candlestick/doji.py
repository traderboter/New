"""
Doji Pattern Detector

Detects Doji candlestick pattern using TALib.
Doji is a reversal pattern indicating indecision.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class DojiPattern(BasePattern):
    """
    Doji candlestick pattern detector.

    Characteristics:
    - Reversal pattern
    - Open and close are very close (virtually same price)
    - Indicates market indecision
    - Can be bullish or bearish depending on context

    Strength: 1/3 (Weak - needs confirmation)
    """

    def _get_pattern_name(self) -> str:
        return "Doji"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "reversal"  # Direction depends on context

    def _get_base_strength(self) -> int:
        return 1  # Weak pattern, needs confirmation

    def detect(
        self,
        df: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> bool:
        """Detect Doji pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        try:
            result = talib.CDLDOJI(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Doji detection."""
        last_candle = df.iloc[-1]

        body_size = abs(last_candle['close'] - last_candle['open'])
        full_range = last_candle['high'] - last_candle['low']

        # Body should be very small relative to range
        body_ratio = body_size / full_range if full_range > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.60 - (body_ratio * 5), 0.80),  # Smaller body = higher confidence
            'metadata': {
                'body_size': float(body_size),
                'full_range': float(full_range),
                'body_ratio': float(body_ratio)
            }
        }
