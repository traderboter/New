"""
Morning Doji Star Pattern Detector

Detects Morning Doji Star candlestick pattern using TALib.
Morning Doji Star is a bullish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class MorningDojiStarPattern(BasePattern):
    """
    Morning Doji Star candlestick pattern detector.

    Characteristics:
    - Bullish reversal pattern (3 candles)
    - First candle: Large bearish
    - Second candle: Doji (very small body) - gaps down
    - Third candle: Large bullish - closes above midpoint of first

    Strength: 2/3 (Medium)
    """

    def _get_pattern_name(self) -> str:
        return "Morning Doji Star"

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
        """Detect Morning Doji Star pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        if len(df) < 3:
            return False

        try:
            result = talib.CDLMORNINGDOJISTAR(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Morning Doji Star detection."""
        if len(df) < 3:
            return super()._get_detection_details(df)

        first_candle = df.iloc[-3]
        doji_candle = df.iloc[-2]
        last_candle = df.iloc[-1]

        first_body = abs(first_candle['close'] - first_candle['open'])
        doji_body = abs(doji_candle['close'] - doji_candle['open'])
        last_body = abs(last_candle['close'] - last_candle['open'])
        first_full_range = first_candle['high'] - first_candle['low']
        doji_full_range = doji_candle['high'] - doji_candle['low']
        last_full_range = last_candle['high'] - last_candle['low']

        # Doji should be very small
        # Use safe division: minimum threshold is 30% of candle's full range
        safe_first_body = max(first_body, first_full_range * 0.3) if first_full_range > 0 else 0.0001
        doji_ratio = doji_body / safe_first_body if safe_first_body > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.75 - (doji_ratio * 2), 0.95),  # Smaller doji = higher confidence
            'metadata': {
                'first_body': float(first_body),
                'doji_body': float(doji_body),
                'last_body': float(last_body),
                'first_full_range': float(first_full_range),
                'doji_full_range': float(doji_full_range),
                'last_full_range': float(last_full_range),
                'doji_ratio': float(doji_ratio)
            }
        }
