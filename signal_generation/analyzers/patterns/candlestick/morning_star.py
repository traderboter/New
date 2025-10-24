"""
Morning Star Pattern Detector

Detects Morning Star candlestick pattern using TALib.
Morning Star is a strong bullish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class MorningStarPattern(BasePattern):
    """
    Morning Star candlestick pattern detector.

    Characteristics:
    - Strong bullish reversal pattern (3 candles)
    - First candle: Large bearish candle
    - Second candle: Small body (star) - gaps down
    - Third candle: Large bullish candle - closes above midpoint of first

    Strength: 3/3 (Strong)
    """

    def _get_pattern_name(self) -> str:
        return "Morning Star"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "bullish"

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
        """Detect Morning Star pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        if len(df) < 3:
            return False

        try:
            # Use TALib to detect
            result = talib.CDLMORNINGSTAR(
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
        """Get additional details about Morning Star detection."""
        # Validate minimum candles
        if len(df) < 3:
            return super()._get_detection_details(df)

        try:
            # Get last three candles
            first_candle = df.iloc[-3]
            star_candle = df.iloc[-2]
            last_candle = df.iloc[-1]

            # Calculate body sizes
            first_body = abs(first_candle['close'] - first_candle['open'])
            star_body = abs(star_candle['close'] - star_candle['open'])
            last_body = abs(last_candle['close'] - last_candle['open'])

            # Star should be small (lower star_ratio = better)
            star_ratio = star_body / first_body if first_body > 0 else 0

            # Last candle should be strong (higher strength_ratio = better)
            strength_ratio = last_body / first_body if first_body > 0 else 1

            # Calculate confidence: higher strength + smaller star = higher confidence
            # Subtract star_ratio to reward smaller stars
            confidence_score = 0.80 + (strength_ratio / 10) - (star_ratio / 20)
            confidence = min(max(confidence_score, 0.70), 0.95)  # Keep in valid range

            # Determine candle directions
            first_direction = 'bearish' if first_candle['close'] < first_candle['open'] else 'bullish'
            last_direction = 'bullish' if last_candle['close'] > last_candle['open'] else 'bearish'

            return {
                'location': 'current',
                'candles_ago': 0,
                'confidence': confidence,
                'metadata': {
                    'first_body': float(first_body),
                    'star_body': float(star_body),
                    'last_body': float(last_body),
                    'star_ratio': float(star_ratio),
                    'strength_ratio': float(strength_ratio),
                    'first_candle_direction': first_direction,
                    'last_candle_direction': last_direction
                }
            }
        except Exception:
            return super()._get_detection_details(df)
