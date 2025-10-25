"""
Engulfing Pattern Detector

Detects Bullish and Bearish Engulfing candlestick patterns using TALib.
Engulfing patterns are strong reversal patterns.

Version: 2.0.0 (2025-10-25) - MAJOR CHANGE
- 🔄 BREAKING: Fix TA-Lib integration (3+ candles required)
- 🔬 بر اساس تحقیقات در talib-test/:
  * TA-Lib نیاز به حداقل 3 کندل دارد (2 قبلی + 1 فعلی)
  * با کمتر از 3 کندل: هیچ تشخیصی نمی‌دهد
- 📊 Detection rate در BTC 1-hour data: 1714/10543 = 16.26%
- ✅ بیشترین detection rate در بین همه الگوها!
"""

ENGULFING_PATTERN_VERSION = "2.0.0"

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class EngulfingPattern(BasePattern):
    """
    Engulfing candlestick pattern detector using TA-Lib.

    Characteristics:
    - Can be bullish or bearish
    - Second candle completely engulfs first candle's body
    - Strong reversal signal
    - Direction determined by engulfing candle color

    Strength: 3/3 (Strong)

    TA-Lib Requirements (based on research in talib-test/):
    - Minimum 3 candles (2 previous + 1 current) - CRITICAL!
    - Detection rate on BTC 1-hour: 1714/10543 = 16.26%
    - Highest detection rate among all patterns!
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Engulfing detector."""
        super().__init__(config)
        self.version = ENGULFING_PATTERN_VERSION

    def _get_pattern_name(self) -> str:
        return "Engulfing"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "both"  # Can be bullish or bearish

    def _get_base_strength(self) -> int:
        return 3  # Strong pattern

    def _get_talib_result(self, df: pd.DataFrame) -> np.ndarray:
        """Get TALib CDLENGULFING result (helper method to avoid duplicate calls)."""
        try:
            result = talib.CDLENGULFING(
                df['open'].values,
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            return result
        except Exception:
            return np.array([])

    def detect(
        self,
        df: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> bool:
        """
        Detect Engulfing pattern using TA-Lib CDLENGULFING.

        TA-Lib Requirements (based on research in talib-test/):
        1. Minimum 3 candles (2 previous + 1 current) - CRITICAL!
        2. Can detect both bullish and bearish engulfing
        3. Detection rate: 16.26% (highest among all patterns!)

        Returns:
            bool: True if Engulfing pattern detected on last candle
        """
        if not self._validate_dataframe(df):
            return False

        # TA-Lib needs minimum 3 candles
        if len(df) < 3:
            return False

        try:
            # Use TALib to detect
            # Pass full DataFrame - TA-Lib uses previous candles for context
            result = talib.CDLENGULFING(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            # Check last candle
            # result values: +100 (bullish), -100 (bearish), 0 (no pattern)
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
            # Use helper method to get TALib result
            result = self._get_talib_result(df)

            if len(result) == 0:
                return 'bullish'

            # Positive = bullish, negative = bearish
            return 'bullish' if result[-1] > 0 else 'bearish'

        except Exception:
            return 'bullish'

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Engulfing detection."""
        # Validate minimum candles
        if len(df) < 2:
            return super()._get_detection_details(df)

        try:
            # Get last two candles
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]

            # Calculate body sizes
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            curr_body = abs(curr_candle['close'] - curr_candle['open'])
            prev_full_range = prev_candle['high'] - prev_candle['low']
            curr_full_range = curr_candle['high'] - curr_candle['low']

            # Engulfing ratio (how much bigger is current body)
            # Use safe division: minimum threshold is 30% of candle's full range
            safe_prev_body = max(prev_body, prev_full_range * 0.3) if prev_full_range > 0 else 0.0001
            engulfing_ratio = curr_body / safe_prev_body if safe_prev_body > 0 else 1

            # Determine direction using TALib result
            result = self._get_talib_result(df)
            direction = 'bullish' if len(result) > 0 and result[-1] > 0 else 'bearish'

            return {
                'location': 'current',
                'candles_ago': 0,
                'confidence': min(0.75 + (engulfing_ratio / 10), 0.95),
                'metadata': {
                    'prev_body_size': float(prev_body),
                    'curr_body_size': float(curr_body),
                    'prev_full_range': float(prev_full_range),
                    'curr_full_range': float(curr_full_range),
                    'engulfing_ratio': float(engulfing_ratio),
                    'pattern_direction': direction
                }
            }
        except Exception:
            return super()._get_detection_details(df)
