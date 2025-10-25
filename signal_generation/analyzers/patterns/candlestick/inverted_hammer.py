"""
Inverted Hammer Pattern Detector

Detects Inverted Hammer candlestick pattern using TALib.
Inverted Hammer is a bullish reversal pattern.

Version: 2.0.0 (2025-10-25) - MAJOR CHANGE
- 🔄 BREAKING: Fix TA-Lib integration (12+ candles required)
- 🔬 بر اساس تحقیقات در talib-test/:
  * TA-Lib نیاز به حداقل 12 کندل دارد (11 قبلی + 1 فعلی)
  * با کمتر از 12 کندل: هیچ تشخیصی نمی‌دهد
- 📊 Detection rate در BTC 1-hour data: 59/10543 = 0.56%
"""

INVERTED_HAMMER_PATTERN_VERSION = "2.0.0"

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class InvertedHammerPattern(BasePattern):
    """
    Inverted Hammer candlestick pattern detector using TA-Lib.

    Characteristics:
    - Bullish reversal pattern
    - Small body at bottom of candle
    - Long upper shadow (at least 2x body)
    - Little to no lower shadow
    - Best when appears after downtrend

    Strength: 2/3 (Medium)

    TA-Lib Requirements (based on research in talib-test/):
    - Minimum 12 candles (11 previous + 1 current) - CRITICAL!
    - Detection rate on BTC 1-hour: 59/10543 = 0.56%
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Inverted Hammer detector."""
        super().__init__(config)
        self.version = INVERTED_HAMMER_PATTERN_VERSION

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
        """
        Detect Inverted Hammer pattern using TA-Lib CDLINVERTEDHAMMER.

        TA-Lib Requirements (based on research in talib-test/):
        1. Minimum 12 candles (11 previous + 1 current) - CRITICAL!
        2. Detection rate: 0.56%

        Returns:
            bool: True if Inverted Hammer pattern detected on last candle
        """
        if not self._validate_dataframe(df):
            return False

        # TA-Lib needs minimum 12 candles
        if len(df) < 12:
            return False

        try:
            # Use TALib to detect
            # Pass full DataFrame - TA-Lib uses previous candles for context
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
            full_range = last_candle['high'] - last_candle['low']

            # Use full_range instead of body_size to avoid zero/very small body issues
            shadow_ratio = upper_shadow / full_range if full_range > 0 else 0

            return {
                'location': 'current',
                'candles_ago': 0,
                'confidence': min(0.7 + (shadow_ratio / 10), 0.95),
                'metadata': {
                    'body_size': float(body_size),
                    'lower_shadow': float(lower_shadow),
                    'upper_shadow': float(upper_shadow),
                    'full_range': float(full_range),
                    'shadow_ratio': float(shadow_ratio)
                }
            }
        except Exception:
            return super()._get_detection_details(df)
