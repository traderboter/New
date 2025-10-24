"""
Doji Pattern Detector

Detects Doji candlestick pattern with configurable threshold.
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

    Threshold for body_ratio:
    - Standard: 0.10 (10%) - Default
    - Strict: 0.05 (5%)
    - Relaxed: 0.15 (15%)
    """

    def __init__(self, config: Dict[str, Any] = None, body_ratio_threshold: float = None):
        """
        Initialize Doji detector.

        Args:
            config: Configuration dictionary (from orchestrator)
            body_ratio_threshold: Maximum ratio of body to full range (default: 0.10 = 10%)
                                 Can also be set via config['doji_threshold']
        """
        super().__init__(config)

        # Determine threshold from multiple sources (priority order):
        # 1. Direct parameter
        # 2. Config dictionary
        # 3. Default value (0.10)
        if body_ratio_threshold is not None:
            self.body_ratio_threshold = body_ratio_threshold
        elif config and 'doji_threshold' in config:
            self.body_ratio_threshold = config['doji_threshold']
        else:
            self.body_ratio_threshold = 0.10

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
        """
        Detect Doji pattern using custom threshold.

        A candle is considered a Doji if:
        1. Body size is very small relative to the full range
        2. body_ratio = |close - open| / (high - low) < threshold
        """
        if not self._validate_dataframe(df):
            return False

        try:
            last_candle = df.iloc[-1]

            # محاسبه اندازه body و range کامل
            body_size = abs(last_candle[close_col] - last_candle[open_col])
            full_range = last_candle[high_col] - last_candle[low_col]

            # جلوگیری از تقسیم بر صفر
            if full_range == 0:
                return False

            # محاسبه نسبت body به range
            body_ratio = body_size / full_range

            # تشخیص Doji بر اساس threshold
            return body_ratio <= self.body_ratio_threshold

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Doji detection."""
        last_candle = df.iloc[-1]

        body_size = abs(last_candle['close'] - last_candle['open'])
        full_range = last_candle['high'] - last_candle['low']

        # Body should be very small relative to range
        body_ratio = body_size / full_range if full_range > 0 else 0

        # محاسبه confidence بر اساس اینکه body چقدر کوچک است
        # هر چه body_ratio کمتر باشد، confidence بیشتر است
        confidence = 1.0 - (body_ratio / self.body_ratio_threshold)
        confidence = max(0.3, min(confidence, 0.9))  # محدود کردن به بازه 0.3 تا 0.9

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': confidence,
            'metadata': {
                'body_size': float(body_size),
                'full_range': float(full_range),
                'body_ratio': float(body_ratio),
                'threshold': float(self.body_ratio_threshold)
            }
        }
