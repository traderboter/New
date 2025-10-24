"""
Doji Pattern Detector

Detects Doji candlestick pattern with configurable threshold.
Doji is a reversal pattern indicating indecision.

Version: 1.2.0 (2025-10-24)
- جایگزینی TA-Lib با detector دستی
- threshold قابل تنظیم (default: 0.10)
- Quality scoring system (0-100)
- Shadow analysis و Doji type detection

Quality Score:
- هرچه body_ratio کمتر → Quality بیشتر
- quality_score = 100 * (1 - body_ratio / threshold)
- مثال: body_ratio=0.01, threshold=0.10 → quality=90%
"""

DOJI_PATTERN_VERSION = "1.2.0"

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

        self.version = DOJI_PATTERN_VERSION

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

    def _calculate_quality_metrics(self, candle: pd.Series) -> Dict[str, Any]:
        """
        محاسبه معیارهای کیفیت Doji.

        Quality Score (0-100):
        - بر اساس کوچکی body نسبت به threshold
        - 100 = Doji کامل (body = 0)
        - 0 = حد آستانه (body_ratio = threshold)

        Doji Types:
        - Standard: سایه‌های بالا و پایین نزدیک به هم
        - Dragonfly: سایه پایین بلند، بدون سایه بالا (صعودی)
        - Gravestone: سایه بالا بلند، بدون سایه پایین (نزولی)
        - Long-legged: سایه‌های بالا و پایین بلند (عدم قطعیت زیاد)
        """
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']

        # محاسبه اندازه‌ها
        body_size = abs(close - open_price)
        full_range = high - low

        if full_range == 0:
            return self._default_quality_metrics()

        body_ratio = body_size / full_range

        # 1. Quality Score (0-100)
        # هرچه body کوچکتر، کیفیت بالاتر
        quality_score = 100 * (1.0 - (body_ratio / self.body_ratio_threshold))
        quality_score = max(0, min(100, quality_score))  # محدود به 0-100

        # 2. Shadow Analysis
        body_midpoint = (open_price + close) / 2
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low

        upper_shadow_ratio = upper_shadow / full_range if full_range > 0 else 0
        lower_shadow_ratio = lower_shadow / full_range if full_range > 0 else 0

        # 3. Symmetry Score (0-100)
        # هرچه سایه‌ها متقارن‌تر، امتیاز بیشتر
        if upper_shadow + lower_shadow > 0:
            shadow_diff = abs(upper_shadow - lower_shadow)
            shadow_sum = upper_shadow + lower_shadow
            symmetry_score = 100 * (1.0 - (shadow_diff / shadow_sum))
        else:
            symmetry_score = 0

        # 4. Doji Type Detection
        doji_type = self._detect_doji_type(
            upper_shadow_ratio,
            lower_shadow_ratio,
            body_ratio
        )

        # 5. Overall Quality (ترکیب همه معیارها)
        # 60% quality_score + 20% symmetry + 20% shadow balance
        shadow_balance = 100 * min(upper_shadow_ratio, lower_shadow_ratio) / max(
            upper_shadow_ratio, lower_shadow_ratio, 0.001
        )

        overall_quality = (
            0.60 * quality_score +
            0.20 * symmetry_score +
            0.20 * shadow_balance
        )

        return {
            'quality_score': round(quality_score, 2),
            'overall_quality': round(overall_quality, 2),
            'symmetry_score': round(symmetry_score, 2),
            'body_ratio': float(body_ratio),
            'body_size': float(body_size),
            'full_range': float(full_range),
            'upper_shadow': float(upper_shadow),
            'lower_shadow': float(lower_shadow),
            'upper_shadow_ratio': float(upper_shadow_ratio),
            'lower_shadow_ratio': float(lower_shadow_ratio),
            'doji_type': doji_type
        }

    def _detect_doji_type(
        self,
        upper_shadow_ratio: float,
        lower_shadow_ratio: float,
        body_ratio: float
    ) -> str:
        """تشخیص نوع Doji بر اساس سایه‌ها."""

        # Dragonfly Doji: سایه پایین بلند، بدون/کم سایه بالا (صعودی)
        if lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
            return "Dragonfly"

        # Gravestone Doji: سایه بالا بلند، بدون/کم سایه پایین (نزولی)
        if upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
            return "Gravestone"

        # Long-legged Doji: هر دو سایه بلند (عدم قطعیت زیاد)
        if upper_shadow_ratio > 0.4 and lower_shadow_ratio > 0.4:
            return "Long-legged"

        # Standard Doji: سایه‌های متعادل
        return "Standard"

    def _default_quality_metrics(self) -> Dict[str, Any]:
        """مقادیر پیش‌فرض برای زمانی که محاسبه ممکن نیست."""
        return {
            'quality_score': 0.0,
            'overall_quality': 0.0,
            'symmetry_score': 0.0,
            'body_ratio': 0.0,
            'body_size': 0.0,
            'full_range': 0.0,
            'upper_shadow': 0.0,
            'lower_shadow': 0.0,
            'upper_shadow_ratio': 0.0,
            'lower_shadow_ratio': 0.0,
            'doji_type': 'Unknown'
        }

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get additional details about Doji detection with quality metrics.

        Returns:
            Dictionary containing:
            - confidence: Trading confidence (0-1)
            - metadata: Detailed quality metrics
        """
        last_candle = df.iloc[-1]

        # محاسبه معیارهای کیفیت
        quality_metrics = self._calculate_quality_metrics(last_candle)

        # محاسبه confidence بر اساس overall_quality
        # overall_quality: 0-100 → confidence: 0.3-0.95
        confidence = 0.3 + (quality_metrics['overall_quality'] / 100) * 0.65
        confidence = max(0.3, min(0.95, confidence))

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': confidence,
            'metadata': {
                **quality_metrics,
                'threshold': float(self.body_ratio_threshold),
                'detector_version': DOJI_PATTERN_VERSION,
                'price_info': {
                    'open': float(last_candle['open']),
                    'high': float(last_candle['high']),
                    'low': float(last_candle['low']),
                    'close': float(last_candle['close'])
                }
            }
        }
