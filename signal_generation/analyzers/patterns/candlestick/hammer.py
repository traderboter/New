"""
Hammer Pattern Detector

Detects Hammer candlestick pattern with configurable thresholds.
Hammer is a bullish reversal pattern.

Version: 1.2.0 (2025-10-24)
- جایگزینی TA-Lib با detector دستی
- threshold های قابل تنظیم
- Quality scoring system (0-100)
- Hammer type detection و context analysis

Quality Score:
- هرچه lower_shadow بلندتر → Quality بیشتر
- upper_shadow کوچکتر → Quality بیشتر
- Body position در بالا → Quality بیشتر
"""

HAMMER_PATTERN_VERSION = "1.2.0"

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class HammerPattern(BasePattern):
    """
    Hammer candlestick pattern detector.

    Characteristics:
    - Bullish reversal pattern
    - Small body at top of candle
    - Long lower shadow (at least 2x body)
    - Little to no upper shadow
    - Best when appears after downtrend

    Strength: 2/3 (Medium-Strong)

    Configurable Thresholds:
    - min_lower_shadow_ratio: حداقل نسبت lower shadow به body (default: 2.0)
    - max_upper_shadow_ratio: حداکثر نسبت upper shadow به body (default: 0.1)
    - min_body_position: حداقل موقعیت body در range (default: 0.66 = top 1/3)
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        min_lower_shadow_ratio: float = None,
        max_upper_shadow_ratio: float = None,
        min_body_position: float = None
    ):
        """
        Initialize Hammer detector.

        Args:
            config: Configuration dictionary
            min_lower_shadow_ratio: حداقل نسبت lower shadow/body (default: 2.0)
            max_upper_shadow_ratio: حداکثر نسبت upper shadow/body (default: 0.1)
            min_body_position: حداقل موقعیت body (0.66 = top 1/3)
        """
        super().__init__(config)

        # تعیین thresholds از مصادر مختلف
        self.min_lower_shadow_ratio = (
            min_lower_shadow_ratio
            if min_lower_shadow_ratio is not None
            else config.get('hammer_min_lower_shadow_ratio', 2.0) if config else 2.0
        )

        self.max_upper_shadow_ratio = (
            max_upper_shadow_ratio
            if max_upper_shadow_ratio is not None
            else config.get('hammer_max_upper_shadow_ratio', 0.1) if config else 0.1
        )

        self.min_body_position = (
            min_body_position
            if min_body_position is not None
            else config.get('hammer_min_body_position', 0.66) if config else 0.66
        )

        self.version = HAMMER_PATTERN_VERSION

    def _get_pattern_name(self) -> str:
        return "Hammer"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "bullish"

    def _get_base_strength(self) -> int:
        return 2  # Medium-Strong pattern

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
        Detect Hammer pattern using custom thresholds.

        شرایط Hammer:
        1. Lower shadow >= min_lower_shadow_ratio * body
        2. Upper shadow <= max_upper_shadow_ratio * body
        3. Body position >= min_body_position (در بالای کندل)
        """
        if not self._validate_dataframe(df):
            return False

        try:
            last_candle = df.iloc[-1]

            open_price = last_candle[open_col]
            high = last_candle[high_col]
            low = last_candle[low_col]
            close = last_candle[close_col]

            # محاسبه اندازه‌ها
            body_size = abs(close - open_price)
            lower_shadow = min(open_price, close) - low
            upper_shadow = high - max(open_price, close)
            full_range = high - low

            if full_range == 0:
                return False

            # برای جلوگیری از division by zero در body_size
            # استفاده از full_range به عنوان fallback
            body_for_ratio = max(body_size, full_range * 0.01)

            # شرط 1: Lower shadow باید بلند باشد
            lower_shadow_ratio = lower_shadow / body_for_ratio
            if lower_shadow_ratio < self.min_lower_shadow_ratio:
                return False

            # شرط 2: Upper shadow باید کوچک باشد
            upper_shadow_ratio = upper_shadow / body_for_ratio
            if upper_shadow_ratio > self.max_upper_shadow_ratio:
                return False

            # شرط 3: Body باید در بالای کندل باشد
            body_bottom = min(open_price, close)
            body_position = (body_bottom - low) / full_range
            if body_position < self.min_body_position:
                return False

            return True

        except Exception as e:
            return False

    def _calculate_quality_metrics(self, candle: pd.Series, df: pd.DataFrame) -> Dict[str, Any]:
        """
        محاسبه معیارهای کیفیت Hammer.

        Quality Score (0-100):
        - بر اساس قدرت lower shadow
        - کوچکی upper shadow
        - موقعیت body
        - context (downtrend یا نه)

        Hammer Types:
        - Perfect: همه شرایط ایده‌آل (lower_shadow >= 3x body, no upper shadow)
        - Strong: شرایط خوب (lower_shadow >= 2.5x body)
        - Standard: شرایط استاندارد (lower_shadow >= 2x body)
        """
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']

        # محاسبه اندازه‌ها
        body_size = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        full_range = high - low

        if full_range == 0:
            return self._default_quality_metrics()

        body_for_ratio = max(body_size, full_range * 0.01)

        # 1. Lower Shadow Quality (0-100)
        # هرچه بلندتر، بهتر
        lower_shadow_ratio = lower_shadow / body_for_ratio
        lower_shadow_score = min(100, (lower_shadow_ratio / 4.0) * 100)

        # 2. Upper Shadow Quality (0-100)
        # هرچه کوچکتر، بهتر
        upper_shadow_ratio = upper_shadow / body_for_ratio
        upper_shadow_score = max(0, 100 - (upper_shadow_ratio * 100))

        # 3. Body Position Quality (0-100)
        # body باید در بالا باشد
        body_bottom = min(open_price, close)
        body_position = (body_bottom - low) / full_range
        body_position_score = body_position * 100

        # 4. Body Size Quality (0-100)
        # body نباید خیلی بزرگ باشد
        body_size_ratio = body_size / full_range
        body_size_score = max(0, 100 - (body_size_ratio * 100))

        # 5. Overall Quality (weighted average)
        overall_quality = (
            0.40 * lower_shadow_score +      # مهم‌ترین معیار
            0.25 * upper_shadow_score +
            0.20 * body_position_score +
            0.15 * body_size_score
        )

        # 6. Context Analysis (downtrend detection)
        context_score = self._analyze_context(df)

        # 7. Hammer Type Detection
        hammer_type = self._detect_hammer_type(
            lower_shadow_ratio,
            upper_shadow_ratio,
            body_position
        )

        # 8. با context adjustment
        final_quality = (overall_quality * 0.8) + (context_score * 0.2)

        return {
            'quality_score': round(overall_quality, 2),
            'overall_quality': round(final_quality, 2),
            'lower_shadow_score': round(lower_shadow_score, 2),
            'upper_shadow_score': round(upper_shadow_score, 2),
            'body_position_score': round(body_position_score, 2),
            'body_size_score': round(body_size_score, 2),
            'context_score': round(context_score, 2),
            'body_size': float(body_size),
            'lower_shadow': float(lower_shadow),
            'upper_shadow': float(upper_shadow),
            'full_range': float(full_range),
            'lower_shadow_ratio': float(lower_shadow_ratio),
            'upper_shadow_ratio': float(upper_shadow_ratio),
            'body_position': float(body_position),
            'body_size_ratio': float(body_size_ratio),
            'hammer_type': hammer_type,
            'is_after_downtrend': context_score > 50
        }

    def _detect_hammer_type(
        self,
        lower_shadow_ratio: float,
        upper_shadow_ratio: float,
        body_position: float
    ) -> str:
        """تشخیص نوع Hammer بر اساس معیارها."""

        # Perfect Hammer: شرایط ایده‌آل
        if (lower_shadow_ratio >= 3.0 and
            upper_shadow_ratio <= 0.05 and
            body_position >= 0.80):
            return "Perfect"

        # Strong Hammer: شرایط خوب
        if (lower_shadow_ratio >= 2.5 and
            upper_shadow_ratio <= 0.1 and
            body_position >= 0.70):
            return "Strong"

        # Standard Hammer: شرایط استاندارد
        return "Standard"

    def _analyze_context(self, df: pd.DataFrame) -> float:
        """
        تحلیل context برای تشخیص downtrend.

        Returns:
            Score 0-100: هرچه بیشتر، احتمال downtrend بیشتر
        """
        if len(df) < 10:
            return 50  # نمی‌دانیم

        try:
            # بررسی 10 کندل قبلی
            recent = df.tail(10)

            # 1. شیب قیمت (slope)
            closes = recent['close'].values
            indices = np.arange(len(closes))
            slope = np.polyfit(indices, closes, 1)[0]

            # اگر slope منفی → downtrend
            if slope < 0:
                slope_score = min(100, abs(slope) / np.mean(closes) * 10000)
            else:
                slope_score = 0

            # 2. تعداد کندل‌های نزولی
            bearish_count = sum(recent['close'] < recent['open'])
            bearish_score = (bearish_count / len(recent)) * 100

            # 3. Lower lows
            lows = recent['low'].values
            lower_lows = sum(lows[i] < lows[i-1] for i in range(1, len(lows)))
            lower_lows_score = (lower_lows / (len(lows) - 1)) * 100

            # Combined score
            context_score = (
                0.40 * slope_score +
                0.30 * bearish_score +
                0.30 * lower_lows_score
            )

            return min(100, context_score)

        except Exception:
            return 50

    def _default_quality_metrics(self) -> Dict[str, Any]:
        """مقادیر پیش‌فرض برای زمانی که محاسبه ممکن نیست."""
        return {
            'quality_score': 0.0,
            'overall_quality': 0.0,
            'lower_shadow_score': 0.0,
            'upper_shadow_score': 0.0,
            'body_position_score': 0.0,
            'body_size_score': 0.0,
            'context_score': 50.0,
            'body_size': 0.0,
            'lower_shadow': 0.0,
            'upper_shadow': 0.0,
            'full_range': 0.0,
            'lower_shadow_ratio': 0.0,
            'upper_shadow_ratio': 0.0,
            'body_position': 0.0,
            'body_size_ratio': 0.0,
            'hammer_type': 'Unknown',
            'is_after_downtrend': False
        }

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get additional details about Hammer detection with quality metrics.

        Returns:
            Dictionary containing:
            - confidence: Trading confidence (0-1)
            - metadata: Detailed quality metrics
        """
        if len(df) == 0:
            return super()._get_detection_details(df)

        try:
            last_candle = df.iloc[-1]

            # محاسبه معیارهای کیفیت
            quality_metrics = self._calculate_quality_metrics(last_candle, df)

            # محاسبه confidence بر اساس overall_quality
            # overall_quality: 0-100 → confidence: 0.4-0.95
            confidence = 0.4 + (quality_metrics['overall_quality'] / 100) * 0.55
            confidence = max(0.4, min(0.95, confidence))

            return {
                'location': 'current',
                'candles_ago': 0,
                'confidence': confidence,
                'metadata': {
                    **quality_metrics,
                    'thresholds': {
                        'min_lower_shadow_ratio': float(self.min_lower_shadow_ratio),
                        'max_upper_shadow_ratio': float(self.max_upper_shadow_ratio),
                        'min_body_position': float(self.min_body_position)
                    },
                    'detector_version': HAMMER_PATTERN_VERSION,
                    'price_info': {
                        'open': float(last_candle['open']),
                        'high': float(last_candle['high']),
                        'low': float(last_candle['low']),
                        'close': float(last_candle['close'])
                    }
                }
            }
        except Exception:
            return super()._get_detection_details(df)
