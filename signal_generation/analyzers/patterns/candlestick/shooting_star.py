"""
Shooting Star Pattern Detector

Detects Shooting Star candlestick pattern with configurable thresholds.
Shooting Star is a bearish reversal pattern (opposite of Hammer).

Version: 1.2.2 (2025-10-24)
- رفع ادامه مشکل threshold - max_lower_shadow: 0.5 → 1.0
- با این تغییر، Shooting Star می‌تواند lower shadow تا 1x body داشته باشد

Version: 1.2.1 (2025-10-24)
- رفع مشکل threshold های خیلی سخت (relaxed defaults)
- min_upper_shadow: 2.0 → 1.5
- max_lower_shadow: 0.1 → 0.5
- max_body_position: 0.33 → 0.4

Version: 1.2.0 (2025-10-24)
- جایگزینی TA-Lib با detector دستی
- threshold های قابل تنظیم
- Quality scoring system (0-100)
- Shooting Star type detection و context analysis

Quality Score:
- هرچه upper_shadow بلندتر → Quality بیشتر
- lower_shadow کوچکتر → Quality بیشتر
- Body position در پایین → Quality بیشتر
"""

SHOOTING_STAR_PATTERN_VERSION = "1.2.2"

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class ShootingStarPattern(BasePattern):
    """
    Shooting Star candlestick pattern detector.

    Characteristics:
    - Bearish reversal pattern (opposite of Hammer)
    - Small body at bottom of candle
    - Long upper shadow (at least 2x body)
    - Little to no lower shadow
    - Best when appears after uptrend

    Strength: 2/3 (Medium-Strong)

    Configurable Thresholds:
    - min_upper_shadow_ratio: حداقل نسبت upper shadow به body (default: 1.5)
    - max_lower_shadow_ratio: حداکثر نسبت lower shadow به body (default: 1.0)
    - max_body_position: حداکثر موقعیت body در range (default: 0.4 = bottom 40%)
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        min_upper_shadow_ratio: float = None,
        max_lower_shadow_ratio: float = None,
        max_body_position: float = None
    ):
        """
        Initialize Shooting Star detector.

        Args:
            config: Configuration dictionary
            min_upper_shadow_ratio: حداقل نسبت upper shadow/body (default: 1.5)
            max_lower_shadow_ratio: حداکثر نسبت lower shadow/body (default: 1.0)
            max_body_position: حداکثر موقعیت body (0.4 = bottom 40%)
        """
        super().__init__(config)

        # تعیین thresholds از مصادر مختلف
        self.min_upper_shadow_ratio = (
            min_upper_shadow_ratio
            if min_upper_shadow_ratio is not None
            else config.get('shooting_star_min_upper_shadow_ratio', 1.5) if config else 1.5
        )

        self.max_lower_shadow_ratio = (
            max_lower_shadow_ratio
            if max_lower_shadow_ratio is not None
            else config.get('shooting_star_max_lower_shadow_ratio', 1.0) if config else 1.0
        )

        self.max_body_position = (
            max_body_position
            if max_body_position is not None
            else config.get('shooting_star_max_body_position', 0.4) if config else 0.4
        )

        self.version = SHOOTING_STAR_PATTERN_VERSION

    def _get_pattern_name(self) -> str:
        return "Shooting Star"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "bearish"

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
        Detect Shooting Star pattern using custom thresholds.

        شرایط Shooting Star:
        1. Upper shadow >= min_upper_shadow_ratio * body
        2. Lower shadow <= max_lower_shadow_ratio * body
        3. Body position <= max_body_position (در پایین کندل)
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
            upper_shadow = high - max(open_price, close)
            lower_shadow = min(open_price, close) - low
            full_range = high - low

            if full_range == 0:
                return False

            # برای جلوگیری از division by zero در body_size
            body_for_ratio = max(body_size, full_range * 0.01)

            # شرط 1: Upper shadow باید بلند باشد
            upper_shadow_ratio = upper_shadow / body_for_ratio
            if upper_shadow_ratio < self.min_upper_shadow_ratio:
                return False

            # شرط 2: Lower shadow باید کوچک باشد
            lower_shadow_ratio = lower_shadow / body_for_ratio
            if lower_shadow_ratio > self.max_lower_shadow_ratio:
                return False

            # شرط 3: Body باید در پایین کندل باشد
            body_bottom = min(open_price, close)
            body_position = (body_bottom - low) / full_range
            if body_position > self.max_body_position:
                return False

            return True

        except Exception as e:
            return False

    def _calculate_quality_metrics(self, candle: pd.Series, df: pd.DataFrame) -> Dict[str, Any]:
        """
        محاسبه معیارهای کیفیت Shooting Star.

        Quality Score (0-100):
        - بر اساس قدرت upper shadow
        - کوچکی lower shadow
        - موقعیت body
        - context (uptrend یا نه)

        Shooting Star Types:
        - Perfect: همه شرایط ایده‌آل (upper_shadow >= 3x body, no lower shadow)
        - Strong: شرایط خوب (upper_shadow >= 2.5x body)
        - Standard: شرایط استاندارد (upper_shadow >= 2x body)
        """
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']

        # محاسبه اندازه‌ها
        body_size = abs(close - open_price)
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        full_range = high - low

        if full_range == 0:
            return self._default_quality_metrics()

        body_for_ratio = max(body_size, full_range * 0.01)

        # 1. Upper Shadow Quality (0-100)
        # هرچه بلندتر، بهتر
        upper_shadow_ratio = upper_shadow / body_for_ratio
        upper_shadow_score = min(100, (upper_shadow_ratio / 4.0) * 100)

        # 2. Lower Shadow Quality (0-100)
        # هرچه کوچکتر، بهتر
        lower_shadow_ratio = lower_shadow / body_for_ratio
        lower_shadow_score = max(0, 100 - (lower_shadow_ratio * 100))

        # 3. Body Position Quality (0-100)
        # body باید در پایین باشد
        body_bottom = min(open_price, close)
        body_position = (body_bottom - low) / full_range
        # برای Shooting Star: موقعیت پایین‌تر = بهتر
        body_position_score = (1.0 - body_position) * 100

        # 4. Body Size Quality (0-100)
        # body نباید خیلی بزرگ باشد
        body_size_ratio = body_size / full_range
        body_size_score = max(0, 100 - (body_size_ratio * 100))

        # 5. Overall Quality (weighted average)
        overall_quality = (
            0.40 * upper_shadow_score +      # مهم‌ترین معیار
            0.25 * lower_shadow_score +
            0.20 * body_position_score +
            0.15 * body_size_score
        )

        # 6. Context Analysis (uptrend detection)
        context_score = self._analyze_context(df)

        # 7. Shooting Star Type Detection
        shooting_star_type = self._detect_shooting_star_type(
            upper_shadow_ratio,
            lower_shadow_ratio,
            body_position
        )

        # 8. با context adjustment
        final_quality = (overall_quality * 0.8) + (context_score * 0.2)

        return {
            'quality_score': round(overall_quality, 2),
            'overall_quality': round(final_quality, 2),
            'upper_shadow_score': round(upper_shadow_score, 2),
            'lower_shadow_score': round(lower_shadow_score, 2),
            'body_position_score': round(body_position_score, 2),
            'body_size_score': round(body_size_score, 2),
            'context_score': round(context_score, 2),
            'body_size': float(body_size),
            'upper_shadow': float(upper_shadow),
            'lower_shadow': float(lower_shadow),
            'full_range': float(full_range),
            'upper_shadow_ratio': float(upper_shadow_ratio),
            'lower_shadow_ratio': float(lower_shadow_ratio),
            'body_position': float(body_position),
            'body_size_ratio': float(body_size_ratio),
            'shooting_star_type': shooting_star_type,
            'is_after_uptrend': context_score > 50
        }

    def _detect_shooting_star_type(
        self,
        upper_shadow_ratio: float,
        lower_shadow_ratio: float,
        body_position: float
    ) -> str:
        """تشخیص نوع Shooting Star بر اساس معیارها."""

        # Perfect Shooting Star: شرایط ایده‌آل
        if (upper_shadow_ratio >= 3.0 and
            lower_shadow_ratio <= 0.05 and
            body_position <= 0.20):
            return "Perfect"

        # Strong Shooting Star: شرایط خوب
        if (upper_shadow_ratio >= 2.5 and
            lower_shadow_ratio <= 0.1 and
            body_position <= 0.30):
            return "Strong"

        # Standard Shooting Star: شرایط استاندارد
        return "Standard"

    def _analyze_context(self, df: pd.DataFrame) -> float:
        """
        تحلیل context برای تشخیص uptrend.

        Returns:
            Score 0-100: هرچه بیشتر، احتمال uptrend بیشتر
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

            # اگر slope مثبت → uptrend
            if slope > 0:
                slope_score = min(100, abs(slope) / np.mean(closes) * 10000)
            else:
                slope_score = 0

            # 2. تعداد کندل‌های صعودی
            bullish_count = sum(recent['close'] > recent['open'])
            bullish_score = (bullish_count / len(recent)) * 100

            # 3. Higher highs
            highs = recent['high'].values
            higher_highs = sum(highs[i] > highs[i-1] for i in range(1, len(highs)))
            higher_highs_score = (higher_highs / (len(highs) - 1)) * 100

            # Combined score
            context_score = (
                0.40 * slope_score +
                0.30 * bullish_score +
                0.30 * higher_highs_score
            )

            return min(100, context_score)

        except Exception:
            return 50

    def _default_quality_metrics(self) -> Dict[str, Any]:
        """مقادیر پیش‌فرض برای زمانی که محاسبه ممکن نیست."""
        return {
            'quality_score': 0.0,
            'overall_quality': 0.0,
            'upper_shadow_score': 0.0,
            'lower_shadow_score': 0.0,
            'body_position_score': 0.0,
            'body_size_score': 0.0,
            'context_score': 50.0,
            'body_size': 0.0,
            'upper_shadow': 0.0,
            'lower_shadow': 0.0,
            'full_range': 0.0,
            'upper_shadow_ratio': 0.0,
            'lower_shadow_ratio': 0.0,
            'body_position': 0.0,
            'body_size_ratio': 0.0,
            'shooting_star_type': 'Unknown',
            'is_after_uptrend': False
        }

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get additional details about Shooting Star detection with quality metrics.

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
                        'min_upper_shadow_ratio': float(self.min_upper_shadow_ratio),
                        'max_lower_shadow_ratio': float(self.max_lower_shadow_ratio),
                        'max_body_position': float(self.max_body_position)
                    },
                    'detector_version': SHOOTING_STAR_PATTERN_VERSION,
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
