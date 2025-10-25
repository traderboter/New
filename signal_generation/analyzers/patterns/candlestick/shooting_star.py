"""
Shooting Star Pattern Detector

Detects Shooting Star candlestick pattern with configurable thresholds.
Shooting Star is a bearish reversal pattern (opposite of Hammer).

Version: 1.4.1 (2025-10-25)
- ⚡ OPTIMIZATION: اضافه شدن cache برای _analyze_context()
- حذف محاسبات تکراری - context_score فقط یک بار برای هر کندل محاسبه می‌شود
- Cache با تغییر طول DataFrame به‌روز می‌شود

Version: 1.4.0 (2025-10-25)
- 🎯 FIX CRITICAL: اضافه شدن شرط uptrend برای detection
- Shooting Star فقط در uptrend معتبر است (الگوی بازگشتی)
- پارامترهای جدید:
  * require_uptrend: آیا uptrend اجباری است؟ (default: True)
  * min_uptrend_score: حداقل امتیاز uptrend (default: 50.0)

Version: 1.3.0 (2025-10-24)
- 🔧 FIX CRITICAL: تغییر منطق detection از body-based به range-based
- قبلاً: مقایسه shadows با body size (منطق اشتباه!)
- حالا: مقایسه shadows با full range (منطق صحیح!)
- Thresholds جدید:
  * min_upper_shadow_pct: حداقل درصد upper shadow از range (default: 50%)
  * max_lower_shadow_pct: حداکثر درصد lower shadow از range (default: 20%)
  * max_body_pct: حداکثر درصد body از range (default: 30%)
  * max_body_position: موقعیت body در پایین (default: 0.4)

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

SHOOTING_STAR_PATTERN_VERSION = "1.4.1"

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
    - Long upper shadow (at least 50% of full range)
    - Little to no lower shadow (max 20% of full range)
    - Best when appears after uptrend

    Strength: 2/3 (Medium-Strong)

    Configurable Thresholds (all relative to full candle range):
    - min_upper_shadow_pct: حداقل درصد upper shadow (default: 0.5 = 50%)
    - max_lower_shadow_pct: حداکثر درصد lower shadow (default: 0.2 = 20%)
    - max_body_pct: حداکثر درصد body (default: 0.3 = 30%)
    - max_body_position: حداکثر موقعیت body (default: 0.4 = bottom 40%)
    - require_uptrend: آیا uptrend اجباری است؟ (default: True)
    - min_uptrend_score: حداقل امتیاز uptrend (default: 50.0 = 0-100 scale)
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        min_upper_shadow_pct: float = None,
        max_lower_shadow_pct: float = None,
        max_body_pct: float = None,
        max_body_position: float = None,
        require_uptrend: bool = None,
        min_uptrend_score: float = None
    ):
        """
        Initialize Shooting Star detector.

        Args:
            config: Configuration dictionary
            min_upper_shadow_pct: حداقل درصد upper shadow از range (default: 0.5 = 50%)
            max_lower_shadow_pct: حداکثر درصد lower shadow از range (default: 0.2 = 20%)
            max_body_pct: حداکثر درصد body از range (default: 0.3 = 30%)
            max_body_position: حداکثر موقعیت body (0.4 = bottom 40%)
            require_uptrend: آیا uptrend برای detection اجباری است؟ (default: True)
            min_uptrend_score: حداقل امتیاز uptrend برای detection (default: 50.0)
        """
        super().__init__(config)

        # تعیین thresholds از مصادر مختلف - همه نسبت به full range
        self.min_upper_shadow_pct = (
            min_upper_shadow_pct
            if min_upper_shadow_pct is not None
            else config.get('shooting_star_min_upper_shadow_pct', 0.5) if config else 0.5
        )

        self.max_lower_shadow_pct = (
            max_lower_shadow_pct
            if max_lower_shadow_pct is not None
            else config.get('shooting_star_max_lower_shadow_pct', 0.2) if config else 0.2
        )

        self.max_body_pct = (
            max_body_pct
            if max_body_pct is not None
            else config.get('shooting_star_max_body_pct', 0.3) if config else 0.3
        )

        self.max_body_position = (
            max_body_position
            if max_body_position is not None
            else config.get('shooting_star_max_body_position', 0.4) if config else 0.4
        )

        # پارامترهای uptrend (جدید در v1.4.0)
        self.require_uptrend = (
            require_uptrend
            if require_uptrend is not None
            else config.get('shooting_star_require_uptrend', True) if config else True
        )

        self.min_uptrend_score = (
            min_uptrend_score
            if min_uptrend_score is not None
            else config.get('shooting_star_min_uptrend_score', 50.0) if config else 50.0
        )

        self.version = SHOOTING_STAR_PATTERN_VERSION

        # Cache for context_score to avoid duplicate calculations
        self._cached_context_score = None
        self._cached_df_length = None

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
        Detect Shooting Star pattern using range-based thresholds.

        شرایط Shooting Star (همه نسبت به full range):
        1. Upper shadow >= min_upper_shadow_pct از range (مثلاً 50%)
        2. Lower shadow <= max_lower_shadow_pct از range (مثلاً 20%)
        3. Body size <= max_body_pct از range (مثلاً 30%)
        4. Body position <= max_body_position (در پایین کندل)
        5. (اختیاری) Uptrend detection: context score >= min_uptrend_score
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

            # محاسبه درصدها نسبت به full range
            upper_shadow_pct = upper_shadow / full_range
            lower_shadow_pct = lower_shadow / full_range
            body_pct = body_size / full_range

            # شرط 1: Upper shadow باید بلند باشد (حداقل 50% از range)
            if upper_shadow_pct < self.min_upper_shadow_pct:
                return False

            # شرط 2: Lower shadow باید کوچک باشد (حداکثر 20% از range)
            if lower_shadow_pct > self.max_lower_shadow_pct:
                return False

            # شرط 3: Body باید کوچک باشد (حداکثر 30% از range)
            if body_pct > self.max_body_pct:
                return False

            # شرط 4: Body باید در پایین کندل باشد
            body_bottom = min(open_price, close)
            body_position = (body_bottom - low) / full_range
            if body_position > self.max_body_position:
                return False

            # شرط 5: چک کردن uptrend (جدید در v1.4.0)
            if self.require_uptrend:
                context_score = self._get_cached_context_score(df)
                if context_score < self.min_uptrend_score:
                    return False  # Shooting Star فقط در uptrend معتبر است

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
        - Perfect: همه شرایط ایده‌آل (upper_shadow >= 70%, lower_shadow <= 5%)
        - Strong: شرایط خوب (upper_shadow >= 60%, lower_shadow <= 10%)
        - Standard: شرایط استاندارد (upper_shadow >= 50%, lower_shadow <= 20%)
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

        # محاسبه درصدها نسبت به full range
        upper_shadow_pct = upper_shadow / full_range
        lower_shadow_pct = lower_shadow / full_range
        body_size_pct = body_size / full_range

        # 1. Upper Shadow Quality (0-100)
        # هرچه بلندتر، بهتر (0.5 → 50 points, 1.0 → 100 points)
        upper_shadow_score = min(100, upper_shadow_pct * 100 * 2)

        # 2. Lower Shadow Quality (0-100)
        # هرچه کوچکتر، بهتر (0.0 → 100 points, 0.2 → 0 points)
        lower_shadow_score = max(0, 100 - (lower_shadow_pct * 500))

        # 3. Body Position Quality (0-100)
        # body باید در پایین باشد
        body_bottom = min(open_price, close)
        body_position = (body_bottom - low) / full_range
        # برای Shooting Star: موقعیت پایین‌تر = بهتر
        body_position_score = (1.0 - body_position) * 100

        # 4. Body Size Quality (0-100)
        # body نباید خیلی بزرگ باشد (0.0 → 100 points, 0.3 → 0 points)
        body_size_score = max(0, 100 - (body_size_pct * 333))

        # 5. Overall Quality (weighted average)
        overall_quality = (
            0.40 * upper_shadow_score +      # مهم‌ترین معیار
            0.25 * lower_shadow_score +
            0.20 * body_position_score +
            0.15 * body_size_score
        )

        # 6. Context Analysis (uptrend detection) - use cached value
        context_score = self._get_cached_context_score(df)

        # 7. Shooting Star Type Detection
        shooting_star_type = self._detect_shooting_star_type(
            upper_shadow_pct,
            lower_shadow_pct,
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
            'upper_shadow_pct': float(upper_shadow_pct),
            'lower_shadow_pct': float(lower_shadow_pct),
            'body_position': float(body_position),
            'body_size_pct': float(body_size_pct),
            'shooting_star_type': shooting_star_type,
            'is_after_uptrend': context_score > 50
        }

    def _detect_shooting_star_type(
        self,
        upper_shadow_pct: float,
        lower_shadow_pct: float,
        body_position: float
    ) -> str:
        """تشخیص نوع Shooting Star بر اساس معیارها (percentage-based)."""

        # Perfect Shooting Star: شرایط ایده‌آل
        if (upper_shadow_pct >= 0.70 and
            lower_shadow_pct <= 0.05 and
            body_position <= 0.20):
            return "Perfect"

        # Strong Shooting Star: شرایط خوب
        if (upper_shadow_pct >= 0.60 and
            lower_shadow_pct <= 0.10 and
            body_position <= 0.30):
            return "Strong"

        # Standard Shooting Star: شرایط استاندارد
        return "Standard"

    def _get_cached_context_score(self, df: pd.DataFrame) -> float:
        """
        Get context score with caching to avoid duplicate calculations.

        Cache is invalidated when df length changes (new candle added).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Context score (0-100)
        """
        current_df_length = len(df)

        # Check if cache is valid
        if (self._cached_context_score is not None and
            self._cached_df_length == current_df_length):
            return self._cached_context_score

        # Calculate and cache
        self._cached_context_score = self._analyze_context(df)
        self._cached_df_length = current_df_length

        return self._cached_context_score

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
            'upper_shadow_pct': 0.0,
            'lower_shadow_pct': 0.0,
            'body_position': 0.0,
            'body_size_pct': 0.0,
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
                        'min_upper_shadow_pct': float(self.min_upper_shadow_pct),
                        'max_lower_shadow_pct': float(self.max_lower_shadow_pct),
                        'max_body_pct': float(self.max_body_pct),
                        'max_body_position': float(self.max_body_position),
                        'require_uptrend': bool(self.require_uptrend),
                        'min_uptrend_score': float(self.min_uptrend_score)
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
