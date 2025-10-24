"""
تست دیباگ برای Shooting Star - ساخت کندل‌های مصنوعی

این اسکریپت کندل‌های مصنوعی با شرایط روشن Shooting Star می‌سازد
و می‌بیند آیا detector آنها را تشخیص می‌دهد یا نه.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from signal_generation.analyzers.patterns.candlestick.shooting_star import ShootingStarPattern


def create_shooting_star_candle():
    """
    ساخت یک کندل Shooting Star واضح و آشکار

    Shooting Star باید:
    - Upper shadow بلند (>= 1.5x body)
    - Lower shadow کوچک (<= 0.5x body)
    - Body در پایین (<= 40% from bottom)
    """
    # مثال 1: Perfect Shooting Star
    # High: 100, Low: 90, Open: 91, Close: 92
    # Body = 1, Upper shadow = 8, Lower shadow = 1
    # upper_shadow_ratio = 8 / 1 = 8.0 (>> 1.5 ✓)
    # lower_shadow_ratio = 1 / 1 = 1.0 (> 0.5 ✗)

    # مثال 2: بهتر - Lower shadow کوچکتر
    # High: 100, Low: 90, Open: 90.5, Close: 91
    # Body = 0.5, Upper shadow = 9, Lower shadow = 0.5
    # upper_shadow_ratio = 9 / 0.5 = 18.0 (>> 1.5 ✓)
    # lower_shadow_ratio = 0.5 / 0.5 = 1.0 (> 0.5 ✗)

    # مثال 3: Lower shadow خیلی کوچک
    # High: 100, Low: 90, Open: 90.1, Close: 90.5
    # Body = 0.4, Upper shadow = 9.5, Lower shadow = 0.1
    # upper_shadow_ratio = 9.5 / 0.4 = 23.75 (>> 1.5 ✓)
    # lower_shadow_ratio = 0.1 / 0.4 = 0.25 (< 0.5 ✓)
    # body_position = (90.1 - 90) / (100 - 90) = 0.1 / 10 = 0.01 (< 0.4 ✓)

    candles = []

    # Perfect Shooting Star #1
    candles.append({
        'high': 100.0,
        'low': 90.0,
        'open': 90.1,
        'close': 90.5,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 00:00:00'),
        'description': 'Perfect Shooting Star (upper=23.75x, lower=0.25x, pos=0.01)'
    })

    # Strong Shooting Star #2
    candles.append({
        'high': 50.0,
        'low': 45.0,
        'open': 45.2,
        'close': 45.5,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 01:00:00'),
        'description': 'Strong Shooting Star (upper=15x, lower=0.67x, pos=0.04) - Lower shadow کمی بزرگ'
    })

    # Borderline Shooting Star #3 (دقیقاً روی threshold)
    candles.append({
        'high': 60.0,
        'low': 50.0,
        'open': 50.5,
        'close': 51.0,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 02:00:00'),
        'description': 'Borderline (upper=18x, lower=1x, pos=0.05) - Lower shadow بزرگتر از threshold'
    })

    # Not a Shooting Star #4 (upper shadow کوتاه)
    candles.append({
        'high': 70.0,
        'low': 65.0,
        'open': 65.5,
        'close': 69.0,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 03:00:00'),
        'description': 'NOT Shooting Star (upper=0.29x, lower=0.14x) - Upper shadow خیلی کوتاه'
    })

    return candles


def test_shooting_star_detector():
    """تست detector با کندل‌های مصنوعی"""
    print("\n" + "="*80)
    print("🧪 DEBUG TEST: Shooting Star Detector")
    print("="*80 + "\n")

    # ساخت detector
    detector = ShootingStarPattern()

    print(f"Detector thresholds:")
    print(f"  min_upper_shadow_ratio: {detector.min_upper_shadow_ratio}")
    print(f"  max_lower_shadow_ratio: {detector.max_lower_shadow_ratio}")
    print(f"  max_body_position: {detector.max_body_position}")
    print(f"  version: {detector.version}\n")

    # ساخت کندل‌های تست
    test_candles = create_shooting_star_candle()

    print(f"Testing {len(test_candles)} synthetic candles:\n")

    detected_count = 0

    for i, candle_dict in enumerate(test_candles, 1):
        # ساخت DataFrame با کندل‌های قبلی (برای context)
        # باید حداقل 10 کندل داشته باشیم
        rows = []
        for j in range(10):
            rows.append({
                'open': 50.0,
                'high': 51.0,
                'low': 49.0,
                'close': 50.5,
                'volume': 1000,
                'timestamp': pd.Timestamp('2024-01-01 00:00:00') + pd.Timedelta(minutes=j*5)
            })

        # اضافه کردن کندل تست
        rows.append({
            'open': candle_dict['open'],
            'high': candle_dict['high'],
            'low': candle_dict['low'],
            'close': candle_dict['close'],
            'volume': candle_dict['volume'],
            'timestamp': candle_dict['timestamp']
        })

        df = pd.DataFrame(rows)

        # محاسبه دستی معیارها
        o = candle_dict['open']
        h = candle_dict['high']
        l = candle_dict['low']
        c = candle_dict['close']

        body_size = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        full_range = h - l

        body_for_ratio = max(body_size, full_range * 0.01)
        upper_ratio = upper_shadow / body_for_ratio
        lower_ratio = lower_shadow / body_for_ratio
        body_bottom = min(o, c)
        body_pos = (body_bottom - l) / full_range if full_range > 0 else 0

        print(f"Test #{i}: {candle_dict['description']}")
        print(f"  Candle: O={o}, H={h}, L={l}, C={c}")
        print(f"  Body: {body_size:.2f}, Upper shadow: {upper_shadow:.2f}, Lower shadow: {lower_shadow:.2f}")
        print(f"  Upper ratio: {upper_ratio:.2f}x (need >= {detector.min_upper_shadow_ratio})")
        print(f"  Lower ratio: {lower_ratio:.2f}x (need <= {detector.max_lower_shadow_ratio})")
        print(f"  Body position: {body_pos:.2f} (need <= {detector.max_body_position})")

        # تست detection
        result = detector.detect(df)

        if result:
            print(f"  ✅ DETECTED as Shooting Star")
            detected_count += 1
        else:
            print(f"  ❌ NOT detected")

            # تحلیل چرا detect نشد
            reasons = []
            if upper_ratio < detector.min_upper_shadow_ratio:
                reasons.append(f"Upper shadow کوتاه ({upper_ratio:.2f} < {detector.min_upper_shadow_ratio})")
            if lower_ratio > detector.max_lower_shadow_ratio:
                reasons.append(f"Lower shadow بلند ({lower_ratio:.2f} > {detector.max_lower_shadow_ratio})")
            if body_pos > detector.max_body_position:
                reasons.append(f"Body بالا ({body_pos:.2f} > {detector.max_body_position})")

            if reasons:
                print(f"  Reasons: {', '.join(reasons)}")

        print()

    print("="*80)
    print(f"SUMMARY: {detected_count}/{len(test_candles)} candles detected as Shooting Star")
    print("="*80)

    if detected_count == 0:
        print("\n⚠️  WARNING: No candles detected! There might be a bug in the detector!")
    elif detected_count >= 2:
        print("\n✅ Detector is working! At least some Shooting Stars were detected.")

    print()


if __name__ == '__main__':
    test_shooting_star_detector()
