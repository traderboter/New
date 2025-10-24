"""
تست دیباگ برای Shooting Star - ساخت کندل‌های مصنوعی

این اسکریپت کندل‌های مصنوعی با شرایط روشن Shooting Star می‌سازد
و می‌بیند آیا detector آنها را تشخیص می‌دهد یا نه.

Version: 1.4.0 - Added uptrend detection requirement
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

    Shooting Star باید (range-based thresholds):
    - Upper shadow بلند (>= 50% of range)
    - Lower shadow کوچک (<= 20% of range)
    - Body کوچک (<= 30% of range)
    - Body در پایین (<= 40% from bottom)
    """
    # مثال 1: Perfect Shooting Star
    # High: 100, Low: 90, Range: 10
    # Open: 90.1, Close: 90.5, Body: 0.4 (4% of range)
    # Upper shadow: 9.5 (95% of range) ✓
    # Lower shadow: 0.1 (1% of range) ✓
    # Body position: 0.01 (1% from bottom) ✓

    # مثال 2: Strong Shooting Star
    # High: 50, Low: 45, Range: 5
    # Open: 45.2, Close: 45.5, Body: 0.3 (6% of range)
    # Upper shadow: 4.5 (90% of range) ✓
    # Lower shadow: 0.2 (4% of range) ✓
    # Body position: 0.04 (4% from bottom) ✓

    # مثال 3: Borderline Shooting Star (دقیقاً روی threshold)
    # High: 60, Low: 50, Range: 10
    # Open: 50.5, Close: 51.0, Body: 0.5 (5% of range)
    # Upper shadow: 9.0 (90% of range) ✓
    # Lower shadow: 0.5 (5% of range) ✓
    # Body position: 0.05 (5% from bottom) ✓

    # مثال 4: NOT a Shooting Star (upper shadow کوتاه)
    # High: 70, Low: 65, Range: 5
    # Open: 65.5, Close: 69.0, Body: 3.5 (70% of range)
    # Upper shadow: 1.0 (20% of range) ✗
    # Lower shadow: 0.5 (10% of range) ✓

    candles = []

    # Perfect Shooting Star #1
    candles.append({
        'high': 100.0,
        'low': 90.0,
        'open': 90.1,
        'close': 90.5,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 00:00:00'),
        'description': 'Perfect Shooting Star (upper=95%, lower=1%, body=4%, pos=1%)'
    })

    # Strong Shooting Star #2
    candles.append({
        'high': 50.0,
        'low': 45.0,
        'open': 45.2,
        'close': 45.5,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 01:00:00'),
        'description': 'Strong Shooting Star (upper=90%, lower=4%, body=6%, pos=4%)'
    })

    # Borderline Shooting Star #3 (دقیقاً روی threshold)
    candles.append({
        'high': 60.0,
        'low': 50.0,
        'open': 50.5,
        'close': 51.0,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 02:00:00'),
        'description': 'Borderline Shooting Star (upper=90%, lower=5%, body=5%, pos=5%)'
    })

    # Not a Shooting Star #4 (upper shadow کوتاه)
    candles.append({
        'high': 70.0,
        'low': 65.0,
        'open': 65.5,
        'close': 69.0,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 03:00:00'),
        'description': 'NOT Shooting Star (upper=20%, lower=10%, body=70%) - Upper shadow کوتاه'
    })

    return candles


def test_shooting_star_detector():
    """تست detector با کندل‌های مصنوعی"""
    print("\n" + "="*80)
    print("🧪 DEBUG TEST: Shooting Star Detector (v1.4.0)")
    print("="*80 + "\n")

    # ساخت detector با uptrend detection غیرفعال برای تست اولیه
    detector = ShootingStarPattern(require_uptrend=False)

    print(f"Detector thresholds (range-based):")
    print(f"  min_upper_shadow_pct: {detector.min_upper_shadow_pct} (>= {detector.min_upper_shadow_pct * 100}% of range)")
    print(f"  max_lower_shadow_pct: {detector.max_lower_shadow_pct} (<= {detector.max_lower_shadow_pct * 100}% of range)")
    print(f"  max_body_pct: {detector.max_body_pct} (<= {detector.max_body_pct * 100}% of range)")
    print(f"  max_body_position: {detector.max_body_position}")
    print(f"  require_uptrend: {detector.require_uptrend} (disabled for this test)")
    print(f"  min_uptrend_score: {detector.min_uptrend_score}")
    print(f"  version: {detector.version}\n")

    # ساخت کندل‌های تست
    test_candles = create_shooting_star_candle()

    print(f"Testing {len(test_candles)} synthetic candles:\n")

    detected_count = 0

    for i, candle_dict in enumerate(test_candles, 1):
        # ساخت DataFrame با کندل‌های قبلی (برای context)
        # باید حداقل 10 کندل داشته باشیم - uptrend مصنوعی
        rows = []
        for j in range(10):
            # ایجاد یک uptrend مصنوعی (قیمت‌ها رو به بالا)
            base_price = 40.0 + (j * 1.0)  # افزایش تدریجی قیمت
            rows.append({
                'open': base_price,
                'high': base_price + 1.0,
                'low': base_price - 0.5,
                'close': base_price + 0.5,
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

        # محاسبه دستی معیارها (range-based)
        o = candle_dict['open']
        h = candle_dict['high']
        l = candle_dict['low']
        c = candle_dict['close']

        body_size = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        full_range = h - l

        # محاسبه درصدها نسبت به full range
        upper_shadow_pct = upper_shadow / full_range if full_range > 0 else 0
        lower_shadow_pct = lower_shadow / full_range if full_range > 0 else 0
        body_pct = body_size / full_range if full_range > 0 else 0
        body_bottom = min(o, c)
        body_pos = (body_bottom - l) / full_range if full_range > 0 else 0

        print(f"Test #{i}: {candle_dict['description']}")
        print(f"  Candle: O={o}, H={h}, L={l}, C={c}, Range={full_range}")
        print(f"  Body: {body_size:.2f} ({body_pct*100:.1f}%), Upper shadow: {upper_shadow:.2f} ({upper_shadow_pct*100:.1f}%), Lower shadow: {lower_shadow:.2f} ({lower_shadow_pct*100:.1f}%)")
        print(f"  Upper shadow: {upper_shadow_pct*100:.1f}% (need >= {detector.min_upper_shadow_pct*100:.0f}%)")
        print(f"  Lower shadow: {lower_shadow_pct*100:.1f}% (need <= {detector.max_lower_shadow_pct*100:.0f}%)")
        print(f"  Body size: {body_pct*100:.1f}% (need <= {detector.max_body_pct*100:.0f}%)")
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
            if upper_shadow_pct < detector.min_upper_shadow_pct:
                reasons.append(f"Upper shadow کوتاه ({upper_shadow_pct*100:.1f}% < {detector.min_upper_shadow_pct*100:.0f}%)")
            if lower_shadow_pct > detector.max_lower_shadow_pct:
                reasons.append(f"Lower shadow بلند ({lower_shadow_pct*100:.1f}% > {detector.max_lower_shadow_pct*100:.0f}%)")
            if body_pct > detector.max_body_pct:
                reasons.append(f"Body بزرگ ({body_pct*100:.1f}% > {detector.max_body_pct*100:.0f}%)")
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
    else:
        print("\n⚠️  Only 1 candle detected - may need to adjust thresholds.")

    print()


def test_with_uptrend_detection():
    """تست detector با uptrend detection فعال"""
    print("\n" + "="*80)
    print("🧪 UPTREND TEST: Testing with uptrend detection ENABLED")
    print("="*80 + "\n")

    # ساخت detector با uptrend detection فعال
    detector = ShootingStarPattern(require_uptrend=True, min_uptrend_score=50.0)

    print(f"Detector settings:")
    print(f"  require_uptrend: {detector.require_uptrend} ✅ ENABLED")
    print(f"  min_uptrend_score: {detector.min_uptrend_score}\n")

    # تست 1: کندل Shooting Star در uptrend
    print("Test #1: Shooting Star in UPTREND (should be DETECTED)")
    rows = []
    for j in range(10):
        # uptrend قوی
        base_price = 40.0 + (j * 2.0)
        rows.append({
            'open': base_price,
            'high': base_price + 2.0,
            'low': base_price - 0.5,
            'close': base_price + 1.8,
            'volume': 1000,
            'timestamp': pd.Timestamp('2024-01-01 00:00:00') + pd.Timedelta(minutes=j*5)
        })
    # Shooting Star کندل
    rows.append({
        'open': 90.1,
        'high': 100.0,
        'low': 90.0,
        'close': 90.5,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 01:00:00')
    })
    df_uptrend = pd.DataFrame(rows)

    # محاسبه context score
    context_score = detector._analyze_context(df_uptrend)
    result_uptrend = detector.detect(df_uptrend)

    print(f"  Context score: {context_score:.1f} (min required: {detector.min_uptrend_score})")
    print(f"  Result: {'✅ DETECTED' if result_uptrend else '❌ NOT detected'}")
    print()

    # تست 2: کندل Shooting Star در downtrend/sideways
    print("Test #2: Shooting Star in DOWNTREND (should be REJECTED)")
    rows = []
    for j in range(10):
        # downtrend
        base_price = 60.0 - (j * 1.0)
        rows.append({
            'open': base_price,
            'high': base_price + 0.5,
            'low': base_price - 1.5,
            'close': base_price - 1.0,
            'volume': 1000,
            'timestamp': pd.Timestamp('2024-01-01 00:00:00') + pd.Timedelta(minutes=j*5)
        })
    # همان Shooting Star کندل
    rows.append({
        'open': 90.1,
        'high': 100.0,
        'low': 90.0,
        'close': 90.5,
        'volume': 1000,
        'timestamp': pd.Timestamp('2024-01-01 01:00:00')
    })
    df_downtrend = pd.DataFrame(rows)

    context_score = detector._analyze_context(df_downtrend)
    result_downtrend = detector.detect(df_downtrend)

    print(f"  Context score: {context_score:.1f} (min required: {detector.min_uptrend_score})")
    print(f"  Result: {'✅ DETECTED' if result_downtrend else '❌ NOT detected'}")
    print()

    print("="*80)
    if result_uptrend and not result_downtrend:
        print("✅ UPTREND DETECTION WORKING CORRECTLY!")
        print("   - Shooting Star detected in uptrend")
        print("   - Shooting Star rejected in downtrend")
    else:
        print("⚠️  Unexpected results - check uptrend detection logic")
    print("="*80)
    print()


if __name__ == '__main__':
    # Test 1: کندل‌های مصنوعی بدون uptrend check
    test_shooting_star_detector()

    # Test 2: تست با uptrend detection
    test_with_uptrend_detection()
