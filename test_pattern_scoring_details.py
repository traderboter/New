#!/usr/bin/env python3
"""
تست سیستم جزئیات امتیازدهی الگوها

این اسکریپت برای تست کردن قابلیت جدید نمایش جزئیات الگوها نوشته شده است.
"""

import sys
from signal_generation.signal_score import SignalScore


def test_pattern_details():
    """تست ذخیره و نمایش جزئیات الگوها"""

    print("=" * 80)
    print("تست سیستم جزئیات امتیازدهی الگوها")
    print("=" * 80)

    # ایجاد یک SignalScore نمونه
    score = SignalScore()

    # اضافه کردن امتیازات پایه
    score.trend_score = 80.0
    score.momentum_score = 75.0
    score.volume_score = 60.0
    score.pattern_score = 85.0

    # ✨ اضافه کردن الگوهای تشخیص داده شده
    score.detected_patterns = [
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'base_strength': 2.0,
            'adjusted_strength': 3.0,
            'location': 'current',
            'timeframe': '1h',  # ✅ تایم‌فریم مشخص است
            'candles_ago': 0
        },
        {
            'name': 'Morning Star',
            'type': 'candlestick',
            'direction': 'bullish',
            'base_strength': 3.0,
            'adjusted_strength': 4.5,
            'location': 'current',
            'timeframe': '4h',  # ✅ تایم‌فریم مشخص است
            'candles_ago': 0
        },
        {
            'name': 'Double Bottom',
            'type': 'chart',
            'direction': 'bullish',
            'base_strength': 3.0,
            'adjusted_strength': 4.0,
            'location': 'recent',
            'timeframe': '1h',  # ✅ تایم‌فریم مشخص است
            'completion': 0.8
        }
    ]

    # ✨ اضافه کردن سهم هر الگو
    score.pattern_contributions = {
        'Hammer': 60.0,
        'Morning Star': 90.0,
        'Double Bottom': 80.0
    }

    # محاسبه امتیاز نهایی
    score.contributing_analyzers = ['trend', 'momentum', 'volume', 'patterns']
    score.aligned_analyzers = 4
    score.calculate_final_score()
    score.determine_signal_strength()
    score.calculate_confidence()
    score.build_breakdown()

    # نمایش نتایج
    print("\n📊 اطلاعات امتیاز:")
    print(f"   امتیاز نهایی: {score.final_score:.2f}")
    print(f"   قدرت سیگنال: {score.signal_strength}")
    print(f"   اطمینان: {score.confidence:.2f}")
    print(f"   تحلیل‌گرهای مشارکت‌کننده: {', '.join(score.contributing_analyzers)}")

    # ✨ نمایش جزئیات الگوها
    print("\n📈 جزئیات الگوهای تشخیص داده شده:")
    print(score.get_pattern_summary())

    # نمایش breakdown کامل
    print("\n🔍 تفصیل کامل امتیازدهی:")
    import json
    print(json.dumps(score.breakdown, indent=2, ensure_ascii=False))

    # بررسی وجود اطلاعات الگوها در breakdown
    assert 'patterns' in score.breakdown, "❌ فیلد patterns در breakdown وجود ندارد!"
    assert 'detected' in score.breakdown['patterns'], "❌ فیلد detected در patterns وجود ندارد!"
    assert 'contributions' in score.breakdown['patterns'], "❌ فیلد contributions در patterns وجود ندارد!"

    print("\n✅ همه تست‌ها با موفقیت انجام شد!")
    print("=" * 80)

    return True


def test_empty_patterns():
    """تست حالتی که هیچ الگویی تشخیص داده نشده"""

    print("\n" + "=" * 80)
    print("تست حالت بدون الگو")
    print("=" * 80)

    score = SignalScore()
    score.trend_score = 70.0
    score.momentum_score = 65.0
    score.detected_patterns = []
    score.pattern_contributions = {}

    score.calculate_final_score()
    score.build_breakdown()

    print("\n📊 خلاصه الگوها:")
    summary = score.get_pattern_summary()
    print(summary)

    assert summary == "هیچ الگویی تشخیص داده نشد", "❌ پیام خلاصه برای حالت بدون الگو صحیح نیست!"

    print("\n✅ تست حالت بدون الگو موفق بود!")
    print("=" * 80)

    return True


def test_mixed_direction_patterns():
    """تست الگوهای با جهت‌های مختلف"""

    print("\n" + "=" * 80)
    print("تست الگوهای با جهت‌های مختلف")
    print("=" * 80)

    score = SignalScore()

    score.detected_patterns = [
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'adjusted_strength': 3.0,
            'timeframe': '15m'
        },
        {
            'name': 'Shooting Star',
            'type': 'candlestick',
            'direction': 'bearish',
            'adjusted_strength': 2.5,
            'timeframe': '1h'
        },
        {
            'name': 'Doji',
            'type': 'candlestick',
            'direction': 'reversal',
            'adjusted_strength': 1.5,
            'timeframe': '5m'
        }
    ]

    score.pattern_contributions = {
        'Hammer': 60.0,
        'Shooting Star': 0,  # این الگو در جهت مخالف است و سهمی ندارد
        'Doji': 30.0
    }

    score.build_breakdown()

    print("\n📊 خلاصه الگوها:")
    print(score.get_pattern_summary())

    print("\n✅ تست الگوهای با جهت‌های مختلف موفق بود!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        # اجرای تست‌ها
        test_pattern_details()
        test_empty_patterns()
        test_mixed_direction_patterns()

        print("\n" + "🎉" * 40)
        print("همه تست‌ها با موفقیت اجرا شدند!")
        print("سیستم جزئیات امتیازدهی الگوها به درستی کار می‌کند.")
        print("🎉" * 40)

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ خطا در اجرای تست‌ها: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
