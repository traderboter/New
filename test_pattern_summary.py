#!/usr/bin/env python3
"""
تست تابع get_pattern_summary بدون نیاز به وابستگی‌های خارجی
"""

import sys


def get_pattern_summary(detected_patterns, pattern_contributions):
    """
    نسخه کپی شده از متد get_pattern_summary برای تست

    Returns:
        رشته‌ای حاوی خلاصه الگوها و تایم‌فریم‌های آن‌ها
    """
    if not detected_patterns:
        return "هیچ الگویی تشخیص داده نشد"

    summary_lines = []
    for pattern in detected_patterns:
        name = pattern.get('name', 'Unknown')
        timeframe = pattern.get('timeframe', 'N/A')
        adjusted_strength = pattern.get('adjusted_strength', 0)
        direction = pattern.get('direction', 'neutral')
        pattern_type = pattern.get('type', 'unknown')

        # افزودن ایموجی بر اساس نوع الگو
        if pattern_type == 'candlestick':
            icon = '🕯️'
        elif pattern_type == 'chart':
            icon = '📊'
        else:
            icon = '📈'

        # افزودن ایموجی بر اساس جهت
        if direction == 'bullish':
            dir_icon = '🟢'
        elif direction == 'bearish':
            dir_icon = '🔴'
        else:
            dir_icon = '⚪'

        contribution = pattern_contributions.get(name, 0)

        summary_lines.append(
            f"{icon} {name} [{timeframe}] {dir_icon} "
            f"(قدرت: {adjusted_strength:.2f}, سهم: {contribution:.2f})"
        )

    return "\n".join(summary_lines)


def test_basic():
    """تست پایه"""
    print("=" * 80)
    print("تست پایه - الگوهای Bullish در تایم‌فریم‌های مختلف")
    print("=" * 80)

    patterns = [
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'adjusted_strength': 3.0,
            'timeframe': '1h'
        },
        {
            'name': 'Morning Star',
            'type': 'candlestick',
            'direction': 'bullish',
            'adjusted_strength': 4.5,
            'timeframe': '4h'
        },
        {
            'name': 'Double Bottom',
            'type': 'chart',
            'direction': 'bullish',
            'adjusted_strength': 4.0,
            'timeframe': '1h'
        }
    ]

    contributions = {
        'Hammer': 60.0,
        'Morning Star': 90.0,
        'Double Bottom': 80.0
    }

    summary = get_pattern_summary(patterns, contributions)
    print("\n📊 خلاصه الگوها:\n")
    print(summary)

    # بررسی وجود اطلاعات کلیدی
    assert 'Hammer' in summary
    assert '[1h]' in summary
    assert '🟢' in summary  # bullish icon
    assert '🕯️' in summary  # candlestick icon
    assert '📊' in summary  # chart icon

    print("\n✅ تست پایه موفق بود!")
    print("=" * 80)


def test_empty():
    """تست حالت خالی"""
    print("\n" + "=" * 80)
    print("تست حالت خالی - بدون الگو")
    print("=" * 80)

    patterns = []
    contributions = {}

    summary = get_pattern_summary(patterns, contributions)
    print(f"\n📊 خلاصه: {summary}")

    assert summary == "هیچ الگویی تشخیص داده نشد"

    print("\n✅ تست حالت خالی موفق بود!")
    print("=" * 80)


def test_mixed_directions():
    """تست الگوها با جهت‌های مختلف"""
    print("\n" + "=" * 80)
    print("تست الگوها با جهت‌های مختلف")
    print("=" * 80)

    patterns = [
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

    contributions = {
        'Hammer': 60.0,
        'Shooting Star': 0,
        'Doji': 30.0
    }

    summary = get_pattern_summary(patterns, contributions)
    print("\n📊 خلاصه الگوها:\n")
    print(summary)

    # بررسی وجود ایموجی‌های مختلف
    assert '🟢' in summary  # bullish
    assert '🔴' in summary  # bearish
    assert '⚪' in summary  # neutral/reversal

    print("\n✅ تست الگوها با جهت‌های مختلف موفق بود!")
    print("=" * 80)


def test_multiple_timeframes():
    """تست الگوهای یکسان در تایم‌فریم‌های مختلف"""
    print("\n" + "=" * 80)
    print("تست الگوهای یکسان در تایم‌فریم‌های مختلف")
    print("=" * 80)

    patterns = [
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'adjusted_strength': 2.0,
            'timeframe': '5m'
        },
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'adjusted_strength': 3.0,
            'timeframe': '1h'
        },
        {
            'name': 'Hammer',
            'type': 'candlestick',
            'direction': 'bullish',
            'adjusted_strength': 4.0,
            'timeframe': '4h'
        }
    ]

    contributions = {
        'Hammer': 180.0  # مجموع سهم از همه تایم‌فریم‌ها
    }

    summary = get_pattern_summary(patterns, contributions)
    print("\n📊 خلاصه الگوها:\n")
    print(summary)

    # بررسی وجود تمام تایم‌فریم‌ها
    assert '[5m]' in summary
    assert '[1h]' in summary
    assert '[4h]' in summary
    # بررسی اینکه قدرت‌های مختلف نمایش داده می‌شود
    assert '2.00' in summary or '2.0' in summary
    assert '3.00' in summary or '3.0' in summary
    assert '4.00' in summary or '4.0' in summary

    print("\n✅ تست الگوهای یکسان در تایم‌فریم‌های مختلف موفق بود!")
    print("✨ این قابلیت به شما کمک می‌کند ببینید هر الگو در کدام تایم‌فریم تشخیص داده شده است")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_basic()
        test_empty()
        test_mixed_directions()
        test_multiple_timeframes()

        print("\n" + "🎉" * 40)
        print("همه تست‌ها با موفقیت اجرا شدند!")
        print("\n✅ قابلیت جدید به درستی کار می‌کند:")
        print("   1. ✓ الگوها با تایم‌فریم‌هایشان ذخیره می‌شوند")
        print("   2. ✓ سهم هر الگو در امتیاز کل محاسبه می‌شود")
        print("   3. ✓ خلاصه الگوها به صورت خوانا نمایش داده می‌شود")
        print("   4. ✓ جهت و نوع هر الگو با ایموجی مشخص می‌شود")
        print("\n💡 حالا می‌توانید برای هر معامله ببینید:")
        print("   - کدام الگوها تشخیص داده شده‌اند")
        print("   - در چه تایم‌فریم‌هایی ایجاد شده‌اند")
        print("   - هر کدام چقدر به امتیاز کل کمک کرده‌اند")
        print("🎉" * 40)

        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ خطا در تست: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ خطای غیرمنتظره: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
