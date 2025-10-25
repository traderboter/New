"""
Test Simple: آزمایش یک الگو با داده‌های ساختگی

این اسکریپت برای تست یک الگوی خاص با داده‌های دست‌ساز است.
هدف: پیدا کردن حداقل تعداد کندل مورد نیاز برای تشخیص الگو

روش کار:
1. یک الگوی کامل و واضح می‌سازیم
2. با 1 کندل شروع می‌کنیم
3. تدریجی کندل اضافه می‌کنیم
4. می‌بینیم با چند کندل، TA-Lib الگو را تشخیص می‌دهد

نویسنده: Test Team
تاریخ: 2025-10-25
"""

import talib
import pandas as pd
import numpy as np

# =============================================================================
# انتخاب الگو برای تست
# =============================================================================

# شما می‌توانید الگوی مختلف را تست کنید:
PATTERN_TO_TEST = "HAMMER"  # یا: HAMMER, SHOOTINGSTAR, DOJI, MORNINGSTAR, ...

PATTERN_INFO = {
    "ENGULFING": {
        "name": "Engulfing (صعودی)",
        "talib_func": talib.CDLENGULFING,
        "description": "2-candle pattern: کندل دوم بدنه کندل اول را کاملاً می‌بلعد"
    },
    "HAMMER": {
        "name": "Hammer (چکش)",
        "talib_func": talib.CDLHAMMER,
        "description": "1-candle pattern: بدنه کوچک، سایه پایین بلند"
    },
    "SHOOTINGSTAR": {
        "name": "Shooting Star (ستاره دنباله‌دار)",
        "talib_func": talib.CDLSHOOTINGSTAR,
        "description": "1-candle pattern: بدنه کوچک، سایه بالا بلند"
    },
    "DOJI": {
        "name": "Doji",
        "talib_func": talib.CDLDOJI,
        "description": "1-candle pattern: open ≈ close"
    },
    "MORNINGSTAR": {
        "name": "Morning Star (ستاره صبحگاهی)",
        "talib_func": talib.CDLMORNINGSTAR,
        "description": "3-candle pattern: نزولی، Doji، صعودی"
    },
}

# =============================================================================
# ساخت داده‌های تست
# =============================================================================

def create_bullish_engulfing():
    """
    ساخت یک الگوی Bullish Engulfing واضح

    الگو:
    - کندل 1: نزولی (قرمز) - open=105, close=100
    - کندل 2: صعودی (سبز) که کندل 1 را می‌بلعد - open=98, close=108
    """
    data = {
        'open':   [105.0, 98.0],
        'high':   [106.0, 110.0],
        'low':    [99.0,  97.0],
        'close':  [100.0, 108.0],
    }
    df = pd.DataFrame(data)

    print("\n📊 الگوی ساخته شده: Bullish Engulfing")
    print("="*60)
    print(df)
    print("\nتوضیح:")
    print("  کندل 0: نزولی (open=105 > close=100)")
    print("  کندل 1: صعودی (open=98 < close=108)")
    print("  → کندل 2 کندل 1 را کاملاً می‌بلعد ✅")

    return df

def create_hammer():
    """
    ساخت یک الگوی Hammer واضح

    الگو:
    - بدنه کوچک در بالای کندل
    - سایه پایین بلند (حداقل 2× بدنه)
    - سایه بالا کوچک یا صفر
    """
    data = {
        'open':   [102.0],
        'high':   [103.0],
        'low':    [95.0],   # سایه پایین بلند
        'close':  [101.0],
    }
    df = pd.DataFrame(data)

    body = abs(102.0 - 101.0)  # 1
    lower_shadow = 101.0 - 95.0  # 6
    upper_shadow = 103.0 - 102.0  # 1

    print("\n📊 الگوی ساخته شده: Hammer")
    print("="*60)
    print(df)
    print("\nتوضیح:")
    print(f"  Body size: {body}")
    print(f"  Lower shadow: {lower_shadow} (= {lower_shadow/body:.1f}× body)")
    print(f"  Upper shadow: {upper_shadow}")
    print("  → سایه پایین بلند، بدنه کوچک ✅")

    return df

def create_shooting_star():
    """
    ساخت یک الگوی Shooting Star واضح

    الگو:
    - بدنه کوچک در پایین کندل
    - سایه بالا بلند (حداقل 2× بدنه)
    - سایه پایین کوچک یا صفر
    """
    data = {
        'open':   [100.0],
        'high':   [110.0],  # سایه بالا بلند
        'low':    [99.0],
        'close':  [101.0],
    }
    df = pd.DataFrame(data)

    body = abs(101.0 - 100.0)  # 1
    upper_shadow = 110.0 - 101.0  # 9
    lower_shadow = 100.0 - 99.0  # 1

    print("\n📊 الگوی ساخته شده: Shooting Star")
    print("="*60)
    print(df)
    print("\nتوضیح:")
    print(f"  Body size: {body}")
    print(f"  Upper shadow: {upper_shadow} (= {upper_shadow/body:.1f}× body)")
    print(f"  Lower shadow: {lower_shadow}")
    print("  → سایه بالا بلند، بدنه کوچک ✅")

    return df

def create_doji():
    """
    ساخت یک الگوی Doji واضح

    الگو:
    - open ≈ close (تقریباً برابر)
    - سایه‌های بالا و پایین
    """
    data = {
        'open':   [100.0],
        'high':   [105.0],
        'low':    [95.0],
        'close':  [100.1],  # تقریباً برابر با open
    }
    df = pd.DataFrame(data)

    body = abs(100.1 - 100.0)
    full_range = 105.0 - 95.0

    print("\n📊 الگوی ساخته شده: Doji")
    print("="*60)
    print(df)
    print("\nتوضیح:")
    print(f"  Body size: {body}")
    print(f"  Full range: {full_range}")
    print(f"  Body ratio: {body/full_range*100:.1f}%")
    print("  → open ≈ close ✅")

    return df

def create_morning_star():
    """
    ساخت یک الگوی Morning Star واضح

    الگو:
    - کندل 1: نزولی بزرگ
    - کندل 2: Doji/کوچک (gap down)
    - کندل 3: صعودی بزرگ
    """
    data = {
        'open':   [110.0, 95.0,  96.0],
        'high':   [112.0, 97.0,  110.0],
        'low':    [94.0,  94.0,  95.0],
        'close':  [95.0,  96.0,  109.0],
    }
    df = pd.DataFrame(data)

    print("\n📊 الگوی ساخته شده: Morning Star")
    print("="*60)
    print(df)
    print("\nتوضیح:")
    print("  کندل 0: نزولی بزرگ (110 → 95)")
    print("  کندل 1: کوچک/Doji (95 → 96)")
    print("  کندل 2: صعودی بزرگ (96 → 109)")
    print("  → Morning Star pattern ✅")

    return df

# =============================================================================
# تابع اصلی تست
# =============================================================================

def add_neutral_candles(df, count):
    """
    اضافه کردن کندل‌های خنثی به ابتدای DataFrame

    این کندل‌ها فقط برای context هستند و الگوی خاصی ندارند.
    """
    neutral_data = []

    for i in range(count):
        # کندل‌های خنثی با تغییرات جزئی
        price = 100.0 + (i % 5)
        neutral_data.append({
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price + 0.5,
        })

    df_neutral = pd.DataFrame(neutral_data)
    df_combined = pd.concat([df_neutral, df], ignore_index=True)

    return df_combined

def test_pattern_with_different_candle_counts(pattern_df, pattern_func, pattern_name):
    """
    تست الگو با تعداد مختلف کندل

    Args:
        pattern_df: DataFrame حاوی الگو
        pattern_func: تابع TA-Lib (مثل talib.CDLENGULFING)
        pattern_name: نام الگو برای نمایش
    """

    print("\n" + "="*60)
    print(f"🔬 شروع تست: {pattern_name}")
    print("="*60)

    # تعداد کندل‌های الگو
    pattern_size = len(pattern_df)
    print(f"\n📌 تعداد کندل‌های الگو: {pattern_size}")

    # لیست تعداد کندل‌های قبلی برای تست
    # منطقی است که با 0 شروع کنیم و تدریجی افزایش دهیم
    previous_candle_counts = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 20, 30]

    print(f"\n🧪 تست با تعداد کندل‌های قبلی مختلف:")
    print("-"*60)

    results = []
    minimum_found = None

    for prev_count in previous_candle_counts:
        total_candles = prev_count + pattern_size

        # اضافه کردن کندل‌های قبلی
        df_test = add_neutral_candles(pattern_df.copy(), prev_count)

        # فراخوانی TA-Lib
        try:
            result = pattern_func(
                df_test['open'].values,
                df_test['high'].values,
                df_test['low'].values,
                df_test['close'].values
            )

            # چک کردن آخرین کندل
            detected = result[-1] != 0

            # نمایش نتیجه
            icon = "✅" if detected else "❌"
            print(f"{icon} کندل‌های قبلی: {prev_count:2d} | کل کندل: {total_candles:2d} | "
                  f"تشخیص: {'YES' if detected else 'NO'}")

            results.append({
                'prev_candles': prev_count,
                'total_candles': total_candles,
                'detected': detected
            })

            # اولین بار که تشخیص داده شد
            if detected and minimum_found is None:
                minimum_found = prev_count
                print(f"   ⭐ اولین تشخیص با {prev_count} کندل قبلی!")

        except Exception as e:
            print(f"❌ کندل‌های قبلی: {prev_count:2d} | کل کندل: {total_candles:2d} | "
                  f"خطا: {str(e)}")

    # خلاصه نتایج
    print("\n" + "="*60)
    print("📊 خلاصه نتایج:")
    print("="*60)

    if minimum_found is not None:
        print(f"\n✅ حداقل کندل قبلی لازم: {minimum_found}")
        print(f"✅ حداقل کل کندل لازم: {minimum_found + pattern_size}")

        if minimum_found == 0:
            print(f"\n💡 نتیجه: این الگو با فقط {pattern_size} کندل (خود الگو) کار می‌کند!")
        else:
            print(f"\n💡 نتیجه: این الگو به {minimum_found} کندل قبلی + {pattern_size} کندل الگو نیاز دارد")
            print(f"   یعنی TA-Lib از کندل‌های قبلی برای context استفاده می‌کند")
    else:
        print("\n❌ الگو در هیچ حالتی تشخیص داده نشد!")
        print("   ممکن است:")
        print("   - الگوی ساخته شده اشتباه باشد")
        print("   - به کندل‌های بیشتری نیاز باشد")
        print("   - TA-Lib معیارهای دقیق‌تری دارد")

    return results, minimum_found

# =============================================================================
# MAIN
# =============================================================================

def main():
    """اجرای تست برای الگوی انتخابی"""

    print("="*60)
    print("🔬 تست ساده: بررسی حداقل کندل مورد نیاز برای تشخیص الگو")
    print("="*60)

    # بررسی الگوی انتخابی
    if PATTERN_TO_TEST not in PATTERN_INFO:
        print(f"\n❌ خطا: الگوی '{PATTERN_TO_TEST}' پیدا نشد!")
        print(f"الگوهای موجود: {list(PATTERN_INFO.keys())}")
        return

    info = PATTERN_INFO[PATTERN_TO_TEST]
    print(f"\n📌 الگوی انتخابی: {info['name']}")
    print(f"📝 توضیح: {info['description']}")

    # ساخت داده‌های تست بر اساس الگو
    if PATTERN_TO_TEST == "ENGULFING":
        pattern_df = create_bullish_engulfing()
    elif PATTERN_TO_TEST == "HAMMER":
        pattern_df = create_hammer()
    elif PATTERN_TO_TEST == "SHOOTINGSTAR":
        pattern_df = create_shooting_star()
    elif PATTERN_TO_TEST == "DOJI":
        pattern_df = create_doji()
    elif PATTERN_TO_TEST == "MORNINGSTAR":
        pattern_df = create_morning_star()
    else:
        print(f"\n❌ تابع ساخت داده برای {PATTERN_TO_TEST} هنوز پیاده‌سازی نشده است!")
        return

    # تست الگو
    results, minimum = test_pattern_with_different_candle_counts(
        pattern_df,
        info['talib_func'],
        info['name']
    )

    print("\n" + "="*60)
    print("✅ تست به پایان رسید")
    print("="*60)

    # راهنمای استفاده
    print("\n💡 برای تست الگوی دیگر:")
    print("   1. در خط 25 این فایل، PATTERN_TO_TEST را تغییر دهید")
    print("   2. الگوهای موجود:")
    for key, val in PATTERN_INFO.items():
        print(f"      - {key}: {val['name']}")
    print("\n   3. دوباره اسکریپت را اجرا کنید:")
    print("      python3 test_single_pattern_simple.py")

if __name__ == '__main__':
    main()
