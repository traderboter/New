"""
Test with Real Data: آزمایش الگو با داده واقعی BTC

این اسکریپت:
1. داده واقعی BTC را می‌خواند
2. یک detection واقعی را پیدا می‌کند
3. با تعداد مختلف کندل تست می‌کند
4. حداقل تعداد کندل مورد نیاز را پیدا می‌کند

نویسنده: Test Team
تاریخ: 2025-10-25
"""

import talib
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# تنظیمات
# =============================================================================

PATTERN_TO_TEST = "HAMMER"  # یا: ENGULFING, SHOOTINGSTAR, DOJI, ...

PATTERN_INFO = {
    # ✅ الگوهای تست شده قبلی
    "ENGULFING": {
        "name": "Engulfing",
        "talib_func": talib.CDLENGULFING,
    },
    "HAMMER": {
        "name": "Hammer",
        "talib_func": talib.CDLHAMMER,
    },
    "SHOOTINGSTAR": {
        "name": "Shooting Star",
        "talib_func": talib.CDLSHOOTINGSTAR,
    },
    "DOJI": {
        "name": "Doji",
        "talib_func": talib.CDLDOJI,
    },
    "MORNINGSTAR": {
        "name": "Morning Star",
        "talib_func": talib.CDLMORNINGSTAR,
    },
    "INVERTEDHAMMER": {
        "name": "Inverted Hammer",
        "talib_func": talib.CDLINVERTEDHAMMER,
    },

    # 🆕 الگوهای جدید برای تست
    "DARKCLOUDCOVER": {
        "name": "Dark Cloud Cover",
        "talib_func": talib.CDLDARKCLOUDCOVER,
    },
    "EVENINGSTAR": {
        "name": "Evening Star",
        "talib_func": talib.CDLEVENINGSTAR,
    },
    "EVENINGDOJISTAR": {
        "name": "Evening Doji Star",
        "talib_func": talib.CDLEVENINGDOJISTAR,
    },
    "HARAMI": {
        "name": "Harami",
        "talib_func": talib.CDLHARAMI,
    },
    "HARAMICROSS": {
        "name": "Harami Cross",
        "talib_func": talib.CDLHARAMICROSS,
    },
    "HANGINGMAN": {
        "name": "Hanging Man",
        "talib_func": talib.CDLHANGINGMAN,
    },
    "PIERCINGLINE": {
        "name": "Piercing Line",
        "talib_func": talib.CDLPIERCING,
    },
    "MORNINGDOJISTAR": {
        "name": "Morning Doji Star",
        "talib_func": talib.CDLMORNINGDOJISTAR,
    },
    "THREEWHITESOLDIERS": {
        "name": "Three White Soldiers",
        "talib_func": talib.CDL3WHITESOLDIERS,
    },
    "THREEBLACKCROWS": {
        "name": "Three Black Crows",
        "talib_func": talib.CDL3BLACKCROWS,
    },
}

# =============================================================================
# بارگذاری داده
# =============================================================================

def load_btc_data():
    """Load BTC 1-hour data"""
    csv_path = Path(__file__).parent.parent / 'historical' / 'BTC-USDT' / '1hour.csv'

    if not csv_path.exists():
        print(f"❌ ERROR: {csv_path} not found!")
        return None

    df = pd.read_csv(csv_path)
    df = df.astype({
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    })

    print(f"✅ Loaded {len(df)} BTC candles")
    return df

# =============================================================================
# تست
# =============================================================================

def find_first_detection(df, pattern_func):
    """پیدا کردن اولین detection در داده واقعی"""

    print("\n🔍 جستجو برای اولین detection در داده واقعی...")

    result = pattern_func(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    # پیدا کردن اولین detection
    detections = np.where(result != 0)[0]

    if len(detections) == 0:
        print("❌ هیچ detection در داده واقعی پیدا نشد!")
        return None

    first_idx = detections[0]
    print(f"✅ اولین detection در index {first_idx} پیدا شد")

    return first_idx

def show_candle_details(df, idx):
    """نمایش جزئیات یک کندل"""

    candle = df.iloc[idx]

    print(f"\n📊 کندل در index {idx}:")
    print("="*60)
    print(f"  Open:   {candle['open']:.2f}")
    print(f"  High:   {candle['high']:.2f}")
    print(f"  Low:    {candle['low']:.2f}")
    print(f"  Close:  {candle['close']:.2f}")

    # محاسبه اندازه‌ها
    body = abs(candle['close'] - candle['open'])
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    full_range = candle['high'] - candle['low']

    print(f"\n  Body size:     {body:.2f} ({body/full_range*100:.1f}%)")
    print(f"  Upper shadow:  {upper_shadow:.2f} ({upper_shadow/full_range*100:.1f}%)")
    print(f"  Lower shadow:  {lower_shadow:.2f} ({lower_shadow/full_range*100:.1f}%)")
    print(f"  Full range:    {full_range:.2f}")

    direction = "صعودی" if candle['close'] > candle['open'] else "نزولی" if candle['close'] < candle['open'] else "Doji"
    print(f"  Direction:     {direction}")

def test_minimum_lookback(df, detection_idx, pattern_func, pattern_name):
    """
    تست حداقل تعداد کندل مورد نیاز

    با استفاده از یک detection واقعی، تعداد کندل را کم می‌کنیم
    تا ببینیم با چند کندل minimum کار می‌کند.
    """

    print("\n" + "="*60)
    print(f"🔬 تست Minimum Lookback: {pattern_name}")
    print("="*60)

    print(f"\n📌 Detection index: {detection_idx}")
    print(f"📌 تعداد کندل موجود قبل از detection: {detection_idx}")

    # نمایش جزئیات کندل
    show_candle_details(df, detection_idx)

    # تست با تعداد مختلف کندل
    print(f"\n🧪 تست با تعداد کندل‌های مختلف:")
    print("-"*60)

    lookback_values = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 20, 30, 50]
    minimum_found = None

    for lookback in lookback_values:
        # اگر lookback بیشتر از کندل‌های موجود باشد
        if lookback > detection_idx:
            continue

        # گرفتن کندل‌ها از lookback تا detection_idx
        start_idx = detection_idx - lookback
        df_test = df.iloc[start_idx:detection_idx + 1].copy()

        try:
            result = pattern_func(
                df_test['open'].values,
                df_test['high'].values,
                df_test['low'].values,
                df_test['close'].values
            )

            # چک کردن آخرین کندل
            detected = result[-1] != 0

            icon = "✅" if detected else "❌"
            print(f"{icon} Lookback: {lookback:2d} | کل کندل: {len(df_test):2d} | "
                  f"تشخیص: {'YES' if detected else 'NO'}")

            if detected and minimum_found is None:
                minimum_found = lookback
                print(f"   ⭐ حداقل lookback پیدا شد: {lookback}")

        except Exception as e:
            print(f"❌ Lookback: {lookback:2d} | خطا: {str(e)}")

    # نتیجه
    print("\n" + "="*60)
    print("📊 نتیجه:")
    print("="*60)

    if minimum_found is not None:
        print(f"\n✅ حداقل lookback: {minimum_found}")
        print(f"✅ حداقل کل کندل: {minimum_found + 1}")

        if minimum_found == 0:
            print(f"\n💡 این الگو با فقط 1 کندل کار می‌کند!")
        else:
            print(f"\n💡 این الگو به {minimum_found} کندل قبلی نیاز دارد")
    else:
        print("\n❌ حداقل lookback پیدا نشد!")
        print("   ممکن است به کندل‌های بیشتری نیاز باشد")

    return minimum_found

def test_multiple_detections(df, pattern_func, pattern_name, num_samples=10):
    """
    تست با چند detection مختلف

    برای اطمینان بیشتر، چند detection را تست می‌کنیم
    """

    print("\n" + "="*60)
    print(f"🔬 تست با {num_samples} Detection مختلف")
    print("="*60)

    # پیدا کردن همه detections
    result = pattern_func(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    detection_indices = np.where(result != 0)[0]

    if len(detection_indices) == 0:
        print("❌ هیچ detection پیدا نشد!")
        return []

    print(f"\n✅ تعداد کل detections: {len(detection_indices)}")

    # انتخاب تصادفی
    sample_size = min(num_samples, len(detection_indices))
    sample_indices = np.random.choice(detection_indices, sample_size, replace=False)

    print(f"🎲 انتخاب {sample_size} detection تصادفی برای تست...")

    minimum_lookbacks = []

    for i, det_idx in enumerate(sample_indices, 1):
        print(f"\n--- Detection {i}/{sample_size} (index: {det_idx}) ---")

        # تست سریع با lookback های مختلف
        lookback_values = [0, 1, 2, 5, 10, 11, 12, 15]
        minimum = None

        for lookback in lookback_values:
            if lookback > det_idx:
                continue

            start_idx = det_idx - lookback
            df_test = df.iloc[start_idx:det_idx + 1].copy()

            try:
                result_test = pattern_func(
                    df_test['open'].values,
                    df_test['high'].values,
                    df_test['low'].values,
                    df_test['close'].values
                )

                if result_test[-1] != 0:
                    minimum = lookback
                    print(f"  ✅ Minimum lookback: {lookback}")
                    break

            except:
                pass

        if minimum is not None:
            minimum_lookbacks.append(minimum)
        else:
            print(f"  ⚠️ Minimum lookback پیدا نشد")

    # خلاصه
    print("\n" + "="*60)
    print("📊 خلاصه نتایج:")
    print("="*60)

    if len(minimum_lookbacks) > 0:
        print(f"\n✅ تعداد نمونه‌های موفق: {len(minimum_lookbacks)}/{sample_size}")
        print(f"📊 آمار:")
        print(f"   - میانگین: {np.mean(minimum_lookbacks):.1f}")
        print(f"   - میانه: {np.median(minimum_lookbacks):.1f}")
        print(f"   - حداقل: {np.min(minimum_lookbacks)}")
        print(f"   - حداکثر: {np.max(minimum_lookbacks)}")
        print(f"\n💡 پیشنهاد: استفاده از حداقل {int(np.max(minimum_lookbacks))} lookback")
    else:
        print("\n❌ هیچ نمونه موفقی پیدا نشد!")

    return minimum_lookbacks

# =============================================================================
# MAIN
# =============================================================================

def main():
    """اجرای تست"""

    print("="*60)
    print("🔬 تست با داده واقعی BTC")
    print("="*60)

    # بررسی الگو
    if PATTERN_TO_TEST not in PATTERN_INFO:
        print(f"\n❌ الگوی '{PATTERN_TO_TEST}' پیدا نشد!")
        print(f"الگوهای موجود: {list(PATTERN_INFO.keys())}")
        return

    info = PATTERN_INFO[PATTERN_TO_TEST]
    print(f"\n📌 الگوی انتخابی: {info['name']}")

    # بارگذاری داده
    df = load_btc_data()
    if df is None:
        return

    # پیدا کردن اولین detection
    first_idx = find_first_detection(df, info['talib_func'])
    if first_idx is None:
        return

    # تست minimum lookback
    minimum = test_minimum_lookback(df, first_idx, info['talib_func'], info['name'])

    # تست با چند detection مختلف
    minimum_lookbacks = test_multiple_detections(df, info['talib_func'], info['name'], num_samples=10)

    print("\n" + "="*60)
    print("✅ تست کامل شد")
    print("="*60)

    # راهنما
    print("\n💡 برای تست الگوی دیگر:")
    print("   1. در خط 23 فایل، PATTERN_TO_TEST را تغییر دهید")
    print("   2. الگوهای موجود:")
    for key, val in PATTERN_INFO.items():
        print(f"      - {key}: {val['name']}")
    print("\n   3. دوباره اسکریپت را اجرا کنید:")
    print("      python3 test_pattern_with_real_data.py")

if __name__ == '__main__':
    main()
