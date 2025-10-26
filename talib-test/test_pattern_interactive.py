#!/usr/bin/env python3
"""
Interactive Pattern Test: تست تعاملی الگوها

این اسکریپت به شما اجازه می‌دهد الگوی دلخواه را انتخاب کنید
و تست کامل انجام دهید.

نویسنده: Test Team
تاریخ: 2025-10-26
"""

import talib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# =============================================================================
# تنظیمات
# =============================================================================

PATTERN_INFO = {
    # الگوهای اولیه
    "ENGULFING": {"name": "Engulfing", "talib_func": talib.CDLENGULFING, "category": "قدیمی"},
    "HAMMER": {"name": "Hammer", "talib_func": talib.CDLHAMMER, "category": "قدیمی"},
    "SHOOTINGSTAR": {"name": "Shooting Star", "talib_func": talib.CDLSHOOTINGSTAR, "category": "قدیمی"},
    "DOJI": {"name": "Doji", "talib_func": talib.CDLDOJI, "category": "قدیمی"},
    "MORNINGSTAR": {"name": "Morning Star", "talib_func": talib.CDLMORNINGSTAR, "category": "قدیمی"},
    "INVERTEDHAMMER": {"name": "Inverted Hammer", "talib_func": talib.CDLINVERTEDHAMMER, "category": "قدیمی"},
    "DARKCLOUDCOVER": {"name": "Dark Cloud Cover", "talib_func": talib.CDLDARKCLOUDCOVER, "category": "قدیمی"},
    "EVENINGSTAR": {"name": "Evening Star", "talib_func": talib.CDLEVENINGSTAR, "category": "قدیمی"},
    "EVENINGDOJISTAR": {"name": "Evening Doji Star", "talib_func": talib.CDLEVENINGDOJISTAR, "category": "قدیمی"},
    "HARAMI": {"name": "Harami", "talib_func": talib.CDLHARAMI, "category": "قدیمی"},
    "HARAMICROSS": {"name": "Harami Cross", "talib_func": talib.CDLHARAMICROSS, "category": "قدیمی"},
    "HANGINGMAN": {"name": "Hanging Man", "talib_func": talib.CDLHANGINGMAN, "category": "قدیمی"},
    "PIERCINGLINE": {"name": "Piercing Line", "talib_func": talib.CDLPIERCING, "category": "قدیمی"},
    "MORNINGDOJISTAR": {"name": "Morning Doji Star", "talib_func": talib.CDLMORNINGDOJISTAR, "category": "قدیمی"},
    "THREEWHITESOLDIERS": {"name": "Three White Soldiers", "talib_func": talib.CDL3WHITESOLDIERS, "category": "قدیمی"},
    "THREEBLACKCROWS": {"name": "Three Black Crows", "talib_func": talib.CDL3BLACKCROWS, "category": "قدیمی"},

    # Phase 1 - الگوهای قدرتمند جدید
    "MARUBOZU": {"name": "Marubozu", "talib_func": talib.CDLMARUBOZU, "category": "Phase 1"},
    "DRAGONFLYDOJI": {"name": "Dragonfly Doji", "talib_func": talib.CDLDRAGONFLYDOJI, "category": "Phase 1"},
    "GRAVESTONEDOJI": {"name": "Gravestone Doji", "talib_func": talib.CDLGRAVESTONEDOJI, "category": "Phase 1"},
    "SPINNINGTOP": {"name": "Spinning Top", "talib_func": talib.CDLSPINNINGTOP, "category": "Phase 1"},
    "LONGLEGGEDDOJI": {"name": "Long-Legged Doji", "talib_func": talib.CDLLONGLEGGEDDOJI, "category": "Phase 1"},

    # Phase 2 - الگوهای ادامه و تایید جدید
    "THREEINSIDE": {"name": "Three Inside Up/Down", "talib_func": talib.CDL3INSIDE, "category": "Phase 2"},
    "THREEOUTSIDE": {"name": "Three Outside Up/Down", "talib_func": talib.CDL3OUTSIDE, "category": "Phase 2"},
    "BELTHOLD": {"name": "Belt Hold", "talib_func": talib.CDLBELTHOLD, "category": "Phase 2"},
    "THREEMETHODS": {"name": "Rising/Falling Three Methods", "talib_func": talib.CDLRISEFALL3METHODS, "category": "Phase 2"},
    "MATHOLD": {"name": "Mat Hold", "talib_func": talib.CDLMATHOLD, "category": "Phase 2"},
}

# =============================================================================
# Functions (copied from test_pattern_with_real_data.py)
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

    return df

def find_first_detection(df, pattern_func):
    """پیدا کردن اولین detection در داده واقعی"""

    result = pattern_func(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    detections = np.where(result != 0)[0]

    if len(detections) == 0:
        return None

    return detections[0]

def show_candle_details(df, idx):
    """نمایش جزئیات یک کندل"""

    candle = df.iloc[idx]

    print(f"\n📊 کندل در index {idx}:")
    print("="*60)
    print(f"  Open:   {candle['open']:.2f}")
    print(f"  High:   {candle['high']:.2f}")
    print(f"  Low:    {candle['low']:.2f}")
    print(f"  Close:  {candle['close']:.2f}")

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
    """تست حداقل تعداد کندل مورد نیاز"""

    print("\n" + "="*60)
    print(f"🔬 تست Minimum Lookback: {pattern_name}")
    print("="*60)

    print(f"\n📌 Detection index: {detection_idx}")
    show_candle_details(df, detection_idx)

    print(f"\n🧪 تست با تعداد کندل‌های مختلف:")
    print("-"*60)

    lookback_values = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 20, 30, 50]
    minimum_found = None

    for lookback in lookback_values:
        if lookback > detection_idx:
            continue

        start_idx = detection_idx - lookback
        df_test = df.iloc[start_idx:detection_idx + 1].copy()

        try:
            result = pattern_func(
                df_test['open'].values,
                df_test['high'].values,
                df_test['low'].values,
                df_test['close'].values
            )

            detected = result[-1] != 0

            icon = "✅" if detected else "❌"
            print(f"{icon} Lookback: {lookback:2d} | کل کندل: {len(df_test):2d} | "
                  f"تشخیص: {'YES' if detected else 'NO'}")

            if detected and minimum_found is None:
                minimum_found = lookback

        except Exception as e:
            print(f"❌ Lookback: {lookback:2d} | خطا: {str(e)}")

    print("\n" + "="*60)
    print("📊 نتیجه:")
    print("="*60)

    if minimum_found is not None:
        print(f"\n✅ حداقل lookback: {minimum_found}")
        print(f"✅ حداقل کل کندل: {minimum_found + 1}")
    else:
        print("\n❌ حداقل lookback پیدا نشد!")

    return minimum_found

# =============================================================================
# MAIN
# =============================================================================

def show_menu():
    """نمایش منوی انتخاب الگو"""

    print("=" * 80)
    print("🎯 انتخاب الگوی مورد نظر برای تست")
    print("=" * 80)
    print()

    # گروه‌بندی بر اساس دسته
    categories = {}
    for key, info in PATTERN_INFO.items():
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((key, info))

    # نمایش هر دسته
    index = 1
    pattern_list = []

    for category in ['قدیمی', 'Phase 1', 'Phase 2']:
        if category not in categories:
            continue

        print(f"\n📁 {category}:")
        print("-" * 80)

        for key, info in categories[category]:
            print(f"  {index:2d}. {info['name']}")
            pattern_list.append(key)
            index += 1

    print()
    print("-" * 80)
    print(f"  0. خروج")
    print()

    return pattern_list

def main():
    """اجرای تست تعاملی"""

    print("\n" + "=" * 80)
    print("🔬 تست تعاملی الگوهای کندل استیک")
    print("=" * 80)
    print()

    # بارگذاری داده
    print("📊 بارگذاری داده BTC...")
    df = load_btc_data()

    if df is None:
        return

    print(f"✅ {len(df)} کندل بارگذاری شد")

    while True:
        # نمایش منو
        pattern_list = show_menu()

        # دریافت انتخاب کاربر
        try:
            choice = input("لطفا شماره الگو را وارد کنید (یا 0 برای خروج): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                print("\n👋 خروج از برنامه...")
                break

            if choice_num < 1 or choice_num > len(pattern_list):
                print(f"\n❌ شماره نامعتبر! لطفا عددی بین 1 تا {len(pattern_list)} وارد کنید.")
                input("\nEnter را فشار دهید تا ادامه دهید...")
                continue

            # الگوی انتخاب شده
            pattern_key = pattern_list[choice_num - 1]
            pattern_info = PATTERN_INFO[pattern_key]
            pattern_name = pattern_info['name']
            pattern_func = pattern_info['talib_func']

            print("\n" + "=" * 80)
            print(f"🔍 تست الگو: {pattern_name}")
            print("=" * 80)

            # پیدا کردن اولین detection
            print("\n🔍 جستجو برای detections...")
            first_idx = find_first_detection(df, pattern_func)

            if first_idx is None:
                print("❌ هیچ detection در داده واقعی پیدا نشد!")
                input("\nEnter را فشار دهید تا ادامه دهید...")
                continue

            print(f"✅ اولین detection در index {first_idx} پیدا شد")

            # تست minimum lookback
            minimum = test_minimum_lookback(df, first_idx, pattern_func, pattern_name)

            print("\n" + "=" * 80)
            print("✅ تست کامل شد")
            print("=" * 80)

            input("\nEnter را فشار دهید تا به منو برگردید...")

        except ValueError:
            print("\n❌ ورودی نامعتبر! لطفا یک عدد وارد کنید.")
            input("\nEnter را فشار دهید تا ادامه دهید...")
        except KeyboardInterrupt:
            print("\n\n👋 خروج از برنامه...")
            break
        except Exception as e:
            print(f"\n❌ خطا: {e}")
            input("\nEnter را فشار دهید تا ادامه دهید...")

if __name__ == '__main__':
    main()
