#!/usr/bin/env python3
"""
Test All Patterns: تست خودکار همه 26 الگوی کندل استیک

این اسکریپت همه الگوهای موجود را به صورت خودکار تست می‌کند.
برای هر الگو:
1. تعداد detections را می‌شمارد
2. اولین detection را پیدا می‌کند
3. حداقل lookback مورد نیاز را محاسبه می‌کند

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
    # ✅ الگوهای اولیه
    "ENGULFING": {
        "name": "Engulfing",
        "talib_func": talib.CDLENGULFING,
        "category": "قدیمی"
    },
    "HAMMER": {
        "name": "Hammer",
        "talib_func": talib.CDLHAMMER,
        "category": "قدیمی"
    },
    "SHOOTINGSTAR": {
        "name": "Shooting Star",
        "talib_func": talib.CDLSHOOTINGSTAR,
        "category": "قدیمی"
    },
    "DOJI": {
        "name": "Doji",
        "talib_func": talib.CDLDOJI,
        "category": "قدیمی"
    },
    "MORNINGSTAR": {
        "name": "Morning Star",
        "talib_func": talib.CDLMORNINGSTAR,
        "category": "قدیمی"
    },
    "INVERTEDHAMMER": {
        "name": "Inverted Hammer",
        "talib_func": talib.CDLINVERTEDHAMMER,
        "category": "قدیمی"
    },
    "DARKCLOUDCOVER": {
        "name": "Dark Cloud Cover",
        "talib_func": talib.CDLDARKCLOUDCOVER,
        "category": "قدیمی"
    },
    "EVENINGSTAR": {
        "name": "Evening Star",
        "talib_func": talib.CDLEVENINGSTAR,
        "category": "قدیمی"
    },
    "EVENINGDOJISTAR": {
        "name": "Evening Doji Star",
        "talib_func": talib.CDLEVENINGDOJISTAR,
        "category": "قدیمی"
    },
    "HARAMI": {
        "name": "Harami",
        "talib_func": talib.CDLHARAMI,
        "category": "قدیمی"
    },
    "HARAMICROSS": {
        "name": "Harami Cross",
        "talib_func": talib.CDLHARAMICROSS,
        "category": "قدیمی"
    },
    "HANGINGMAN": {
        "name": "Hanging Man",
        "talib_func": talib.CDLHANGINGMAN,
        "category": "قدیمی"
    },
    "PIERCINGLINE": {
        "name": "Piercing Line",
        "talib_func": talib.CDLPIERCING,
        "category": "قدیمی"
    },
    "MORNINGDOJISTAR": {
        "name": "Morning Doji Star",
        "talib_func": talib.CDLMORNINGDOJISTAR,
        "category": "قدیمی"
    },
    "THREEWHITESOLDIERS": {
        "name": "Three White Soldiers",
        "talib_func": talib.CDL3WHITESOLDIERS,
        "category": "قدیمی"
    },
    "THREEBLACKCROWS": {
        "name": "Three Black Crows",
        "talib_func": talib.CDL3BLACKCROWS,
        "category": "قدیمی"
    },

    # 🆕 Phase 1 - الگوهای قدرتمند جدید
    "MARUBOZU": {
        "name": "Marubozu",
        "talib_func": talib.CDLMARUBOZU,
        "category": "Phase 1"
    },
    "DRAGONFLYDOJI": {
        "name": "Dragonfly Doji",
        "talib_func": talib.CDLDRAGONFLYDOJI,
        "category": "Phase 1"
    },
    "GRAVESTONEDOJI": {
        "name": "Gravestone Doji",
        "talib_func": talib.CDLGRAVESTONEDOJI,
        "category": "Phase 1"
    },
    "SPINNINGTOP": {
        "name": "Spinning Top",
        "talib_func": talib.CDLSPINNINGTOP,
        "category": "Phase 1"
    },
    "LONGLEGGEDDOJI": {
        "name": "Long-Legged Doji",
        "talib_func": talib.CDLLONGLEGGEDDOJI,
        "category": "Phase 1"
    },

    # 🆕 Phase 2 - الگوهای ادامه و تایید جدید
    "THREEINSIDE": {
        "name": "Three Inside Up/Down",
        "talib_func": talib.CDL3INSIDE,
        "category": "Phase 2"
    },
    "THREEOUTSIDE": {
        "name": "Three Outside Up/Down",
        "talib_func": talib.CDL3OUTSIDE,
        "category": "Phase 2"
    },
    "BELTHOLD": {
        "name": "Belt Hold",
        "talib_func": talib.CDLBELTHOLD,
        "category": "Phase 2"
    },
    "THREEMETHODS": {
        "name": "Rising/Falling Three Methods",
        "talib_func": talib.CDLRISEFALL3METHODS,
        "category": "Phase 2"
    },
    "MATHOLD": {
        "name": "Mat Hold",
        "talib_func": talib.CDLMATHOLD,
        "category": "Phase 2"
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

    return df

# =============================================================================
# تست
# =============================================================================

def test_pattern(df, pattern_key, pattern_info):
    """
    تست یک الگو

    Returns:
        dict: نتایج تست شامل تعداد detections، اولین index و حداقل lookback
    """

    pattern_func = pattern_info['talib_func']

    try:
        # اجرای تابع TALib
        result = pattern_func(
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values
        )

        # پیدا کردن detections
        detections = np.where(result != 0)[0]
        num_detections = len(detections)

        if num_detections == 0:
            return {
                'status': 'no_detection',
                'num_detections': 0,
                'first_idx': None,
                'min_lookback': None
            }

        # پیدا کردن حداقل lookback برای اولین detection
        first_idx = detections[0]
        min_lookback = find_minimum_lookback(df, first_idx, pattern_func)

        return {
            'status': 'ok',
            'num_detections': num_detections,
            'first_idx': first_idx,
            'min_lookback': min_lookback
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'num_detections': 0,
            'first_idx': None,
            'min_lookback': None
        }

def find_minimum_lookback(df, detection_idx, pattern_func):
    """پیدا کردن حداقل lookback برای یک detection"""

    lookback_values = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 20, 30]

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

            if result[-1] != 0:
                return lookback

        except:
            pass

    return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    """تست همه الگوها"""

    print("=" * 80)
    print("🔬 تست خودکار همه الگوهای کندل استیک")
    print("=" * 80)
    print()

    # بارگذاری داده
    print("📊 بارگذاری داده BTC...")
    df = load_btc_data()

    if df is None:
        return

    print(f"✅ {len(df)} کندل بارگذاری شد")
    print()

    # تست همه الگوها
    print("🔍 شروع تست الگوها...")
    print("-" * 80)

    results = {}

    for pattern_key, pattern_info in PATTERN_INFO.items():
        pattern_name = pattern_info['name']
        category = pattern_info['category']

        print(f"  Testing: {pattern_name:30s} ({category:10s})...", end=" ")
        sys.stdout.flush()

        result = test_pattern(df, pattern_key, pattern_info)
        results[pattern_key] = result

        if result['status'] == 'ok':
            print(f"✅ {result['num_detections']:4d} detections, min_lookback={result['min_lookback']}")
        elif result['status'] == 'no_detection':
            print(f"⚠️  0 detections")
        else:
            print(f"❌ ERROR: {result.get('error', 'Unknown')}")

    print()

    # خلاصه نتایج
    print("=" * 80)
    print("📊 خلاصه نتایج")
    print("=" * 80)
    print()

    # گروه‌بندی بر اساس دسته
    categories = {}
    for pattern_key, pattern_info in PATTERN_INFO.items():
        category = pattern_info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((pattern_key, pattern_info, results[pattern_key]))

    # نمایش هر دسته
    for category in ['قدیمی', 'Phase 1', 'Phase 2']:
        if category not in categories:
            continue

        patterns_in_category = categories[category]

        print(f"📁 {category}:")
        print("-" * 80)

        total_patterns = len(patterns_in_category)
        successful_patterns = sum(1 for _, _, r in patterns_in_category if r['status'] == 'ok')
        no_detection_patterns = sum(1 for _, _, r in patterns_in_category if r['status'] == 'no_detection')
        error_patterns = sum(1 for _, _, r in patterns_in_category if r['status'] == 'error')

        print(f"  کل الگوها: {total_patterns}")
        print(f"  موفق (با detection): {successful_patterns}")
        print(f"  بدون detection: {no_detection_patterns}")
        print(f"  خطا: {error_patterns}")

        if successful_patterns > 0:
            lookbacks = [r['min_lookback'] for _, _, r in patterns_in_category if r['status'] == 'ok' and r['min_lookback'] is not None]
            if lookbacks:
                print(f"  میانگین min_lookback: {np.mean(lookbacks):.1f}")
                print(f"  حداقل min_lookback: {np.min(lookbacks)}")
                print(f"  حداکثر min_lookback: {np.max(lookbacks)}")

        print()

    # جدول کامل
    print("=" * 80)
    print("📋 جدول کامل نتایج")
    print("=" * 80)
    print()
    print(f"{'الگو':<35} {'دسته':<12} {'Detections':<12} {'Min Lookback':<12} {'وضعیت':<10}")
    print("-" * 80)

    for pattern_key, pattern_info in PATTERN_INFO.items():
        result = results[pattern_key]
        name = pattern_info['name']
        category = pattern_info['category']

        num_det = result['num_detections'] if result['num_detections'] else 0
        lookback = result['min_lookback'] if result['min_lookback'] is not None else '-'
        status = '✅' if result['status'] == 'ok' else '⚠️' if result['status'] == 'no_detection' else '❌'

        print(f"{name:<35} {category:<12} {num_det:<12} {str(lookback):<12} {status:<10}")

    print()
    print("=" * 80)
    print("✅ تست کامل شد!")
    print("=" * 80)
    print()

    # آمار کلی
    total_patterns = len(PATTERN_INFO)
    successful = sum(1 for r in results.values() if r['status'] == 'ok')
    no_detection = sum(1 for r in results.values() if r['status'] == 'no_detection')
    errors = sum(1 for r in results.values() if r['status'] == 'error')

    print(f"📊 آمار کلی:")
    print(f"  کل الگوها: {total_patterns}")
    print(f"  موفق: {successful} ({successful/total_patterns*100:.1f}%)")
    print(f"  بدون detection: {no_detection} ({no_detection/total_patterns*100:.1f}%)")
    print(f"  خطا: {errors} ({errors/total_patterns*100:.1f}%)")
    print()

if __name__ == '__main__':
    main()
