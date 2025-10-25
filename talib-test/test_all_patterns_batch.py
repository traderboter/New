"""
Test All Patterns Batch: تست خودکار همه الگوها

این اسکریپت:
1. همه الگوهای موجود را تست می‌کند
2. حداقل candles requirement را پیدا می‌کند
3. نتایج را در یک فایل JSON ذخیره می‌کند
4. یک گزارش خلاصه تولید می‌کند

نویسنده: Test Team
تاریخ: 2025-10-25
"""

import talib
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# تنظیمات
# =============================================================================

PATTERN_INFO = {
    # ✅ الگوهای تست شده قبلی
    "ENGULFING": {
        "name": "Engulfing",
        "talib_func": talib.CDLENGULFING,
        "category": "reversal",
    },
    "HAMMER": {
        "name": "Hammer",
        "talib_func": talib.CDLHAMMER,
        "category": "reversal",
    },
    "SHOOTINGSTAR": {
        "name": "Shooting Star",
        "talib_func": talib.CDLSHOOTINGSTAR,
        "category": "reversal",
    },
    "DOJI": {
        "name": "Doji",
        "talib_func": talib.CDLDOJI,
        "category": "reversal",
    },
    "MORNINGSTAR": {
        "name": "Morning Star",
        "talib_func": talib.CDLMORNINGSTAR,
        "category": "reversal",
    },
    "INVERTEDHAMMER": {
        "name": "Inverted Hammer",
        "talib_func": talib.CDLINVERTEDHAMMER,
        "category": "reversal",
    },

    # 🆕 الگوهای جدید برای تست
    "DARKCLOUDCOVER": {
        "name": "Dark Cloud Cover",
        "talib_func": talib.CDLDARKCLOUDCOVER,
        "category": "reversal",
    },
    "EVENINGSTAR": {
        "name": "Evening Star",
        "talib_func": talib.CDLEVENINGSTAR,
        "category": "reversal",
    },
    "EVENINGDOJISTAR": {
        "name": "Evening Doji Star",
        "talib_func": talib.CDLEVENINGDOJISTAR,
        "category": "reversal",
    },
    "HARAMI": {
        "name": "Harami",
        "talib_func": talib.CDLHARAMI,
        "category": "reversal",
    },
    "HARAMICROSS": {
        "name": "Harami Cross",
        "talib_func": talib.CDLHARAMICROSS,
        "category": "reversal",
    },
    "HANGINGMAN": {
        "name": "Hanging Man",
        "talib_func": talib.CDLHANGINGMAN,
        "category": "reversal",
    },
    "PIERCINGLINE": {
        "name": "Piercing Line",
        "talib_func": talib.CDLPIERCING,
        "category": "reversal",
    },
    "MORNINGDOJISTAR": {
        "name": "Morning Doji Star",
        "talib_func": talib.CDLMORNINGDOJISTAR,
        "category": "reversal",
    },
    "THREEWHITESOLDIERS": {
        "name": "Three White Soldiers",
        "talib_func": talib.CDL3WHITESOLDIERS,
        "category": "continuation",
    },
    "THREEBLACKCROWS": {
        "name": "Three Black Crows",
        "talib_func": talib.CDL3BLACKCROWS,
        "category": "continuation",
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
# تست الگو
# =============================================================================

def test_pattern(df, pattern_key, pattern_info, num_samples=10):
    """
    تست یک الگو و پیدا کردن minimum candles requirement

    Returns:
        dict: نتایج تست شامل min_lookback, min_candles, detection_rate
    """

    print(f"\n{'='*70}")
    print(f"🔬 تست الگو: {pattern_info['name']}")
    print(f"{'='*70}")

    pattern_func = pattern_info['talib_func']

    # اجرای detection روی کل داده
    try:
        result = pattern_func(
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
    except Exception as e:
        print(f"❌ خطا در اجرای TA-Lib: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

    # پیدا کردن همه detections
    detection_indices = np.where(result != 0)[0]

    if len(detection_indices) == 0:
        print(f"❌ هیچ detection در داده BTC پیدا نشد!")
        return {
            'success': False,
            'error': 'No detections found in BTC data'
        }

    detection_count = len(detection_indices)
    detection_rate = (detection_count / len(df)) * 100

    print(f"✅ تعداد detections: {detection_count}/{len(df)} = {detection_rate:.2f}%")

    # انتخاب نمونه‌های تصادفی
    sample_size = min(num_samples, len(detection_indices))
    # فقط detection هایی را انتخاب کن که حداقل 50 کندل قبلی دارند
    valid_detections = [idx for idx in detection_indices if idx >= 50]

    if len(valid_detections) == 0:
        print(f"❌ هیچ detection با کندل کافی پیدا نشد!")
        return {
            'success': False,
            'error': 'No detections with enough history'
        }

    sample_indices = np.random.choice(valid_detections, min(sample_size, len(valid_detections)), replace=False)

    print(f"🎲 تست {len(sample_indices)} detection تصادفی...")

    # تست هر نمونه
    minimum_lookbacks = []

    for det_idx in sample_indices:
        # تست با lookback های مختلف
        lookback_values = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 20, 30]
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
                    break

            except:
                pass

        if minimum is not None:
            minimum_lookbacks.append(minimum)

    # تحلیل نتایج
    if len(minimum_lookbacks) == 0:
        print(f"❌ نتوانستیم minimum lookback را پیدا کنیم!")
        return {
            'success': False,
            'error': 'Could not find minimum lookback'
        }

    min_lookback = int(np.max(minimum_lookbacks))  # بیشترین (محافظه‌کارانه‌ترین)
    avg_lookback = float(np.mean(minimum_lookbacks))
    median_lookback = float(np.median(minimum_lookbacks))

    min_candles = min_lookback + 1

    print(f"\n📊 نتایج:")
    print(f"   ✅ حداقل lookback: {min_lookback}")
    print(f"   ✅ حداقل کل کندل: {min_candles}")
    print(f"   📈 میانگین lookback: {avg_lookback:.1f}")
    print(f"   📊 میانه lookback: {median_lookback:.1f}")
    print(f"   📉 Detection rate: {detection_rate:.2f}%")

    return {
        'success': True,
        'min_lookback': min_lookback,
        'min_candles': min_candles,
        'avg_lookback': avg_lookback,
        'median_lookback': median_lookback,
        'detection_count': detection_count,
        'detection_rate': detection_rate,
        'total_candles': len(df),
        'samples_tested': len(minimum_lookbacks)
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    """اجرای تست برای همه الگوها"""

    print("="*70)
    print("🔬 تست خودکار همه الگوها")
    print("="*70)
    print(f"\n📅 تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # بارگذاری داده
    print(f"\n📂 بارگذاری داده BTC...")
    df = load_btc_data()
    if df is None:
        return

    print(f"✅ Loaded {len(df)} BTC candles")

    # تست همه الگوها
    results = {}
    total_patterns = len(PATTERN_INFO)

    for i, (pattern_key, pattern_info) in enumerate(PATTERN_INFO.items(), 1):
        print(f"\n\n{'#'*70}")
        print(f"# [{i}/{total_patterns}] تست الگو: {pattern_info['name']}")
        print(f"{'#'*70}")

        result = test_pattern(df, pattern_key, pattern_info, num_samples=10)

        results[pattern_key] = {
            'name': pattern_info['name'],
            'category': pattern_info.get('category', 'unknown'),
            **result
        }

    # ذخیره نتایج
    output_file = Path(__file__).parent / 'pattern_test_results.json'

    output_data = {
        'test_date': datetime.now().isoformat(),
        'btc_candles': len(df),
        'patterns_tested': len(results),
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}")
    print(f"✅ نتایج در {output_file} ذخیره شد")
    print(f"{'='*70}")

    # گزارش خلاصه
    print(f"\n\n{'='*70}")
    print("📊 خلاصه نتایج")
    print(f"{'='*70}")

    print(f"\n{'Pattern':<25} {'Min Candles':<15} {'Detection Rate':<15} {'Status'}")
    print("-"*70)

    for pattern_key, result in results.items():
        name = result['name']

        if result['success']:
            min_candles = result['min_candles']
            det_rate = result['detection_rate']
            status = "✅"
        else:
            min_candles = "N/A"
            det_rate = "N/A"
            status = "❌"

        print(f"{name:<25} {str(min_candles):<15} {str(det_rate) if det_rate != 'N/A' else det_rate:<15} {status}")

    # آمار کلی
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful

    print(f"\n{'='*70}")
    print(f"📈 آمار کلی:")
    print(f"   ✅ موفق: {successful}/{len(results)}")
    print(f"   ❌ ناموفق: {failed}/{len(results)}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
