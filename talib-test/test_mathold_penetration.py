"""
تست Mat Hold با مقادیر مختلف penetration

این اسکریپت مقادیر مختلف penetration را تست می‌کند تا ببیند
کدام یک detections بیشتری دارد.
"""

import talib
import pandas as pd
import numpy as np
from pathlib import Path

def load_btc_data(timeframe="1h"):
    """Load BTC data"""
    timeframe_files = {
        "5m": "5min.csv",
        "15m": "15min.csv",
        "1h": "1hour.csv",
        "4h": "4hour.csv"
    }

    if timeframe not in timeframe_files:
        return None

    csv_path = Path(__file__).parent.parent / 'historical' / 'BTC-USDT' / timeframe_files[timeframe]

    if not csv_path.exists():
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

def test_penetration(df, penetration_value):
    """Test a specific penetration value"""
    try:
        result = talib.CDLMATHOLD(
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values,
            penetration=penetration_value
        )

        detections = np.where(result != 0)[0]
        return len(detections), detections
    except Exception as e:
        return 0, []

def main():
    print("="*70)
    print("🔬 تست Mat Hold با مقادیر مختلف Penetration")
    print("="*70)

    # تست برای همه تایم فریم‌ها
    timeframes = ["5m", "15m", "1h", "4h"]

    # مقادیر مختلف penetration برای تست
    penetration_values = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]

    for tf in timeframes:
        print(f"\n{'='*70}")
        print(f"📊 تایم فریم: {tf}")
        print(f"{'='*70}")

        df = load_btc_data(tf)
        if df is None:
            print(f"❌ فایل پیدا نشد")
            continue

        print(f"✅ {len(df)} کندل بارگذاری شد")
        print(f"\n{'Penetration':>12} | {'Detections':>10} | {'نتیجه'}")
        print("-"*70)

        best_penetration = None
        best_count = 0

        for pen in penetration_values:
            count, detections = test_penetration(df, pen)

            # آیکون
            if count == 0:
                icon = "❌"
            elif count < 5:
                icon = "⚠️ "
            elif count < 20:
                icon = "✅"
            else:
                icon = "🌟"

            print(f"{pen:>12.2f} | {count:>10} | {icon}")

            if count > best_count:
                best_count = count
                best_penetration = pen

        if best_count > 0:
            print(f"\n💡 بهترین مقدار: penetration={best_penetration:.2f} با {best_count} detection")
        else:
            print(f"\n❌ هیچ detection با هیچ مقداری پیدا نشد!")

    print("\n" + "="*70)
    print("📋 نتیجه‌گیری:")
    print("="*70)
    print("""
1. مقدار پیش‌فرض TALib (0.5) خیلی سخت‌گیرانه است
2. مقادیر پایین‌تر (0.2-0.3) معمولاً detections بیشتری دارند
3. Mat Hold یک الگوی نادر است حتی با penetration کم

💡 توصیه: از penetration=0.3 استفاده کنید (تعادل بین دقت و تعداد)
""")

if __name__ == '__main__':
    main()
