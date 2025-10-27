"""
بررسی سریع: آیا الگوی Mat Hold در داده واقعی BTC وجود دارد؟
"""

import talib
import pandas as pd
import numpy as np
from pathlib import Path

def check_timeframe(timeframe, filename):
    """بررسی یک تایم فریم"""

    csv_path = Path(__file__).parent.parent / 'historical' / 'BTC-USDT' / filename

    if not csv_path.exists():
        print(f"❌ {timeframe}: فایل پیدا نشد")
        return

    df = pd.read_csv(csv_path)
    df = df.astype({
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    })

    # اجرای TALib
    result = talib.CDLMATHOLD(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    # پیدا کردن detections
    detections = np.where(result != 0)[0]

    print(f"\n📊 {timeframe} ({len(df)} کندل):")
    if len(detections) > 0:
        print(f"   ✅ {len(detections)} الگو پیدا شد!")
        print(f"   📍 اولین detection: index {detections[0]}")
        print(f"   📍 آخرین detection: index {detections[-1]}")
    else:
        print(f"   ❌ هیچ الگویی پیدا نشد")

def main():
    print("="*60)
    print("🔍 بررسی الگوی Mat Hold در تمام داده‌های BTC")
    print("="*60)

    timeframes = [
        ("5m", "5min.csv"),
        ("15m", "15min.csv"),
        ("1h", "1hour.csv"),
        ("4h", "4hour.csv"),
    ]

    for tf, filename in timeframes:
        check_timeframe(tf, filename)

    print("\n" + "="*60)
    print("💡 نتیجه‌گیری:")
    print("="*60)
    print("""
الگوی Mat Hold یک الگوی بسیار نادر است که شامل این شرایط است:
1. یک کندل صعودی قوی (شروع صعود)
2. یک gap صعودی کوچک
3. 2-3 کندل کوچک که عقب‌نشینی را نشان می‌دهند
4. یک کندل صعودی قوی که بالاتر از همه بسته می‌شود

اگر هیچ detection پیدا نشد، ممکن است:
- این الگو واقعاً در BTC رخ نداده باشد
- یا معیارهای TALib بسیار سخت‌گیرانه است
    """)

if __name__ == '__main__':
    main()
