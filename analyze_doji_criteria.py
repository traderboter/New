"""
تحلیل معیارهای Doji در داده‌های واقعی
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path

def analyze_doji_distribution(csv_file: str):
    """تحلیل توزیع body_ratio در داده‌های واقعی"""

    df = pd.read_csv(csv_file)

    # محاسبه body_ratio برای همه کندل‌ها
    df['body_size'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['body_ratio'] = df['body_size'] / df['full_range']

    # حذف کندل‌های با range صفر
    df = df[df['full_range'] > 0]

    print(f"\n{'='*80}")
    print(f"📊 تحلیل فایل: {Path(csv_file).name}")
    print(f"{'='*80}")
    print(f"تعداد کل کندل‌ها: {len(df):,}")

    # تحلیل با آستانه‌های مختلف
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]

    print(f"\n{'آستانه body_ratio':<20} {'تعداد':<15} {'درصد':<15}")
    print(f"{'-'*50}")

    for threshold in thresholds:
        count = len(df[df['body_ratio'] <= threshold])
        percentage = (count / len(df)) * 100
        print(f"<= {threshold:<17.2f} {count:<15,} {percentage:>6.2f}%")

    # بررسی تشخیص TA-Lib
    talib_doji = talib.CDLDOJI(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    talib_count = np.sum(talib_doji != 0)
    talib_percentage = (talib_count / len(df)) * 100

    print(f"\n{'TA-Lib CDLDOJI':<20} {talib_count:<15,} {talib_percentage:>6.3f}%")

    # نمایش آمار body_ratio
    print(f"\n{'='*80}")
    print(f"📈 آمار body_ratio:")
    print(f"{'='*80}")
    print(f"میانگین: {df['body_ratio'].mean():.4f}")
    print(f"میانه: {df['body_ratio'].median():.4f}")
    print(f"انحراف معیار: {df['body_ratio'].std():.4f}")
    print(f"کمترین: {df['body_ratio'].min():.4f}")
    print(f"بیشترین: {df['body_ratio'].max():.4f}")

    # نمایش نمونه‌هایی که TA-Lib تشخیص داده
    if talib_count > 0:
        print(f"\n{'='*80}")
        print(f"🔍 نمونه‌های تشخیص داده شده توسط TA-Lib:")
        print(f"{'='*80}")
        detected = df[talib_doji != 0].head(5)
        for idx, row in detected.iterrows():
            print(f"\nکندل {idx}:")
            print(f"  Open: {row['open']:.2f}, Close: {row['close']:.2f}")
            print(f"  High: {row['high']:.2f}, Low: {row['low']:.2f}")
            print(f"  Body Size: {row['body_size']:.2f}")
            print(f"  Full Range: {row['full_range']:.2f}")
            print(f"  Body Ratio: {row['body_ratio']:.4f}")


if __name__ == "__main__":
    data_dir = Path("historical/BTC-USDT")

    # تحلیل هر تایم‌فریم
    for timeframe in ["5min.csv", "15min.csv", "1hour.csv", "4hour.csv"]:
        file_path = data_dir / timeframe
        if file_path.exists():
            analyze_doji_distribution(str(file_path))
        else:
            print(f"\n⚠️  فایل یافت نشد: {file_path}")

    print(f"\n{'='*80}")
    print("✅ تحلیل کامل شد!")
    print(f"{'='*80}")
