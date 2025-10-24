"""
تحلیل معیارهای Shooting Star در داده واقعی

این اسکریپت همه کندل‌ها را اسکن می‌کند و توزیع معیارهای Shooting Star را نمایش می‌دهد:
- upper_shadow_ratio (upper shadow / body)
- lower_shadow_ratio (lower shadow / body)
- body_position (موقعیت body در کندل)

هدف: تعیین threshold های واقع‌بینانه برای detection
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_shooting_star_metrics(csv_path: str):
    """
    تحلیل معیارهای Shooting Star در یک فایل CSV.

    Args:
        csv_path: مسیر فایل CSV
    """
    print(f"\n{'='*80}")
    print(f"📊 Analyzing Shooting Star Metrics: {Path(csv_path).name}")
    print(f"{'='*80}\n")

    # خواندن داده
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Total candles: {len(df)}")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")

    # لیست برای ذخیره metrics
    metrics = []

    for i, row in df.iterrows():
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        # محاسبات
        body_size = abs(close - open_price)
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        full_range = high - low

        if full_range == 0:
            continue

        # برای جلوگیری از division by zero
        body_for_ratio = max(body_size, full_range * 0.01)

        # نسبت‌ها
        upper_shadow_ratio = upper_shadow / body_for_ratio
        lower_shadow_ratio = lower_shadow / body_for_ratio

        # body position (از پایین)
        body_bottom = min(open_price, close)
        body_position = (body_bottom - low) / full_range

        # ذخیره
        metrics.append({
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'body_position': body_position,
            'body_size': body_size,
            'full_range': full_range,
            'timestamp': row['timestamp']
        })

    # تبدیل به DataFrame
    metrics_df = pd.DataFrame(metrics)

    print(f"Valid candles for analysis: {len(metrics_df)}\n")

    # توزیع upper_shadow_ratio
    print(f"{'='*80}")
    print(f"📊 Upper Shadow Ratio Distribution (upper_shadow / body)")
    print(f"{'='*80}")
    print(f"  Mean:      {metrics_df['upper_shadow_ratio'].mean():.3f}")
    print(f"  Median:    {metrics_df['upper_shadow_ratio'].median():.3f}")
    print(f"  Std Dev:   {metrics_df['upper_shadow_ratio'].std():.3f}")
    print(f"  Min:       {metrics_df['upper_shadow_ratio'].min():.3f}")
    print(f"  Max:       {metrics_df['upper_shadow_ratio'].max():.3f}")
    print(f"\n  Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(metrics_df['upper_shadow_ratio'], p)
        print(f"    {p}th:     {val:.3f}")

    # چند درصد کندل‌ها upper_shadow >= 2.0x body دارند؟
    print(f"\n  Candles with upper_shadow >= 2.0x body: {len(metrics_df[metrics_df['upper_shadow_ratio'] >= 2.0])} ({len(metrics_df[metrics_df['upper_shadow_ratio'] >= 2.0])/len(metrics_df)*100:.2f}%)")
    print(f"  Candles with upper_shadow >= 1.5x body: {len(metrics_df[metrics_df['upper_shadow_ratio'] >= 1.5])} ({len(metrics_df[metrics_df['upper_shadow_ratio'] >= 1.5])/len(metrics_df)*100:.2f}%)")
    print(f"  Candles with upper_shadow >= 1.0x body: {len(metrics_df[metrics_df['upper_shadow_ratio'] >= 1.0])} ({len(metrics_df[metrics_df['upper_shadow_ratio'] >= 1.0])/len(metrics_df)*100:.2f}%)")

    # توزیع lower_shadow_ratio
    print(f"\n{'='*80}")
    print(f"📊 Lower Shadow Ratio Distribution (lower_shadow / body)")
    print(f"{'='*80}")
    print(f"  Mean:      {metrics_df['lower_shadow_ratio'].mean():.3f}")
    print(f"  Median:    {metrics_df['lower_shadow_ratio'].median():.3f}")
    print(f"  Std Dev:   {metrics_df['lower_shadow_ratio'].std():.3f}")
    print(f"  Min:       {metrics_df['lower_shadow_ratio'].min():.3f}")
    print(f"  Max:       {metrics_df['lower_shadow_ratio'].max():.3f}")
    print(f"\n  Percentiles:")
    for p in [5, 10, 25, 50]:
        val = np.percentile(metrics_df['lower_shadow_ratio'], p)
        print(f"    {p}th:     {val:.3f}")

    # چند درصد کندل‌ها lower_shadow <= 0.1x body دارند؟
    print(f"\n  Candles with lower_shadow <= 0.1x body: {len(metrics_df[metrics_df['lower_shadow_ratio'] <= 0.1])} ({len(metrics_df[metrics_df['lower_shadow_ratio'] <= 0.1])/len(metrics_df)*100:.2f}%)")
    print(f"  Candles with lower_shadow <= 0.3x body: {len(metrics_df[metrics_df['lower_shadow_ratio'] <= 0.3])} ({len(metrics_df[metrics_df['lower_shadow_ratio'] <= 0.3])/len(metrics_df)*100:.2f}%)")
    print(f"  Candles with lower_shadow <= 0.5x body: {len(metrics_df[metrics_df['lower_shadow_ratio'] <= 0.5])} ({len(metrics_df[metrics_df['lower_shadow_ratio'] <= 0.5])/len(metrics_df)*100:.2f}%)")

    # توزیع body_position
    print(f"\n{'='*80}")
    print(f"📊 Body Position Distribution (0=bottom, 1=top)")
    print(f"{'='*80}")
    print(f"  Mean:      {metrics_df['body_position'].mean():.3f}")
    print(f"  Median:    {metrics_df['body_position'].median():.3f}")
    print(f"  Std Dev:   {metrics_df['body_position'].std():.3f}")
    print(f"  Min:       {metrics_df['body_position'].min():.3f}")
    print(f"  Max:       {metrics_df['body_position'].max():.3f}")
    print(f"\n  Percentiles:")
    for p in [5, 10, 25, 33, 40, 50]:
        val = np.percentile(metrics_df['body_position'], p)
        print(f"    {p}th:     {val:.3f}")

    # چند درصد کندل‌ها body_position <= 0.33 دارند؟
    print(f"\n  Candles with body_position <= 0.33 (bottom 1/3): {len(metrics_df[metrics_df['body_position'] <= 0.33])} ({len(metrics_df[metrics_df['body_position'] <= 0.33])/len(metrics_df)*100:.2f}%)")
    print(f"  Candles with body_position <= 0.40: {len(metrics_df[metrics_df['body_position'] <= 0.40])} ({len(metrics_df[metrics_df['body_position'] <= 0.40])/len(metrics_df)*100:.2f}%)")
    print(f"  Candles with body_position <= 0.50: {len(metrics_df[metrics_df['body_position'] <= 0.50])} ({len(metrics_df[metrics_df['body_position'] <= 0.50])/len(metrics_df)*100:.2f}%)")

    # ترکیب شرایط
    print(f"\n{'='*80}")
    print(f"📊 Combined Conditions (Shooting Star Detection)")
    print(f"{'='*80}")

    # Current strict thresholds
    strict = metrics_df[
        (metrics_df['upper_shadow_ratio'] >= 2.0) &
        (metrics_df['lower_shadow_ratio'] <= 0.1) &
        (metrics_df['body_position'] <= 0.33)
    ]
    print(f"\n  Current STRICT thresholds (upper>=2.0, lower<=0.1, pos<=0.33):")
    print(f"    Matching candles: {len(strict)} ({len(strict)/len(metrics_df)*100:.3f}%)")

    # Relaxed thresholds option 1
    relaxed1 = metrics_df[
        (metrics_df['upper_shadow_ratio'] >= 1.5) &
        (metrics_df['lower_shadow_ratio'] <= 0.3) &
        (metrics_df['body_position'] <= 0.4)
    ]
    print(f"\n  RELAXED thresholds v1 (upper>=1.5, lower<=0.3, pos<=0.4):")
    print(f"    Matching candles: {len(relaxed1)} ({len(relaxed1)/len(metrics_df)*100:.3f}%)")

    # Relaxed thresholds option 2
    relaxed2 = metrics_df[
        (metrics_df['upper_shadow_ratio'] >= 1.5) &
        (metrics_df['lower_shadow_ratio'] <= 0.5) &
        (metrics_df['body_position'] <= 0.4)
    ]
    print(f"\n  RELAXED thresholds v2 (upper>=1.5, lower<=0.5, pos<=0.4):")
    print(f"    Matching candles: {len(relaxed2)} ({len(relaxed2)/len(metrics_df)*100:.3f}%)")

    # Medium thresholds
    medium = metrics_df[
        (metrics_df['upper_shadow_ratio'] >= 1.8) &
        (metrics_df['lower_shadow_ratio'] <= 0.3) &
        (metrics_df['body_position'] <= 0.35)
    ]
    print(f"\n  MEDIUM thresholds (upper>=1.8, lower<=0.3, pos<=0.35):")
    print(f"    Matching candles: {len(medium)} ({len(medium)/len(metrics_df)*100:.3f}%)")

    print(f"\n{'='*80}")
    print(f"💡 Recommendations:")
    print(f"{'='*80}")

    if len(strict) == 0:
        print(f"\n  ⚠️  Current thresholds are TOO STRICT - No matching candles!")
        print(f"  ✓ Recommended: Use RELAXED v1 or MEDIUM thresholds")
    elif len(strict) < len(metrics_df) * 0.001:  # < 0.1%
        print(f"\n  ⚠️  Current thresholds are very strict - Only {len(strict)/len(metrics_df)*100:.3f}%")
        print(f"  ✓ Consider relaxing to get 0.5-2% detection rate")
    else:
        print(f"\n  ✓ Current thresholds seem reasonable - {len(strict)/len(metrics_df)*100:.3f}%")

    print(f"\n  Based on analysis:")
    print(f"    • Target detection rate: 0.5% - 2% is typical for Shooting Star")
    print(f"    • RELAXED v1 gives {len(relaxed1)/len(metrics_df)*100:.2f}% detection rate")
    print(f"    • MEDIUM gives {len(medium)/len(metrics_df)*100:.2f}% detection rate")

    print(f"\n{'='*80}\n")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Shooting Star metrics in historical data')
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file (e.g., historical/BTC-USDT/BTC-USDT-5m.csv)'
    )

    args = parser.parse_args()

    analyze_shooting_star_metrics(args.csv)


if __name__ == '__main__':
    main()
