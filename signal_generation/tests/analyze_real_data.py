"""
تحلیل عمیق داده‌های واقعی برای Shooting Star

این اسکریپت:
1. می‌بیند چند کندل فیزیک Shooting Star دارند (بدون uptrend check)
2. می‌بیند چند کندل در uptrend هستند
3. می‌بیند چند کندل هم فیزیک + هم uptrend دارند
4. threshold ها را کمی راحت‌تر می‌کند و دوباره تست می‌کند

Version: 1.0.0
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from signal_generation.analyzers.patterns.candlestick.shooting_star import ShootingStarPattern


def load_data(data_dir: str, timeframe: str = '1h') -> pd.DataFrame:
    """Load historical data"""
    filepath = os.path.join(data_dir, f'{timeframe}.csv')

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]

    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing required columns in {filepath}")
        return None

    return df


def analyze_shooting_star_distribution(df: pd.DataFrame) -> dict:
    """
    تحلیل عمیق کندل‌ها برای Shooting Star

    Returns:
        dict با آمار مختلف
    """
    total_candles = len(df)

    # Detector با uptrend غیرفعال
    detector_no_uptrend = ShootingStarPattern(require_uptrend=False)

    # Detector با uptrend فعال
    detector_with_uptrend = ShootingStarPattern(require_uptrend=True, min_uptrend_score=50.0)

    # آمارهای جمع‌آوری شده
    candles_with_physics = 0  # فیزیک Shooting Star دارند
    candles_in_uptrend = 0    # در uptrend هستند
    candles_both = 0           # هر دو شرط را دارند

    # نمونه‌های near-miss (نزدیک به threshold)
    near_miss_upper = []
    near_miss_lower = []
    near_miss_body = []
    near_miss_uptrend = []

    # برای tracking progress
    print(f"\n📊 Analyzing {total_candles} candles...")
    print_interval = max(1, total_candles // 20)  # Print every 5%

    for i in range(10, total_candles):  # شروع از کندل 10 (برای context)
        if i % print_interval == 0:
            progress = (i / total_candles) * 100
            print(f"  Progress: {progress:.1f}% ({i}/{total_candles})")

        df_window = df.iloc[:i+1]

        # تست با uptrend غیرفعال
        has_physics = detector_no_uptrend.detect(df_window)

        # تست uptrend
        context_score = detector_with_uptrend._analyze_context(df_window)
        in_uptrend = context_score >= detector_with_uptrend.min_uptrend_score

        # تست با هر دو شرط
        has_both = detector_with_uptrend.detect(df_window)

        if has_physics:
            candles_with_physics += 1

        if in_uptrend:
            candles_in_uptrend += 1

        if has_both:
            candles_both += 1

        # بررسی near-miss (کندل‌هایی که نزدیک threshold هستند)
        if not has_physics:
            candle = df_window.iloc[-1]
            o = candle['open']
            h = candle['high']
            l = candle['low']
            c = candle['close']

            body_size = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            full_range = h - l

            if full_range > 0:
                upper_shadow_pct = upper_shadow / full_range
                lower_shadow_pct = lower_shadow / full_range
                body_pct = body_size / full_range

                # Near-miss upper shadow (40-50% range)
                if 0.4 <= upper_shadow_pct < 0.5 and lower_shadow_pct <= 0.2 and body_pct <= 0.3:
                    near_miss_upper.append({
                        'index': i,
                        'upper_shadow_pct': upper_shadow_pct,
                        'context_score': context_score
                    })

                # Near-miss lower shadow (20-30% range)
                if upper_shadow_pct >= 0.5 and 0.2 < lower_shadow_pct <= 0.3 and body_pct <= 0.3:
                    near_miss_lower.append({
                        'index': i,
                        'lower_shadow_pct': lower_shadow_pct,
                        'context_score': context_score
                    })

                # Near-miss body (30-40% range)
                if upper_shadow_pct >= 0.5 and lower_shadow_pct <= 0.2 and 0.3 < body_pct <= 0.4:
                    near_miss_body.append({
                        'index': i,
                        'body_pct': body_pct,
                        'context_score': context_score
                    })

        # Near-miss uptrend (40-50 score)
        if has_physics and not in_uptrend:
            if 40 <= context_score < 50:
                near_miss_uptrend.append({
                    'index': i,
                    'context_score': context_score
                })

    return {
        'total_candles': total_candles,
        'candles_with_physics': candles_with_physics,
        'candles_in_uptrend': candles_in_uptrend,
        'candles_both': candles_both,
        'near_miss_upper': near_miss_upper[:5],  # Top 5
        'near_miss_lower': near_miss_lower[:5],
        'near_miss_body': near_miss_body[:5],
        'near_miss_uptrend': near_miss_uptrend[:5]
    }


def test_relaxed_thresholds(df: pd.DataFrame):
    """تست با threshold های راحت‌تر"""
    print("\n" + "="*80)
    print("🔧 Testing with RELAXED thresholds")
    print("="*80 + "\n")

    relaxed_configs = [
        {
            'name': 'Slightly Relaxed',
            'min_upper_shadow_pct': 0.45,  # 50% → 45%
            'max_lower_shadow_pct': 0.25,  # 20% → 25%
            'max_body_pct': 0.35,           # 30% → 35%
            'min_uptrend_score': 45.0       # 50 → 45
        },
        {
            'name': 'Moderately Relaxed',
            'min_upper_shadow_pct': 0.40,  # 50% → 40%
            'max_lower_shadow_pct': 0.30,  # 20% → 30%
            'max_body_pct': 0.40,           # 30% → 40%
            'min_uptrend_score': 40.0       # 50 → 40
        },
        {
            'name': 'Very Relaxed',
            'min_upper_shadow_pct': 0.35,  # 50% → 35%
            'max_lower_shadow_pct': 0.35,  # 20% → 35%
            'max_body_pct': 0.45,           # 30% → 45%
            'min_uptrend_score': 35.0       # 50 → 35
        }
    ]

    results = []

    for config in relaxed_configs:
        print(f"\nTesting: {config['name']}")
        print(f"  Thresholds: upper>={config['min_upper_shadow_pct']*100:.0f}%, "
              f"lower<={config['max_lower_shadow_pct']*100:.0f}%, "
              f"body<={config['max_body_pct']*100:.0f}%, "
              f"uptrend>={config['min_uptrend_score']}")

        detector = ShootingStarPattern(
            min_upper_shadow_pct=config['min_upper_shadow_pct'],
            max_lower_shadow_pct=config['max_lower_shadow_pct'],
            max_body_pct=config['max_body_pct'],
            require_uptrend=True,
            min_uptrend_score=config['min_uptrend_score']
        )

        count = 0
        for i in range(10, len(df)):
            df_window = df.iloc[:i+1]
            if detector.detect(df_window):
                count += 1

        rate = (count / len(df)) * 100
        print(f"  ✓ Detected: {count} patterns ({rate:.3f}%)")

        results.append({
            'name': config['name'],
            'count': count,
            'rate': rate
        })

    return results


def main():
    """Main function"""
    print("\n" + "="*80)
    print("🔍 DEEP ANALYSIS: Shooting Star in Real BTC Data")
    print("="*80 + "\n")

    # تعیین مسیر داده
    data_dir = 'historical/BTC-USDT'

    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run this from the project root directory")
        return

    # بارگذاری داده
    print(f"📂 Loading data from: {data_dir}")
    df = load_data(data_dir, '1h')

    if df is None:
        return

    print(f"✓ Loaded {len(df)} candles")
    print(f"  Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

    # تحلیل توزیع
    print("\n" + "="*80)
    print("📊 DISTRIBUTION ANALYSIS")
    print("="*80)

    stats = analyze_shooting_star_distribution(df)

    print(f"\n✓ Analysis complete!")
    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    print(f"Total candles analyzed: {stats['total_candles']:,}")
    print(f"")
    print(f"Candles with Shooting Star PHYSICS (no uptrend check): {stats['candles_with_physics']:,}")
    print(f"  → Rate: {(stats['candles_with_physics'] / stats['total_candles'] * 100):.3f}%")
    print(f"")
    print(f"Candles in UPTREND (score >= 50): {stats['candles_in_uptrend']:,}")
    print(f"  → Rate: {(stats['candles_in_uptrend'] / stats['total_candles'] * 100):.1f}%")
    print(f"")
    print(f"Candles with BOTH (physics + uptrend): {stats['candles_both']:,}")
    print(f"  → Rate: {(stats['candles_both'] / stats['total_candles'] * 100):.3f}%")
    print("-"*80)

    # Near-miss analysis
    if stats['near_miss_upper']:
        print(f"\n⚠️  Near-miss: Upper shadow (40-50%): {len(stats['near_miss_upper'])} examples")
        for nm in stats['near_miss_upper'][:3]:
            print(f"    Index {nm['index']}: upper={nm['upper_shadow_pct']*100:.1f}%, context={nm['context_score']:.1f}")

    if stats['near_miss_lower']:
        print(f"\n⚠️  Near-miss: Lower shadow (20-30%): {len(stats['near_miss_lower'])} examples")
        for nm in stats['near_miss_lower'][:3]:
            print(f"    Index {nm['index']}: lower={nm['lower_shadow_pct']*100:.1f}%, context={nm['context_score']:.1f}")

    if stats['near_miss_body']:
        print(f"\n⚠️  Near-miss: Body size (30-40%): {len(stats['near_miss_body'])} examples")
        for nm in stats['near_miss_body'][:3]:
            print(f"    Index {nm['index']}: body={nm['body_pct']*100:.1f}%, context={nm['context_score']:.1f}")

    if stats['near_miss_uptrend']:
        print(f"\n⚠️  Near-miss: Uptrend (40-50 score): {len(stats['near_miss_uptrend'])} examples")
        for nm in stats['near_miss_uptrend'][:3]:
            print(f"    Index {nm['index']}: context={nm['context_score']:.1f}")

    # تست threshold های راحت‌تر
    relaxed_results = test_relaxed_thresholds(df)

    # خلاصه نهایی
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)

    if stats['candles_with_physics'] == 0:
        print("\n❌ NO candles have Shooting Star physics in this dataset!")
        print("   → The thresholds are too strict for BTC data")
        print("   → Consider relaxing the thresholds")
    elif stats['candles_both'] == 0:
        print(f"\n⚠️  {stats['candles_with_physics']} candles have physics, but NONE are in uptrend!")
        print("   → Shooting Star patterns exist, but not in uptrend context")
        print("   → Consider lowering min_uptrend_score")
    else:
        print(f"\n✅ Found {stats['candles_both']} valid Shooting Stars (physics + uptrend)")
        print("   → Pattern detection is working correctly")

    print("\n" + "="*80)
    print()


if __name__ == '__main__':
    main()
