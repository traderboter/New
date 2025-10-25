"""
Demo: Recency Scoring System

این اسکریپت نشان می‌دهد سیستم recency scoring چطور کار می‌کند.

1. یک Hammer واقعی را در BTC data پیدا می‌کند
2. با lookback مختلف تست می‌کند
3. نشان می‌دهد score چطور تغییر می‌کند
"""

import talib
import pandas as pd
import numpy as np
from pathlib import Path

# Load BTC data
csv_path = Path(__file__).parent.parent / 'historical' / 'BTC-USDT' / '1hour.csv'
df = pd.read_csv(csv_path)

print("="*70)
print("🎯 Demo: Pattern Recency Scoring System")
print("="*70)

print(f"\n✅ Loaded {len(df)} BTC candles")

# =============================================================================
# Step 1: Find a real Hammer in the data
# =============================================================================

print("\n" + "="*70)
print("Step 1: Finding a real Hammer in BTC data")
print("="*70)

result = talib.CDLHAMMER(
    df['open'].values,
    df['high'].values,
    df['low'].values,
    df['close'].values
)

# Find all Hammers
hammer_indices = np.where(result != 0)[0]

if len(hammer_indices) == 0:
    print("❌ No Hammer found in data!")
    exit(1)

print(f"\n✅ Found {len(hammer_indices)} Hammers")
print(f"First Hammer at index: {hammer_indices[0]}")

# Pick a Hammer in the middle for demo
demo_hammer_idx = hammer_indices[len(hammer_indices) // 2]
print(f"Using Hammer at index {demo_hammer_idx} for demo")

# Show the candle
candle = df.iloc[demo_hammer_idx]
body = abs(candle['close'] - candle['open'])
lower_shadow = min(candle['open'], candle['close']) - candle['low']
upper_shadow = candle['high'] - max(candle['open'], candle['close'])
full_range = candle['high'] - candle['low']

print(f"\n📊 Hammer Candle Details (index {demo_hammer_idx}):")
print(f"  Open:   {candle['open']:.2f}")
print(f"  High:   {candle['high']:.2f}")
print(f"  Low:    {candle['low']:.2f}")
print(f"  Close:  {candle['close']:.2f}")
print(f"  Body:   {body/full_range*100:.1f}%")
print(f"  Upper:  {upper_shadow/full_range*100:.1f}%")
print(f"  Lower:  {lower_shadow/full_range*100:.1f}% ← characteristic!")

# =============================================================================
# Step 2: Simulate different recency scenarios
# =============================================================================

print("\n" + "="*70)
print("Step 2: Simulating Recency Scenarios")
print("="*70)

# Config
WEIGHT = 3  # Hammer weight
BASE_CONFIDENCE = 0.85  # Base confidence for this Hammer
RECENCY_MULTIPLIERS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

print(f"\n⚙️  Configuration:")
print(f"  Pattern weight: {WEIGHT}")
print(f"  Base confidence: {BASE_CONFIDENCE}")
print(f"  Recency multipliers: {RECENCY_MULTIPLIERS}")

print(f"\n📊 Recency Scenarios:")
print("-"*70)
print(f"{'Scenario':<30} {'Candles':<10} {'Mult':<8} {'Conf':<8} {'Score':<10}")
print("-"*70)

for candles_ago in range(6):
    multiplier = RECENCY_MULTIPLIERS[candles_ago]
    adjusted_confidence = BASE_CONFIDENCE * multiplier
    score = WEIGHT * adjusted_confidence * multiplier

    scenario_name = f"Hammer در کندل {candles_ago} قبل"
    if candles_ago == 0:
        scenario_name = "Hammer در کندل آخر ⭐"

    print(f"{scenario_name:<30} {candles_ago:<10} {multiplier:<8.2f} "
          f"{adjusted_confidence:<8.3f} {score:<10.3f}")

# =============================================================================
# Step 3: Actual detection with different df slices
# =============================================================================

print("\n" + "="*70)
print("Step 3: Testing with Real DataFrame Slices")
print("="*70)

print(f"\nHammer is at index {demo_hammer_idx}")
print(f"We'll test with df ending at different positions:")

# Test scenarios:
# - Exact: df ends exactly at Hammer (candles_ago = 0)
# - +1: df ends 1 candle after (candles_ago = 1)
# - +2: df ends 2 candles after (candles_ago = 2)
# etc.

test_scenarios = [
    (demo_hammer_idx + 1, 0, "Hammer در کندل آخر"),
    (demo_hammer_idx + 2, 1, "Hammer در 1 کندل قبل"),
    (demo_hammer_idx + 3, 2, "Hammer در 2 کندل قبل"),
    (demo_hammer_idx + 4, 3, "Hammer در 3 کندل قبل"),
    (demo_hammer_idx + 5, 4, "Hammer در 4 کندل قبل"),
    (demo_hammer_idx + 6, 5, "Hammer در 5 کندل قبل"),
]

print("\n" + "-"*70)
print(f"{'Scenario':<30} {'DF End':<10} {'Detected?':<12} {'Score':<10}")
print("-"*70)

for end_idx, expected_candles_ago, description in test_scenarios:
    if end_idx > len(df):
        continue

    # Get slice
    df_slice = df.iloc[:end_idx]

    # Run TA-Lib
    result_slice = talib.CDLHAMMER(
        df_slice['open'].values,
        df_slice['high'].values,
        df_slice['low'].values,
        df_slice['close'].values
    )

    # Check last 6 candles
    detected = False
    actual_candles_ago = None

    for i in range(min(6, len(result_slice))):
        idx = -(i + 1)
        if result_slice[idx] != 0:
            detected = True
            actual_candles_ago = i
            break

    if detected:
        multiplier = RECENCY_MULTIPLIERS[actual_candles_ago]
        adjusted_confidence = BASE_CONFIDENCE * multiplier
        score = WEIGHT * adjusted_confidence * multiplier

        print(f"{description:<30} {end_idx:<10} ✅ Yes (ago={actual_candles_ago}) {score:<10.3f}")
    else:
        print(f"{description:<30} {end_idx:<10} ❌ No           -")

# =============================================================================
# Step 4: Comparison
# =============================================================================

print("\n" + "="*70)
print("Step 4: Score Comparison")
print("="*70)

print(f"\n💡 مقایسه:")
print(f"  Hammer در کندل آخر:    score = {WEIGHT * BASE_CONFIDENCE * 1.0:.3f}")
print(f"  Hammer در 1 کندل قبل:  score = {WEIGHT * BASE_CONFIDENCE * 0.9:.3f} (-10%)")
print(f"  Hammer در 2 کندل قبل:  score = {WEIGHT * BASE_CONFIDENCE * 0.8:.3f} (-20%)")
print(f"  Hammer در 3 کندل قبل:  score = {WEIGHT * BASE_CONFIDENCE * 0.7:.3f} (-30%)")
print(f"  Hammer در 4 کندل قبل:  score = {WEIGHT * BASE_CONFIDENCE * 0.6:.3f} (-40%)")
print(f"  Hammer در 5 کندل قبل:  score = {WEIGHT * BASE_CONFIDENCE * 0.5:.3f} (-50%)")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("✅ Summary")
print("="*70)

print(f"""
🎯 با سیستم Recency Scoring:

1. ✅ الگوهای اخیر را نمی‌بریم
   - قبلاً: فقط کندل آخر → الگوهای 2-3 کندل قبل از دست می‌رفت
   - حالا: 5-6 کندل آخر → همه الگوهای اخیر را می‌گیریم

2. ✅ امتیازدهی منصفانه
   - الگوهای تازه‌تر → امتیاز بیشتر
   - الگوهای قدیمی‌تر → امتیاز کمتر

3. ✅ قابل تنظیم
   - برای هر الگو multipliers متفاوت
   - با backtesting بهینه می‌کنیم

4. ✅ شفاف
   - می‌دانیم چه الگویی کجا پیدا شده
   - تحلیل معاملات راحت‌تر
""")

print("="*70)
