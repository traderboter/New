"""
Test TA-Lib CDLHAMMER function - Phase by Phase

This script tests Hammer pattern detection similar to Shooting Star tests.

Hammer Characteristics:
- Bullish reversal pattern (opposite of Shooting Star)
- Small body at TOP of candle
- Long LOWER shadow (at least 2x body)
- Little to no UPPER shadow
- Best when appears after downtrend

Test Phases:
Phase 1: Single candle test (if this works, we're done!)
Phase 2: Multiple candles test (if Phase 1 fails)
Phase 3: Real BTC data test
Phase 4: Find minimum lookback (like we did for Shooting Star)
"""

import numpy as np
import pandas as pd
import talib


def create_hammer_candle(base_price=100):
    """
    Create a perfect Hammer candle.

    Characteristics:
    - Small body at top (bullish or bearish)
    - Long lower shadow (>= 2x body)
    - Very small or no upper shadow
    """
    body_size = 2.0  # Small body
    lower_shadow = 8.0  # Long lower shadow
    upper_shadow = 0.2  # Very small upper shadow

    # Bullish candle (close > open)
    low = base_price
    open_price = base_price + lower_shadow
    close = base_price + lower_shadow + body_size
    high = close + upper_shadow

    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }


def create_downtrend_candles(n_candles=10, start_price=120):
    """Create candles forming a downtrend."""
    candles = []
    price = start_price

    for i in range(n_candles):
        # Bearish candle with decreasing prices
        candle_size = np.random.uniform(1.5, 3.0)
        open_price = price
        close = price - candle_size
        high = open_price + np.random.uniform(0.1, 0.5)
        low = close - np.random.uniform(0.1, 0.5)

        candles.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

        price = close - np.random.uniform(0.5, 1.0)  # Gap down

    return candles


def test_hammer(df, scenario_name):
    """Test TA-Lib CDLHAMMER on given dataframe."""

    # Ensure float64 type for TA-Lib
    df = df.astype(np.float64)

    # Call TA-Lib
    pattern = talib.CDLHAMMER(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    # Find detections
    detections = np.where(pattern != 0)[0]

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")
    print(f"Total candles: {len(df)}")
    print(f"Detections found: {len(detections)}")

    if len(detections) > 0:
        print(f"\n✅ DETECTED! Indices: {detections}")
        print(f"\nDetection details:")
        for idx in detections:
            print(f"\n  Candle #{idx}:")
            print(f"    Pattern value: {pattern[idx]}")
            print(f"    OHLC: O={df.iloc[idx]['open']:.2f}, H={df.iloc[idx]['high']:.2f}, "
                  f"L={df.iloc[idx]['low']:.2f}, C={df.iloc[idx]['close']:.2f}")

            # Calculate candle characteristics
            candle = df.iloc[idx]
            body = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            full_range = candle['high'] - candle['low']

            print(f"    Body: {body:.2f} ({body/full_range*100:.1f}% of range)")
            print(f"    Upper shadow: {upper_shadow:.2f} ({upper_shadow/full_range*100:.1f}% of range)")
            print(f"    Lower shadow: {lower_shadow:.2f} ({lower_shadow/full_range*100:.1f}% of range)")
    else:
        print("\n❌ NO DETECTIONS")
        print(f"\nLast candle details:")
        candle = df.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        full_range = candle['high'] - candle['low']

        print(f"  OHLC: O={candle['open']:.2f}, H={candle['high']:.2f}, "
              f"L={candle['low']:.2f}, C={candle['close']:.2f}")
        print(f"  Body: {body:.2f} ({body/full_range*100:.1f}% of range)")
        print(f"  Upper shadow: {upper_shadow:.2f} ({upper_shadow/full_range*100:.1f}% of range)")
        print(f"  Lower shadow: {lower_shadow:.2f} ({lower_shadow/full_range*100:.1f}% of range)")

    return len(detections) > 0


def phase1_single_candle():
    """PHASE 1: Test with single candle only."""

    print("\n" + "="*70)
    print("PHASE 1: SINGLE CANDLE TEST")
    print("="*70)
    print("\nGoal: See if TA-Lib can detect Hammer with just 1 candle")

    # Test 1: Perfect Hammer alone
    hammer = create_hammer_candle(100)
    df_single = pd.DataFrame([hammer])
    detected = test_hammer(df_single, "Perfect Hammer - Single Candle")

    if detected:
        print("\n🎉 SUCCESS! TA-Lib can detect Hammer with 1 candle!")
        return True
    else:
        print("\n⚠️ FAILED! TA-Lib needs more candles (like Shooting Star)")
        return False


def phase2_multiple_candles():
    """PHASE 2: Test with multiple candles."""

    print("\n" + "="*70)
    print("PHASE 2: MULTIPLE CANDLES TEST")
    print("="*70)
    print("\nGoal: Find minimum number of candles needed")

    # Test with different numbers of context candles
    for n_candles in [1, 2, 3, 5, 10, 12, 15, 20]:
        if n_candles == 1:
            # Already tested in Phase 1
            continue

        downtrend = create_downtrend_candles(n_candles - 1, start_price=120)
        hammer = create_hammer_candle(90)
        df = pd.DataFrame(downtrend + [hammer])

        detected = test_hammer(df, f"Hammer after {n_candles-1}-candle downtrend")

        if detected:
            print(f"\n✅ MINIMUM FOUND: {n_candles} candles!")
            return n_candles

        print(f"❌ NOT detected with {n_candles} candles")

    print("\n⚠️ Could not find detection even with 20 candles")
    return None


def phase3_real_data():
    """PHASE 3: Test with real BTC data."""

    print("\n" + "="*70)
    print("PHASE 3: REAL BTC DATA TEST")
    print("="*70)

    # Load BTC data
    try:
        df = pd.read_csv('historical/BTC-USDT/1hour.csv')
        df = df.astype({
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.float64
        })
    except Exception as e:
        print(f"\n❌ Could not load BTC data: {e}")
        return None

    print(f"\nTotal candles: {len(df):,}")

    # Run TA-Lib on full dataset
    pattern = talib.CDLHAMMER(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    detections = np.where(pattern != 0)[0]

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total detections: {len(detections)}")
    print(f"Detection rate: {len(detections)/len(df)*100:.2f}%")

    if len(detections) > 0:
        print(f"\n✅ Found Hammer patterns in real data!")
        print(f"\nFirst 5 detections at indices: {detections[:5]}")
        print(f"Last 5 detections at indices: {detections[-5:]}")

        # Show first detection details
        idx = detections[0]
        candle = df.iloc[idx]
        body = abs(candle['close'] - candle['open'])
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        full_range = candle['high'] - candle['low']

        print(f"\nFirst detection (index {idx}):")
        print(f"  OHLC: O={candle['open']:.2f}, H={candle['high']:.2f}, "
              f"L={candle['low']:.2f}, C={candle['close']:.2f}")
        print(f"  Direction: {'Bullish' if candle['close'] > candle['open'] else 'Bearish'}")
        print(f"  Upper shadow: {upper_shadow/full_range*100:.1f}%")
        print(f"  Body: {body/full_range*100:.1f}%")
        print(f"  Lower shadow: {lower_shadow/full_range*100:.1f}%")

        return len(detections)
    else:
        print(f"\n❌ No Hammer patterns found in real data!")
        return 0


def phase4_minimum_lookback():
    """PHASE 4: Find exact minimum lookback needed."""

    print("\n" + "="*70)
    print("PHASE 4: MINIMUM LOOKBACK TEST (like Shooting Star)")
    print("="*70)

    try:
        df = pd.read_csv('historical/BTC-USDT/1hour.csv')
        df = df.astype({
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.float64
        })
    except Exception as e:
        print(f"\n❌ Could not load BTC data: {e}")
        return None

    # Run TA-Lib to find detections
    pattern = talib.CDLHAMMER(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    detections = np.where(pattern != 0)[0]

    if len(detections) == 0:
        print("\n❌ No detections in full dataset, cannot test lookback")
        return None

    print(f"\nTotal detections in full data: {len(detections)}")
    print(f"\nTesting minimum lookback for first detection at index {detections[0]}...")

    detection_idx = detections[0]

    print(f"\n{'='*70}")
    print("Testing different lookback values:")
    print(f"{'='*70}")

    for lookback in [0, 1, 2, 3, 5, 10, 11, 12, 15, 20, 50, 100]:
        start_idx = max(0, detection_idx - lookback)
        end_idx = detection_idx + 1

        subset = df.iloc[start_idx:end_idx].copy()

        pattern_subset = talib.CDLHAMMER(
            subset['open'].values,
            subset['high'].values,
            subset['low'].values,
            subset['close'].values
        )

        detections_subset = np.where(pattern_subset != 0)[0]

        if len(detections_subset) > 0 and detections_subset[-1] == len(subset) - 1:
            print(f"✅ lookback={lookback:3d}: DETECTED (with {len(subset)} total candles)")
        else:
            print(f"❌ lookback={lookback:3d}: NOT detected (with {len(subset)} total candles)")

    # Find exact minimum
    print(f"\n{'='*70}")
    print("Finding exact minimum for first 10 detections:")
    print(f"{'='*70}")

    min_lookbacks = []

    for det_idx in detections[:10]:
        min_lookback = None

        for lookback in range(0, 101):
            start_idx = max(0, det_idx - lookback)
            end_idx = det_idx + 1

            subset = df.iloc[start_idx:end_idx].copy()

            pattern_subset = talib.CDLHAMMER(
                subset['open'].values,
                subset['high'].values,
                subset['low'].values,
                subset['close'].values
            )

            detections_subset = np.where(pattern_subset != 0)[0]

            if len(detections_subset) > 0 and detections_subset[-1] == len(subset) - 1:
                min_lookback = lookback
                break

        if min_lookback is not None:
            min_lookbacks.append(min_lookback)
            print(f"Detection at index {det_idx}: min lookback = {min_lookback}")
        else:
            print(f"Detection at index {det_idx}: NOT FOUND (??)")

    if len(min_lookbacks) > 0:
        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"Minimum lookback needed:")
        print(f"  Min: {min(min_lookbacks)}")
        print(f"  Max: {max(min_lookbacks)}")
        print(f"  Mean: {np.mean(min_lookbacks):.1f}")
        print(f"  Median: {np.median(min_lookbacks):.0f}")

        return int(np.median(min_lookbacks))

    return None


def main():
    """Run all test phases."""

    print("\n" + "="*70)
    print("TA-LIB HAMMER DETECTION TEST")
    print("="*70)
    print("\nGoal: Understand how TA-Lib CDLHAMMER works")
    print("Hammer is opposite of Shooting Star:")
    print("  - Bullish reversal (vs bearish)")
    print("  - Long LOWER shadow (vs upper)")
    print("  - In downtrend (vs uptrend)")

    # Phase 1: Single candle
    phase1_success = phase1_single_candle()

    if phase1_success:
        print("\n" + "="*70)
        print("✅ PHASE 1 PASSED - No need for Phase 2!")
        print("="*70)
    else:
        # Phase 2: Multiple candles
        min_candles = phase2_multiple_candles()

        if min_candles:
            print(f"\n{'='*70}")
            print(f"✅ PHASE 2 COMPLETE - Minimum {min_candles} candles needed")
            print(f"{'='*70}")

    # Phase 3: Real data
    detection_count = phase3_real_data()

    if detection_count and detection_count > 0:
        # Phase 4: Minimum lookback
        min_lookback = phase4_minimum_lookback()

        if min_lookback is not None:
            print(f"\n{'='*70}")
            print(f"✅ PHASE 4 COMPLETE - Minimum lookback: {min_lookback}")
            print(f"{'='*70}")

    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nComparison with Shooting Star:")
    print("  Shooting Star: Minimum 12 candles (11 lookback)")
    print(f"  Hammer: ??? (to be determined from tests above)")
    print("\nNext steps:")
    print("  1. If Hammer also needs 12 candles → same fix as Shooting Star")
    print("  2. If different → document and fix accordingly")


if __name__ == "__main__":
    main()
