"""
Test TA-Lib by copying EXACT candles from real BTC data

Since TA-Lib detected 75 Shooting Stars in real data but not in our synthetic data,
let's copy exact candles from BTC data to understand what triggers detection.
"""

import pandas as pd
import numpy as np
import talib


def load_btc_data(filename='historical/BTC-USDT/1hour.csv'):
    """Load BTC OHLCV data."""
    df = pd.read_csv(filename)
    df = df.astype({
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    })
    return df


def test_exact_copy():
    """
    Copy exact candles from BTC data where TA-Lib detected Shooting Star.

    Test scenarios:
    1. Just the detection candle (should FAIL)
    2. Detection candle + 1 previous (should FAIL)
    3. Detection candle + 5 previous (might work)
    4. Detection candle + 10 previous (should work)
    5. Detection candle + 20 previous (should work)
    """

    # Load BTC data
    df = load_btc_data()

    # Run TA-Lib to find detections
    pattern = talib.CDLSHOOTINGSTAR(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    detections = np.where(pattern != 0)[0]
    print(f"\nTotal detections in full BTC data: {len(detections)}")

    if len(detections) == 0:
        print("No detections found!")
        return

    # Use first detection for testing
    detection_idx = detections[0]
    print(f"\nUsing detection at index {detection_idx}")

    print(f"\n{'='*70}")
    print("Testing how many previous candles TA-Lib needs")
    print(f"{'='*70}")

    # Test with different lookback periods
    for lookback in [0, 1, 2, 3, 5, 10, 15, 20, 50, 100]:
        start_idx = max(0, detection_idx - lookback)
        end_idx = detection_idx + 1  # Include detection candle

        # Extract subset
        subset = df.iloc[start_idx:end_idx].copy()

        # Run TA-Lib on subset
        pattern_subset = talib.CDLSHOOTINGSTAR(
            subset['open'].values,
            subset['high'].values,
            subset['low'].values,
            subset['close'].values
        )

        detections_subset = np.where(pattern_subset != 0)[0]

        # Check if last candle was detected
        if len(detections_subset) > 0 and detections_subset[-1] == len(subset) - 1:
            print(f"✅ lookback={lookback:3d}: DETECTED (with {len(subset)} total candles)")
        else:
            print(f"❌ lookback={lookback:3d}: NOT detected (with {len(subset)} total candles)")

    # Show the detection candle details
    print(f"\n{'='*70}")
    print("Detection candle details:")
    print(f"{'='*70}")

    candle = df.iloc[detection_idx]
    body = abs(candle['close'] - candle['open'])
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    full_range = candle['high'] - candle['low']

    print(f"\nOHLC: O={candle['open']:.2f}, H={candle['high']:.2f}, "
          f"L={candle['low']:.2f}, C={candle['close']:.2f}")
    print(f"Direction: {'Bullish' if candle['close'] > candle['open'] else 'Bearish'}")
    print(f"Upper shadow: {upper_shadow:.2f} ({upper_shadow/full_range*100:.1f}%)")
    print(f"Body: {body:.2f} ({body/full_range*100:.1f}%)")
    print(f"Lower shadow: {lower_shadow:.2f} ({lower_shadow/full_range*100:.1f}%)")
    print(f"Body position: {(min(candle['open'], candle['close']) - candle['low'])/full_range:.3f}")

    # Show previous candles
    print(f"\n{'='*70}")
    print("Previous 3 candles:")
    print(f"{'='*70}")

    for i in range(max(0, detection_idx - 3), detection_idx):
        c = df.iloc[i]
        direction = "🟢" if c['close'] > c['open'] else "🔴"
        print(f"{direction} Candle #{i}: O={c['open']:.2f}, H={c['high']:.2f}, "
              f"L={c['low']:.2f}, C={c['close']:.2f}")


def find_minimum_lookback():
    """
    For each detection in BTC data, find minimum number of previous candles needed.
    """

    # Load BTC data
    df = load_btc_data()

    # Run TA-Lib
    pattern = talib.CDLSHOOTINGSTAR(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    detections = np.where(pattern != 0)[0]

    print(f"\n{'='*70}")
    print("Finding minimum lookback for each detection")
    print(f"{'='*70}")

    min_lookbacks = []

    # Test first 10 detections
    for det_idx in detections[:10]:
        # Binary search for minimum lookback
        min_lookback = None

        for lookback in range(0, 101):
            start_idx = max(0, det_idx - lookback)
            end_idx = det_idx + 1

            subset = df.iloc[start_idx:end_idx].copy()

            pattern_subset = talib.CDLSHOOTINGSTAR(
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


def main():
    """Run all tests."""

    print("\n" + "="*70)
    print("TA-LIB REAL CONTEXT TEST")
    print("="*70)
    print("\nGoal: Understand how many previous candles TA-Lib needs")

    # Test 1
    test_exact_copy()

    # Test 2
    find_minimum_lookback()

    print("\n" + "="*70)
    print("FINAL CONCLUSIONS")
    print("="*70)
    print("\nNow we know:")
    print("1. Minimum number of candles TA-Lib needs")
    print("2. Whether context matters or just candle count")
    print("3. How to properly call TA-Lib in our code")


if __name__ == "__main__":
    main()
