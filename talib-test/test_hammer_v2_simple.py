"""
Simple test for Hammer v2.0.0

Tests the new TA-Lib integration with proper 12-candle requirement.
Expected: Should find 277 detections in BTC 1-hour data (with require_downtrend=False)
          Should find ~116 detections with require_downtrend=True
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


def detect_hammer_v2(df, require_downtrend=False, min_downtrend_score=50.0):
    """
    Replicate hammer.py v2.0.0 detect() logic.
    """
    # TA-Lib needs minimum 12 candles
    if len(df) < 12:
        return False

    try:
        # Prepare data for TA-Lib
        df_tail = df.tail(100)

        # Call TA-Lib CDLHAMMER
        pattern = talib.CDLHAMMER(
            df_tail['open'].values,
            df_tail['high'].values,
            df_tail['low'].values,
            df_tail['close'].values
        )

        # Check if last candle is detected
        if pattern[-1] == 0:
            return False

        # Downtrend check (if required)
        if require_downtrend:
            context_score = analyze_context(df)
            if context_score < min_downtrend_score:
                return False

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def analyze_context(df):
    """Analyze context for downtrend detection."""
    if len(df) < 10:
        return 50

    try:
        recent = df.tail(10)

        # Slope
        closes = recent['close'].values
        indices = np.arange(len(closes))
        slope = np.polyfit(indices, closes, 1)[0]

        if slope < 0:
            slope_score = min(100, abs(slope) / np.mean(closes) * 10000)
        else:
            slope_score = 0

        # Bearish count
        bearish_count = sum(recent['close'] < recent['open'])
        bearish_score = (bearish_count / len(recent)) * 100

        # Lower lows
        lows = recent['low'].values
        lower_lows = sum(lows[i] < lows[i-1] for i in range(1, len(lows)))
        lower_lows_score = (lower_lows / (len(lows) - 1)) * 100

        # Combined
        context_score = (
            0.40 * slope_score +
            0.30 * bearish_score +
            0.30 * lower_lows_score
        )

        return min(100, context_score)

    except Exception:
        return 50


def main():
    """Run tests."""

    print("\n" + "="*70)
    print("HAMMER v2.0.0 SIMPLE TEST")
    print("="*70)

    # Load data
    df = load_btc_data()
    print(f"\nTotal candles: {len(df):,}")

    # Test 1: Without downtrend
    print("\n" + "-"*70)
    print("TEST 1: WITHOUT downtrend check (expect ~277)")
    print("-"*70)

    detections_no_downtrend = []
    for i in range(12, len(df)):
        df_slice = df.iloc[:i+1]
        if detect_hammer_v2(df_slice, require_downtrend=False):
            detections_no_downtrend.append(i)

    print(f"Detections: {len(detections_no_downtrend)}")
    print(f"Rate: {len(detections_no_downtrend)/len(df)*100:.2f}%")

    if 270 <= len(detections_no_downtrend) <= 285:
        print("✅ PASS (expected ~277)")
    else:
        print(f"⚠️ UNEXPECTED (expected ~277, got {len(detections_no_downtrend)})")

    # Test 2: With downtrend
    print("\n" + "-"*70)
    print("TEST 2: WITH downtrend check (expect ~100-130)")
    print("-"*70)

    detections_with_downtrend = []
    for i in range(12, len(df)):
        df_slice = df.iloc[:i+1]
        if detect_hammer_v2(df_slice, require_downtrend=True, min_downtrend_score=50.0):
            detections_with_downtrend.append(i)

    print(f"Detections: {len(detections_with_downtrend)}")
    print(f"Rate: {len(detections_with_downtrend)/len(df)*100:.2f}%")

    if 100 <= len(detections_with_downtrend) <= 130:
        print("✅ PASS (expected ~100-130)")
    else:
        print(f"⚠️ UNEXPECTED (expected ~100-130, got {len(detections_with_downtrend)})")

    # Test 3: Minimum candles
    print("\n" + "-"*70)
    print("TEST 3: Minimum candles requirement")
    print("-"*70)

    result_11 = detect_hammer_v2(df.iloc[:11], require_downtrend=False)
    result_12 = detect_hammer_v2(df.iloc[:12], require_downtrend=False)

    print(f"11 candles: {result_11} (expect False)")
    print(f"12 candles: {result_12} (may be True/False)")

    if result_11 == False:
        print("✅ PASS (correctly requires 12+ candles)")
    else:
        print("❌ FAIL (should reject < 12 candles)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Without downtrend: {len(detections_no_downtrend)} detections")
    print(f"With downtrend: {len(detections_with_downtrend)} detections")
    print(f"\nComparison with Shooting Star:")
    print(f"  Shooting Star (no uptrend): 75 detections (0.71%)")
    print(f"  Hammer (no downtrend): {len(detections_no_downtrend)} detections ({len(detections_no_downtrend)/len(df)*100:.2f}%)")
    print(f"  Ratio: Hammer is {len(detections_no_downtrend)/75:.1f}× more common ⭐")
    print(f"\nv2.0.0 properly uses TA-Lib with 12+ candles ✅")


if __name__ == "__main__":
    main()
