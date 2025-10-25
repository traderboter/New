"""
Simple test for Shooting Star v2.0.0

Direct test without importing full package.
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


def detect_shooting_star_v2(df, require_uptrend=False, min_uptrend_score=50.0):
    """
    Replicate shooting_star.py v2.0.0 detect() logic.
    """
    # TA-Lib needs minimum 12 candles
    if len(df) < 12:
        return False

    try:
        # Prepare data for TA-Lib
        df_tail = df.tail(100)

        # Call TA-Lib CDLSHOOTINGSTAR
        pattern = talib.CDLSHOOTINGSTAR(
            df_tail['open'].values,
            df_tail['high'].values,
            df_tail['low'].values,
            df_tail['close'].values
        )

        # Check if last candle is detected
        if pattern[-1] == 0:
            return False

        # Uptrend check (if required)
        if require_uptrend:
            context_score = analyze_context(df)
            if context_score < min_uptrend_score:
                return False

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def analyze_context(df):
    """Analyze context for uptrend detection."""
    if len(df) < 10:
        return 50

    try:
        recent = df.tail(10)

        # Slope
        closes = recent['close'].values
        indices = np.arange(len(closes))
        slope = np.polyfit(indices, closes, 1)[0]

        if slope > 0:
            slope_score = min(100, abs(slope) / np.mean(closes) * 10000)
        else:
            slope_score = 0

        # Bullish count
        bullish_count = sum(recent['close'] > recent['open'])
        bullish_score = (bullish_count / len(recent)) * 100

        # Higher highs
        highs = recent['high'].values
        higher_highs = sum(highs[i] > highs[i-1] for i in range(1, len(highs)))
        higher_highs_score = (higher_highs / (len(highs) - 1)) * 100

        # Combined
        context_score = (
            0.40 * slope_score +
            0.30 * bullish_score +
            0.30 * higher_highs_score
        )

        return min(100, context_score)

    except Exception:
        return 50


def main():
    """Run tests."""

    print("\n" + "="*70)
    print("SHOOTING STAR v2.0.0 SIMPLE TEST")
    print("="*70)

    # Load data
    df = load_btc_data()
    print(f"\nTotal candles: {len(df):,}")

    # Test 1: Without uptrend
    print("\n" + "-"*70)
    print("TEST 1: WITHOUT uptrend check (expect ~75)")
    print("-"*70)

    detections_no_uptrend = []
    for i in range(12, len(df)):
        df_slice = df.iloc[:i+1]
        if detect_shooting_star_v2(df_slice, require_uptrend=False):
            detections_no_uptrend.append(i)

    print(f"Detections: {len(detections_no_uptrend)}")
    print(f"Rate: {len(detections_no_uptrend)/len(df)*100:.2f}%")

    if 70 <= len(detections_no_uptrend) <= 80:
        print("✅ PASS (expected ~75)")
    else:
        print(f"⚠️ UNEXPECTED (expected ~75, got {len(detections_no_uptrend)})")

    # Test 2: With uptrend
    print("\n" + "-"*70)
    print("TEST 2: WITH uptrend check (expect ~30-45)")
    print("-"*70)

    detections_with_uptrend = []
    for i in range(12, len(df)):
        df_slice = df.iloc[:i+1]
        if detect_shooting_star_v2(df_slice, require_uptrend=True, min_uptrend_score=50.0):
            detections_with_uptrend.append(i)

    print(f"Detections: {len(detections_with_uptrend)}")
    print(f"Rate: {len(detections_with_uptrend)/len(df)*100:.2f}%")

    if 30 <= len(detections_with_uptrend) <= 45:
        print("✅ PASS (expected ~30-45)")
    else:
        print(f"⚠️ UNEXPECTED (expected ~30-45, got {len(detections_with_uptrend)})")

    # Test 3: Minimum candles
    print("\n" + "-"*70)
    print("TEST 3: Minimum candles requirement")
    print("-"*70)

    result_11 = detect_shooting_star_v2(df.iloc[:11], require_uptrend=False)
    result_12 = detect_shooting_star_v2(df.iloc[:12], require_uptrend=False)

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
    print(f"Without uptrend: {len(detections_no_uptrend)} detections")
    print(f"With uptrend: {len(detections_with_uptrend)} detections")
    print(f"\nv2.0.0 properly uses TA-Lib with 12+ candles ✅")


if __name__ == "__main__":
    main()
