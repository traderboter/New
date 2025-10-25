"""
Test Shooting Star v2.0.0 with TA-Lib integration

This tests the new version that properly uses TA-Lib with 12+ candles.
Expected: Should find 75 detections in BTC 1-hour data (with require_uptrend=False)
          Should find ~37-38 detections with require_uptrend=True
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import talib

# Import directly without going through __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "shooting_star",
    "signal_generation/analyzers/patterns/candlestick/shooting_star.py"
)
shooting_star_module = importlib.util.module_from_spec(spec)

# Mock the base_pattern import
class BasePattern:
    def _validate_dataframe(self, df):
        return df is not None and len(df) > 0

sys.modules['signal_generation.analyzers.patterns.base_pattern'] = type('module', (), {'BasePattern': BasePattern})()

spec.loader.exec_module(shooting_star_module)
ShootingStarPattern = shooting_star_module.ShootingStarPattern


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


def test_v2_without_uptrend():
    """Test v2.0.0 without uptrend requirement (should match TA-Lib: 75 detections)."""

    print("\n" + "="*70)
    print("TEST 1: Shooting Star v2.0.0 WITHOUT uptrend check")
    print("="*70)
    print("Expected: ~75 detections (same as raw TA-Lib)")

    # Load data
    df = load_btc_data()
    print(f"\nTotal candles: {len(df):,}")

    # Create detector WITHOUT uptrend requirement
    detector = ShootingStarPattern(require_uptrend=False)
    print(f"Detector version: {detector.version}")
    print(f"Require uptrend: {detector.require_uptrend}")

    # Test each candle
    detections = []
    for i in range(12, len(df)):  # Start from 12 (minimum for TA-Lib)
        df_slice = df.iloc[:i+1]
        if detector.detect(df_slice):
            detections.append(i)

    print(f"\n✅ Detections found: {len(detections)}")
    print(f"Detection rate: {len(detections)/len(df)*100:.2f}%")

    if len(detections) > 0:
        print(f"\nFirst 5 detections at indices: {detections[:5]}")
        print(f"Last 5 detections at indices: {detections[-5:]}")

    # Compare with expected
    expected = 75
    if abs(len(detections) - expected) <= 2:  # Allow small difference
        print(f"\n✅ SUCCESS! Found {len(detections)} vs expected {expected}")
        return True
    else:
        print(f"\n⚠️ UNEXPECTED! Found {len(detections)} vs expected {expected}")
        return False


def test_v2_with_uptrend():
    """Test v2.0.0 with uptrend requirement (should be ~37-38 detections)."""

    print("\n" + "="*70)
    print("TEST 2: Shooting Star v2.0.0 WITH uptrend check")
    print("="*70)
    print("Expected: ~37-38 detections (49.3% of 75 in uptrend)")

    # Load data
    df = load_btc_data()
    print(f"\nTotal candles: {len(df):,}")

    # Create detector WITH uptrend requirement
    detector = ShootingStarPattern(require_uptrend=True, min_uptrend_score=50.0)
    print(f"Detector version: {detector.version}")
    print(f"Require uptrend: {detector.require_uptrend}")
    print(f"Min uptrend score: {detector.min_uptrend_score}")

    # Test each candle
    detections = []
    for i in range(12, len(df)):
        df_slice = df.iloc[:i+1]
        if detector.detect(df_slice):
            detections.append(i)

    print(f"\n✅ Detections found: {len(detections)}")
    print(f"Detection rate: {len(detections)/len(df)*100:.2f}%")

    if len(detections) > 0:
        print(f"\nFirst 5 detections at indices: {detections[:5]}")
        print(f"Last 5 detections at indices: {detections[-5:]}")

    # Expected is approximately 49% of 75 = ~37
    expected_min = 30
    expected_max = 45
    if expected_min <= len(detections) <= expected_max:
        print(f"\n✅ SUCCESS! Found {len(detections)} (expected {expected_min}-{expected_max})")
        return True
    else:
        print(f"\n⚠️ UNEXPECTED! Found {len(detections)} (expected {expected_min}-{expected_max})")
        return False


def test_minimum_candles():
    """Test that detector requires minimum 12 candles."""

    print("\n" + "="*70)
    print("TEST 3: Minimum candles requirement")
    print("="*70)

    df = load_btc_data()
    detector = ShootingStarPattern(require_uptrend=False)

    # Test with 11 candles (should fail)
    df_11 = df.iloc[:11]
    result_11 = detector.detect(df_11)
    print(f"\n11 candles: {result_11} (expected: False)")

    # Test with 12 candles (should work)
    df_12 = df.iloc[:12]
    result_12 = detector.detect(df_12)
    print(f"12 candles: {result_12} (may be True or False depending on data)")

    if result_11 == False:
        print("\n✅ SUCCESS! Correctly requires minimum 12 candles")
        return True
    else:
        print("\n❌ FAILED! Should reject < 12 candles")
        return False


def main():
    """Run all tests."""

    print("\n" + "="*70)
    print("SHOOTING STAR v2.0.0 TEST SUITE")
    print("="*70)
    print("\nTesting new TA-Lib integration with proper 12-candle requirement")

    results = []

    # Test 1
    results.append(test_v2_without_uptrend())

    # Test 2
    results.append(test_v2_with_uptrend())

    # Test 3
    results.append(test_minimum_candles())

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests passed: {sum(results)}/{len(results)}")

    if all(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("\nv2.0.0 is working correctly:")
        print("- Uses TA-Lib properly with 12+ candles ✅")
        print("- Detects ~75 patterns without uptrend check ✅")
        print("- Detects ~37 patterns with uptrend check ✅")
        print("- Requires minimum 12 candles ✅")
    else:
        print("\n⚠️ SOME TESTS FAILED!")
        print("Please review the results above.")


if __name__ == "__main__":
    main()
