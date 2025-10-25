"""
Test all TA-Lib candlestick patterns to determine minimum lookback requirements.

This script systematically tests each TA-Lib candlestick pattern function to determine:
1. Does it work with just 1 candle?
2. If not, what is the minimum number of candles required?
3. What is the detection rate on real BTC data?

Based on findings from Shooting Star and Hammer research:
- Both required minimum 12 candles (11 lookback)
- With 1 candle: 0 detections
- With 12+ candles: proper detections

Author: Research Team
Date: 2025-10-25
"""

import talib
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATTERN DEFINITIONS
# ============================================================================

PATTERNS_TO_TEST = {
    # Already Fixed (v2.0.0)
    'CDLSHOOTINGSTAR': {'name': 'Shooting Star', 'status': '✅ Fixed', 'expected_lookback': 11},
    'CDLHAMMER': {'name': 'Hammer', 'status': '✅ Fixed', 'expected_lookback': 11},

    # Need Testing - Single Candle Patterns
    'CDLINVERTEDHAMMER': {'name': 'Inverted Hammer', 'status': '❓ Unknown'},
    'CDLHANGINGMAN': {'name': 'Hanging Man', 'status': '❓ Unknown'},
    'CDLDOJI': {'name': 'Doji', 'status': '❓ Unknown'},

    # Need Testing - Two Candle Patterns
    'CDLENGULFING': {'name': 'Engulfing', 'status': '❓ Unknown'},
    'CDLPIERCING': {'name': 'Piercing Line', 'status': '❓ Unknown'},
    'CDLDARKCLOUDCOVER': {'name': 'Dark Cloud Cover', 'status': '❓ Unknown'},
    'CDLHARAMI': {'name': 'Harami', 'status': '❓ Unknown'},
    'CDLHARAMICROSS': {'name': 'Harami Cross', 'status': '❓ Unknown'},

    # Need Testing - Three Candle Patterns
    'CDLMORNINGSTAR': {'name': 'Morning Star', 'status': '❓ Unknown'},
    'CDLEVENINGSTAR': {'name': 'Evening Star', 'status': '❓ Unknown'},
    'CDLMORNINGDOJISTAR': {'name': 'Morning Doji Star', 'status': '❓ Unknown'},
    'CDLEVENINGDOJISTAR': {'name': 'Evening Doji Star', 'status': '❓ Unknown'},
    'CDL3WHITESOLDIERS': {'name': '3 White Soldiers', 'status': '❓ Unknown'},
    'CDL3BLACKCROWS': {'name': '3 Black Crows', 'status': '❓ Unknown'},
}

# ============================================================================
# LOAD BTC DATA
# ============================================================================

def load_btc_data():
    """Load BTC hourly data for testing."""
    # Use the historical data path
    csv_path = Path(__file__).parent.parent / 'historical' / 'BTC-USDT' / '1hour.csv'

    if not csv_path.exists():
        print(f"❌ ERROR: {csv_path} not found!")
        print("Please ensure historical/BTC-USDT/1hour.csv exists")
        return None

    df = pd.read_csv(csv_path)
    df = df.astype({
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    })
    print(f"✅ Loaded {len(df)} BTC candles from {csv_path}")
    return df

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_pattern_with_lookback(pattern_func, df, lookback):
    """
    Test a pattern with specific lookback.

    Args:
        pattern_func: TA-Lib pattern function (e.g., talib.CDLHAMMER)
        df: DataFrame with OHLC data
        lookback: Number of previous candles to include (0 = only last candle)

    Returns:
        Number of detections found
    """
    try:
        # Get last N candles (lookback + 1)
        df_subset = df.tail(lookback + 1)

        if len(df_subset) < (lookback + 1):
            return 0

        # Call TA-Lib pattern function
        pattern = pattern_func(
            df_subset['open'].values,
            df_subset['high'].values,
            df_subset['low'].values,
            df_subset['close'].values
        )

        # Count non-zero detections
        detections = np.count_nonzero(pattern)
        return detections

    except Exception as e:
        return 0

def find_minimum_lookback(pattern_func, df, pattern_name):
    """
    Find minimum lookback required for a pattern.

    Tests with increasing lookback until pattern is detected.

    Args:
        pattern_func: TA-Lib pattern function
        df: Full DataFrame with OHLC data
        pattern_name: Name of pattern for display

    Returns:
        dict with test results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {pattern_name}")
    print(f"{'='*70}")

    # First, get total detections with full data
    try:
        pattern_full = pattern_func(
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
        total_detections = np.count_nonzero(pattern_full)
        detection_rate = (total_detections / len(df)) * 100

        print(f"Total detections in full data: {total_detections}/{len(df)} ({detection_rate:.2f}%)")

        if total_detections == 0:
            print(f"⚠️  No detections found in full data - pattern may not occur in this dataset")
            return {
                'pattern_name': pattern_name,
                'total_detections': 0,
                'detection_rate': 0,
                'minimum_lookback': 'N/A',
                'works_with_1_candle': False,
                'status': 'No detections in dataset'
            }

        # Find first detection index
        first_detection_idx = np.argmax(pattern_full != 0)
        print(f"First detection at index: {first_detection_idx}")

    except Exception as e:
        print(f"❌ Error with full data: {e}")
        return {
            'pattern_name': pattern_name,
            'total_detections': 0,
            'detection_rate': 0,
            'minimum_lookback': 'Error',
            'works_with_1_candle': False,
            'status': f'Error: {e}'
        }

    # Now test with increasing lookback values
    print(f"\nTesting minimum lookback requirement:")

    # Use the first detection to test minimum lookback
    # Get data up to and including first detection
    df_test = df.iloc[:first_detection_idx + 1].copy()

    lookback_values = [0, 1, 2, 3, 4, 5, 10, 11, 12, 15, 20, 30, 50]
    minimum_lookback = None

    for lookback in lookback_values:
        if lookback >= len(df_test):
            continue

        detections = test_pattern_with_lookback(pattern_func, df_test, lookback)
        detected = detections > 0

        status_icon = "✅" if detected else "❌"
        print(f"{status_icon} lookback={lookback:2d}: {'DETECTED' if detected else 'NOT detected'} ({lookback + 1} candles)")

        if detected and minimum_lookback is None:
            minimum_lookback = lookback
            print(f"   ⭐ MINIMUM FOUND: {lookback} lookback ({lookback + 1} candles total)")

    # Verify with 10 random detections
    if minimum_lookback is not None and total_detections >= 10:
        print(f"\n🔬 Verifying with 10 random detections...")
        detection_indices = np.where(pattern_full != 0)[0]
        random_indices = np.random.choice(detection_indices, min(10, len(detection_indices)), replace=False)

        lookback_results = []
        for idx in random_indices:
            df_verify = df.iloc[:idx + 1].copy()

            # Test with minimum_lookback
            detections = test_pattern_with_lookback(pattern_func, df_verify, minimum_lookback)
            lookback_results.append(minimum_lookback if detections > 0 else None)

        valid_results = [r for r in lookback_results if r is not None]
        if len(valid_results) == len(lookback_results):
            print(f"✅ All 10 detections confirmed with lookback={minimum_lookback}")
        else:
            print(f"⚠️  Only {len(valid_results)}/10 confirmed - may need higher lookback")

    return {
        'pattern_name': pattern_name,
        'total_detections': int(total_detections),
        'detection_rate': float(detection_rate),
        'minimum_lookback': minimum_lookback if minimum_lookback is not None else 'Unknown',
        'works_with_1_candle': minimum_lookback == 0 if minimum_lookback is not None else False,
        'status': '✅ Works with 1 candle' if minimum_lookback == 0 else f'⚠️ Needs {minimum_lookback + 1} candles'
    }

# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    """Run comprehensive pattern lookback testing."""

    print("="*70)
    print("TA-Lib Pattern Lookback Requirements Test")
    print("="*70)
    print(f"\nTesting {len(PATTERNS_TO_TEST)} patterns...")

    # Load data
    df = load_btc_data()
    if df is None:
        return

    # Test each pattern
    results = []

    for pattern_func_name, pattern_info in PATTERNS_TO_TEST.items():
        pattern_name = pattern_info['name']
        status = pattern_info['status']

        if status == '✅ Fixed':
            print(f"\n{'='*70}")
            print(f"Skipping {pattern_name} - Already Fixed (v2.0.0)")
            print(f"Expected lookback: {pattern_info['expected_lookback']}")
            print(f"{'='*70}")
            results.append({
                'pattern_name': pattern_name,
                'total_detections': 'N/A',
                'detection_rate': 'N/A',
                'minimum_lookback': pattern_info['expected_lookback'],
                'works_with_1_candle': False,
                'status': '✅ Already Fixed'
            })
            continue

        # Get TA-Lib function
        try:
            pattern_func = getattr(talib, pattern_func_name)
        except AttributeError:
            print(f"❌ ERROR: {pattern_func_name} not found in TA-Lib")
            continue

        # Test pattern
        result = find_minimum_lookback(pattern_func, df, pattern_name)
        results.append(result)

    # ============================================================================
    # SUMMARY REPORT
    # ============================================================================

    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    print(f"\n{'Pattern':<25} {'Detections':<12} {'Rate':<10} {'Min Lookback':<15} {'Status'}")
    print("-"*70)

    for result in results:
        pattern = result['pattern_name']
        detections = result['total_detections']
        rate = f"{result['detection_rate']:.2f}%" if isinstance(result['detection_rate'], (int, float)) else 'N/A'
        lookback = result['minimum_lookback']
        status = result['status']

        # Format detections
        det_str = str(detections) if detections != 'N/A' else 'N/A'
        lookback_str = str(lookback) if lookback != 'N/A' else 'N/A'

        print(f"{pattern:<25} {det_str:<12} {rate:<10} {lookback_str:<15} {status}")

    # ============================================================================
    # CATEGORIES
    # ============================================================================

    print("\n" + "="*70)
    print("CATEGORIZATION")
    print("="*70)

    works_with_1 = [r for r in results if r.get('works_with_1_candle') == True]
    needs_more = [r for r in results if r.get('works_with_1_candle') == False and r['status'] not in ['✅ Already Fixed', 'No detections in dataset']]
    already_fixed = [r for r in results if r['status'] == '✅ Already Fixed']
    no_detections = [r for r in results if 'No detections' in r['status']]

    print(f"\n✅ Works with 1 candle ({len(works_with_1)}):")
    for r in works_with_1:
        print(f"   - {r['pattern_name']}")

    print(f"\n⚠️  Needs more than 1 candle ({len(needs_more)}):")
    for r in needs_more:
        lookback = r['minimum_lookback']
        total_candles = lookback + 1 if isinstance(lookback, int) else 'Unknown'
        print(f"   - {r['pattern_name']}: {total_candles} candles (lookback={lookback})")

    print(f"\n✅ Already Fixed ({len(already_fixed)}):")
    for r in already_fixed:
        print(f"   - {r['pattern_name']}")

    print(f"\n🔍 No detections in dataset ({len(no_detections)}):")
    for r in no_detections:
        print(f"   - {r['pattern_name']}")

    # ============================================================================
    # RECOMMENDATIONS
    # ============================================================================

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if len(needs_more) > 0:
        print(f"\n⚠️  {len(needs_more)} patterns need to be fixed:")
        for r in needs_more:
            pattern_file = r['pattern_name'].lower().replace(' ', '_') + '.py'
            lookback = r['minimum_lookback']
            total_candles = lookback + 1 if isinstance(lookback, int) else 'Unknown'
            print(f"\n   📝 {r['pattern_name']} ({pattern_file}):")
            print(f"      - Current: probably passing 1 candle")
            print(f"      - Required: {total_candles} candles (lookback={lookback})")
            print(f"      - Detection rate: {r['detection_rate']:.2f}%")
            print(f"      - Action: Update detect() method to use df.tail(100)")
    else:
        print("\n✅ All tested patterns are either fixed or work with 1 candle!")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
