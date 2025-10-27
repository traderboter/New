"""
Test SMA (Simple Moving Average) Indicator

این اسکریپت صحت محاسبه SMA را با روش‌های زیر بررسی می‌کند:
1. مقایسه با TA-Lib (استاندارد طلایی)
2. مقایسه با محاسبات دستی
3. تست با داده‌های واقعی BTC
4. بررسی موارد مرزی (edge cases)
5. تست چند period همزمان

نویسنده: Test Team
تاریخ: 2025-10-27
"""

import sys
import pandas as pd
import numpy as np
import talib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_generation.analyzers.indicators.sma import SMAIndicator

# =============================================================================
# Constants
# =============================================================================

TOLERANCE = 0.01  # 1% tolerance for comparison
MAX_DIFF_PERCENTAGE = 0.001  # Maximum 0.001% difference (SMA should be exact!)

# =============================================================================
# Helper Functions
# =============================================================================

def load_btc_data(timeframe="1h", max_rows=1000):
    """Load BTC historical data for testing"""
    timeframe_files = {
        "5m": "5min.csv",
        "15m": "15min.csv",
        "1h": "1hour.csv",
        "4h": "4hour.csv"
    }

    csv_path = Path(__file__).parent.parent / 'historical' / 'BTC-USDT' / timeframe_files[timeframe]

    if not csv_path.exists():
        print(f"⚠️  Warning: {csv_path} not found. Using synthetic data.")
        return None

    df = pd.read_csv(csv_path)
    df = df.astype({
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    })

    # Limit rows for faster testing
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    return df

def create_synthetic_data(n_rows=100):
    """Create synthetic OHLC data for testing"""
    np.random.seed(42)

    # Generate price data with some volatility
    base_price = 50000
    changes = np.random.randn(n_rows) * 500
    close_prices = base_price + np.cumsum(changes)

    # Generate OHLC with proper relationships
    data = []
    for i, close in enumerate(close_prices):
        high = close + abs(np.random.randn() * 200)
        low = close - abs(np.random.randn() * 200)
        open_price = low + (high - low) * np.random.rand()

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.rand() * 1000000
        })

    return pd.DataFrame(data)

def compare_series(series1, series2, name1="Series 1", name2="Series 2", tolerance=TOLERANCE):
    """Compare two series and return statistics"""
    # Remove NaN values
    mask = ~(pd.isna(series1) | pd.isna(series2))
    s1 = series1[mask]
    s2 = series2[mask]

    if len(s1) == 0 or len(s2) == 0:
        return {
            'success': False,
            'error': 'No valid values to compare',
            'count': 0
        }

    # Calculate differences
    abs_diff = np.abs(s1 - s2)
    pct_diff = np.abs((s1 - s2) / s2) * 100

    # Statistics
    stats = {
        'count': len(s1),
        'mean_abs_diff': abs_diff.mean(),
        'max_abs_diff': abs_diff.max(),
        'mean_pct_diff': pct_diff.mean(),
        'max_pct_diff': pct_diff.max(),
        'within_tolerance': (pct_diff < tolerance * 100).sum() / len(pct_diff) * 100,
        'success': pct_diff.max() < MAX_DIFF_PERCENTAGE
    }

    return stats

def print_comparison_results(stats, indicator_name, test_name):
    """Print comparison statistics"""
    print(f"\n{'='*70}")
    print(f"📊 {test_name}")
    print(f"{'='*70}")

    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
        return False

    print(f"📈 Number of compared values: {stats['count']}")

    print(f"\n📉 Absolute Differences:")
    print(f"   - Mean: {stats['mean_abs_diff']:.10f}")
    print(f"   - Max:  {stats['max_abs_diff']:.10f}")

    print(f"\n📊 Percentage Differences:")
    print(f"   - Mean: {stats['mean_pct_diff']:.6f}%")
    print(f"   - Max:  {stats['max_pct_diff']:.6f}%")
    print(f"   - Within {TOLERANCE*100}% tolerance: {stats['within_tolerance']:.2f}%")

    if stats['success']:
        print(f"\n✅ PASSED: Max difference ({stats['max_pct_diff']:.6f}%) is within acceptable range")
    else:
        print(f"\n❌ FAILED: Max difference ({stats['max_pct_diff']:.6f}%) exceeds {MAX_DIFF_PERCENTAGE}%")

    return stats['success']

# =============================================================================
# Test Functions
# =============================================================================

def test_sma_basic():
    """Test 1: Basic SMA calculation with synthetic data"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Basic SMA Calculation")
    print("="*70)

    # Create test data
    df = create_synthetic_data(300)
    period = 20

    # Calculate with our implementation
    indicator = SMAIndicator({'sma_periods': [period]})
    result = indicator.calculate(df)

    # Calculate with TA-Lib
    talib_sma = talib.SMA(df['close'], timeperiod=period)

    # Compare
    stats = compare_series(result[f'sma_{period}'], talib_sma,
                          name1="Our SMA", name2="TA-Lib SMA")

    success = print_comparison_results(stats, "SMA", "Basic Calculation Test")

    # Show sample values
    print(f"\n📋 Sample Values (last 5 rows):")
    print(f"{'Index':<8} {'Close':<15} {'Our SMA':<15} {'TA-Lib SMA':<15} {'Diff':<15}")
    print("-" * 70)
    for i in range(-5, 0):
        close = df['close'].iloc[i]
        our_val = result[f'sma_{period}'].iloc[i]
        talib_val = talib_sma.iloc[i]
        diff = abs(our_val - talib_val)
        print(f"{len(df)+i:<8} {close:<15.6f} {our_val:<15.6f} {talib_val:<15.6f} {diff:<15.10f}")

    return success

def test_sma_manual_calculation():
    """Test 2: Manual calculation verification"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Manual Calculation Verification")
    print("="*70)

    # Create simple test data with known values
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    df = pd.DataFrame({'close': prices})
    period = 5

    print(f"Test data: {prices}")
    print(f"Period: {period}")

    # Calculate with our implementation
    indicator = SMAIndicator({'sma_periods': [period]})
    result = indicator.calculate(df)

    # Manual calculation
    print(f"\n📐 Manual Calculations:")
    manual_smas = []
    for i in range(len(prices)):
        if i >= period - 1:
            window = prices[i - period + 1: i + 1]
            manual_sma = sum(window) / period
            manual_smas.append(manual_sma)
            print(f"   Index {i}: SMA of {window} = {manual_sma:.2f}")
        else:
            manual_smas.append(np.nan)
            print(f"   Index {i}: Not enough data (NaN)")

    # Compare
    our_smas = result[f'sma_{period}'].values

    print(f"\n📊 Comparison:")
    all_match = True
    for i in range(period - 1, len(prices)):
        our_val = our_smas[i]
        manual_val = manual_smas[i]
        match = abs(our_val - manual_val) < 0.0001
        status = "✅" if match else "❌"
        print(f"   {status} Index {i}: Our={our_val:.6f}, Manual={manual_val:.6f}, Diff={abs(our_val - manual_val):.10f}")
        if not match:
            all_match = False

    if all_match:
        print(f"\n✅ PASSED: All manual calculations match!")
    else:
        print(f"\n❌ FAILED: Some calculations don't match!")

    return all_match

def test_sma_real_data():
    """Test 3: SMA calculation with real BTC data"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Real BTC Data Test")
    print("="*70)

    # Try to load real data
    df = load_btc_data("1h", max_rows=500)

    if df is None:
        print("⚠️  Skipping: Real data not available")
        return True

    print(f"✅ Loaded {len(df)} BTC candles (1h timeframe)")

    periods = [20, 50, 200]
    all_passed = True

    # Calculate with our implementation
    indicator = SMAIndicator({'sma_periods': periods})
    result = indicator.calculate(df)

    for period in periods:
        print(f"\n--- Testing Period = {period} ---")

        # Calculate with TA-Lib
        talib_sma = talib.SMA(df['close'], timeperiod=period)

        # Compare
        stats = compare_series(result[f'sma_{period}'], talib_sma,
                              name1=f"Our SMA({period})",
                              name2=f"TA-Lib SMA({period})")

        if stats['success']:
            print(f"✅ Period {period}: PASSED (max diff: {stats['max_pct_diff']:.8f}%)")
        else:
            print(f"❌ Period {period}: FAILED (max diff: {stats['max_pct_diff']:.8f}%)")
            all_passed = False

        # Show last 3 values
        print(f"\n   Last 3 values:")
        for i in range(-3, 0):
            our_val = result[f'sma_{period}'].iloc[i]
            talib_val = talib_sma.iloc[i]
            diff = abs(our_val - talib_val)
            print(f"   Index {len(df)+i}: Our={our_val:.6f}, TA-Lib={talib_val:.6f}, Diff={diff:.10f}")

    return all_passed

def test_sma_edge_cases():
    """Test 4: Edge cases and boundary conditions"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Edge Cases Test")
    print("="*70)

    all_passed = True

    # Test 1: Minimum data (exactly period rows)
    print("\n--- Test 4.1: Minimum Data (20 rows for period=20) ---")
    df_min = create_synthetic_data(20)
    indicator = SMAIndicator({'sma_periods': [20]})
    result = indicator.calculate(df_min)

    # Last value should be calculated, all others should be NaN
    non_nan = result['sma_20'].notna().sum()
    if non_nan == 1:
        print(f"✅ Correct: Only 1 non-NaN value (last one)")
    else:
        print(f"❌ Failed: Expected 1 non-NaN value, got {non_nan}")
        all_passed = False

    # Test 2: Flat data (all same prices)
    print("\n--- Test 4.2: Flat Data (all same prices) ---")
    df_flat = pd.DataFrame({
        'close': [100.0] * 50
    })

    indicator = SMAIndicator({'sma_periods': [10]})
    result = indicator.calculate(df_flat)

    # All SMA values should be exactly 100.0
    sma_values = result['sma_10'].dropna()
    if (sma_values == 100.0).all():
        print(f"✅ Correct: All SMA values are 100.0")
    else:
        print(f"❌ Failed: Expected all 100.0, got range [{sma_values.min():.6f}, {sma_values.max():.6f}]")
        all_passed = False

    # Test 3: Linear trend
    print("\n--- Test 4.3: Linear Trend ---")
    df_linear = pd.DataFrame({
        'close': list(range(1, 101))  # 1, 2, 3, ..., 100
    })

    indicator = SMAIndicator({'sma_periods': [10]})
    result = indicator.calculate(df_linear)

    # For linear data, SMA should be exactly in the middle of the window
    # e.g., SMA of [1,2,3,4,5,6,7,8,9,10] = 5.5
    expected_sma_at_10 = 5.5
    actual_sma_at_10 = result['sma_10'].iloc[9]

    if abs(actual_sma_at_10 - expected_sma_at_10) < 0.0001:
        print(f"✅ Correct: SMA at index 9 = {actual_sma_at_10:.6f} (expected {expected_sma_at_10})")
    else:
        print(f"❌ Failed: SMA at index 9 = {actual_sma_at_10:.6f}, expected {expected_sma_at_10}")
        all_passed = False

    # Test 4: Multiple periods simultaneously
    print("\n--- Test 4.4: Multiple Periods Simultaneously ---")
    df = create_synthetic_data(300)
    periods = [5, 10, 20, 50, 100]

    indicator = SMAIndicator({'sma_periods': periods})
    result = indicator.calculate(df)

    # Check that all SMA columns exist
    missing_cols = []
    for period in periods:
        col_name = f'sma_{period}'
        if col_name not in result.columns:
            missing_cols.append(col_name)

    if len(missing_cols) == 0:
        print(f"✅ All {len(periods)} SMA columns exist")
    else:
        print(f"❌ Missing columns: {missing_cols}")
        all_passed = False

    # Compare with TA-Lib for all periods
    all_match = True
    for period in periods:
        talib_sma = talib.SMA(df['close'], timeperiod=period)
        our_sma = result[f'sma_{period}']

        # Check if they're identical
        mask = ~(pd.isna(our_sma) | pd.isna(talib_sma))
        max_diff = (our_sma[mask] - talib_sma[mask]).abs().max()

        if max_diff < 0.0001:
            print(f"   ✅ Period {period}: Max diff = {max_diff:.10f}")
        else:
            print(f"   ❌ Period {period}: Max diff = {max_diff:.10f}")
            all_match = False

    if not all_match:
        all_passed = False

    return all_passed

def test_sma_precision():
    """Test 5: Numerical precision test"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Numerical Precision Test")
    print("="*70)

    # Create data with very large values
    df_large = pd.DataFrame({
        'close': [1000000.0 + i * 0.01 for i in range(100)]
    })

    indicator = SMAIndicator({'sma_periods': [20]})
    result = indicator.calculate(df_large)

    talib_sma = talib.SMA(df_large['close'], timeperiod=20)

    # Compare
    stats = compare_series(result['sma_20'], talib_sma,
                          name1="Our SMA", name2="TA-Lib SMA")

    print(f"\n📊 Large Values Test:")
    print(f"   - Mean absolute difference: {stats['mean_abs_diff']:.15f}")
    print(f"   - Max absolute difference: {stats['max_abs_diff']:.15f}")

    # Create data with very small values
    df_small = pd.DataFrame({
        'close': [0.00001 + i * 0.000001 for i in range(100)]
    })

    indicator = SMAIndicator({'sma_periods': [20]})
    result = indicator.calculate(df_small)

    talib_sma = talib.SMA(df_small['close'], timeperiod=20)

    # Compare
    stats2 = compare_series(result['sma_20'], talib_sma,
                           name1="Our SMA", name2="TA-Lib SMA")

    print(f"\n📊 Small Values Test:")
    print(f"   - Mean absolute difference: {stats2['mean_abs_diff']:.15f}")
    print(f"   - Max absolute difference: {stats2['max_abs_diff']:.15f}")

    success = stats['success'] and stats2['success']

    if success:
        print(f"\n✅ PASSED: Precision is acceptable for both large and small values")
    else:
        print(f"\n❌ FAILED: Precision issues detected")

    return success

# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 SMA INDICATOR TEST SUITE")
    print("="*70)
    print(f"Testing: signal_generation/analyzers/indicators/sma.py")
    print(f"Date: 2025-10-27")
    print("="*70)

    results = {}

    # Run all tests
    results['test_1_basic'] = test_sma_basic()
    results['test_2_manual'] = test_sma_manual_calculation()
    results['test_3_real_data'] = test_sma_real_data()
    results['test_4_edge_cases'] = test_sma_edge_cases()
    results['test_5_precision'] = test_sma_precision()

    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<25} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print("\n" + "="*70)
    print(f"Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {total_tests - passed_tests} test(s) failed")

    print("="*70)

    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
