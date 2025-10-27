"""
Test RSI (Relative Strength Index) Indicator

این اسکریپت صحت محاسبه RSI را با روش‌های زیر بررسی می‌کند:
1. مقایسه با TA-Lib (استاندارد طلایی)
2. تست با داده‌های واقعی BTC
3. بررسی موارد مرزی (edge cases)
4. تست محدوده RSI (0-100)
5. تست overbought/oversold

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

from signal_generation.analyzers.indicators.rsi import RSIIndicator

# =============================================================================
# Constants
# =============================================================================

TOLERANCE = 0.01  # 1% tolerance for comparison
MAX_DIFF_PERCENTAGE = 1.0  # Maximum 1% difference allowed (after warm-up)

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

def compare_series(series1, series2, name1="Series 1", name2="Series 2", tolerance=TOLERANCE, skip_warmup=0):
    """Compare two series and return statistics"""
    # Remove NaN values
    mask = ~(pd.isna(series1) | pd.isna(series2))
    s1 = series1[mask]
    s2 = series2[mask]

    # Skip warm-up period if specified
    if skip_warmup > 0 and len(s1) > skip_warmup:
        s1 = s1.iloc[skip_warmup:]
        s2 = s2.iloc[skip_warmup:]

    if len(s1) == 0 or len(s2) == 0:
        return {
            'success': False,
            'error': 'No valid values to compare',
            'count': 0
        }

    # Calculate differences
    abs_diff = np.abs(s1 - s2)
    # For RSI, use absolute difference not percentage (since RSI is 0-100)

    # Statistics
    stats = {
        'count': len(s1),
        'mean_abs_diff': abs_diff.mean(),
        'max_abs_diff': abs_diff.max(),
        'within_tolerance': (abs_diff < 1.0).sum() / len(abs_diff) * 100,  # Within 1 point
        'success': abs_diff.max() < 1.0,  # Max 1 point difference
        'skipped_warmup': skip_warmup
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
    if stats.get('skipped_warmup', 0) > 0:
        print(f"   (Skipped first {stats['skipped_warmup']} values for warm-up)")

    print(f"\n📉 Absolute Differences (RSI points):")
    print(f"   - Mean: {stats['mean_abs_diff']:.6f}")
    print(f"   - Max:  {stats['max_abs_diff']:.6f}")

    print(f"\n📊 Statistics:")
    print(f"   - Within 1.0 point tolerance: {stats['within_tolerance']:.2f}%")

    if stats['success']:
        print(f"\n✅ PASSED: Max difference ({stats['max_abs_diff']:.6f} points) is within acceptable range")
    else:
        print(f"\n❌ FAILED: Max difference ({stats['max_abs_diff']:.6f} points) exceeds 1.0 point")

    return stats['success']

# =============================================================================
# Test Functions
# =============================================================================

def test_rsi_basic():
    """Test 1: Basic RSI calculation with synthetic data"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Basic RSI Calculation")
    print("="*70)

    # Create test data
    df = create_synthetic_data(300)
    period = 14

    # Calculate with our implementation
    indicator = RSIIndicator({'rsi_period': period})
    result = indicator.calculate(df)

    # Calculate with TA-Lib
    talib_rsi = talib.RSI(df['close'], timeperiod=period)

    # Compare (skip first 2*period for warm-up)
    stats = compare_series(result['rsi'], talib_rsi,
                          name1="Our RSI", name2="TA-Lib RSI",
                          skip_warmup=period*3)

    success = print_comparison_results(stats, "RSI", "Basic Calculation Test")

    # Show sample values
    print(f"\n📋 Sample Values (last 5 rows):")
    print(f"{'Index':<8} {'Close':<15} {'Our RSI':<15} {'TA-Lib RSI':<15} {'Diff':<12}")
    print("-" * 70)
    for i in range(-5, 0):
        close = df['close'].iloc[i]
        our_val = result['rsi'].iloc[i]
        talib_val = talib_rsi.iloc[i]
        diff = abs(our_val - talib_val) if not (pd.isna(our_val) or pd.isna(talib_val)) else 0
        print(f"{len(df)+i:<8} {close:<15.2f} {our_val:<15.6f} {talib_val:<15.6f} {diff:<12.6f}")

    return success

def test_rsi_real_data():
    """Test 2: RSI calculation with real BTC data"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Real BTC Data Test")
    print("="*70)

    # Try to load real data
    df = load_btc_data("1h", max_rows=500)

    if df is None:
        print("⚠️  Skipping: Real data not available")
        return True

    print(f"✅ Loaded {len(df)} BTC candles (1h timeframe)")

    period = 14

    # Calculate with our implementation
    indicator = RSIIndicator({'rsi_period': period})
    result = indicator.calculate(df)

    # Calculate with TA-Lib
    talib_rsi = talib.RSI(df['close'], timeperiod=period)

    # Compare (skip warm-up)
    stats = compare_series(result['rsi'], talib_rsi,
                          name1="Our RSI", name2="TA-Lib RSI",
                          skip_warmup=period*3)

    success = print_comparison_results(stats, "RSI", "Real BTC Data Test")

    # Show sample values
    print(f"\n📋 Sample Values (last 5 rows):")
    print(f"{'Index':<8} {'Close':<12} {'Our RSI':<15} {'TA-Lib RSI':<15} {'Diff':<12}")
    print("-" * 65)
    for i in range(-5, 0):
        close = df['close'].iloc[i]
        our_val = result['rsi'].iloc[i]
        talib_val = talib_rsi.iloc[i]
        diff = abs(our_val - talib_val)
        print(f"{len(df)+i:<8} {close:<12.2f} {our_val:<15.6f} {talib_val:<15.6f} {diff:<12.6f}")

    return success

def test_rsi_different_periods():
    """Test 3: RSI with different periods"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Different Periods Test")
    print("="*70)

    df = create_synthetic_data(500)

    periods = [7, 14, 21, 28]
    all_passed = True

    for period in periods:
        print(f"\n--- Testing Period = {period} ---")

        # Calculate with our implementation
        indicator = RSIIndicator({'rsi_period': period})
        result = indicator.calculate(df)

        # Calculate with TA-Lib
        talib_rsi = talib.RSI(df['close'], timeperiod=period)

        # Compare (skip warm-up)
        stats = compare_series(result['rsi'], talib_rsi,
                              name1=f"Our RSI({period})",
                              name2=f"TA-Lib RSI({period})",
                              skip_warmup=period*3)

        if stats['success']:
            print(f"✅ Period {period}: PASSED (max diff: {stats['max_abs_diff']:.6f} points)")
        else:
            print(f"❌ Period {period}: FAILED (max diff: {stats['max_abs_diff']:.6f} points)")
            all_passed = False

    return all_passed

def test_rsi_range():
    """Test 4: RSI range validation (0-100)"""
    print("\n" + "="*70)
    print("🧪 TEST 4: RSI Range Test (0-100)")
    print("="*70)

    all_passed = True

    # Test 1: Very bullish data (should approach 100)
    print("\n--- Test 4.1: Very Bullish Data ---")
    bullish_prices = [100 + i*10 for i in range(100)]  # Strong uptrend
    df_bullish = pd.DataFrame({'close': bullish_prices})

    indicator = RSIIndicator({'rsi_period': 14})
    result = indicator.calculate(df_bullish)

    rsi_values = result['rsi'].dropna()
    max_rsi = rsi_values.max()
    min_rsi = rsi_values.min()

    print(f"RSI Range: [{min_rsi:.2f}, {max_rsi:.2f}]")

    if 0 <= min_rsi <= 100 and 0 <= max_rsi <= 100:
        print(f"✅ RSI within valid range [0, 100]")
        if max_rsi > 70:
            print(f"✅ RSI correctly shows overbought condition (>{max_rsi:.2f})")
    else:
        print(f"❌ RSI outside valid range!")
        all_passed = False

    # Test 2: Very bearish data (should approach 0)
    print("\n--- Test 4.2: Very Bearish Data ---")
    bearish_prices = [1000 - i*10 for i in range(100)]  # Strong downtrend
    df_bearish = pd.DataFrame({'close': bearish_prices})

    indicator = RSIIndicator({'rsi_period': 14})
    result = indicator.calculate(df_bearish)

    rsi_values = result['rsi'].dropna()
    max_rsi = rsi_values.max()
    min_rsi = rsi_values.min()

    print(f"RSI Range: [{min_rsi:.2f}, {max_rsi:.2f}]")

    if 0 <= min_rsi <= 100 and 0 <= max_rsi <= 100:
        print(f"✅ RSI within valid range [0, 100]")
        if min_rsi < 30:
            print(f"✅ RSI correctly shows oversold condition (<{min_rsi:.2f})")
    else:
        print(f"❌ RSI outside valid range!")
        all_passed = False

    # Test 3: Flat data (should be around 50)
    print("\n--- Test 4.3: Flat Data ---")
    df_flat = pd.DataFrame({'close': [100.0] * 100})

    indicator = RSIIndicator({'rsi_period': 14})
    result = indicator.calculate(df_flat)

    rsi_values = result['rsi'].dropna()
    # For flat data, RSI should be NaN or undefined (no gains/losses)
    # TA-Lib returns NaN for this case

    if len(rsi_values) == 0 or rsi_values.isna().all():
        print(f"✅ RSI correctly returns NaN for flat data (no price changes)")
    else:
        print(f"⚠️  RSI values for flat data: {rsi_values.iloc[-1]:.2f}")

    return all_passed

def test_rsi_edge_cases():
    """Test 5: Edge cases"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Edge Cases Test")
    print("="*70)

    all_passed = True

    # Test 1: Minimum data
    print("\n--- Test 5.1: Minimum Data ---")
    df_min = create_synthetic_data(20)
    indicator = RSIIndicator({'rsi_period': 14})
    result = indicator.calculate(df_min)

    rsi_values = result['rsi'].dropna()
    if len(rsi_values) > 0:
        print(f"✅ RSI calculated with minimum data ({len(rsi_values)} values)")
    else:
        print(f"❌ RSI failed with minimum data")
        all_passed = False

    # Test 2: High volatility
    print("\n--- Test 5.2: High Volatility Data ---")
    df_volatile = create_synthetic_data(300)
    df_volatile['close'] = df_volatile['close'] * (1 + np.random.randn(len(df_volatile)) * 0.2)

    indicator = RSIIndicator({'rsi_period': 14})
    result = indicator.calculate(df_volatile)
    talib_rsi = talib.RSI(df_volatile['close'], timeperiod=14)

    stats = compare_series(result['rsi'], talib_rsi, skip_warmup=42)
    if stats['success']:
        print(f"✅ Handles high volatility (max diff: {stats['max_abs_diff']:.6f})")
    else:
        print(f"❌ Failed with high volatility (max diff: {stats['max_abs_diff']:.6f})")
        all_passed = False

    return all_passed

# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 RSI INDICATOR TEST SUITE")
    print("="*70)
    print(f"Testing: signal_generation/analyzers/indicators/rsi.py")
    print(f"Date: 2025-10-27")
    print("="*70)

    results = {}

    # Run all tests
    results['test_1_basic'] = test_rsi_basic()
    results['test_2_real_data'] = test_rsi_real_data()
    results['test_3_periods'] = test_rsi_different_periods()
    results['test_4_range'] = test_rsi_range()
    results['test_5_edge_cases'] = test_rsi_edge_cases()

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
