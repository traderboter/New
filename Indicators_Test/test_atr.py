"""
Test ATR (Average True Range) Indicator

این اسکریپت صحت محاسبه ATR را با روش‌های زیر بررسی می‌کند:
1. مقایسه با TA-Lib (استاندارد طلایی)
2. تست با داده‌های واقعی BTC
3. بررسی موارد مرزی (edge cases)
4. تحلیل دقت محاسبات

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

from signal_generation.analyzers.indicators.atr import ATRIndicator

# =============================================================================
# Constants
# =============================================================================

TOLERANCE = 0.01  # 1% tolerance for comparison
MAX_DIFF_PERCENTAGE = 1.0  # Maximum 1% difference allowed (after warm-up period)

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
    """Compare two series and return statistics

    Args:
        series1: First series to compare
        series2: Second series to compare
        name1: Name of first series
        name2: Name of second series
        tolerance: Tolerance for comparison
        skip_warmup: Number of initial values to skip (warm-up period)
    """
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
    pct_diff = np.abs((s1 - s2) / s2) * 100

    # Statistics
    stats = {
        'count': len(s1),
        'mean_abs_diff': abs_diff.mean(),
        'max_abs_diff': abs_diff.max(),
        'mean_pct_diff': pct_diff.mean(),
        'max_pct_diff': pct_diff.max(),
        'within_tolerance': (pct_diff < tolerance * 100).sum() / len(pct_diff) * 100,
        'success': pct_diff.max() < MAX_DIFF_PERCENTAGE,
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

    print(f"\n📉 Absolute Differences:")
    print(f"   - Mean: {stats['mean_abs_diff']:.6f}")
    print(f"   - Max:  {stats['max_abs_diff']:.6f}")

    print(f"\n📊 Percentage Differences:")
    print(f"   - Mean: {stats['mean_pct_diff']:.4f}%")
    print(f"   - Max:  {stats['max_pct_diff']:.4f}%")
    print(f"   - Within {TOLERANCE*100}% tolerance: {stats['within_tolerance']:.2f}%")

    if stats['success']:
        print(f"\n✅ PASSED: Max difference ({stats['max_pct_diff']:.4f}%) is within acceptable range")
    else:
        print(f"\n❌ FAILED: Max difference ({stats['max_pct_diff']:.4f}%) exceeds {MAX_DIFF_PERCENTAGE}%")

    return stats['success']

# =============================================================================
# Test Functions
# =============================================================================

def test_atr_basic():
    """Test 1: Basic ATR calculation with synthetic data"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Basic ATR Calculation")
    print("="*70)

    # Create test data
    df = create_synthetic_data(100)
    period = 14

    # Calculate with our implementation
    indicator = ATRIndicator({'atr_period': period})
    result = indicator.calculate(df)

    # Calculate with TA-Lib
    talib_atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)

    # Compare (skip first 3*period values for warm-up on synthetic data)
    stats = compare_series(result['atr'], talib_atr,
                          name1="Our ATR", name2="TA-Lib ATR",
                          skip_warmup=period*3)

    success = print_comparison_results(stats, "ATR", "Basic Calculation Test")

    # Show sample values
    print(f"\n📋 Sample Values (last 5 rows):")
    print(f"{'Index':<8} {'Our ATR':<15} {'TA-Lib ATR':<15} {'Diff %':<10}")
    print("-" * 50)
    for i in range(-5, 0):
        our_val = result['atr'].iloc[i]
        talib_val = talib_atr.iloc[i]
        diff_pct = abs((our_val - talib_val) / talib_val * 100) if not pd.isna(talib_val) else 0
        print(f"{len(df)+i:<8} {our_val:<15.6f} {talib_val:<15.6f} {diff_pct:<10.4f}%")

    return success

def test_atr_real_data():
    """Test 2: ATR calculation with real BTC data"""
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
    indicator = ATRIndicator({'atr_period': period})
    result = indicator.calculate(df)

    # Calculate with TA-Lib
    talib_atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)

    # Compare (skip first 2*period values for warm-up)
    stats = compare_series(result['atr'], talib_atr,
                          name1="Our ATR", name2="TA-Lib ATR",
                          skip_warmup=period*2)

    success = print_comparison_results(stats, "ATR", "Real BTC Data Test")

    # Show sample values
    print(f"\n📋 Sample Values (last 5 rows):")
    print(f"{'Index':<8} {'Close':<12} {'Our ATR':<15} {'TA-Lib ATR':<15} {'Diff %':<10}")
    print("-" * 65)
    for i in range(-5, 0):
        close = df['close'].iloc[i]
        our_val = result['atr'].iloc[i]
        talib_val = talib_atr.iloc[i]
        diff_pct = abs((our_val - talib_val) / talib_val * 100) if not pd.isna(talib_val) else 0
        print(f"{len(df)+i:<8} {close:<12.2f} {our_val:<15.6f} {talib_val:<15.6f} {diff_pct:<10.4f}%")

    return success

def test_atr_different_periods():
    """Test 3: ATR with different periods"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Different Periods Test")
    print("="*70)

    # Create more data for larger periods
    df = create_synthetic_data(500)

    periods = [7, 14, 21, 50]
    all_passed = True

    for period in periods:
        print(f"\n--- Testing Period = {period} ---")

        # Calculate with our implementation
        indicator = ATRIndicator({'atr_period': period})
        result = indicator.calculate(df)

        # Calculate with TA-Lib
        talib_atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)

        # Compare (skip warm-up period, more for larger periods)
        warmup = min(period * 4, len(df) // 3)  # Use 4x period or 1/3 of data
        stats = compare_series(result['atr'], talib_atr,
                              name1=f"Our ATR({period})",
                              name2=f"TA-Lib ATR({period})",
                              skip_warmup=warmup)

        if stats['success']:
            print(f"✅ Period {period}: PASSED (max diff: {stats['max_pct_diff']:.4f}%)")
        else:
            print(f"❌ Period {period}: FAILED (max diff: {stats['max_pct_diff']:.4f}%)")
            all_passed = False

    return all_passed

def test_atr_edge_cases():
    """Test 4: Edge cases and boundary conditions"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Edge Cases Test")
    print("="*70)

    all_passed = True

    # Test 1: Minimum data (exactly period + 1 rows)
    print("\n--- Test 4.1: Minimum Data (15 rows for period=14) ---")
    df_min = create_synthetic_data(15)
    indicator = ATRIndicator({'atr_period': 14})
    result = indicator.calculate(df_min)

    if 'atr' in result.columns and not result['atr'].isna().all():
        print("✅ Handles minimum data correctly")
    else:
        print("❌ Failed with minimum data")
        all_passed = False

    # Test 2: Very high volatility
    print("\n--- Test 4.2: High Volatility Data ---")
    df_volatile = create_synthetic_data(100)
    df_volatile['high'] = df_volatile['high'] * 2
    df_volatile['low'] = df_volatile['low'] * 0.5

    indicator = ATRIndicator({'atr_period': 14})
    result = indicator.calculate(df_volatile)
    talib_atr = talib.ATR(df_volatile['high'], df_volatile['low'],
                          df_volatile['close'], timeperiod=14)

    stats = compare_series(result['atr'], talib_atr, skip_warmup=28)
    if stats['success']:
        print(f"✅ Handles high volatility (max diff: {stats['max_pct_diff']:.4f}%)")
    else:
        print(f"❌ Failed with high volatility (max diff: {stats['max_pct_diff']:.4f}%)")
        all_passed = False

    # Test 3: Zero volatility (all same prices)
    print("\n--- Test 4.3: Zero Volatility (all same prices) ---")
    df_flat = pd.DataFrame({
        'open': [100.0] * 50,
        'high': [100.0] * 50,
        'low': [100.0] * 50,
        'close': [100.0] * 50,
        'volume': [1000.0] * 50
    })

    indicator = ATRIndicator({'atr_period': 14})
    result = indicator.calculate(df_flat)

    # ATR should be zero or very close to zero
    atr_values = result['atr'].dropna()
    if len(atr_values) > 0 and atr_values.max() < 0.001:
        print(f"✅ Handles zero volatility correctly (max ATR: {atr_values.max():.6f})")
    else:
        print(f"⚠️  Zero volatility result: max ATR = {atr_values.max():.6f}")

    return all_passed

def test_atr_formula_verification():
    """Test 5: Verify ATR formula step by step"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Formula Verification")
    print("="*70)

    # Create simple test data
    df = create_synthetic_data(50)
    period = 14

    # Manual calculation
    print("\n📐 Manual ATR Calculation:")

    # Step 1: Calculate True Range
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    print(f"✅ Step 1: True Range calculated")
    print(f"   - First non-NaN TR: {true_range.dropna().iloc[0]:.4f}")
    print(f"   - Mean TR: {true_range.mean():.4f}")

    # Step 2: Apply EMA (Wilder's smoothing)
    # TA-Lib uses alpha = 1/period for Wilder's smoothing
    atr_manual = true_range.ewm(alpha=1/period, adjust=False).mean()

    print(f"\n✅ Step 2: EMA applied (alpha=1/{period})")
    print(f"   - Last ATR value: {atr_manual.iloc[-1]:.4f}")

    # Compare with TA-Lib
    talib_atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)

    stats = compare_series(atr_manual, talib_atr,
                          name1="Manual ATR", name2="TA-Lib ATR")

    print(f"\n📊 Manual vs TA-Lib:")
    print(f"   - Mean difference: {stats['mean_pct_diff']:.4f}%")
    print(f"   - Max difference: {stats['max_pct_diff']:.4f}%")

    # Now compare our implementation
    indicator = ATRIndicator({'atr_period': period})
    result = indicator.calculate(df)

    stats2 = compare_series(result['atr'], talib_atr,
                           name1="Our ATR", name2="TA-Lib ATR")

    print(f"\n📊 Our Implementation vs TA-Lib:")
    print(f"   - Mean difference: {stats2['mean_pct_diff']:.4f}%")
    print(f"   - Max difference: {stats2['max_pct_diff']:.4f}%")

    # Check if using alpha gives better results than span
    print(f"\n🔍 Analysis:")
    if stats['max_pct_diff'] < stats2['max_pct_diff']:
        print(f"⚠️  Using alpha=1/period gives better accuracy than span=period")
        print(f"   - Alpha method: {stats['max_pct_diff']:.4f}% max diff")
        print(f"   - Span method: {stats2['max_pct_diff']:.4f}% max diff")
        print(f"\n💡 Recommendation: Update ATR implementation to use alpha=1/period")
        return False
    else:
        print(f"✅ Current implementation is optimal")
        return True

# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 ATR INDICATOR TEST SUITE")
    print("="*70)
    print(f"Testing: signal_generation/analyzers/indicators/atr.py")
    print(f"Date: 2025-10-27")
    print("="*70)

    results = {}

    # Run all tests
    results['test_1_basic'] = test_atr_basic()
    results['test_2_real_data'] = test_atr_real_data()
    results['test_3_periods'] = test_atr_different_periods()
    results['test_4_edge_cases'] = test_atr_edge_cases()
    results['test_5_formula'] = test_atr_formula_verification()

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
