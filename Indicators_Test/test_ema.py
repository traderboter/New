"""
Test EMA (Exponential Moving Average) Indicator

این اسکریپت صحت محاسبه EMA را با روش‌های زیر بررسی می‌کند:
1. مقایسه با TA-Lib (استاندارد طلایی)
2. مقایسه با محاسبات دستی
3. تست با داده‌های واقعی BTC
4. بررسی موارد مرزی (edge cases)
5. مقایسه با SMA (تفاوت EMA و SMA)

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

from signal_generation.analyzers.indicators.ema import EMAIndicator

# =============================================================================
# Constants
# =============================================================================

TOLERANCE = 0.01  # 1% tolerance for comparison
MAX_DIFF_PERCENTAGE = 0.01  # Maximum 0.01% difference allowed

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

def test_ema_basic():
    """Test 1: Basic EMA calculation with synthetic data"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Basic EMA Calculation")
    print("="*70)

    # Create test data
    df = create_synthetic_data(300)
    period = 20

    # Calculate with our implementation
    indicator = EMAIndicator({'ema_periods': [period]})
    result = indicator.calculate(df)

    # Calculate with TA-Lib
    talib_ema = talib.EMA(df['close'], timeperiod=period)

    # Compare (skip first few for warm-up)
    stats = compare_series(result[f'ema_{period}'], talib_ema,
                          name1="Our EMA", name2="TA-Lib EMA",
                          skip_warmup=period*2)

    success = print_comparison_results(stats, "EMA", "Basic Calculation Test")

    # Show sample values
    print(f"\n📋 Sample Values (last 5 rows):")
    print(f"{'Index':<8} {'Close':<15} {'Our EMA':<15} {'TA-Lib EMA':<15} {'Diff %':<12}")
    print("-" * 75)
    for i in range(-5, 0):
        close = df['close'].iloc[i]
        our_val = result[f'ema_{period}'].iloc[i]
        talib_val = talib_ema.iloc[i]
        diff_pct = abs((our_val - talib_val) / talib_val * 100) if not pd.isna(talib_val) else 0
        print(f"{len(df)+i:<8} {close:<15.6f} {our_val:<15.6f} {talib_val:<15.6f} {diff_pct:<12.8f}%")

    return success

def test_ema_formula_verification():
    """Test 2: Verify EMA formula step by step"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Formula Verification")
    print("="*70)

    # Create simple test data
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    df = pd.DataFrame({'close': prices})
    period = 5

    print(f"Test data: {prices}")
    print(f"Period: {period}")

    # EMA formula:
    # alpha = 2 / (period + 1)
    # EMA[i] = price[i] * alpha + EMA[i-1] * (1 - alpha)
    # First EMA = SMA of first period values

    alpha = 2 / (period + 1)
    print(f"\n📐 Alpha = 2/(period+1) = 2/({period}+1) = {alpha:.6f}")

    # Manual calculation
    print(f"\n📐 Manual EMA Calculation:")
    manual_emas = []

    for i in range(len(prices)):
        if i < period - 1:
            manual_emas.append(np.nan)
            print(f"   Index {i}: Not enough data (NaN)")
        elif i == period - 1:
            # First EMA is SMA
            ema = sum(prices[0:period]) / period
            manual_emas.append(ema)
            print(f"   Index {i}: First EMA (SMA) = {ema:.6f}")
        else:
            # EMA = price * alpha + prev_EMA * (1-alpha)
            ema = prices[i] * alpha + manual_emas[i-1] * (1 - alpha)
            manual_emas.append(ema)
            print(f"   Index {i}: EMA = {prices[i]:.2f}*{alpha:.6f} + {manual_emas[i-1]:.6f}*(1-{alpha:.6f}) = {ema:.6f}")

    # Calculate with our implementation
    indicator = EMAIndicator({'ema_periods': [period]})
    result = indicator.calculate(df)

    # Calculate with TA-Lib
    talib_ema = talib.EMA(df['close'], timeperiod=period)

    # Compare all three
    print(f"\n📊 Comparison:")
    print(f"{'Index':<8} {'Price':<12} {'Manual':<15} {'Our EMA':<15} {'TA-Lib':<15}")
    print("-" * 70)

    all_match = True
    for i in range(len(prices)):
        manual_val = manual_emas[i]
        our_val = result[f'ema_{period}'].iloc[i]
        talib_val = talib_ema.iloc[i]

        manual_str = f"{manual_val:.6f}" if not pd.isna(manual_val) else "NaN"
        our_str = f"{our_val:.6f}" if not pd.isna(our_val) else "NaN"
        talib_str = f"{talib_val:.6f}" if not pd.isna(talib_val) else "NaN"

        print(f"{i:<8} {prices[i]:<12.2f} {manual_str:<15} {our_str:<15} {talib_str:<15}")

        if not pd.isna(manual_val) and not pd.isna(our_val):
            if abs(manual_val - our_val) > 0.0001:
                all_match = False

    if all_match:
        print(f"\n✅ PASSED: Manual calculations match our implementation!")
    else:
        print(f"\n❌ FAILED: Some calculations don't match!")

    return all_match

def test_ema_real_data():
    """Test 3: EMA calculation with real BTC data"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Real BTC Data Test")
    print("="*70)

    # Try to load real data
    df = load_btc_data("1h", max_rows=500)

    if df is None:
        print("⚠️  Skipping: Real data not available")
        return True

    print(f"✅ Loaded {len(df)} BTC candles (1h timeframe)")

    periods = [12, 26, 50, 200]
    all_passed = True

    # Calculate with our implementation
    indicator = EMAIndicator({'ema_periods': periods})
    result = indicator.calculate(df)

    for period in periods:
        print(f"\n--- Testing Period = {period} ---")

        # Calculate with TA-Lib
        talib_ema = talib.EMA(df['close'], timeperiod=period)

        # Compare (skip warm-up)
        stats = compare_series(result[f'ema_{period}'], talib_ema,
                              name1=f"Our EMA({period})",
                              name2=f"TA-Lib EMA({period})",
                              skip_warmup=period*2)

        if stats['success']:
            print(f"✅ Period {period}: PASSED (max diff: {stats['max_pct_diff']:.8f}%)")
        else:
            print(f"❌ Period {period}: FAILED (max diff: {stats['max_pct_diff']:.8f}%)")
            all_passed = False

        # Show last 3 values
        print(f"\n   Last 3 values:")
        for i in range(-3, 0):
            close = df['close'].iloc[i]
            our_val = result[f'ema_{period}'].iloc[i]
            talib_val = talib_ema.iloc[i]
            diff_pct = abs((our_val - talib_val) / talib_val * 100)
            print(f"   Index {len(df)+i}: Close={close:.2f}, Our={our_val:.6f}, TA-Lib={talib_val:.6f}, Diff={diff_pct:.8f}%")

    return all_passed

def test_ema_vs_sma():
    """Test 4: Compare EMA behavior vs SMA"""
    print("\n" + "="*70)
    print("🧪 TEST 4: EMA vs SMA Comparison")
    print("="*70)

    # Create data with a trend change
    # First 50: uptrend, Next 50: downtrend
    prices = list(range(1, 51)) + list(range(50, 0, -1))
    df = pd.DataFrame({'close': prices})
    period = 10

    print(f"Test with {len(prices)} prices (uptrend then downtrend)")

    # Calculate EMA
    from signal_generation.analyzers.indicators.ema import EMAIndicator
    ema_indicator = EMAIndicator({'ema_periods': [period]})
    ema_result = ema_indicator.calculate(df)

    # Calculate SMA
    from signal_generation.analyzers.indicators.sma import SMAIndicator
    sma_indicator = SMAIndicator({'sma_periods': [period]})
    sma_result = sma_indicator.calculate(df)

    # At the trend change point (index 50), EMA should react faster than SMA
    change_idx = 50
    window_after_change = 10

    print(f"\n📊 Response to trend change at index {change_idx}:")
    print(f"{'Index':<8} {'Price':<12} {'EMA':<15} {'SMA':<15} {'Diff':<12}")
    print("-" * 65)

    for i in range(change_idx, change_idx + window_after_change):
        price = prices[i]
        ema_val = ema_result[f'ema_{period}'].iloc[i]
        sma_val = sma_result[f'sma_{period}'].iloc[i]
        diff = ema_val - sma_val

        print(f"{i:<8} {price:<12} {ema_val:<15.6f} {sma_val:<15.6f} {diff:<12.6f}")

    # EMA should be more responsive (lower values after downtrend starts)
    ema_at_change_plus_5 = ema_result[f'ema_{period}'].iloc[change_idx + 5]
    sma_at_change_plus_5 = sma_result[f'sma_{period}'].iloc[change_idx + 5]

    print(f"\n📈 Analysis:")
    print(f"   5 periods after trend change:")
    print(f"   - EMA: {ema_at_change_plus_5:.6f}")
    print(f"   - SMA: {sma_at_change_plus_5:.6f}")

    if ema_at_change_plus_5 < sma_at_change_plus_5:
        print(f"   ✅ EMA reacts faster (lower value = quicker response to downtrend)")
        return True
    else:
        print(f"   ❌ Expected EMA to react faster than SMA")
        return False

def test_ema_edge_cases():
    """Test 5: Edge cases and boundary conditions"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Edge Cases Test")
    print("="*70)

    all_passed = True

    # Test 1: Flat data
    print("\n--- Test 5.1: Flat Data (all same prices) ---")
    df_flat = pd.DataFrame({
        'close': [100.0] * 50
    })

    indicator = EMAIndicator({'ema_periods': [10]})
    result = indicator.calculate(df_flat)

    # EMA of flat data should converge to that value
    ema_values = result['ema_10'].dropna()
    last_ema = ema_values.iloc[-1]

    if abs(last_ema - 100.0) < 0.001:
        print(f"✅ Correct: EMA converges to 100.0 (got {last_ema:.6f})")
    else:
        print(f"❌ Failed: Expected ~100.0, got {last_ema:.6f}")
        all_passed = False

    # Test 2: Multiple periods simultaneously
    print("\n--- Test 5.2: Multiple Periods Simultaneously ---")
    df = create_synthetic_data(300)
    periods = [5, 10, 20, 50, 100, 200]

    indicator = EMAIndicator({'ema_periods': periods})
    result = indicator.calculate(df)

    # Check that all EMA columns exist
    missing_cols = []
    for period in periods:
        col_name = f'ema_{period}'
        if col_name not in result.columns:
            missing_cols.append(col_name)

    if len(missing_cols) == 0:
        print(f"✅ All {len(periods)} EMA columns exist")

        # Verify with TA-Lib for each
        all_match = True
        for period in periods:
            talib_ema = talib.EMA(df['close'], timeperiod=period)
            our_ema = result[f'ema_{period}']

            stats = compare_series(our_ema, talib_ema, skip_warmup=period*2)

            if stats['success']:
                print(f"   ✅ Period {period}: Max diff = {stats['max_pct_diff']:.6f}%")
            else:
                print(f"   ❌ Period {period}: Max diff = {stats['max_pct_diff']:.6f}%")
                all_match = False

        if not all_match:
            all_passed = False
    else:
        print(f"❌ Missing columns: {missing_cols}")
        all_passed = False

    # Test 3: Very volatile data
    print("\n--- Test 5.3: High Volatility Data ---")
    df_volatile = create_synthetic_data(200)
    df_volatile['close'] = df_volatile['close'] * (1 + np.random.randn(len(df_volatile)) * 0.1)

    indicator = EMAIndicator({'ema_periods': [20]})
    result = indicator.calculate(df_volatile)
    talib_ema = talib.EMA(df_volatile['close'], timeperiod=20)

    stats = compare_series(result['ema_20'], talib_ema, skip_warmup=40)
    if stats['success']:
        print(f"✅ Handles high volatility (max diff: {stats['max_pct_diff']:.6f}%)")
    else:
        print(f"❌ Failed with high volatility (max diff: {stats['max_pct_diff']:.6f}%)")
        all_passed = False

    return all_passed

# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 EMA INDICATOR TEST SUITE")
    print("="*70)
    print(f"Testing: signal_generation/analyzers/indicators/ema.py")
    print(f"Date: 2025-10-27")
    print("="*70)

    results = {}

    # Run all tests
    results['test_1_basic'] = test_ema_basic()
    results['test_2_formula'] = test_ema_formula_verification()
    results['test_3_real_data'] = test_ema_real_data()
    results['test_4_vs_sma'] = test_ema_vs_sma()
    results['test_5_edge_cases'] = test_ema_edge_cases()

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
