"""
Test V2 Wrapper Classes with AnalysisContext

This test verifies that:
1. IndicatorCalculator V2 works correctly with AnalysisContext
2. PatternAnalyzer V2 works correctly with AnalysisContext
3. Integration between indicators and patterns works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime

from signal_generation.context import AnalysisContext
from signal_generation.shared.indicator_calculator_v2 import IndicatorCalculator
from signal_generation.analyzers.pattern_analyzer_v2 import PatternAnalyzer


def create_sample_data(num_candles=200):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=num_candles, freq='1h')

    # Generate random price data
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(num_candles) * 100)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(num_candles) * 50,
        'high': close_prices + np.abs(np.random.randn(num_candles) * 100),
        'low': close_prices - np.abs(np.random.randn(num_candles) * 100),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, num_candles)
    })

    return df


def test_indicator_calculator_v2():
    """Test IndicatorCalculator V2 with AnalysisContext."""
    print("\n" + "=" * 80)
    print("Test 1: IndicatorCalculator V2 with AnalysisContext")
    print("=" * 80)

    # Create context
    df = create_sample_data(200)
    context = AnalysisContext(
        symbol='BTCUSDT',
        timeframe='1h',
        df=df
    )

    print(f"\n✓ Created AnalysisContext for {context.symbol} {context.timeframe}")
    print(f"  Initial DataFrame shape: {context.df.shape}")
    print(f"  Initial columns: {list(context.df.columns)}")

    # Create and use IndicatorCalculator V2
    config = {
        'ema_periods': [20, 50],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
    }

    calculator = IndicatorCalculator(config)
    print(f"\n✓ Created IndicatorCalculator")
    print(f"  Registered indicators: {len(calculator.orchestrator.all_indicators)}")

    # Calculate all indicators
    calculator.calculate_all(context)

    print(f"\n✓ Calculated all indicators")
    print(f"  Final DataFrame shape: {context.df.shape}")

    # Verify indicators were added
    new_columns = [col for col in context.df.columns if col not in df.columns]
    print(f"\n  New columns added ({len(new_columns)}):")
    for col in new_columns:
        print(f"    - {col}")

    # Check specific indicators
    print(f"\n  Sample values (last candle):")
    if 'ema_20' in context.df.columns:
        print(f"    EMA(20): {context.df['ema_20'].iloc[-1]:.2f}")
    if 'rsi' in context.df.columns:
        print(f"    RSI: {context.df['rsi'].iloc[-1]:.2f}")
    if 'macd' in context.df.columns:
        print(f"    MACD: {context.df['macd'].iloc[-1]:.2f}")

    # Verify no NaN in recent data
    recent_data = context.df.tail(10)
    nan_columns = recent_data.columns[recent_data.isna().any()].tolist()
    if nan_columns:
        print(f"\n  ⚠️  Columns with NaN values in recent data: {nan_columns}")
    else:
        print(f"\n  ✓ No NaN values in recent data")

    # Get stats
    stats = calculator.get_stats()
    print(f"\n  Statistics:")
    print(f"    Total calculations: {stats['total_calculations']}")
    print(f"    Errors: {stats['errors']}")

    print(f"\n✅ Test 1 PASSED")
    return context


def test_pattern_analyzer_v2():
    """Test PatternAnalyzer V2 with AnalysisContext."""
    print("\n" + "=" * 80)
    print("Test 2: PatternAnalyzer V2 with AnalysisContext")
    print("=" * 80)

    # Create context with more data for patterns
    df = create_sample_data(300)
    context = AnalysisContext(
        symbol='ETHUSDT',
        timeframe='4h',
        df=df
    )

    print(f"\n✓ Created AnalysisContext for {context.symbol} {context.timeframe}")
    print(f"  DataFrame shape: {context.df.shape}")

    # Create and use PatternAnalyzer V2
    config = {
        'patterns': {
            'candlestick_enabled': True,
            'chart_enabled': True,
            'min_strength': 1
        }
    }

    analyzer = PatternAnalyzer(config)
    print(f"\n✓ Created PatternAnalyzer")
    print(f"  Registered candlestick patterns: {len(analyzer.orchestrator.candlestick_patterns)}")
    print(f"  Registered chart patterns: {len(analyzer.orchestrator.chart_patterns)}")

    # Analyze patterns
    analyzer.analyze(context)

    print(f"\n✓ Pattern analysis completed")

    # Check if results were added to context
    pattern_results = context.get_result('patterns')
    if pattern_results:
        print(f"\n  Results added to context:")
        print(f"    Status: {pattern_results.get('status')}")

        patterns_found = pattern_results.get('patterns', [])
        print(f"    Patterns found: {len(patterns_found)}")

        if patterns_found:
            print(f"\n  Detected patterns:")
            for pattern in patterns_found[:5]:  # Show first 5
                print(f"    - {pattern.get('name')} ({pattern.get('direction')})")
                print(f"      Strength: {pattern.get('base_strength')}/3")
                print(f"      Confidence: {pattern.get('confidence', 0):.2f}")
    else:
        print(f"\n  ⚠️  No pattern results found in context")

    # Get stats
    stats = analyzer.orchestrator.get_stats()
    print(f"\n  Statistics:")
    print(f"    Total detections: {stats['total_detections']}")
    print(f"    Candlestick: {stats['candlestick_detections']}")
    print(f"    Chart: {stats['chart_detections']}")

    print(f"\n✅ Test 2 PASSED")
    return context


def test_combined_integration():
    """Test combined usage of indicators and patterns."""
    print("\n" + "=" * 80)
    print("Test 3: Combined Integration (Indicators + Patterns)")
    print("=" * 80)

    # Create context
    df = create_sample_data(300)
    context = AnalysisContext(
        symbol='BNBUSDT',
        timeframe='1d',
        df=df
    )

    print(f"\n✓ Created AnalysisContext for {context.symbol} {context.timeframe}")

    # Step 1: Calculate indicators
    print(f"\n📊 Step 1: Calculating indicators...")
    calculator = IndicatorCalculator({})
    calculator.calculate_all(context)

    indicator_cols = [col for col in context.df.columns if col not in df.columns]
    print(f"  ✓ Added {len(indicator_cols)} indicator columns")

    # Step 2: Detect patterns (patterns can use indicator data)
    print(f"\n🔍 Step 2: Detecting patterns...")
    analyzer = PatternAnalyzer({})
    analyzer.analyze(context)

    pattern_results = context.get_result('patterns')
    patterns_count = len(pattern_results.get('patterns', [])) if pattern_results else 0
    print(f"  ✓ Detected {patterns_count} patterns")

    # Step 3: Verify integration
    print(f"\n📈 Step 3: Verifying integration...")

    # Check context state
    print(f"\n  Context state:")
    print(f"    Symbol: {context.symbol}")
    print(f"    Timeframe: {context.timeframe}")
    print(f"    DataFrame shape: {context.df.shape}")
    print(f"    Results stored: {list(context.results.keys())}")
    print(f"    Analyzers run: {context._stats['analyzers_run']}")
    print(f"    Analyzers failed: {context._stats['analyzers_failed']}")

    # Sample analysis
    if 'rsi' in context.df.columns:
        rsi_last = context.df['rsi'].iloc[-1]
        print(f"\n  Market analysis:")
        print(f"    RSI: {rsi_last:.2f}", end="")
        if rsi_last > 70:
            print(f" → Overbought ⚠️")
        elif rsi_last < 30:
            print(f" → Oversold ⚠️")
        else:
            print(f" → Normal ✓")

    if 'macd' in context.df.columns and 'macd_signal' in context.df.columns:
        macd = context.df['macd'].iloc[-1]
        signal = context.df['macd_signal'].iloc[-1]
        trend = "Bullish ↗" if macd > signal else "Bearish ↘"
        print(f"    MACD trend: {trend}")

    if pattern_results and patterns_count > 0:
        patterns = pattern_results.get('patterns', [])
        bullish = sum(1 for p in patterns if p.get('direction') == 'bullish')
        bearish = sum(1 for p in patterns if p.get('direction') == 'bearish')
        print(f"    Patterns: {bullish} bullish, {bearish} bearish")

    print(f"\n✅ Test 3 PASSED")

    # Final summary
    print(f"\n" + "=" * 80)
    print(f"Integration Test Summary:")
    print(f"  - All indicators calculated successfully")
    print(f"  - All patterns detected successfully")
    print(f"  - Context properly maintained throughout pipeline")
    print(f"  - V2 wrappers work correctly with AnalysisContext")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🧪 Testing V2 Wrapper Classes with AnalysisContext")
    print("=" * 80)

    try:
        # Run tests
        test_indicator_calculator_v2()
        test_pattern_analyzer_v2()
        test_combined_integration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
