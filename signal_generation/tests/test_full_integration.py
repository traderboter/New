"""
تست کامل Integration - بررسی عملکرد کل سیستم Signal Generation

این تست شبیه‌سازی می‌کند که آیا همه اجزا با هم به درستی کار می‌کنند:
1. IndicatorCalculator (نسخه جدید)
2. PatternAnalyzer (نسخه جدید)
3. سایر Analyzers
4. AnalysisContext
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components
from signal_generation.context import AnalysisContext
from signal_generation.shared.indicator_calculator import IndicatorCalculator
from signal_generation.analyzers import (
    PatternAnalyzer,
    TrendAnalyzer,
    MomentumAnalyzer,
    VolumeAnalyzer,
)


def create_realistic_ohlcv_data(num_candles=300, base_price=50000):
    """Create realistic OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=num_candles, freq='1h')

    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, num_candles)  # 2% volatility
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * (1 + np.random.uniform(-0.005, 0.005, num_candles)),
        'high': close_prices * (1 + np.abs(np.random.uniform(0, 0.01, num_candles))),
        'low': close_prices * (1 - np.abs(np.random.uniform(0, 0.01, num_candles))),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, num_candles).astype(float)
    })

    return df


def test_full_pipeline():
    """Test complete signal generation pipeline."""
    print("\n" + "=" * 80)
    print("🧪 Full Integration Test - Signal Generation Pipeline")
    print("=" * 80)

    # Create config
    config = {
        'analyzers': {
            'pattern': {'enabled': True},
            'trend': {'enabled': True},
            'momentum': {'enabled': True},
            'volume': {'enabled': True},
        },
        'indicators': {
            'ema_periods': [20, 50],
            'sma_periods': [20, 50, 200],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
        }
    }

    # Step 1: Create context
    print("\n📊 Step 1: Creating AnalysisContext...")
    df = create_realistic_ohlcv_data(300)
    context = AnalysisContext(
        symbol='BTCUSDT',
        timeframe='1h',
        df=df
    )
    print(f"  ✓ Created context for {context.symbol} {context.timeframe}")
    print(f"  ✓ DataFrame shape: {context.df.shape}")

    # Step 2: Calculate indicators
    print("\n📈 Step 2: Calculating indicators...")
    indicator_calc = IndicatorCalculator(config)

    initial_columns = set(context.df.columns)
    indicator_calc.calculate_all(context)
    new_columns = set(context.df.columns) - initial_columns

    print(f"  ✓ Indicator calculation completed")
    print(f"  ✓ Added {len(new_columns)} new columns")
    print(f"  ✓ New columns: {', '.join(sorted(new_columns))}")

    # Verify key indicators
    required_indicators = ['ema_20', 'sma_20', 'rsi', 'macd']
    missing = [ind for ind in required_indicators if ind not in context.df.columns]
    if missing:
        print(f"  ❌ Missing indicators: {missing}")
        return False
    else:
        print(f"  ✓ All required indicators present")

    # Step 3: Run analyzers
    print("\n🔍 Step 3: Running analyzers...")

    # 3a. Trend Analyzer
    print("\n  3a. Trend Analyzer...")
    try:
        trend_analyzer = TrendAnalyzer(config)
        trend_analyzer.analyze(context)
        trend_result = context.get_result('trend')
        if trend_result and trend_result.get('status') == 'ok':
            print(f"     ✓ Trend: {trend_result.get('direction', 'unknown')}")
            print(f"     ✓ Strength: {trend_result.get('strength', 0):.2f}")
        else:
            print(f"     ⚠️  Trend analysis incomplete")
    except Exception as e:
        print(f"     ❌ Trend analyzer error: {e}")
        return False

    # 3b. Momentum Analyzer
    print("\n  3b. Momentum Analyzer...")
    try:
        momentum_analyzer = MomentumAnalyzer(config)
        momentum_analyzer.analyze(context)
        momentum_result = context.get_result('momentum')
        if momentum_result and momentum_result.get('status') == 'ok':
            print(f"     ✓ Momentum: {momentum_result.get('direction', 'unknown')}")
            print(f"     ✓ RSI: {momentum_result.get('rsi', 0):.2f}")
        else:
            print(f"     ⚠️  Momentum analysis incomplete")
    except Exception as e:
        print(f"     ❌ Momentum analyzer error: {e}")
        return False

    # 3c. Volume Analyzer
    print("\n  3c. Volume Analyzer...")
    try:
        volume_analyzer = VolumeAnalyzer(config)
        volume_analyzer.analyze(context)
        volume_result = context.get_result('volume')
        if volume_result and volume_result.get('status') == 'ok':
            print(f"     ✓ Volume analysis completed")
            print(f"     ✓ Confirmed: {volume_result.get('is_confirmed', False)}")
        else:
            print(f"     ⚠️  Volume analysis incomplete")
    except Exception as e:
        print(f"     ❌ Volume analyzer error: {e}")
        return False

    # 3d. Pattern Analyzer (NEW - با orchestrator)
    print("\n  3d. Pattern Analyzer (NEW with Orchestrator)...")
    try:
        pattern_analyzer = PatternAnalyzer(config)
        print(f"     ✓ Registered {len(pattern_analyzer.orchestrator.candlestick_patterns)} candlestick patterns")
        print(f"     ✓ Registered {len(pattern_analyzer.orchestrator.chart_patterns)} chart patterns")

        pattern_analyzer.analyze(context)
        pattern_result = context.get_result('patterns')

        if pattern_result and pattern_result.get('status') == 'ok':
            candlestick = pattern_result.get('candlestick_patterns', [])
            chart = pattern_result.get('chart_patterns', [])
            total = pattern_result.get('total_patterns', 0)

            print(f"     ✓ Pattern detection completed")
            print(f"     ✓ Found {total} patterns ({len(candlestick)} candlestick, {len(chart)} chart)")

            if pattern_result.get('strongest_pattern'):
                strongest = pattern_result['strongest_pattern']
                print(f"     ✓ Strongest: {strongest.get('name')} ({strongest.get('direction')})")
                print(f"     ✓ Strength: {strongest.get('adjusted_strength', 0):.2f}")

            print(f"     ✓ Overall confidence: {pattern_result.get('confidence', 0):.2f}")
            print(f"     ✓ Trend aligned: {pattern_result.get('alignment_with_trend', False)}")

            # Check orchestrator stats
            if 'orchestrator_stats' in pattern_result:
                stats = pattern_result['orchestrator_stats']
                print(f"     ✓ Orchestrator stats: {stats}")
        else:
            print(f"     ⚠️  Pattern analysis incomplete")
    except Exception as e:
        print(f"     ❌ Pattern analyzer error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Summary
    print("\n" + "=" * 80)
    print("📋 Integration Test Summary")
    print("=" * 80)

    print(f"\n  Context State:")
    print(f"    Symbol: {context.symbol}")
    print(f"    Timeframe: {context.timeframe}")
    print(f"    DataFrame shape: {context.df.shape}")
    print(f"    Results stored: {list(context.results.keys())}")
    print(f"    Analyzers run: {context._stats['analyzers_run']}")
    print(f"    Analyzers failed: {context._stats['analyzers_failed']}")

    print(f"\n  Indicators:")
    print(f"    Total columns: {len(context.df.columns)}")
    print(f"    Indicator columns: {len(new_columns)}")

    print(f"\n  Analysis Results:")
    for analyzer_name, result in context.results.items():
        status = result.get('status', 'unknown')
        print(f"    {analyzer_name}: {status}")

    # Final check
    required_results = ['trend', 'momentum', 'volume', 'patterns']
    missing_results = [r for r in required_results if r not in context.results]

    if missing_results:
        print(f"\n  ❌ Missing results: {missing_results}")
        return False

    print("\n" + "=" * 80)
    print("✅ Integration Test PASSED!")
    print("=" * 80)
    print("\n  سیستم Signal Generation به درستی کار می‌کند:")
    print("  ✓ IndicatorCalculator جدید با Orchestrator")
    print("  ✓ PatternAnalyzer جدید با Orchestrator")
    print("  ✓ تمام Analyzers")
    print("  ✓ AnalysisContext")
    print("  ✓ Integration کامل")

    return True


def main():
    """Run integration test."""
    try:
        success = test_full_pipeline()

        if success:
            print("\n✅ همه تست‌ها موفق بود!")
            return 0
        else:
            print("\n❌ تست ناموفق بود!")
            return 1

    except Exception as e:
        print(f"\n❌ خطای غیرمنتظره: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
