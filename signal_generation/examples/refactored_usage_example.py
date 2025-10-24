"""
مثال استفاده از سیستم refactored شده signal_generation

این فایل نشان می‌دهد چطور از معماری جدید با Orchestrator Pattern استفاده کنیم.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import new components
from signal_generation.analyzers.patterns.pattern_orchestrator import PatternOrchestrator
from signal_generation.analyzers.indicators.indicator_orchestrator import IndicatorOrchestrator

# Import pattern classes
from signal_generation.analyzers.patterns.candlestick import (
    HammerPattern,
    EngulfingPattern,
    MorningStarPattern,
)
from signal_generation.analyzers.patterns.chart import (
    DoubleTopBottomPattern,
    HeadShouldersPattern,
)

# Import indicator classes
from signal_generation.analyzers.indicators import (
    EMAIndicator,
    RSIIndicator,
    MACDIndicator,
    BollingerBandsIndicator,
)


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


def example_1_indicator_orchestrator():
    """
    مثال 1: استفاده از IndicatorOrchestrator

    نشان می‌دهد چطور اندیکاتورها را محاسبه کنیم.
    """
    print("\n" + "=" * 80)
    print("مثال 1: استفاده از IndicatorOrchestrator")
    print("=" * 80)

    # Create sample data
    df = create_sample_data()
    print(f"\n✓ داده‌های نمونه ایجاد شد: {len(df)} کندل")

    # Initialize orchestrator
    config = {
        'ema_periods': [20, 50, 200],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
    }

    orchestrator = IndicatorOrchestrator(config)
    print(f"✓ IndicatorOrchestrator ایجاد شد")

    # Register indicators
    orchestrator.register_indicator(EMAIndicator)
    orchestrator.register_indicator(RSIIndicator)
    orchestrator.register_indicator(MACDIndicator)
    orchestrator.register_indicator(BollingerBandsIndicator)

    print(f"✓ {len(orchestrator.all_indicators)} اندیکاتور ثبت شد")

    # Calculate all indicators
    enriched_df = orchestrator.calculate_all(df)
    print(f"\n✓ تمام اندیکاتورها محاسبه شدند")

    # Show results
    print(f"\nستون‌های جدید اضافه شده:")
    new_columns = [col for col in enriched_df.columns if col not in df.columns]
    for col in new_columns:
        print(f"  - {col}")

    # Show last values
    print(f"\n📊 مقادیر آخرین کندل:")
    if 'ema_20' in enriched_df.columns:
        print(f"  EMA(20): {enriched_df['ema_20'].iloc[-1]:.2f}")
    if 'rsi' in enriched_df.columns:
        print(f"  RSI: {enriched_df['rsi'].iloc[-1]:.2f}")
    if 'macd' in enriched_df.columns:
        print(f"  MACD: {enriched_df['macd'].iloc[-1]:.2f}")

    # Get stats
    stats = orchestrator.get_stats()
    print(f"\n📈 آمار:")
    print(f"  محاسبات انجام شده: {stats['total_calculations']}")
    print(f"  خطاها: {stats['errors']}")

    return enriched_df


def example_2_pattern_orchestrator():
    """
    مثال 2: استفاده از PatternOrchestrator

    نشان می‌دهد چطور الگوها را شناسایی کنیم.
    """
    print("\n" + "=" * 80)
    print("مثال 2: استفاده از PatternOrchestrator")
    print("=" * 80)

    # Create sample data
    df = create_sample_data()
    print(f"\n✓ داده‌های نمونه ایجاد شد: {len(df)} کندل")

    # Initialize orchestrator
    config = {
        'patterns': {
            'candlestick_enabled': True,
            'chart_enabled': True,
            'min_strength': 1
        }
    }

    orchestrator = PatternOrchestrator(config)
    print(f"✓ PatternOrchestrator ایجاد شد")

    # Register patterns
    orchestrator.register_pattern(HammerPattern)
    orchestrator.register_pattern(EngulfingPattern)
    orchestrator.register_pattern(MorningStarPattern)
    orchestrator.register_pattern(DoubleTopBottomPattern)
    orchestrator.register_pattern(HeadShouldersPattern)

    print(f"✓ {len(orchestrator.candlestick_patterns)} الگوی candlestick ثبت شد")
    print(f"✓ {len(orchestrator.chart_patterns)} الگوی chart ثبت شد")

    # Detect patterns
    detected_patterns = orchestrator.detect_all_patterns(
        df=df,
        timeframe='1h',
        context=None
    )

    print(f"\n✓ {len(detected_patterns)} الگو شناسایی شد")

    # Show detected patterns
    if detected_patterns:
        print(f"\n📊 الگوهای شناسایی شده:")
        for pattern in detected_patterns:
            print(f"\n  🔹 {pattern['name']}")
            print(f"     نوع: {pattern['type']}")
            print(f"     جهت: {pattern['direction']}")
            print(f"     قدرت: {pattern['base_strength']:.1f}/3")
            print(f"     اطمینان: {pattern.get('confidence', 0):.2f}")
            if 'metadata' in pattern:
                print(f"     metadata: {list(pattern['metadata'].keys())}")
    else:
        print(f"\n⚠️  هیچ الگویی شناسایی نشد")

    # Get stats
    stats = orchestrator.get_stats()
    print(f"\n📈 آمار:")
    print(f"  کل تشخیص‌ها: {stats['total_detections']}")
    print(f"  Candlestick: {stats['candlestick_detections']}")
    print(f"  Chart: {stats['chart_detections']}")

    return detected_patterns


def example_3_combined_usage():
    """
    مثال 3: استفاده ترکیبی

    نشان می‌دهد چطور اندیکاتورها و الگوها را با هم استفاده کنیم.
    """
    print("\n" + "=" * 80)
    print("مثال 3: استفاده ترکیبی (اندیکاتورها + الگوها)")
    print("=" * 80)

    # Create sample data
    df = create_sample_data(300)
    print(f"\n✓ داده‌های نمونه ایجاد شد: {len(df)} کندل")

    # Step 1: Calculate indicators
    print(f"\n📊 مرحله 1: محاسبه اندیکاتورها")
    indicator_orch = IndicatorOrchestrator({})
    indicator_orch.register_indicator(EMAIndicator)
    indicator_orch.register_indicator(RSIIndicator)
    indicator_orch.register_indicator(MACDIndicator)

    enriched_df = indicator_orch.calculate_all(df)
    print(f"✓ {len(indicator_orch.all_indicators)} اندیکاتور محاسبه شد")

    # Step 2: Detect patterns
    print(f"\n🔍 مرحله 2: شناسایی الگوها")
    pattern_orch = PatternOrchestrator({})
    pattern_orch.register_pattern(HammerPattern)
    pattern_orch.register_pattern(EngulfingPattern)
    pattern_orch.register_pattern(DoubleTopBottomPattern)

    patterns = pattern_orch.detect_all_patterns(
        df=enriched_df,
        timeframe='1h',
        context=None
    )
    print(f"✓ {len(patterns)} الگو شناسایی شد")

    # Step 3: Analyze results
    print(f"\n📈 مرحله 3: تحلیل نتایج")

    # Check RSI
    if 'rsi' in enriched_df.columns:
        rsi_last = enriched_df['rsi'].iloc[-1]
        print(f"\n  RSI: {rsi_last:.2f}")
        if rsi_last > 70:
            print(f"    ⚠️  Overbought (بیش از حد خرید)")
        elif rsi_last < 30:
            print(f"    ⚠️  Oversold (بیش از حد فروش)")
        else:
            print(f"    ✓ در محدوده نرمال")

    # Check MACD
    if 'macd' in enriched_df.columns and 'macd_signal' in enriched_df.columns:
        macd = enriched_df['macd'].iloc[-1]
        signal = enriched_df['macd_signal'].iloc[-1]
        print(f"\n  MACD: {macd:.2f}")
        print(f"  Signal: {signal:.2f}")
        if macd > signal:
            print(f"    ✓ سیگنال صعودی (MACD > Signal)")
        else:
            print(f"    ✓ سیگنال نزولی (MACD < Signal)")

    # Show patterns
    if patterns:
        print(f"\n  الگوهای فعلی:")
        for pattern in patterns[:3]:  # Show first 3
            print(f"    - {pattern['name']} ({pattern['direction']})")

    print(f"\n✅ تحلیل کامل شد!")


def main():
    """اجرای تمام مثال‌ها"""
    print("\n" + "=" * 80)
    print("🎯 نمونه‌های استفاده از سیستم Refactored Signal Generation")
    print("=" * 80)

    try:
        # Run examples
        example_1_indicator_orchestrator()
        example_2_pattern_orchestrator()
        example_3_combined_usage()

        print("\n" + "=" * 80)
        print("✅ تمام مثال‌ها با موفقیت اجرا شدند!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ خطا: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
