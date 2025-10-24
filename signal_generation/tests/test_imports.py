"""
Simple test to verify all imports work correctly.
This doesn't require external dependencies.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")
print("=" * 80)

# Test base classes
try:
    from signal_generation.analyzers.patterns.base_pattern import BasePattern
    print("✓ BasePattern imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BasePattern: {e}")

try:
    from signal_generation.analyzers.indicators.base_indicator import BaseIndicator
    print("✓ BaseIndicator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BaseIndicator: {e}")

# Test orchestrators
try:
    from signal_generation.analyzers.patterns.pattern_orchestrator import PatternOrchestrator
    print("✓ PatternOrchestrator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PatternOrchestrator: {e}")

try:
    from signal_generation.analyzers.indicators.indicator_orchestrator import IndicatorOrchestrator
    print("✓ IndicatorOrchestrator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import IndicatorOrchestrator: {e}")

# Test candlestick patterns
print("\nCandlestick Patterns:")
candlestick_patterns = [
    'hammer', 'inverted_hammer', 'engulfing', 'morning_star',
    'piercing_line', 'three_white_soldiers', 'morning_doji_star',
    'shooting_star', 'hanging_man', 'evening_star',
    'dark_cloud_cover', 'three_black_crows', 'evening_doji_star',
    'doji', 'harami', 'harami_cross'
]

for pattern in candlestick_patterns:
    try:
        module = __import__(
            f'signal_generation.analyzers.patterns.candlestick.{pattern}',
            fromlist=['']
        )
        print(f"  ✓ {pattern}")
    except ImportError as e:
        print(f"  ✗ {pattern}: {e}")

# Test chart patterns
print("\nChart Patterns:")
chart_patterns = ['double_top_bottom', 'head_shoulders', 'triangle', 'wedge']

for pattern in chart_patterns:
    try:
        module = __import__(
            f'signal_generation.analyzers.patterns.chart.{pattern}',
            fromlist=['']
        )
        print(f"  ✓ {pattern}")
    except ImportError as e:
        print(f"  ✗ {pattern}: {e}")

# Test indicators
print("\nIndicators:")
indicators = ['ema', 'sma', 'rsi', 'macd', 'atr', 'bollinger_bands', 'stochastic', 'obv']

for indicator in indicators:
    try:
        module = __import__(
            f'signal_generation.analyzers.indicators.{indicator}',
            fromlist=['']
        )
        print(f"  ✓ {indicator}")
    except ImportError as e:
        print(f"  ✗ {indicator}: {e}")

# Test V2 wrappers
print("\nV2 Wrappers:")
try:
    from signal_generation.analyzers.pattern_analyzer_v2 import PatternAnalyzer
    print("✓ PatternAnalyzer V2 imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PatternAnalyzer V2: {e}")

try:
    from signal_generation.shared.indicator_calculator_v2 import IndicatorCalculator
    print("✓ IndicatorCalculator V2 imported successfully")
except ImportError as e:
    print(f"✗ Failed to import IndicatorCalculator V2: {e}")

print("\n" + "=" * 80)
print("Import test completed!")
