"""
Debug version of test_pattern.py for Shooting Star

این اسکریپت دقیقاً مثل test_pattern.py کار می‌کند
اما logging بیشتری دارد تا ببینیم چه اتفاقی می‌افتد.
"""

import pandas as pd
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from signal_generation.analyzers.patterns.pattern_orchestrator import PatternOrchestrator
from signal_generation.analyzers.patterns.candlestick.shooting_star import ShootingStarPattern


def main():
    print("\n" + "="*80)
    print("🐛 DEBUG: test_pattern.py workflow for Shooting Star")
    print("="*80 + "\n")

    # Load data (exactly like test_pattern.py)
    data_file = 'historical/BTC-USDT/1hour.csv'
    df = pd.read_csv(data_file)

    # Ensure lowercase columns
    df.columns = [col.lower() for col in df.columns]

    print(f"✓ Loaded {len(df)} candles\n")

    # Create orchestrator (exactly like test_pattern.py)
    orchestrator = PatternOrchestrator({})

    # Get pattern class
    pattern_class = ShootingStarPattern

    print(f"Pattern class: {pattern_class}")
    print(f"Pattern class name: {pattern_class.__name__}\n")

    # Register pattern (exactly like test_pattern.py)
    orchestrator.register_pattern(pattern_class)

    print(f"✓ Registered pattern")
    print(f"  Candlestick patterns: {list(orchestrator.candlestick_patterns.keys())}\n")

    # Check the registered pattern instance
    if 'Shooting Star' in orchestrator.candlestick_patterns:
        pattern_instance = orchestrator.candlestick_patterns['Shooting Star']
        print(f"Registered pattern instance:")
        print(f"  Name: {pattern_instance.name}")
        print(f"  Version: {pattern_instance.version}")
        print(f"  require_uptrend: {pattern_instance.require_uptrend}")
        print(f"  min_uptrend_score: {pattern_instance.min_uptrend_score}\n")

    # Scan (exactly like test_pattern.py)
    pattern_name = 'shooting_star'
    min_window = 50
    total_candles = len(df)
    target_detections = []

    print(f"Scanning with test_pattern.py logic...")
    print(f"  Pattern name for filtering: '{pattern_name}'\n")

    # Test first 1000 only
    test_limit = min(1000, total_candles)

    for i in range(min_window, test_limit):
        # Window (exactly like test_pattern.py)
        window_df = df.iloc[:i+1].copy()

        # Detect (exactly like test_pattern.py)
        detections = orchestrator.detect_all_patterns(
            df=window_df,
            timeframe='1h',
            context={}
        )

        # Filter (exactly like test_pattern.py)
        for d in detections:
            detection_name_lower = d['name'].lower()

            if len(target_detections) < 3:  # First 3 only
                print(f"  Index {i}: Detection '{d['name']}' (lower: '{detection_name_lower}')")
                print(f"    Check: '{pattern_name}' in '{detection_name_lower}' = {pattern_name in detection_name_lower}")

            if pattern_name in detection_name_lower:
                d['detected_at_index'] = i
                target_detections.append(d)

                if len(target_detections) <= 3:
                    print(f"    ✓ ADDED to target_detections (total: {len(target_detections)})")

    print(f"\n" + "="*80)
    print(f"RESULTS")
    print(f"="*80)
    print(f"\nTotal detections found: {len(target_detections)}")

    if target_detections:
        print(f"\nFirst 3 detections:")
        for i, d in enumerate(target_detections[:3], 1):
            print(f"  #{i}. Index {d['detected_at_index']}: {d['name']}")
    else:
        print(f"\n❌ NO DETECTIONS FOUND!")
        print(f"\nDEBUGGING INFO:")
        print(f"  Pattern name used for filtering: '{pattern_name}'")
        print(f"  Registered patterns: {list(orchestrator.candlestick_patterns.keys())}")

        # Try one detection manually
        print(f"\n  Testing index 19 manually (where we know there's a detection):")
        window_df = df.iloc[:20].copy()
        detections = orchestrator.detect_all_patterns(
            df=window_df,
            timeframe='1h',
            context={}
        )
        print(f"    Detections: {len(detections)}")
        if detections:
            for d in detections:
                print(f"      - {d['name']} (lower: '{d['name'].lower()}')")
                print(f"        '{pattern_name}' in '{d['name'].lower()}' = {pattern_name in d['name'].lower()}")

    print()


if __name__ == '__main__':
    main()
