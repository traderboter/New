"""
تست ساده برای Shooting Star با PatternOrchestrator

این اسکریپت مستقیماً PatternOrchestrator را روی داده‌های واقعی تست می‌کند
تا ببیند آیا orchestrator درست کار می‌کند یا نه.
"""

import pandas as pd
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from signal_generation.analyzers.patterns.pattern_orchestrator import PatternOrchestrator
from signal_generation.analyzers.patterns.candlestick.shooting_star import ShootingStarPattern


def main():
    print("\n" + "="*80)
    print("🔬 SIMPLE TEST: ShootingStarPattern with PatternOrchestrator")
    print("="*80 + "\n")

    # Load data
    data_file = 'historical/BTC-USDT/1hour.csv'

    if not os.path.exists(data_file):
        print(f"❌ File not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    df.columns = [col.lower() for col in df.columns]

    print(f"✓ Loaded {len(df)} candles from {data_file}\n")

    # Test 1: Direct detection (without orchestrator)
    print("="*80)
    print("TEST 1: Direct ShootingStarPattern.detect()")
    print("="*80 + "\n")

    detector = ShootingStarPattern(require_uptrend=True, min_uptrend_score=50.0)

    print(f"Detector settings:")
    print(f"  require_uptrend: {detector.require_uptrend}")
    print(f"  min_uptrend_score: {detector.min_uptrend_score}\n")

    direct_detections = 0

    print("Scanning...")
    for i in range(10, min(1000, len(df))):  # Test first 1000 only
        df_window = df.iloc[:i+1]
        if detector.detect(df_window):
            direct_detections += 1
            if direct_detections <= 3:
                print(f"  ✓ Detection at index {i}")

    print(f"\nDirect detections (first 1000): {direct_detections}\n")

    # Test 2: Through PatternOrchestrator
    print("="*80)
    print("TEST 2: Through PatternOrchestrator.detect_all_patterns()")
    print("="*80 + "\n")

    orchestrator = PatternOrchestrator({})

    # Register pattern
    orchestrator.register_pattern(ShootingStarPattern)

    print(f"✓ Registered ShootingStarPattern")
    print(f"  Candlestick patterns: {list(orchestrator.candlestick_patterns.keys())}\n")

    orchestrator_detections = 0

    print("Scanning...")
    for i in range(10, min(1000, len(df))):  # Test first 1000 only
        df_window = df.iloc[:i+1]
        detections = orchestrator.detect_all_patterns(
            df=df_window,
            timeframe='1h',
            context={}
        )

        if detections:
            orchestrator_detections += 1
            if orchestrator_detections <= 3:
                print(f"  ✓ Detection at index {i}: {detections[0]['name']}")

    print(f"\nOrchestrator detections (first 1000): {orchestrator_detections}\n")

    # Test 3: Check specific index from analyze_real_data.py results
    # از نتایج analyze_real_data.py می‌دانیم که 50 detection داریم
    # بیایید یکی از آنها را پیدا کنیم
    print("="*80)
    print("TEST 3: Finding first detection in full dataset")
    print("="*80 + "\n")

    print("Scanning full dataset...")
    first_detection_index = None

    for i in range(10, len(df)):
        df_window = df.iloc[:i+1]
        if detector.detect(df_window):
            first_detection_index = i
            print(f"  ✓ First detection found at index: {i}")

            # Show candle details
            candle = df.iloc[i]
            print(f"\n  Candle details:")
            print(f"    Timestamp: {candle.get('timestamp', 'N/A')}")
            print(f"    Open: {candle['open']:.2f}")
            print(f"    High: {candle['high']:.2f}")
            print(f"    Low: {candle['low']:.2f}")
            print(f"    Close: {candle['close']:.2f}")

            # Calculate metrics
            o = candle['open']
            h = candle['high']
            l = candle['low']
            c = candle['close']

            body_size = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            full_range = h - l

            if full_range > 0:
                upper_shadow_pct = upper_shadow / full_range
                lower_shadow_pct = lower_shadow / full_range
                body_pct = body_size / full_range

                print(f"\n  Metrics:")
                print(f"    Upper shadow: {upper_shadow_pct*100:.1f}% (need >= 50%)")
                print(f"    Lower shadow: {lower_shadow_pct*100:.1f}% (need <= 20%)")
                print(f"    Body size: {body_pct*100:.1f}% (need <= 30%)")

            # Check uptrend
            context_score = detector._analyze_context(df_window)
            print(f"    Context score: {context_score:.1f} (need >= 50.0)")

            break

        # Show progress every 1000
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(df)}")

    if first_detection_index is None:
        print("  ❌ No detection found in full dataset!")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDirect detections (first 1000):      {direct_detections}")
    print(f"Orchestrator detections (first 1000): {orchestrator_detections}")
    print(f"First detection in full dataset:      {'Found at index ' + str(first_detection_index) if first_detection_index else 'Not found'}")

    if direct_detections > 0 and orchestrator_detections == 0:
        print("\n⚠️  ISSUE: Direct detection works, but orchestrator doesn't!")
        print("   There may be a bug in PatternOrchestrator.detect_all_patterns()")
    elif direct_detections == 0:
        print("\n⚠️  ISSUE: No detections even with direct method!")
        print("   Check ShootingStarPattern.detect() logic")
    elif direct_detections == orchestrator_detections:
        print("\n✅ Both methods work correctly!")
    else:
        print("\n⚠️  Different results between direct and orchestrator!")

    print()


if __name__ == '__main__':
    main()
