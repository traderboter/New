"""
Test Recency Scoring Implementation

تست سیستم امتیازدهی بر اساس تازگی الگو
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from signal_generation.analyzers.patterns.candlestick.hammer import HammerPattern

def test_recency_scoring():
    """Test Hammer pattern with recency scoring"""

    print("="*70)
    print("🧪 Test: Recency Scoring System")
    print("="*70)

    # Load config
    config_path = Path(__file__).parent / 'signal_generation' / 'analyzers' / 'patterns' / 'pattern_config.json'

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\n✅ Loaded config from: {config_path}")
    print(f"   Hammer lookback_window: {config['patterns']['hammer']['lookback_window']}")
    print(f"   Hammer recency_multipliers: {config['patterns']['hammer']['recency_multipliers']}")

    # Load BTC data
    csv_path = Path(__file__).parent / 'historical' / 'BTC-USDT' / '1hour.csv'

    if not csv_path.exists():
        print(f"❌ Data file not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded {len(df)} BTC candles from: {csv_path}")

    # Create Hammer detector
    detector = HammerPattern(config=config)

    print(f"\n📋 Detector info:")
    print(f"   Version: {detector.version}")
    print(f"   Lookback window: {detector.lookback_window}")
    print(f"   Recency multipliers: {detector.recency_multipliers}")

    # Test with a slice of data
    print("\n" + "="*70)
    print("Testing Detection with Different Data Slices")
    print("="*70)

    test_cases = [
        (1000, "Early data"),
        (5000, "Middle data"),
        (10000, "Later data"),
        (len(df), "Full data (latest)")
    ]

    for end_idx, description in test_cases:
        df_slice = df.iloc[:end_idx]

        # Detect
        detected = detector.detect(df_slice)

        if detected:
            # Get details
            details = detector._get_detection_details(df_slice)

            print(f"\n📊 {description} (rows: 0-{end_idx}):")
            print(f"  ✅ Detected: Yes")
            print(f"  📍 Location: {details['location']}")
            print(f"  📍 Candles ago: {details['candles_ago']}")
            print(f"  🔢 Recency multiplier: {details['recency_multiplier']:.2f}")
            print(f"  💯 Confidence: {details['confidence']:.3f}")

            # Show recency info
            recency_info = details['metadata'].get('recency_info', {})
            if recency_info:
                print(f"  📈 Base confidence: {recency_info.get('base_confidence', 0):.3f}")
                print(f"  📉 Adjusted confidence: {recency_info.get('adjusted_confidence', 0):.3f}")
        else:
            print(f"\n📊 {description} (rows: 0-{end_idx}):")
            print(f"  ❌ Detected: No")

    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)

    return True


if __name__ == '__main__':
    try:
        success = test_recency_scoring()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
