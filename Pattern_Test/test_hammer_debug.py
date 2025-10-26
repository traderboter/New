"""
Quick debug test for Hammer pattern detection
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import csv
from signal_generation.analyzers.patterns.candlestick.hammer import HammerPattern

# Load some data
data_file = project_root / 'historical' / 'BTC-USDT' / '5min.csv'

print("Loading data...")
data = []
with open(data_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 1000:  # Load first 1000 candles
            break
        data.append({
            'timestamp': row['timestamp'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} candles")

# Test 1: Default Hammer detector
print("\n" + "="*60)
print("Test 1: Default Hammer detector (v4.0.0 - no trend checking)")
print("="*60)
detector1 = HammerPattern()
print(f"Config: lookback_window={detector1.lookback_window}")
print(f"Note: Trend checking removed - handled separately now")

# Test with different window sizes
for window_size in [20, 50, 100, 200]:
    df_slice = df.tail(window_size)
    result = detector1.detect(df_slice)
    print(f"Window size {window_size:3d}: detected = {result}")

    if result:
        # Get pattern info
        info = detector1.get_pattern_info(df_slice, '5min')
        print(f"   -> Pattern found!")
        print(f"   -> Candles ago: {info.get('candles_ago', 0)}")
        print(f"   -> Confidence: {info.get('confidence', 0):.2%}")
        break

# Test 2: Scan through entire dataset
print("\n" + "="*60)
print("Test 2: Scan first 1000 candles")
print("="*60)
detector2 = HammerPattern()
detections = 0

for i in range(100, len(df)):
    df_slice = df.iloc[max(0, i-100):i+1]  # Last 100 candles
    if detector2.detect(df_slice):
        detections += 1
        if detections <= 5:  # Show first 5
            candle = df.iloc[i]
            print(f"Detection {detections}: Candle {i} at {candle['timestamp']}")

print(f"\nTotal detections: {detections}/{len(df)-100}")
print(f"Detection rate: {detections/(len(df)-100)*100:.2f}%")
