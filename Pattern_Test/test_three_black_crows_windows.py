"""
Three Black Crows Pattern Test on Historical BTC Data - Windows Version

Author: Claude Code
Date: 2025-10-26
"""

import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import csv
import json

try:
    from signal_generation.analyzers.patterns.candlestick.three_black_crows import ThreeBlackCrowsPattern
    print("ThreeBlackCrowsPattern imported successfully")
except ImportError as e:
    print(f"Error importing ThreeBlackCrowsPattern: {e}")
    sys.exit(1)


class ILocIndexer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.dataframe.data[index]
        elif isinstance(index, slice):
            return SimpleDataFrame(self.dataframe.data[index], self.dataframe.columns)
        else:
            raise TypeError(f"Type {type(index)} not supported")


class SimpleDataFrame:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns
        self._iloc = ILocIndexer(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.data]
        elif isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return SimpleDataFrame(self.data[key], self.columns)
        return None

    @property
    def iloc(self):
        return self._iloc

    def copy(self):
        return SimpleDataFrame([row.copy() for row in self.data], self.columns.copy())


class ThreeBlackCrowsPatternTester:
    def __init__(self):
        self.base_dir = project_root
        self.data_dir = self.base_dir / 'historical' / 'BTC-USDT'
        self.output_dir = self.base_dir / 'Pattern_Test'
        self.charts_dir = self.output_dir / 'Charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self._clear_old_charts()
        self.detector = ThreeBlackCrowsPattern()
        self.results = []
        print(f"ThreeBlackCrowsPatternTester initialized")

    def _clear_old_charts(self):
        if self.charts_dir.exists():
            chart_files = list(self.charts_dir.glob('three_black_crows_*.png'))
            for chart_file in chart_files:
                try:
                    chart_file.unlink()
                except Exception:
                    pass

    def load_csv(self, timeframe='5min'):
        csv_file = self.data_dir / f"{timeframe}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"File {csv_file} not found!")
        print(f"\nReading {csv_file}...")
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })
        df = SimpleDataFrame(data, list(data[0].keys()))
        print(f"   Loaded {len(df)} candles")
        return df

    def test_candle_by_candle(self, df, timeframe='5min', start_from=100, window_size=20):
        print(f"\nStarting test from candle {start_from}...")
        print(f"Lookback: 12, Min candles: 13")
        print(f"Pattern: 3-candle bearish")
        total_candles = len(df)
        detections = []

        try:
            import pandas as pd
        except ImportError:
            print("Error: pandas not available!")
            return []

        for i in range(start_from, total_candles):
            if (i - start_from) % 100 == 0:
                progress = ((i - start_from) / (total_candles - start_from)) * 100
                print(f"   Progress: {progress:.1f}%")

            start_idx = max(0, i + 1 - window_size)
            df_slice = pd.DataFrame(df.iloc[start_idx:i+1].data)

            try:
                is_detected = self.detector.detect(df_slice)
            except Exception:
                continue

            if is_detected:
                try:
                    pattern_info = self.detector.get_pattern_info(df_slice, timeframe)
                    candles_ago = pattern_info.get('candles_ago', 0)
                except Exception:
                    pattern_info = None
                    candles_ago = 0

                pattern_candle_index = i - candles_ago
                candle_info = df[pattern_candle_index]

                detections.append({
                    'index': pattern_candle_index,
                    'timestamp': candle_info['timestamp'],
                    'confidence': pattern_info.get('confidence', 0) if pattern_info else 0
                })
                print(f"\nThree Black Crows #{len(detections)} at candle {pattern_candle_index}")

        unique_patterns = set(det['index'] for det in detections)
        print(f"\nTotal detections: {len(detections)}, Unique: {len(unique_patterns)}")

        self.results.extend(detections)
        self._save_results(timeframe)
        return detections

    def _save_results(self, timeframe):
        results_file = self.output_dir / f'three_black_crows_detections_{timeframe}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved in {results_file.name}")


def main():
    print("="*80)
    print("Three Black Crows Pattern Test")
    print("="*80)

    try:
        tester = ThreeBlackCrowsPatternTester()
        df = tester.load_csv('5min')
        tester.test_candle_by_candle(df, '5min', 100, 20)
        print("\nTest completed!")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    if os.name == 'nt':
        input("\nPress Enter to exit...")
    sys.exit(exit_code)
