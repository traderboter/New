"""
Morning Star Pattern Test on Historical BTC Data - Windows Version

This script tests the Morning Star pattern candle-by-candle on historical data.

Author: Claude Code
Date: 2025-10-26

How to run in PyCharm:
1. Right-click on file -> Run 'test_morning_star_windows'
OR
1. Open Terminal in PyCharm
2. python Pattern_Test/test_morning_star_windows.py
"""

import sys
import os
from pathlib import Path

# Configure project path for Windows
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # From Pattern_Test to New

# Add to sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python paths: {sys.path[:3]}")

import csv
from datetime import datetime
import json
import shutil

# Now we can import
try:
    from signal_generation.analyzers.patterns.candlestick.morning_star import MorningStarPattern
    print("MorningStarPattern imported successfully")
except ImportError as e:
    print(f"Error importing MorningStarPattern: {e}")
    print(f"Please make sure you are in the project path: {project_root}")
    sys.exit(1)


class ILocIndexer:
    """Helper class for iloc that works like pandas"""

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
    """A simple DataFrame to replace pandas"""

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
            sliced_data = self.data[key]
            return SimpleDataFrame(sliced_data, self.columns)
        else:
            raise TypeError(f"Type {type(key)} not supported")

    @property
    def iloc(self):
        """Return ILocIndexer for pandas-like access"""
        return self._iloc

    def copy(self):
        return SimpleDataFrame([row.copy() for row in self.data], self.columns.copy())


class MorningStarPatternTester:
    """Morning Star pattern tester class for Windows"""

    def __init__(self):
        """Initialize"""
        # Configure paths for Windows
        self.base_dir = project_root
        self.data_dir = self.base_dir / 'historical' / 'BTC-USDT'
        self.output_dir = self.base_dir / 'Pattern_Test'
        self.charts_dir = self.output_dir / 'Charts'

        # Create output folder
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        # Clear old charts
        self._clear_old_charts()

        # Create pattern detector
        self.morning_star_detector = MorningStarPattern()

        # Store results
        self.results = []

        print(f"MorningStarPatternTester initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Charts directory: {self.charts_dir}")

    def _clear_old_charts(self):
        """Clear all previous Morning Star charts"""
        if self.charts_dir.exists():
            chart_files = list(self.charts_dir.glob('morning_star_*.png'))
            for chart_file in chart_files:
                try:
                    chart_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {chart_file.name}: {e}")
            if chart_files:
                print(f"Cleared {len(chart_files)} previous Morning Star charts")

    def load_csv(self, timeframe='5min'):
        """Load CSV data"""
        csv_file = self.data_dir / f"{timeframe}.csv"

        if not csv_file.exists():
            print(f"Error: File {csv_file} not found!")
            raise FileNotFoundError(f"File {csv_file} not found!")

        print(f"\nReading {csv_file}...")

        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_row = {
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                data.append(data_row)

        df = SimpleDataFrame(data, list(data[0].keys()))
        print(f"   Loaded {len(df)} candles")
        return df

    def test_candle_by_candle(self, df, timeframe='5min', start_from=100, window_size=20):
        """Test candle-by-candle - simulates real bot behavior"""
        print(f"\nStarting candle-by-candle test from candle {start_from}...")
        print(f"Window size: {window_size} candles")
        print(f"Detector lookback: 12 candles")
        print(f"TA-Lib requirement: minimum 13 candles")
        print(f"Pattern: 3-candle bullish reversal (Bearish -> Star -> Bullish)")

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
                print(f"   Progress: {progress:.1f}% ({i}/{total_candles})")

            start_idx = max(0, i + 1 - window_size)
            df_slice_simple = df.iloc[start_idx:i+1]
            df_slice = pd.DataFrame(df_slice_simple.data)

            try:
                is_detected = self.morning_star_detector.detect(df_slice)
            except Exception as e:
                if i == start_from:
                    print(f"Error in pattern detection: {e}")
                continue

            if is_detected:
                try:
                    pattern_info = self.morning_star_detector.get_pattern_info(df_slice, timeframe)
                    candles_ago = pattern_info.get('candles_ago', 0)
                except Exception:
                    pattern_info = None
                    candles_ago = 0

                pattern_candle_index = i - candles_ago
                candle_info = df[pattern_candle_index]

                detection_info = {
                    'index': pattern_candle_index,
                    'detected_at_index': i,
                    'candles_ago': candles_ago,
                    'timestamp': candle_info['timestamp'],
                    'open': candle_info['open'],
                    'high': candle_info['high'],
                    'low': candle_info['low'],
                    'close': candle_info['close'],
                    'volume': candle_info['volume'],
                    'timeframe': timeframe,
                    'confidence': pattern_info.get('confidence', 0) if pattern_info else 0
                }

                detections.append(detection_info)
                print(f"\nMorning Star pattern #{len(detections)} detected!")
                print(f"   Pattern candle {pattern_candle_index}: {candle_info['timestamp']}")

                self._plot_detection(df, i, pattern_candle_index, timeframe,
                                    pattern_info, detections, start_idx, window_size)

        print(f"\nFinal results:")
        print(f"   Total detections: {len(detections)}")
        unique_patterns = set(det['index'] for det in detections)
        print(f"   Unique patterns: {len(unique_patterns)}")

        self.results.extend(detections)
        self._save_results(timeframe)

        if unique_patterns:
            self._plot_all_patterns(df, list(unique_patterns), timeframe)

        return detections

    def _plot_detection(self, df, detected_at_index, pattern_index, timeframe,
                        pattern_info, all_detections, window_start_idx, window_size):
        """Plot candlestick chart showing detection"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            window_end_idx = detected_at_index + 1
            df_plot = df.iloc[window_start_idx:window_end_idx]

            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

            # Draw candles
            for idx in range(len(df_plot)):
                row = df_plot[idx]
                x_pos = idx
                color = 'green' if row['close'] >= row['open'] else 'red'

                ax.plot([x_pos, x_pos], [row['low'], row['high']], color='black', linewidth=1)

                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                rect = Rectangle((x_pos - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
                ax.add_patch(rect)

            # Mark pattern candles
            patterns_in_range = [det for det in all_detections
                               if window_start_idx <= det['index'] < window_end_idx]

            for det in patterns_in_range:
                det_idx = det['index']
                if det_idx - 2 >= window_start_idx:
                    first_pos = det_idx - 2 - window_start_idx
                    second_pos = det_idx - 1 - window_start_idx
                    third_pos = det_idx - window_start_idx

                    first_candle = df[det_idx - 2]
                    second_candle = df[det_idx - 1]
                    third_candle = df[det_idx]

                    if det_idx == pattern_index:
                        # Mark all 3 candles
                        ax.scatter([first_pos], [first_candle['low']], color='red', s=200,
                                 marker='v', zorder=5, edgecolors='darkred', linewidths=2,
                                 label=f'1st: Bearish')
                        ax.scatter([second_pos], [second_candle['low']], color='gold', s=250,
                                 marker='*', zorder=5, edgecolors='orange', linewidths=2,
                                 label=f'2nd: Star')
                        ax.scatter([third_pos], [third_candle['low']], color='darkgreen', s=250,
                                 marker='^', zorder=5, edgecolors='green', linewidths=2,
                                 label=f'3rd: Bullish')

            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price (USDT)', fontsize=12)

            pattern_candle = df[pattern_index]
            ax.set_title(
                f'Morning Star Pattern Detection - BTC/USDT {timeframe}\n'
                f'3-Candle Sequence: Bearish (v) -> Star (*) -> Bullish (^)',
                fontsize=11, fontweight='bold'
            )

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=9)

            if pattern_info:
                info_text = f"Confidence: {pattern_info.get('confidence', 0):.1%}\n"
                info_text += f"Direction: {pattern_info.get('direction', 'N/A')}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

            timestamp_str = pattern_candle["timestamp"].replace(' ', '_').replace(':', '').replace('-', '')
            filename = f"morning_star_{timeframe}_detect{detected_at_index}_pattern{pattern_index}_{timestamp_str}.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"   Chart saved: {filename}")

        except Exception as e:
            print(f"   Warning: Error plotting chart: {e}")

    def _plot_all_patterns(self, df, pattern_indices, timeframe):
        """Plot summary chart with all patterns"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            if not pattern_indices:
                return

            min_idx = min(pattern_indices)
            max_idx = max(pattern_indices)
            start_idx = max(0, min_idx - 50)
            end_idx = min(len(df), max_idx + 50)
            df_plot = df.iloc[start_idx:end_idx]

            fig, ax = plt.subplots(figsize=(24, 12), dpi=100)

            for idx in range(len(df_plot)):
                row = df_plot[idx]
                color = 'green' if row['close'] >= row['open'] else 'red'
                ax.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=1)
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                rect = Rectangle((idx - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
                ax.add_patch(rect)

            for pattern_idx in pattern_indices:
                if start_idx <= pattern_idx < end_idx:
                    pattern_position = pattern_idx - start_idx
                    pattern_candle = df[pattern_idx]
                    ax.scatter([pattern_position], [pattern_candle['low']],
                             color='darkgreen', s=150, marker='^', zorder=5, alpha=0.7)

            ax.set_title(f'All Morning Star Patterns - BTC/USDT {timeframe}\n'
                        f'Total {len(pattern_indices)} patterns',
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            filename = f"morning_star_{timeframe}_ALL_PATTERNS_summary.png"
            filepath = self.charts_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"Summary chart saved: {filename}")

        except Exception as e:
            print(f"Warning: Error creating summary chart: {e}")

    def _save_results(self, timeframe):
        """Save results to JSON file"""
        results_file = self.output_dir / f'morning_star_detections_{timeframe}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved in {results_file.name}")


def main():
    """Main function"""
    print("="*80)
    print("Morning Star Pattern Test on Historical BTC/USDT Data - Windows")
    print("="*80)

    try:
        tester = MorningStarPatternTester()
        timeframe = '5min'
        df = tester.load_csv(timeframe)
        detections = tester.test_candle_by_candle(df=df, timeframe=timeframe,
                                                  start_from=100, window_size=20)

        unique_patterns = set(det['index'] for det in detections)
        print("\n" + "="*80)
        print("Test completed successfully!")
        print(f"Total detections: {len(detections)}")
        print(f"Unique patterns: {len(unique_patterns)}")
        print("="*80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    if os.name == 'nt':
        input("\nPress Enter to exit...")
    sys.exit(exit_code)
