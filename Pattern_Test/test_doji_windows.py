"""
Doji Pattern Test on Historical BTC Data - Windows Version

This script tests the Doji pattern candle-by-candle on historical data.

Author: Claude Code
Date: 2025-10-26

How to run in PyCharm:
1. Right-click on file -> Run 'test_doji_windows'
OR
1. Open Terminal in PyCharm
2. python Pattern_Test/test_doji_windows.py
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
    from signal_generation.analyzers.patterns.candlestick.doji import DojiPattern
    print("DojiPattern imported successfully")
except ImportError as e:
    print(f"Error importing DojiPattern: {e}")
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


class DojiPatternTester:
    """Doji pattern tester class for Windows"""

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
        self.doji_detector = DojiPattern()

        # Store results
        self.results = []

        print(f"DojiPatternTester initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Charts directory: {self.charts_dir}")

    def _clear_old_charts(self):
        """Clear all previous charts"""
        if self.charts_dir.exists():
            chart_files = list(self.charts_dir.glob('*.png'))
            for chart_file in chart_files:
                try:
                    chart_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {chart_file.name}: {e}")
            if chart_files:
                print(f"Cleared {len(chart_files)} previous charts")

    def load_csv(self, timeframe='5min'):
        """
        Load CSV data

        Args:
            timeframe: Desired timeframe (5min, 15min, 1hour, 4hour)

        Returns:
            SimpleDataFrame
        """
        csv_file = self.data_dir / f"{timeframe}.csv"

        if not csv_file.exists():
            print(f"Error: File {csv_file} not found!")
            print(f"Please make sure CSV files are in {self.data_dir}")
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
        print(f"   From {df[0]['timestamp']} to {df[-1]['timestamp']}")

        return df

    def test_candle_by_candle(self, df, timeframe='5min', lookback=50, start_from=100):
        """
        Test candle-by-candle

        Args:
            df: SimpleDataFrame of data
            timeframe: Timeframe name
            lookback: Number of previous candles to display (50)
            start_from: Start test from which candle (100)
        """
        print(f"\nStarting candle-by-candle test from candle {start_from}...")

        total_candles = len(df)
        detections = []

        # Import pandas for pattern detector
        try:
            import pandas as pd
            pandas_available = True
        except ImportError:
            print("Error: pandas not available!")
            print("Please install pandas: pip install pandas")
            return []

        # Loop from start_from to end
        for i in range(start_from, total_candles):
            # Progress
            if (i - start_from) % 100 == 0:
                progress = ((i - start_from) / (total_candles - start_from)) * 100
                print(f"   Progress: {progress:.1f}% ({i}/{total_candles})")

            # Extract data up to current candle
            df_slice_simple = df.iloc[:i+1]
            df_slice = pd.DataFrame(df_slice_simple.data)

            # Test pattern
            try:
                is_detected = self.doji_detector.detect(df_slice)
            except Exception as e:
                if i == start_from:
                    print(f"Error in pattern detection: {e}")
                continue

            if is_detected:
                # Pattern found!
                # Get pattern details to find which candle has the pattern
                try:
                    pattern_info = self.doji_detector.get_pattern_info(df_slice, timeframe)
                    candles_ago = pattern_info.get('candles_ago', 0)
                except Exception as e:
                    pattern_info = None
                    candles_ago = 0

                # Calculate the actual index of the pattern candle
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
                    'timeframe': timeframe
                }

                detections.append(detection_info)

                print(f"\nDoji pattern #{len(detections)} detected!")
                print(f"   Pattern candle {pattern_candle_index}: {candle_info['timestamp']}")
                print(f"   Detected at candle {i} ({candles_ago} candles ago)")
                print(f"   OHLC: O={candle_info['open']:.2f} H={candle_info['high']:.2f} "
                      f"L={candle_info['low']:.2f} C={candle_info['close']:.2f}")
                if pattern_info:
                    print(f"   Confidence: {pattern_info.get('confidence', 0):.2%}")
                    print(f"   Location: {pattern_info.get('location', 'current')}")
                    print(f"   Recency multiplier: {pattern_info.get('recency_multiplier', 1.0):.2f}")

                # Plot chart
                self._plot_detection(df, pattern_candle_index, lookback, timeframe, pattern_info)

        print(f"\nFinal results:")
        print(f"   Candles checked: {total_candles - start_from}")
        print(f"   Patterns found: {len(detections)}")

        # Save results
        self.results.extend(detections)
        self._save_results(timeframe)

        return detections

    def _plot_detection(self, df, pattern_index, lookback, timeframe, pattern_info):
        """Plot candlestick chart"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            start_idx = max(0, pattern_index - lookback)
            end_idx = pattern_index + lookback
            end_idx = min(end_idx, len(df))
            df_plot = df.iloc[start_idx:end_idx]

            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

            # Draw candles
            for idx in range(len(df_plot)):
                row = df_plot[idx]
                x_pos = idx
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                color = 'green' if close_price >= open_price else 'red'

                ax.plot([x_pos, x_pos], [low_price, high_price],
                       color='black', linewidth=1)

                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                candle_width = 0.6

                rect = Rectangle(
                    (x_pos - candle_width/2, body_bottom),
                    candle_width,
                    body_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
                ax.add_patch(rect)

            # Mark the pattern candle
            pattern_position = pattern_index - start_idx
            pattern_candle = df[pattern_index]

            # Use simple marker instead of emoji
            ax.scatter([pattern_position], [pattern_candle['high']],
                      color='blue', s=200, marker='v', zorder=5,
                      label='Doji Pattern Detected')

            # X-axis settings
            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price (USDT)', fontsize=12)
            ax.set_title(
                f'Doji Pattern Detection - BTC/USDT {timeframe}\n'
                f'Pattern at: {pattern_candle["timestamp"]} (Candle #{pattern_index})',
                fontsize=14,
                fontweight='bold'
            )

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=10)

            if pattern_info:
                info_text = f"Confidence: {pattern_info.get('confidence', 0):.1%}\n"
                info_text += f"Direction: {pattern_info.get('direction', 'N/A')}\n"
                info_text += f"Location: {pattern_info.get('location', 'current')}\n"
                info_text += f"Candles ago: {pattern_info.get('candles_ago', 0)}\n"
                info_text += f"Recency mult: {pattern_info.get('recency_multiplier', 1.0):.2f}"

                ax.text(
                    0.02, 0.98, info_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            timestamp_str = pattern_candle["timestamp"].replace(' ', '_').replace(':', '')
            filename = f"doji_{timeframe}_candle_{pattern_index}_{timestamp_str}.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"   Chart saved: {filename}")

        except Exception as e:
            print(f"   Warning: Error plotting chart: {e}")

    def _save_results(self, timeframe):
        """Save results to JSON file"""
        results_file = self.output_dir / f'doji_detections_{timeframe}.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved in {results_file.name}")


def main():
    """Main function"""

    print("="*80)
    print("Doji Pattern Test on Historical BTC/USDT Data - Windows")
    print("="*80)

    try:
        # Create tester
        tester = DojiPatternTester()

        # Select timeframe
        timeframe = '5min'
        print(f"\nSelected timeframe: {timeframe}")

        # Load data
        df = tester.load_csv(timeframe)

        # Test candle-by-candle
        detections = tester.test_candle_by_candle(
            df=df,
            timeframe=timeframe,
            lookback=50,
            start_from=100
        )

        print("\n" + "="*80)
        print("Test completed successfully!")
        print(f"Found {len(detections)} Doji patterns")
        print(f"Charts saved in {tester.charts_dir}")
        print("="*80)

        # Show first few results
        if detections:
            print("\nSample results (first 5):")
            for i, det in enumerate(detections[:5], 1):
                print(f"   {i}. Candle {det['index']}: {det['timestamp']} - "
                      f"C={det['close']:.2f} (detected {det['candles_ago']} candles ago)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()

    # On Windows, wait for user to press Enter
    if os.name == 'nt':  # Windows
        input("\nPress Enter to exit...")

    sys.exit(exit_code)
