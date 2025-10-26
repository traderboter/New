"""
Engulfing Pattern Test on Historical BTC Data - Windows Version

This script tests the Engulfing pattern candle-by-candle on historical data.
Tests both Bullish and Bearish Engulfing patterns.

Author: Claude Code
Date: 2025-10-26

How to run in PyCharm:
1. Right-click on file -> Run 'test_engulfing_windows'
OR
1. Open Terminal in PyCharm
2. python Pattern_Test/test_engulfing_windows.py
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
    from signal_generation.analyzers.patterns.candlestick.engulfing import EngulfingPattern
    print("EngulfingPattern imported successfully")
except ImportError as e:
    print(f"Error importing EngulfingPattern: {e}")
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


class EngulfingPatternTester:
    """Engulfing pattern tester class for Windows"""

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
        # NOTE: Trend checking is handled separately by Trend Analyzer
        self.engulfing_detector = EngulfingPattern()

        # Store results
        self.results = []

        print(f"EngulfingPatternTester initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Charts directory: {self.charts_dir}")

    def _clear_old_charts(self):
        """Clear all previous Engulfing charts"""
        if self.charts_dir.exists():
            chart_files = list(self.charts_dir.glob('engulfing_*.png'))
            for chart_file in chart_files:
                try:
                    chart_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {chart_file.name}: {e}")
            if chart_files:
                print(f"Cleared {len(chart_files)} previous Engulfing charts")

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

    def test_candle_by_candle(self, df, timeframe='5min', start_from=100,
                              window_size=20):
        """
        Test candle-by-candle - simulates real bot behavior

        IMPORTANT: This test simulates how the real bot works:
        - Bot only sees last N candles (window_size, default 20)
        - Bot moves forward one candle at a time
        - Detector searches in last 2 candles within that window
        - Charts show ONLY the data sent to detector (no future data!)

        Args:
            df: SimpleDataFrame of data
            timeframe: Timeframe name
            start_from: Start test from which candle (100)
            window_size: Number of candles to send to detector (20, like real bot)
        """
        print(f"\nStarting candle-by-candle test from candle {start_from}...")
        print(f"Window size: {window_size} candles (simulating real bot)")
        print(f"Detector lookback: 2 candles (last 2 in window)")
        print(f"TA-Lib requirement: minimum 3 candles")
        print(f"Pattern types: Bullish Engulfing (+100) and Bearish Engulfing (-100)")
        print(f"Charts: Show ONLY window data (no future data!)")

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

            # CRITICAL: Simulate real bot - only send last N candles
            # Real bot doesn't have access to all history, just recent window
            start_idx = max(0, i + 1 - window_size)
            df_slice_simple = df.iloc[start_idx:i+1]
            df_slice = pd.DataFrame(df_slice_simple.data)

            # Test pattern
            try:
                is_detected = self.engulfing_detector.detect(df_slice)
            except Exception as e:
                if i == start_from:
                    print(f"Error in pattern detection: {e}")
                continue

            if is_detected:
                # Pattern found!
                # Get pattern details to find which candle has the pattern
                try:
                    pattern_info = self.engulfing_detector.get_pattern_info(df_slice, timeframe)
                    candles_ago = pattern_info.get('candles_ago', 0)
                except Exception as e:
                    pattern_info = None
                    candles_ago = 0

                # Calculate the actual index of the pattern candle
                pattern_candle_index = i - candles_ago
                candle_info = df[pattern_candle_index]

                # Get pattern direction from metadata
                metadata = pattern_info.get('metadata', {}) if pattern_info else {}
                pattern_direction = metadata.get('pattern_direction', 'unknown')

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
                    'confidence': pattern_info.get('confidence', 0) if pattern_info else 0,
                    'direction': pattern_direction
                }

                detections.append(detection_info)

                print(f"\nEngulfing pattern #{len(detections)} detected!")
                print(f"   Pattern type: {pattern_direction.upper()}")
                print(f"   Pattern candle {pattern_candle_index}: {candle_info['timestamp']}")
                print(f"   Detected at candle {i} ({candles_ago} candles ago)")
                print(f"   OHLC: O={candle_info['open']:.2f} H={candle_info['high']:.2f} "
                      f"L={candle_info['low']:.2f} C={candle_info['close']:.2f}")
                if pattern_info:
                    print(f"   Confidence: {pattern_info.get('confidence', 0):.2%}")
                    print(f"   Location: {pattern_info.get('location', 'current')}")
                    print(f"   Recency multiplier: {pattern_info.get('recency_multiplier', 1.0):.2f}")
                    # Engulfing specific info
                    if metadata:
                        print(f"   Engulfing ratio: {metadata.get('engulfing_ratio', 0):.2f}")

                # Plot chart - show ONLY the window data sent to detector
                # No future data! Just like real bot sees
                self._plot_detection(df, i, pattern_candle_index, timeframe,
                                    pattern_info, detections, start_idx, window_size)

        print(f"\nFinal results:")
        print(f"   Candles checked: {total_candles - start_from}")
        print(f"   Total detections: {len(detections)}")

        # Count unique patterns and by direction
        unique_patterns = set(det['index'] for det in detections)
        bullish_count = sum(1 for det in detections if det.get('direction') == 'bullish')
        bearish_count = sum(1 for det in detections if det.get('direction') == 'bearish')

        print(f"   Unique patterns: {len(unique_patterns)}")
        print(f"   Bullish Engulfing: {bullish_count}")
        print(f"   Bearish Engulfing: {bearish_count}")

        # Save results
        self.results.extend(detections)
        self._save_results(timeframe)

        # Create summary chart with all unique patterns
        if unique_patterns:
            print(f"\nCreating summary chart with all {len(unique_patterns)} unique patterns...")
            self._plot_all_patterns(df, list(unique_patterns), detections, timeframe)

        return detections

    def _plot_detection(self, df, detected_at_index, pattern_index, timeframe,
                        pattern_info, all_detections, window_start_idx, window_size):
        """
        Plot candlestick chart showing ONLY the data sent to detector

        IMPORTANT: Charts show exactly what the bot sees - NO FUTURE DATA!
        - Plot range: ONLY the window sent to detector (window_size candles)
        - This simulates real bot that doesn't have access to future

        Args:
            df: Full dataframe (for reference only)
            detected_at_index: The candle index where detection happened
            pattern_index: The actual candle index where the pattern is
            timeframe: Timeframe string
            pattern_info: Pattern information dict
            all_detections: List of all detections so far
            window_start_idx: Start index of the window sent to detector
            window_size: Size of window sent to detector
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            # CRITICAL: Plot ONLY the window data (no future data!)
            # This is exactly what was sent to the detector
            window_end_idx = detected_at_index + 1
            df_plot = df.iloc[window_start_idx:window_end_idx]

            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

            # Draw all candles
            for idx in range(len(df_plot)):
                row = df_plot[idx]
                x_pos = idx
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                color = 'green' if close_price >= open_price else 'red'

                # Wick
                ax.plot([x_pos, x_pos], [low_price, high_price],
                       color='black', linewidth=1)

                # Body
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

            # Find all patterns in this window
            patterns_in_range = []
            for det in all_detections:
                det_idx = det['index']
                if window_start_idx <= det_idx < window_end_idx:
                    patterns_in_range.append(det)

            # Mark all patterns in this window
            if patterns_in_range:
                for det in patterns_in_range:
                    det_idx = det['index']
                    position = det_idx - window_start_idx
                    candle = df[det_idx]
                    direction = det.get('direction', 'unknown')

                    # Different marker based on direction and whether it's the main pattern
                    if det_idx == pattern_index:
                        # Main pattern
                        if direction == 'bullish':
                            # Bullish - green triangle up at bottom
                            ax.scatter([position], [candle['low']],
                                      color='darkgreen', s=250, marker='^', zorder=5,
                                      edgecolors='green', linewidths=2,
                                      label=f'Main Pattern: Bullish (candle {det_idx})')
                        else:
                            # Bearish - red triangle down at top
                            ax.scatter([position], [candle['high']],
                                      color='darkred', s=250, marker='v', zorder=5,
                                      edgecolors='red', linewidths=2,
                                      label=f'Main Pattern: Bearish (candle {det_idx})')
                    else:
                        # Other patterns
                        if direction == 'bullish':
                            ax.scatter([position], [candle['low']],
                                      color='lightgreen', s=150, marker='^', zorder=4,
                                      alpha=0.6,
                                      label=f'Other: Bullish (candle {det_idx})')
                        else:
                            ax.scatter([position], [candle['high']],
                                      color='orange', s=150, marker='v', zorder=4,
                                      alpha=0.6,
                                      label=f'Other: Bearish (candle {det_idx})')

            # X-axis settings
            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price (USDT)', fontsize=12)

            pattern_candle = df[pattern_index]
            metadata = pattern_info.get('metadata', {}) if pattern_info else {}
            pattern_direction = metadata.get('pattern_direction', 'unknown')

            ax.set_title(
                f'Engulfing Pattern Detection ({pattern_direction.upper()}) - BTC/USDT {timeframe}\n'
                f'Main Pattern: Candle #{pattern_index} at {pattern_candle["timestamp"]}\n'
                f'Detected at: Candle #{detected_at_index} ({detected_at_index - pattern_index} candles later)\n'
                f'Chart shows: ONLY window data sent to detector (#{window_start_idx} to #{detected_at_index}) - NO FUTURE DATA',
                fontsize=11,
                fontweight='bold'
            )

            ax.grid(True, alpha=0.3, linestyle='--')

            # Legend - only show unique labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

            # Info text
            if pattern_info:
                info_text = f"REALISTIC SIMULATION:\n"
                info_text += f"Chart = Exact data sent to detector\n"
                info_text += f"Window: {window_size} candles\n"
                info_text += f"Lookback: 2 candles\n"
                info_text += f"TA-Lib minimum: 3 candles\n"
                info_text += f"NO future data!\n\n"
                info_text += f"PATTERN INFO:\n"
                info_text += f"Type: {pattern_direction.upper()}\n"
                info_text += f"Confidence: {pattern_info.get('confidence', 0):.1%}\n"
                info_text += f"Direction: {pattern_info.get('direction', 'N/A')}\n"
                info_text += f"Location: {pattern_info.get('location', 'current')}\n"
                info_text += f"Candles ago: {pattern_info.get('candles_ago', 0)}\n"
                info_text += f"Recency: {pattern_info.get('recency_multiplier', 1.0):.2f}\n"

                # Engulfing specific info
                if metadata:
                    info_text += f"Engulf ratio: {metadata.get('engulfing_ratio', 0):.2f}\n"

                info_text += f"Patterns shown: {len(patterns_in_range)}"

                # Color based on direction
                box_color = 'lightgreen' if pattern_direction == 'bullish' else 'lightcoral'

                ax.text(
                    0.02, 0.98, info_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7)
                )

            # Filename includes detection index and direction
            timestamp_str = pattern_candle["timestamp"].replace(' ', '_').replace(':', '').replace('-', '')
            filename = f"engulfing_{pattern_direction}_{timeframe}_detect{detected_at_index}_pattern{pattern_index}_{timestamp_str}.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"   Chart saved: {filename}")

        except Exception as e:
            print(f"   Warning: Error plotting chart: {e}")
            import traceback
            traceback.print_exc()

    def _plot_all_patterns(self, df, pattern_indices, all_detections, timeframe):
        """Plot a large chart showing all detected patterns"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            # Determine range to plot
            if not pattern_indices:
                return

            min_idx = min(pattern_indices)
            max_idx = max(pattern_indices)

            # Add some padding
            start_idx = max(0, min_idx - 50)
            end_idx = min(len(df), max_idx + 50)

            df_plot = df.iloc[start_idx:end_idx]

            # Create large figure
            fig, ax = plt.subplots(figsize=(24, 12), dpi=100)

            # Draw all candles
            for idx in range(len(df_plot)):
                row = df_plot[idx]
                x_pos = idx
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                color = 'green' if close_price >= open_price else 'red'

                # Wick
                ax.plot([x_pos, x_pos], [low_price, high_price],
                       color='black', linewidth=1)

                # Body
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

            # Create detection map
            detection_map = {det['index']: det for det in all_detections}

            # Mark all pattern candles
            bullish_count = 0
            bearish_count = 0

            for pattern_idx in pattern_indices:
                if start_idx <= pattern_idx < end_idx:
                    pattern_position = pattern_idx - start_idx
                    pattern_candle = df[pattern_idx]

                    # Get direction from detection
                    detection = detection_map.get(pattern_idx, {})
                    direction = detection.get('direction', 'unknown')

                    if direction == 'bullish':
                        # Bullish - green up arrow at bottom
                        ax.scatter([pattern_position], [pattern_candle['low']],
                                  color='darkgreen', s=150, marker='^', zorder=5, alpha=0.7)
                        bullish_count += 1
                    else:
                        # Bearish - red down arrow at top
                        ax.scatter([pattern_position], [pattern_candle['high']],
                                  color='darkred', s=150, marker='v', zorder=5, alpha=0.7)
                        bearish_count += 1

            # X-axis settings - show fewer labels for readability
            num_labels = min(20, len(df_plot))
            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // num_labels)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Price (USDT)', fontsize=14)
            ax.set_title(
                f'All Engulfing Patterns - BTC/USDT {timeframe}\n'
                f'Total {len(pattern_indices)} patterns: {bullish_count} Bullish (▲) + {bearish_count} Bearish (▼)\n'
                f'From candle {min_idx} to {max_idx}',
                fontsize=16,
                fontweight='bold'
            )

            ax.grid(True, alpha=0.3, linestyle='--')

            # Add legend
            ax.scatter([], [], color='darkgreen', s=150, marker='^',
                      label=f'Bullish Engulfing ({bullish_count} total)')
            ax.scatter([], [], color='darkred', s=150, marker='v',
                      label=f'Bearish Engulfing ({bearish_count} total)')
            ax.legend(loc='upper left', fontsize=12)

            # Add info text
            info_text = f"Total: {len(pattern_indices)} patterns\n"
            info_text += f"Bullish: {bullish_count}\n"
            info_text += f"Bearish: {bearish_count}\n"
            info_text += f"Range: candles {min_idx}-{max_idx}\n"
            info_text += f"Timeframe: {timeframe}"

            ax.text(
                0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
            )

            filename = f"engulfing_{timeframe}_ALL_PATTERNS_summary.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"Summary chart saved: {filename}")

        except Exception as e:
            print(f"Warning: Error creating summary chart: {e}")
            import traceback
            traceback.print_exc()

    def _save_results(self, timeframe):
        """Save results to JSON file"""
        results_file = self.output_dir / f'engulfing_detections_{timeframe}.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved in {results_file.name}")


def main():
    """Main function"""

    print("="*80)
    print("Engulfing Pattern Test on Historical BTC/USDT Data - Windows")
    print("Testing both Bullish and Bearish Engulfing patterns")
    print("="*80)

    try:
        # Create tester
        tester = EngulfingPatternTester()

        # Select timeframe
        timeframe = '5min'
        print(f"\nSelected timeframe: {timeframe}")

        # Load data
        df = tester.load_csv(timeframe)

        # Test candle-by-candle - simulates real bot with 20-candle window
        detections = tester.test_candle_by_candle(
            df=df,
            timeframe=timeframe,
            start_from=100,
            window_size=20  # Same as real bot
        )

        # Count unique patterns and directions
        unique_patterns = set(det['index'] for det in detections)
        bullish_count = sum(1 for det in detections if det.get('direction') == 'bullish')
        bearish_count = sum(1 for det in detections if det.get('direction') == 'bearish')

        print("\n" + "="*80)
        print("Test completed successfully!")
        print(f"Total detections: {len(detections)}")
        print(f"Unique patterns: {len(unique_patterns)}")
        print(f"  - Bullish Engulfing: {bullish_count}")
        print(f"  - Bearish Engulfing: {bearish_count}")
        print(f"Charts saved in {tester.charts_dir}")
        print("="*80)

        # Show first few unique results by direction
        if detections:
            # Get first detection of each unique pattern
            seen_patterns = set()
            unique_detections = []
            for det in detections:
                if det['index'] not in seen_patterns:
                    unique_detections.append(det)
                    seen_patterns.add(det['index'])

            print("\nFirst 10 unique patterns:")
            for i, det in enumerate(unique_detections[:10], 1):
                direction_emoji = "🟢" if det.get('direction') == 'bullish' else "🔴"
                print(f"   {i}. {direction_emoji} {det.get('direction', 'unknown').upper()}: "
                      f"Candle {det['index']} at {det['timestamp']} - C={det['close']:.2f}")

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
