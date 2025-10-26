"""
Harami Cross Pattern Test on Historical BTC Data - Windows Version

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
    from signal_generation.analyzers.patterns.candlestick.harami_cross import HaramiCrossPattern
    print("HaramiCrossPattern imported successfully")
except ImportError as e:
    print(f"Error importing HaramiCrossPattern: {e}")
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


class HaramiCrossPatternTester:
    def __init__(self):
        self.base_dir = project_root
        self.data_dir = self.base_dir / 'historical' / 'BTC-USDT'
        self.output_dir = self.base_dir / 'Pattern_Test'
        self.charts_dir = self.output_dir / 'Charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self._clear_old_charts()
        self.detector = HaramiCrossPattern()
        self.results = []
        print(f"HaramiCrossPatternTester initialized")

    def _clear_old_charts(self):
        if self.charts_dir.exists():
            chart_files = list(self.charts_dir.glob('harami_cross_*.png'))
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
        print(f"Lookback: 11, Min candles: 12")
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

                print(f"\nHarami Cross #{len(detections)} detected!")
                print(f"   Pattern candle {pattern_candle_index}: {candle_info['timestamp']}")
                print(f"   Detected at candle {i} ({candles_ago} candles ago)")
                if pattern_info:
                    print(f"   Confidence: {pattern_info.get('confidence', 0):.2%}")

                self._plot_detection(df, i, pattern_candle_index, timeframe,
                                    pattern_info, detections, start_idx, window_size)

        unique_patterns = set(det['index'] for det in detections)
        print(f"\nTotal detections: {len(detections)}, Unique: {len(unique_patterns)}")

        self.results.extend(detections)
        self._save_results(timeframe)

        if unique_patterns:
            print(f"\nCreating summary chart with all {len(unique_patterns)} unique patterns...")
            self._plot_all_patterns(df, list(unique_patterns), timeframe)

        return detections

    def _plot_detection(self, df, detected_at_index, pattern_index, timeframe,
                        pattern_info, all_detections, window_start_idx, window_size):
        """Plot candlestick chart showing the detected pattern"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            window_end_idx = detected_at_index + 1
            df_plot = df.iloc[window_start_idx:window_end_idx]

            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

            for idx in range(len(df_plot)):
                row = df_plot[idx]
                x_pos = idx
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                color = 'green' if close_price >= open_price else 'red'
                ax.plot([x_pos, x_pos], [low_price, high_price], color='black', linewidth=1)

                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                candle_width = 0.6

                rect = Rectangle((x_pos - candle_width/2, body_bottom), candle_width, body_height,
                                facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
                ax.add_patch(rect)

            patterns_in_range = [det for det in all_detections if window_start_idx <= det['index'] < window_end_idx]

            if patterns_in_range:
                for det in patterns_in_range:
                    det_idx = det['index']
                    position = det_idx - window_start_idx
                    candle = df[det_idx]

                    if det_idx == pattern_index:
                        ax.scatter([position], [candle['high']], color='blue', s=250, marker='v', zorder=5,
                                  edgecolors='darkblue', linewidths=2, label=f'Main Pattern (candle {det_idx})')
                    else:
                        secondary_color = 'cyan' if 'blue' == 'blue' else 'orange'
                        ax.scatter([position], [candle['high']], color=secondary_color, s=150, marker='v', zorder=4,
                                  alpha=0.6, label=f'Other Pattern (candle {det_idx})')

            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price (USDT)', fontsize=12)

            pattern_candle = df[pattern_index]
            ax.set_title(
                f'Harami Cross Pattern Detection - BTC/USDT {timeframe}\n'
                f'Main Pattern: Candle #{pattern_index} at {pattern_candle["timestamp"]}\n'
                f'Detected at: Candle #{detected_at_index} ({detected_at_index - pattern_index} candles later)\n'
                f'Chart shows: ONLY window data sent to detector (#{window_start_idx} to #{detected_at_index})',
                fontsize=11, fontweight='bold'
            )

            ax.grid(True, alpha=0.3, linestyle='--')

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

            if pattern_info:
                info_text = f"Window: {window_size} candles\n"
                info_text += f"Confidence: {pattern_info.get('confidence', 0):.1%}\n"
                info_text += f"Direction: {pattern_info.get('direction', 'N/A')}\n"
                info_text += f"Location: {pattern_info.get('location', 'current')}\n"
                info_text += f"Candles ago: {pattern_info.get('candles_ago', 0)}\n"
                info_text += f"Patterns shown: {len(patterns_in_range)}"

                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            timestamp_str = pattern_candle["timestamp"].replace(' ', '_').replace(':', '').replace('-', '')
            filename = f"harami_cross_{timeframe}_detect{detected_at_index}_pattern{pattern_index}_{timestamp_str}.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"   Chart saved: {filename}")

        except Exception as e:
            print(f"   Warning: Error plotting chart: {e}")

    def _plot_all_patterns(self, df, pattern_indices, timeframe):
        """Plot a large chart showing all detected patterns"""
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
                x_pos = idx
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                color = 'green' if close_price >= open_price else 'red'
                ax.plot([x_pos, x_pos], [low_price, high_price], color='black', linewidth=1)

                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                candle_width = 0.6

                rect = Rectangle((x_pos - candle_width/2, body_bottom), candle_width, body_height,
                                facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
                ax.add_patch(rect)

            for pattern_idx in pattern_indices:
                if start_idx <= pattern_idx < end_idx:
                    pattern_position = pattern_idx - start_idx
                    pattern_candle = df[pattern_idx]
                    ax.scatter([pattern_position], [pattern_candle['high']],
                              color='blue', s=150, marker='v', zorder=5, alpha=0.7)

            num_labels = min(20, len(df_plot))
            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // num_labels)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Price (USDT)', fontsize=14)
            ax.set_title(
                f'All Harami Cross Patterns - BTC/USDT {timeframe}\n'
                f'Total {len(pattern_indices)} patterns from candle {min_idx} to {max_idx}',
                fontsize=16, fontweight='bold'
            )

            ax.grid(True, alpha=0.3, linestyle='--')

            ax.scatter([], [], color='blue', s=150, marker='v',
                      label=f'Harami Cross Pattern ({len(pattern_indices)} total)')
            ax.legend(loc='upper left', fontsize=12)

            info_text = f"Patterns: {len(pattern_indices)}\n"
            info_text += f"Range: candles {min_idx}-{max_idx}\n"
            info_text += f"Timeframe: {timeframe}"

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            filename = f"harami_cross_{timeframe}_ALL_PATTERNS_summary.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"Summary chart saved: {filename}")

        except Exception as e:
            print(f"Warning: Error creating summary chart: {e}")

    def _save_results(self, timeframe):
        results_file = self.output_dir / f'harami_cross_detections_{timeframe}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved in {results_file.name}")


def main():
    print("="*80)
    print("Harami Cross Pattern Test")
    print("="*80)

    try:
        tester = HaramiCrossPatternTester()
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
