"""
تست الگوی Doji روی داده‌های تاریخی BTC - نسخه ویندوز

این اسکریپت الگوی Doji را کندل به کندل روی داده‌های تاریخی تست می‌کند.

نویسنده: Claude Code
تاریخ: 2025-10-26

نحوه اجرا در PyCharm:
1. کلیک راست روی فایل → Run 'test_doji_windows'
یا
1. باز کردن Terminal در PyCharm
2. python Pattern_Test/test_doji_windows.py
"""

import sys
import os
from pathlib import Path

# تنظیم مسیر پروژه برای Windows
# دریافت مسیر فعلی و رفتن به ریشه پروژه
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # از Pattern_Test به New

# اضافه کردن به sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"📁 Project root: {project_root}")
print(f"📁 Python paths: {sys.path[:3]}")

import csv
from datetime import datetime
import json
import shutil

# حالا می‌توانیم import کنیم
try:
    from signal_generation.analyzers.patterns.candlestick.doji import DojiPattern
    print("✅ DojiPattern imported successfully")
except ImportError as e:
    print(f"❌ خطا در import DojiPattern: {e}")
    print(f"💡 لطفاً مطمئن شوید که در مسیر پروژه هستید: {project_root}")
    sys.exit(1)


class SimpleDataFrame:
    """یک DataFrame ساده برای جایگزینی pandas"""

    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

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
            raise TypeError(f"نوع {type(key)} پشتیبانی نمی‌شود")

    def iloc(self, index):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, slice):
            return SimpleDataFrame(self.data[index], self.columns)
        else:
            raise TypeError(f"نوع {type(index)} پشتیبانی نمی‌شود")

    def copy(self):
        return SimpleDataFrame([row.copy() for row in self.data], self.columns.copy())


class DojiPatternTester:
    """کلاس تست الگوی Doji برای Windows"""

    def __init__(self):
        """مقداردهی اولیه"""
        # تنظیم مسیرها برای Windows
        self.base_dir = project_root
        self.data_dir = self.base_dir / 'historical' / 'BTC-USDT'
        self.output_dir = self.base_dir / 'Pattern_Test'
        self.charts_dir = self.output_dir / 'Charts'

        # ایجاد فولدر خروجی
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        # پاک کردن چارت‌های قبلی
        self._clear_old_charts()

        # ایجاد pattern detector
        self.doji_detector = DojiPattern()

        # ذخیره نتایج
        self.results = []

        print(f"✅ DojiPatternTester initialized")
        print(f"   📂 Data directory: {self.data_dir}")
        print(f"   📂 Output directory: {self.output_dir}")
        print(f"   📊 Charts directory: {self.charts_dir}")

    def _clear_old_charts(self):
        """پاک کردن تمام چارت‌های قبلی"""
        if self.charts_dir.exists():
            chart_files = list(self.charts_dir.glob('*.png'))
            for chart_file in chart_files:
                try:
                    chart_file.unlink()
                except Exception as e:
                    print(f"⚠️  نتوانستیم {chart_file.name} را پاک کنیم: {e}")
            if chart_files:
                print(f"🗑️  {len(chart_files)} چارت قبلی پاک شد")

    def load_csv(self, timeframe='5min'):
        """
        بارگذاری داده‌های CSV

        Args:
            timeframe: تایم‌فریم مورد نظر (5min, 15min, 1hour, 4hour)

        Returns:
            SimpleDataFrame
        """
        csv_file = self.data_dir / f"{timeframe}.csv"

        if not csv_file.exists():
            print(f"❌ فایل {csv_file} پیدا نشد!")
            print(f"💡 لطفاً مطمئن شوید فایل‌های CSV در {self.data_dir} موجود هستند")
            raise FileNotFoundError(f"فایل {csv_file} پیدا نشد!")

        print(f"\n📖 در حال خواندن {csv_file}...")

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

        print(f"   ✅ {len(df)} کندل بارگذاری شد")
        print(f"   📅 از {df[0]['timestamp']} تا {df[-1]['timestamp']}")

        return df

    def test_candle_by_candle(self, df, timeframe='5min', lookback=50, start_from=100):
        """
        تست کندل به کندل

        Args:
            df: SimpleDataFrame داده‌ها
            timeframe: نام تایم‌فریم
            lookback: تعداد کندل‌های قبلی برای نمایش (50)
            start_from: شروع تست از کندل چندم (100)
        """
        print(f"\n🔍 شروع تست کندل به کندل از کندل {start_from}...")

        total_candles = len(df)
        detections = []

        # Import pandas برای pattern detector
        try:
            import pandas as pd
            pandas_available = True
        except ImportError:
            print("❌ pandas موجود نیست!")
            print("💡 لطفاً pandas را نصب کنید: pip install pandas")
            return []

        # حلقه از start_from تا آخر
        for i in range(start_from, total_candles):
            # پیشرفت
            if (i - start_from) % 100 == 0:
                progress = ((i - start_from) / (total_candles - start_from)) * 100
                print(f"   ⏳ پیشرفت: {progress:.1f}% ({i}/{total_candles})")

            # استخراج داده‌های تا کندل فعلی
            df_slice_simple = df.iloc[:i+1]
            df_slice = pd.DataFrame(df_slice_simple.data)

            # تست pattern
            try:
                is_detected = self.doji_detector.detect(df_slice)
            except Exception as e:
                if i == start_from:
                    print(f"❌ خطا در تشخیص الگو: {e}")
                continue

            if is_detected:
                # الگو پیدا شد!
                candle_info = df[i]
                detection_info = {
                    'index': i,
                    'timestamp': candle_info['timestamp'],
                    'open': candle_info['open'],
                    'high': candle_info['high'],
                    'low': candle_info['low'],
                    'close': candle_info['close'],
                    'volume': candle_info['volume'],
                    'timeframe': timeframe
                }

                detections.append(detection_info)

                # دریافت جزئیات الگو
                try:
                    pattern_info = self.doji_detector.get_pattern_info(df_slice, timeframe)
                except Exception as e:
                    pattern_info = None

                print(f"\n🎯 الگوی Doji #{len(detections)} پیدا شد!")
                print(f"   📍 کندل {i}: {candle_info['timestamp']}")
                print(f"   💰 OHLC: O={candle_info['open']:.2f} H={candle_info['high']:.2f} "
                      f"L={candle_info['low']:.2f} C={candle_info['close']:.2f}")
                if pattern_info:
                    print(f"   ⭐ Confidence: {pattern_info.get('confidence', 0):.2%}")
                    print(f"   📊 Location: {pattern_info.get('location', 'current')}")
                    print(f"   🔍 Candles ago: {pattern_info.get('candles_ago', 0)}")

                # رسم نمودار
                self._plot_detection(df, i, lookback, timeframe, pattern_info)

        print(f"\n📊 نتایج نهایی:")
        print(f"   🔍 تعداد کندل‌های بررسی شده: {total_candles - start_from}")
        print(f"   ✅ تعداد الگوهای پیدا شده: {len(detections)}")

        # ذخیره نتایج
        self.results.extend(detections)
        self._save_results(timeframe)

        return detections

    def _plot_detection(self, df, detection_index, lookback, timeframe, pattern_info):
        """رسم نمودار کندلی"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            start_idx = max(0, detection_index - lookback)
            end_idx = detection_index + 1
            df_plot = df.iloc(slice(start_idx, end_idx))

            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

            # رسم کندل‌ها
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

            # مشخص کردن کندل تشخیص داده شده
            detection_position = detection_index - start_idx
            detection_candle = df[detection_index]

            ax.scatter([detection_position], [detection_candle['high']],
                      color='blue', s=200, marker='v', zorder=5,
                      label='🎯 Doji Pattern Detected')

            # تنظیمات محور X
            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price (USDT)', fontsize=12)
            ax.set_title(
                f'Doji Pattern Detection - BTC/USDT {timeframe}\n'
                f'Detected at: {detection_candle["timestamp"]} (Candle #{detection_index})',
                fontsize=14,
                fontweight='bold'
            )

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=10)

            if pattern_info:
                info_text = f"Confidence: {pattern_info.get('confidence', 0):.1%}\n"
                info_text += f"Direction: {pattern_info.get('direction', 'N/A')}\n"
                info_text += f"Location: {pattern_info.get('location', 'current')}\n"
                info_text += f"Candles ago: {pattern_info.get('candles_ago', 0)}"

                ax.text(
                    0.02, 0.98, info_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            timestamp_str = detection_candle["timestamp"].replace(' ', '_').replace(':', '')
            filename = f"doji_{timeframe}_candle_{detection_index}_{timestamp_str}.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"   💾 نمودار ذخیره شد: {filename}")

        except Exception as e:
            print(f"   ⚠️  خطا در رسم نمودار: {e}")

    def _save_results(self, timeframe):
        """ذخیره نتایج در فایل JSON"""
        results_file = self.output_dir / f'doji_detections_{timeframe}.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 نتایج در {results_file.name} ذخیره شد")


def main():
    """تابع اصلی"""

    print("="*80)
    print("🧪 تست الگوی Doji روی داده‌های تاریخی BTC/USDT - Windows")
    print("="*80)

    try:
        # ایجاد tester
        tester = DojiPatternTester()

        # انتخاب تایم‌فریم
        timeframe = '5min'
        print(f"\n📊 تایم‌فریم انتخاب شده: {timeframe}")

        # بارگذاری داده‌ها
        df = tester.load_csv(timeframe)

        # تست کندل به کندل
        detections = tester.test_candle_by_candle(
            df=df,
            timeframe=timeframe,
            lookback=50,
            start_from=100
        )

        print("\n" + "="*80)
        print("✅ تست با موفقیت انجام شد!")
        print(f"📊 {len(detections)} الگوی Doji پیدا شد")
        print(f"📁 نمودارها در {tester.charts_dir} ذخیره شدند")
        print("="*80)

        # نمایش چند نتیجه اول
        if detections:
            print("\n📋 نمونه نتایج (5 اولی):")
            for i, det in enumerate(detections[:5], 1):
                print(f"   {i}. کندل {det['index']}: {det['timestamp']} - "
                      f"C={det['close']:.2f}")

    except Exception as e:
        print(f"\n❌ خطا: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()

    # در ویندوز، صبر کنید تا کاربر Enter بزند
    if os.name == 'nt':  # Windows
        input("\n⏸️  Press Enter to exit...")

    sys.exit(exit_code)
