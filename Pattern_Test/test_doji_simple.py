"""
تست الگوی Doji روی داده‌های تاریخی BTC (نسخه ساده بدون pandas)

این اسکریپت الگوی Doji را کندل به کندل روی داده‌های تاریخی تست می‌کند.

نویسنده: Claude Code
تاریخ: 2025-10-26
"""

import sys
import os
import csv
from datetime import datetime
from pathlib import Path
import json
import shutil

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.insert(0, '/home/user/New')


class SimpleDataFrame:
    """یک DataFrame ساده برای جایگزینی pandas"""

    def __init__(self, data, columns):
        """
        Args:
            data: لیستی از دیکشنری‌ها
            columns: نام ستون‌ها
        """
        self.data = data
        self.columns = columns
        self._index = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """دسترسی به ستون یا سطر"""
        if isinstance(key, str):
            # دسترسی به ستون
            return [row[key] for row in self.data]
        elif isinstance(key, int):
            # دسترسی به سطر
            return self.data[key]
        elif isinstance(key, slice):
            # برش
            sliced_data = self.data[key]
            return SimpleDataFrame(sliced_data, self.columns)
        else:
            raise TypeError(f"نوع {type(key)} پشتیبانی نمی‌شود")

    def iloc(self, index):
        """دسترسی به سطر با ایندکس"""
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, slice):
            return SimpleDataFrame(self.data[index], self.columns)
        else:
            raise TypeError(f"نوع {type(index)} پشتیبانی نمی‌شود")

    def copy(self):
        """کپی از DataFrame"""
        return SimpleDataFrame([row.copy() for row in self.data], self.columns.copy())

    def to_pandas_like(self):
        """تبدیل به فرمت pandas-like برای سازگاری با pattern detectors"""
        import pandas as pd
        return pd.DataFrame(self.data)


class DojiPatternTester:
    """
    کلاس تست الگوی Doji (نسخه ساده)
    """

    def __init__(self, data_dir='historical/BTC-USDT', output_dir='Pattern_Test'):
        """
        مقداردهی اولیه

        Args:
            data_dir: مسیر فولدر داده‌های تاریخی
            output_dir: مسیر فولدر خروجی
        """
        self.base_dir = Path('/home/user/New')
        self.data_dir = self.base_dir / data_dir
        self.output_dir = self.base_dir / output_dir
        self.charts_dir = self.output_dir / 'Charts'

        # ایجاد فولدر خروجی
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        # پاک کردن چارت‌های قبلی
        self._clear_old_charts()

        # ایجاد pattern detector
        try:
            from signal_generation.analyzers.patterns.candlestick.doji import DojiPattern
            self.doji_detector = DojiPattern()
            print(f"✅ DojiPattern بارگذاری شد")
        except Exception as e:
            print(f"❌ خطا در بارگذاری DojiPattern: {e}")
            raise

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
                chart_file.unlink()
            if chart_files:
                print(f"🗑️  {len(chart_files)} چارت قبلی پاک شد")

    def load_csv(self, timeframe='5min'):
        """
        بارگذاری داده‌های CSV

        Args:
            timeframe: تایم‌فریم مورد نظر (5min, 15min, 1hour, 4hour)

        Returns:
            SimpleDataFrame با ستون‌های: timestamp, open, high, low, close, volume
        """
        csv_file = self.data_dir / f"{timeframe}.csv"

        if not csv_file.exists():
            raise FileNotFoundError(f"فایل {csv_file} پیدا نشد!")

        print(f"\n📖 در حال خواندن {csv_file}...")

        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # تبدیل مقادیر به float
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
            timeframe: نام تایم‌فریم (برای نام‌گذاری فایل‌ها)
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
            print("⚠️  pandas موجود نیست - از نسخه ساده استفاده می‌شود")
            pandas_available = False
            return []

        # حلقه از start_from تا آخر
        for i in range(start_from, total_candles):
            # استخراج داده‌های تا کندل فعلی
            df_slice_simple = df.iloc[:i+1]

            # تبدیل به pandas DataFrame برای pattern detector
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

                print(f"\n🎯 الگوی Doji پیدا شد!")
                print(f"   📍 کندل {i}: {candle_info['timestamp']}")
                print(f"   💰 OHLC: O={candle_info['open']:.2f} H={candle_info['high']:.2f} "
                      f"L={candle_info['low']:.2f} C={candle_info['close']:.2f}")
                if pattern_info:
                    print(f"   ⭐ Confidence: {pattern_info.get('confidence', 0):.2%}")
                    print(f"   📊 Location: {pattern_info.get('location', 'current')}")
                    print(f"   🔍 Candles ago: {pattern_info.get('candles_ago', 0)}")

                # رسم نمودار
                self._plot_detection(
                    df=df,
                    detection_index=i,
                    lookback=lookback,
                    timeframe=timeframe,
                    pattern_info=pattern_info
                )

        print(f"\n📊 نتایج:")
        print(f"   🔍 تعداد کندل‌های بررسی شده: {total_candles - start_from}")
        print(f"   ✅ تعداد الگوهای پیدا شده: {len(detections)}")

        # ذخیره نتایج
        self.results.extend(detections)
        self._save_results(timeframe)

        return detections

    def _plot_detection(self, df, detection_index, lookback, timeframe, pattern_info):
        """
        رسم نمودار کندلی برای الگوی تشخیص داده شده

        Args:
            df: SimpleDataFrame کامل
            detection_index: ایندکس کندلی که الگو در آن تشخیص داده شد
            lookback: تعداد کندل‌های قبلی برای نمایش
            timeframe: تایم‌فریم
            pattern_info: اطلاعات الگو
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # استفاده از backend غیر GUI
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from datetime import datetime as dt

            # استخراج داده‌ها برای نمایش (50 کندل قبل + کندل فعلی)
            start_idx = max(0, detection_index - lookback)
            end_idx = detection_index + 1
            df_plot = df.iloc(slice(start_idx, end_idx))

            # ایجاد figure
            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

            # رسم کندل‌ها
            for idx in range(len(df_plot)):
                row = df_plot[idx]

                # تبدیل timestamp به شماره برای رسم
                x_pos = idx

                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                # رنگ کندل
                color = 'green' if close_price >= open_price else 'red'

                # رسم shadow (فتیله)
                ax.plot([x_pos, x_pos], [low_price, high_price],
                       color='black', linewidth=1)

                # رسم body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)

                # عرض body
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

            # دایره آبی روی کندل تشخیص داده شده
            ax.scatter([detection_position], [detection_candle['high']],
                      color='blue', s=200, marker='v', zorder=5,
                      label='🎯 Doji Pattern Detected')

            # تنظیمات محور X
            # نمایش برچسب‌های زمانی هر 10 کندل
            x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
            x_labels = [df_plot[i]['timestamp'] for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

            # برچسب‌ها
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price (USDT)', fontsize=12)
            ax.set_title(
                f'Doji Pattern Detection - BTC/USDT {timeframe}\n'
                f'Detected at: {detection_candle["timestamp"]} (Candle #{detection_index})',
                fontsize=14,
                fontweight='bold'
            )

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')

            # Legend
            ax.legend(loc='upper left', fontsize=10)

            # اطلاعات الگو در گوشه
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

            # ذخیره فایل
            timestamp_str = detection_candle["timestamp"].replace(' ', '_').replace(':', '')
            filename = f"doji_{timeframe}_candle_{detection_index}_{timestamp_str}.png"
            filepath = self.charts_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"   💾 نمودار ذخیره شد: {filename}")

        except Exception as e:
            print(f"   ❌ خطا در رسم نمودار: {e}")
            import traceback
            traceback.print_exc()

    def _save_results(self, timeframe):
        """ذخیره نتایج در فایل JSON"""
        results_file = self.output_dir / f'doji_detections_{timeframe}.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 نتایج در {results_file} ذخیره شد")


def main():
    """تابع اصلی"""

    print("="*80)
    print("🧪 تست الگوی Doji روی داده‌های تاریخی BTC/USDT")
    print("="*80)

    # ایجاد tester
    tester = DojiPatternTester()

    # انتخاب تایم‌فریم برای تست
    timeframe = '5min'

    print(f"\n📊 تایم‌فریم انتخاب شده: {timeframe}")

    # بارگذاری داده‌ها
    df = tester.load_csv(timeframe)

    # تست کندل به کندل
    # از کندل 100 شروع می‌کنیم و 50 کندل قبلی را نمایش می‌دهیم
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ خطا: {e}")
        import traceback
        traceback.print_exc()
