"""
تست الگوی Doji روی داده‌های تاریخی BTC

این اسکریپت الگوی Doji را کندل به کندل روی داده‌های تاریخی تست می‌کند
و برای هر الگوی پیدا شده، نمودار کندلی رسم می‌کند.

نویسنده: Claude Code
تاریخ: 2025-10-26
"""

import sys
import os

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.insert(0, '/home/user/New')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

# Import pattern detector
from signal_generation.analyzers.patterns.candlestick.doji import DojiPattern


class DojiPatternTester:
    """
    کلاس تست الگوی Doji

    این کلاس داده‌های تاریخی را می‌خواند و کندل به کندل الگوی Doji را
    تست می‌کند و نتایج را ذخیره می‌کند.
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
                chart_file.unlink()
            if chart_files:
                print(f"🗑️  {len(chart_files)} چارت قبلی پاک شد")

    def load_csv(self, timeframe='5min'):
        """
        بارگذاری داده‌های CSV

        Args:
            timeframe: تایم‌فریم مورد نظر (5min, 15min, 1hour, 4hour)

        Returns:
            DataFrame با ستون‌های: timestamp, open, high, low, close, volume
        """
        csv_file = self.data_dir / f"{timeframe}.csv"

        if not csv_file.exists():
            raise FileNotFoundError(f"فایل {csv_file} پیدا نشد!")

        print(f"\n📖 در حال خواندن {csv_file}...")
        df = pd.read_csv(csv_file)

        # تبدیل نام ستون‌ها به lowercase
        df.columns = df.columns.str.lower()

        # بررسی ستون‌های مورد نیاز
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"ستون‌های مورد نیاز پیدا نشد: {missing_cols}")

        # تبدیل timestamp به datetime
        if df['timestamp'].dtype == 'object' or df['timestamp'].dtype == 'int64':
            # اگر timestamp عدد است (Unix timestamp)
            if df['timestamp'].iloc[0] > 1000000000000:  # milliseconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:  # seconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # مرتب‌سازی بر اساس زمان
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"   ✅ {len(df)} کندل بارگذاری شد")
        print(f"   📅 از {df['timestamp'].iloc[0]} تا {df['timestamp'].iloc[-1]}")

        return df

    def test_candle_by_candle(self, df, timeframe='5min', lookback=50, start_from=100):
        """
        تست کندل به کندل

        Args:
            df: DataFrame داده‌ها
            timeframe: نام تایم‌فریم (برای نام‌گذاری فایل‌ها)
            lookback: تعداد کندل‌های قبلی برای نمایش (50)
            start_from: شروع تست از کندل چندم (100)
        """
        print(f"\n🔍 شروع تست کندل به کندل از کندل {start_from}...")

        total_candles = len(df)
        detections = []

        # حلقه از start_from تا آخر
        for i in range(start_from, total_candles):
            # استخراج داده‌های تا کندل فعلی
            df_slice = df.iloc[:i+1].copy()

            # تست pattern
            is_detected = self.doji_detector.detect(df_slice)

            if is_detected:
                # الگو پیدا شد!
                candle_info = df.iloc[i]
                detection_info = {
                    'index': i,
                    'timestamp': str(candle_info['timestamp']),
                    'open': float(candle_info['open']),
                    'high': float(candle_info['high']),
                    'low': float(candle_info['low']),
                    'close': float(candle_info['close']),
                    'volume': float(candle_info['volume']),
                    'timeframe': timeframe
                }

                detections.append(detection_info)

                # دریافت جزئیات الگو
                pattern_info = self.doji_detector.get_pattern_info(df_slice, timeframe)

                print(f"\n🎯 الگوی Doji پیدا شد!")
                print(f"   📍 کندل {i}: {candle_info['timestamp']}")
                print(f"   💰 OHLC: O={candle_info['open']:.2f} H={candle_info['high']:.2f} "
                      f"L={candle_info['low']:.2f} C={candle_info['close']:.2f}")
                if pattern_info:
                    print(f"   ⭐ Confidence: {pattern_info.get('confidence', 0):.2%}")
                    print(f"   📊 Location: {pattern_info.get('location', 'current')}")

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
            df: DataFrame کامل
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
            import matplotlib.dates as mdates

            # استخراج داده‌ها برای نمایش (50 کندل قبل + کندل فعلی)
            start_idx = max(0, detection_index - lookback)
            end_idx = detection_index + 1
            df_plot = df.iloc[start_idx:end_idx].copy()

            # ایجاد figure
            fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

            # رسم کندل‌ها
            for idx, row in df_plot.iterrows():
                timestamp = row['timestamp']
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']

                # رنگ کندل
                color = 'green' if close_price >= open_price else 'red'

                # رسم shadow (فتیله)
                ax.plot([timestamp, timestamp], [low_price, high_price],
                       color='black', linewidth=1)

                # رسم body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)

                # عرض body
                candle_width = pd.Timedelta(minutes=1) if timeframe == '5min' else \
                               pd.Timedelta(minutes=5) if timeframe == '15min' else \
                               pd.Timedelta(minutes=20) if timeframe == '1hour' else \
                               pd.Timedelta(hours=1)

                rect = Rectangle(
                    (timestamp - candle_width/2, body_bottom),
                    candle_width,
                    body_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
                ax.add_patch(rect)

            # مشخص کردن کندل تشخیص داده شده
            detection_candle = df.iloc[detection_index]
            detection_timestamp = detection_candle['timestamp']

            # دایره قرمز روی کندل تشخیص داده شده
            ax.scatter([detection_timestamp], [detection_candle['high']],
                      color='blue', s=200, marker='v', zorder=5,
                      label='🎯 Doji Pattern Detected')

            # تنظیمات محور X (زمان)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45, ha='right')

            # برچسب‌ها
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price (USDT)', fontsize=12)
            ax.set_title(
                f'Doji Pattern Detection - BTC/USDT {timeframe}\n'
                f'Detected at: {detection_timestamp} (Candle #{detection_index})',
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
                info_text += f"Location: {pattern_info.get('location', 'current')}"

                ax.text(
                    0.02, 0.98, info_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            # ذخیره فایل
            filename = f"doji_{timeframe}_candle_{detection_index}_{detection_timestamp.strftime('%Y%m%d_%H%M')}.png"
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
    # می‌توانید بین 5min, 15min, 1hour, 4hour انتخاب کنید
    timeframe = '5min'

    print(f"\n📊 تایم‌فریم انتخاب شده: {timeframe}")

    # بارگذاری داده‌ها
    df = tester.load_csv(timeframe)

    # تست کندل به کندل
    # از کندل 100 شروع می‌کنیم (تا 100 کندل اول برای warm-up)
    # و 50 کندل قبلی را برای هر الگو نمایش می‌دهیم
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
    main()
