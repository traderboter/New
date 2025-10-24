"""
Pattern Testing Framework - تست تک‌تک الگوها روی داده‌های واقعی

این اسکریپت یک الگوی خاص را روی داده‌های تاریخی تست می‌کند و
نتایج را به صورت دقیق و قابل بررسی نمایش می‌دهد.

استفاده:
    python test_pattern.py --pattern doji --data-dir historical/BTC-USDT
"""

import sys
import os

# Add project root to Python path
# __file__ is in: New/signal_generation/tests/test_pattern.py
# We need to add: New/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pattern detection
from signal_generation.context import AnalysisContext
from signal_generation.analyzers.patterns.pattern_orchestrator import PatternOrchestrator


class PatternTester:
    """
    تست‌کننده الگوها روی داده‌های واقعی

    قابلیت‌ها:
    - تست یک الگوی خاص
    - اجرا روی چند تایم‌فریم
    - نمایش نتایج با جزئیات
    - آمار دقیق
    """

    def __init__(self, data_dir: str, pattern_name: str):
        """
        مقداردهی اولیه

        Args:
            data_dir: مسیر دایرکتوری داده‌ها (مثلا: historical/BTC-USDT)
            pattern_name: نام الگو (مثلا: doji, hammer)
        """
        self.data_dir = Path(data_dir)
        self.pattern_name = pattern_name.lower()

        # تایم‌فریم‌های موجود
        self.timeframes = ['5m', '15m', '1h', '4h']

        # نتایج
        self.results = {}

        logger.info(f"PatternTester initialized for pattern: {pattern_name}")
        logger.info(f"Data directory: {data_dir}")

    def load_data(self, timeframe: str) -> pd.DataFrame:
        """
        لود داده‌های یک تایم‌فریم

        Args:
            timeframe: تایم‌فریم (5m, 15m, 1h, 4h)

        Returns:
            DataFrame با داده‌های OHLCV
        """
        csv_file = self.data_dir / f"{timeframe}.csv"

        if not csv_file.exists():
            logger.warning(f"File not found: {csv_file}")
            return None

        try:
            df = pd.read_csv(csv_file)

            # تبدیل timestamp به datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # چک کردن ستون‌های ضروری
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    logger.error(f"Missing column: {col}")
                    return None

            logger.info(f"Loaded {len(df)} candles from {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error loading data from {csv_file}: {e}")
            return None

    def test_pattern_on_timeframe(self, timeframe: str) -> dict:
        """
        تست الگو روی یک تایم‌فریم

        Args:
            timeframe: تایم‌فریم

        Returns:
            دیکشنری با نتایج
        """
        print(f"\n{'='*80}")
        print(f"🔍 Testing {self.pattern_name.upper()} on {timeframe}")
        print(f"{'='*80}")

        # لود داده
        df = self.load_data(timeframe)
        if df is None or len(df) < 50:
            return {
                'status': 'error',
                'message': 'Data not found or insufficient',
                'detections': []
            }

        print(f"✓ Loaded {len(df)} candles")
        print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # ایجاد orchestrator
        try:
            orchestrator = PatternOrchestrator({})

            # ثبت الگو
            pattern_class = self._get_pattern_class(self.pattern_name)
            if pattern_class is None:
                return {
                    'status': 'error',
                    'message': f'Pattern not found: {self.pattern_name}',
                    'detections': []
                }

            orchestrator.register_pattern(pattern_class)
            print(f"✓ Pattern registered: {pattern_class.__name__}")

        except Exception as e:
            logger.error(f"Error initializing orchestrator: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'detections': []
            }

        # تشخیص الگو
        try:
            print(f"\n🔎 Scanning for {self.pattern_name} patterns...")

            detections = orchestrator.detect_all_patterns(
                df=df,
                timeframe=timeframe,
                context={}
            )

            # فیلتر کردن فقط الگوی مورد نظر
            target_detections = [
                d for d in detections
                if self.pattern_name in d['name'].lower()
            ]

            print(f"✓ Found {len(target_detections)} {self.pattern_name} patterns")

            # نمایش جزئیات
            if target_detections:
                self._display_detections(target_detections, df, timeframe)
            else:
                print(f"  ℹ️  No {self.pattern_name} pattern detected in this timeframe")

            return {
                'status': 'ok',
                'timeframe': timeframe,
                'total_candles': len(df),
                'detections': target_detections,
                'detection_count': len(target_detections),
                'detection_rate': len(target_detections) / len(df) * 100
            }

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'detections': []
            }

    def _display_detections(self, detections: list, df: pd.DataFrame, timeframe: str):
        """
        نمایش جزئیات الگوهای تشخیص داده شده

        Args:
            detections: لیست الگوهای تشخیص داده شده
            df: DataFrame داده‌ها
            timeframe: تایم‌فریم
        """
        print(f"\n📊 Pattern Detections ({len(detections)}):")
        print(f"{'='*80}")

        for i, detection in enumerate(detections[:10], 1):  # نمایش 10 مورد اول
            print(f"\n  #{i}. {detection['name']}")
            print(f"     Direction: {detection['direction']}")
            print(f"     Strength: {detection['base_strength']}/3")
            print(f"     Confidence: {detection.get('confidence', 0):.2f}")

            # نمایش metadata اگر موجود باشد
            if 'metadata' in detection:
                print(f"     Metadata: {detection['metadata']}")

            # نمایش context کندل (5 کندل قبل و 2 کندل بعد)
            if 'detected_at_index' in detection:
                idx = detection['detected_at_index']
                self._show_candle_context(df, idx, timeframe)

        if len(detections) > 10:
            print(f"\n  ... and {len(detections) - 10} more detections")

    def _show_candle_context(self, df: pd.DataFrame, index: int, timeframe: str):
        """
        نمایش context کندل (کندل‌های قبل و بعد)

        Args:
            df: DataFrame
            index: اندیس کندل
            timeframe: تایم‌فریم
        """
        # محدوده نمایش
        start = max(0, index - 5)
        end = min(len(df), index + 3)

        context_df = df.iloc[start:end]

        print(f"\n     Context (index {index}):")
        print(f"     {'Date':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Pattern'}")
        print(f"     {'-'*75}")

        for i, row in context_df.iterrows():
            marker = " <<<" if i == index else ""
            date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M') if 'timestamp' in row else str(i)

            print(f"     {date_str:<20} "
                  f"{row['open']:<10.2f} "
                  f"{row['high']:<10.2f} "
                  f"{row['low']:<10.2f} "
                  f"{row['close']:<10.2f} "
                  f"{marker}")

    def _get_pattern_class(self, pattern_name: str):
        """
        دریافت کلاس الگو بر اساس نام

        Args:
            pattern_name: نام الگو

        Returns:
            کلاس الگو یا None
        """
        # Candlestick patterns
        from signal_generation.analyzers.patterns.candlestick import (
            HammerPattern,
            InvertedHammerPattern,
            EngulfingPattern,
            MorningStarPattern,
            PiercingLinePattern,
            ThreeWhiteSoldiersPattern,
            MorningDojiStarPattern,
            ShootingStarPattern,
            HangingManPattern,
            EveningStarPattern,
            DarkCloudCoverPattern,
            ThreeBlackCrowsPattern,
            EveningDojiStarPattern,
            DojiPattern,
            HaramiPattern,
            HaramiCrossPattern,
        )

        # Chart patterns
        from signal_generation.analyzers.patterns.chart import (
            DoubleTopBottomPattern,
            HeadShouldersPattern,
            TrianglePattern,
            WedgePattern,
        )

        pattern_map = {
            'hammer': HammerPattern,
            'inverted_hammer': InvertedHammerPattern,
            'inverted hammer': InvertedHammerPattern,
            'engulfing': EngulfingPattern,
            'morning_star': MorningStarPattern,
            'morning star': MorningStarPattern,
            'piercing_line': PiercingLinePattern,
            'piercing line': PiercingLinePattern,
            'three_white_soldiers': ThreeWhiteSoldiersPattern,
            'three white soldiers': ThreeWhiteSoldiersPattern,
            'morning_doji_star': MorningDojiStarPattern,
            'morning doji star': MorningDojiStarPattern,
            'shooting_star': ShootingStarPattern,
            'shooting star': ShootingStarPattern,
            'hanging_man': HangingManPattern,
            'hanging man': HangingManPattern,
            'evening_star': EveningStarPattern,
            'evening star': EveningStarPattern,
            'dark_cloud_cover': DarkCloudCoverPattern,
            'dark cloud cover': DarkCloudCoverPattern,
            'three_black_crows': ThreeBlackCrowsPattern,
            'three black crows': ThreeBlackCrowsPattern,
            'evening_doji_star': EveningDojiStarPattern,
            'evening doji star': EveningDojiStarPattern,
            'doji': DojiPattern,
            'harami': HaramiPattern,
            'harami_cross': HaramiCrossPattern,
            'harami cross': HaramiCrossPattern,
            'double_top_bottom': DoubleTopBottomPattern,
            'double top bottom': DoubleTopBottomPattern,
            'head_shoulders': HeadShouldersPattern,
            'head shoulders': HeadShouldersPattern,
            'triangle': TrianglePattern,
            'wedge': WedgePattern,
        }

        return pattern_map.get(pattern_name.lower())

    def run_all_timeframes(self):
        """
        اجرای تست روی همه تایم‌فریم‌ها
        """
        print(f"\n{'='*80}")
        print(f"🎯 Pattern Testing: {self.pattern_name.upper()}")
        print(f"{'='*80}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Timeframes: {', '.join(self.timeframes)}")

        results = {}

        for tf in self.timeframes:
            result = self.test_pattern_on_timeframe(tf)
            results[tf] = result

        # خلاصه نتایج
        self._print_summary(results)

        self.results = results
        return results

    def _print_summary(self, results: dict):
        """
        نمایش خلاصه نتایج

        Args:
            results: دیکشنری نتایج
        """
        print(f"\n{'='*80}")
        print(f"📋 Summary Report - {self.pattern_name.upper()}")
        print(f"{'='*80}")

        total_detections = 0
        total_candles = 0

        print(f"\n{'Timeframe':<12} {'Candles':<12} {'Detections':<15} {'Rate':<12} {'Status'}")
        print(f"{'-'*80}")

        for tf, result in results.items():
            if result['status'] == 'ok':
                candles = result['total_candles']
                detections = result['detection_count']
                rate = result['detection_rate']
                status = '✓ OK'

                total_detections += detections
                total_candles += candles

                print(f"{tf:<12} {candles:<12} {detections:<15} {rate:<11.3f}% {status}")
            else:
                print(f"{tf:<12} {'N/A':<12} {'N/A':<15} {'N/A':<12} ✗ Error")

        print(f"{'-'*80}")
        print(f"{'TOTAL':<12} {total_candles:<12} {total_detections:<15} "
              f"{total_detections/total_candles*100 if total_candles > 0 else 0:<11.3f}%")

        # تحلیل
        print(f"\n💡 Analysis:")
        if total_detections == 0:
            print(f"  ⚠️  No {self.pattern_name} patterns detected in any timeframe!")
            print(f"  Possible reasons:")
            print(f"    - Pattern is rare in this dataset")
            print(f"    - Pattern detection parameters too strict")
            print(f"    - Issue with pattern detection logic")
        elif total_detections < 10:
            print(f"  ℹ️  Low detection rate ({total_detections} total)")
            print(f"  This pattern appears to be rare in this dataset")
        else:
            print(f"  ✓ Pattern detected successfully across timeframes")
            print(f"  Total: {total_detections} detections in {total_candles} candles")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test a specific pattern on historical data'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        required=True,
        help='Pattern name (e.g., doji, hammer, engulfing)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='historical/BTC-USDT',
        help='Data directory path (default: historical/BTC-USDT)'
    )

    args = parser.parse_args()

    try:
        tester = PatternTester(
            data_dir=args.data_dir,
            pattern_name=args.pattern
        )

        results = tester.run_all_timeframes()

        print(f"\n✅ Testing completed!")
        print(f"\nTo test another pattern, run:")
        print(f"  python test_pattern.py --pattern <pattern_name> --data-dir {args.data_dir}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
