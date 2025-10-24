"""
Pattern Testing Framework - تست تک‌تک الگوها روی داده‌های واقعی

این اسکریپت یک الگوی خاص را روی داده‌های تاریخی تست می‌کند و
نتایج را به صورت دقیق و قابل بررسی نمایش می‌دهد.

استفاده:
    python test_pattern.py --pattern doji --data-dir historical/BTC-USDT
"""

TEST_PATTERN_VERSION = "1.2.0"
TEST_PATTERN_DATE = "2025-10-24"

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

    def __init__(self, data_dir: str, pattern_name: str, threshold: float = None):
        """
        مقداردهی اولیه

        Args:
            data_dir: مسیر دایرکتوری داده‌ها (مثلا: historical/BTC-USDT)
            pattern_name: نام الگو (مثلا: doji, hammer)
            threshold: آستانه body_ratio برای الگوی Doji (پیش‌فرض: 0.10)
        """
        self.data_dir = Path(data_dir)
        self.pattern_name = pattern_name.lower()
        self.threshold = threshold

        # تایم‌فریم‌های موجود
        self.timeframes = ['5m', '15m', '1h', '4h']

        # Mapping بین نام تایم‌فریم و نام فایل CSV
        # این mapping با انواع مختلف نام‌گذاری فایل‌ها سازگار است
        self.timeframe_to_filename = {
            '5m': ['5m.csv', '5min.csv'],
            '15m': ['15m.csv', '15min.csv'],
            '1h': ['1h.csv', '1hour.csv'],
            '4h': ['4h.csv', '4hour.csv']
        }

        # نتایج
        self.results = {}

        logger.info(f"PatternTester initialized for pattern: {pattern_name}")
        logger.info(f"Data directory: {data_dir}")
        if threshold is not None:
            logger.info(f"Doji threshold: {threshold}")

    def load_data(self, timeframe: str) -> pd.DataFrame:
        """
        لود داده‌های یک تایم‌فریم

        Args:
            timeframe: تایم‌فریم (5m, 15m, 1h, 4h)

        Returns:
            DataFrame با داده‌های OHLCV
        """
        # سعی می‌کنیم فایل را با نام‌های مختلف پیدا کنیم
        possible_filenames = self.timeframe_to_filename.get(timeframe, [f"{timeframe}.csv"])

        csv_file = None
        for filename in possible_filenames:
            file_path = self.data_dir / filename
            if file_path.exists():
                csv_file = file_path
                break

        if csv_file is None:
            logger.warning(f"File not found for timeframe {timeframe}. Tried: {possible_filenames}")
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

            # اگر الگو Doji است و threshold مشخص شده، آن را بفرست
            if self.pattern_name == 'doji' and self.threshold is not None:
                pattern_instance = pattern_class(body_ratio_threshold=self.threshold)
                orchestrator.register_pattern(pattern_instance)
                print(f"✓ Pattern registered: {pattern_class.__name__} (threshold={self.threshold})")
            else:
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

            # برای تست کامل، همه کندل‌ها را اسکن می‌کنیم
            # نه فقط آخرین کندل!
            target_detections = []

            # حداقل window size برای pattern detection (برای اکثر الگوها)
            min_window = 50

            # Loop روی همه کندل‌ها (با پیشرفت هر 1000 کندل)
            total_candles = len(df)
            progress_step = max(1000, total_candles // 20)  # حداقل 20 گام

            for i in range(min_window, total_candles):
                # نمایش پیشرفت
                if i % progress_step == 0:
                    progress = (i / total_candles) * 100
                    print(f"  Progress: {progress:.1f}% ({i}/{total_candles})", end='\r')

                # Window از داده‌ها (از شروع تا کندل فعلی)
                window_df = df.iloc[:i+1].copy()

                # تشخیص الگو در این window
                detections = orchestrator.detect_all_patterns(
                    df=window_df,
                    timeframe=timeframe,
                    context={}
                )

                # اگر الگویی یافت شد، آن را ذخیره کن
                for d in detections:
                    if self.pattern_name in d['name'].lower():
                        # اضافه کردن index کندل برای reference
                        d['detected_at_index'] = i
                        target_detections.append(d)

            print(f"\n✓ Found {len(target_detections)} {self.pattern_name} patterns")

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

            # نمایش quality metrics برای Doji و Hammer
            if 'metadata' in detection and self.pattern_name in ['doji', 'hammer']:
                meta = detection['metadata']
                print(f"\n     📈 Quality Metrics:")
                print(f"        Quality Score:    {meta.get('quality_score', 0):.2f}/100")
                print(f"        Overall Quality:  {meta.get('overall_quality', 0):.2f}/100")

                if self.pattern_name == 'doji':
                    print(f"        Symmetry Score:   {meta.get('symmetry_score', 0):.2f}/100")
                    print(f"        Doji Type:        {meta.get('doji_type', 'Unknown')}")
                elif self.pattern_name == 'hammer':
                    print(f"        Lower Shadow:     {meta.get('lower_shadow_score', 0):.2f}/100")
                    print(f"        Upper Shadow:     {meta.get('upper_shadow_score', 0):.2f}/100")
                    print(f"        Body Position:    {meta.get('body_position_score', 0):.2f}/100")
                    print(f"        Context Score:    {meta.get('context_score', 0):.2f}/100")
                    print(f"        Hammer Type:      {meta.get('hammer_type', 'Unknown')}")
                    print(f"        After Downtrend:  {'Yes ✓' if meta.get('is_after_downtrend', False) else 'No ✗'}")

                print(f"\n     🔍 Technical Details:")
                if self.pattern_name == 'doji':
                    print(f"        Body Ratio:       {meta.get('body_ratio', 0):.4f} ({meta.get('body_ratio', 0)*100:.2f}%)")
                    print(f"        Upper Shadow:     {meta.get('upper_shadow_ratio', 0):.2%}")
                    print(f"        Lower Shadow:     {meta.get('lower_shadow_ratio', 0):.2%}")
                    print(f"        Threshold:        {meta.get('threshold', 0):.2f}")
                elif self.pattern_name == 'hammer':
                    print(f"        Lower Shadow Ratio: {meta.get('lower_shadow_ratio', 0):.2f}x body")
                    print(f"        Upper Shadow Ratio: {meta.get('upper_shadow_ratio', 0):.2f}x body")
                    print(f"        Body Position:      {meta.get('body_position', 0):.2%} from low")
                    if 'thresholds' in meta:
                        th = meta['thresholds']
                        print(f"        Thresholds:         lower>={th.get('min_lower_shadow_ratio', 0):.1f}x, "
                              f"upper<={th.get('max_upper_shadow_ratio', 0):.1f}x, "
                              f"pos>={th.get('min_body_position', 0):.1%}")
            elif 'metadata' in detection:
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
        print(f"📦 Version: {TEST_PATTERN_VERSION} ({TEST_PATTERN_DATE})")
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

            # Quality Statistics for Doji and Hammer
            if self.pattern_name in ['doji', 'hammer'] and total_detections > 0:
                self._print_quality_stats(results)

    def _print_quality_stats(self, results: dict):
        """
        نمایش آمار کیفیت برای الگوهای Doji و Hammer

        Args:
            results: دیکشنری نتایج
        """
        print(f"\n{'='*80}")
        print(f"📊 Quality Statistics - {self.pattern_name.upper()}")
        print(f"{'='*80}")

        all_qualities = []
        all_types = []

        for tf, result in results.items():
            if result['status'] == 'ok' and result['detections']:
                for detection in result['detections']:
                    if 'metadata' in detection:
                        meta = detection['metadata']
                        if 'overall_quality' in meta:
                            all_qualities.append(meta['overall_quality'])
                        # برای Doji: doji_type، برای Hammer: hammer_type
                        type_key = f'{self.pattern_name}_type'
                        if type_key in meta:
                            all_types.append(meta[type_key])

        if all_qualities:
            import numpy as np

            qualities = np.array(all_qualities)

            print(f"\nOverall Quality Distribution:")
            print(f"  Mean:     {qualities.mean():.2f}")
            print(f"  Median:   {np.median(qualities):.2f}")
            print(f"  Std Dev:  {qualities.std():.2f}")
            print(f"  Min:      {qualities.min():.2f}")
            print(f"  Max:      {qualities.max():.2f}")

            # Quality ranges
            print(f"\nQuality Ranges:")
            high_q = len(qualities[qualities >= 70])
            med_q = len(qualities[(qualities >= 40) & (qualities < 70)])
            low_q = len(qualities[qualities < 40])

            print(f"  High Quality (≥70):    {high_q:>5} ({high_q/len(qualities)*100:>5.1f}%)")
            print(f"  Medium Quality (40-69): {med_q:>5} ({med_q/len(qualities)*100:>5.1f}%)")
            print(f"  Low Quality (<40):      {low_q:>5} ({low_q/len(qualities)*100:>5.1f}%)")

        if all_types:
            from collections import Counter
            type_counts = Counter(all_types)

            print(f"\n{self.pattern_name.title()} Type Distribution:")
            for pattern_type, count in type_counts.most_common():
                percentage = count / len(all_types) * 100
                print(f"  {pattern_type:<15} {count:>5} ({percentage:>5.1f}%)")

        # Pattern-specific insights
        if self.pattern_name == 'hammer':
            self._print_hammer_specific_stats(results)

        print(f"\n💡 Trading Insights:")
        if all_qualities:
            avg_quality = qualities.mean()
            if avg_quality >= 60:
                print(f"  ✓ High average quality ({avg_quality:.1f}) - Strong signals")
            elif avg_quality >= 40:
                print(f"  ℹ️  Medium average quality ({avg_quality:.1f}) - Use with confirmation")
            else:
                print(f"  ⚠️  Low average quality ({avg_quality:.1f}) - Requires additional filters")

        print(f"\n📝 Filtering Recommendations:")
        print(f"  • For high-confidence trades: overall_quality >= 70")
        print(f"  • For moderate trades: overall_quality >= 50")
        print(f"  • Consider timeframe: Higher timeframes typically more reliable")

        if self.pattern_name == 'hammer':
            print(f"  • For Hammer: Prefer 'Perfect' or 'Strong' types after confirmed downtrend")

    def _print_hammer_specific_stats(self, results: dict):
        """نمایش آمار خاص Hammer."""
        downtrend_count = 0
        total_count = 0

        for tf, result in results.items():
            if result['status'] == 'ok' and result['detections']:
                for detection in result['detections']:
                    if 'metadata' in detection:
                        meta = detection['metadata']
                        total_count += 1
                        if meta.get('is_after_downtrend', False):
                            downtrend_count += 1

        if total_count > 0:
            print(f"\nContext Analysis:")
            downtrend_pct = (downtrend_count / total_count) * 100
            print(f"  After Downtrend:  {downtrend_count:>5} ({downtrend_pct:>5.1f}%)")
            print(f"  Other Context:    {total_count - downtrend_count:>5} ({100 - downtrend_pct:>5.1f}%)")


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
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Body ratio threshold for Doji pattern (default: 0.10). '
             'Examples: 0.05 (strict), 0.10 (standard), 0.15 (relaxed)'
    )

    args = parser.parse_args()

    try:
        tester = PatternTester(
            data_dir=args.data_dir,
            pattern_name=args.pattern,
            threshold=args.threshold
        )

        results = tester.run_all_timeframes()

        print(f"\n✅ Testing completed!")
        print(f"\nTo test another pattern, run:")
        print(f"  python test_pattern.py --pattern <pattern_name> --data-dir {args.data_dir}")
        if args.pattern.lower() == 'doji':
            print(f"\nFor Doji pattern, you can adjust threshold:")
            print(f"  python test_pattern.py --pattern doji --threshold 0.05  (strict)")
            print(f"  python test_pattern.py --pattern doji --threshold 0.10  (standard)")
            print(f"  python test_pattern.py --pattern doji --threshold 0.15  (relaxed)")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
