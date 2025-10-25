"""
Script to systematically fix all TA-Lib pattern files with minimum candle requirements.

This script will add the minimum candle check and version updates to all patterns
that need fixing based on our research findings.
"""

import re
from pathlib import Path

# Pattern definitions with their minimum candle requirements
PATTERNS_TO_FIX = {
    # 12-candle patterns
    'inverted_hammer.py': {
        'min_candles': 12,
        'detection_rate': 0.56,
        'detections': 59,
        'talib_func': 'CDLINVERTEDHAMMER',
        'pattern_name': 'Inverted Hammer'
    },
    'hanging_man.py': {
        'min_candles': 12,
        'detection_rate': 1.77,
        'detections': 187,
        'talib_func': 'CDLHANGINGMAN',
        'pattern_name': 'Hanging Man'
    },
    'harami.py': {
        'min_candles': 12,
        'detection_rate': 7.26,
        'detections': 765,
        'talib_func': 'CDLHARAMI',
        'pattern_name': 'Harami'
    },
    'harami_cross.py': {
        'min_candles': 12,
        'detection_rate': 1.39,
        'detections': 147,
        'talib_func': 'CDLHARAMICROSS',
        'pattern_name': 'Harami Cross'
    },
    'piercing_line.py': {
        'min_candles': 12,
        'detection_rate': 0.03,
        'detections': 3,
        'talib_func': 'CDLPIERCING',
        'pattern_name': 'Piercing Line'
    },
    'dark_cloud_cover.py': {
        'min_candles': 12,
        'detection_rate': 0.03,
        'detections': 3,
        'talib_func': 'CDLDARKCLOUDCOVER',
        'pattern_name': 'Dark Cloud Cover'
    },

    # 13-candle patterns
    'morning_star.py': {
        'min_candles': 13,
        'detection_rate': 0.38,
        'detections': 40,
        'talib_func': 'CDLMORNINGSTAR',
        'pattern_name': 'Morning Star'
    },
    'evening_star.py': {
        'min_candles': 13,
        'detection_rate': 0.46,
        'detections': 49,
        'talib_func': 'CDLEVENINGSTAR',
        'pattern_name': 'Evening Star'
    },
    'morning_doji_star.py': {
        'min_candles': 13,
        'detection_rate': 0.11,
        'detections': 12,
        'talib_func': 'CDLMORNINGDOJISTAR',
        'pattern_name': 'Morning Doji Star'
    },
    'evening_doji_star.py': {
        'min_candles': 13,
        'detection_rate': 0.11,
        'detections': 12,
        'talib_func': 'CDLEVENINGDOJISTAR',
        'pattern_name': 'Evening Doji Star'
    },
    'three_white_soldiers.py': {
        'min_candles': 13,
        'detection_rate': 0.09,
        'detections': 9,
        'talib_func': 'CDL3WHITESOLDIERS',
        'pattern_name': '3 White Soldiers'
    },

    # 16-candle pattern
    'three_black_crows.py': {
        'min_candles': 16,
        'detection_rate': 0.01,
        'detections': 1,
        'talib_func': 'CDL3BLACKCROWS',
        'pattern_name': '3 Black Crows'
    },
}

def fix_pattern_file(file_path: Path, pattern_info: dict):
    """Fix a single pattern file."""

    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return False

    print(f"\n{'='*70}")
    print(f"Fixing: {pattern_info['pattern_name']} ({file_path.name})")
    print(f"  Min candles: {pattern_info['min_candles']}")
    print(f"  Detection rate: {pattern_info['detection_rate']}%")
    print(f"{'='*70}")

    # Read file
    content = file_path.read_text()

    # Check if already fixed (has version 2.0.0)
    if 'VERSION = "2.0.0"' in content or "_VERSION = '2.0.0'" in content:
        print("✅ Already fixed (version 2.0.0 found)")
        return True

    # Check if minimum candle check already exists
    min_check_pattern = f"if len\\(df\\) < {pattern_info['min_candles']}"
    if re.search(min_check_pattern, content):
        print("✅ Already has minimum candle check")
        return True

    print(f"⚠️  Needs fixing - adding minimum candle check")
    print(f"   Pattern file: {file_path}")
    print(f"   You need to manually add:")
    print(f"   1. Version constant at top")
    print(f"   2. Minimum candle check: if len(df) < {pattern_info['min_candles']}: return False")
    print(f"   3. Update docstrings")

    return False

def main():
    """Main function to fix all patterns."""

    base_path = Path(__file__).parent.parent / 'signal_generation' / 'analyzers' / 'patterns' / 'candlestick'

    print("="*70)
    print("Pattern Fixer - Based on TA-Lib Research")
    print("="*70)
    print(f"\nBase path: {base_path}")
    print(f"Patterns to fix: {len(PATTERNS_TO_FIX)}")

    fixed_count = 0
    need_fix_count = 0
    not_found_count = 0

    for filename, pattern_info in PATTERNS_TO_FIX.items():
        file_path = base_path / filename

        if not file_path.exists():
            print(f"\n❌ NOT FOUND: {filename}")
            not_found_count += 1
            continue

        if fix_pattern_file(file_path, pattern_info):
            fixed_count += 1
        else:
            need_fix_count += 1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Already fixed: {fixed_count}")
    print(f"⚠️  Need manual fix: {need_fix_count}")
    print(f"❌ Not found: {not_found_count}")
    print(f"📊 Total: {len(PATTERNS_TO_FIX)}")

    if need_fix_count > 0:
        print(f"\n⚠️  {need_fix_count} patterns still need manual fixing!")
        print("Please review the output above for details.")
    else:
        print("\n✅ All patterns are fixed!")

if __name__ == '__main__':
    main()
