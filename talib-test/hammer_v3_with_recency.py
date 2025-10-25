"""
Hammer Pattern v3.0.0 - با سیستم Recency Scoring

این نسخه نمونه برای نشان دادن پیاده‌سازی recency scoring است.

تغییرات نسبت به v2.0.0:
- چک کردن N کندل آخر (lookback_window)
- امتیازدهی بر اساس تازگی (recency multipliers)
- metadata کامل‌تر برای debugging

نویسنده: Development Team
تاریخ: 2025-10-25
"""

HAMMER_PATTERN_VERSION = "3.0.0"

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Mock BasePattern for demonstration
class BasePattern:
    """Base pattern class (simplified for demo)"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame has required columns"""
        required = ['open', 'high', 'low', 'close']
        return all(col in df.columns for col in required) and len(df) > 0


class HammerPattern(BasePattern):
    """
    Hammer Pattern Detector v3.0.0 with Recency Scoring

    Features:
    - TA-Lib CDLHAMMER detection
    - Multi-candle lookback (checks last N candles)
    - Recency-based scoring
    - Detailed metadata

    Config Parameters:
    - lookback_window: چند کندل آخر را چک کنیم (default: 5)
    - recency_multipliers: ضرایب هر کندل (default: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    - min_candles: حداقل کندل برای detection (12)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Hammer detector with recency scoring"""
        super().__init__(config)

        # Pattern-specific config
        pattern_config = self.config.get('patterns', {}).get('hammer', {})

        # Recency scoring parameters
        self.lookback_window = pattern_config.get('lookback_window', 5)
        self.recency_multipliers = pattern_config.get(
            'recency_multipliers',
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # default decay
        )

        # Detection parameters
        self.min_candles = pattern_config.get('min_candles', 12)
        self.weight = pattern_config.get('weight', 3)

        # Version
        self.version = HAMMER_PATTERN_VERSION

        # Cache for last detection
        self._last_detection_candles_ago = None
        self._last_detection_index = None

    def detect(
        self,
        df: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> bool:
        """
        Detect Hammer pattern in last N candles.

        Strategy:
        1. Check minimum candles (12 for Hammer)
        2. Run TA-Lib on full data
        3. Check last N candles (lookback_window)
        4. Return True if found in any of them
        5. Store which candle for scoring

        Args:
            df: DataFrame with OHLC data
            open_col, high_col, low_col, close_col: Column names

        Returns:
            bool: True if Hammer detected in last N candles
        """
        if not self._validate_dataframe(df):
            return False

        # Reset cache
        self._last_detection_candles_ago = None
        self._last_detection_index = None

        # Check minimum candles
        if len(df) < self.min_candles:
            return False

        try:
            # Run TA-Lib on full data
            result = talib.CDLHAMMER(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            # Check last N candles
            # lookback_window = 5 → check last 5 candles
            lookback = min(self.lookback_window, len(result))

            for i in range(lookback):
                # Check from newest to oldest
                # i=0: last candle (result[-1])
                # i=1: second to last (result[-2])
                # etc.
                idx = -(i + 1)

                if result[idx] != 0:
                    # Pattern found!
                    self._last_detection_candles_ago = i
                    self._last_detection_index = len(df) + idx  # absolute index
                    return True

            # Not found in last N candles
            return False

        except Exception as e:
            print(f"Error in Hammer detection: {e}")
            return False

    def get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed information about the detection.

        Returns:
            {
                'location': 'current' or 'recent',
                'candles_ago': 0-5,
                'recency_multiplier': 0.5-1.0,
                'confidence': adjusted confidence,
                'metadata': {
                    'pattern': 'Hammer',
                    'version': '3.0.0',
                    'candle_details': {...},
                    'recency_info': {...}
                }
            }
        """
        if len(df) == 0:
            return self._default_details()

        # Get detection info
        candles_ago = self._last_detection_candles_ago
        if candles_ago is None:
            candles_ago = 0  # assume current if not set

        # Get recency multiplier
        if candles_ago < len(self.recency_multipliers):
            recency_multiplier = self.recency_multipliers[candles_ago]
        else:
            # Too old → minimal score
            recency_multiplier = 0.0

        # Get the candle where pattern was detected
        candle_idx = -(candles_ago + 1)
        candle = df.iloc[candle_idx]

        # Calculate candle details
        candle_details = self._analyze_candle(candle)

        # Base confidence (can be based on quality)
        base_confidence = 0.75  # Hammer is medium-strong

        # Adjust confidence with recency
        # If very recent → full confidence
        # If old → reduced confidence
        adjusted_confidence = min(base_confidence * recency_multiplier, 0.95)

        return {
            'location': 'current' if candles_ago == 0 else 'recent',
            'candles_ago': candles_ago,
            'recency_multiplier': recency_multiplier,
            'confidence': adjusted_confidence,
            'metadata': {
                'pattern': 'Hammer',
                'version': self.version,
                'weight': self.weight,
                'candle_details': candle_details,
                'recency_info': {
                    'candles_ago': candles_ago,
                    'lookback_window': self.lookback_window,
                    'multiplier': recency_multiplier,
                    'multipliers_config': self.recency_multipliers
                },
                'detection_index': self._last_detection_index,
                'dataframe_length': len(df)
            }
        }

    def _analyze_candle(self, candle: pd.Series) -> Dict[str, Any]:
        """Analyze candle characteristics"""

        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']

        # Calculate sizes
        body = abs(close - open_price)
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        full_range = high - low

        if full_range == 0:
            return {'error': 'zero range'}

        return {
            'open': float(open_price),
            'high': float(high),
            'low': float(low),
            'close': float(close),
            'body_size': float(body),
            'body_pct': float(body / full_range * 100),
            'upper_shadow': float(upper_shadow),
            'upper_shadow_pct': float(upper_shadow / full_range * 100),
            'lower_shadow': float(lower_shadow),
            'lower_shadow_pct': float(lower_shadow / full_range * 100),
            'full_range': float(full_range),
            'direction': 'bullish' if close > open_price else 'bearish' if close < open_price else 'doji'
        }

    def calculate_score(self, detection_details: Dict[str, Any]) -> float:
        """
        Calculate final score with recency multiplier.

        Score = weight * confidence * recency_multiplier

        Args:
            detection_details: من get_detection_details()

        Returns:
            float: Final score (0-10 معمولاً)
        """
        confidence = detection_details.get('confidence', 0)
        recency_multiplier = detection_details.get('recency_multiplier', 1.0)

        score = self.weight * confidence * recency_multiplier

        return round(score, 3)

    def _default_details(self) -> Dict[str, Any]:
        """Default details when detection fails"""
        return {
            'location': 'none',
            'candles_ago': None,
            'recency_multiplier': 0.0,
            'confidence': 0.0,
            'metadata': {}
        }


# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

def test_hammer_recency():
    """Test Hammer with recency scoring"""

    print("="*70)
    print("Testing Hammer v3.0.0 with Recency Scoring")
    print("="*70)

    # Load BTC data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    csv_path = Path(__file__).parent.parent / 'historical' / 'BTC-USDT' / '1hour.csv'
    df = pd.read_csv(csv_path)

    print(f"\n✅ Loaded {len(df)} BTC candles")

    # Config with recency scoring
    config = {
        'patterns': {
            'hammer': {
                'enabled': True,
                'weight': 3,
                'min_candles': 12,
                'lookback_window': 5,
                'recency_multipliers': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
            }
        }
    }

    # Create detector
    detector = HammerPattern(config)

    print(f"\n📋 Config:")
    print(f"  Lookback window: {detector.lookback_window}")
    print(f"  Recency multipliers: {detector.recency_multipliers}")
    print(f"  Min candles: {detector.min_candles}")

    # Test with different slices to simulate different recencies
    test_cases = [
        (100, "Early data"),
        (1000, "Middle data"),
        (5000, "Later data"),
        (10000, "Recent data"),
        (len(df), "All data (latest)")
    ]

    print("\n" + "="*70)
    print("Testing Detection & Scoring")
    print("="*70)

    for end_idx, description in test_cases:
        df_slice = df.iloc[:end_idx]

        # Detect
        detected = detector.detect(df_slice)

        if detected:
            # Get details
            details = detector.get_detection_details(df_slice)

            # Calculate score
            score = detector.calculate_score(details)

            print(f"\n📊 {description} (index 0-{end_idx}):")
            print(f"  ✅ Detected: Yes")
            print(f"  📍 Candles ago: {details['candles_ago']}")
            print(f"  🔢 Recency multiplier: {details['recency_multiplier']}")
            print(f"  💯 Confidence: {details['confidence']:.3f}")
            print(f"  ⭐ Final score: {score:.3f}")

            # Candle details
            candle_info = details['metadata']['candle_details']
            print(f"  📊 Candle: body={candle_info['body_pct']:.1f}%, "
                  f"lower_shadow={candle_info['lower_shadow_pct']:.1f}%")
        else:
            print(f"\n📊 {description} (index 0-{end_idx}):")
            print(f"  ❌ Detected: No")

    print("\n" + "="*70)
    print("✅ Test completed")
    print("="*70)


if __name__ == '__main__':
    test_hammer_recency()
