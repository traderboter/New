"""
Three White Soldiers Pattern Detector

Detects Three White Soldiers candlestick pattern using TALib.
Three White Soldiers is a strong bullish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class ThreeWhiteSoldiersPattern(BasePattern):
    """
    Three White Soldiers candlestick pattern detector.

    Characteristics:
    - Strong bullish reversal pattern (3 candles)
    - Three consecutive long bullish candles
    - Each opens within previous body
    - Each closes progressively higher
    - Little to no upper shadows

    Strength: 3/3 (Strong)
    """

    def _get_pattern_name(self) -> str:
        return "Three White Soldiers"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "bullish"

    def _get_base_strength(self) -> int:
        return 3  # Strong pattern

    def detect(
        self,
        df: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> bool:
        """Detect Three White Soldiers pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        if len(df) < 3:
            return False

        try:
            result = talib.CDL3WHITESOLDIERS(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Three White Soldiers detection."""
        if len(df) < 3:
            return super()._get_detection_details(df)

        # Get last three candles
        candles = df.iloc[-3:].copy()

        # Calculate body sizes
        bodies = [abs(candles.iloc[i]['close'] - candles.iloc[i]['open']) for i in range(3)]
        avg_body = sum(bodies) / 3

        # Calculate consistency (similar body sizes)
        body_consistency = 1.0 - (max(bodies) - min(bodies)) / max(bodies) if max(bodies) > 0 else 0

        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': min(0.80 + (body_consistency / 5), 0.95),
            'metadata': {
                'body_sizes': [float(b) for b in bodies],
                'avg_body': float(avg_body),
                'body_consistency': float(body_consistency)
            }
        }
