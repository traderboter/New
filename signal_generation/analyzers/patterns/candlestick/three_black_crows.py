"""
Three Black Crows Pattern Detector

Detects Three Black Crows candlestick pattern using TALib.
Three Black Crows is a strong bearish reversal pattern.
"""

import talib
import pandas as pd
import numpy as np
from typing import Dict, Any

from signal_generation.analyzers.patterns.base_pattern import BasePattern


class ThreeBlackCrowsPattern(BasePattern):
    """
    Three Black Crows candlestick pattern detector.

    Characteristics:
    - Strong bearish reversal pattern (3 candles)
    - Three consecutive long bearish candles
    - Each opens within previous body
    - Each closes progressively lower
    - Little to no lower shadows

    Strength: 3/3 (Strong)
    """

    def _get_pattern_name(self) -> str:
        return "Three Black Crows"

    def _get_pattern_type(self) -> str:
        return "candlestick"

    def _get_direction(self) -> str:
        return "bearish"

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
        """Detect Three Black Crows pattern using TALib."""
        if not self._validate_dataframe(df):
            return False

        if len(df) < 3:
            return False

        try:
            result = talib.CDL3BLACKCROWS(
                df[open_col].values,
                df[high_col].values,
                df[low_col].values,
                df[close_col].values
            )

            return result[-1] != 0

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get additional details about Three Black Crows detection."""
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
