"""
Base Pattern - Base class for all pattern detectors

This is the abstract base class that all pattern detectors must inherit from.
It provides a consistent interface for pattern detection and scoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BasePattern(ABC):
    """
    Abstract base class for all pattern detectors.

    All pattern classes (candlestick and chart) must inherit from this class
    and implement the abstract methods.

    Key features:
    1. Standardized interface for pattern detection
    2. Automatic pattern metadata
    3. Context-aware scoring support
    4. Consistent output format
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pattern detector.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.name = self._get_pattern_name()
        self.pattern_type = self._get_pattern_type()
        self.direction = self._get_direction()
        self.base_strength = self._get_base_strength()

        logger.debug(f"Pattern initialized: {self.name}")

    @abstractmethod
    def _get_pattern_name(self) -> str:
        """
        Get the pattern name.

        Returns:
            Pattern name (e.g., 'Hammer', 'Double Top')
        """
        pass

    @abstractmethod
    def _get_pattern_type(self) -> str:
        """
        Get the pattern type.

        Returns:
            'candlestick' or 'chart'
        """
        pass

    @abstractmethod
    def _get_direction(self) -> str:
        """
        Get the pattern direction.

        Returns:
            'bullish', 'bearish', 'both', or 'reversal'
        """
        pass

    @abstractmethod
    def _get_base_strength(self) -> int:
        """
        Get the base strength of the pattern.

        Returns:
            Strength value (1-3)
            1 = weak, 2 = medium, 3 = strong
        """
        pass

    @abstractmethod
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
        Detect if the pattern exists in the data.

        Args:
            df: DataFrame with OHLCV data
            open_col: Name of open price column
            high_col: Name of high price column
            low_col: Name of low price column
            close_col: Name of close price column
            volume_col: Name of volume column

        Returns:
            True if pattern detected, False otherwise
        """
        pass

    def get_pattern_info(
        self,
        df: pd.DataFrame,
        timeframe: str = '1h',
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete pattern information if detected.

        This is the main public method that should be called.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string (e.g., '1h', '4h')
            context: Optional context for context-aware scoring

        Returns:
            Dictionary with pattern info if detected, None otherwise
        """
        try:
            # Detect pattern
            is_detected = self.detect(df)

            if not is_detected:
                return None

            # Get detection details
            detection_details = self._get_detection_details(df)

            # Calculate base strength (might be dynamic)
            base_strength = self._calculate_dynamic_strength(df, detection_details)

            # Build pattern info
            pattern_info = {
                'name': self.name,
                'type': self.pattern_type,
                'direction': self._get_actual_direction(df, detection_details),
                'base_strength': base_strength,
                'adjusted_strength': base_strength,  # Will be adjusted by analyzer
                'timeframe': timeframe,
                'location': detection_details.get('location', 'current'),
                'candles_ago': detection_details.get('candles_ago', 0),
                'confidence': detection_details.get('confidence', 0.5),
                'metadata': detection_details.get('metadata', {})
            }

            # Apply context-aware adjustments if context provided
            if context:
                pattern_info = self._apply_context_adjustments(pattern_info, context)

            return pattern_info

        except Exception as e:
            logger.error(f"Error getting pattern info for {self.name}: {e}", exc_info=True)
            return None

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get additional details about the detection.

        Subclasses can override this to provide more information.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with detection details
        """
        return {
            'location': 'current',
            'candles_ago': 0,
            'confidence': 0.7,
            'metadata': {}
        }

    def _calculate_dynamic_strength(
        self,
        df: pd.DataFrame,
        detection_details: Dict[str, Any]
    ) -> float:
        """
        Calculate dynamic strength based on pattern quality.

        Subclasses can override this for more sophisticated strength calculation.

        Args:
            df: DataFrame with OHLCV data
            detection_details: Detection details

        Returns:
            Strength value (can be float for more precision)
        """
        # Default: return base strength
        return float(self.base_strength)

    def _get_actual_direction(
        self,
        df: pd.DataFrame,
        detection_details: Dict[str, Any]
    ) -> str:
        """
        Get the actual direction of the pattern.

        For patterns with direction='both', this determines the actual direction.

        Args:
            df: DataFrame with OHLCV data
            detection_details: Detection details

        Returns:
            'bullish' or 'bearish'
        """
        # Default: return configured direction
        return self.direction

    def _apply_context_adjustments(
        self,
        pattern_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply context-aware adjustments to pattern info.

        Subclasses can override for custom context handling.

        Args:
            pattern_info: Pattern information
            context: Context data (trend, momentum, volume, etc.)

        Returns:
            Updated pattern info
        """
        # Default: no adjustments (done by PatternAnalyzer)
        return pattern_info

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required data.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        if df is None or len(df) == 0:
            logger.warning(f"{self.name}: DataFrame is empty")
            return False

        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"{self.name}: Missing column {col}")
                return False

        return True

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} ({self.pattern_type}, {self.direction}, strength={self.base_strength})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type='{self.pattern_type}', direction='{self.direction}', "
            f"strength={self.base_strength})"
        )
