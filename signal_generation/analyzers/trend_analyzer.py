"""
TrendAnalyzer - Market Trend Detection

This analyzer determines the market trend direction, strength, and phase
using Exponential Moving Averages (EMAs) and price position analysis.

Uses indicators (pre-calculated by IndicatorCalculator):
- ema_20, ema_50, ema_200
- sma_20, sma_50, sma_200
- close price

Outputs to context:
- trend: {
    'direction': 'bullish' | 'bearish' | 'sideways' | 'bullish_pullback' | 'bearish_pullback',
    'strength': float (-3 to 3),
    'phase': 'early' | 'developing' | 'mature' | 'pullback' | 'transition' | 'undefined',
    'ema_alignment': bool,
    'price_position': str,
    'ema_slopes': dict,
    'confidence': float (0-1)
  }
"""

from typing import Dict, Any
import logging
import pandas as pd

# Assuming BaseAnalyzer and AnalysisContext are in parent directory
try:
    from signal_generation.analyzers.base_analyzer import BaseAnalyzer
    from signal_generation.context import AnalysisContext
except ImportError:
    # Fallback for testing
    import sys
    sys.path.append('..')
    from analyzers.base_analyzer import BaseAnalyzer
    from context import AnalysisContext

logger = logging.getLogger(__name__)


class TrendAnalyzer(BaseAnalyzer):
    """
    Analyzes market trend using EMAs and price position.
    
    Determines:
    1. Trend Direction (bullish/bearish/sideways)
    2. Trend Strength (0-3 scale)
    3. Trend Phase (early/developing/mature/late)
    4. EMA Alignment
    5. Confidence level
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TrendAnalyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get trend-specific configuration
        trend_config = config.get('trend', {})
        
        # Minimum slope threshold for trend confirmation
        self.min_slope_threshold = trend_config.get('min_slope_threshold', 0.0001)
        
        # Lookback period for slope calculation
        self.slope_lookback = trend_config.get('slope_lookback', 5)
        
        # Enable/disable this analyzer
        self.enabled = config.get('analyzers', {}).get('trend', {}).get('enabled', True)
        
        logger.info("TrendAnalyzer initialized successfully")
    
    def analyze(self, context: AnalysisContext) -> None:
        """
        Main analysis method - determines market trend.
        
        Args:
            context: AnalysisContext with pre-calculated indicators
        """
        # 1. Check if enabled
        if not self._check_enabled():
            logger.debug(f"TrendAnalyzer disabled, skipping analysis for {context.symbol}")
            return
        
        # 2. Validate context has required data
        if not self._validate_context(context):
            logger.warning(f"TrendAnalyzer: Invalid context for {context.symbol}")
            return
        
        try:
            # 3. Read pre-calculated indicators from context.df
            df = context.df
            
            # Ensure we have enough data
            if len(df) < 200:
                logger.warning(
                    f"Insufficient data for TrendAnalyzer on {context.symbol} "
                    f"({len(df)} rows, need 200+)"
                )
                context.add_result('trend', {
                    'status': 'insufficient_data',
                    'direction': 'neutral',
                    'strength': 0,
                    'phase': 'undefined'
                })
                return
            
            # Get current values (last row)
            current_close = df['close'].iloc[-1]
            current_ema20 = df['ema_20'].iloc[-1]
            current_ema50 = df['ema_50'].iloc[-1]
            current_ema200 = df['ema_200'].iloc[-1]
            
            # 4. Calculate EMA slopes (rate of change)
            ema_slopes = self._calculate_ema_slopes(df)
            
            # 5. Determine EMA arrangement pattern
            ema_alignment = self._determine_ema_alignment(
                current_close, current_ema20, current_ema50, current_ema200
            )
            
            # 6. Detect trend direction and strength
            trend_result = self._detect_trend(
                current_close,
                current_ema20,
                current_ema50,
                current_ema200,
                ema_slopes,
                ema_alignment
            )
            
            # 7. Determine trend phase
            trend_phase = self._determine_trend_phase(
                trend_result['direction'],
                trend_result['strength'],
                ema_alignment,
                ema_slopes
            )
            
            # 8. Calculate confidence score
            confidence = self._calculate_confidence(
                trend_result['strength'],
                ema_alignment,
                ema_slopes
            )
            
            # 9. Build final result
            result = {
                'status': 'ok',
                'direction': trend_result['direction'],
                'strength': trend_result['strength'],
                'phase': trend_phase,
                'ema_alignment': ema_alignment,
                'price_position': self._get_price_position(current_close, current_ema20, current_ema50),
                'ema_slopes': ema_slopes,
                'confidence': confidence,
                'details': {
                    'close': round(current_close, 5),
                    'ema20': round(current_ema20, 5),
                    'ema50': round(current_ema50, 5),
                    'ema200': round(current_ema200, 5),
                }
            }
            
            # 10. Store result in context
            context.add_result('trend', result)
            
            logger.info(
                f"TrendAnalyzer completed for {context.symbol} {context.timeframe}: "
                f"{result['direction']} (strength: {result['strength']}, "
                f"phase: {result['phase']}, confidence: {confidence:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Error in TrendAnalyzer for {context.symbol}: {e}", exc_info=True)
            context.add_result('trend', {
                'status': 'error',
                'direction': 'neutral',
                'strength': 0,
                'phase': 'undefined',
                'error': str(e)
            })
    
    def _calculate_ema_slopes(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate rate of change (slope) for EMAs.
        
        Args:
            df: DataFrame with EMA columns
            
        Returns:
            Dictionary with slopes for each EMA
        """
        lookback = min(self.slope_lookback, len(df) - 1)
        
        if lookback < 2:
            return {'ema20': 0.0, 'ema50': 0.0, 'ema200': 0.0}
        
        # Calculate slopes (current - past) / past
        ema20_slope = (
            (df['ema_20'].iloc[-1] - df['ema_20'].iloc[-lookback]) 
            / df['ema_20'].iloc[-lookback]
        ) if df['ema_20'].iloc[-lookback] != 0 else 0.0
        
        ema50_slope = (
            (df['ema_50'].iloc[-1] - df['ema_50'].iloc[-lookback]) 
            / df['ema_50'].iloc[-lookback]
        ) if df['ema_50'].iloc[-lookback] != 0 else 0.0
        
        ema200_slope = (
            (df['ema_200'].iloc[-1] - df['ema_200'].iloc[-lookback]) 
            / df['ema_200'].iloc[-lookback]
        ) if df['ema_200'].iloc[-lookback] != 0 else 0.0
        
        return {
            'ema20': ema20_slope,
            'ema50': ema50_slope,
            'ema200': ema200_slope
        }
    
    def _determine_ema_alignment(
        self, 
        close: float, 
        ema20: float, 
        ema50: float, 
        ema200: float
    ) -> str:
        """
        Determine the arrangement/alignment pattern of EMAs.
        
        Args:
            close: Current close price
            ema20, ema50, ema200: Current EMA values
            
        Returns:
            String describing the alignment pattern
        """
        if ema20 > ema50 > ema200:
            return 'bullish_aligned'
        elif ema20 < ema50 < ema200:
            return 'bearish_aligned'
        elif ema20 > ema50 and ema50 < ema200:
            return 'potential_bullish_reversal'
        elif ema20 < ema50 and ema50 > ema200:
            return 'potential_bearish_reversal'
        elif ema20 > ema50 > ema200 and close < ema20:
            return 'bullish_pullback'
        elif ema20 < ema50 < ema200 and close > ema20:
            return 'bearish_pullback'
        else:
            return 'mixed'
    
    def _detect_trend(
        self,
        close: float,
        ema20: float,
        ema50: float,
        ema200: float,
        slopes: Dict[str, float],
        alignment: str
    ) -> Dict[str, Any]:
        """
        Detect trend direction and strength.
        
        Args:
            close: Current close price
            ema20, ema50, ema200: Current EMA values
            slopes: Dictionary of EMA slopes
            alignment: EMA alignment pattern
            
        Returns:
            Dictionary with 'direction' and 'strength'
        """
        direction = 'neutral'
        strength = 0
        
        ema20_slope = slopes['ema20']
        ema50_slope = slopes['ema50']
        
        # Strong Bullish (strength = 3)
        if (close > ema20 > ema50 > ema200 and 
            ema20_slope > self.min_slope_threshold and 
            ema50_slope > self.min_slope_threshold):
            direction = 'bullish'
            strength = 3
        
        # Moderate Bullish (strength = 2)
        elif (close > ema20 > ema50 and 
              ema20_slope > self.min_slope_threshold):
            direction = 'bullish'
            strength = 2
        
        # Weak Bullish (strength = 1)
        elif close > ema20 and ema20_slope > self.min_slope_threshold:
            direction = 'bullish'
            strength = 1
        
        # Strong Bearish (strength = -3)
        elif (close < ema20 < ema50 < ema200 and 
              ema20_slope < -self.min_slope_threshold and 
              ema50_slope < -self.min_slope_threshold):
            direction = 'bearish'
            strength = -3
        
        # Moderate Bearish (strength = -2)
        elif (close < ema20 < ema50 and 
              ema20_slope < -self.min_slope_threshold):
            direction = 'bearish'
            strength = -2
        
        # Weak Bearish (strength = -1)
        elif close < ema20 and ema20_slope < -self.min_slope_threshold:
            direction = 'bearish'
            strength = -1
        
        # Bullish Pullback
        elif close < ema50 and ema20 > ema50 and ema50_slope > 0:
            direction = 'bullish_pullback'
            strength = 1
        
        # Bearish Pullback
        elif close > ema50 and ema20 < ema50 and ema50_slope < 0:
            direction = 'bearish_pullback'
            strength = -1
        
        # Sideways / Neutral
        else:
            direction = 'sideways'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength
        }
    
    def _determine_trend_phase(
        self,
        direction: str,
        strength: int,
        alignment: str,
        slopes: Dict[str, float]
    ) -> str:
        """
        Determine the phase of the trend.
        
        Phases:
        - early: Trend just starting
        - developing: Trend gaining momentum
        - mature: Trend fully established
        - pullback: Temporary retracement in trend
        - transition: Changing from one trend to another
        - undefined: No clear trend
        
        Args:
            direction: Trend direction
            strength: Trend strength
            alignment: EMA alignment pattern
            slopes: EMA slopes
            
        Returns:
            String describing trend phase
        """
        if direction == 'sideways' or direction == 'neutral':
            return 'undefined'
        
        if 'pullback' in direction:
            return 'pullback'
        
        # Check if it's a mature trend (all EMAs aligned)
        if abs(strength) == 3:
            if 'aligned' in alignment:
                return 'mature'
            else:
                return 'developing'
        
        # Check if it's early trend (weak but starting)
        if abs(strength) == 1:
            return 'early'
        
        # Medium strength is developing
        if abs(strength) == 2:
            return 'developing'
        
        # Check for transition
        if 'reversal' in alignment:
            return 'transition'
        
        return 'undefined'
    
    def _get_price_position(self, close: float, ema20: float, ema50: float) -> str:
        """
        Describe price position relative to EMAs.
        
        Args:
            close: Current close price
            ema20, ema50: Current EMA values
            
        Returns:
            String describing price position
        """
        if close > ema20 and close > ema50:
            return 'above_both_emas'
        elif close > ema20 and close < ema50:
            return 'between_emas'
        elif close < ema20 and close < ema50:
            return 'below_both_emas'
        else:
            return 'at_ema'
    
    def _calculate_confidence(
        self,
        strength: int,
        alignment: str,
        slopes: Dict[str, float]
    ) -> float:
        """
        Calculate confidence score for the trend detection.
        
        Args:
            strength: Trend strength
            alignment: EMA alignment
            slopes: EMA slopes
            
        Returns:
            Confidence score (0 to 1)
        """
        confidence = 0.5  # Base confidence
        
        # Strong trend increases confidence
        if abs(strength) == 3:
            confidence += 0.3
        elif abs(strength) == 2:
            confidence += 0.2
        elif abs(strength) == 1:
            confidence += 0.1
        
        # Aligned EMAs increase confidence
        if 'aligned' in alignment:
            confidence += 0.2
        elif 'reversal' in alignment:
            confidence += 0.1
        
        # Consistent slopes increase confidence
        all_slopes = list(slopes.values())
        if all(s > 0 for s in all_slopes):  # All positive
            confidence += 0.1
        elif all(s < 0 for s in all_slopes):  # All negative
            confidence += 0.1
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        """
        Validate that context has required indicators.
        
        Args:
            context: AnalysisContext to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['close', 'ema_20', 'ema_50', 'ema_200']
        
        df = context.df
        
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        return True
    
    def _check_enabled(self) -> bool:
        """Check if this analyzer is enabled."""
        return self.enabled
