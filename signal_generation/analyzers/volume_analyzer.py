"""
VolumeAnalyzer - Volume Analysis and Confirmation

Analyzes trading volume to confirm price movements and detect breakouts.

Uses indicators (pre-calculated by IndicatorCalculator):
- volume
- volume_sma (Simple Moving Average of volume)
- obv (On-Balance Volume)

Can read from context (context-aware):
- trend: To validate volume confirmation with trend
- momentum: To check volume support for momentum signals

Outputs to context:
- volume: {
    'is_confirmed': bool,
    'volume_ratio': float,
    'volume_trend': 'increasing' | 'decreasing' | 'stable',
    'breakout_volume': bool,
    'obv_trend': 'bullish' | 'bearish' | 'neutral',
    'strength': float (0-3),
    'confidence': float (0-1)
  }
"""

from typing import Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

from signal_generation.analyzers.base_analyzer import BaseAnalyzer
from signal_generation.context import AnalysisContext

logger = logging.getLogger(__name__)


class VolumeAnalyzer(BaseAnalyzer):
    """
    Analyzes trading volume for signal confirmation.
    
    Key features:
    1. Volume confirmation (current vs average)
    2. Volume trend analysis (increasing/decreasing)
    3. Breakout volume detection
    4. OBV (On-Balance Volume) trend analysis
    5. Context-aware validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VolumeAnalyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get volume-specific configuration
        vol_config = config.get('volume', {})
        
        # Volume confirmation threshold (ratio to average)
        self.volume_threshold = vol_config.get('volume_threshold', 1.5)
        
        # Breakout volume threshold
        self.breakout_threshold = vol_config.get('breakout_threshold', 2.0)
        
        # OBV trend lookback
        self.obv_lookback = vol_config.get('obv_lookback', 10)
        
        # Enable/disable
        self.enabled = config.get('analyzers', {}).get('volume', {}).get('enabled', True)
        
        logger.info("VolumeAnalyzer initialized successfully")
    
    def analyze(self, context: AnalysisContext) -> None:
        """
        Main analysis method - analyzes volume patterns.
        
        Args:
            context: AnalysisContext with pre-calculated indicators
        """
        # 1. Check if enabled
        if not self._check_enabled():
            logger.debug(f"VolumeAnalyzer disabled for {context.symbol}")
            return
        
        # 2. Validate context
        if not self._validate_context(context):
            logger.warning(f"VolumeAnalyzer: Invalid context for {context.symbol}")
            return
        
        try:
            # 3. Read pre-calculated indicators
            df = context.df
            
            # Ensure we have enough data
            if len(df) < 30:
                logger.warning(f"Insufficient data for VolumeAnalyzer on {context.symbol}")
                context.add_result('volume', {
                    'status': 'insufficient_data',
                    'is_confirmed': False,
                    'volume_ratio': 1.0
                })
                return
            
            # Get current values
            current_volume = df['volume'].iloc[-1]
            volume_sma = df['volume_sma'].iloc[-1]
            current_obv = df['obv'].iloc[-1]
            
            # 4. Calculate volume ratio
            volume_ratio = self._calculate_volume_ratio(current_volume, volume_sma)
            
            # 5. Analyze volume trend
            volume_trend = self._analyze_volume_trend(df)
            
            # 6. Detect breakout volume
            is_breakout = self._detect_breakout_volume(volume_ratio)
            
            # 7. Analyze OBV trend
            obv_analysis = self._analyze_obv(df)
            
            # 8. Check if volume confirms price movement
            is_confirmed = self._check_volume_confirmation(
                volume_ratio,
                volume_trend,
                obv_analysis
            )
            
            # 9. Context-aware validation (read trend/momentum if available)
            trend_context = context.get_result('trend')
            momentum_context = context.get_result('momentum')
            
            validation = self._validate_with_context(
                is_confirmed,
                volume_trend,
                obv_analysis,
                trend_context,
                momentum_context
            )
            
            # 10. Calculate strength
            strength = self._calculate_strength(
                volume_ratio,
                is_breakout,
                obv_analysis,
                validation
            )
            
            # 11. Calculate confidence
            confidence = self._calculate_confidence(
                is_confirmed,
                volume_trend,
                obv_analysis,
                validation
            )
            
            # 12. Build result
            result = {
                'status': 'ok',
                'is_confirmed': is_confirmed,
                'volume_ratio': round(volume_ratio, 2),
                'volume_trend': volume_trend,
                'breakout_volume': is_breakout,
                'obv_trend': obv_analysis['trend'],
                'strength': strength,
                'confidence': confidence,
                'context_validated': validation['validated'],
                'validation_details': validation,
                'details': {
                    'current_volume': round(current_volume, 2),
                    'volume_sma': round(volume_sma, 2),
                    'obv': round(current_obv, 2)
                }
            }
            
            # 13. Store in context
            context.add_result('volume', result)
            
            logger.info(
                f"VolumeAnalyzer completed for {context.symbol} {context.timeframe}: "
                f"confirmed={is_confirmed}, ratio={volume_ratio:.2f}, "
                f"trend={volume_trend}, confidence={confidence:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error in VolumeAnalyzer for {context.symbol}: {e}", exc_info=True)
            context.add_result('volume', {
                'status': 'error',
                'is_confirmed': False,
                'error': str(e)
            })
    
    def _calculate_volume_ratio(self, current: float, average: float) -> float:
        """
        Calculate ratio of current volume to average.
        
        Args:
            current: Current volume
            average: Average volume (SMA)
            
        Returns:
            Volume ratio
        """
        if average == 0 or pd.isna(average):
            return 1.0
        
        return current / average
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """
        Analyze volume trend (increasing/decreasing/stable).
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            'increasing', 'decreasing', or 'stable'
        """
        # Look at last 5 periods
        recent_volumes = df['volume'].tail(5).values
        
        if len(recent_volumes) < 5:
            return 'stable'
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_volumes))
        try:
            slope = np.polyfit(x, recent_volumes, 1)[0]
            
            # Normalize by average volume
            avg_volume = recent_volumes.mean()
            if avg_volume > 0:
                normalized_slope = slope / avg_volume
                
                if normalized_slope > 0.1:
                    return 'increasing'
                elif normalized_slope < -0.1:
                    return 'decreasing'
                else:
                    return 'stable'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _detect_breakout_volume(self, volume_ratio: float) -> bool:
        """
        Detect if current volume indicates a breakout.
        
        Args:
            volume_ratio: Ratio of current to average volume
            
        Returns:
            True if breakout volume detected
        """
        return volume_ratio >= self.breakout_threshold
    
    def _analyze_obv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze On-Balance Volume trend.
        
        Args:
            df: DataFrame with OBV column
            
        Returns:
            Dictionary with OBV analysis
        """
        lookback = min(self.obv_lookback, len(df))
        
        if lookback < 3:
            return {
                'trend': 'neutral',
                'slope': 0.0,
                'strength': 0
            }
        
        recent_obv = df['obv'].tail(lookback).values
        
        # Calculate OBV trend
        x = np.arange(len(recent_obv))
        try:
            slope = np.polyfit(x, recent_obv, 1)[0]

            # Normalize slope by average OBV value for meaningful strength calculation
            avg_obv = abs(np.mean(recent_obv))
            if avg_obv > 0:
                normalized_slope = abs(slope) / avg_obv
            else:
                normalized_slope = 0

            # Determine trend and strength based on normalized slope
            if slope > 0:
                trend = 'bullish'
                # Strength thresholds: 0.01 = weak, 0.05 = moderate, 0.1+ = strong
                if normalized_slope >= 0.1:
                    strength = 3
                elif normalized_slope >= 0.05:
                    strength = 2
                elif normalized_slope >= 0.01:
                    strength = 1
                else:
                    strength = 0
            elif slope < 0:
                trend = 'bearish'
                if normalized_slope >= 0.1:
                    strength = 3
                elif normalized_slope >= 0.05:
                    strength = 2
                elif normalized_slope >= 0.01:
                    strength = 1
                else:
                    strength = 0
            else:
                trend = 'neutral'
                strength = 0

            return {
                'trend': trend,
                'slope': slope,
                'strength': strength
            }
            
        except Exception:
            return {
                'trend': 'neutral',
                'slope': 0.0,
                'strength': 0
            }
    
    def _check_volume_confirmation(
        self,
        volume_ratio: float,
        volume_trend: str,
        obv_analysis: Dict
    ) -> bool:
        """
        Check if volume confirms the price movement.
        
        Args:
            volume_ratio: Current/average volume ratio
            volume_trend: Volume trend direction
            obv_analysis: OBV analysis results
            
        Returns:
            True if volume confirms
        """
        # Volume should be above threshold
        if volume_ratio < self.volume_threshold:
            return False
        
        # Volume should be increasing or at least stable
        if volume_trend == 'decreasing':
            return False
        
        # OBV should not be contradicting
        # (if OBV is strongly opposite, volume might not confirm)
        if obv_analysis['strength'] >= 2:
            # We'll handle this in context validation
            pass
        
        return True
    
    def _validate_with_context(
        self,
        is_confirmed: bool,
        volume_trend: str,
        obv_analysis: Dict,
        trend_context: Optional[Dict],
        momentum_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Validate volume analysis with trend and momentum context.
        
        Args:
            is_confirmed: Basic volume confirmation
            volume_trend: Volume trend
            obv_analysis: OBV analysis
            trend_context: Trend analyzer results
            momentum_context: Momentum analyzer results
            
        Returns:
            Validation results
        """
        validation = {
            'validated': is_confirmed,
            'trend_aligned': False,
            'momentum_aligned': False,
            'notes': []
        }
        
        # Check trend alignment
        if trend_context:
            trend_direction = trend_context.get('direction', 'neutral')
            obv_trend = obv_analysis['trend']
            
            if trend_direction == 'bullish' and obv_trend == 'bullish':
                validation['trend_aligned'] = True
                validation['validated'] = validation['validated'] and True
                validation['notes'].append('OBV confirms bullish trend')
            elif trend_direction == 'bearish' and obv_trend == 'bearish':
                validation['trend_aligned'] = True
                validation['validated'] = validation['validated'] and True
                validation['notes'].append('OBV confirms bearish trend')
            elif trend_direction in ['bullish', 'bearish'] and obv_trend != trend_direction:
                validation['notes'].append('OBV diverges from trend - potential reversal')
                # Don't fully invalidate, but note the divergence
        
        # Check momentum alignment
        if momentum_context:
            momentum_direction = momentum_context.get('direction', 'neutral')
            
            if volume_trend == 'increasing':
                if momentum_direction in ['bullish', 'bearish']:
                    validation['momentum_aligned'] = True
                    validation['notes'].append('Volume supports momentum move')
            elif volume_trend == 'decreasing':
                if momentum_direction != 'neutral':
                    validation['notes'].append('Weakening volume may signal momentum fade')
        
        return validation
    
    def _calculate_strength(
        self,
        volume_ratio: float,
        is_breakout: bool,
        obv_analysis: Dict,
        validation: Dict
    ) -> float:
        """
        Calculate volume strength score.
        
        Args:
            volume_ratio: Volume ratio
            is_breakout: Breakout flag
            obv_analysis: OBV analysis
            validation: Context validation
            
        Returns:
            Strength score (0-3)
        """
        strength = 0.0
        
        # Base strength from volume ratio
        if volume_ratio >= 3.0:
            strength += 2.0
        elif volume_ratio >= 2.0:
            strength += 1.5
        elif volume_ratio >= 1.5:
            strength += 1.0
        elif volume_ratio >= 1.2:
            strength += 0.5
        
        # Bonus for breakout
        if is_breakout:
            strength += 0.5
        
        # Bonus for OBV confirmation
        if obv_analysis['strength'] >= 2:
            strength += 0.5
        
        # Bonus for context validation
        if validation['validated']:
            if validation['trend_aligned']:
                strength += 0.3
            if validation['momentum_aligned']:
                strength += 0.2
        
        return min(strength, 3.0)
    
    def _calculate_confidence(
        self,
        is_confirmed: bool,
        volume_trend: str,
        obv_analysis: Dict,
        validation: Dict
    ) -> float:
        """
        Calculate confidence score.
        
        Args:
            is_confirmed: Volume confirmation
            volume_trend: Volume trend
            obv_analysis: OBV analysis
            validation: Context validation
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5
        
        # Volume confirmation adds confidence
        if is_confirmed:
            confidence += 0.2
        
        # Increasing volume adds confidence
        if volume_trend == 'increasing':
            confidence += 0.1
        
        # OBV confirmation adds confidence
        if obv_analysis['strength'] >= 2:
            confidence += 0.1
        
        # Context validation adds confidence
        if validation['validated']:
            confidence += 0.1
            if validation['trend_aligned']:
                confidence += 0.1
            if validation['momentum_aligned']:
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        """Validate required indicators"""
        required = ['volume', 'volume_sma', 'obv']
        
        df = context.df
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        return True
