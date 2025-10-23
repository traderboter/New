"""
VolatilityAnalyzer - Market Volatility and Risk Assessment

Analyzes market volatility using ATR and Bollinger Bands, provides risk adjustment recommendations.

Uses indicators (pre-calculated by IndicatorCalculator):
- atr (Average True Range)
- bb_upper, bb_middle, bb_lower (Bollinger Bands)
- close price

Can read from context (context-aware):
- trend: To assess volatility in trend context
- volume: To validate volatility signals

Outputs to context:
- volatility: {
    'atr_value': float,
    'atr_percentile': float,
    'volatility_regime': 'low' | 'normal' | 'high',
    'bb_width': float,
    'bb_squeeze': bool,
    'bb_breakout': 'upper' | 'lower' | None,
    'risk_multiplier': float,
    'recommended_stop_atr': float,
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


class VolatilityAnalyzer(BaseAnalyzer):
    """
    Analyzes market volatility for risk management.
    
    Key features:
    1. ATR analysis and percentile calculation
    2. Bollinger Band analysis (width, squeeze, breakout)
    3. Volatility regime detection
    4. Risk multiplier calculation
    5. Stop loss recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VolatilityAnalyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get volatility-specific configuration
        vol_config = config.get('volatility', {})
        
        # ATR lookback for percentile calculation
        self.atr_lookback = vol_config.get('atr_lookback', 100)
        
        # Volatility thresholds (percentiles)
        self.low_vol_threshold = vol_config.get('low_vol_threshold', 30)
        self.high_vol_threshold = vol_config.get('high_vol_threshold', 70)
        
        # BB squeeze threshold
        self.squeeze_threshold = vol_config.get('squeeze_threshold', 0.02)
        
        # Default risk multipliers for different regimes
        self.risk_multipliers = vol_config.get('risk_multipliers', {
            'low': 1.5,
            'normal': 1.0,
            'high': 0.6
        })
        
        # Enable/disable
        self.enabled = config.get('analyzers', {}).get('volatility', {}).get('enabled', True)
        
        logger.info("VolatilityAnalyzer initialized successfully")
    
    def analyze(self, context: AnalysisContext) -> None:
        """
        Main analysis method - analyzes volatility.
        
        Args:
            context: AnalysisContext with pre-calculated indicators
        """
        # 1. Check if enabled
        if not self._check_enabled():
            logger.debug(f"VolatilityAnalyzer disabled for {context.symbol}")
            return
        
        # 2. Validate context
        if not self._validate_context(context):
            logger.warning(f"VolatilityAnalyzer: Invalid context for {context.symbol}")
            return
        
        try:
            # 3. Read data
            df = context.df
            
            # Ensure we have enough data
            if len(df) < 50:
                logger.warning(f"Insufficient data for VolatilityAnalyzer on {context.symbol}")
                context.add_result('volatility', {
                    'status': 'insufficient_data',
                    'volatility_regime': 'unknown'
                })
                return
            
            # Get current values
            current_atr = df['atr'].iloc[-1]
            current_close = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            # 4. Calculate ATR percentile
            atr_percentile = self._calculate_atr_percentile(df)
            
            # 5. Determine volatility regime
            volatility_regime = self._determine_volatility_regime(atr_percentile)
            
            # 6. Analyze Bollinger Bands
            bb_analysis = self._analyze_bollinger_bands(
                current_close,
                bb_upper,
                bb_middle,
                bb_lower,
                df
            )
            
            # 7. Calculate risk multiplier
            risk_multiplier = self._calculate_risk_multiplier(
                volatility_regime,
                bb_analysis
            )
            
            # 8. Calculate recommended stop loss (in ATR multiples)
            recommended_stop = self._calculate_recommended_stop(
                volatility_regime,
                current_atr
            )
            
            # 9. Context-aware adjustments
            trend_context = context.get_result('trend')
            volume_context = context.get_result('volume')
            
            adjusted_risk = self._adjust_risk_for_context(
                risk_multiplier,
                trend_context,
                volume_context,
                bb_analysis
            )
            
            # 10. Calculate confidence
            confidence = self._calculate_confidence(
                atr_percentile,
                bb_analysis,
                volatility_regime
            )
            
            # 11. Build result
            result = {
                'status': 'ok',
                'atr_value': round(current_atr, 2),
                'atr_percentile': round(atr_percentile, 2),
                'volatility_regime': volatility_regime,
                'bb_width': bb_analysis['width'],
                'bb_width_percentile': bb_analysis['width_percentile'],
                'bb_squeeze': bb_analysis['squeeze'],
                'bb_breakout': bb_analysis['breakout'],
                'risk_multiplier': round(adjusted_risk, 2),
                'recommended_stop_atr': round(recommended_stop, 1),
                'confidence': confidence,
                'context_adjusted': trend_context is not None,
                'details': {
                    'bb_upper': round(bb_upper, 2),
                    'bb_middle': round(bb_middle, 2),
                    'bb_lower': round(bb_lower, 2),
                    'price_position': bb_analysis['position']
                }
            }
            
            # 12. Store in context
            context.add_result('volatility', result)
            
            logger.info(
                f"VolatilityAnalyzer completed for {context.symbol} {context.timeframe}: "
                f"regime={volatility_regime}, risk_mult={adjusted_risk:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error in VolatilityAnalyzer for {context.symbol}: {e}", exc_info=True)
            context.add_result('volatility', {
                'status': 'error',
                'volatility_regime': 'unknown',
                'error': str(e)
            })
    
    def _calculate_atr_percentile(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR percentile (where current ATR stands historically).
        
        Args:
            df: DataFrame with ATR column
            
        Returns:
            Percentile (0-100)
        """
        lookback = min(self.atr_lookback, len(df))
        
        historical_atr = df['atr'].tail(lookback).values
        current_atr = df['atr'].iloc[-1]
        
        # Calculate percentile
        percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
        
        return percentile
    
    def _determine_volatility_regime(self, atr_percentile: float) -> str:
        """
        Determine volatility regime based on ATR percentile.
        
        Args:
            atr_percentile: ATR percentile
            
        Returns:
            'low', 'normal', or 'high'
        """
        if atr_percentile < self.low_vol_threshold:
            return 'low'
        elif atr_percentile > self.high_vol_threshold:
            return 'high'
        else:
            return 'normal'
    
    def _analyze_bollinger_bands(
        self,
        close: float,
        bb_upper: float,
        bb_middle: float,
        bb_lower: float,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze Bollinger Bands.
        
        Args:
            close: Current close price
            bb_upper, bb_middle, bb_lower: BB values
            df: DataFrame for historical analysis
            
        Returns:
            BB analysis dict
        """
        # Calculate BB width
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Calculate BB width percentile
        lookback = min(self.atr_lookback, len(df))
        historical_widths = []
        
        for i in range(-lookback, 0):
            try:
                upper = df['bb_upper'].iloc[i]
                lower = df['bb_lower'].iloc[i]
                middle = df['bb_middle'].iloc[i]
                width = (upper - lower) / middle
                historical_widths.append(width)
            except:
                continue
        
        width_percentile = 50.0  # Default
        if historical_widths:
            width_percentile = (np.array(historical_widths) < bb_width).sum() / len(historical_widths) * 100
        
        # Detect BB squeeze
        is_squeeze = bb_width < self.squeeze_threshold
        
        # Detect breakout
        breakout = None
        prev_close = df['close'].iloc[-2]
        prev_upper = df['bb_upper'].iloc[-2]
        prev_lower = df['bb_lower'].iloc[-2]
        
        # Upper breakout
        if close > bb_upper and prev_close <= prev_upper:
            breakout = 'upper'
        # Lower breakout
        elif close < bb_lower and prev_close >= prev_lower:
            breakout = 'lower'
        
        # Price position within bands
        if close > bb_middle:
            if close > bb_upper:
                position = 'above_upper'
            else:
                position = 'upper_half'
        else:
            if close < bb_lower:
                position = 'below_lower'
            else:
                position = 'lower_half'
        
        return {
            'width': round(bb_width, 4),
            'width_percentile': round(width_percentile, 2),
            'squeeze': is_squeeze,
            'breakout': breakout,
            'position': position
        }
    
    def _calculate_risk_multiplier(
        self,
        volatility_regime: str,
        bb_analysis: Dict
    ) -> float:
        """
        Calculate risk multiplier based on volatility.
        
        Args:
            volatility_regime: Volatility regime
            bb_analysis: BB analysis results
            
        Returns:
            Risk multiplier
        """
        # Base multiplier from regime
        multiplier = self.risk_multipliers.get(volatility_regime, 1.0)
        
        # Adjust for BB squeeze (expansion expected)
        if bb_analysis['squeeze']:
            multiplier *= 0.8  # Reduce risk during squeeze
        
        # Adjust for BB breakout
        if bb_analysis['breakout']:
            multiplier *= 1.2  # Can increase risk on breakout
        
        return max(0.5, min(multiplier, 2.0))  # Clamp between 0.5 and 2.0
    
    def _calculate_recommended_stop(
        self,
        volatility_regime: str,
        current_atr: float
    ) -> float:
        """
        Calculate recommended stop loss in ATR multiples.
        
        Args:
            volatility_regime: Volatility regime
            current_atr: Current ATR value
            
        Returns:
            Recommended stop (ATR multiples)
        """
        # Base ATR multiples for different regimes
        base_stops = {
            'low': 1.5,      # Tighter stops in low volatility
            'normal': 2.0,   # Standard stops
            'high': 3.0      # Wider stops in high volatility
        }
        
        return base_stops.get(volatility_regime, 2.0)
    
    def _adjust_risk_for_context(
        self,
        base_risk: float,
        trend_context: Optional[Dict],
        volume_context: Optional[Dict],
        bb_analysis: Dict
    ) -> float:
        """
        Adjust risk multiplier based on context (context-aware).
        
        Args:
            base_risk: Base risk multiplier
            trend_context: Trend analyzer results
            volume_context: Volume analyzer results
            bb_analysis: BB analysis
            
        Returns:
            Adjusted risk multiplier
        """
        adjusted = base_risk
        
        # Strong trend allows higher risk
        if trend_context:
            trend_strength = abs(trend_context.get('strength', 0))
            if trend_strength >= 3:
                adjusted *= 1.2
            elif trend_strength <= 1:
                adjusted *= 0.9
        
        # Volume confirmation allows higher risk
        if volume_context and volume_context.get('is_confirmed'):
            adjusted *= 1.1
        
        # Price near BB edges - reduce risk
        if bb_analysis['position'] in ['above_upper', 'below_lower']:
            adjusted *= 0.8
        
        return max(0.5, min(adjusted, 2.0))
    
    def _calculate_confidence(
        self,
        atr_percentile: float,
        bb_analysis: Dict,
        volatility_regime: str
    ) -> float:
        """
        Calculate confidence in volatility assessment.
        
        Args:
            atr_percentile: ATR percentile
            bb_analysis: BB analysis
            volatility_regime: Volatility regime
            
        Returns:
            Confidence (0-1)
        """
        confidence = 0.6  # Base confidence
        
        # Clear regime increases confidence
        if volatility_regime in ['low', 'high']:
            if atr_percentile < 20 or atr_percentile > 80:
                confidence += 0.2
        
        # BB squeeze increases confidence (clear signal)
        if bb_analysis['squeeze']:
            confidence += 0.1
        
        # BB breakout increases confidence
        if bb_analysis['breakout']:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        """Validate required columns"""
        required = ['atr', 'bb_upper', 'bb_middle', 'bb_lower', 'close']
        
        df = context.df
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        return True
