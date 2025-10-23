"""
HTFAnalyzer - Higher Timeframe Analysis

Analyzes higher timeframe structure for multi-timeframe confirmation.

Note: This analyzer requires HTF data to be passed in context metadata.
If HTF data is not available, it will skip analysis.

Uses indicators:
- HTF OHLC data (from context.metadata)
- HTF trend structure

Outputs to context:
- htf: {
    'htf_trend': 'bullish' | 'bearish' | 'neutral',
    'htf_structure': 'higher_highs' | 'lower_lows' | 'ranging',
    'alignment': bool,
    'htf_support': float | None,
    'htf_resistance': float | None,
    'structure_shift': bool,
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


class HTFAnalyzer(BaseAnalyzer):
    """Analyzes higher timeframe structure."""
    
    # Timeframe hierarchy (in minutes)
    TF_HIERARCHY = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240,
        '1d': 1440, '1w': 10080
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        htf_config = config.get('htf', {})
        self.lookback = htf_config.get('lookback', 50)
        
        self.enabled = config.get('analyzers', {}).get('htf', {}).get('enabled', True)
        
        logger.info("HTFAnalyzer initialized")
    
    def analyze(self, context: AnalysisContext) -> None:
        """Main analysis method."""
        if not self._check_enabled():
            return
        
        try:
            # Check if HTF data is available in metadata
            htf_data = context.metadata.get('htf_data')
            current_tf = context.timeframe
            
            if not htf_data:
                logger.debug(f"No HTF data available for {context.symbol}")
                context.add_result('htf', {
                    'status': 'no_htf_data',
                    'htf_trend': 'unknown'
                })
                return
            
            # Get higher timeframe
            htf = self._get_higher_timeframe(current_tf)
            
            if htf not in htf_data:
                logger.debug(f"HTF {htf} not available")
                context.add_result('htf', {
                    'status': 'htf_not_available',
                    'htf_trend': 'unknown'
                })
                return
            
            htf_df = htf_data[htf]
            
            if len(htf_df) < 20:
                context.add_result('htf', {
                    'status': 'insufficient_htf_data',
                    'htf_trend': 'unknown'
                })
                return
            
            # Analyze HTF trend
            htf_trend = self._analyze_htf_trend(htf_df)
            
            # Analyze HTF structure
            htf_structure = self._analyze_structure(htf_df)
            
            # Find HTF support/resistance
            htf_support, htf_resistance = self._find_htf_levels(htf_df)
            
            # Check alignment with current timeframe
            current_trend = context.get_result('trend')
            alignment = self._check_alignment(htf_trend, current_trend)
            
            # Detect structure shift
            structure_shift = self._detect_structure_shift(htf_df)
            
            result = {
                'status': 'ok',
                'htf_timeframe': htf,
                'htf_trend': htf_trend,
                'htf_structure': htf_structure,
                'alignment': alignment,
                'htf_support': htf_support,
                'htf_resistance': htf_resistance,
                'structure_shift': structure_shift,
                'confidence': 0.7 if alignment else 0.5
            }
            
            context.add_result('htf', result)
            
            logger.info(f"HTFAnalyzer: {htf} trend={htf_trend} for {context.symbol}")
            
        except Exception as e:
            logger.error(f"Error in HTFAnalyzer: {e}", exc_info=True)
            context.add_result('htf', {
                'status': 'error',
                'htf_trend': 'unknown',
                'error': str(e)
            })
    
    def _get_higher_timeframe(self, current_tf: str) -> str:
        """Get the next higher timeframe."""
        current_minutes = self.TF_HIERARCHY.get(current_tf, 60)
        
        # Find next higher timeframe
        higher_tfs = [tf for tf, minutes in self.TF_HIERARCHY.items() 
                      if minutes > current_minutes]
        
        if higher_tfs:
            return min(higher_tfs, key=lambda tf: self.TF_HIERARCHY[tf])
        
        return '1d'  # Default to daily
    
    def _analyze_htf_trend(self, htf_df: pd.DataFrame) -> str:
        """Analyze trend on higher timeframe."""
        try:
            # Simple EMA-based trend
            close = htf_df['close'].values
            
            if len(close) < 50:
                return 'neutral'
            
            # Calculate EMAs
            ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().iloc[-1]
            ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1]
            
            current_price = close[-1]
            
            if current_price > ema_20 > ema_50:
                return 'bullish'
            elif current_price < ema_20 < ema_50:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.debug(f"HTF trend analysis failed: {e}")
            return 'neutral'
    
    def _analyze_structure(self, htf_df: pd.DataFrame) -> str:
        """Analyze market structure."""
        try:
            highs = htf_df['high'].tail(10).values
            lows = htf_df['low'].tail(10).values
            
            # Check for higher highs and higher lows
            recent_highs = highs[-3:]
            recent_lows = lows[-3:]
            
            higher_highs = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
            higher_lows = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
            
            if higher_highs and higher_lows:
                return 'higher_highs'
            
            lower_highs = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
            lower_lows = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
            
            if lower_highs and lower_lows:
                return 'lower_lows'
            
            return 'ranging'
            
        except Exception as e:
            logger.debug(f"Structure analysis failed: {e}")
            return 'unknown'
    
    def _find_htf_levels(self, htf_df: pd.DataFrame) -> tuple:
        """Find HTF support and resistance."""
        try:
            lookback = min(self.lookback, len(htf_df))
            recent = htf_df.tail(lookback)
            
            current_price = htf_df['close'].iloc[-1]
            
            # Find recent swing lows (support)
            lows = recent['low'].values
            support = None
            for low in sorted(lows)[:5]:
                if low < current_price:
                    support = low
                    break
            
            # Find recent swing highs (resistance)
            highs = recent['high'].values
            resistance = None
            for high in sorted(highs, reverse=True)[:5]:
                if high > current_price:
                    resistance = high
                    break
            
            return support, resistance
            
        except Exception as e:
            logger.debug(f"HTF level finding failed: {e}")
            return None, None
    
    def _check_alignment(self, htf_trend: str, current_trend: Optional[Dict]) -> bool:
        """Check if current trend aligns with HTF."""
        if not current_trend or htf_trend == 'neutral':
            return False
        
        current_direction = current_trend.get('direction', 'neutral')
        
        return htf_trend == current_direction
    
    def _detect_structure_shift(self, htf_df: pd.DataFrame) -> bool:
        """Detect if structure has shifted recently."""
        try:
            if len(htf_df) < 10:
                return False
            
            recent_highs = htf_df['high'].tail(5).values
            recent_lows = htf_df['low'].tail(5).values
            
            # Check if recent price broke previous structure
            prev_high = htf_df['high'].iloc[-6]
            prev_low = htf_df['low'].iloc[-6]
            
            current_high = recent_highs[-1]
            current_low = recent_lows[-1]
            
            # Break of structure
            if current_high > prev_high * 1.02 or current_low < prev_low * 0.98:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Structure shift detection failed: {e}")
            return False
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        return True  # HTF data is optional
