"""
CyclicalAnalyzer - Cyclical Pattern and Cycle Detection

Detects recurring cycles and forecasts potential reversal points.

Uses indicators:
- close price for cycle detection
- Simple cycle detection algorithms

Outputs to context:
- cyclical: {
    'dominant_cycle': int (in periods),
    'cycle_phase': 'top' | 'bottom' | 'rising' | 'falling',
    'next_reversal_in': int (periods),
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


class CyclicalAnalyzer(BaseAnalyzer):
    """Analyzes cyclical patterns in price."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        cyclical_config = config.get('cyclical', {})
        self.lookback = cyclical_config.get('lookback', 200)
        self.min_cycle = cyclical_config.get('min_cycle', 10)
        self.max_cycle = cyclical_config.get('max_cycle', 100)
        
        self.enabled = config.get('analyzers', {}).get('cyclical', {}).get('enabled', True)
        
        logger.info("CyclicalAnalyzer initialized")
    
    def analyze(self, context: AnalysisContext) -> None:
        """Main analysis method."""
        if not self._check_enabled():
            return
        
        if not self._validate_context(context):
            return
        
        try:
            df = context.df
            
            if len(df) < 100:
                context.add_result('cyclical', {
                    'status': 'insufficient_data',
                    'dominant_cycle': None
                })
                return
            
            lookback = min(self.lookback, len(df))
            prices = df['close'].tail(lookback).values
            
            # Detect dominant cycle using autocorrelation
            dominant_cycle = self._detect_dominant_cycle(prices)
            
            # Determine current phase
            cycle_phase = self._determine_cycle_phase(prices, dominant_cycle)
            
            # Estimate next reversal
            next_reversal = self._estimate_next_reversal(dominant_cycle, cycle_phase)
            
            result = {
                'status': 'ok',
                'dominant_cycle': dominant_cycle,
                'cycle_phase': cycle_phase,
                'next_reversal_in': next_reversal,
                'confidence': 0.5 if dominant_cycle else 0.2
            }
            
            context.add_result('cyclical', result)
            
            logger.info(f"CyclicalAnalyzer: cycle={dominant_cycle} for {context.symbol}")
            
        except Exception as e:
            logger.error(f"Error in CyclicalAnalyzer: {e}", exc_info=True)
            context.add_result('cyclical', {
                'status': 'error',
                'dominant_cycle': None,
                'error': str(e)
            })
    
    def _detect_dominant_cycle(self, prices: np.ndarray) -> Optional[int]:
        """Detect dominant cycle using autocorrelation."""
        try:
            # Simple autocorrelation
            normalized = (prices - np.mean(prices)) / np.std(prices)
            
            autocorr = np.correlate(normalized, normalized, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(self.min_cycle, min(self.max_cycle, len(autocorr) - 1)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append((i, autocorr[i]))
            
            if peaks:
                # Return period with highest correlation
                dominant = max(peaks, key=lambda x: x[1])
                return dominant[0]
            
            return None
            
        except Exception as e:
            logger.debug(f"Cycle detection failed: {e}")
            return None
    
    def _determine_cycle_phase(self, prices: np.ndarray, cycle: Optional[int]) -> str:
        """Determine current phase of cycle."""
        if not cycle or len(prices) < cycle:
            return 'unknown'
        
        recent_prices = prices[-cycle:]
        current = prices[-1]
        
        max_price = np.max(recent_prices)
        min_price = np.min(recent_prices)
        
        # Determine phase based on position and recent movement
        if current >= max_price * 0.95:
            return 'top'
        elif current <= min_price * 1.05:
            return 'bottom'
        elif current > prices[-2]:
            return 'rising'
        else:
            return 'falling'
    
    def _estimate_next_reversal(self, cycle: Optional[int], phase: str) -> Optional[int]:
        """Estimate periods until next reversal."""
        if not cycle:
            return None
        
        # Simple estimate: half cycle from top/bottom
        if phase in ['top', 'bottom']:
            return cycle // 2
        elif phase in ['rising', 'falling']:
            return cycle // 4
        
        return None
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        return 'close' in context.df.columns
