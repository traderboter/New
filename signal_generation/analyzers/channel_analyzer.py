"""
ChannelAnalyzer - Price Channel Detection

Detects price channels (ascending, descending, horizontal).

Uses indicators:
- OHLC data for channel detection
- Linear regression for trendlines

Outputs to context:
- channel: {
    'channel_type': 'ascending' | 'descending' | 'horizontal',
    'upper_bound': float,
    'lower_bound': float,
    'channel_width': float,
    'price_position': 'upper' | 'middle' | 'lower',
    'breakout': bool,
    'strength': float (0-3)
  }
"""

from typing import Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

from signal_generation.analyzers.base_analyzer import BaseAnalyzer
from signal_generation.context import AnalysisContext

logger = logging.getLogger(__name__)


class ChannelAnalyzer(BaseAnalyzer):
    """Analyzes price channels."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        channel_config = config.get('channel', {})
        self.lookback = channel_config.get('lookback', 50)
        self.enabled = config.get('analyzers', {}).get('channel', {}).get('enabled', True)
        
        logger.info("ChannelAnalyzer initialized")
    
    def analyze(self, context: AnalysisContext) -> None:
        """Main analysis method."""
        if not self._check_enabled():
            return
        
        if not self._validate_context(context):
            return
        
        try:
            df = context.df
            
            if len(df) < 30:
                context.add_result('channel', {
                    'status': 'insufficient_data',
                    'channel_type': 'unknown'
                })
                return
            
            lookback = min(self.lookback, len(df))
            recent_df = df.tail(lookback)
            
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            current_price = df['close'].iloc[-1]
            
            # Fit linear regression for highs and lows
            x = np.arange(len(highs))
            upper_slope, upper_intercept = np.polyfit(x, highs, 1)
            lower_slope, lower_intercept = np.polyfit(x, lows, 1)
            
            # Determine channel type
            if abs(upper_slope) < 0.0001 and abs(lower_slope) < 0.0001:
                channel_type = 'horizontal'
            elif upper_slope > 0.0001 and lower_slope > 0.0001:
                channel_type = 'ascending'
            elif upper_slope < -0.0001 and lower_slope < -0.0001:
                channel_type = 'descending'
            else:
                channel_type = 'irregular'
            
            # Calculate current channel bounds
            upper_bound = upper_slope * (len(x) - 1) + upper_intercept
            lower_bound = lower_slope * (len(x) - 1) + lower_intercept
            channel_width = upper_bound - lower_bound
            
            # Price position
            if current_price > upper_bound:
                position = 'above'
                breakout = True
            elif current_price < lower_bound:
                position = 'below'
                breakout = True
            else:
                mid = (upper_bound + lower_bound) / 2
                position = 'upper' if current_price > mid else 'lower'
                breakout = False
            
            # Calculate strength (based on how well prices fit channel)
            upper_fit = highs - (upper_slope * x + upper_intercept)
            lower_fit = lows - (lower_slope * x + lower_intercept)
            fit_error = np.mean(np.abs(upper_fit)) + np.mean(np.abs(lower_fit))
            
            strength = 3 if fit_error < channel_width * 0.1 else 2 if fit_error < channel_width * 0.2 else 1
            
            result = {
                'status': 'ok',
                'channel_type': channel_type,
                'upper_bound': round(upper_bound, 2),
                'lower_bound': round(lower_bound, 2),
                'channel_width': round(channel_width, 2),
                'price_position': position,
                'breakout': breakout,
                'strength': strength
            }
            
            context.add_result('channel', result)
            
            logger.info(f"ChannelAnalyzer: {channel_type} channel for {context.symbol}")
            
        except Exception as e:
            logger.error(f"Error in ChannelAnalyzer: {e}", exc_info=True)
            context.add_result('channel', {
                'status': 'error',
                'channel_type': 'unknown',
                'error': str(e)
            })
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        required = ['high', 'low', 'close']
        return all(col in context.df.columns for col in required)
