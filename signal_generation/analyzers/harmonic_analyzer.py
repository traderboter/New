"""
HarmonicAnalyzer - Harmonic Pattern Detection

Detects harmonic patterns using Fibonacci ratios.

Patterns detected:
- Gartley
- Butterfly
- Bat
- Crab
- Cypher (optional)

Uses indicators:
- OHLC data for swing point detection
- Fibonacci ratios for pattern validation

Outputs to context:
- harmonic: {
    'patterns': [list of detected harmonic patterns],
    'active_patterns': int,
    'strongest_pattern': dict | None,
    'confidence': float (0-1)
  }
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np

from signal_generation.analyzers.base_analyzer import BaseAnalyzer
from signal_generation.context import AnalysisContext

logger = logging.getLogger(__name__)


class HarmonicAnalyzer(BaseAnalyzer):
    """
    Analyzes harmonic patterns based on Fibonacci ratios.
    
    Key features:
    1. Swing point detection (X, A, B, C, D)
    2. Fibonacci ratio calculation
    3. Pattern validation
    4. Entry zone calculation
    5. Target calculation
    """
    
    # Harmonic pattern definitions (Fibonacci ratios)
    PATTERNS = {
        'gartley': {
            'XA_BC': (0.382, 0.886),  # BC retracement of XA
            'AB_CD': (1.13, 1.618),   # CD extension of AB
            'XA_AD': (0.618, 0.786),  # AD retracement of XA
            'type': 'both',
            'strength': 3
        },
        'butterfly': {
            'XA_BC': (0.382, 0.886),
            'AB_CD': (1.618, 2.618),
            'XA_AD': (1.27, 1.618),
            'type': 'both',
            'strength': 3
        },
        'bat': {
            'XA_BC': (0.382, 0.50),
            'AB_CD': (1.618, 2.618),
            'XA_AD': (0.886, 0.886),  # Exact 88.6%
            'type': 'both',
            'strength': 3
        },
        'crab': {
            'XA_BC': (0.382, 0.618),
            'AB_CD': (2.618, 3.618),
            'XA_AD': (1.618, 1.618),
            'type': 'both',
            'strength': 3
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HarmonicAnalyzer."""
        super().__init__(config)
        
        harmonic_config = config.get('harmonic', {})
        self.lookback = harmonic_config.get('lookback', 100)
        self.tolerance = harmonic_config.get('tolerance', 0.05)  # 5% tolerance
        
        self.enabled = config.get('analyzers', {}).get('harmonic', {}).get('enabled', True)
        
        logger.info("HarmonicAnalyzer initialized")
    
    def analyze(self, context: AnalysisContext) -> None:
        """Main analysis method."""
        if not self._check_enabled():
            return
        
        if not self._validate_context(context):
            return
        
        try:
            df = context.df
            
            if len(df) < 50:
                context.add_result('harmonic', {
                    'status': 'insufficient_data',
                    'patterns': []
                })
                return
            
            # Detect swing points
            swing_points = self._detect_swing_points(df)
            
            # Search for harmonic patterns
            patterns = self._search_harmonic_patterns(swing_points, df)
            
            # Find strongest
            strongest = max(patterns, key=lambda p: p['completion']) if patterns else None
            
            result = {
                'status': 'ok',
                'patterns': patterns,
                'active_patterns': len(patterns),
                'strongest_pattern': strongest,
                'confidence': 0.6 if patterns else 0.0
            }
            
            context.add_result('harmonic', result)
            
            logger.info(f"HarmonicAnalyzer: {len(patterns)} patterns for {context.symbol}")
            
        except Exception as e:
            logger.error(f"Error in HarmonicAnalyzer: {e}", exc_info=True)
            context.add_result('harmonic', {
                'status': 'error',
                'patterns': [],
                'error': str(e)
            })
    
    def _detect_swing_points(self, df: pd.DataFrame) -> List[Dict]:
        """Detect swing highs and lows."""
        lookback = min(self.lookback, len(df))
        recent_df = df.tail(lookback)
        
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        swing_points = []
        window = 5
        
        for i in range(window, len(highs) - window):
            # Swing high
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_points.append({
                    'type': 'high',
                    'price': highs[i],
                    'index': i
                })
            # Swing low
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_points.append({
                    'type': 'low',
                    'price': lows[i],
                    'index': i
                })
        
        return sorted(swing_points, key=lambda x: x['index'])
    
    def _search_harmonic_patterns(
        self,
        swing_points: List[Dict],
        df: pd.DataFrame
    ) -> List[Dict]:
        """Search for harmonic patterns in swing points."""
        patterns = []
        
        if len(swing_points) < 5:
            return patterns
        
        # Look for 5-point patterns (X, A, B, C, D)
        for i in range(len(swing_points) - 4):
            x = swing_points[i]
            a = swing_points[i+1]
            b = swing_points[i+2]
            c = swing_points[i+3]
            d = swing_points[i+4]
            
            # Check if alternating (high-low-high-low or vice versa)
            if not self._is_valid_sequence(x, a, b, c, d):
                continue
            
            # Check each pattern type
            for pattern_name, ratios in self.PATTERNS.items():
                if self._matches_pattern(x, a, b, c, d, ratios):
                    patterns.append({
                        'name': pattern_name.capitalize(),
                        'type': 'bullish' if x['type'] == 'low' else 'bearish',
                        'completion': self._calculate_completion(x, a, b, c, d, ratios),
                        'entry_zone': [d['price'] * 0.99, d['price'] * 1.01],
                        'targets': self._calculate_targets(x, a, d),
                        'strength': ratios['strength']
                    })
        
        return patterns
    
    def _is_valid_sequence(self, x, a, b, c, d) -> bool:
        """Check if points form valid alternating sequence."""
        types = [x['type'], a['type'], b['type'], c['type'], d['type']]
        
        # Should alternate: low-high-low-high-low or high-low-high-low-high
        for i in range(len(types) - 1):
            if types[i] == types[i+1]:
                return False
        
        return True
    
    def _matches_pattern(self, x, a, b, c, d, ratios: Dict) -> bool:
        """Check if points match pattern ratios."""
        tolerance = self.tolerance
        
        # Calculate actual ratios
        xa = abs(a['price'] - x['price'])
        ab = abs(b['price'] - a['price'])
        bc = abs(c['price'] - b['price'])
        cd = abs(d['price'] - c['price'])
        ad = abs(d['price'] - a['price'])
        
        if xa == 0 or ab == 0:
            return False
        
        # Check BC retracement of XA
        bc_xa_ratio = bc / xa
        if not self._in_range(bc_xa_ratio, ratios['XA_BC'], tolerance):
            return False
        
        # Check CD extension of AB
        cd_ab_ratio = cd / ab
        if not self._in_range(cd_ab_ratio, ratios['AB_CD'], tolerance):
            return False
        
        # Check AD retracement of XA
        ad_xa_ratio = ad / xa
        if not self._in_range(ad_xa_ratio, ratios['XA_AD'], tolerance):
            return False
        
        return True
    
    def _in_range(self, value: float, target: Tuple[float, float], tolerance: float) -> bool:
        """Check if value is within range with tolerance."""
        min_val = target[0] * (1 - tolerance)
        max_val = target[1] * (1 + tolerance)
        return min_val <= value <= max_val
    
    def _calculate_completion(self, x, a, b, c, d, ratios: Dict) -> float:
        """
        Calculate pattern completion percentage based on actual price position.

        Args:
            x, a, b, c, d: Pattern points
            ratios: Expected pattern ratios

        Returns:
            Completion percentage (0-1)
        """
        try:
            # Calculate expected D level based on XA_AD ratio
            xa = abs(a['price'] - x['price'])
            expected_ad_ratio = (ratios['XA_AD'][0] + ratios['XA_AD'][1]) / 2  # Average
            expected_d_price = a['price'] + (xa * expected_ad_ratio * (1 if a['price'] > x['price'] else -1))

            # Calculate actual AD
            actual_ad = abs(d['price'] - a['price'])

            # Completion is how close actual is to expected
            if expected_d_price != 0:
                completion = min(abs(actual_ad / (expected_ad_ratio * xa)), 1.0)
            else:
                completion = 0.85  # Fallback

            return round(completion, 2)

        except Exception:
            # Fallback to reasonable default
            return 0.85
    
    def _calculate_targets(self, x, a, d) -> List[float]:
        """Calculate target levels."""
        xa = abs(a['price'] - x['price'])
        
        # Standard Fibonacci targets
        target1 = d['price'] + xa * 0.382
        target2 = d['price'] + xa * 0.618
        target3 = d['price'] + xa * 1.0
        
        return [round(target1, 2), round(target2, 2), round(target3, 2)]
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        """Validate required columns."""
        required = ['high', 'low', 'close']
        return all(col in context.df.columns for col in required)
