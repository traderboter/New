"""
MomentumAnalyzer - Momentum Indicator Analysis

Analyzes momentum indicators (RSI, MACD, Stochastic) to detect:
- Overbought/oversold conditions
- Bullish/bearish divergences
- MACD crossovers
- Momentum strength and direction

Uses indicators (pre-calculated by IndicatorCalculator):
- rsi (Relative Strength Index)
- macd, macd_signal, macd_hist
- slowk, slowd (Stochastic)
- mfi (Money Flow Index)

Can read from context (context-aware):
- trend: To align momentum signals with trend direction

Outputs to context:
- momentum: {
    'direction': 'bullish' | 'bearish' | 'neutral',
    'strength': float (0-3),
    'rsi_signal': 'overbought' | 'oversold' | 'neutral',
    'macd_signal': dict,
    'stoch_signal': dict,
    'divergence': dict | None,
    'confidence': float (0-1),
    'signals': list of momentum signals
  }
"""

from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np

from signal_generation.analyzers.base_analyzer import BaseAnalyzer
from signal_generation.context import AnalysisContext

logger = logging.getLogger(__name__)


class MomentumAnalyzer(BaseAnalyzer):
    """
    Analyzes momentum indicators for trading signals.
    
    Key features:
    1. RSI analysis (overbought/oversold)
    2. MACD analysis (crossovers, histogram)
    3. Stochastic analysis
    4. Divergence detection
    5. Context-aware scoring (considers trend)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MomentumAnalyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get momentum-specific configuration
        mom_config = config.get('momentum', {})
        
        # RSI thresholds
        self.rsi_overbought = mom_config.get('rsi_overbought', 70)
        self.rsi_oversold = mom_config.get('rsi_oversold', 30)
        
        # Stochastic thresholds
        self.stoch_overbought = mom_config.get('stoch_overbought', 80)
        self.stoch_oversold = mom_config.get('stoch_oversold', 20)
        
        # Divergence detection lookback
        self.divergence_lookback = mom_config.get('divergence_lookback', 14)
        
        # Enable/disable
        self.enabled = config.get('analyzers', {}).get('momentum', {}).get('enabled', True)
        
        logger.info("MomentumAnalyzer initialized successfully")
    
    def analyze(self, context: AnalysisContext) -> None:
        """
        Main analysis method - analyzes momentum indicators.
        
        Args:
            context: AnalysisContext with pre-calculated indicators
        """
        # 1. Check if enabled
        if not self._check_enabled():
            logger.debug(f"MomentumAnalyzer disabled for {context.symbol}")
            return
        
        # 2. Validate context
        if not self._validate_context(context):
            logger.warning(f"MomentumAnalyzer: Invalid context for {context.symbol}")
            return
        
        try:
            # 3. Read pre-calculated indicators
            df = context.df
            
            # Ensure we have enough data
            if len(df) < 50:
                logger.warning(f"Insufficient data for MomentumAnalyzer on {context.symbol}")
                context.add_result('momentum', {
                    'status': 'insufficient_data',
                    'direction': 'neutral',
                    'strength': 0
                })
                return
            
            # Get current indicator values
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_macd_signal = df['macd_signal'].iloc[-1]
            current_macd_hist = df['macd_hist'].iloc[-1]
            current_slowk = df['slowk'].iloc[-1]
            current_slowd = df['slowd'].iloc[-1]
            
            # 4. Analyze RSI
            rsi_analysis = self._analyze_rsi(df)
            
            # 5. Analyze MACD
            macd_analysis = self._analyze_macd(df)
            
            # 6. Analyze Stochastic
            stoch_analysis = self._analyze_stochastic(df)
            
            # 7. Detect divergences
            divergence = self._detect_divergences(df)
            
            # 8. Calculate overall momentum direction and strength
            momentum_result = self._calculate_momentum(
                rsi_analysis,
                macd_analysis,
                stoch_analysis,
                divergence
            )
            
            # 9. Context-aware scoring (read trend from context if available)
            trend_context = context.get_result('trend')
            if trend_context:
                momentum_result = self._adjust_for_trend_alignment(
                    momentum_result,
                    trend_context
                )
            
            # 10. Generate momentum signals
            signals = self._generate_signals(
                rsi_analysis,
                macd_analysis,
                stoch_analysis,
                divergence
            )
            
            # 11. Calculate confidence
            confidence = self._calculate_confidence(
                momentum_result,
                rsi_analysis,
                macd_analysis,
                stoch_analysis,
                divergence
            )
            
            # 12. Build final result
            result = {
                'status': 'ok',
                'direction': momentum_result['direction'],
                'strength': momentum_result['strength'],
                'rsi_signal': rsi_analysis['signal'],
                'macd_signal': macd_analysis,
                'stoch_signal': stoch_analysis,
                'divergence': divergence,
                'confidence': confidence,
                'signals': signals,
                'details': {
                    'rsi': round(current_rsi, 2),
                    'macd': round(current_macd, 5),
                    'macd_signal': round(current_macd_signal, 5),
                    'macd_hist': round(current_macd_hist, 5),
                    'slowk': round(current_slowk, 2),
                    'slowd': round(current_slowd, 2)
                }
            }
            
            # 13. Store in context
            context.add_result('momentum', result)
            
            logger.info(
                f"MomentumAnalyzer completed for {context.symbol} {context.timeframe}: "
                f"{result['direction']} (strength: {result['strength']}, "
                f"confidence: {confidence:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Error in MomentumAnalyzer for {context.symbol}: {e}", exc_info=True)
            context.add_result('momentum', {
                'status': 'error',
                'direction': 'neutral',
                'strength': 0,
                'error': str(e)
            })
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze RSI indicator.
        
        Args:
            df: DataFrame with RSI column
            
        Returns:
            Dictionary with RSI analysis
        """
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2] if len(df) > 1 else current_rsi
        
        # Determine signal
        if current_rsi >= self.rsi_overbought:
            signal = 'overbought'
        elif current_rsi <= self.rsi_oversold:
            signal = 'oversold'
        else:
            signal = 'neutral'
        
        # Check for RSI crossing levels
        rsi_crossing_up = prev_rsi < self.rsi_oversold <= current_rsi
        rsi_crossing_down = prev_rsi > self.rsi_overbought >= current_rsi
        
        return {
            'value': current_rsi,
            'signal': signal,
            'crossing_up': rsi_crossing_up,
            'crossing_down': rsi_crossing_down,
            'bullish': current_rsi < 50 and signal == 'oversold',
            'bearish': current_rsi > 50 and signal == 'overbought'
        }
    
    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze MACD indicator.
        
        Args:
            df: DataFrame with MACD columns
            
        Returns:
            Dictionary with MACD analysis
        """
        current_macd = df['macd'].iloc[-1]
        current_signal = df['macd_signal'].iloc[-1]
        current_hist = df['macd_hist'].iloc[-1]
        
        prev_macd = df['macd'].iloc[-2] if len(df) > 1 else current_macd
        prev_signal = df['macd_signal'].iloc[-2] if len(df) > 1 else current_signal
        prev_hist = df['macd_hist'].iloc[-2] if len(df) > 1 else current_hist
        
        # Detect crossovers
        bullish_crossover = (prev_macd <= prev_signal and 
                            current_macd > current_signal)
        bearish_crossover = (prev_macd >= prev_signal and 
                            current_macd < current_signal)
        
        # Histogram analysis
        hist_increasing = current_hist > prev_hist
        hist_positive = current_hist > 0
        
        # Determine direction
        if current_macd > current_signal:
            direction = 'bullish'
        elif current_macd < current_signal:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'value': current_macd,
            'signal_value': current_signal,
            'histogram': current_hist,
            'direction': direction,
            'bullish_crossover': bullish_crossover,
            'bearish_crossover': bearish_crossover,
            'hist_increasing': hist_increasing,
            'hist_positive': hist_positive
        }
    
    def _analyze_stochastic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze Stochastic indicator.
        
        Args:
            df: DataFrame with Stochastic columns
            
        Returns:
            Dictionary with Stochastic analysis
        """
        current_k = df['slowk'].iloc[-1]
        current_d = df['slowd'].iloc[-1]
        
        prev_k = df['slowk'].iloc[-2] if len(df) > 1 else current_k
        prev_d = df['slowd'].iloc[-2] if len(df) > 1 else current_d
        
        # Determine signal
        if current_k >= self.stoch_overbought:
            signal = 'overbought'
        elif current_k <= self.stoch_oversold:
            signal = 'oversold'
        else:
            signal = 'neutral'
        
        # Detect crossovers
        bullish_crossover = prev_k <= prev_d and current_k > current_d
        bearish_crossover = prev_k >= prev_d and current_k < current_d
        
        return {
            'k_value': current_k,
            'd_value': current_d,
            'signal': signal,
            'bullish_crossover': bullish_crossover,
            'bearish_crossover': bearish_crossover
        }
    
    def _detect_divergences(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect bullish/bearish divergences between price and RSI.
        
        Args:
            df: DataFrame with price and RSI
            
        Returns:
            Divergence dict if found, None otherwise
        """
        try:
            lookback = min(self.divergence_lookback, len(df))
            if lookback < 5:
                return None
            
            recent_df = df.tail(lookback)
            
            # Find price lows and highs
            price_lows = recent_df['low'].rolling(window=3, center=True).min()
            price_highs = recent_df['high'].rolling(window=3, center=True).max()
            
            # Find RSI lows and highs
            rsi_lows = recent_df['rsi'].rolling(window=3, center=True).min()
            rsi_highs = recent_df['rsi'].rolling(window=3, center=True).max()
            
            # Bullish divergence: price making lower low, RSI making higher low
            # Ensure we have enough data before accessing indices
            price_lower_low = False
            rsi_higher_low = False

            if len(price_lows) >= 6:  # Need at least 6 to safely access iloc[-5]
                price_lower_low = price_lows.iloc[-1] < price_lows.iloc[-5]
                rsi_higher_low = rsi_lows.iloc[-1] > rsi_lows.iloc[-5]

            if price_lower_low and rsi_higher_low:
                return {
                    'type': 'bullish',
                    'strength': 'strong' if rsi_lows.iloc[-1] < 40 else 'moderate'
                }

            # Bearish divergence: price making higher high, RSI making lower high
            price_higher_high = False
            rsi_lower_high = False

            if len(price_highs) >= 6:  # Need at least 6 to safely access iloc[-5]
                price_higher_high = price_highs.iloc[-1] > price_highs.iloc[-5]
                rsi_lower_high = rsi_highs.iloc[-1] < rsi_highs.iloc[-5]

            if price_higher_high and rsi_lower_high:
                return {
                    'type': 'bearish',
                    'strength': 'strong' if rsi_highs.iloc[-1] > 60 else 'moderate'
                }

            return None
            
        except Exception as e:
            logger.debug(f"Divergence detection failed: {e}")
            return None
    
    def _calculate_momentum(
        self,
        rsi: Dict,
        macd: Dict,
        stoch: Dict,
        divergence: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate overall momentum direction and strength.
        
        Args:
            rsi: RSI analysis
            macd: MACD analysis
            stoch: Stochastic analysis
            divergence: Divergence analysis
            
        Returns:
            Dictionary with momentum direction and strength
        """
        bullish_score = 0
        bearish_score = 0
        
        # RSI contribution
        if rsi['bullish']:
            bullish_score += 1
        if rsi['bearish']:
            bearish_score += 1
        
        # MACD contribution
        if macd['direction'] == 'bullish':
            bullish_score += 1
            if macd['bullish_crossover']:
                bullish_score += 1
        elif macd['direction'] == 'bearish':
            bearish_score += 1
            if macd['bearish_crossover']:
                bearish_score += 1
        
        # Stochastic contribution
        if stoch['signal'] == 'oversold' and stoch['bullish_crossover']:
            bullish_score += 1
        if stoch['signal'] == 'overbought' and stoch['bearish_crossover']:
            bearish_score += 1
        
        # Divergence contribution
        if divergence:
            if divergence['type'] == 'bullish':
                bullish_score += 2
            else:
                bearish_score += 2
        
        # Determine direction and strength
        if bullish_score > bearish_score:
            direction = 'bullish'
            strength = min(bullish_score - bearish_score, 3)
        elif bearish_score > bullish_score:
            direction = 'bearish'
            strength = min(bearish_score - bullish_score, 3)
        else:
            direction = 'neutral'
            strength = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score
        }
    
    def _adjust_for_trend_alignment(
        self,
        momentum: Dict,
        trend: Dict
    ) -> Dict:
        """
        Adjust momentum scores based on trend alignment (context-aware).
        
        Args:
            momentum: Momentum result
            trend: Trend result from context
            
        Returns:
            Adjusted momentum result
        """
        trend_direction = trend.get('direction', 'neutral')
        momentum_direction = momentum['direction']
        
        # Bonus for trend alignment
        if trend_direction == momentum_direction:
            momentum['strength'] = min(momentum['strength'] * 1.2, 3)
            momentum['trend_aligned'] = True
        else:
            momentum['trend_aligned'] = False
        
        return momentum
    
    def _generate_signals(
        self,
        rsi: Dict,
        macd: Dict,
        stoch: Dict,
        divergence: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Generate specific momentum signals.
        
        Args:
            rsi, macd, stoch: Indicator analyses
            divergence: Divergence analysis
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # RSI signals
        if rsi['crossing_up']:
            signals.append({
                'type': 'rsi_oversold_bounce',
                'direction': 'bullish',
                'strength': 2,
                'description': 'RSI crossed above oversold level'
            })
        
        if rsi['crossing_down']:
            signals.append({
                'type': 'rsi_overbought_reversal',
                'direction': 'bearish',
                'strength': 2,
                'description': 'RSI crossed below overbought level'
            })
        
        # MACD signals
        if macd['bullish_crossover']:
            signals.append({
                'type': 'macd_bullish_cross',
                'direction': 'bullish',
                'strength': 2,
                'description': 'MACD bullish crossover'
            })
        
        if macd['bearish_crossover']:
            signals.append({
                'type': 'macd_bearish_cross',
                'direction': 'bearish',
                'strength': 2,
                'description': 'MACD bearish crossover'
            })
        
        # Stochastic signals
        if stoch['bullish_crossover'] and stoch['signal'] == 'oversold':
            signals.append({
                'type': 'stoch_bullish_cross',
                'direction': 'bullish',
                'strength': 1,
                'description': 'Stochastic bullish crossover in oversold'
            })
        
        if stoch['bearish_crossover'] and stoch['signal'] == 'overbought':
            signals.append({
                'type': 'stoch_bearish_cross',
                'direction': 'bearish',
                'strength': 1,
                'description': 'Stochastic bearish crossover in overbought'
            })
        
        # Divergence signals
        if divergence:
            signals.append({
                'type': f'{divergence["type"]}_divergence',
                'direction': divergence['type'],
                'strength': 3 if divergence['strength'] == 'strong' else 2,
                'description': f'{divergence["type"].capitalize()} divergence detected'
            })
        
        return signals
    
    def _calculate_confidence(
        self,
        momentum: Dict,
        rsi: Dict,
        macd: Dict,
        stoch: Dict,
        divergence: Optional[Dict]
    ) -> float:
        """
        Calculate confidence score.
        
        Args:
            momentum, rsi, macd, stoch, divergence: Analysis results
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5
        
        # Strong momentum increases confidence
        if momentum['strength'] >= 3:
            confidence += 0.3
        elif momentum['strength'] == 2:
            confidence += 0.2
        elif momentum['strength'] == 1:
            confidence += 0.1
        
        # Multiple confirming indicators
        if macd['direction'] == momentum['direction']:
            confidence += 0.1
        
        # Divergence is strong signal
        if divergence:
            confidence += 0.2
        
        # Trend alignment
        if momentum.get('trend_aligned'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _validate_context(self, context: AnalysisContext) -> bool:
        """Validate required indicators"""
        required = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'slowk', 'slowd']
        
        df = context.df
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        return True
