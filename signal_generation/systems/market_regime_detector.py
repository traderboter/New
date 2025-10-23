"""
Market Regime Detector
Detects market regime (trend, volatility) and adapts parameters.
"""

import logging
import time
import copy
import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DataFrame = pd.DataFrame


class MarketRegimeDetector:
    """Detects market regime (trend, volatility) and adapts parameters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config.get('market_regime', {})
        self.enabled = self.config.get('enabled', True)

        # Indicator parameters
        self.adx_period = self.config.get('adx_period', 14)
        self.volatility_period = self.config.get('volatility_period', 20)  # ATR period

        # Detection thresholds
        self.strong_trend_threshold = self.config.get('strong_trend_threshold', 25)
        self.weak_trend_threshold = self.config.get('weak_trend_threshold', 20)
        self.high_volatility_threshold = self.config.get('high_volatility_threshold', 1.5)  # ATR %
        self.low_volatility_threshold = self.config.get('low_volatility_threshold', 0.5)  # ATR %

        # Cache results
        self._regime_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

        # Required samples
        self._required_samples = max(self.adx_period, self.volatility_period) + 10  # Extra buffer

        # Regime history for transition analysis
        self._regime_history = deque(maxlen=10)  # Keep last 10 regimes
        self._regime_transition_probabilities = defaultdict(lambda: defaultdict(int))

        logger.info(f"MarketRegimeDetector initialized. Enabled: {self.enabled}")

    def detect_regime(self, df: DataFrame) -> Dict[str, Any]:
        """
        Detect market regime based on ADX and ATR.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with regime info
        """
        if not self.enabled:
            return {'regime': 'disabled', 'confidence': 1.0, 'details': {}}

        # Check cache
        cache_key = self._generate_cache_key(df)
        if cache_key in self._regime_cache:
            cached_result, timestamp = self._regime_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        # Minimum data needed
        if df is None or len(df) < self._required_samples:
            logger.debug(
                f"Not enough data ({len(df) if df is not None else 0} rows) "
                f"to detect market regime (requires {self._required_samples})."
            )
            return {'regime': 'unknown_data', 'confidence': 0.0, 'details': {}}

        try:
            # Prepare data
            df_copy = df.copy()
            high_prices = df_copy['high'].values.astype(np.float64)
            low_prices = df_copy['low'].values.astype(np.float64)
            close_prices = df_copy['close'].values.astype(np.float64)

            # Calculate ADX, +DI, -DI
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=self.adx_period)

            # Calculate ATR%
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.volatility_period)
            atr_percent = np.where(close_prices > 0, (atr / close_prices) * 100, 0)

            # Get last valid values
            last_valid_idx = self._find_last_valid_index([adx, atr_percent])
            if last_valid_idx is None:
                logger.warning("Could not find valid ADX/ATR values for regime detection.")
                return {'regime': 'unknown_calc', 'confidence': 0.0, 'details': {}}

            current_adx = adx[last_valid_idx]
            current_plus_di = plus_di[last_valid_idx]
            current_minus_di = minus_di[last_valid_idx]
            current_atr_percent = atr_percent[last_valid_idx]

            # Determine trend strength
            if current_adx >= self.strong_trend_threshold:
                trend_strength = 'strong_trend'
            elif current_adx >= self.weak_trend_threshold:
                trend_strength = 'weak_trend'
            else:
                trend_strength = 'no_trend'

            # Determine trend direction
            if current_plus_di > current_minus_di:
                trend_direction = 'bullish'
            elif current_minus_di > current_plus_di:
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'

            # Determine volatility
            if current_atr_percent >= self.high_volatility_threshold:
                volatility = 'high'
            elif current_atr_percent <= self.low_volatility_threshold:
                volatility = 'low'
            else:
                volatility = 'normal'

            # Build regime string
            regime = f"{trend_strength}_{trend_direction}_{volatility}"

            # Calculate confidence
            confidence = min(1.0, current_adx / 50.0)  # ADX-based confidence

            # Details
            details = {
                'adx': round(current_adx, 2),
                'plus_di': round(current_plus_di, 2),
                'minus_di': round(current_minus_di, 2),
                'atr_percent': round(current_atr_percent, 3)
            }

            # Regime transition probabilities
            if self._regime_history:
                prev_regime = self._regime_history[-1]
                self._regime_transition_probabilities[prev_regime][regime] += 1
                next_regime_probs = self._calculate_next_regime_probabilities(regime)
                details['next_regime_probabilities'] = next_regime_probs

            # Add to history
            self._regime_history.append(regime)

            logger.debug(
                f"Regime Detected: {regime}, Strength: {trend_strength} ({details['adx']}), "
                f"Direction: {trend_direction}, Volatility: {volatility} ({details['atr_percent']}), "
                f"Confidence: {confidence:.2f}"
            )

            result = {
                'regime': regime,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'volatility': volatility,
                'confidence': confidence,
                'details': details
            }

            # Save to cache
            self._regime_cache[cache_key] = (result, time.time())

            return result

        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}", exc_info=True)
            return {'regime': 'error', 'confidence': 0.0, 'details': {'error': str(e)}}

    def _calculate_next_regime_probabilities(self, current_regime: str) -> Dict[str, float]:
        """Calculate transition probabilities to next regimes."""
        transitions = self._regime_transition_probabilities[current_regime]
        total_transitions = sum(transitions.values())

        if total_transitions == 0:
            return {}

        return {
            next_regime: count / total_transitions
            for next_regime, count in transitions.items()
        }

    def _find_last_valid_index(self, arrays: List[np.ndarray]) -> Optional[int]:
        """Find the last valid index across multiple arrays."""
        if not arrays:
            return None

        max_len = min(len(arr) for arr in arrays)
        if max_len == 0:
            return None

        # Search from end to beginning
        for i in range(-1, -max_len - 1, -1):
            if all(not np.isnan(arr[i]) for arr in arrays):
                return i

        return None

    def _generate_cache_key(self, df: DataFrame) -> str:
        """Generate cache key from dataframe."""
        if df is None or len(df) == 0:
            return "empty_dataframe"

        try:
            last_idx = df.index[-1]
            last_close = df['close'].iloc[-1]

            timestamp_str = str(last_idx) if isinstance(last_idx, (int, float)) else (
                str(last_idx.timestamp()) if hasattr(last_idx, 'timestamp') else str(last_idx)
            )
            return f"{timestamp_str}_{last_close:.6f}"
        except (IndexError, KeyError):
            return f"dataframe_len_{len(df)}"

    def get_adapted_parameters(
            self,
            regime_info: Dict[str, Any],
            base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on market regime.

        Args:
            regime_info: Regime detection results
            base_config: Base configuration

        Returns:
            Adapted configuration
        """
        if not self.enabled or regime_info.get('regime', 'disabled') in [
            'disabled', 'unknown_data', 'unknown_calc', 'error'
        ]:
            return base_config

        # Deep copy to avoid changing original
        adapted_config = copy.deepcopy(base_config)

        regime = regime_info.get('regime')
        trend_strength = regime_info.get('trend_strength')
        volatility = regime_info.get('volatility')
        confidence = regime_info.get('confidence', 0.5)

        # Get config sections
        risk_params = adapted_config.setdefault('risk_management', {})
        signal_params = adapted_config.setdefault('signal_generation', {})

        # Base values
        base_risk = copy.deepcopy(self.config.get('risk_management', {}))
        base_signal = copy.deepcopy(self.config.get('signal_generation', {}))

        # --- Adjust Risk Parameters ---

        # 1. Max risk per trade
        base_risk_pct = base_risk.get('max_risk_per_trade_percent', 2.0)
        risk_modifier = 1.0

        if volatility == 'high':
            risk_modifier = 0.7  # Reduce risk
        elif volatility == 'low':
            risk_modifier = 1.2  # Can increase slightly

        risk_params['max_risk_per_trade_percent'] = (
                base_risk_pct * (1.0 + (risk_modifier - 1.0) * confidence)
        )

        # 2. Preferred RR ratio
        base_rr = base_risk.get('preferred_risk_reward_ratio', 2.5)
        rr_modifier = 1.0

        if trend_strength == 'strong_trend':
            rr_modifier = 1.2  # Higher targets in strong trends
        elif trend_strength == 'no_trend':
            rr_modifier = 0.8  # Lower targets in ranging

        risk_params['preferred_risk_reward_ratio'] = (
                base_rr * (1.0 + (rr_modifier - 1.0) * confidence)
        )

        # 3. Stop loss percentage
        base_sl_percent = base_risk.get('default_stop_loss_percent', 1.5)
        sl_modifier = 1.0

        if volatility == 'high':
            sl_modifier = 1.3  # Wider stop loss
        elif volatility == 'low':
            sl_modifier = 0.8  # Tighter stop loss

        risk_params['default_stop_loss_percent'] = (
                base_sl_percent * (1.0 + (sl_modifier - 1.0) * confidence)
        )

        # --- Adjust Signal Parameters ---

        # 1. Minimum signal score
        base_min_score = base_signal.get('minimum_signal_score', 33)
        score_modifier = 1.0

        if trend_strength == 'no_trend' or volatility == 'high':
            score_modifier = 1.1  # More strict

        signal_params['minimum_signal_score'] = (
                base_min_score * (1.0 + (score_modifier - 1.0) * confidence)
        )

        # Round values
        for params in [risk_params, signal_params]:
            for key, value in params.items():
                if isinstance(value, float):
                    params[key] = round(value, 2)

        logger.debug(
            f"Regime '{regime}' (Conf: {confidence:.2f}) -> Adapted Params: "
            f"Risk%: {risk_params['max_risk_per_trade_percent']:.2f}, "
            f"RR: {risk_params['preferred_risk_reward_ratio']:.2f}, "
            f"MinScore: {signal_params['minimum_signal_score']:.2f}"
        )

        return adapted_config