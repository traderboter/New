"""
Market Regime Detector
Detects market regime (trend, volatility) and adapts parameters.
"""

import logging
import time
import copy
import hashlib
import numpy as np
import pandas as pd
import talib
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Enums for market regime states
class TrendStrength(str, Enum):
    """Trend strength classification."""
    STRONG = 'strong_trend'
    WEAK = 'weak_trend'
    NONE = 'no_trend'


class TrendDirection(str, Enum):
    """Trend direction classification."""
    BULLISH = 'bullish'
    BEARISH = 'bearish'
    NEUTRAL = 'neutral'


class Volatility(str, Enum):
    """Volatility classification."""
    HIGH = 'high'
    NORMAL = 'normal'
    LOW = 'low'


class RegimeStatus(str, Enum):
    """Regime detection status."""
    DISABLED = 'disabled'
    UNKNOWN_DATA = 'unknown_data'
    UNKNOWN_CALC = 'unknown_calc'
    ERROR = 'error'


# Constants
ADX_MAX_REFERENCE = 50.0  # Maximum ADX value for normalization
CACHE_TTL_SECONDS = 300  # Cache time-to-live (5 minutes)
REGIME_HISTORY_MAXLEN = 10  # Maximum regime history to keep
EXTRA_DATA_BUFFER = 10  # Extra samples for indicator calculation


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero or invalid.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if division fails

    Returns:
        Result of division or default value
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        return default
    return result


class MarketRegimeDetector:
    """Detects market regime (trend, volatility) and adapts parameters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config.get('market_regime', {})
        self.enabled = self.config.get('enabled', True)

        # Indicator parameters with validation
        self.adx_period = self._validate_period(
            self.config.get('adx_period', 14),
            'adx_period',
            min_value=5,
            max_value=100
        )
        self.volatility_period = self._validate_period(
            self.config.get('volatility_period', 20),
            'volatility_period',
            min_value=5,
            max_value=100
        )

        # Detection thresholds with validation
        self.strong_trend_threshold = self._validate_threshold(
            self.config.get('strong_trend_threshold', 25),
            'strong_trend_threshold',
            min_value=0,
            max_value=100
        )
        self.weak_trend_threshold = self._validate_threshold(
            self.config.get('weak_trend_threshold', 20),
            'weak_trend_threshold',
            min_value=0,
            max_value=100
        )
        self.high_volatility_threshold = self._validate_threshold(
            self.config.get('high_volatility_threshold', 1.5),
            'high_volatility_threshold',
            min_value=0,
            max_value=10.0
        )
        self.low_volatility_threshold = self._validate_threshold(
            self.config.get('low_volatility_threshold', 0.5),
            'low_volatility_threshold',
            min_value=0,
            max_value=10.0
        )

        # Validate threshold relationships
        if self.weak_trend_threshold > self.strong_trend_threshold:
            logger.warning(
                f"weak_trend_threshold ({self.weak_trend_threshold}) > "
                f"strong_trend_threshold ({self.strong_trend_threshold}). Swapping values."
            )
            self.weak_trend_threshold, self.strong_trend_threshold = (
                self.strong_trend_threshold, self.weak_trend_threshold
            )

        if self.low_volatility_threshold > self.high_volatility_threshold:
            logger.warning(
                f"low_volatility_threshold ({self.low_volatility_threshold}) > "
                f"high_volatility_threshold ({self.high_volatility_threshold}). Swapping values."
            )
            self.low_volatility_threshold, self.high_volatility_threshold = (
                self.high_volatility_threshold, self.low_volatility_threshold
            )

        # Cache results
        self._regime_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_ttl_seconds = CACHE_TTL_SECONDS

        # Required samples
        self._required_samples = max(self.adx_period, self.volatility_period) + EXTRA_DATA_BUFFER

        # Regime history for transition analysis
        self._regime_history = deque(maxlen=REGIME_HISTORY_MAXLEN)
        self._regime_transition_probabilities = defaultdict(lambda: defaultdict(int))

        logger.info(f"MarketRegimeDetector initialized. Enabled: {self.enabled}")

    def _validate_period(self, value: Any, name: str, min_value: int = 1, max_value: int = 1000) -> int:
        """
        Validate period parameter.

        Args:
            value: Value to validate
            name: Parameter name for logging
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer value

        Raises:
            ValueError: If value is invalid
        """
        try:
            int_value = int(value)
            if int_value < min_value or int_value > max_value:
                raise ValueError(
                    f"{name} must be between {min_value} and {max_value}, got {int_value}"
                )
            return int_value
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid {name}: {value}. Error: {e}")
            raise ValueError(f"Invalid {name}: {value}") from e

    def _validate_threshold(self, value: Any, name: str, min_value: float = 0.0, max_value: float = 100.0) -> float:
        """
        Validate threshold parameter.

        Args:
            value: Value to validate
            name: Parameter name for logging
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated float value

        Raises:
            ValueError: If value is invalid
        """
        try:
            float_value = float(value)
            if float_value < min_value or float_value > max_value:
                raise ValueError(
                    f"{name} must be between {min_value} and {max_value}, got {float_value}"
                )
            return float_value
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid {name}: {value}. Error: {e}")
            raise ValueError(f"Invalid {name}: {value}") from e

    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market regime based on ADX and ATR.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with regime info
        """
        if not self.enabled:
            return {
                'regime': RegimeStatus.DISABLED.value,
                'confidence': 1.0,
                'details': {}
            }

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
            return {
                'regime': RegimeStatus.UNKNOWN_DATA.value,
                'confidence': 0.0,
                'details': {}
            }

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
                return {
                    'regime': RegimeStatus.UNKNOWN_CALC.value,
                    'confidence': 0.0,
                    'details': {}
                }

            current_adx = adx[last_valid_idx]
            current_plus_di = plus_di[last_valid_idx]
            current_minus_di = minus_di[last_valid_idx]
            current_atr_percent = atr_percent[last_valid_idx]

            # Validate that values are not NaN
            assert not np.isnan(current_adx), "current_adx is NaN"
            assert not np.isnan(current_atr_percent), "current_atr_percent is NaN"

            # Determine trend strength
            if current_adx >= self.strong_trend_threshold:
                trend_strength = TrendStrength.STRONG
            elif current_adx >= self.weak_trend_threshold:
                trend_strength = TrendStrength.WEAK
            else:
                trend_strength = TrendStrength.NONE

            # Determine trend direction
            if current_plus_di > current_minus_di:
                trend_direction = TrendDirection.BULLISH
            elif current_minus_di > current_plus_di:
                trend_direction = TrendDirection.BEARISH
            else:
                trend_direction = TrendDirection.NEUTRAL

            # Determine volatility
            if current_atr_percent >= self.high_volatility_threshold:
                volatility = Volatility.HIGH
            elif current_atr_percent <= self.low_volatility_threshold:
                volatility = Volatility.LOW
            else:
                volatility = Volatility.NORMAL

            # Build regime string
            regime = f"{trend_strength.value}_{trend_direction.value}_{volatility.value}"

            # Calculate confidence using safe division
            confidence = min(1.0, safe_divide(current_adx, ADX_MAX_REFERENCE, 0.0))

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
                f"Regime Detected: {regime}, Strength: {trend_strength.value} ({details['adx']}), "
                f"Direction: {trend_direction.value}, Volatility: {volatility.value} ({details['atr_percent']}), "
                f"Confidence: {confidence:.2f}"
            )

            result = {
                'regime': regime,
                'trend_strength': trend_strength.value,
                'trend_direction': trend_direction.value,
                'volatility': volatility.value,
                'confidence': confidence,
                'details': details
            }

            # Save to cache
            self._regime_cache[cache_key] = (result, time.time())

            return result

        except AssertionError as e:
            logger.error(f"Assertion failed in regime detection: {str(e)}", exc_info=True)
            return {
                'regime': RegimeStatus.ERROR.value,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}", exc_info=True)
            return {
                'regime': RegimeStatus.ERROR.value,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }

    def _calculate_next_regime_probabilities(self, current_regime: str) -> Dict[str, float]:
        """Calculate transition probabilities to next regimes."""
        transitions = self._regime_transition_probabilities[current_regime]
        total_transitions = sum(transitions.values())

        if total_transitions == 0:
            return {}

        return {
            next_regime: safe_divide(count, total_transitions, 0.0)
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

    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """
        Generate cache key from dataframe using hash for uniqueness.

        Args:
            df: Input dataframe

        Returns:
            Unique cache key string
        """
        if df is None or len(df) == 0:
            return "empty_dataframe"

        try:
            last_idx = df.index[-1]
            last_close = df['close'].iloc[-1]
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]

            # Create a unique string representation
            timestamp_str = str(last_idx) if isinstance(last_idx, (int, float)) else (
                str(last_idx.timestamp()) if hasattr(last_idx, 'timestamp') else str(last_idx)
            )

            # Use hash for better uniqueness guarantee
            data_str = f"{timestamp_str}_{last_close:.8f}_{last_high:.8f}_{last_low:.8f}_{len(df)}"
            hash_value = hashlib.md5(data_str.encode()).hexdigest()[:16]

            return f"regime_{hash_value}"
        except (IndexError, KeyError, AttributeError) as e:
            logger.debug(f"Error generating cache key: {e}")
            return f"dataframe_len_{len(df)}"

    def _apply_adjustment(
            self,
            base_value: float,
            modifier: float,
            confidence: float
    ) -> float:
        """
        Apply adjustment with confidence weighting.

        Args:
            base_value: Base parameter value
            modifier: Adjustment modifier (1.0 = no change)
            confidence: Confidence level (0.0 to 1.0)

        Returns:
            Adjusted value
        """
        return base_value * (1.0 + (modifier - 1.0) * confidence)

    def _adapt_risk_parameters(
            self,
            risk_params: Dict[str, Any],
            base_risk: Dict[str, Any],
            trend_strength: str,
            volatility: str,
            confidence: float
    ) -> None:
        """
        Adapt risk management parameters based on market regime.

        Args:
            risk_params: Risk parameters to modify (modified in-place)
            base_risk: Base risk configuration
            trend_strength: Trend strength classification
            volatility: Volatility classification
            confidence: Confidence level
        """
        # 1. Max risk per trade
        base_risk_pct = base_risk.get('max_risk_per_trade_percent', 2.0)
        risk_modifier = 1.0

        if volatility == Volatility.HIGH.value:
            risk_modifier = 0.7  # Reduce risk
        elif volatility == Volatility.LOW.value:
            risk_modifier = 1.2  # Can increase slightly

        risk_params['max_risk_per_trade_percent'] = self._apply_adjustment(
            base_risk_pct, risk_modifier, confidence
        )

        # 2. Preferred RR ratio
        base_rr = base_risk.get('preferred_risk_reward_ratio', 2.5)
        rr_modifier = 1.0

        if trend_strength == TrendStrength.STRONG.value:
            rr_modifier = 1.2  # Higher targets in strong trends
        elif trend_strength == TrendStrength.NONE.value:
            rr_modifier = 0.8  # Lower targets in ranging

        risk_params['preferred_risk_reward_ratio'] = self._apply_adjustment(
            base_rr, rr_modifier, confidence
        )

        # 3. Stop loss percentage
        base_sl_percent = base_risk.get('default_stop_loss_percent', 1.5)
        sl_modifier = 1.0

        if volatility == Volatility.HIGH.value:
            sl_modifier = 1.3  # Wider stop loss
        elif volatility == Volatility.LOW.value:
            sl_modifier = 0.8  # Tighter stop loss

        risk_params['default_stop_loss_percent'] = self._apply_adjustment(
            base_sl_percent, sl_modifier, confidence
        )

    def _adapt_signal_parameters(
            self,
            signal_params: Dict[str, Any],
            base_signal: Dict[str, Any],
            trend_strength: str,
            volatility: str,
            confidence: float
    ) -> None:
        """
        Adapt signal generation parameters based on market regime.

        Args:
            signal_params: Signal parameters to modify (modified in-place)
            base_signal: Base signal configuration
            trend_strength: Trend strength classification
            volatility: Volatility classification
            confidence: Confidence level
        """
        # Minimum signal score
        base_min_score = base_signal.get('minimum_signal_score', 33)
        score_modifier = 1.0

        if trend_strength == TrendStrength.NONE.value or volatility == Volatility.HIGH.value:
            score_modifier = 1.1  # More strict

        signal_params['minimum_signal_score'] = self._apply_adjustment(
            base_min_score, score_modifier, confidence
        )

    def _round_parameters(self, *param_dicts: Dict[str, Any]) -> None:
        """
        Round float parameters to 2 decimal places.

        Args:
            *param_dicts: Variable number of parameter dictionaries to round
        """
        for params in param_dicts:
            for key, value in params.items():
                if isinstance(value, float):
                    params[key] = round(value, 2)

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
        # Check if adaptation is needed
        disabled_regimes = [
            RegimeStatus.DISABLED.value,
            RegimeStatus.UNKNOWN_DATA.value,
            RegimeStatus.UNKNOWN_CALC.value,
            RegimeStatus.ERROR.value
        ]
        if not self.enabled or regime_info.get('regime', 'disabled') in disabled_regimes:
            return base_config

        # Deep copy to avoid changing original
        adapted_config = copy.deepcopy(base_config)

        # Extract regime information
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

        # Apply adaptations
        self._adapt_risk_parameters(risk_params, base_risk, trend_strength, volatility, confidence)
        self._adapt_signal_parameters(signal_params, base_signal, trend_strength, volatility, confidence)

        # Round all parameter values
        self._round_parameters(risk_params, signal_params)

        # Log the adaptation
        logger.debug(
            f"Regime '{regime}' (Conf: {confidence:.2f}) -> Adapted Params: "
            f"Risk%: {risk_params.get('max_risk_per_trade_percent', 'N/A')}, "
            f"RR: {risk_params.get('preferred_risk_reward_ratio', 'N/A')}, "
            f"MinScore: {signal_params.get('minimum_signal_score', 'N/A')}"
        )

        return adapted_config