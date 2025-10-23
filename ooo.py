"""
سیستم پیشرفته تولید سیگنال معاملاتی چند تایم‌فریم

این سیستم با ترکیب بهترین ویژگی‌های تحلیل تکنیکال، یادگیری تطبیقی، مدیریت همبستگی
و سیستم قطع اضطراری، سیگنال‌های معاملاتی با کیفیت بالا تولید می‌کند.

بهینه‌سازی‌های انجام شده:
1. کش هوشمند برای کاهش محاسبات تکراری
2. فیلترینگ پیشرفته برای کاهش سیگنال‌های نویزی
3. وزن‌دهی دینامیک بر اساس شرایط بازار
4. مدیریت حافظه بهینه
5. پردازش موازی کارآمد
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Set, TypeVar, cast, Callable, DefaultDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
import copy
import asyncio
import random
import time
import json
import os
from functools import lru_cache, partial
import warnings
from collections import defaultdict, deque
from pathlib import Path

# کتابخانه‌های تحلیل تکنیکال
import talib
from scipy import signal as sig_processing
from scipy import stats
import scipy

# کتابخانه‌های بهینه‌سازی
try:
    import bottleneck as bn

    use_bottleneck = True
except ImportError:
    use_bottleneck = False

# پردازش موازی
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Type definitions
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)
TimeSeriesData = Union[np.ndarray, pd.Series]
T = TypeVar('T')

# تنظیمات لاگر
logger = logging.getLogger(__name__)
if use_bottleneck:
    logger.info("کتابخانه Bottleneck یافت شد، از محاسبات بهینه استفاده می‌شود.")
else:
    logger.info("کتابخانه Bottleneck یافت نشد، از محاسبات استاندارد استفاده می‌شود.")


# ===============================================
#      کلاس‌های Data برای اطلاعات سیگنال
# ===============================================
@dataclass
class SignalScore:
    """جزئیات امتیاز سیگنال برای ارزیابی کیفیت"""
    base_score: float = 0.0  # امتیاز پایه
    timeframe_weight: float = 1.0  # وزن تایم‌فریم
    trend_alignment: float = 1.0  # همراستایی با روند
    volume_confirmation: float = 1.0  # تایید حجم
    pattern_quality: float = 1.0  # کیفیت الگو
    confluence_score: float = 0.0  # امتیاز همگرایی
    final_score: float = 0.0  # امتیاز نهایی
    symbol_performance_factor: float = 1.0  # عملکرد تاریخی سمبل
    correlation_safety_factor: float = 1.0  # ایمنی همبستگی
    macd_analysis_score: float = 1.0  # امتیاز تحلیل MACD
    structure_score: float = 1.0  # امتیاز ساختار تایم‌فریم بالاتر
    volatility_score: float = 1.0  # امتیاز نوسان
    harmonic_pattern_score: float = 1.0  # امتیاز الگوهای هارمونیک
    price_channel_score: float = 1.0  # امتیاز کانال قیمتی
    cyclical_pattern_score: float = 1.0  # امتیاز الگوهای چرخه‌ای

    # فیلدهای جدید برای بهینه‌سازی
    market_strength: float = 1.0  # قدرت کلی بازار
    signal_clarity: float = 1.0  # وضوح سیگنال
    risk_adjusted_score: float = 0.0  # امتیاز تعدیل‌شده براساس ریسک

    def to_dict(self) -> Dict[str, float]:
        """تبدیل به دیکشنری برای ذخیره‌سازی"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SignalScore':
        """ایجاد از دیکشنری"""
        return cls(**data)

    def calculate_risk_adjusted_score(self, risk_reward_ratio: float,
                                      market_volatility: float) -> None:
        """محاسبه امتیاز تعدیل‌شده براساس ریسک"""
        # امتیاز بیشتر برای RR بهتر
        rr_factor = min(2.0, max(0.5, risk_reward_ratio / 2.0))

        # کاهش امتیاز در نوسان بالا
        volatility_penalty = max(0.5, 1.0 - (market_volatility - 1.0) * 0.3)

        self.risk_adjusted_score = self.final_score * rr_factor * volatility_penalty


@dataclass
class SignalInfo:
    """اطلاعات کامل سیگنال معاملاتی"""
    symbol: str
    timeframe: str  # تایم‌فریم اصلی
    signal_type: str  # نوع سیگنال
    direction: str  # 'long' یا 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timestamp: datetime  # زمان تولید سیگنال
    pattern_names: List[str] = field(default_factory=list)  # الگوهای شناسایی شده
    score: SignalScore = field(default_factory=SignalScore)  # امتیاز دقیق
    confirmation_timeframes: List[str] = field(default_factory=list)  # تایم‌فریم‌های تایید شده
    rejected_reason: Optional[str] = None  # دلیل رد شدن

    # اطلاعات اضافی
    regime: Optional[str] = None  # رژیم بازار
    is_reversal: bool = False  # آیا سیگنال برگشتی است
    adapted_config: Optional[Dict[str, Any]] = None  # پیکربندی تطبیقی

    # جزئیات تحلیل پیشرفته
    macd_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    volatility_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    htf_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    harmonic_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    channel_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    cyclical_details: Optional[Dict[str, Any]] = field(default_factory=dict)

    # فیلدهای جدید
    correlated_symbols: List[Tuple[str, float]] = field(default_factory=list)
    signal_id: str = ""
    market_context: Dict[str, Any] = field(default_factory=dict)
    trade_result: Optional[Dict[str, Any]] = None

    # فیلدهای بهینه‌سازی
    confidence_level: float = 0.0  # سطح اطمینان (0-1)
    expected_profit: float = 0.0  # سود مورد انتظار
    max_drawdown: float = 0.0  # حداکثر ضرر احتمالی
    time_validity: int = 3600  # مدت اعتبار سیگنال (ثانیه)
    priority: int = 0  # اولویت سیگنال

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری برای ذخیره‌سازی"""
        result = asdict(self)
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
        if self.score:
            result['score'] = self.score.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalInfo':
        """ایجاد از دیکشنری"""
        data_copy = data.copy()
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        if 'score' in data_copy and isinstance(data_copy['score'], dict):
            data_copy['score'] = SignalScore.from_dict(data_copy['score'])
        return cls(**data_copy)

    def ensure_aware_timestamp(self) -> None:
        """اطمینان از اینکه timestamp دارای timezone است"""
        if self.timestamp and self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def generate_signal_id(self) -> None:
        """تولید شناسه یکتا برای سیگنال"""
        if not self.signal_id:
            time_part = int(time.time() * 1000)  # میلی‌ثانیه برای دقت بیشتر
            random_part = random.randint(10000, 99999)
            symbol_part = ''.join(c for c in self.symbol if c.isalnum())[:6]
            direction_part = 'L' if self.direction == 'long' else 'S'
            self.signal_id = f"{symbol_part}_{direction_part}_{time_part}_{random_part}"

    def calculate_confidence(self) -> None:
        """محاسبه سطح اطمینان سیگنال"""
        # فاکتورهای موثر در اطمینان
        score_factor = min(1.0, self.score.final_score / 300)  # نرمالایز امتیاز
        rr_factor = min(1.0, self.risk_reward_ratio / 3.0)  # RR بالای 3 عالی
        pattern_factor = min(1.0, len(self.pattern_names) / 5)  # تعداد الگوها
        tf_factor = min(1.0, len(self.confirmation_timeframes) / 3)  # تایید تایم‌فریم‌ها

        # محاسبه اطمینان کلی
        self.confidence_level = (
                score_factor * 0.4 +
                rr_factor * 0.3 +
                pattern_factor * 0.2 +
                tf_factor * 0.1
        )

    def calculate_expected_profit(self) -> None:
        """محاسبه سود مورد انتظار"""
        # براساس RR و احتمال موفقیت
        win_probability = 0.5 + (self.confidence_level - 0.5) * 0.3  # 35-65%

        profit_if_win = (self.take_profit - self.entry_price) / self.entry_price
        loss_if_lose = (self.entry_price - self.stop_loss) / self.entry_price

        if self.direction == 'short':
            profit_if_win = (self.entry_price - self.take_profit) / self.entry_price
            loss_if_lose = (self.stop_loss - self.entry_price) / self.entry_price

        self.expected_profit = (win_probability * profit_if_win) - ((1 - win_probability) * loss_if_lose)

    def set_priority(self) -> None:
        """تعیین اولویت سیگنال برای اجرا"""
        # اولویت بالاتر = بهتر
        base_priority = int(self.score.final_score)

        # جایزه برای RR بالا
        if self.risk_reward_ratio >= 3:
            base_priority += 50
        elif self.risk_reward_ratio >= 2.5:
            base_priority += 30
        elif self.risk_reward_ratio >= 2:
            base_priority += 10

        # جایزه برای اطمینان بالا
        base_priority += int(self.confidence_level * 100)

        # جایزه برای سیگنال‌های برگشتی در نقاط کلیدی
        if self.is_reversal:
            base_priority += 20

        self.priority = base_priority

    def is_valid(self) -> bool:
        """بررسی اعتبار سیگنال"""
        # بررسی زمان اعتبار
        if self.timestamp:
            age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
            if age > self.time_validity:
                return False

        # بررسی قیمت‌ها
        if self.stop_loss <= 0 or self.take_profit <= 0 or self.entry_price <= 0:
            return False

        # بررسی منطقی بودن قیمت‌ها
        if self.direction == 'long':
            if self.stop_loss >= self.entry_price or self.take_profit <= self.entry_price:
                return False
        else:  # short
            if self.stop_loss <= self.entry_price or self.take_profit >= self.entry_price:
                return False

        return True


@dataclass
class TradeResult:
    """نتیجه معامله برای سیستم یادگیری تطبیقی"""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str  # 'tp', 'sl', 'manual', 'trailing'
    profit_pct: float
    profit_r: float  # سود/زیان بر حسب R
    market_regime: Optional[str] = None
    pattern_names: List[str] = field(default_factory=list)
    timeframe: str = ""
    signal_score: float = 0.0
    trade_duration: Optional[timedelta] = None
    signal_type: str = ""

    # فیلدهای جدید
    slippage: float = 0.0  # اسلیپیج
    commission: float = 0.0  # کمیسیون
    market_impact: float = 0.0  # تاثیر بر بازار
    actual_profit: float = 0.0  # سود واقعی پس از کسر هزینه‌ها

    def __post_init__(self):
        """محاسبات پس از مقداردهی اولیه"""
        if self.entry_time and self.exit_time and not self.trade_duration:
            self.trade_duration = self.exit_time - self.entry_time

        # محاسبه سود واقعی
        self.calculate_actual_profit()

    def calculate_actual_profit(self) -> None:
        """محاسبه سود واقعی با در نظر گرفتن هزینه‌ها"""
        gross_profit = self.profit_pct / 100

        # کسر هزینه‌ها
        total_costs = self.slippage + self.commission + self.market_impact

        self.actual_profit = gross_profit - total_costs

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        result = asdict(self)
        if self.entry_time:
            result['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            result['exit_time'] = self.exit_time.isoformat()
        if self.trade_duration:
            result['trade_duration'] = str(self.trade_duration)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeResult':
        """ایجاد از دیکشنری"""
        data_copy = data.copy()
        if 'entry_time' in data_copy and isinstance(data_copy['entry_time'], str):
            data_copy['entry_time'] = datetime.fromisoformat(data_copy['entry_time'])
        if 'exit_time' in data_copy and isinstance(data_copy['exit_time'], str):
            data_copy['exit_time'] = datetime.fromisoformat(data_copy['exit_time'])
        if 'trade_duration' in data_copy:
            del data_copy['trade_duration']
        return cls(**data_copy)


# ===============================================
#      تشخیص رژیم بازار بهینه‌شده
# ===============================================
class MarketRegimeDetector:
    """تشخیص رژیم بازار و تطبیق پارامترها - نسخه بهینه‌شده"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('market_regime', {})
        self.enabled = self.config.get('enabled', True)

        # پارامترهای اندیکاتور
        self.adx_period = self.config.get('adx_period', 14)
        self.volatility_period = self.config.get('volatility_period', 20)

        # آستانه‌های تشخیص
        self.strong_trend_threshold = self.config.get('strong_trend_threshold', 25)
        self.weak_trend_threshold = self.config.get('weak_trend_threshold', 20)
        self.high_volatility_threshold = self.config.get('high_volatility_threshold', 1.5)
        self.low_volatility_threshold = self.config.get('low_volatility_threshold', 0.5)

        # کش پیشرفته
        self._regime_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_ttl_seconds = 300
        self._cache_size_limit = 100  # حداکثر اندازه کش

        # داده‌های مورد نیاز
        self._required_samples = max(self.adx_period, self.volatility_period) + 10

        # تاریخچه رژیم‌ها
        self._regime_history = deque(maxlen=20)
        self._regime_transition_probabilities: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        # آنالیز بازار کلی
        self._market_sentiment = 'neutral'
        self._last_sentiment_update = 0

        logger.info(f"تشخیص‌دهنده رژیم بازار فعال شد. وضعیت: {self.enabled}")

    def detect_regime(self, df: DataFrame) -> Dict[str, Any]:
        """تشخیص رژیم بازار با الگوریتم بهینه‌شده"""
        if not self.enabled:
            return {'regime': 'disabled', 'confidence': 1.0, 'details': {}}

        # بررسی کش
        cache_key = self._generate_cache_key(df)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        # حداقل داده مورد نیاز
        if df is None or len(df) < self._required_samples:
            logger.debug(f"داده ناکافی برای تشخیص رژیم: {len(df) if df is not None else 0} ردیف")
            return {'regime': 'unknown_data', 'confidence': 0.0, 'details': {}}

        try:
            # کپی برای جلوگیری از تغییرات ناخواسته
            df_copy = df.copy()

            # آماده‌سازی داده‌ها
            high_prices = df_copy['high'].values.astype(np.float64)
            low_prices = df_copy['low'].values.astype(np.float64)
            close_prices = df_copy['close'].values.astype(np.float64)

            # محاسبه ADX با بهینه‌سازی
            adx, plus_di, minus_di = self._calculate_adx_optimized(
                high_prices, low_prices, close_prices
            )

            # محاسبه ATR% با بهینه‌سازی
            atr_percent = self._calculate_atr_percent_optimized(
                high_prices, low_prices, close_prices
            )

            # یافتن آخرین مقادیر معتبر
            last_valid_idx = self._find_last_valid_index([adx, atr_percent])
            if last_valid_idx is None:
                logger.warning("مقادیر معتبر ADX/ATR یافت نشد")
                return {'regime': 'unknown_calc', 'confidence': 0.0, 'details': {}}

            current_adx = adx[last_valid_idx]
            current_plus_di = plus_di[last_valid_idx]
            current_minus_di = minus_di[last_valid_idx]
            current_atr_percent = atr_percent[last_valid_idx]

            # تشخیص قدرت و جهت روند
            trend_strength = self._determine_trend_strength(current_adx)
            trend_direction = 'bullish' if current_plus_di > current_minus_di else 'bearish'

            # تشخیص سطح نوسان
            volatility = self._determine_volatility_level(current_atr_percent)

            # تعیین رژیم نهایی
            regime = self._determine_final_regime(trend_strength, volatility)

            # محاسبه اطمینان با فاکتورهای بیشتر
            confidence = self._calculate_confidence(
                current_adx, current_atr_percent, adx, atr_percent
            )

            # تحلیل احتمالات انتقال رژیم
            transition_probs = self._analyze_regime_transitions(regime)

            # جزئیات نتیجه
            details = {
                'adx': round(current_adx, 2),
                'plus_di': round(current_plus_di, 2),
                'minus_di': round(current_minus_di, 2),
                'atr_percent': round(current_atr_percent, 3),
                'next_regime_probabilities': transition_probs,
                'regime_stability': self._calculate_regime_stability(),
                'market_sentiment': self._update_market_sentiment(df_copy)
            }

            # به‌روزرسانی تاریخچه
            self._regime_history.append(regime)
            if len(self._regime_history) > 1:
                prev_regime = self._regime_history[-2]
                self._regime_transition_probabilities[prev_regime][regime] += 1

            result = {
                'regime': regime,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'volatility': volatility,
                'confidence': confidence,
                'details': details
            }

            # ذخیره در کش
            self._save_to_cache(cache_key, result)

            logger.debug(
                f"رژیم تشخیص داده شد: {regime}, قدرت: {trend_strength} ({details['adx']}), "
                f"جهت: {trend_direction}, نوسان: {volatility} ({details['atr_percent']}), "
                f"اطمینان: {confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"خطا در تشخیص رژیم بازار: {str(e)}", exc_info=True)
            return {'regime': 'error', 'confidence': 0.0, 'details': {'error': str(e)}}

    def _calculate_adx_optimized(self, high: np.ndarray, low: np.ndarray,
                                 close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """محاسبه بهینه ADX"""
        # استفاده از تابع talib که در C++ نوشته شده و بهینه است
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

        return adx, plus_di, minus_di

    def _calculate_atr_percent_optimized(self, high: np.ndarray, low: np.ndarray,
                                         close: np.ndarray) -> np.ndarray:
        """محاسبه بهینه ATR درصدی"""
        atr = talib.ATR(high, low, close, timeperiod=self.volatility_period)

        # محاسبه درصدی با بهینه‌سازی
        atr_percent = np.where(close > 0, (atr / close) * 100, 0)

        return atr_percent

    def _determine_trend_strength(self, adx: float) -> str:
        """تعیین قدرت روند"""
        if adx > self.strong_trend_threshold:
            return 'strong'
        elif adx > self.weak_trend_threshold:
            return 'weak'
        else:
            return 'no_trend'

    def _determine_volatility_level(self, atr_percent: float) -> str:
        """تعیین سطح نوسان"""
        if atr_percent > self.high_volatility_threshold:
            return 'high'
        elif atr_percent < self.low_volatility_threshold:
            return 'low'
        else:
            return 'normal'

    def _determine_final_regime(self, trend_strength: str, volatility: str) -> str:
        """تعیین رژیم نهایی بازار"""
        if trend_strength == 'strong':
            regime = f'strong_trend_{volatility}'
        elif trend_strength == 'weak':
            regime = f'weak_trend_{volatility}'
        else:
            regime = f'range_{volatility}'

        return regime

    def _calculate_confidence(self, current_adx: float, current_atr: float,
                              adx_series: np.ndarray, atr_series: np.ndarray) -> float:
        """محاسبه اطمینان با فاکتورهای متعدد"""
        # اطمینان براساس فاصله از آستانه‌ها
        adx_confidence = min(1.0, abs(current_adx - self.weak_trend_threshold) /
                             (self.strong_trend_threshold - self.weak_trend_threshold + 1e-6))

        # اطمینان براساس ثبات اندیکاتورها
        adx_stability = 1.0 - (np.std(adx_series[-10:]) / (np.mean(adx_series[-10:]) + 1e-6))
        atr_stability = 1.0 - (np.std(atr_series[-10:]) / (np.mean(atr_series[-10:]) + 1e-6))

        # ترکیب فاکتورها
        confidence = (adx_confidence * 0.5 +
                      adx_stability * 0.25 +
                      atr_stability * 0.25)

        return max(0.1, min(1.0, confidence))

    def _calculate_regime_stability(self) -> float:
        """محاسبه ثبات رژیم فعلی"""
        if len(self._regime_history) < 5:
            return 0.5

        # بررسی تعداد تغییرات رژیم
        changes = sum(1 for i in range(1, len(self._regime_history))
                      if self._regime_history[i] != self._regime_history[i - 1])

        stability = 1.0 - (changes / (len(self._regime_history) - 1))
        return stability

    def _update_market_sentiment(self, df: DataFrame) -> str:
        """به‌روزرسانی احساسات بازار"""
        current_time = time.time()

        # به‌روزرسانی هر 5 دقیقه
        if current_time - self._last_sentiment_update < 300:
            return self._market_sentiment

        try:
            # محاسبه شاخص‌های احساسات
            close_prices = df['close'].values

            # میانگین متحرک‌ها
            ma_short = np.mean(close_prices[-20:])
            ma_long = np.mean(close_prices[-50:])

            # قدرت حرکت
            momentum = (close_prices[-1] - close_prices[-20]) / close_prices[-20]

            # تعیین احساسات
            if ma_short > ma_long and momentum > 0.02:
                self._market_sentiment = 'bullish'
            elif ma_short < ma_long and momentum < -0.02:
                self._market_sentiment = 'bearish'
            else:
                self._market_sentiment = 'neutral'

            self._last_sentiment_update = current_time

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی احساسات بازار: {e}")

        return self._market_sentiment

    def _analyze_regime_transitions(self, current_regime: str) -> Dict[str, float]:
        """تحلیل احتمالات انتقال رژیم"""
        transitions = self._regime_transition_probabilities.get(current_regime, {})
        total = sum(transitions.values())

        if total == 0:
            return {}

        return {
            next_regime: count / total
            for next_regime, count in transitions.items()
        }

    def _find_last_valid_index(self, arrays: List[np.ndarray]) -> Optional[int]:
        """یافتن آخرین ایندکس معتبر در آرایه‌ها"""
        if not arrays:
            return None

        max_len = min(len(arr) for arr in arrays)
        if max_len == 0:
            return None

        # جستجو از انتها به ابتدا
        for i in range(-1, -max_len - 1, -1):
            if all(not np.isnan(arr[i]) for arr in arrays):
                return i

        return None

    def _generate_cache_key(self, df: DataFrame) -> str:
        """تولید کلید کش"""
        if df is None or len(df) == 0:
            return "empty_dataframe"

        try:
            # استفاده از هش برای کارایی بهتر
            last_close = df['close'].iloc[-1]
            last_time = df.index[-1]

            # ترکیب زمان و قیمت برای کلید یکتا
            time_str = str(last_time.timestamp() if hasattr(last_time, 'timestamp') else last_time)
            key = f"{time_str}_{last_close:.6f}_{len(df)}"

            return key
        except Exception:
            return f"dataframe_error_{id(df)}"

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """دریافت از کش با بررسی اعتبار"""
        if key in self._regime_cache:
            result, timestamp = self._regime_cache[key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return result
            else:
                # حذف ورودی منقضی شده
                del self._regime_cache[key]
        return None

    def _save_to_cache(self, key: str, result: Dict[str, Any]) -> None:
        """ذخیره در کش با مدیریت اندازه"""
        # بررسی اندازه کش
        if len(self._regime_cache) >= self._cache_size_limit:
            # حذف قدیمی‌ترین ورودی‌ها
            oldest_keys = sorted(
                self._regime_cache.keys(),
                key=lambda k: self._regime_cache[k][1]
            )[:10]
            for old_key in oldest_keys:
                del self._regime_cache[old_key]

        self._regime_cache[key] = (result, time.time())

    def get_adapted_parameters(self, regime_info: Dict[str, Any],
                               base_config: Dict[str, Any]) -> Dict[str, Any]:
        """تطبیق پارامترها براساس رژیم بازار - نسخه بهینه‌شده"""
        if not self.enabled or regime_info.get('regime') in ['disabled', 'unknown_data', 'error']:
            return base_config

        # کپی عمیق برای جلوگیری از تغییر کانفیگ اصلی
        adapted_config = copy.deepcopy(base_config)

        regime = regime_info.get('regime')
        trend_strength = regime_info.get('trend_strength')
        volatility = regime_info.get('volatility')
        confidence = regime_info.get('confidence', 0.5)
        market_sentiment = regime_info.get('details', {}).get('market_sentiment', 'neutral')

        # دریافت بخش‌های مربوطه
        risk_params = adapted_config.setdefault('risk_management', {})
        signal_params = adapted_config.setdefault('signal_generation', {})

        # مقادیر پایه
        base_risk_percent = base_config.get('risk_management', {}).get('max_risk_per_trade_percent', 1.5)
        base_rr = base_config.get('risk_management', {}).get('preferred_risk_reward_ratio', 2.5)
        base_min_score = base_config.get('signal_generation', {}).get('minimum_signal_score', 180)

        # --- تنظیم پارامترهای ریسک ---
        risk_modifier = self._calculate_risk_modifier(trend_strength, volatility, market_sentiment)
        risk_params['max_risk_per_trade_percent'] = base_risk_percent * risk_modifier

        # --- تنظیم نسبت ریسک به ریوارد ---
        rr_modifier = self._calculate_rr_modifier(trend_strength, volatility)
        risk_params['preferred_risk_reward_ratio'] = base_rr * rr_modifier

        # اطمینان از حداقل RR
        min_rr = base_config.get('risk_management', {}).get('min_risk_reward_ratio', 1.5)
        risk_params['preferred_risk_reward_ratio'] = max(min_rr, risk_params['preferred_risk_reward_ratio'])

        # --- تنظیم حد ضرر ---
        sl_modifier = self._calculate_sl_modifier(volatility)
        base_sl = base_config.get('risk_management', {}).get('default_stop_loss_percent', 1.5)
        risk_params['default_stop_loss_percent'] = base_sl * sl_modifier

        # --- تنظیم حداقل امتیاز سیگنال ---
        score_modifier = self._calculate_score_modifier(trend_strength, volatility, market_sentiment)
        signal_params['minimum_signal_score'] = base_min_score * score_modifier

        # --- پارامترهای اضافی براساس رژیم ---
        if volatility == 'high':
            # در نوسان بالا، محافظه‌کارتر
            signal_params['max_signals_per_day'] = 3
            signal_params['require_volume_confirmation'] = True
        elif trend_strength == 'strong':
            # در روند قوی، سیگنال‌های بیشتر
            signal_params['max_signals_per_day'] = 10
            signal_params['trend_following_weight'] = 1.2

        # گرد کردن مقادیر
        for params in [risk_params, signal_params]:
            for key, value in params.items():
                if isinstance(value, float):
                    params[key] = round(value, 2)

        logger.debug(
            f"رژیم '{regime}' (اطمینان: {confidence:.2f}) -> پارامترهای تطبیقی: "
            f"ریسک%: {risk_params['max_risk_per_trade_percent']:.2f}, "
            f"RR: {risk_params['preferred_risk_reward_ratio']:.2f}, "
            f"حداقل امتیاز: {signal_params['minimum_signal_score']:.2f}"
        )

        return adapted_config

    def _calculate_risk_modifier(self, trend_strength: str, volatility: str,
                                 sentiment: str) -> float:
        """محاسبه ضریب تعدیل ریسک"""
        modifier = 1.0

        # تعدیل براساس قدرت روند
        if trend_strength == 'strong':
            modifier *= 1.1
        elif trend_strength == 'no_trend':
            modifier *= 0.8

        # تعدیل براساس نوسان
        if volatility == 'high':
            modifier *= 0.7
        elif volatility == 'low':
            modifier *= 0.9

        # تعدیل براساس احساسات بازار
        if sentiment == 'bullish':
            modifier *= 1.05
        elif sentiment == 'bearish':
            modifier *= 0.95

        return max(0.5, min(1.5, modifier))

    def _calculate_rr_modifier(self, trend_strength: str, volatility: str) -> float:
        """محاسبه ضریب تعدیل RR"""
        modifier = 1.0

        if trend_strength == 'strong':
            modifier *= 1.2  # RR بالاتر در روند قوی
        elif trend_strength == 'no_trend':
            modifier *= 0.8  # RR پایین‌تر در رنج

        if volatility == 'high':
            modifier *= 1.1  # هدف بزرگتر در نوسان بالا

        return max(0.8, min(1.5, modifier))

    def _calculate_sl_modifier(self, volatility: str) -> float:
        """محاسبه ضریب تعدیل حد ضرر"""
        if volatility == 'high':
            return 1.3  # حد ضرر بزرگتر
        elif volatility == 'low':
            return 0.8  # حد ضرر کوچکتر
        return 1.0

    def _calculate_score_modifier(self, trend_strength: str, volatility: str,
                                  sentiment: str) -> float:
        """محاسبه ضریب تعدیل امتیاز حداقل"""
        modifier = 1.0

        if trend_strength == 'no_trend' or volatility == 'high':
            modifier *= 1.1  # سخت‌گیرانه‌تر

        if sentiment == 'bearish':
            modifier *= 1.05  # کمی سخت‌گیرانه‌تر در بازار نزولی

        return max(0.9, min(1.3, modifier))


# ===============================================
#      سیستم یادگیری تطبیقی بهینه‌شده
# ===============================================
class AdaptiveLearningSystem:
    """سیستم یادگیری تطبیقی برای بهبود پارامترها - نسخه بهینه‌شده"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('adaptive_learning', {})
        self.enabled = self.config.get('enabled', True)
        self.data_file = self.config.get('data_file', 'adaptive_learning_data.json')
        self.max_history_per_symbol = self.config.get('max_history_per_symbol', 200)
        self.learning_rate = self.config.get('learning_rate', 0.1)

        # وزن‌های یادگیری
        self.symbol_performance_weight = self.config.get('symbol_performance_weight', 0.3)
        self.pattern_performance_weight = self.config.get('pattern_performance_weight', 0.3)
        self.regime_performance_weight = self.config.get('regime_performance_weight', 0.2)
        self.timeframe_performance_weight = self.config.get('timeframe_performance_weight', 0.2)

        # داده‌های یادگیری
        self.trade_history: List[TradeResult] = []
        self.symbol_performance: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.pattern_performance: Dict[str, Dict[str, float]] = {}
        self.regime_performance: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.timeframe_performance: Dict[str, Dict[str, Dict[str, float]]] = {}

        # آمار پیشرفته
        self.pattern_combinations: Dict[str, Dict[str, float]] = {}  # ترکیب الگوها
        self.time_of_day_performance: Dict[int, Dict[str, float]] = {}  # عملکرد براساس ساعت
        self.consecutive_patterns: Dict[str, int] = {}  # الگوهای متوالی

        # کش محاسبات
        self._performance_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl_seconds = 3600
        self._cache_size_limit = 500

        # بارگذاری داده‌های موجود
        self._load_data()

        # تایمر ذخیره‌سازی خودکار
        self._last_save_time = time.time()
        self._auto_save_interval = 300  # 5 دقیقه

        logger.info(
            f"سیستم یادگیری تطبیقی فعال شد. وضعیت: {self.enabled}, "
            f"فایل داده: {self.data_file}"
        )

    def _load_data(self) -> None:
        """بارگذاری داده‌های یادگیری با مدیریت خطا"""
        try:
            if not os.path.exists(self.data_file):
                logger.info(f"فایل داده یافت نشد: {self.data_file}، شروع با داده خالی")
                return

            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # بازیابی تاریخچه معاملات
            if 'trade_history' in data:
                self.trade_history = [
                    TradeResult.from_dict(trade)
                    for trade in data['trade_history'][-self.max_history_per_symbol * 10:]
                ]

            # بازیابی داده‌های عملکرد
            self.symbol_performance = data.get('symbol_performance', {})
            self.pattern_performance = data.get('pattern_performance', {})
            self.regime_performance = data.get('regime_performance', {})
            self.timeframe_performance = data.get('timeframe_performance', {})

            # بازیابی آمار پیشرفته
            self.pattern_combinations = data.get('pattern_combinations', {})
            self.time_of_day_performance = data.get('time_of_day_performance', {})

            # تبدیل کلیدهای رشته‌ای به عدد برای time_of_day
            self.time_of_day_performance = {
                int(k): v for k, v in self.time_of_day_performance.items()
            }

            logger.info(
                f"داده‌های یادگیری بارگذاری شد: {len(self.trade_history)} معامله، "
                f"{len(self.symbol_performance)} سمبل، {len(self.pattern_performance)} الگو"
            )

        except Exception as e:
            logger.error(f"خطا در بارگذاری داده‌های یادگیری: {e}", exc_info=True)

    def save_data(self, force: bool = False) -> None:
        """ذخیره داده‌های یادگیری با بهینه‌سازی"""
        if not self.enabled:
            return

        # بررسی نیاز به ذخیره‌سازی
        current_time = time.time()
        if not force and current_time - self._last_save_time < self._auto_save_interval:
            return

        try:
            # ایجاد دایرکتوری در صورت نیاز
            os.makedirs(os.path.dirname(os.path.abspath(self.data_file)), exist_ok=True)

            # آماده‌سازی داده‌ها
            data = {
                'trade_history': [
                    trade.to_dict()
                    for trade in self.trade_history[-self.max_history_per_symbol * 10:]
                ],
                'symbol_performance': self.symbol_performance,
                'pattern_performance': self.pattern_performance,
                'regime_performance': self.regime_performance,
                'timeframe_performance': self.timeframe_performance,
                'pattern_combinations': self.pattern_combinations,
                'time_of_day_performance': self.time_of_day_performance,
                'last_updated': datetime.now().isoformat(),
                'version': '2.0'  # نسخه فرمت داده
            }

            # ذخیره در فایل موقت ابتدا
            temp_file = f"{self.data_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # جابجایی فایل موقت به اصلی
            os.replace(temp_file, self.data_file)

            self._last_save_time = current_time
            logger.info(f"داده‌های یادگیری ذخیره شد در {self.data_file}")

        except Exception as e:
            logger.error(f"خطا در ذخیره داده‌های یادگیری: {e}", exc_info=True)

    def add_trade_result(self, trade_result: TradeResult) -> None:
        """اضافه کردن نتیجه معامله و به‌روزرسانی آمار"""
        if not self.enabled:
            return

        try:
            # اضافه به تاریخچه
            self.trade_history.append(trade_result)

            # محدود کردن اندازه تاریخچه
            if len(self.trade_history) > self.max_history_per_symbol * 10:
                self.trade_history = self.trade_history[-self.max_history_per_symbol * 10:]

            # به‌روزرسانی آمار عملکرد
            self._update_symbol_performance(trade_result)
            self._update_pattern_performance(trade_result)
            self._update_regime_performance(trade_result)
            self._update_timeframe_performance(trade_result)

            # به‌روزرسانی آمار پیشرفته
            self._update_pattern_combinations(trade_result)
            self._update_time_of_day_performance(trade_result)
            self._analyze_consecutive_patterns(trade_result)

            # پاک کردن کش
            self._clear_old_cache()

            # ذخیره خودکار
            if len(self.trade_history) % 10 == 0:
                self.save_data()

            logger.debug(
                f"نتیجه معامله اضافه شد برای {trade_result.symbol}: "
                f"سود R: {trade_result.profit_r:.2f}, خروج: {trade_result.exit_reason}"
            )

        except Exception as e:
            logger.error(f"خطا در اضافه کردن نتیجه معامله: {e}", exc_info=True)

    def _update_symbol_performance(self, trade: TradeResult) -> None:
        """به‌روزرسانی عملکرد سمبل با آمار دقیق‌تر"""
        symbol = trade.symbol
        direction = trade.direction

        # ایجاد ساختار در صورت عدم وجود
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                'long': self._create_empty_performance_dict(),
                'short': self._create_empty_performance_dict(),
                'total': self._create_empty_performance_dict()
            }

        # به‌روزرسانی آمار برای جهت خاص
        self._update_performance_dict(
            self.symbol_performance[symbol][direction], trade
        )

        # به‌روزرسانی آمار کل
        self._update_performance_dict(
            self.symbol_performance[symbol]['total'], trade
        )

    def _update_pattern_performance(self, trade: TradeResult) -> None:
        """به‌روزرسانی عملکرد الگوها"""
        for pattern in trade.pattern_names:
            if pattern not in self.pattern_performance:
                self.pattern_performance[pattern] = self._create_empty_performance_dict()

            self._update_performance_dict(
                self.pattern_performance[pattern], trade
            )

    def _update_regime_performance(self, trade: TradeResult) -> None:
        """به‌روزرسانی عملکرد رژیم‌های بازار"""
        if not trade.market_regime:
            return

        regime = trade.market_regime
        direction = trade.direction

        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                'long': self._create_empty_performance_dict(),
                'short': self._create_empty_performance_dict(),
                'total': self._create_empty_performance_dict()
            }

        self._update_performance_dict(
            self.regime_performance[regime][direction], trade
        )
        self._update_performance_dict(
            self.regime_performance[regime]['total'], trade
        )

    def _update_timeframe_performance(self, trade: TradeResult) -> None:
        """به‌روزرسانی عملکرد تایم‌فریم‌ها"""
        if not trade.timeframe:
            return

        timeframe = trade.timeframe
        direction = trade.direction

        if timeframe not in self.timeframe_performance:
            self.timeframe_performance[timeframe] = {
                'long': self._create_empty_performance_dict(),
                'short': self._create_empty_performance_dict(),
                'total': self._create_empty_performance_dict()
            }

        self._update_performance_dict(
            self.timeframe_performance[timeframe][direction], trade
        )
        self._update_performance_dict(
            self.timeframe_performance[timeframe]['total'], trade
        )

    def _update_pattern_combinations(self, trade: TradeResult) -> None:
        """به‌روزرسانی عملکرد ترکیب الگوها"""
        if len(trade.pattern_names) < 2:
            return

        # ترکیب الگوها به صورت جفتی
        patterns_sorted = sorted(trade.pattern_names)
        for i in range(len(patterns_sorted)):
            for j in range(i + 1, len(patterns_sorted)):
                combo_key = f"{patterns_sorted[i]}+{patterns_sorted[j]}"

                if combo_key not in self.pattern_combinations:
                    self.pattern_combinations[combo_key] = self._create_empty_performance_dict()

                self._update_performance_dict(
                    self.pattern_combinations[combo_key], trade
                )

    def _update_time_of_day_performance(self, trade: TradeResult) -> None:
        """به‌روزرسانی عملکرد براساس ساعت روز"""
        if not trade.entry_time:
            return

        hour = trade.entry_time.hour

        if hour not in self.time_of_day_performance:
            self.time_of_day_performance[hour] = self._create_empty_performance_dict()

        self._update_performance_dict(
            self.time_of_day_performance[hour], trade
        )

    def _analyze_consecutive_patterns(self, trade: TradeResult) -> None:
        """تحلیل الگوهای متوالی موفق/ناموفق"""
        for pattern in trade.pattern_names:
            if pattern not in self.consecutive_patterns:
                self.consecutive_patterns[pattern] = 0

            if trade.profit_r > 0:
                # معامله موفق
                if self.consecutive_patterns[pattern] < 0:
                    self.consecutive_patterns[pattern] = 1
                else:
                    self.consecutive_patterns[pattern] += 1
            else:
                # معامله ناموفق
                if self.consecutive_patterns[pattern] > 0:
                    self.consecutive_patterns[pattern] = -1
                else:
                    self.consecutive_patterns[pattern] -= 1

    def _create_empty_performance_dict(self) -> Dict[str, float]:
        """ایجاد دیکشنری عملکرد خالی"""
        return {
            'count': 0,
            'win_count': 0,
            'avg_profit_r': 0.0,
            'win_rate': 0.0,
            'max_profit_r': 0.0,
            'max_loss_r': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'avg_duration_minutes': 0.0,
            'total_profit_r': 0.0,
            'total_loss_r': 0.0
        }

    def _update_performance_dict(self, perf: Dict[str, float], trade: TradeResult) -> None:
        """به‌روزرسانی دیکشنری عملکرد با معامله جدید"""
        perf['count'] += 1
        is_win = trade.profit_r > 0

        if is_win:
            perf['win_count'] += 1
            perf['total_profit_r'] += trade.profit_r
            perf['max_profit_r'] = max(perf['max_profit_r'], trade.profit_r)
        else:
            perf['total_loss_r'] += abs(trade.profit_r)
            perf['max_loss_r'] = min(perf['max_loss_r'], trade.profit_r)

        # به‌روزرسانی میانگین سود
        perf['avg_profit_r'] = (
                (perf['avg_profit_r'] * (perf['count'] - 1) + trade.profit_r) /
                perf['count']
        )

        # به‌روزرسانی نرخ برد
        perf['win_rate'] = perf['win_count'] / perf['count']

        # محاسبه profit factor
        if perf['total_loss_r'] > 0:
            perf['profit_factor'] = perf['total_profit_r'] / perf['total_loss_r']

        # به‌روزرسانی مدت زمان معامله
        if trade.trade_duration:
            duration_minutes = trade.trade_duration.total_seconds() / 60
            perf['avg_duration_minutes'] = (
                    (perf['avg_duration_minutes'] * (perf['count'] - 1) + duration_minutes) /
                    perf['count']
            )

        # محاسبه Sharpe Ratio (ساده‌شده)
        if perf['count'] >= 10:
            returns = []
            for t in self.trade_history[-10:]:
                if t.symbol == trade.symbol:
                    returns.append(t.profit_r)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    perf['sharpe_ratio'] = avg_return / std_return

    def get_symbol_performance_factor(self, symbol: str, direction: str) -> float:
        """محاسبه فاکتور عملکرد برای سمبل - نسخه بهینه‌شده"""
        if not self.enabled or symbol not in self.symbol_performance:
            return 1.0

        # بررسی کش
        cache_key = f"symbol_{symbol}_{direction}_perf"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            perf = self.symbol_performance[symbol][direction]

            # حداقل معاملات مورد نیاز
            if perf['count'] < 5:
                result = 1.0
            else:
                # محاسبه فاکتور براساس معیارهای مختلف
                win_rate_factor = perf['win_rate'] / 0.5
                profit_factor_score = min(2.0, perf.get('profit_factor', 1.0))
                sharpe_factor = min(2.0, 1.0 + perf.get('sharpe_ratio', 0) * 0.3)

                # ترکیب فاکتورها
                result = (
                        win_rate_factor * 0.4 +
                        profit_factor_score * 0.4 +
                        sharpe_factor * 0.2
                )

                # محدود کردن نتیجه
                result = max(0.5, min(1.5, result))

            self._save_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"خطا در محاسبه فاکتور عملکرد سمبل: {e}", exc_info=True)
            return 1.0

    def get_pattern_performance_factors(self, patterns: List[str]) -> Dict[str, float]:
        """محاسبه فاکتور عملکرد برای الگوها"""
        if not self.enabled or not patterns:
            return {pattern: 1.0 for pattern in patterns}

        result = {}

        for pattern in patterns:
            cache_key = f"pattern_{pattern}_perf"
            cached = self._get_from_cache(cache_key)

            if cached is not None:
                result[pattern] = cached
                continue

            try:
                if pattern not in self.pattern_performance or \
                        self.pattern_performance[pattern]['count'] < 10:
                    factor = 1.0
                else:
                    perf = self.pattern_performance[pattern]

                    # فاکتورهای عملکرد
                    win_rate_factor = perf['win_rate'] / 0.5
                    avg_profit_factor = 1.0 + perf['avg_profit_r'] * 0.3

                    # بررسی الگوهای متوالی
                    consecutive_bonus = 1.0
                    if pattern in self.consecutive_patterns:
                        consec = self.consecutive_patterns[pattern]
                        if consec >= 3:  # 3 برد متوالی
                            consecutive_bonus = 1.1
                        elif consec <= -3:  # 3 باخت متوالی
                            consecutive_bonus = 0.9

                    # محاسبه نهایی
                    factor = (
                            win_rate_factor * 0.5 +
                            avg_profit_factor * 0.3 +
                            consecutive_bonus * 0.2
                    )

                    factor = max(0.5, min(1.5, factor))

                self._save_to_cache(cache_key, factor)
                result[pattern] = factor

            except Exception as e:
                logger.error(f"خطا در محاسبه فاکتور عملکرد الگو {pattern}: {e}")
                result[pattern] = 1.0

        return result

    def get_regime_performance_factor(self, regime: str, direction: str) -> float:
        """محاسبه فاکتور عملکرد برای رژیم بازار"""
        if not self.enabled or not regime or regime not in self.regime_performance:
            return 1.0

        cache_key = f"regime_{regime}_{direction}_perf"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            perf = self.regime_performance[regime][direction]

            if perf['count'] < 10:
                result = 1.0
            else:
                # محاسبه فاکتور
                win_rate_factor = perf['win_rate'] / 0.5
                profit_factor = min(2.0, perf.get('profit_factor', 1.0))

                result = (win_rate_factor * 0.6 + profit_factor * 0.4)
                result = max(0.7, min(1.3, result))

            self._save_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"خطا در محاسبه فاکتور عملکرد رژیم: {e}")
            return 1.0

    def get_timeframe_performance_factor(self, timeframe: str, direction: str) -> float:
        """محاسبه فاکتور عملکرد برای تایم‌فریم"""
        if not self.enabled or not timeframe or timeframe not in self.timeframe_performance:
            return 1.0

        cache_key = f"timeframe_{timeframe}_{direction}_perf"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            perf = self.timeframe_performance[timeframe][direction]

            if perf['count'] < 10:
                result = 1.0
            else:
                # محاسبه فاکتور
                win_rate_factor = perf['win_rate'] / 0.5
                avg_profit_factor = 1.0 + perf['avg_profit_r'] * 0.2

                result = (win_rate_factor * 0.7 + avg_profit_factor * 0.3)
                result = max(0.8, min(1.2, result))

            self._save_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"خطا در محاسبه فاکتور عملکرد تایم‌فریم: {e}")
            return 1.0

    def get_adaptive_pattern_scores(self, pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """محاسبه امتیازات تطبیقی برای الگوها"""
        if not self.enabled:
            return pattern_scores

        cache_key = "adaptive_pattern_scores"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            adjusted_scores = copy.deepcopy(pattern_scores)

            # تنظیم امتیازها براساس عملکرد
            for pattern, score in pattern_scores.items():
                if pattern in self.pattern_performance and \
                        self.pattern_performance[pattern]['count'] >= 10:
                    performance_factor = self.get_pattern_performance_factors([pattern])[pattern]

                    # تنظیم تدریجی با نرخ یادگیری
                    adjusted_score = score * (1.0 + (performance_factor - 1.0) * self.learning_rate)
                    adjusted_scores[pattern] = adjusted_score

            self._save_to_cache(cache_key, adjusted_scores)
            return adjusted_scores

        except Exception as e:
            logger.error(f"خطا در محاسبه امتیازات تطبیقی الگو: {e}")
            return pattern_scores

    def get_adaptive_sl_percent(self, base_sl_percent: float, symbol: str,
                                timeframe: str, direction: str) -> float:
        """محاسبه درصد حد ضرر تطبیقی"""
        if not self.enabled:
            return base_sl_percent

        cache_key = f"adaptive_sl_{symbol}_{timeframe}_{direction}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            # فیلتر معاملات مرتبط
            relevant_trades = [
                t for t in self.trade_history
                if t.symbol == symbol and
                   t.timeframe == timeframe and
                   t.direction == direction and
                   t.exit_reason == 'sl'
            ]

            if len(relevant_trades) < 5:
                return base_sl_percent

            # تحلیل حدضررهای قبلی
            sl_distances = []
            for trade in relevant_trades[-20:]:  # آخرین 20 معامله
                if trade.direction == 'long':
                    sl_distance_percent = (trade.entry_price - trade.stop_loss) / trade.entry_price * 100
                else:
                    sl_distance_percent = (trade.stop_loss - trade.entry_price) / trade.entry_price * 100
                sl_distances.append(sl_distance_percent)

            # محاسبه میانگین و انحراف معیار
            avg_sl = np.mean(sl_distances)
            std_sl = np.std(sl_distances)

            # تنظیم حد ضرر
            # اگر انحراف معیار بالاست، از میانه استفاده کن
            if std_sl > avg_sl * 0.3:
                target_sl = np.median(sl_distances)
            else:
                target_sl = avg_sl

            # تنظیم تدریجی
            adjusted_sl_percent = base_sl_percent + (target_sl - base_sl_percent) * self.learning_rate

            # محدود کردن تغییرات
            adjusted_sl_percent = max(
                base_sl_percent * 0.7,
                min(base_sl_percent * 1.3, adjusted_sl_percent)
            )

            self._save_to_cache(cache_key, adjusted_sl_percent)
            return adjusted_sl_percent

        except Exception as e:
            logger.error(f"خطا در محاسبه SL تطبیقی: {e}")
            return base_sl_percent

    def get_best_time_of_day(self, symbol: str = None) -> List[int]:
        """دریافت بهترین ساعات روز برای معامله"""
        if not self.time_of_day_performance:
            return list(range(24))  # همه ساعات

        # فیلتر براساس سمبل اگر مشخص شده
        if symbol:
            relevant_hours = {}
            for hour, perf in self.time_of_day_performance.items():
                # باید معاملات این سمبل را در این ساعت بررسی کنیم
                symbol_trades = [
                    t for t in self.trade_history
                    if t.symbol == symbol and t.entry_time.hour == hour
                ]
                if len(symbol_trades) >= 3:
                    relevant_hours[hour] = perf
        else:
            relevant_hours = {
                h: p for h, p in self.time_of_day_performance.items()
                if p['count'] >= 5
            }

        if not relevant_hours:
            return list(range(24))

        # رتبه‌بندی براساس عملکرد
        hour_scores = {}
        for hour, perf in relevant_hours.items():
            score = (
                    perf['win_rate'] * 0.5 +
                    (1.0 + perf['avg_profit_r']) * 0.3 +
                    perf.get('profit_factor', 1.0) * 0.2
            )
            hour_scores[hour] = score

        # بازگشت بهترین ساعات
        sorted_hours = sorted(hour_scores.items(), key=lambda x: x[1], reverse=True)
        best_hours = [h for h, s in sorted_hours if s > 1.0]

        return best_hours if best_hours else list(range(24))

    def get_pattern_combination_score(self, patterns: List[str]) -> float:
        """محاسبه امتیاز ترکیب الگوها"""
        if len(patterns) < 2:
            return 1.0

        total_score = 0.0
        count = 0

        patterns_sorted = sorted(patterns)
        for i in range(len(patterns_sorted)):
            for j in range(i + 1, len(patterns_sorted)):
                combo_key = f"{patterns_sorted[i]}+{patterns_sorted[j]}"

                if combo_key in self.pattern_combinations:
                    perf = self.pattern_combinations[combo_key]
                    if perf['count'] >= 5:
                        score = perf['win_rate'] * 2.0
                        total_score += score
                        count += 1

        if count == 0:
            return 1.0

        avg_score = total_score / count
        return max(0.8, min(1.2, avg_score))

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """دریافت از کش"""
        if key in self._performance_cache:
            value, timestamp = self._performance_cache[key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return value
            else:
                del self._performance_cache[key]
        return None

    def _save_to_cache(self, key: str, value: Any) -> None:
        """ذخیره در کش"""
        self._performance_cache[key] = (value, time.time())

        # مدیریت اندازه کش
        if len(self._performance_cache) > self._cache_size_limit:
            self._clear_old_cache()

    def _clear_old_cache(self) -> None:
        """پاک کردن ورودی‌های قدیمی کش"""
        current_time = time.time()

        # حذف ورودی‌های منقضی شده
        expired_keys = [
            k for k, (v, t) in self._performance_cache.items()
            if current_time - t > self._cache_ttl_seconds
        ]

        for key in expired_keys:
            del self._performance_cache[key]

        # اگر هنوز زیاد است، قدیمی‌ترین‌ها را حذف کن
        if len(self._performance_cache) > self._cache_size_limit:
            sorted_keys = sorted(
                self._performance_cache.keys(),
                key=lambda k: self._performance_cache[k][1]
            )

            # حذف 20% قدیمی‌ترین
            to_remove = int(self._cache_size_limit * 0.2)
            for key in sorted_keys[:to_remove]:
                del self._performance_cache[key]

    def generate_performance_report(self) -> Dict[str, Any]:
        """تولید گزارش عملکرد کامل"""
        report = {
            'summary': {
                'total_trades': len(self.trade_history),
                'total_symbols': len(self.symbol_performance),
                'total_patterns': len(self.pattern_performance),
                'last_update': datetime.now().isoformat()
            },
            'top_symbols': self._get_top_performers(self.symbol_performance, 'total'),
            'top_patterns': self._get_top_performers(self.pattern_performance),
            'best_hours': self.get_best_time_of_day(),
            'regime_analysis': self._analyze_regime_performance(),
            'pattern_combinations': self._get_best_pattern_combinations()
        }

        return report

    def _get_top_performers(self, data: Dict, key: str = None) -> List[Dict[str, Any]]:
        """دریافت بهترین عملکردها"""
        performers = []

        for name, perf in data.items():
            if key and isinstance(perf, dict) and key in perf:
                perf_data = perf[key]
            else:
                perf_data = perf

            if isinstance(perf_data, dict) and perf_data.get('count', 0) >= 5:
                performers.append({
                    'name': name,
                    'win_rate': perf_data.get('win_rate', 0),
                    'avg_profit_r': perf_data.get('avg_profit_r', 0),
                    'profit_factor': perf_data.get('profit_factor', 0),
                    'count': perf_data.get('count', 0)
                })

        # رتبه‌بندی براساس profit factor
        performers.sort(key=lambda x: x.get('profit_factor', 0), reverse=True)

        return performers[:10]  # 10 مورد برتر

    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """تحلیل عملکرد رژیم‌های بازار"""
        analysis = {}

        for regime, data in self.regime_performance.items():
            if data['total']['count'] >= 5:
                analysis[regime] = {
                    'total_trades': data['total']['count'],
                    'win_rate': data['total']['win_rate'],
                    'best_direction': 'long' if data['long'].get('win_rate', 0) > data['short'].get('win_rate',
                                                                                                    0) else 'short',
                    'avg_profit_r': data['total']['avg_profit_r']
                }

        return analysis

    def _get_best_pattern_combinations(self) -> List[Dict[str, Any]]:
        """دریافت بهترین ترکیبات الگو"""
        combinations = []

        for combo, perf in self.pattern_combinations.items():
            if perf['count'] >= 5:
                combinations.append({
                    'combination': combo,
                    'win_rate': perf['win_rate'],
                    'avg_profit_r': perf['avg_profit_r'],
                    'count': perf['count']
                })

        combinations.sort(key=lambda x: x['win_rate'], reverse=True)

        return combinations[:10]


# ===============================================
#      مدیریت همبستگی بهینه‌شده
# ===============================================
class CorrelationManager:
    """مدیریت همبستگی بین سمبل‌ها برای تنوع‌بخشی - نسخه بهینه‌شده"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('correlation_management', {})
        self.enabled = self.config.get('enabled', True)
        self.data_file = self.config.get('data_file', 'correlation_data.json')
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.max_exposure_per_group = self.config.get('max_exposure_per_group', 3)
        self.update_interval = self.config.get('update_interval', 86400)  # 24 ساعت
        self.lookback_periods = self.config.get('lookback_periods', 100)

        # ساختارهای داده
        self.correlation_groups: Dict[str, List[str]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.last_update_time = 0
        self.active_positions: Dict[str, Dict[str, Any]] = {}

        # تنظیمات همبستگی با بیت‌کوین
        self.btc_correlation_enabled = self.config.get('btc_correlation_enabled', True)
        self.btc_positive_threshold = self.config.get('btc_positive_threshold', 0.5)
        self.btc_negative_threshold = self.config.get('btc_negative_threshold', -0.5)
        self.btc_trend_data = {}

        # کش محاسبات
        self._correlation_cache: Dict[str, Tuple[float, float]] = {}
        self._cache_ttl = 3600  # 1 ساعت

        # بارگذاری داده‌های موجود
        self._load_data()

        logger.info(
            f"مدیر همبستگی فعال شد. وضعیت: {self.enabled}, "
            f"آستانه همبستگی: {self.correlation_threshold}"
        )

    def _load_data(self) -> None:
        """بارگذاری داده‌های همبستگی"""
        try:
            if not os.path.exists(self.data_file):
                logger.info(f"فایل داده همبستگی یافت نشد: {self.data_file}")
                return

            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.correlation_matrix = data.get('correlation_matrix', {})
            self.correlation_groups = data.get('correlation_groups', {})
            self.last_update_time = data.get('last_update_time', 0)
            self.btc_trend_data = data.get('btc_trend_data', {})

            logger.info(
                f"داده‌های همبستگی بارگذاری شد: {len(self.correlation_matrix)} سمبل، "
                f"{len(self.correlation_groups)} گروه"
            )

        except Exception as e:
            logger.error(f"خطا در بارگذاری داده‌های همبستگی: {e}")

    def save_data(self) -> None:
        """ذخیره داده‌های همبستگی"""
        if not self.enabled:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.data_file)), exist_ok=True)

            data = {
                'correlation_matrix': self.correlation_matrix,
                'correlation_groups': self.correlation_groups,
                'last_update_time': self.last_update_time,
                'btc_trend_data': self.btc_trend_data,
                'update_timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }

            temp_file = f"{self.data_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            os.replace(temp_file, self.data_file)

            logger.info(f"داده‌های همبستگی ذخیره شد در {self.data_file}")

        except Exception as e:
            logger.error(f"خطا در ذخیره داده‌های همبستگی: {e}")

    def update_correlations(self, symbols_data: Dict[str, pd.DataFrame]) -> None:
        """به‌روزرسانی ماتریس همبستگی - نسخه بهینه‌شده"""
        if not self.enabled or len(symbols_data) < 2:
            return

        # بررسی نیاز به به‌روزرسانی
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            logger.debug("همبستگی‌ها اخیراً به‌روزرسانی شده‌اند")
            return

        try:
            logger.info(f"به‌روزرسانی همبستگی برای {len(symbols_data)} سمبل...")

            # استخراج قیمت‌های بسته
            symbol_prices = {}
            for symbol, df in symbols_data.items():
                if df is not None and len(df) >= self.lookback_periods:
                    # استفاده از بازده‌ها به جای قیمت برای همبستگی بهتر
                    close_prices = df['close'].iloc[-self.lookback_periods:].values
                    returns = np.diff(np.log(close_prices))
                    symbol_prices[symbol] = returns

            # محاسبه همبستگی با استفاده از numpy برای سرعت بهتر
            self._calculate_correlation_matrix_optimized(symbol_prices)

            # به‌روزرسانی گروه‌های همبستگی
            self._update_correlation_groups_optimized()

            # به‌روزرسانی زمان
            self.last_update_time = current_time

            # ذخیره داده‌ها
            self.save_data()

            logger.info(
                f"همبستگی‌ها به‌روزرسانی شد: {len(self.correlation_matrix)} سمبل، "
                f"{len(self.correlation_groups)} گروه"
            )

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی همبستگی‌ها: {e}")

    def _calculate_correlation_matrix_optimized(self, returns_data: Dict[str, np.ndarray]) -> None:
        """محاسبه بهینه ماتریس همبستگی"""
        symbols = list(returns_data.keys())
        n_symbols = len(symbols)

        if n_symbols < 2:
            return

        # ایجاد ماتریس بازده‌ها
        returns_matrix = np.column_stack([returns_data[s] for s in symbols])

        # محاسبه ماتریس همبستگی با numpy
        correlation_matrix = np.corrcoef(returns_matrix.T)

        # تبدیل به دیکشنری
        new_correlation_matrix = {}
        for i, symbol1 in enumerate(symbols):
            new_correlation_matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    corr = correlation_matrix[i, j]
                    if not np.isnan(corr):
                        new_correlation_matrix[symbol1][symbol2] = float(corr)
                    else:
                        new_correlation_matrix[symbol1][symbol2] = 0.0
                else:
                    new_correlation_matrix[symbol1][symbol2] = 1.0

        self.correlation_matrix = new_correlation_matrix

    def _update_correlation_groups_optimized(self) -> None:
        """به‌روزرسانی گروه‌های همبستگی با الگوریتم بهینه"""
        try:
            self.correlation_groups = {}

            if not self.correlation_matrix:
                return

            symbols = list(self.correlation_matrix.keys())
            if not symbols:
                return

            # استفاده از Union-Find برای گروه‌بندی کارآمد
            parent = {s: s for s in symbols}

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # ایجاد گروه‌ها براساس همبستگی
            for symbol1 in symbols:
                for symbol2, corr in self.correlation_matrix[symbol1].items():
                    if abs(corr) >= self.correlation_threshold:
                        union(symbol1, symbol2)

            # جمع‌آوری گروه‌ها
            groups = defaultdict(list)
            for symbol in symbols:
                root = find(symbol)
                groups[root].append(symbol)

            # ذخیره گروه‌های با بیش از یک عضو
            group_id = 0
            for members in groups.values():
                if len(members) > 1:
                    self.correlation_groups[f"group_{group_id}"] = sorted(members)
                    group_id += 1

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی گروه‌های همبستگی: {e}")

    def get_correlated_symbols(self, symbol: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """دریافت سمبل‌های همبسته"""
        if not self.enabled or symbol not in self.correlation_matrix:
            return []

        # بررسی کش
        cache_key = f"{symbol}_{threshold}"
        current_time = time.time()

        if cache_key in self._correlation_cache:
            correlated, timestamp = self._correlation_cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                return correlated

        try:
            corr_threshold = threshold if threshold is not None else self.correlation_threshold
            correlated = []

            for other_symbol, corr in self.correlation_matrix[symbol].items():
                if other_symbol != symbol and abs(corr) >= corr_threshold:
                    correlated.append((other_symbol, corr))

            # مرتب‌سازی براساس قدر مطلق همبستگی
            correlated.sort(key=lambda x: abs(x[1]), reverse=True)

            # ذخیره در کش
            self._correlation_cache[cache_key] = (correlated, current_time)

            return correlated

        except Exception as e:
            logger.error(f"خطا در دریافت سمبل‌های همبسته برای {symbol}: {e}")
            return []

    async def analyze_btc_trend(self, data_fetcher) -> Dict[str, Any]:
        """تحلیل روند بیت‌کوین برای استفاده در فیلترینگ سیگنال‌ها"""
        if not self.btc_correlation_enabled:
            return {'trend': 'neutral', 'strength': 0.0, 'last_price': 0.0}

        try:
            # دریافت داده‌های بیت‌کوین
            btc_data = await data_fetcher.fetch_data('BTCUSDT', ['15m', '1h', '4h'])

            if not btc_data or '1h' not in btc_data:
                return {'trend': 'unknown', 'strength': 0.0, 'last_price': 0.0}

            df = btc_data['1h']
            if df is None or len(df) < 50:
                return {'trend': 'unknown', 'strength': 0.0, 'last_price': 0.0}

            # محاسبه اندیکاتورها
            close_prices = df['close'].values
            ema20 = talib.EMA(close_prices, timeperiod=20)
            ema50 = talib.EMA(close_prices, timeperiod=50)
            rsi = talib.RSI(close_prices, timeperiod=14)

            # تشخیص روند
            last_close = close_prices[-1]
            last_ema20 = ema20[-1]
            last_ema50 = ema50[-1]
            last_rsi = rsi[-1]

            trend = 'neutral'
            strength = 0.0

            if last_ema20 > last_ema50 and last_close > last_ema20:
                trend = 'bullish'
                strength = min(1.0, (last_close - last_ema50) / last_ema50 * 10)
            elif last_ema20 < last_ema50 and last_close < last_ema20:
                trend = 'bearish'
                strength = min(1.0, (last_ema50 - last_close) / last_ema50 * 10)

            # تعدیل قدرت براساس RSI
            if last_rsi > 70:
                if trend == 'bullish':
                    strength *= 0.8  # احتمال اصلاح
            elif last_rsi < 30:
                if trend == 'bearish':
                    strength *= 0.8  # احتمال ریباند

            # ذخیره در داده‌های روند
            self.btc_trend_data = {
                'trend': trend,
                'strength': strength,
                'last_price': last_close,
                'last_update': datetime.now().isoformat(),
                'ema20': last_ema20,
                'ema50': last_ema50,
                'rsi': last_rsi
            }

            return self.btc_trend_data

        except Exception as e:
            logger.error(f"خطا در تحلیل روند بیت‌کوین: {e}")
            return {'trend': 'error', 'strength': 0.0, 'last_price': 0.0}

    async def check_btc_correlation_compatibility(self, symbol: str, direction: str,
                                                  data_fetcher) -> Dict[str, Any]:
        """بررسی سازگاری سیگنال با روند بیت‌کوین"""
        if not self.btc_correlation_enabled or symbol == 'BTCUSDT':
            return {'is_compatible': True, 'reason': 'btc_check_disabled'}

        try:
            # به‌روزرسانی روند بیت‌کوین اگر قدیمی است
            if 'last_update' not in self.btc_trend_data or \
                    (datetime.now() - datetime.fromisoformat(self.btc_trend_data['last_update'])).seconds > 3600:
                await self.analyze_btc_trend(data_fetcher)

            btc_trend = self.btc_trend_data.get('trend', 'neutral')
            btc_strength = self.btc_trend_data.get('strength', 0.0)

            # دریافت همبستگی با بیت‌کوین
            correlation_with_btc = 0.0
            if symbol in self.correlation_matrix and 'BTCUSDT' in self.correlation_matrix[symbol]:
                correlation_with_btc = self.correlation_matrix[symbol]['BTCUSDT']

            # تعیین نوع همبستگی
            correlation_type = 'zero'
            if correlation_with_btc > self.btc_positive_threshold:
                correlation_type = 'positive'
            elif correlation_with_btc < self.btc_negative_threshold:
                correlation_type = 'inverse'

            # بررسی سازگاری
            is_compatible = True
            reason = 'compatible'

            # قوانین سازگاری
            if btc_trend == 'bearish' and btc_strength > 0.5:
                if direction == 'long' and correlation_type == 'positive':
                    # Long در ارز با همبستگی مثبت در بازار نزولی بیت‌کوین
                    is_compatible = False
                    reason = 'long_positive_corr_in_bearish_btc'
                elif direction == 'short' and correlation_type == 'inverse':
                    # Short در ارز با همبستگی معکوس در بازار نزولی بیت‌کوین
                    is_compatible = False
                    reason = 'short_inverse_corr_in_bearish_btc'

            elif btc_trend == 'bullish' and btc_strength > 0.5:
                if direction == 'short' and correlation_type == 'positive':
                    # Short در ارز با همبستگی مثبت در بازار صعودی بیت‌کوین
                    is_compatible = False
                    reason = 'short_positive_corr_in_bullish_btc'
                elif direction == 'long' and correlation_type == 'inverse':
                    # Long در ارز با همبستگی معکوس در بازار صعودی بیت‌کوین
                    is_compatible = False
                    reason = 'long_inverse_corr_in_bullish_btc'

            return {
                'is_compatible': is_compatible,
                'reason': reason,
                'btc_trend': btc_trend,
                'btc_strength': btc_strength,
                'correlation_with_btc': correlation_with_btc,
                'correlation_type': correlation_type
            }

        except Exception as e:
            logger.error(f"خطا در بررسی سازگاری با بیت‌کوین: {e}")
            return {'is_compatible': True, 'reason': f'error: {str(e)}'}

    def update_active_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """به‌روزرسانی لیست پوزیشن‌های فعال"""
        if not self.enabled:
            return

        self.active_positions = positions

        # تحلیل تمرکز در گروه‌ها
        group_exposure = self._analyze_group_exposure()
        if group_exposure:
            logger.info(f"وضعیت تمرکز در گروه‌های همبستگی: {group_exposure}")

    def _analyze_group_exposure(self) -> Dict[str, int]:
        """تحلیل میزان تمرکز در گروه‌های همبستگی"""
        group_exposure = {}

        for group_id, symbols in self.correlation_groups.items():
            exposure_count = sum(
                1 for symbol in symbols
                if symbol in self.active_positions
            )

            if exposure_count > 0:
                group_exposure[group_id] = exposure_count

        return group_exposure

    def get_correlation_safety_factor(self, symbol: str, direction: str) -> float:
        """محاسبه فاکتور ایمنی همبستگی - نسخه بهینه‌شده"""
        if not self.enabled or not self.active_positions:
            return 1.0

        try:
            # یافتن گروه همبستگی سمبل
            symbol_group = None
            for group_id, group_symbols in self.correlation_groups.items():
                if symbol in group_symbols:
                    symbol_group = group_id
                    break

            if not symbol_group:
                return 1.0  # سمبل در هیچ گروهی نیست

            # شمارش پوزیشن‌های فعال در گروه
            group_positions = []
            for pos_symbol, pos_info in self.active_positions.items():
                if pos_symbol in self.correlation_groups.get(symbol_group, []):
                    pos_direction = pos_info.get('direction', '')
                    if direction == pos_direction:
                        group_positions.append(pos_symbol)

            group_count = len(group_positions)

            # محاسبه فاکتور ایمنی پیشرفته
            if group_count >= self.max_exposure_per_group:
                safety_factor = 0.3  # کاهش شدید امتیاز
            else:
                # کاهش تدریجی
                reduction_rate = 0.7 / self.max_exposure_per_group
                safety_factor = 1.0 - (reduction_rate * group_count)

            # در نظر گرفتن قدرت همبستگی
            if group_count > 0:
                # محاسبه میانگین همبستگی با سمبل‌های موجود
                avg_correlation = 0.0
                count = 0

                for pos_symbol in group_positions:
                    if pos_symbol in self.correlation_matrix.get(symbol, {}):
                        avg_correlation += abs(self.correlation_matrix[symbol][pos_symbol])
                        count += 1

                if count > 0:
                    avg_correlation /= count
                    # تعدیل فاکتور براساس قدرت همبستگی
                    correlation_penalty = (avg_correlation - self.correlation_threshold) * 2
                    safety_factor *= (1.0 - correlation_penalty)

            # محدود کردن فاکتور
            safety_factor = max(0.1, min(1.0, safety_factor))

            if safety_factor < 1.0:
                logger.debug(
                    f"فاکتور ایمنی همبستگی برای {symbol}: {safety_factor:.2f} "
                    f"(گروه {symbol_group}: {group_count} پوزیشن فعال)"
                )

            return safety_factor

        except Exception as e:
            logger.error(f"خطا در محاسبه فاکتور ایمنی همبستگی برای {symbol}: {e}")
            return 1.0

    def get_portfolio_correlation_matrix(self) -> pd.DataFrame:
        """دریافت ماتریس همبستگی پورتفولیو فعلی"""
        if not self.active_positions:
            return pd.DataFrame()

        active_symbols = list(self.active_positions.keys())
        n = len(active_symbols)

        if n < 2:
            return pd.DataFrame()

        # ایجاد ماتریس
        corr_matrix = np.eye(n)

        for i, symbol1 in enumerate(active_symbols):
            for j, symbol2 in enumerate(active_symbols):
                if i != j and symbol1 in self.correlation_matrix:
                    corr = self.correlation_matrix[symbol1].get(symbol2, 0)
                    corr_matrix[i, j] = corr

        return pd.DataFrame(corr_matrix, index=active_symbols, columns=active_symbols)

    def suggest_diversification(self, new_signals: List[SignalInfo]) -> List[SignalInfo]:
        """پیشنهاد سیگنال‌ها برای تنوع‌بخشی بهتر"""
        if not self.enabled or not new_signals:
            return new_signals

        # محاسبه همبستگی پورتفولیو فعلی
        current_portfolio_corr = self.get_portfolio_correlation_matrix()

        # امتیازدهی به سیگنال‌ها براساس تنوع‌بخشی
        scored_signals = []

        for signal in new_signals:
            diversification_score = self._calculate_diversification_score(
                signal.symbol,
                current_portfolio_corr
            )

            # تعدیل امتیاز نهایی
            adjusted_score = signal.score.final_score * diversification_score

            scored_signals.append((signal, adjusted_score, diversification_score))

        # مرتب‌سازی براساس امتیاز تعدیل شده
        scored_signals.sort(key=lambda x: x[1], reverse=True)

        # انتخاب بهترین سیگنال‌ها با در نظر گرفتن تنوع
        selected_signals = []
        selected_symbols = set(self.active_positions.keys())

        for signal, adj_score, div_score in scored_signals:
            # بررسی افزودن به تنوع
            if self._improves_diversification(signal.symbol, selected_symbols):
                selected_signals.append(signal)
                selected_symbols.add(signal.symbol)

                logger.info(
                    f"سیگنال {signal.symbol} انتخاب شد "
                    f"(امتیاز تنوع: {div_score:.2f})"
                )

        return selected_signals

    def _calculate_diversification_score(self, symbol: str,
                                         current_portfolio: pd.DataFrame) -> float:
        """محاسبه امتیاز تنوع‌بخشی برای سمبل جدید"""
        if current_portfolio.empty:
            return 1.0

        # محاسبه میانگین همبستگی با پورتفولیو موجود
        correlations = []

        for active_symbol in current_portfolio.index:
            if active_symbol in self.correlation_matrix.get(symbol, {}):
                corr = abs(self.correlation_matrix[symbol][active_symbol])
                correlations.append(corr)

        if not correlations:
            return 1.0  # بدون همبستگی = تنوع خوب

        avg_correlation = np.mean(correlations)

        # امتیاز بیشتر برای همبستگی کمتر
        diversity_score = 1.0 - avg_correlation

        return max(0.5, diversity_score)

    def _improves_diversification(self, symbol: str,
                                  selected_symbols: Set[str]) -> bool:
        """بررسی اینکه آیا افزودن سمبل به تنوع کمک می‌کند"""
        if not selected_symbols:
            return True

        # محاسبه حداکثر همبستگی با سمبل‌های انتخاب شده
        max_correlation = 0.0

        for selected in selected_symbols:
            if selected in self.correlation_matrix.get(symbol, {}):
                corr = abs(self.correlation_matrix[symbol][selected])
                max_correlation = max(max_correlation, corr)

        # اگر همبستگی زیاد است، تنوع را بهبود نمی‌دهد
        return max_correlation < self.correlation_threshold * 0.8

# ===============================================
#      قطع‌کننده اضطراری بهینه‌شده
# ===============================================

class EmergencyCircuitBreaker:
    """مکانیزم توقف اضطراری برای جلوگیری از ضررهای متوالی - نسخه بهینه‌شده"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('circuit_breaker', {})
        self.enabled = self.config.get('enabled', True)
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        self.max_daily_losses_r = self.config.get('max_daily_losses_r', 5.0)
        self.max_hourly_losses_r = self.config.get('max_hourly_losses_r', 3.0)
        self.cool_down_period_minutes = self.config.get('cool_down_period_minutes', 60)
        self.reset_period_hours = self.config.get('reset_period_hours', 24)

        # متغیرهای داخلی
        self.consecutive_losses = 0
        self.daily_loss_r = 0.0
        self.hourly_loss_r = 0.0
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = ""
        self.last_reset_time = datetime.now(timezone.utc)
        self.last_hour_reset = datetime.now(timezone.utc)
        self.trade_log: deque = deque(maxlen=1000)

        # آمار پیشرفته
        self.market_anomaly_triggers = 0
        self.loss_velocity = 0.0  # سرعت ضرر
        self.recovery_mode = False
        self.performance_metrics = {
            'total_triggers': 0,
            'false_triggers': 0,
            'prevented_losses': 0.0
        }

        logger.info(
            f"قطع‌کننده اضطراری فعال شد. "
            f"حداکثر ضرر متوالی: {self.max_consecutive_losses}, "
            f"حداکثر ضرر روزانه: {self.max_daily_losses_r}R"
        )

    def add_trade_result(self, trade_result: TradeResult) -> None:
        """ثبت نتیجه معامله و بررسی شرایط توقف"""
        if not self.enabled:
            return

        try:
            current_time = datetime.now(timezone.utc)

            # بررسی نیاز به ریست آمار
            self._check_reset_periods(current_time)

            # ثبت معامله
            trade_info = {
                'time': current_time,
                'symbol': trade_result.symbol,
                'direction': trade_result.direction,
                'profit_r': trade_result.profit_r,
                'exit_reason': trade_result.exit_reason,
                'duration': trade_result.trade_duration
            }
            self.trade_log.append(trade_info)

            # به‌روزرسانی آمار
            self._update_statistics(trade_result)

            # محاسبه سرعت ضرر
            self._calculate_loss_velocity()

            # بررسی شرایط توقف
            self._check_trigger_conditions()

            # حالت بازیابی
            if self.recovery_mode and trade_result.profit_r > 0.5:
                self._check_recovery_progress()

            logger.debug(
                f"وضعیت قطع‌کننده: ضرر متوالی={self.consecutive_losses}, "
                f"ضرر روزانه={self.daily_loss_r:.2f}R, "
                f"ضرر ساعتی={self.hourly_loss_r:.2f}R, "
                f"فعال={self.triggered}"
            )

        except Exception as e:
            logger.error(f"خطا در پردازش نتیجه معامله در قطع‌کننده: {e}")

    def _check_reset_periods(self, current_time: datetime) -> None:
        """بررسی و ریست دوره‌های زمانی"""
        # ریست روزانه
        hours_since_reset = (current_time - self.last_reset_time).total_seconds() / 3600
        if hours_since_reset >= self.reset_period_hours:
            self._reset_daily_stats()

        # ریست ساعتی
        minutes_since_hour = (current_time - self.last_hour_reset).total_seconds() / 60
        if minutes_since_hour >= 60:
            self.hourly_loss_r = 0.0
            self.last_hour_reset = current_time

    def _update_statistics(self, trade_result: TradeResult) -> None:
        """به‌روزرسانی آمار معاملات"""
        if trade_result.profit_r < 0:
            # معامله ضررده
            self.consecutive_losses += 1
            loss_amount = abs(trade_result.profit_r)
            self.daily_loss_r += loss_amount
            self.hourly_loss_r += loss_amount
        else:
            # معامله سودده
            self.consecutive_losses = 0
            # کاهش ضرر تجمعی
            profit_amount = trade_result.profit_r
            self.daily_loss_r = max(0, self.daily_loss_r - profit_amount * 0.5)
            self.hourly_loss_r = max(0, self.hourly_loss_r - profit_amount * 0.5)

    def _calculate_loss_velocity(self) -> None:
        """محاسبه سرعت ضرر"""
        # بررسی ضررهای 30 دقیقه اخیر
        thirty_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=30)
        recent_losses = [
            t['profit_r'] for t in self.trade_log
            if t['time'] > thirty_minutes_ago and t['profit_r'] < 0
        ]

        if len(recent_losses) >= 2:
            # ضرر در واحد زمان
            total_loss = sum(abs(loss) for loss in recent_losses)
            time_span = 30  # دقیقه
            self.loss_velocity = total_loss / time_span * 60  # ضرر در ساعت
        else:
            self.loss_velocity = 0.0

    def _check_trigger_conditions(self) -> None:
        """بررسی شرایط فعال‌سازی قطع‌کننده"""
        trigger_reasons = []

        # بررسی ضرر متوالی
        if self.consecutive_losses >= self.max_consecutive_losses:
            trigger_reasons.append(f"{self.consecutive_losses} ضرر متوالی")

        # بررسی ضرر روزانه
        if self.daily_loss_r >= self.max_daily_losses_r:
            trigger_reasons.append(f"ضرر روزانه {self.daily_loss_r:.2f}R")

        # بررسی ضرر ساعتی
        if self.hourly_loss_r >= self.max_hourly_losses_r:
            trigger_reasons.append(f"ضرر ساعتی {self.hourly_loss_r:.2f}R")

        # بررسی سرعت ضرر
        if self.loss_velocity > 2.0:  # بیش از 2R در ساعت
            trigger_reasons.append(f"سرعت ضرر بالا: {self.loss_velocity:.2f}R/h")

        if trigger_reasons and not self.triggered:
            self._trigger_circuit_breaker(" + ".join(trigger_reasons))

    def _trigger_circuit_breaker(self, reason: str) -> None:
        """فعال کردن توقف اضطراری"""
        if self.triggered:
            return

        self.triggered = True
        self.trigger_time = datetime.now(timezone.utc)
        self.trigger_reason = reason
        self.performance_metrics['total_triggers'] += 1

        # تعیین مدت توقف براساس شدت
        if "سرعت ضرر بالا" in reason:
            self.cool_down_period_minutes = 120  # 2 ساعت
        elif self.daily_loss_r > self.max_daily_losses_r * 1.5:
            self.cool_down_period_minutes = 180  # 3 ساعت

        logger.warning(
            f"🚨 قطع‌کننده اضطراری فعال شد: {reason}. "
            f"معاملات متوقف شد برای {self.cool_down_period_minutes} دقیقه."
        )

        # فعال کردن حالت بازیابی
        self.recovery_mode = True

    def _reset_daily_stats(self) -> None:
        """ریست آمار روزانه"""
        self.daily_loss_r = 0.0
        self.hourly_loss_r = 0.0
        self.last_reset_time = datetime.now(timezone.utc)

        # پاکسازی معاملات قدیمی
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.reset_period_hours)
        self.trade_log = deque(
            (t for t in self.trade_log if t['time'] > cutoff_time),
            maxlen=1000
        )

    def _check_recovery_progress(self) -> None:
        """بررسی پیشرفت در حالت بازیابی"""
        recent_trades = [
            t for t in self.trade_log
            if t['time'] > self.trigger_time
        ]

        if len(recent_trades) >= 3:
            recent_profits = [t['profit_r'] for t in recent_trades[-3:]]
            avg_profit = sum(recent_profits) / len(recent_profits)

            if avg_profit > 0.3:
                # بازیابی موفق
                self.recovery_mode = False
                logger.info("حالت بازیابی با موفقیت تکمیل شد")

    def check_if_active(self) -> Tuple[bool, Optional[str]]:
        """بررسی وضعیت فعال بودن قطع‌کننده"""
        if not self.enabled:
            return False, None

        if not self.triggered:
            return False, None

        current_time = datetime.now(timezone.utc)
        if self.trigger_time:
            minutes_since_trigger = (current_time - self.trigger_time).total_seconds() / 60

            if minutes_since_trigger >= self.cool_down_period_minutes:
                # پایان دوره خنک‌سازی
                self._deactivate()
                return False, None
            else:
                minutes_remaining = self.cool_down_period_minutes - minutes_since_trigger
                return True, f"در حال خنک‌سازی، {int(minutes_remaining)} دقیقه باقی‌مانده"

        return self.triggered, self.trigger_reason

    def _deactivate(self) -> None:
        """غیرفعال کردن قطع‌کننده"""
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = ""
        self.consecutive_losses = 0

        # بررسی عملکرد
        if self.daily_loss_r < self.max_daily_losses_r * 0.5:
            # توقف موثر بوده
            self.performance_metrics['prevented_losses'] += (
                    self.max_daily_losses_r - self.daily_loss_r
            )
        else:
            # احتمالاً توقف زودرس
            self.performance_metrics['false_triggers'] += 1

        logger.info("دوره خنک‌سازی قطع‌کننده تمام شد. معاملات از سر گرفته شد.")

    def is_market_volatile(self, symbols_data: Dict[str, DataFrame]) -> bool:
        """تشخیص نوسان غیرعادی بازار"""
        if not self.enabled or not symbols_data:
            return False

        try:
            volatility_scores = []
            volume_anomalies = 0
            price_gaps = 0

            for symbol, df in symbols_data.items():
                if df is None or len(df) < 30:
                    continue

                # محاسبه ATR
                atr = talib.ATR(
                    df['high'].values.astype(np.float64),
                    df['low'].values.astype(np.float64),
                    df['close'].values.astype(np.float64),
                    timeperiod=14
                )

                # بررسی افزایش ناگهانی ATR
                if len(atr) >= 20:
                    recent_atr = np.nanmean(atr[-5:])
                    past_atr = np.nanmean(atr[-25:-5])

                    if past_atr > 0:
                        volatility_change = recent_atr / past_atr
                        volatility_scores.append(volatility_change)

                # بررسی حجم غیرعادی
                if 'volume' in df.columns and len(df) >= 20:
                    recent_volume = df['volume'].iloc[-5:].mean()
                    avg_volume = df['volume'].iloc[-20:].mean()

                    if avg_volume > 0 and recent_volume > avg_volume * 3:
                        volume_anomalies += 1

                # بررسی گپ‌های قیمتی
                if len(df) >= 2:
                    last_close = df['close'].iloc[-2]
                    current_open = df['open'].iloc[-1]
                    gap_percent = abs(current_open - last_close) / last_close * 100

                    if gap_percent > 2:  # گپ بیش از 2%
                        price_gaps += 1

            # تصمیم‌گیری
            if volatility_scores:
                avg_volatility_change = np.mean(volatility_scores)
                if avg_volatility_change > 1.5:
                    return True

            # بررسی آنومالی‌ها
            total_symbols = len(symbols_data)
            if total_symbols > 0:
                anomaly_ratio = (volume_anomalies + price_gaps) / total_symbols
                if anomaly_ratio > 0.3:  # بیش از 30% سمبل‌ها غیرعادی
                    return True

            return False

        except Exception as e:
            logger.error(f"خطا در بررسی نوسان بازار: {e}")
            return False

    def get_market_anomaly_score(self, symbols_data: Dict[str, DataFrame]) -> float:
        """محاسبه امتیاز آنومالی بازار"""
        if not self.enabled or not symbols_data:
            return 0.0

        try:
            anomaly_factors = []

            for symbol, df in symbols_data.items():
                if df is None or len(df) < 50:
                    continue

                symbol_anomaly = 0.0

                # تحلیل حجم
                if 'volume' in df.columns and len(df) >= 20:
                    vol_ma = df['volume'].rolling(window=20).mean()
                    if not vol_ma.isna().all():
                        last_vol = df['volume'].iloc[-1]
                        last_vol_ma = vol_ma.iloc[-1]

                        if last_vol_ma > 0:
                            vol_ratio = last_vol / last_vol_ma
                            if vol_ratio > 3:
                                symbol_anomaly += min(1.0, (vol_ratio - 3) / 7)

                # تحلیل تغییر قیمت
                if len(df) >= 2:
                    price_changes = df['close'].pct_change().abs()
                    recent_change = price_changes.iloc[-1]
                    avg_change = price_changes.iloc[-20:].mean()

                    if avg_change > 0 and recent_change > avg_change * 3:
                        symbol_anomaly += min(1.0, (recent_change / avg_change - 3) / 5)

                # تحلیل نوسان
                highs = df['high'].values
                lows = df['low'].values

                hl_ranges = (highs - lows) / lows * 100
                recent_range = hl_ranges[-1]
                avg_range = np.mean(hl_ranges[-20:])

                if avg_range > 0 and recent_range > avg_range * 2:
                    symbol_anomaly += min(1.0, (recent_range / avg_range - 2) / 3)

                if symbol_anomaly > 0:
                    anomaly_factors.append(min(1.0, symbol_anomaly / 3))

            if anomaly_factors:
                return np.mean(anomaly_factors)

            return 0.0

        except Exception as e:
            logger.error(f"خطا در محاسبه امتیاز آنومالی بازار: {e}")
            return 0.0

    def get_status_report(self) -> Dict[str, Any]:
        """دریافت گزارش وضعیت قطع‌کننده"""
        return {
            'enabled': self.enabled,
            'triggered': self.triggered,
            'trigger_reason': self.trigger_reason,
            'consecutive_losses': self.consecutive_losses,
            'daily_loss_r': round(self.daily_loss_r, 2),
            'hourly_loss_r': round(self.hourly_loss_r, 2),
            'loss_velocity': round(self.loss_velocity, 2),
            'recovery_mode': self.recovery_mode,
            'performance_metrics': self.performance_metrics,
            'last_trigger': self.trigger_time.isoformat() if self.trigger_time else None
        }

# ===============================================
#      کلاس اصلی تولید سیگنال بهینه‌شده
# ===============================================
class SignalGenerator:
    """تولیدکننده سیگنال معاملاتی چند تایم‌فریم - نسخه بهینه و حرفه‌ای"""

    def __init__(self, config: Dict[str, Any]):
        """مقداردهی اولیه با پیکربندی"""
        self.config = config

        # بخش‌های پیکربندی
        self.signal_config = config.get('signal_generation', {})
        self.signal_processing = config.get('signal_processing', {})
        self.risk_config = config.get('risk_management', {})
        self.core_config = config.get('core', {})

        # تنظیمات اعلان
        notification = config.get('notification', {})
        self.events = notification.get('events', {})
        self.signal_generated = self.events.get('signal_generated')

        # تایم‌فریم‌ها و وزن‌ها
        self.timeframes = self.signal_config.get('timeframes', ['5m', '15m', '1h', '4h'])
        self.timeframe_weights = self.signal_config.get('timeframe_weights', {
            '5m': 0.7, '15m': 0.85, '1h': 1.0, '4h': 1.2
        })

        # آستانه‌ها و پارامترهای پایه
        self.base_minimum_signal_score = self.signal_config.get('minimum_signal_score', 180.0)
        self.base_min_risk_reward_ratio = self.risk_config.get('min_risk_reward_ratio', 1.8)
        self.base_preferred_risk_reward_ratio = self.risk_config.get('preferred_risk_reward_ratio', 2.5)
        self.base_default_sl_percent = self.risk_config.get('default_stop_loss_percent', 1.5)
        self.notification_config = self.signal_processing.get('notification', {})
        self.min_score_to_notify = self.notification_config.get('min_score_to_notify')

        # پارامترهای تحلیل
        self.pattern_scores = self.signal_config.get('pattern_scores', {})
        self.volume_multiplier_threshold = self.signal_config.get('volume_multiplier_threshold', 1.3)
        self.divergence_sensitivity = self.signal_config.get('divergence_sensitivity', 0.75)

        # تنظیمات تشخیص قله/دره
        self.peak_detection_settings = {
            'order': self.signal_config.get('peak_detection_order', 3),
            'distance': self.signal_config.get('peak_detection_distance', 5),
            'prominence_factor': self.signal_config.get('peak_detection_prominence_factor', 0.1)
        }

        # تنظیمات MACD
        self.macd_peak_detection_settings = {
            'smooth_kernel': 5,
            'distance': 5,
            'prominence_factor': 0.1
        }
        self.macd_trendline_period = 80
        self.macd_cross_period = 20
        self.macd_hist_period = 60

        # تنظیمات ساختار تایم‌فریم بالاتر
        self._init_htf_settings()

        # تنظیمات فیلتر نوسان
        self._init_volatility_settings()

        # تنظیمات الگوهای پیشرفته
        self._init_advanced_patterns_settings()

        # مقداردهی سیستم‌های پیشرفته
        self.regime_detector = MarketRegimeDetector(config)
        self.adaptive_learning = AdaptiveLearningSystem(config)
        self.correlation_manager = CorrelationManager(config)
        self.circuit_breaker = EmergencyCircuitBreaker(config)

        # مدیریت اجرای موازی
        self._init_parallel_execution()

        # کش برای محاسبات تکراری
        self._init_caching()

        # به‌روزرسانی امتیازات الگو از سیستم یادگیری
        if self.adaptive_learning.enabled:
            self.pattern_scores = self.adaptive_learning.get_adaptive_pattern_scores(self.pattern_scores)

        logger.info(
            f"تولیدکننده سیگنال مقداردهی شد. "
            f"حداقل امتیاز پایه: {self.base_minimum_signal_score}, "
            f"حداقل RR پایه: {self.base_min_risk_reward_ratio}"
        )

    def _init_htf_settings(self) -> None:
        """مقداردهی تنظیمات ساختار تایم‌فریم بالاتر"""
        self.htf_config = self.signal_config.get('structure_confirmation', {})
        self.htf_enabled = self.htf_config.get('enabled', True)
        self.htf_timeframe_method = self.htf_config.get('timeframe_method', 'next_higher')
        self.htf_fixed_tf1 = self.htf_config.get('fixed_first_timeframe', '1h')
        self.htf_fixed_tf2 = self.htf_config.get('fixed_second_timeframe', '4h')
        self.htf_sr_lookback = self.htf_config.get('support_resistance_lookback', 50)
        self.htf_sr_atr_multiplier = self.htf_config.get('support_resistance_atr_multiplier', 1.0)
        self.htf_trend_indicator = self.htf_config.get('trend_indicator', 'ema')
        self.htf_score_config = {
            'base': self.htf_config.get('base_score', 1.0),
            'confirm_bonus': self.htf_config.get('confirmation_bonus', 0.2),
            'trend_bonus_mult': self.htf_config.get('trend_bonus_multiplier', 1.5),
            'contradict_penalty': self.htf_config.get('contradiction_penalty', 0.3),
            'trend_penalty_mult': self.htf_config.get('trend_penalty_multiplier', 1.5),
            'min_score': self.htf_config.get('min_score', 0.5),
            'max_score': self.htf_config.get('max_score', 1.5),
        }

    def _init_volatility_settings(self) -> None:
        """مقداردهی تنظیمات فیلتر نوسان"""
        self.vol_config = self.signal_config.get('volatility_filter', {})
        self.vol_enabled = self.vol_config.get('enabled', True)
        self.vol_atr_period = self.vol_config.get('atr_period', 14)
        self.vol_atr_ma_period = self.vol_config.get('atr_ma_period', 30)
        self.vol_high_thresh = self.vol_config.get('high_volatility_threshold', 1.3)
        self.vol_low_thresh = self.vol_config.get('low_volatility_threshold', 0.7)
        self.vol_extreme_thresh = self.vol_config.get('extreme_volatility_threshold', 1.8)
        self.vol_scores = self.vol_config.get('scores', {
            'normal': 1.0,
            'low': 0.9,
            'high': 0.8,
            'extreme': 0.5
        })
        self.vol_reject_extreme = self.vol_config.get('reject_on_extreme_volatility', True)

    def _init_advanced_patterns_settings(self) -> None:
        """مقداردهی تنظیمات الگوهای پیشرفته"""
        # الگوهای هارمونیک
        self.harmonic_config = self.signal_config.get('harmonic_patterns', {})
        self.harmonic_enabled = self.harmonic_config.get('enabled', True)
        self.harmonic_lookback = self.harmonic_config.get('lookback', 100)
        self.harmonic_tolerance = self.harmonic_config.get('tolerance', 0.03)
        self.harmonic_min_quality = self.harmonic_config.get('min_quality', 0.7)

        # کانال‌های قیمت
        self.channel_config = self.signal_config.get('price_channels', {})
        self.channel_enabled = self.channel_config.get('enabled', True)
        self.channel_lookback = self.channel_config.get('lookback', 100)
        self.channel_min_touches = self.channel_config.get('min_touches', 3)
        self.channel_quality_threshold = self.channel_config.get('quality_threshold', 0.7)

        # الگوهای چرخه‌ای
        self.cycle_config = self.signal_config.get('cyclical_patterns', {})
        self.cycle_enabled = self.cycle_config.get('enabled', True)
        self.cycle_lookback = self.cycle_config.get('lookback', 200)
        self.cycle_min_cycles = self.cycle_config.get('min_cycles', 2)
        self.cycle_fourier_periods = self.cycle_config.get('fourier_periods', [5, 10, 20, 40, 60])

    def _init_parallel_execution(self) -> None:
        """مقداردهی اجرای موازی"""
        max_workers = self.core_config.get('max_workers', multiprocessing.cpu_count())
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers if max_workers > 0 else None,
            thread_name_prefix="SignalGen"
        )

        # برای محاسبات سنگین
        self.process_executor = ProcessPoolExecutor(
            max_workers=max(1, max_workers // 2)
        )

        logger.info(f"اجرای موازی با {max_workers} thread فعال شد")

    def _init_caching(self) -> None:
        """مقداردهی سیستم کش"""
        self._analysis_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._indicator_cache: Dict[str, np.ndarray] = {}
        self._pattern_cache: Dict[str, List[Dict[str, Any]]] = {}

        self._cache_ttl_seconds = 60  # 1 دقیقه
        self._cache_size_limit = 1000
        self._last_cache_cleanup = time.time()

    def shutdown(self) -> None:
        """خاتمه عملیات و ذخیره داده‌ها"""
        logger.info("در حال خاتمه تولیدکننده سیگنال...")

        try:
            # ذخیره داده‌های یادگیری
            if hasattr(self, 'adaptive_learning'):
                self.adaptive_learning.save_data(force=True)

            # ذخیره داده‌های همبستگی
            if hasattr(self, 'correlation_manager'):
                self.correlation_manager.save_data()

            # خاتمه executor ها
            self.executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)

        except Exception as e:
            logger.error(f"خطا در خاتمه: {e}")

        logger.info("تولیدکننده سیگنال خاتمه یافت.")

    # --- توابع تحلیل بهینه‌شده ---

    def _cache_key(self, symbol: str, timeframe: str, indicator: str, params: tuple) -> str:
        """تولید کلید کش برای اندیکاتورها"""
        params_str = '_'.join(str(p) for p in params)
        return f"{symbol}_{timeframe}_{indicator}_{params_str}"

    def _get_cached_indicator(self, key: str) -> Optional[np.ndarray]:
        """دریافت اندیکاتور از کش"""
        self._cleanup_cache_if_needed()
        return self._indicator_cache.get(key)

    def _cache_indicator(self, key: str, data: np.ndarray) -> None:
        """کش کردن اندیکاتور"""
        if len(self._indicator_cache) > self._cache_size_limit:
            # حذف قدیمی‌ترین
            oldest = min(self._indicator_cache.keys(),
                         key=lambda k: id(self._indicator_cache[k]))
            del self._indicator_cache[oldest]

        self._indicator_cache[key] = data

    def _cleanup_cache_if_needed(self) -> None:
        """پاکسازی کش در صورت نیاز"""
        current_time = time.time()
        if current_time - self._last_cache_cleanup > 300:  # هر 5 دقیقه
            # حذف ورودی‌های قدیمی از analysis_cache
            expired_keys = []
            for key, (data, timestamp) in self._analysis_cache.items():
                if current_time - timestamp > self._cache_ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._analysis_cache[key]

            self._last_cache_cleanup = current_time

    @lru_cache(maxsize=1024)
    def find_peaks_and_valleys(self, data_tuple: tuple, order: int = 3,
                               distance: int = 5, prominence_factor: float = 0.1,
                               window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """یافتن قله‌ها و دره‌ها با کش"""
        # تبدیل tuple به array (برای کش)
        data = np.array(data_tuple)

        if data is None or len(data) < (max(order, distance) * 2 + 1):
            return np.array([], dtype=int), np.array([], dtype=int)

        try:
            if window_size and window_size < len(data):
                data = data[-window_size:]

            # فیلتر داده‌های معتبر
            valid_mask = np.isfinite(data)
            if not np.any(valid_mask):
                return np.array([], dtype=int), np.array([], dtype=int)

            indices = np.arange(len(data))
            valid_indices = indices[valid_mask]
            valid_data = data[valid_mask]

            if len(valid_data) < (max(order, distance) * 2 + 1):
                return np.array([], dtype=int), np.array([], dtype=int)

            # محاسبه prominence
            prominence = np.std(valid_data) * prominence_factor if np.std(valid_data) > 1e-6 else None

            # یافتن قله‌ها
            peaks_rel, peaks_props = sig_processing.find_peaks(
                valid_data, distance=distance, prominence=prominence,
                width=order, rel_height=0.5
            )

            # یافتن دره‌ها
            valleys_rel, valleys_props = sig_processing.find_peaks(
                -valid_data, distance=distance, prominence=prominence,
                width=order, rel_height=0.5
            )

            # فیلتر کیفیت
            if len(peaks_rel) > 0 and 'prominences' in peaks_props:
                quality_threshold = np.median(peaks_props['prominences']) * 0.5
                if quality_threshold > 0:
                    quality_peaks = peaks_rel[peaks_props['prominences'] >= quality_threshold]
                    peaks_rel = quality_peaks

            if len(valleys_rel) > 0 and 'prominences' in valleys_props:
                quality_threshold = np.median(valleys_props['prominences']) * 0.5
                if quality_threshold > 0:
                    quality_valleys = valleys_rel[valleys_props['prominences'] >= quality_threshold]
                    valleys_rel = quality_valleys

            peaks = valid_indices[peaks_rel]
            valleys = valid_indices[valleys_rel]

            return peaks, valleys

        except Exception as e:
            logger.error(f"خطا در یافتن قله/دره: {e}")
            return np.array([], dtype=int), np.array([], dtype=int)

    def analyze_volume_trend(self, df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """تحلیل روند حجم - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'current_ratio': 1.0,
            'trend': 'neutral',
            'pattern': 'normal',
            'is_confirmed_by_volume': False,
            'volume_score': 1.0
        }

        if 'volume' not in df.columns or len(df) < window + 1:
            results['status'] = 'insufficient_data'
            return results

        try:
            # استفاده از numpy برای سرعت بیشتر
            volumes = df['volume'].values

            # محاسبه میانگین متحرک
            if use_bottleneck:
                vol_sma = bn.move_mean(volumes, window=window, min_count=window)
            else:
                vol_sma = np.convolve(volumes, np.ones(window) / window, mode='valid')
                vol_sma = np.pad(vol_sma, (window - 1, 0), mode='constant', constant_values=np.nan)

            # محاسبه نسبت حجم
            valid_indices = ~np.isnan(vol_sma) & (vol_sma > 1e-9)
            vol_ratio = np.full_like(volumes, np.nan, dtype=float)
            vol_ratio[valid_indices] = volumes[valid_indices] / vol_sma[valid_indices]

            # مقدار فعلی
            current_ratio = vol_ratio[-1] if not np.isnan(vol_ratio[-1]) else 1.0
            results['current_ratio'] = round(current_ratio, 3)
            results['is_confirmed_by_volume'] = current_ratio > self.volume_multiplier_threshold

            # تعیین روند و الگو
            if current_ratio > self.volume_multiplier_threshold * 2.0:
                results['trend'] = 'strongly_increasing'
                results['pattern'] = 'climax_volume'
                results['volume_score'] = 1.5
            elif current_ratio > self.volume_multiplier_threshold * 1.5:
                results['trend'] = 'increasing'
                results['pattern'] = 'spike'
                results['volume_score'] = 1.3
            elif current_ratio > self.volume_multiplier_threshold:
                results['trend'] = 'increasing'
                results['pattern'] = 'above_average'
                results['volume_score'] = 1.1
            elif current_ratio < 1.0 / (self.volume_multiplier_threshold * 1.5):
                results['trend'] = 'strongly_decreasing'
                results['pattern'] = 'dry_up'
                results['volume_score'] = 0.7
            elif current_ratio < 1.0 / self.volume_multiplier_threshold:
                results['trend'] = 'decreasing'
                results['pattern'] = 'below_average'
                results['volume_score'] = 0.8
            else:
                results['trend'] = 'neutral'
                results['pattern'] = 'normal'
                results['volume_score'] = 1.0

            # تحلیل روند میانگین متحرک حجم
            if len(vol_sma) >= 10:
                vol_sma_valid = vol_sma[~np.isnan(vol_sma)]
                if len(vol_sma_valid) >= 10:
                    vol_sma_slope = (vol_sma_valid[-1] - vol_sma_valid[-10]) / vol_sma_valid[-10]
                    results['volume_ma_trend'] = (
                        'increasing' if vol_sma_slope > 0.05 else
                        'decreasing' if vol_sma_slope < -0.05 else 'flat'
                    )
                    results['volume_ma_slope'] = round(vol_sma_slope, 3)

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل روند حجم: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def detect_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تشخیص روند با میانگین‌های متحرک - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'trend': 'neutral',
            'strength': 0,
            'method': 'ema',
            'phase': 'undefined',
            'details': {}
        }

        required_len = 100 + 5
        if df is None or len(df) < required_len:
            results['status'] = 'insufficient_data'
            return results

        try:
            # دریافت از کش یا محاسبه
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')
            close_prices = df['close'].values.astype(np.float64)

            # EMAs
            ema_periods = [20, 50, 100]
            emas = {}

            for period in ema_periods:
                ema_key = self._cache_key(symbol, timeframe, 'EMA', (period,))
                ema = self._get_cached_indicator(ema_key)

                if ema is None:
                    ema = talib.EMA(close_prices, timeperiod=period)
                    self._cache_indicator(ema_key, ema)

                emas[period] = ema

            # یافتن آخرین مقادیر معتبر
            last_valid_idx = -1
            while last_valid_idx >= -len(df) and any(
                    np.isnan(emas[p][last_valid_idx]) for p in ema_periods
            ):
                last_valid_idx -= 1

            if abs(last_valid_idx) > len(df):
                results['status'] = 'calculation_error'
                return results

            # مقادیر فعلی
            current_close = close_prices[last_valid_idx]
            current_emas = {p: emas[p][last_valid_idx] for p in ema_periods}

            # محاسبه شیب
            lookback = 5
            slopes = {}
            for period in ema_periods[:2]:  # فقط 20 و 50
                if last_valid_idx >= lookback:
                    slopes[period] = emas[period][last_valid_idx] - emas[period][last_valid_idx - lookback]
                else:
                    slopes[period] = 0

            # تعیین آرایش EMA
            ema_arrangement = self._determine_ema_arrangement(current_emas)

            # تشخیص روند و قدرت
            trend_info = self._analyze_trend_pattern(
                current_close, current_emas, slopes, ema_arrangement
            )

            results.update(trend_info)
            results['details'] = {
                'close': round(current_close, 5),
                **{f'ema{p}': round(current_emas[p], 5) for p in ema_periods},
                'ema20_slope': round(slopes.get(20, 0), 5),
                'ema50_slope': round(slopes.get(50, 0), 5),
                'ema_arrangement': ema_arrangement
            }

            return results

        except Exception as e:
            logger.error(f"خطا در تشخیص روند: {e}")
            results['status'] = 'error'
            return results

    def _determine_ema_arrangement(self, emas: Dict[int, float]) -> str:
        """تعیین آرایش EMAs"""
        if emas[20] > emas[50] > emas[100]:
            return 'bullish_aligned'
        elif emas[20] < emas[50] < emas[100]:
            return 'bearish_aligned'
        elif emas[20] > emas[50] and emas[50] < emas[100]:
            return 'potential_bullish_reversal'
        elif emas[20] < emas[50] and emas[50] > emas[100]:
            return 'potential_bearish_reversal'
        else:
            return 'mixed'

    def _analyze_trend_pattern(self, close: float, emas: Dict[int, float],
                               slopes: Dict[int, float], arrangement: str) -> Dict[str, Any]:
        """تحلیل الگوی روند"""
        trend = 'neutral'
        strength = 0
        phase = 'undefined'

        # روند صعودی
        if close > emas[20] > emas[50] > emas[100] and slopes[20] > 0 and slopes[50] > 0:
            trend = 'bullish'
            strength = 3
            phase = 'mature' if arrangement == 'bullish_aligned' else 'developing'
        elif close > emas[20] > emas[50] and slopes[20] > 0:
            trend = 'bullish'
            strength = 2
            phase = 'developing'
        elif close > emas[20] and slopes[20] > 0:
            trend = 'bullish'
            strength = 1
            phase = 'early'

        # روند نزولی
        elif close < emas[20] < emas[50] < emas[100] and slopes[20] < 0 and slopes[50] < 0:
            trend = 'bearish'
            strength = -3
            phase = 'mature' if arrangement == 'bearish_aligned' else 'developing'
        elif close < emas[20] < emas[50] and slopes[20] < 0:
            trend = 'bearish'
            strength = -2
            phase = 'developing'
        elif close < emas[20] and slopes[20] < 0:
            trend = 'bearish'
            strength = -1
            phase = 'early'

        # پولبک‌ها
        elif close < emas[50] and emas[20] > emas[50] and slopes[50] > 0:
            trend = 'bullish_pullback'
            strength = 1
            phase = 'pullback'
        elif close > emas[50] and emas[20] < emas[50] and slopes[50] < 0:
            trend = 'bearish_pullback'
            strength = -1
            phase = 'pullback'

        return {
            'trend': trend,
            'strength': strength,
            'phase': phase
        }

    async def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """تشخیص الگوهای کندل استیک - نسخه بهینه با اجرای موازی"""
        patterns_found = []

        if df is None or len(df) < 10:
            return patterns_found

        try:
            # بررسی کش
            cache_key = f"candle_patterns_{id(df)}_{len(df)}"
            if cache_key in self._pattern_cache:
                return self._pattern_cache[cache_key]

            # آماده‌سازی داده‌ها
            ohlc_data = (
                df['open'].values.astype(np.float64),
                df['high'].values.astype(np.float64),
                df['low'].values.astype(np.float64),
                df['close'].values.astype(np.float64)
            )

            # الگوهای تالیب
            talib_patterns = [
                (talib.CDLHAMMER, 'hammer', 'bullish', 1.8),
                (talib.CDLINVERTEDHAMMER, 'inverted_hammer', 'bullish', 1.8),
                (talib.CDLENGULFING, 'engulfing', 'neutral', 2.2),
                (talib.CDLMORNINGSTAR, 'morning_star', 'bullish', 3.0),
                (talib.CDLEVENINGSTAR, 'evening_star', 'bearish', 3.0),
                (talib.CDLHARAMI, 'harami', 'neutral', 1.5),
                (talib.CDLDOJI, 'doji', 'neutral', 1.2),
                (talib.CDLSHOOTINGSTAR, 'shooting_star', 'bearish', 2.0),
                (talib.CDLMARUBOZU, 'marubozu', 'neutral', 1.5),
                (talib.CDLHANGINGMAN, 'hanging_man', 'bearish', 2.0),
                (talib.CDLDRAGONFLYDOJI, 'dragonfly_doji', 'bullish', 2.5),
                (talib.CDLGRAVESTONEDOJI, 'gravestone_doji', 'bearish', 2.5),
                (talib.CDLPIERCING, 'piercing', 'bullish', 2.5),
                (talib.CDLDARKCLOUDCOVER, 'dark_cloud_cover', 'bearish', 2.5),
                (talib.CDLTHREEWHITESOLDIERS, 'three_white_soldiers', 'bullish', 3.5),
                (talib.CDLTHREEBLACKCROWS, 'three_black_crows', 'bearish', 3.5),
            ]

            # اجرای موازی
            loop = asyncio.get_running_loop()
            futures = []

            for func, name, direction, base_score in talib_patterns:
                future = loop.run_in_executor(
                    self.executor, func, *ohlc_data
                )
                futures.append((future, name, direction, base_score))

            # جمع‌آوری نتایج
            results = await asyncio.gather(*(f[0] for f in futures), return_exceptions=True)

            last_idx = len(df) - 1

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"خطا در پردازش الگوی {futures[i][1]}: {result}")
                    continue

                if len(result) == len(df):
                    future_obj, pattern_name, default_direction, base_score = futures[i]
                    pattern_value = result[last_idx]

                    if pattern_value != 0:
                        # تعیین جهت
                        pattern_direction = default_direction
                        if pattern_value > 0 and default_direction == 'neutral':
                            pattern_direction = 'bullish'
                        elif pattern_value < 0 and default_direction == 'neutral':
                            pattern_direction = 'bearish'

                        # محاسبه قدرت و امتیاز
                        pattern_strength = min(1.0, abs(pattern_value) / 100)
                        if pattern_strength < 0.1:
                            pattern_strength = 0.7

                        # امتیاز از پیکربندی یا مقدار پیش‌فرض
                        config_score = self.pattern_scores.get(pattern_name, base_score)
                        pattern_score = config_score * pattern_strength

                        patterns_found.append({
                            'type': pattern_name,
                            'direction': pattern_direction,
                            'index': last_idx,
                            'score': pattern_score,
                            'strength': pattern_strength,
                            'value': int(pattern_value)
                        })

            # تشخیص الگوهای چند کندلی
            await self._detect_multi_candle_patterns(df, patterns_found)

            # کش نتیجه
            self._pattern_cache[cache_key] = patterns_found

            return patterns_found

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای کندل استیک: {e}", exc_info=True)
            return []

    async def _detect_multi_candle_patterns(self, df: pd.DataFrame,
                                            patterns: List[Dict[str, Any]]) -> None:
        """تشخیص الگوهای چند کندلی پیشرفته"""
        if df is None or len(df) < 30:
            return

        try:
            # اجرای موازی تشخیص الگوها
            tasks = [
                self._detect_head_and_shoulders(df),
                self._detect_triangle_patterns(df),
                self._detect_flag_patterns(df),
                self._detect_double_patterns(df),
                self._detect_wedge_patterns(df)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    patterns.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"خطا در تشخیص الگوی چند کندلی: {result}")

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای چند کندلی: {e}")

    async def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """تشخیص الگوی سر و شانه"""
        patterns = []

        if df is None or len(df) < 30:
            return patterns

        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values

            # تبدیل به tuple برای کش
            highs_tuple = tuple(highs)
            lows_tuple = tuple(lows)

            peaks, _ = self.find_peaks_and_valleys(
                highs_tuple, distance=5, prominence_factor=0.05
            )
            _, valleys = self.find_peaks_and_valleys(
                lows_tuple, distance=5, prominence_factor=0.05
            )

            # الگوی سر و شانه معمولی
            if len(peaks) >= 3:
                for i in range(len(peaks) - 2):
                    left_shoulder_idx = peaks[i]
                    head_idx = peaks[i + 1]
                    right_shoulder_idx = peaks[i + 2]

                    pattern = self._validate_head_shoulders_pattern(
                        highs, lows, closes, valleys,
                        left_shoulder_idx, head_idx, right_shoulder_idx,
                        is_inverse=False
                    )

                    if pattern:
                        patterns.append(pattern)

            # الگوی سر و شانه معکوس
            if len(valleys) >= 3:
                for i in range(len(valleys) - 2):
                    left_shoulder_idx = valleys[i]
                    head_idx = valleys[i + 1]
                    right_shoulder_idx = valleys[i + 2]

                    pattern = self._validate_head_shoulders_pattern(
                        highs, lows, closes, peaks,
                        left_shoulder_idx, head_idx, right_shoulder_idx,
                        is_inverse=True
                    )

                    if pattern:
                        patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوی سر و شانه: {e}")
            return []

    def _validate_head_shoulders_pattern(self, highs: np.ndarray, lows: np.ndarray,
                                         closes: np.ndarray, opposite_points: np.ndarray,
                                         left_idx: int, head_idx: int, right_idx: int,
                                         is_inverse: bool) -> Optional[Dict[str, Any]]:
        """اعتبارسنجی الگوی سر و شانه"""
        try:
            if is_inverse:
                # سر و شانه معکوس
                left_price = lows[left_idx]
                head_price = lows[head_idx]
                right_price = lows[right_idx]

                # سر باید پایین‌ترین نقطه باشد
                if not (head_price < left_price and head_price < right_price):
                    return None

                pattern_type = 'inverse_head_and_shoulders'
                direction = 'bullish'
            else:
                # سر و شانه معمولی
                left_price = highs[left_idx]
                head_price = highs[head_idx]
                right_price = highs[right_idx]

                # سر باید بالاترین نقطه باشد
                if not (head_price > left_price and head_price > right_price):
                    return None

                pattern_type = 'head_and_shoulders'
                direction = 'bearish'

            # بررسی تقارن شانه‌ها
            shoulder_diff_percent = abs(right_price - left_price) / left_price
            if shoulder_diff_percent > 0.1:  # بیش از 10% اختلاف
                return None

            # بررسی تقارن زمانی
            left_time_gap = head_idx - left_idx
            right_time_gap = right_idx - head_idx
            time_gap_ratio = min(left_time_gap, right_time_gap) / max(left_time_gap, right_time_gap)

            if time_gap_ratio < 0.6:  # عدم تقارن زمانی
                return None

            # یافتن خط گردن
            neckline_points = self._find_neckline_points(
                opposite_points, left_idx, head_idx, right_idx
            )

            if not neckline_points:
                return None

            neckline_price = np.mean([
                (lows if not is_inverse else highs)[p] for p in neckline_points
            ])

            # محاسبه هدف قیمتی
            pattern_height = abs(head_price - neckline_price)

            if is_inverse:
                price_target = neckline_price + pattern_height
            else:
                price_target = neckline_price - pattern_height

            # بررسی شکست خط گردن
            last_price = closes[-1]
            last_idx = len(closes) - 1

            if is_inverse:
                breakout_confirmed = last_price > neckline_price and last_idx > right_idx
            else:
                breakout_confirmed = last_price < neckline_price and last_idx > right_idx

            # محاسبه کیفیت الگو
            pattern_quality = (
                    (1.0 - shoulder_diff_percent) * 0.4 +
                    time_gap_ratio * 0.3 +
                    (0.3 if breakout_confirmed else 0)
            )

            return {
                'type': pattern_type,
                'direction': direction,
                'index': right_idx,
                'breakout_confirmed': breakout_confirmed,
                'neckline_price': float(neckline_price),
                'price_target': float(price_target),
                'pattern_quality': round(pattern_quality, 2),
                'score': self.pattern_scores.get(pattern_type, 4.0) * pattern_quality,
                'points': {
                    'left_shoulder': int(left_idx),
                    'head': int(head_idx),
                    'right_shoulder': int(right_idx)
                }
            }

        except Exception as e:
            logger.error(f"خطا در اعتبارسنجی الگوی سر و شانه: {e}")
            return None

    def _find_neckline_points(self, points: np.ndarray, left: int,
                              head: int, right: int) -> List[int]:
        """یافتن نقاط خط گردن"""
        neckline_points = []

        # نقاط بین شانه چپ و سر
        between_left_head = [p for p in points if left < p < head]
        if between_left_head:
            neckline_points.append(between_left_head[-1])

        # نقاط بین سر و شانه راست
        between_head_right = [p for p in points if head < p < right]
        if between_head_right:
            neckline_points.append(between_head_right[0])

        return neckline_points

    async def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """تشخیص الگوهای مثلث"""
        patterns = []

        if df is None or len(df) < 30:
            return patterns

        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values

            # یافتن قله‌ها و دره‌ها
            peaks, valleys = self.find_peaks_and_valleys(
                tuple(closes),
                distance=self.peak_detection_settings['distance'],
                prominence_factor=self.peak_detection_settings['prominence_factor']
            )

            if len(peaks) < 2 or len(valleys) < 2:
                return patterns

            # انتخاب آخرین نقاط
            last_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            last_valleys = valleys[-3:] if len(valleys) >= 3 else valleys

            if len(last_peaks) < 2 or len(last_valleys) < 2:
                return patterns

            # محاسبه خطوط روند
            peak_x = last_peaks
            peak_y = highs[last_peaks]
            valley_x = last_valleys
            valley_y = lows[last_valleys]

            # رگرسیون خطی
            upper_slope, upper_intercept = np.polyfit(peak_x, peak_y, 1)
            lower_slope, lower_intercept = np.polyfit(valley_x, valley_y, 1)

            # تشخیص نوع مثلث
            triangle_type = self._determine_triangle_type(
                upper_slope, lower_slope
            )

            if not triangle_type:
                return patterns

            # محاسبه نقطه همگرایی
            convergence_info = self._calculate_convergence(
                upper_slope, upper_intercept,
                lower_slope, lower_intercept,
                len(df)
            )

            if not convergence_info['is_valid']:
                return patterns

            # محاسبه جزئیات الگو
            pattern = self._create_triangle_pattern(
                triangle_type, convergence_info,
                last_peaks, last_valleys,
                highs, lows, closes,
                upper_slope, upper_intercept,
                lower_slope, lower_intercept
            )

            if pattern:
                patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای مثلث: {e}")
            return []

    def _determine_triangle_type(self, upper_slope: float,
                                 lower_slope: float) -> Optional[str]:
        """تعیین نوع مثلث"""
        slope_threshold = 0.001

        if abs(upper_slope) < slope_threshold and lower_slope > slope_threshold:
            return 'ascending_triangle'
        elif upper_slope < -slope_threshold and abs(lower_slope) < slope_threshold:
            return 'descending_triangle'
        elif upper_slope < -slope_threshold and lower_slope > slope_threshold:
            return 'symmetric_triangle'

        return None

    def _calculate_convergence(self, upper_slope: float, upper_intercept: float,
                               lower_slope: float, lower_intercept: float,
                               data_length: int) -> Dict[str, Any]:
        """محاسبه نقطه همگرایی خطوط"""
        if abs(upper_slope - lower_slope) < 1e-6:
            return {'is_valid': False}

        convergence_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
        convergence_y = upper_slope * convergence_x + upper_intercept

        # بررسی اعتبار نقطه همگرایی
        last_idx = data_length - 1
        is_valid = last_idx < convergence_x < last_idx + 50

        return {
            'is_valid': is_valid,
            'x': convergence_x,
            'y': convergence_y
        }

    def _create_triangle_pattern(self, triangle_type: str, convergence_info: Dict,
                                 peaks: np.ndarray, valleys: np.ndarray,
                                 highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                 upper_slope: float, upper_intercept: float,
                                 lower_slope: float, lower_intercept: float) -> Optional[Dict[str, Any]]:
        """ایجاد اطلاعات الگوی مثلث"""
        try:
            last_idx = len(closes) - 1

            # محاسبه عرض فعلی مثلث
            current_upper = upper_slope * last_idx + upper_intercept
            current_lower = lower_slope * last_idx + lower_intercept
            pattern_width = current_upper - current_lower

            if pattern_width <= 0:
                return None

            # محاسبه کیفیت الگو
            total_touches = len(peaks) + len(valleys)
            pattern_quality = min(1.0, total_touches / 6)

            # موقعیت قیمت در الگو
            last_close = closes[-1]
            position_in_pattern = (last_close - current_lower) / pattern_width

            # تعیین جهت
            if triangle_type == 'ascending_triangle':
                direction = 'bullish'
            elif triangle_type == 'descending_triangle':
                direction = 'bearish'
            else:  # symmetric
                direction = 'bullish' if position_in_pattern > 0.5 else 'bearish'

            # محاسبه هدف قیمتی
            pattern_height = max(highs[peaks]) - min(lows[valleys])
            price_target = (
                last_close + pattern_height if direction == 'bullish'
                else last_close - pattern_height
            )

            return {
                'type': triangle_type,
                'direction': direction,
                'index': last_idx,
                'score': self.pattern_scores.get(triangle_type, 3.5) * pattern_quality,
                'pattern_quality': round(pattern_quality, 2),
                'convergence_point': {
                    'x': int(convergence_info['x']),
                    'y': float(convergence_info['y'])
                },
                'current_width': float(pattern_width),
                'position_in_pattern': float(position_in_pattern),
                'price_target': float(price_target),
                'upper_line': {
                    'slope': float(upper_slope),
                    'intercept': float(upper_intercept)
                },
                'lower_line': {
                    'slope': float(lower_slope),
                    'intercept': float(lower_intercept)
                }
            }

        except Exception as e:
            logger.error(f"خطا در ایجاد الگوی مثلث: {e}")
            return None

    async def _detect_flag_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """تشخیص الگوهای پرچم"""
        patterns = []

        if df is None or len(df) < 30:
            return patterns

        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values if 'volume' in df.columns else None

            # پارامترها
            pole_window = 5
            flag_window = 15
            min_pole_move = 0.03  # 3%

            # بررسی برای تشکیل پرچم
            for i in range(len(closes) - pole_window - flag_window):
                # تشخیص میله (pole)
                pole_start = i
                pole_end = i + pole_window

                pole_move = (closes[pole_end] - closes[pole_start]) / closes[pole_start]

                # بررسی حرکت قوی
                is_bullish_pole = pole_move > min_pole_move
                is_bearish_pole = pole_move < -min_pole_move

                if not (is_bullish_pole or is_bearish_pole):
                    continue

                # بررسی حجم در میله
                strong_volume = False
                if volumes is not None:
                    avg_volume = np.mean(volumes[max(0, pole_start - 5):pole_start])
                    pole_volume = np.mean(volumes[pole_start:pole_end])
                    strong_volume = pole_volume > avg_volume * 1.5

                # تحلیل پرچم
                flag_start = pole_end
                flag_end = min(flag_start + flag_window, len(closes) - 1)

                if flag_end - flag_start < 5:
                    continue

                # محاسبه خطوط کانال پرچم
                flag_highs = highs[flag_start:flag_end + 1]
                flag_lows = lows[flag_start:flag_end + 1]

                x_indices = np.arange(len(flag_highs))

                try:
                    upper_slope, upper_intercept = np.polyfit(x_indices, flag_highs, 1)
                    lower_slope, lower_intercept = np.polyfit(x_indices, flag_lows, 1)
                except:
                    continue

                # بررسی موازی بودن خطوط
                slopes_difference = abs(upper_slope - lower_slope)
                are_lines_parallel = slopes_difference < 0.0005

                # اعتبارسنجی جهت پرچم
                is_valid_flag = False
                pattern_type = None
                direction = None

                if is_bullish_pole:
                    # پرچم صعودی - باید نزولی یا افقی باشد
                    if (upper_slope < 0 and lower_slope < 0) or are_lines_parallel:
                        is_valid_flag = True
                        pattern_type = 'bull_flag'
                        direction = 'bullish'
                elif is_bearish_pole:
                    # پرچم نزولی - باید صعودی یا افقی باشد
                    if (upper_slope > 0 and lower_slope > 0) or are_lines_parallel:
                        is_valid_flag = True
                        pattern_type = 'bear_flag'
                        direction = 'bearish'

                if is_valid_flag and pattern_type:
                    # محاسبه کیفیت الگو
                    flag_quality = (
                            (1.0 if strong_volume else 0.7) *
                            (1.0 - (slopes_difference / 0.001))
                    )

                    # هدف قیمتی
                    pole_height = abs(closes[pole_end] - closes[pole_start])
                    price_target = (
                        closes[flag_end] + pole_height if direction == 'bullish'
                        else closes[flag_end] - pole_height
                    )

                    pattern = {
                        'type': pattern_type,
                        'direction': direction,
                        'index': flag_end,
                        'score': self.pattern_scores.get(pattern_type, 3.0) * flag_quality,
                        'pattern_quality': round(flag_quality, 2),
                        'price_target': float(price_target),
                        'pole_start': pole_start,
                        'pole_end': pole_end,
                        'flag_start': flag_start,
                        'flag_end': flag_end,
                        'pole_height': float(pole_height),
                        'slope_difference': float(slopes_difference),
                        'strong_volume': strong_volume
                    }

                    patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای پرچم: {e}")
            return []

    async def _detect_double_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """تشخیص الگوهای دوقلو (Double Top/Bottom)"""
        patterns = []

        if df is None or len(df) < 40:
            return patterns

        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values

            # یافتن قله‌ها و دره‌ها
            peaks, valleys = self.find_peaks_and_valleys(
                tuple(closes), distance=10, prominence_factor=0.08
            )

            # Double Top
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    first_peak_idx = peaks[i]
                    second_peak_idx = peaks[i + 1]

                    first_peak_price = highs[first_peak_idx]
                    second_peak_price = highs[second_peak_idx]
                        # بررسی تشابه قله‌ها (حداکثر 3% اختلاف)
                    peak_diff_percent = abs(second_peak_price - first_peak_price) / first_peak_price
                    if peak_diff_percent > 0.03:
                        continue

                    # یافتن دره بین دو قله
                    valley_between = [
                        v for v in valleys
                        if first_peak_idx < v < second_peak_idx
                    ]

                    if not valley_between:
                        continue

                    neckline_idx = valley_between[0]
                    neckline_price = lows[neckline_idx]

                    # محاسبه کیفیت الگو
                    pattern_height = first_peak_price - neckline_price
                    time_symmetry = abs((second_peak_idx - neckline_idx) -
                                      (neckline_idx - first_peak_idx)) / (second_peak_idx - first_peak_idx)

                    pattern_quality = (1.0 - peak_diff_percent) * 0.5 + (1.0 - time_symmetry) * 0.5

                    # بررسی شکست خط گردن
                    last_price = closes[-1]
                    breakout_confirmed = last_price < neckline_price

                    pattern = {
                        'type': 'double_top',
                        'direction': 'bearish',
                        'index': second_peak_idx,
                        'score': self.pattern_scores.get('double_top', 3.8) * pattern_quality,
                        'pattern_quality': round(pattern_quality, 2),
                        'neckline_price': float(neckline_price),
                        'price_target': float(neckline_price - pattern_height),
                        'breakout_confirmed': breakout_confirmed,
                        'points': {
                            'first_peak': int(first_peak_idx),
                            'second_peak': int(second_peak_idx),
                            'neckline': int(neckline_idx)
                        }
                    }

                    patterns.append(pattern)

            # Double Bottom
            if len(valleys) >= 2:
                for i in range(len(valleys) - 1):
                    first_valley_idx = valleys[i]
                    second_valley_idx = valleys[i + 1]

                    first_valley_price = lows[first_valley_idx]
                    second_valley_price = lows[second_valley_idx]

                    # بررسی تشابه دره‌ها
                    valley_diff_percent = abs(second_valley_price - first_valley_price) / first_valley_price
                    if valley_diff_percent > 0.03:
                        continue

                    # یافتن قله بین دو دره
                    peak_between = [
                        p for p in peaks
                        if first_valley_idx < p < second_valley_idx
                    ]

                    if not peak_between:
                        continue

                    neckline_idx = peak_between[0]
                    neckline_price = highs[neckline_idx]

                    # محاسبه کیفیت الگو
                    pattern_height = neckline_price - first_valley_price
                    time_symmetry = abs((second_valley_idx - neckline_idx) -
                                      (neckline_idx - first_valley_idx)) / (second_valley_idx - first_valley_idx)

                    pattern_quality = (1.0 - valley_diff_percent) * 0.5 + (1.0 - time_symmetry) * 0.5

                    # بررسی شکست خط گردن
                    last_price = closes[-1]
                    breakout_confirmed = last_price > neckline_price

                    pattern = {
                        'type': 'double_bottom',
                        'direction': 'bullish',
                        'index': second_valley_idx,
                        'score': self.pattern_scores.get('double_bottom', 3.8) * pattern_quality,
                        'pattern_quality': round(pattern_quality, 2),
                        'neckline_price': float(neckline_price),
                        'price_target': float(neckline_price + pattern_height),
                        'breakout_confirmed': breakout_confirmed,
                        'points': {
                            'first_valley': int(first_valley_idx),
                            'second_valley': int(second_valley_idx),
                            'neckline': int(neckline_idx)
                        }
                    }

                    patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای دوقلو: {e}")
            return []

    async def _detect_wedge_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """تشخیص الگوهای گوه (Wedge)"""
        patterns = []

        if df is None or len(df) < 30:
            return patterns

        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values

            # یافتن قله‌ها و دره‌ها
            peaks, valleys = self.find_peaks_and_valleys(
                tuple(closes), distance=5, prominence_factor=0.05
            )

            if len(peaks) < 3 or len(valleys) < 3:
                return patterns

            # بررسی آخرین نقاط
            recent_peaks = peaks[-5:] if len(peaks) >= 5 else peaks
            recent_valleys = valleys[-5:] if len(valleys) >= 5 else valleys

            if len(recent_peaks) < 3 or len(recent_valleys) < 3:
                return patterns

            # محاسبه خطوط روند
            peak_x = recent_peaks
            peak_y = highs[recent_peaks]
            valley_x = recent_valleys
            valley_y = lows[recent_valleys]

            upper_slope, upper_intercept = np.polyfit(peak_x, peak_y, 1)
            lower_slope, lower_intercept = np.polyfit(valley_x, valley_y, 1)

            # تشخیص نوع گوه
            wedge_type = None
            direction = None

            # هر دو خط باید هم‌جهت باشند
            if upper_slope > 0.001 and lower_slope > 0.001:
                # گوه صعودی
                if upper_slope < lower_slope:
                    wedge_type = 'rising_wedge'
                    direction = 'bearish'  # معمولاً نزولی است
            elif upper_slope < -0.001 and lower_slope < -0.001:
                # گوه نزولی
                if abs(upper_slope) > abs(lower_slope):
                    wedge_type = 'falling_wedge'
                    direction = 'bullish'  # معمولاً صعودی است

            if not wedge_type:
                return patterns

            # محاسبه نقطه همگرایی
            if abs(upper_slope - lower_slope) > 1e-6:
                convergence_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)

                # بررسی اعتبار همگرایی
                last_idx = len(closes) - 1
                if convergence_x <= last_idx or convergence_x > last_idx + 50:
                    return patterns
            else:
                return patterns

            # محاسبه کیفیت الگو
            total_touches = len(recent_peaks) + len(recent_valleys)
            pattern_quality = min(1.0, total_touches / 8) * 0.5

            # بررسی کاهش حجم (در صورت وجود)
            if 'volume' in df.columns:
                volumes = df['volume'].values
                vol_trend = np.polyfit(range(len(recent_peaks)),
                                     volumes[recent_peaks], 1)[0]
                if vol_trend < 0:  # حجم کاهشی
                    pattern_quality += 0.5

            # موقعیت قیمت در الگو
            current_upper = upper_slope * last_idx + upper_intercept
            current_lower = lower_slope * last_idx + lower_intercept
            pattern_width = current_upper - current_lower

            if pattern_width > 0:
                position_in_pattern = (closes[-1] - current_lower) / pattern_width
            else:
                position_in_pattern = 0.5

            # هدف قیمتی
            pattern_height = max(peak_y) - min(valley_y)
            if direction == 'bullish':
                price_target = closes[-1] + pattern_height * 0.7
            else:
                price_target = closes[-1] - pattern_height * 0.7

            pattern = {
                'type': wedge_type,
                'direction': direction,
                'index': last_idx,
                'score': self.pattern_scores.get(wedge_type, 3.2) * pattern_quality,
                'pattern_quality': round(pattern_quality, 2),
                'convergence_x': float(convergence_x),
                'position_in_pattern': float(position_in_pattern),
                'price_target': float(price_target),
                'upper_slope': float(upper_slope),
                'lower_slope': float(lower_slope)
            }

            patterns.append(pattern)

            return patterns

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای گوه: {e}")
            return []

    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """تشخیص سطوح حمایت و مقاومت - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'resistance_levels': [],
            'support_levels': [],
            'details': {},
            'zones': {}
        }

        if df is None or len(df) < lookback:
            results['status'] = 'insufficient_data'
            return results

        try:
            df_window = df.iloc[-lookback:]
            highs = df_window['high'].values.astype(np.float64)
            lows = df_window['low'].values.astype(np.float64)
            closes = df_window['close'].values.astype(np.float64)

            # یافتن قله‌ها و دره‌ها
            resistance_peaks, _ = self.find_peaks_and_valleys(
                tuple(highs),
                order=self.peak_detection_settings['order'],
                distance=self.peak_detection_settings['distance']
            )

            _, support_valleys = self.find_peaks_and_valleys(
                tuple(lows),
                order=self.peak_detection_settings['order'],
                distance=self.peak_detection_settings['distance']
            )

            # استخراج سطوح
            resistance_levels_raw = highs[resistance_peaks] if len(resistance_peaks) > 0 else []
            support_levels_raw = lows[support_valleys] if len(support_valleys) > 0 else []

            # محاسبه ATR برای خوشه‌بندی
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            last_atr = atr[~np.isnan(atr)][-1] if not np.all(np.isnan(atr)) else (highs[-1] - lows[-1])

            # ادغام سطوح نزدیک
            results['resistance_levels'] = self._consolidate_levels(
                resistance_levels_raw, last_atr
            )
            results['support_levels'] = self._consolidate_levels(
                support_levels_raw, last_atr
            )

            # تحلیل وضعیت فعلی
            current_close = closes[-1]

            # شناسایی شکست‌ها
            results['details'] = self._analyze_sr_breakouts(
                df_window, results['resistance_levels'],
                results['support_levels'], current_close
            )

            # تحلیل مناطق (zones)
            results['resistance_zones'] = self._analyze_sr_zones(
                results['resistance_levels'], current_close, 'resistance'
            )
            results['support_zones'] = self._analyze_sr_zones(
                results['support_levels'], current_close, 'support'
            )

            return results

        except Exception as e:
            logger.error(f"خطا در تشخیص حمایت/مقاومت: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _consolidate_levels(self, levels: np.ndarray, atr: float) -> List[Dict[str, Any]]:
        """ادغام سطوح نزدیک به هم"""
        if len(levels) == 0 or atr <= 1e-9:
            return [{'price': float(level), 'strength': 1.0} for level in np.unique(levels)]

        # آستانه برای ادغام
        threshold = atr * 0.3
        if threshold <= 1e-9:
            threshold = np.mean(levels) * 0.001 if np.mean(levels) > 0 else 1e-5

        # خوشه‌بندی
        sorted_levels = np.sort(levels)
        clusters = []

        if len(sorted_levels) > 0:
            current_cluster = [sorted_levels[0]]

            for i in range(1, len(sorted_levels)):
                if abs(sorted_levels[i] - np.mean(current_cluster)) <= threshold:
                    current_cluster.append(sorted_levels[i])
                else:
                    # ذخیره خوشه فعلی
                    cluster_mean = np.mean(current_cluster)
                    cluster_strength = self._calculate_level_strength(current_cluster)
                    clusters.append({
                        'price': float(cluster_mean),
                        'strength': float(cluster_strength),
                        'touches': len(current_cluster)
                    })

                    # شروع خوشه جدید
                    current_cluster = [sorted_levels[i]]

            # آخرین خوشه
            if current_cluster:
                cluster_mean = np.mean(current_cluster)
                cluster_strength = self._calculate_level_strength(current_cluster)
                clusters.append({
                    'price': float(cluster_mean),
                    'strength': float(cluster_strength),
                    'touches': len(current_cluster)
                })

        return sorted(clusters, key=lambda x: x['price'])

    def _calculate_level_strength(self, touches: List[float]) -> float:
        """محاسبه قدرت سطح براساس تعداد برخوردها"""
        count = len(touches)
        if count == 0:
            return 0.0

        # قدرت براساس تعداد
        count_strength = min(1.0, count / 3)

        # قدرت براساس پراکندگی
        if count > 1:
            std_dev = np.std(touches)
            mean_val = np.mean(touches)
            consistency = 1.0 - (std_dev / mean_val if mean_val > 0 else 0)
        else:
            consistency = 1.0

        return count_strength * 0.7 + consistency * 0.3

    def _analyze_sr_breakouts(self, df: pd.DataFrame, resistance_levels: List[Dict],
                            support_levels: List[Dict], current_close: float) -> Dict[str, Any]:
        """تحلیل شکست‌های حمایت و مقاومت"""
        details = {
            'nearest_resistance': None,
            'nearest_support': None,
            'broken_resistance': None,
            'broken_support': None,
            'price_position': 'neutral'
        }

        if len(df) < 2:
            return details

        prev_close = df['close'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_high = df['high'].iloc[-2]

        # یافتن نزدیک‌ترین سطوح
        resistance_candidates = [r for r in resistance_levels if r['price'] > current_close]
        support_candidates = [s for s in support_levels if s['price'] < current_close]

        if resistance_candidates:
            details['nearest_resistance'] = min(
                resistance_candidates,
                key=lambda x: x['price'] - current_close
            )

        if support_candidates:
            details['nearest_support'] = max(
                support_candidates,
                key=lambda x: current_close - x['price']
            )

        # بررسی شکست‌ها
        for level in resistance_levels:
            if (current_close > level['price'] and
                prev_close < level['price'] and
                df['close'].iloc[-1] > df['open'].iloc[-1]):  # کندل صعودی
                details['broken_resistance'] = level
                break

        for level in support_levels:
            if (current_close < level['price'] and
                prev_close > level['price'] and
                df['close'].iloc[-1] < df['open'].iloc[-1]):  # کندل نزولی
                details['broken_support'] = level
                break

        # تعیین موقعیت قیمت
        if details['nearest_resistance'] and details['nearest_support']:
            range_size = details['nearest_resistance']['price'] - details['nearest_support']['price']
            position_in_range = (current_close - details['nearest_support']['price']) / range_size

            if position_in_range > 0.8:
                details['price_position'] = 'near_resistance'
            elif position_in_range < 0.2:
                details['price_position'] = 'near_support'
            else:
                details['price_position'] = 'mid_range'

        return details

    def _analyze_sr_zones(self, levels: List[Dict[str, Any]], current_price: float,
                        zone_type: str) -> Dict[str, Any]:
        """تحلیل مناطق حمایت/مقاومت"""
        if not levels:
            return {'status': 'no_levels', 'zones': []}

        try:
            # مرتب‌سازی براساس فاصله از قیمت فعلی
            sorted_levels = sorted(
                levels,
                key=lambda x: abs(x['price'] - current_price)
            )

            # شناسایی خوشه‌ها
            clusters = []
            current_cluster = [sorted_levels[0]]

            for i in range(1, len(sorted_levels)):
                # بررسی نزدیکی (1% اختلاف)
                if abs(sorted_levels[i]['price'] - sorted_levels[i-1]['price']) / sorted_levels[i-1]['price'] < 0.01:
                    current_cluster.append(sorted_levels[i])
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_levels[i]]

            # آخرین خوشه
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

            # تحلیل مناطق
            zones = []
            for cluster in clusters:
                zone_min = min(level['price'] for level in cluster)
                zone_max = max(level['price'] for level in cluster)
                zone_strength = sum(level['strength'] for level in cluster) / len(cluster)
                zone_center = (zone_min + zone_max) / 2
                zone_width = zone_max - zone_min

                zone = {
                    'min': float(zone_min),
                    'max': float(zone_max),
                    'center': float(zone_center),
                    'width': float(zone_width),
                    'strength': float(zone_strength),
                    'levels_count': len(cluster),
                    'distance_to_price': float(abs(zone_center - current_price)),
                    'type': zone_type
                }
                zones.append(zone)

            # مرتب‌سازی براساس فاصله
            zones = sorted(zones, key=lambda x: x['distance_to_price'])

            return {'status': 'ok', 'zones': zones}

        except Exception as e:
            logger.error(f"خطا در تحلیل مناطق {zone_type}: {e}")
            return {'status': 'error', 'message': str(e), 'zones': []}

    def detect_harmonic_patterns(self, df: pd.DataFrame, lookback: int = 100,
                               tolerance: float = 0.03) -> List[Dict[str, Any]]:
        """تشخیص الگوهای هارمونیک - نسخه بهینه‌شده"""
        patterns = []

        if not self.harmonic_enabled or df is None or len(df) < lookback:
            return patterns

        try:
            df_window = df.iloc[-lookback:].copy()

            # یافتن نقاط کلیدی
            peaks, valleys = self.find_peaks_and_valleys(
                tuple(df_window['close'].values),
                distance=self.peak_detection_settings['distance'],
                prominence_factor=self.peak_detection_settings['prominence_factor']
            )

            if len(peaks) + len(valleys) < 5:
                return patterns

            # ترکیب و مرتب‌سازی نقاط
            all_points = [(idx, 'peak', df_window['high'].iloc[idx]) for idx in peaks]
            all_points.extend([(idx, 'valley', df_window['low'].iloc[idx]) for idx in valleys])
            all_points.sort(key=lambda x: x[0])

            # بررسی الگوهای 5 نقطه‌ای
            for i in range(len(all_points) - 4):
                X, A, B, C, D = all_points[i:i + 5]

                # بررسی تناوب قله و دره
                if not self._check_alternating_points([X, A, B, C, D]):
                    continue

                # استخراج قیمت‌ها
                x_price = X[2]
                a_price = A[2]
                b_price = B[2]
                c_price = C[2]
                d_price = D[2]

                # محاسبه نسبت‌های فیبوناچی
                ratios = self._calculate_harmonic_ratios(
                    x_price, a_price, b_price, c_price, d_price
                )

                if not ratios:
                    continue

                # بررسی الگوهای مختلف
                detected_patterns = []

                # Gartley
                if self._is_gartley_pattern(ratios, tolerance):
                    detected_patterns.append(('gartley', 0.9))

                # Bat
                if self._is_bat_pattern(ratios, tolerance):
                    detected_patterns.append(('bat', 0.85))

                # Butterfly
                if self._is_butterfly_pattern(ratios, tolerance):
                    detected_patterns.append(('butterfly', 0.8))

                # Crab
                if self._is_crab_pattern(ratios, tolerance):
                    detected_patterns.append(('crab', 0.95))

                # Cypher
                if self._is_cypher_pattern(ratios, tolerance):
                    detected_patterns.append(('cypher', 0.75))

                # انتخاب بهترین الگو
                if detected_patterns:
                    best_pattern, base_confidence = max(
                        detected_patterns,
                        key=lambda x: x[1]
                    )

                    # تعیین جهت
                    is_bullish = A[1] == 'valley'
                    pattern_type = f"bullish_{best_pattern}" if is_bullish else f"bearish_{best_pattern}"
                    direction = 'bullish' if is_bullish else 'bearish'

                    # محاسبه اطمینان
                    confidence = self._calculate_harmonic_confidence(
                        ratios, best_pattern, tolerance, base_confidence
                    )

                    if confidence >= self.harmonic_min_quality:
                        pattern = {
                            'type': pattern_type,
                            'direction': direction,
                            'points': {
                                'X': {'index': X[0], 'price': float(x_price)},
                                'A': {'index': A[0], 'price': float(a_price)},
                                'B': {'index': B[0], 'price': float(b_price)},
                                'C': {'index': C[0], 'price': float(c_price)},
                                'D': {'index': D[0], 'price': float(d_price)},
                            },
                            'ratios': {
                                'AB/XA': float(ratios['ab_xa']),
                                'BC/AB': float(ratios['bc_ab']),
                                'CD/BC': float(ratios['cd_bc']),
                                'BD/XA': float(ratios['bd_xa'])
                            },
                            'confidence': float(confidence),
                            'index': D[0],
                            'score': self.pattern_scores.get(pattern_type, 4.0) * confidence,
                            'completion_zone': self._calculate_prz(
                                X, A, B, C, best_pattern, is_bullish
                            )
                        }
                        patterns.append(pattern)

            # مرتب‌سازی براساس اطمینان
            patterns = sorted(patterns, key=lambda x: x['confidence'], reverse=True)

            # حذف الگوهای تکراری
            patterns = self._remove_duplicate_harmonics(patterns)

            return patterns[:5]  # حداکثر 5 الگو

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای هارمونیک: {e}")
            return []

    def _check_alternating_points(self, points: List[Tuple]) -> bool:
        """بررسی تناوب قله و دره"""
        for i in range(len(points) - 1):
            if points[i][1] == points[i + 1][1]:
                return False
        return True

    def _calculate_harmonic_ratios(self, x: float, a: float, b: float,
                                 c: float, d: float) -> Optional[Dict[str, float]]:
        """محاسبه نسبت‌های هارمونیک"""
        xa = abs(x - a)
        ab = abs(a - b)
        bc = abs(b - c)
        cd = abs(c - d)
        bd = abs(b - d)

        if xa == 0 or ab == 0 or bc == 0:
            return None

        return {
            'ab_xa': ab / xa,
            'bc_ab': bc / ab,
            'cd_bc': cd / bc,
            'bd_xa': bd / xa,
            'cd_xa': cd / xa
        }

    def _is_gartley_pattern(self, ratios: Dict[str, float], tolerance: float) -> bool:
        """بررسی الگوی Gartley"""
        return (
            self._is_in_range(ratios['ab_xa'], 0.618, tolerance) and
            self._is_in_range(ratios['bc_ab'], 0.382, tolerance) and
            self._is_in_range(ratios['cd_bc'], 1.272, tolerance) and
            self._is_in_range(ratios['bd_xa'], 0.786, tolerance)
        )

    def _is_bat_pattern(self, ratios: Dict[str, float], tolerance: float) -> bool:
        """بررسی الگوی Bat"""
        return (
            self._is_in_range(ratios['ab_xa'], 0.382, tolerance) and
            self._is_in_range(ratios['bc_ab'], 0.382, tolerance) and
            self._is_in_range(ratios['cd_bc'], 1.618, tolerance) and
            self._is_in_range(ratios['bd_xa'], 0.886, tolerance)
        )

    def _is_butterfly_pattern(self, ratios: Dict[str, float], tolerance: float) -> bool:
        """بررسی الگوی Butterfly"""
        return (
            self._is_in_range(ratios['ab_xa'], 0.786, tolerance) and
            self._is_in_range(ratios['bc_ab'], 0.382, tolerance) and
            self._is_in_range(ratios['cd_bc'], 1.618, tolerance) and
            self._is_in_range(ratios['bd_xa'], 1.27, tolerance)
        )

    def _is_crab_pattern(self, ratios: Dict[str, float], tolerance: float) -> bool:
        """بررسی الگوی Crab"""
        return (
            self._is_in_range(ratios['ab_xa'], 0.382, tolerance) and
            self._is_in_range(ratios['bc_ab'], 0.618, tolerance) and
            self._is_in_range(ratios['cd_bc'], 3.618, tolerance) and
            self._is_in_range(ratios['bd_xa'], 1.618, tolerance)
        )

    def _is_cypher_pattern(self, ratios: Dict[str, float], tolerance: float) -> bool:
        """بررسی الگوی Cypher"""
        return (
            self._is_in_range(ratios['ab_xa'], 0.382, tolerance * 1.5) and
            self._is_in_range(ratios['bc_ab'], 1.272, tolerance) and
            self._is_in_range(ratios['cd_bc'], 0.786, tolerance) and
            self._is_in_range(ratios['cd_xa'], 0.786, tolerance)
        )

    def _is_in_range(self, value: float, target: float, tolerance: float) -> bool:
        """بررسی قرارگیری در محدوده"""
        return abs(value - target) <= tolerance

    def _calculate_harmonic_confidence(self, ratios: Dict[str, float], pattern: str,
                                     tolerance: float, base_confidence: float) -> float:
        """محاسبه اطمینان الگوی هارمونیک"""
        # نسبت‌های هدف برای هر الگو
        target_ratios = {
            'gartley': {'ab_xa': 0.618, 'bc_ab': 0.382, 'cd_bc': 1.272, 'bd_xa': 0.786},
            'bat': {'ab_xa': 0.382, 'bc_ab': 0.382, 'cd_bc': 1.618, 'bd_xa': 0.886},
            'butterfly': {'ab_xa': 0.786, 'bc_ab': 0.382, 'cd_bc': 1.618, 'bd_xa': 1.27},
            'crab': {'ab_xa': 0.382, 'bc_ab': 0.618, 'cd_bc': 3.618, 'bd_xa': 1.618},
            'cypher': {'ab_xa': 0.382, 'bc_ab': 1.272, 'cd_bc': 0.786, 'cd_xa': 0.786}
        }

        if pattern not in target_ratios:
            return base_confidence

        targets = target_ratios[pattern]
        deviations = []

        for key, target in targets.items():
            if key in ratios:
                deviation = abs(ratios[key] - target) / tolerance
                deviations.append(1.0 - min(1.0, deviation))

        if deviations:
            avg_accuracy = sum(deviations) / len(deviations)
            return base_confidence * avg_accuracy

        return base_confidence

    def _calculate_prz(self, X: Tuple, A: Tuple, B: Tuple, C: Tuple,
                     pattern: str, is_bullish: bool) -> Dict[str, float]:
        """محاسبه منطقه بازگشت احتمالی (PRZ)"""
        x_price = X[2]
        a_price = A[2]
        b_price = B[2]
        c_price = C[2]

        # محاسبه سطوح فیبوناچی مختلف
        xa_range = abs(x_price - a_price)
        bc_range = abs(b_price - c_price)

        prz_levels = []

        # سطوح براساس نوع الگو
        if pattern == 'gartley':
            prz_levels.append(a_price + 0.786 * xa_range * (1 if is_bullish else -1))
        elif pattern == 'bat':
            prz_levels.append(a_price + 0.886 * xa_range * (1 if is_bullish else -1))
        elif pattern == 'butterfly':
            prz_levels.append(a_price + 1.27 * xa_range * (1 if is_bullish else -1))
        elif pattern == 'crab':
            prz_levels.append(a_price + 1.618 * xa_range * (1 if is_bullish else -1))

        # سطح پروجکشن BC
        prz_levels.append(c_price + 1.618 * bc_range * (1 if is_bullish else -1))

        # محدوده PRZ
        prz_min = min(prz_levels)
        prz_max = max(prz_levels)

        return {
            'min': float(prz_min),
            'max': float(prz_max),
            'center': float((prz_min + prz_max) / 2),
            'width': float(prz_max - prz_min)
        }

    def _remove_duplicate_harmonics(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """حذف الگوهای هارمونیک تکراری"""
        if len(patterns) <= 1:
            return patterns

        unique_patterns = []

        for pattern in patterns:
            is_duplicate = False

            for unique in unique_patterns:
                # بررسی همپوشانی نقطه D
                if abs(pattern['points']['D']['index'] - unique['points']['D']['index']) < 5:
                    # انتخاب الگو با اطمینان بیشتر
                    if pattern['confidence'] > unique['confidence']:
                        unique_patterns.remove(unique)
                        unique_patterns.append(pattern)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_patterns.append(pattern)

        return unique_patterns

    def detect_price_channels(self, df: pd.DataFrame, lookback: int = 100,
                            min_touches: int = 3) -> Dict[str, Any]:
        """تشخیص کانال‌های قیمتی - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'channels': [],
            'details': {},
            'active_channel': None
        }

        if not self.channel_enabled or df is None or len(df) < lookback:
            results['status'] = 'insufficient_data'
            return results

        try:
            df_window = df.iloc[-lookback:]
            highs = df_window['high'].values
            lows = df_window['low'].values
            closes = df_window['close'].values

            # یافتن قله‌ها و دره‌ها
            peaks, valleys = self.find_peaks_and_valleys(
                tuple(closes),
                distance=self.peak_detection_settings['distance'],
                prominence_factor=self.peak_detection_settings['prominence_factor']
            )

            if len(peaks) < min_touches or len(valleys) < min_touches:
                results['status'] = 'insufficient_points'
                return results

            # تحلیل کانال‌های مختلف
            channels = []

            # کانال اصلی با تمام نقاط
            main_channel = self._analyze_channel(
                peaks, valleys, highs, lows, closes
            )

            if main_channel and main_channel['quality'] >= self.channel_quality_threshold:
                channels.append(main_channel)

            # کانال‌های کوتاه‌مدت (آخرین نقاط)
            if len(peaks) >= min_touches and len(valleys) >= min_touches:
                recent_channel = self._analyze_channel(
                    peaks[-min_touches:], valleys[-min_touches:],
                    highs, lows, closes
                )

                if recent_channel and recent_channel['quality'] >= self.channel_quality_threshold:
                    recent_channel['timeframe'] = 'short'
                    channels.append(recent_channel)

            # انتخاب بهترین کانال فعال
            if channels:
                # اولویت به کانال‌هایی که قیمت داخل آنهاست
                active_channels = [
                    ch for ch in channels
                    if 0.1 <= ch['position_in_channel'] <= 0.9
                ]

                if active_channels:
                    results['active_channel'] = max(
                        active_channels,
                        key=lambda x: x['quality']
                    )
                else:
                    results['active_channel'] = max(
                        channels,
                        key=lambda x: x['quality']
                    )

            results['channels'] = channels

            # تولید سیگنال براساس کانال فعال
            if results['active_channel']:
                signal = self._generate_channel_signal(
                    results['active_channel'], closes[-1]
                )
                if signal:
                    results['signal'] = signal

            return results

        except Exception as e:
            logger.error(f"خطا در تشخیص کانال‌های قیمت: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _analyze_channel(self, peaks: np.ndarray, valleys: np.ndarray,
                       highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray) -> Optional[Dict[str, Any]]:
        """تحلیل یک کانال قیمتی"""
        try:
            if len(peaks) < 2 or len(valleys) < 2:
                return None

            # محاسبه خطوط کانال
            peak_indices = np.array(peaks)
            peak_values = highs[peak_indices]
            valley_indices = np.array(valleys)
            valley_values = lows[valley_indices]

            # رگرسیون خطی
            upper_slope, upper_intercept = np.polyfit(peak_indices, peak_values, 1)
            lower_slope, lower_intercept = np.polyfit(valley_indices, valley_values, 1)

            # بررسی موازی بودن
            slope_diff = abs(upper_slope - lower_slope)
            avg_slope = (upper_slope + lower_slope) / 2

            if avg_slope != 0:
                parallelism = 1.0 - min(1.0, slope_diff / abs(avg_slope))
            else:
                parallelism = 1.0 if slope_diff < 0.0001 else 0.0

            # محاسبه نقاط برخورد معتبر
            upper_touches = self._count_valid_touches(
                highs, peak_indices, upper_slope, upper_intercept, is_upper=True
            )
            lower_touches = self._count_valid_touches(
                lows, valley_indices, lower_slope, lower_intercept, is_upper=False
            )

            # موقعیت قیمت فعلی
            last_idx = len(closes) - 1
            upper_current = upper_slope * last_idx + upper_intercept
            lower_current = lower_slope * last_idx + lower_intercept
            channel_width = upper_current - lower_current

            if channel_width <= 0:
                return None

            last_close = closes[-1]
            position_in_channel = (last_close - lower_current) / channel_width

            # تعیین جهت کانال
            if avg_slope > 0.001:
                channel_direction = 'ascending'
            elif avg_slope < -0.001:
                channel_direction = 'descending'
            else:
                channel_direction = 'horizontal'

            # محاسبه کیفیت کانال
            quality = self._calculate_channel_quality(
                upper_touches, lower_touches, parallelism, position_in_channel
            )

            # بررسی شکست
            breakout = self._detect_channel_breakout(
                last_close, upper_current, lower_current,
                channel_width, closes[-5:]
            )

            return {
                'type': f'{channel_direction}_channel',
                'direction': channel_direction,
                'upper_slope': float(upper_slope),
                'upper_intercept': float(upper_intercept),
                'lower_slope': float(lower_slope),
                'lower_intercept': float(lower_intercept),
                'width': float(channel_width),
                'quality': float(quality),
                'position_in_channel': float(position_in_channel),
                'breakout': breakout,
                'up_touches': int(upper_touches),
                'down_touches': int(lower_touches),
                'parallelism': float(parallelism)
            }

        except Exception as e:
            logger.error(f"خطا در تحلیل کانال: {e}")
            return None

    def _count_valid_touches(self, prices: np.ndarray, touch_indices: np.ndarray,
                           slope: float, intercept: float, is_upper: bool) -> int:
        """شمارش برخوردهای معتبر با خط کانال"""
        valid_touches = 0
        tolerance = np.std(prices) * 0.1  # 10% انحراف معیار

        for idx in touch_indices:
            expected_price = slope * idx + intercept
            actual_price = prices[idx]

            if is_upper:
                if abs(actual_price - expected_price) < tolerance:
                    valid_touches += 1
            else:
                if abs(actual_price - expected_price) < tolerance:
                    valid_touches += 1

        return valid_touches

    def _calculate_channel_quality(self, upper_touches: int, lower_touches: int,
                                 parallelism: float, position: float) -> float:
        """محاسبه کیفیت کانال"""
        # کیفیت براساس تعداد برخوردها
        touch_quality = min(1.0, (upper_touches + lower_touches) / 8)

        # کیفیت براساس موازی بودن
        parallel_quality = parallelism

        # کیفیت براساس موقعیت قیمت
        position_quality = 1.0 if 0.2 <= position <= 0.8 else 0.7

        # ترکیب فاکتورها
        quality = (
            touch_quality * 0.4 +
            parallel_quality * 0.4 +
            position_quality * 0.2
        )

        return quality

    def _detect_channel_breakout(self, last_close: float, upper: float,
                                lower: float, width: float,
                                recent_closes: np.ndarray) -> Optional[str]:
        """تشخیص شکست کانال"""
        # بررسی شکست صعودی
        if last_close > upper:
            # تایید با بسته شدن بالای کانال
            if len(recent_closes) >= 2 and recent_closes[-2] < upper:
                return 'up'

        # بررسی شکست نزولی
        elif last_close < lower:
            # تایید با بسته شدن پایین کانال
            if len(recent_closes) >= 2 and recent_closes[-2] > lower:
                return 'down'

        return None

    def _generate_channel_signal(self, channel: Dict[str, Any],
                               current_price: float) -> Optional[Dict[str, Any]]:
        """تولید سیگنال براساس کانال"""
        position = channel['position_in_channel']
        breakout = channel['breakout']
        direction = channel['direction']
        quality = channel['quality']

        # سیگنال شکست
        if breakout == 'up':
            return {
                'type': 'channel_breakout',
                'direction': 'bullish',
                'score': 4.0 * quality
            }
        elif breakout == 'down':
            return {
                'type': 'channel_breakout',
                'direction': 'bearish',
                'score': 4.0 * quality
            }

        # سیگنال برگشت از کف/سقف کانال
        elif position < 0.2 and direction != 'descending':
            return {
                'type': 'channel_bounce',
                'direction': 'bullish',
                'score': 3.0 * quality
            }
        elif position > 0.8 and direction != 'ascending':
            return {
                'type': 'channel_bounce',
                'direction': 'bearish',
                'score': 3.0 * quality
            }

        return None

    def detect_cyclical_patterns(self, df: pd.DataFrame, lookback: int = 200) -> Dict[str, Any]:
        """تشخیص الگوهای چرخه‌ای - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'cycles': [],
            'forecast': None,
            'dominant_cycle': None,
            'details': {}
        }

        if not self.cycle_enabled or df is None or len(df) < lookback:
            results['status'] = 'disabled_or_insufficient_data'
            return results

        try:
            df_window = df.iloc[-lookback:].copy()
            closes = df_window['close'].values

            # حذف روند برای تمرکز بر نوسانات
            detrended = self._detrend_data(closes)

            # تحلیل فوریه
            cycles = self._analyze_frequency_spectrum(detrended)

            if len(cycles) < self.cycle_min_cycles:
                results['status'] = 'insufficient_cycles'
                return results

            # انتخاب چرخه‌های غالب
            dominant_cycles = cycles[:min(5, len(cycles))]
            results['cycles'] = dominant_cycles

            if dominant_cycles:
                results['dominant_cycle'] = dominant_cycles[0]

            # پیش‌بینی براساس چرخه‌ها
            forecast_result = self._generate_cycle_forecast(
                closes, dominant_cycles, 20
            )

            if forecast_result:
                results['forecast'] = forecast_result

                # تولید سیگنال
                signal = self._generate_cycle_signal(
                    forecast_result, closes[-1]
                )
                if signal:
                    results['signal'] = signal

            # تحلیل‌های اضافی
            results['details'] = {
                'total_cycles_detected': len(cycles),
                'significant_cycles': len(dominant_cycles),
                'cycle_strength': self._calculate_cycle_strength(dominant_cycles),
                'phase_analysis': self._analyze_cycle_phase(detrended, dominant_cycles[0]) if dominant_cycles else None
            }

            return results

        except Exception as e:
            logger.error(f"خطا در تشخیص الگوهای چرخه‌ای: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _detrend_data(self, data: np.ndarray) -> np.ndarray:
        """حذف روند از داده‌ها"""
        x = np.arange(len(data))

        # رگرسیون چندجمله‌ای درجه 2
        coeffs = np.polyfit(x, data, 2)
        trend = np.polyval(coeffs, x)

        return data - trend

    def _analyze_frequency_spectrum(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """تحلیل طیف فرکانسی"""
        from scipy import fft

        # FFT
        fft_values = fft.rfft(data)
        fft_freqs = fft.rfftfreq(len(data))
        fft_magnitudes = np.abs(fft_values)

        # یافتن فرکانس‌های مهم
        threshold = np.mean(fft_magnitudes) + np.std(fft_magnitudes)
        significant_indices = np.where(fft_magnitudes > threshold)[0]

        cycles = []
        for idx in significant_indices:
            if fft_freqs[idx] > 0 and idx > 0:  # حذف DC component
                period = 1 / fft_freqs[idx]

                # فیلتر دوره‌های معقول
                if 2 <= period <= len(data) / 2:
                    amplitude = fft_magnitudes[idx] / len(data)
                    phase = np.angle(fft_values[idx])

                    # قدرت چرخه
                    cycle_power = (amplitude / np.mean(np.abs(data))) * 100

                    cycles.append({
                        'period': int(period),
                        'frequency': float(fft_freqs[idx]),
                        'amplitude': float(amplitude),
                        'amplitude_percent': float(cycle_power),
                        'phase': float(phase),
                        'magnitude': float(fft_magnitudes[idx])
                    })

        # مرتب‌سازی براساس دامنه
        cycles.sort(key=lambda x: x['magnitude'], reverse=True)

        return cycles

    def _generate_cycle_forecast(self, original_data: np.ndarray,
                               cycles: List[Dict[str, Any]],
                               forecast_length: int) -> Optional[Dict[str, Any]]:
        """تولید پیش‌بینی براساس چرخه‌ها"""
        try:
            if not cycles:
                return None

            # استخراج روند
            x = np.arange(len(original_data))
            trend_coeffs = np.polyfit(x, original_data, 2)

            # پیش‌بینی
            forecast = np.zeros(forecast_length)

            for i in range(forecast_length):
                # موقعیت زمانی
                t = len(original_data) + i

                # مولفه روند
                trend_value = np.polyval(trend_coeffs, t)

                # مولفه‌های چرخه‌ای
                cycle_value = 0
                for cycle in cycles[:3]:  # حداکثر 3 چرخه مهم
                    period = cycle['period']
                    amplitude = cycle['amplitude']
                    phase = cycle['phase']

                    cycle_component = amplitude * np.cos(
                        2 * np.pi * t / period + phase
                    )
                    cycle_value += cycle_component

                forecast[i] = trend_value + cycle_value

            # تحلیل پیش‌بینی
            forecast_direction = 'bullish' if forecast[-1] > original_data[-1] else 'bearish'
            forecast_strength = abs(forecast[-1] - original_data[-1]) / original_data[-1]

            # نقاط عطف در پیش‌بینی
            turning_points = self._find_forecast_turning_points(forecast)

            return {
                'values': [float(f) for f in forecast],
                'direction': forecast_direction,
                'strength': float(forecast_strength),
                'turning_points': turning_points,
                'confidence': self._calculate_forecast_confidence(cycles)
            }

        except Exception as e:
            logger.error(f"خطا در تولید پیش‌بینی چرخه‌ای: {e}")
            return None

    def _find_forecast_turning_points(self, forecast: np.ndarray) -> List[Dict[str, Any]]:
        """یافتن نقاط عطف در پیش‌بینی"""
        turning_points = []

        if len(forecast) < 3:
            return turning_points

        for i in range(1, len(forecast) - 1):
            # قله
            if forecast[i] > forecast[i-1] and forecast[i] > forecast[i+1]:
                turning_points.append({
                    'index': i,
                    'type': 'peak',
                    'value': float(forecast[i])
                })
            # دره
            elif forecast[i] < forecast[i-1] and forecast[i] < forecast[i+1]:
                turning_points.append({
                    'index': i,
                    'type': 'valley',
                    'value': float(forecast[i])
                })

        return turning_points

    def _calculate_forecast_confidence(self, cycles: List[Dict[str, Any]]) -> float:
        """محاسبه اطمینان پیش‌بینی"""
        if not cycles:
            return 0.0

        # قدرت چرخه‌ها
        total_strength = sum(c['amplitude_percent'] for c in cycles[:3])
        strength_factor = min(1.0, total_strength / 20)

        # پایداری چرخه‌ها (نسبت بزرگترین به دومین)
        if len(cycles) >= 2:
            stability = cycles[0]['magnitude'] / cycles[1]['magnitude']
            stability_factor = min(1.0, stability / 3)
        else:
            stability_factor = 0.5

        return strength_factor * 0.6 + stability_factor * 0.4

    def _generate_cycle_signal(self, forecast: Dict[str, Any],
                             current_price: float) -> Optional[Dict[str, Any]]:
        """تولید سیگنال براساس پیش‌بینی چرخه‌ای"""
        if not forecast or forecast['confidence'] < 0.5:
            return None

        direction = forecast['direction']
        strength = forecast['strength']
        confidence = forecast['confidence']

        # بررسی نقاط عطف آینده
        turning_points = forecast.get('turning_points', [])

        # سیگنال براساس جهت کلی
        if direction == 'bullish' and strength > 0.02:  # حداقل 2% حرکت
            score = 2.5 * confidence * min(1.0, strength * 10)

            return {
                'type': 'cycle_bullish_forecast',
                'direction': 'bullish',
                'score': score,
                'forecast_change': strength,
                'next_turning_point': turning_points[0] if turning_points else None
            }

        elif direction == 'bearish' and strength > 0.02:
            score = 2.5 * confidence * min(1.0, strength * 10)

            return {
                'type': 'cycle_bearish_forecast',
                'direction': 'bearish',
                'score': score,
                'forecast_change': strength,
                'next_turning_point': turning_points[0] if turning_points else None
            }

        return None

    def _calculate_cycle_strength(self, cycles: List[Dict[str, Any]]) -> float:
        """محاسبه قدرت کلی چرخه‌ها"""
        if not cycles:
            return 0.0

        # مجموع دامنه‌های نرمالایز شده
        total_amplitude = sum(c['amplitude_percent'] for c in cycles)

        # نرمالایز به 0-1
        return min(1.0, total_amplitude / 30)

    def _analyze_cycle_phase(self, data: np.ndarray,
                           dominant_cycle: Dict[str, Any]) -> Dict[str, str]:
        """تحلیل فاز فعلی چرخه"""
        if not dominant_cycle:
            return {'phase': 'unknown', 'position': 0.0}

        period = dominant_cycle['period']
        phase = dominant_cycle['phase']

        # موقعیت در چرخه
        current_position = (len(data) % period) / period

        # تعیین فاز
        if current_position < 0.25:
            phase_name = 'ascending'
        elif current_position < 0.5:
            phase_name = 'peak_formation'
        elif current_position < 0.75:
            phase_name = 'descending'
        else:
            phase_name = 'bottom_formation'

        return {
            'phase': phase_name,
            'position': float(current_position)
        }

    # --- توابع تحلیل اندیکاتورها ---

    def _detect_divergence_generic(self, price_series: pd.Series,
                                 indicator_series: pd.Series,
                                 indicator_name: str) -> List[Dict[str, Any]]:
        """تشخیص واگرایی بین قیمت و اندیکاتور - نسخه بهینه‌شده"""
        signals = []

        if price_series is None or indicator_series is None:
            return signals

        # حداقل داده مورد نیاز
        period = min(len(price_series), len(indicator_series))
        if period < 20:
            return signals

        try:
            # پنجره‌های همسان
            price_window = price_series.iloc[-period:]
            indicator_window = indicator_series.iloc[-period:]

            # بررسی مقادیر معتبر
            if price_window.isna().all() or indicator_window.isna().all():
                return signals

            # یافتن قله‌ها و دره‌ها
            price_values_tuple = tuple(price_window.values)
            indicator_values_tuple = tuple(indicator_window.values)

            price_peaks_idx, price_valleys_idx = self.find_peaks_and_valleys(
                price_values_tuple,
                distance=5,
                prominence_factor=0.05,
                window_size=period
            )

            ind_peaks_idx, ind_valleys_idx = self.find_peaks_and_valleys(
                indicator_values_tuple,
                distance=5,
                prominence_factor=0.1,
                window_size=period
            )

            if (len(price_peaks_idx) == 0 and len(price_valleys_idx) == 0) or \
               (len(ind_peaks_idx) == 0 and len(ind_valleys_idx) == 0):
                return signals

            # تبدیل به ایندکس‌های مطلق
            price_peaks_abs = price_window.index[price_peaks_idx].tolist() if len(price_peaks_idx) > 0 else []
            price_valleys_abs = price_window.index[price_valleys_idx].tolist() if len(price_valleys_idx) > 0 else []
            ind_peaks_abs = indicator_window.index[ind_peaks_idx].tolist() if len(ind_peaks_idx) > 0 else []
            ind_valleys_abs = indicator_window.index[ind_valleys_idx].tolist() if len(ind_valleys_idx) > 0 else []

            # تشخیص واگرایی نزولی
            bearish_divergences = self._detect_bearish_divergences(
                price_window, indicator_window,
                price_peaks_abs, ind_peaks_abs,
                indicator_name
            )
            signals.extend(bearish_divergences)

            # تشخیص واگرایی صعودی
            bullish_divergences = self._detect_bullish_divergences(
                price_window, indicator_window,
                price_valleys_abs, ind_valleys_abs,
                indicator_name
            )
            signals.extend(bullish_divergences)

            # فیلتر سیگنال‌های اخیر
            recent_candle_limit = 10
            if len(signals) > 0 and len(price_window) > recent_candle_limit:
                recent_threshold = price_window.index[-recent_candle_limit]
                signals = [s for s in signals if s['index'] >= recent_threshold]

            # مرتب‌سازی براساس قدرت
            return sorted(signals, key=lambda x: x.get('strength', 0), reverse=True)

        except Exception as e:
            logger.error(f"خطا در تشخیص واگرایی {indicator_name}: {str(e)}")
            return []

    def _detect_bearish_divergences(self, price_window: pd.Series,
                                  indicator_window: pd.Series,
                                  price_peaks: List, ind_peaks: List,
                                  indicator_name: str) -> List[Dict[str, Any]]:
        """تشخیص واگرایی‌های نزولی"""
        divergences = []

        if len(price_peaks) < 2 or len(ind_peaks) < 2:
            return divergences

        max_peaks_to_check = min(len(price_peaks), 5)

        for i in range(max_peaks_to_check - 1):
            cur_idx = len(price_peaks) - 1 - i
            prev_idx = cur_idx - 1

            if prev_idx < 0 or cur_idx >= len(price_peaks):
                continue

            p1_idx = price_peaks[prev_idx]
            p2_idx = price_peaks[cur_idx]

            p1_price = price_window.loc[p1_idx]
            p2_price = price_window.loc[p2_idx]

            # قیمت باید قله بالاتر بسازد
            if p2_price <= p1_price:
                continue

            # یافتن قله‌های متناظر در اندیکاتور
            ind_p1_idx = self._find_closest_peak(ind_peaks, p1_idx)
            ind_p2_idx = self._find_closest_peak(ind_peaks, p2_idx)

            if ind_p1_idx is None or ind_p2_idx is None:
                continue

            ind_p1_val = indicator_window.loc[ind_p1_idx]
            ind_p2_val = indicator_window.loc[ind_p2_idx]

            # اندیکاتور باید قله پایین‌تر بسازد
            if ind_p2_val >= ind_p1_val:
                continue

            # محاسبه قدرت واگرایی
            div_strength = self._calculate_divergence_strength(
                p1_price, p2_price, ind_p1_val, ind_p2_val
            )

            if div_strength >= self.divergence_sensitivity:
                div_score = self.pattern_scores.get(
                    f"{indicator_name}_bearish_divergence", 3.5
                ) * div_strength

                divergences.append({
                    'type': f'{indicator_name}_bearish_divergence',
                    'direction': 'bearish',
                    'index': p2_idx,
                    'score': div_score,
                    'strength': float(div_strength),
                    'details': {
                        'price_p1': float(p1_price),
                        'price_p2': float(p2_price),
                        'ind_p1': float(ind_p1_val),
                        'ind_p2': float(ind_p2_val),
                        'price_change_pct': float((p2_price - p1_price) / p1_price),
                        'ind_change_pct': float((ind_p1_val - ind_p2_val) / ind_p1_val) if ind_p1_val != 0 else 0
                    }
                })

        return divergences

    def _detect_bullish_divergences(self, price_window: pd.Series,
                                  indicator_window: pd.Series,
                                  price_valleys: List, ind_valleys: List,
                                  indicator_name: str) -> List[Dict[str, Any]]:
        """تشخیص واگرایی‌های صعودی"""
        divergences = []

        if len(price_valleys) < 2 or len(ind_valleys) < 2:
            return divergences

        max_valleys_to_check = min(len(price_valleys), 5)

        for i in range(max_valleys_to_check - 1):
            cur_idx = len(price_valleys) - 1 - i
            prev_idx = cur_idx - 1

            if prev_idx < 0 or cur_idx >= len(price_valleys):
                continue

            p1_idx = price_valleys[prev_idx]
            p2_idx = price_valleys[cur_idx]

            p1_price = price_window.loc[p1_idx]
            p2_price = price_window.loc[p2_idx]

            # قیمت باید دره پایین‌تر بسازد
            if p2_price >= p1_price:
                continue

            # یافتن دره‌های متناظر در اندیکاتور
            ind_p1_idx = self._find_closest_peak(ind_valleys, p1_idx)
            ind_p2_idx = self._find_closest_peak(ind_valleys, p2_idx)

            if ind_p1_idx is None or ind_p2_idx is None:
                continue

            ind_p1_val = indicator_window.loc[ind_p1_idx]
            ind_p2_val = indicator_window.loc[ind_p2_idx]

            # اندیکاتور باید دره بالاتر بسازد
            if ind_p2_val <= ind_p1_val:
                continue

            # محاسبه قدرت واگرایی
            div_strength = self._calculate_divergence_strength(
                p1_price, p2_price, ind_p1_val, ind_p2_val
            )

            if div_strength >= self.divergence_sensitivity:
                div_score = self.pattern_scores.get(
                    f"{indicator_name}_bullish_divergence", 3.5
                ) * div_strength

                divergences.append({
                    'type': f'{indicator_name}_bullish_divergence',
                    'direction': 'bullish',
                    'index': p2_idx,
                    'score': div_score,
                    'strength': float(div_strength),
                    'details': {
                        'price_p1': float(p1_price),
                        'price_p2': float(p2_price),
                        'ind_p1': float(ind_p1_val),
                        'ind_p2': float(ind_p2_val),
                        'price_change_pct': float((p1_price - p2_price) / p1_price),
                        'ind_change_pct': float((ind_p2_val - ind_p1_val) / ind_p1_val) if ind_p1_val != 0 else 0
                    }
                })

        return divergences

    def _calculate_divergence_strength(self, p1_price: float, p2_price: float,
                                     ind_p1: float, ind_p2: float) -> float:
        """محاسبه قدرت واگرایی"""
        # تغییرات درصدی
        price_change_pct = abs(p2_price - p1_price) / p1_price if p1_price != 0 else 0
        ind_change_pct = abs(ind_p2 - ind_p1) / abs(ind_p1) if ind_p1 != 0 else 0

        # ترکیب تغییرات
        div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)

        return div_strength

    def _find_closest_peak(self, peaks_list: List, target_idx) -> Optional[Any]:
        """یافتن نزدیک‌ترین قله/دره به ایندکس هدف"""
        if not peaks_list or target_idx is None:
            return None

        try:
            # محاسبه فواصل
            if all(isinstance(idx, pd.Timestamp) for idx in peaks_list):
                if isinstance(target_idx, pd.Timestamp):
                    distances = [(abs((idx - target_idx).total_seconds()), idx) for idx in peaks_list]
                else:
                    # استفاده از موقعیت نسبی
                    distances = [(abs(i - target_idx), idx) for i, idx in enumerate(peaks_list)]
            else:
                # فرض بر این که همه از یک نوع هستند
                distances = [(abs(idx - target_idx), idx) for idx in peaks_list]

            if distances:
                closest_peak = min(distances, key=lambda x: x[0])[1]
                return closest_peak

            return None

        except Exception as e:
            logger.debug(f"خطا در یافتن نزدیک‌ترین قله: {e}")
            return peaks_list[0] if peaks_list else None

    def detect_divergence(self, price_data: np.ndarray, indicator_data: np.ndarray,
                        threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """تشخیص واگرایی بین قیمت و اندیکاتور (سازگاری با نسخه قدیمی)"""
        sensitivity = threshold if threshold is not None else self.divergence_sensitivity

        if price_data is None or indicator_data is None or \
           len(price_data) != len(indicator_data) or len(price_data) < 20:
            return []

        try:
            price_series = pd.Series(price_data)
            indicator_series = pd.Series(indicator_data)

            return self._detect_divergence_generic(
                price_series, indicator_series, 'indicator'
            )

        except Exception as e:
            logger.error(f"خطا در detect_divergence: {e}")
            return []

    # تابع ادامه کد
    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل جامع MACD - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'signals': [],
            'direction': 'neutral',
            'market_type': 'unknown',
            'details': {}
        }

        if df is None or len(df) < 50:
            results['status'] = 'insufficient_data'
            return results

        try:
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')
            close_p = df['close'].values.astype(np.float64)

            # محاسبه EMAs برای تشخیص نوع بازار
            ema20_key = self._cache_key(symbol, timeframe, 'EMA', (20,))
            ema50_key = self._cache_key(symbol, timeframe, 'EMA', (50,))

            ema20 = self._get_cached_indicator(ema20_key)
            if ema20 is None:
                ema20 = talib.EMA(close_p, timeperiod=20)
                self._cache_indicator(ema20_key, ema20)

            ema50 = self._get_cached_indicator(ema50_key)
            if ema50 is None:
                ema50 = talib.EMA(close_p, timeperiod=50)
                self._cache_indicator(ema50_key, ema50)

            # محاسبه MACD
            macd_key = self._cache_key(symbol, timeframe, 'MACD', (12, 26, 9))
            cached_macd = self._get_cached_indicator(macd_key)

            if cached_macd is not None:
                dif, dea, hist = cached_macd
            else:
                dif, dea, hist = talib.MACD(close_p, fastperiod=12, slowperiod=26, signalperiod=9)
                self._cache_indicator(macd_key, (dif, dea, hist))

            # تبدیل به pandas series
            dif_s = pd.Series(dif, index=df.index)
            dea_s = pd.Series(dea, index=df.index)
            hist_s = pd.Series(hist, index=df.index)
            ema20_s = pd.Series(ema20, index=df.index)
            ema50_s = pd.Series(ema50, index=df.index)

            # تشخیص نوع بازار
            market_type = self._detect_macd_market_type(dif_s, hist_s, ema20_s, ema50_s)

            # تحلیل‌های مختلف MACD
            all_signals = []

            # 1. تقاطع‌های MACD
            macd_crosses = self._detect_detailed_macd_crosses(dif_s, dea_s, df.index)
            all_signals.extend(macd_crosses)

            # 2. رفتار خط DIF
            dif_behavior = self._detect_dif_behavior(dif_s, df.index)
            all_signals.extend(dif_behavior)

            # 3. تحلیل هیستوگرام
            hist_analysis = self._analyze_macd_histogram(
                hist_s, pd.Series(close_p, index=df.index), df.index
            )
            all_signals.extend(hist_analysis)

            # 4. واگرایی MACD
            macd_divergence = self._detect_divergence_generic(
                pd.Series(close_p, index=df.index), dif_s, 'macd'
            )
            all_signals.extend(macd_divergence)

            # محاسبه جهت و امتیازات
            bullish_score = sum(
                s.get('score', 0) for s in all_signals
                if s.get('direction', '') == 'bullish'
            )
            bearish_score = sum(
                s.get('score', 0) for s in all_signals
                if s.get('direction', '') == 'bearish'
            )

            # تعیین جهت کلی
            direction = 'neutral'
            if bullish_score > bearish_score * 1.1:
                direction = 'bullish'
            elif bearish_score > bullish_score * 1.1:
                direction = 'bearish'

            # جزئیات فعلی
            last_valid_idx = -1
            while last_valid_idx >= -len(df) and (
                np.isnan(dif[last_valid_idx]) or
                np.isnan(dea[last_valid_idx]) or
                np.isnan(hist[last_valid_idx])
            ):
                last_valid_idx -= 1

            if abs(last_valid_idx) <= len(df):
                curr_dif = dif[last_valid_idx]
                curr_dea = dea[last_valid_idx]
                curr_hist = hist[last_valid_idx]

                # محاسبه شیب‌ها
                lookback = 3
                hist_slope = self._calculate_slope(hist, last_valid_idx, lookback)
                dif_slope = self._calculate_slope(dif, last_valid_idx, lookback)
                dea_slope = self._calculate_slope(dea, last_valid_idx, lookback)

                results['details'] = {
                    'dif': round(float(curr_dif), 6),
                    'dea': round(float(curr_dea), 6),
                    'hist': round(float(curr_hist), 6),
                    'dif_slope': round(float(dif_slope), 6),
                    'dea_slope': round(float(dea_slope), 6),
                    'hist_slope': round(float(hist_slope), 6),
                    'market_type': market_type,
                    'dif_above_zero': curr_dif > 0,
                    'hist_above_zero': curr_hist > 0
                }

            # نتایج نهایی
            results['signals'] = all_signals
            results['direction'] = direction
            results['market_type'] = market_type
            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل MACD: {e}", exc_info=True)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _calculate_slope(self, data: np.ndarray, idx: int, lookback: int) -> float:
        """محاسبه شیب با مدیریت خطا"""
        try:
            if idx - lookback < -len(data):
                return 0.0

            return data[idx] - data[max(-len(data), idx - lookback)]

        except:
            return 0.0

    def _detect_macd_market_type(self, dif: pd.Series, hist: pd.Series,
                                ema20: pd.Series, ema50: pd.Series) -> str:
        """تشخیص نوع بازار براساس MACD"""
        if dif.empty or hist.empty or ema20.empty or ema50.empty or len(dif) < 1:
            return "unknown_data"

        try:
            curr_dif = dif.iloc[-1]
            curr_hist = hist.iloc[-1]
            curr_ema20 = ema20.iloc[-1]
            curr_ema50 = ema50.iloc[-1]

            # تعیین نوع بازار
            if curr_dif > 0 and curr_hist > 0 and curr_ema20 > curr_ema50:
                return "A_bullish_strong"
            elif curr_dif > 0 and curr_hist < 0 and curr_ema20 > curr_ema50:
                return "B_bullish_correction"
            elif curr_dif < 0 and curr_hist < 0 and curr_ema20 < curr_ema50:
                return "C_bearish_strong"
            elif curr_dif < 0 and curr_hist > 0 and curr_ema20 < curr_ema50:
                return "D_bearish_rebound"
            else:
                return "X_transition"

        except:
            return "error"

    def _detect_detailed_macd_crosses(self, dif: pd.Series, dea: pd.Series,
                                    dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تشخیص تقاطع‌های MACD با جزئیات"""
        signals = []
        min_len = max(2, self.macd_cross_period)

        if len(dif) < min_len or len(dea) < min_len:
            return signals

        try:
            # بررسی داده‌های اخیر
            cross_window = min(len(dif), self.macd_cross_period)
            dif_window = dif.iloc[-cross_window:]
            dea_window = dea.iloc[-cross_window:]
            dates_window = dates_index[-cross_window:]

            for i in range(1, len(dif_window)):
                current_idx = dates_window[i]
                dif_curr = dif_window.iloc[i]
                dif_prev = dif_window.iloc[i - 1]
                dea_curr = dea_window.iloc[i]
                dea_prev = dea_window.iloc[i - 1]

                # تقاطع صعودی (Golden Cross)
                if dif_prev < dea_prev and dif_curr > dea_curr:
                    cross_type = (
                        "macd_gold_cross_below_zero" if dif_curr < 0
                        else "macd_gold_cross_above_zero"
                    )

                    cross_strength = min(1.0, abs(dif_curr - dea_curr) * 10)
                    signal_score = self.pattern_scores.get(cross_type, 2.5) * cross_strength

                    signals.append({
                        'type': cross_type,
                        'direction': 'bullish',
                        'index': current_idx,
                        'date': current_idx,
                        'score': signal_score,
                        'strength': cross_strength,
                        'details': {
                            'dif': float(dif_curr),
                            'dea': float(dea_curr),
                            'above_zero': dif_curr > 0,
                            'cross_angle': self._calculate_cross_angle(
                                dif_prev, dif_curr, dea_prev, dea_curr
                            )
                        }
                    })

                # تقاطع نزولی (Death Cross)
                elif dif_prev > dea_prev and dif_curr < dea_curr:
                    cross_type = (
                        "macd_death_cross_above_zero" if dif_curr > 0
                        else "macd_death_cross_below_zero"
                    )

                    cross_strength = min(1.0, abs(dif_curr - dea_curr) * 10)
                    signal_score = self.pattern_scores.get(cross_type, 2.5) * cross_strength

                    signals.append({
                        'type': cross_type,
                        'direction': 'bearish',
                        'index': current_idx,
                        'date': current_idx,
                        'score': signal_score,
                        'strength': cross_strength,
                        'details': {
                            'dif': float(dif_curr),
                            'dea': float(dea_curr),
                            'above_zero': dif_curr > 0,
                            'cross_angle': self._calculate_cross_angle(
                                dif_prev, dif_curr, dea_prev, dea_curr
                            )
                        }
                    })

            # فیلتر سیگنال‌های کندل آخر
            if dates_index is not None and len(dates_index) > 0:
                last_date = dates_index[-1]
                recent_signals = [s for s in signals if s['date'] == last_date]
                return recent_signals

            return signals

        except Exception as e:
            logger.error(f"خطا در تشخیص تقاطع‌های MACD: {e}")
            return []

    def _calculate_cross_angle(self, dif_prev: float, dif_curr: float,
                             dea_prev: float, dea_curr: float) -> float:
        """محاسبه زاویه تقاطع"""
        try:
            # محاسبه شیب خطوط
            dif_slope = dif_curr - dif_prev
            dea_slope = dea_curr - dea_prev

            # زاویه نسبی
            angle_diff = abs(dif_slope - dea_slope)

            return round(angle_diff * 100, 2)  # نرمالایز به 0-100

        except:
            return 0.0

    def _detect_dif_behavior(self, dif: pd.Series, dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تحلیل رفتار خط DIF"""
        signals = []

        if not isinstance(dif, pd.Series) or len(dif) < self.macd_cross_period:
            return signals

        try:
            dif_vals = dif.values

            # تقاطع با خط صفر
            zero_crosses = self._detect_zero_crosses(dif_vals, dates_index)
            signals.extend(zero_crosses)

            # شکست خط روند
            if len(dif) >= self.macd_trendline_period:
                trendline_breaks = self._detect_trendline_breaks(
                    dif.iloc[-self.macd_trendline_period:], dates_index
                )
                signals.extend(trendline_breaks)

            # فیلتر سیگنال‌های اخیر
            if dates_index is not None and len(dates_index) > 0:
                last_date = dates_index[-1]
                recent_signals = [s for s in signals if s.get('date') == last_date]
                return recent_signals

            return signals

        except Exception as e:
            logger.error(f"خطا در تحلیل رفتار DIF: {e}")
            return []

    def _detect_zero_crosses(self, dif_vals: np.ndarray,
                           dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تشخیص تقاطع با خط صفر"""
        signals = []
        cross_up_count = 0
        cross_down_count = 0

        for i in range(1, len(dif_vals)):
            crossed_up = dif_vals[i-1] < 0 and dif_vals[i] > 0
            crossed_down = dif_vals[i-1] > 0 and dif_vals[i] < 0

            if crossed_up or crossed_down:
                current_idx = dates_index[i]

                if crossed_up:
                    cross_up_count += 1
                    signal_type = f"dif_cross_zero_up_{'first' if cross_up_count == 1 else 'second'}"
                    direction = 'bullish'
                else:
                    cross_down_count += 1
                    signal_type = f"dif_cross_zero_down_{'first' if cross_down_count == 1 else 'second'}"
                    direction = 'bearish'

                signals.append({
                    'type': signal_type,
                    'direction': direction,
                    'index': current_idx,
                    'date': current_idx,
                    'score': self.pattern_scores.get(signal_type, 2.0),
                    'details': {
                        'dif_value': float(dif_vals[i]),
                        'cross_count': cross_up_count if crossed_up else cross_down_count
                    }
                })

        return signals

    def _detect_trendline_breaks(self, dif_window: pd.Series,
                                dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تشخیص شکست خط روند"""
        signals = []

        try:
            # هموارسازی داده‌ها
            if len(dif_window) > self.macd_peak_detection_settings['smooth_kernel'] * 2:
                smooth_dif = scipy.signal.medfilt(
                    dif_window.values,
                    kernel_size=self.macd_peak_detection_settings['smooth_kernel']
                )

                # یافتن نقاط کلیدی
                peaks, valleys = self.find_peaks_and_valleys(
                    tuple(smooth_dif),
                    distance=self.macd_peak_detection_settings['distance'],
                    prominence_factor=self.macd_peak_detection_settings['prominence_factor']
                )

                # بررسی شکست مقاومت
                if len(peaks) >= 2:
                    resistance_break = self._check_trendline_break(peaks, smooth_dif, dif_window, dates_index, True
                    )
                    if resistance_break:
                        signals.append(resistance_break)

                # بررسی شکست حمایت
                if len(valleys) >= 2:
                    support_break = self._check_trendline_break(
                        valleys, smooth_dif, dif_window, dates_index, False
                    )
                    if support_break:
                        signals.append(support_break)

            return signals

        except Exception as e:
            logger.error(f"خطا در تشخیص شکست خط روند: {e}")
            return []

    def _check_trendline_break(self, points: np.ndarray, smooth_data: np.ndarray,
                              original_data: pd.Series, dates_index: pd.Index,
                              is_resistance: bool) -> Optional[Dict[str, Any]]:
        """بررسی شکست خط روند"""
        if len(points) < 2:
            return None

        try:
            # انتخاب دو نقطه آخر
            p1_idx, p2_idx = points[-2], points[-1]
            p1_val, p2_val = smooth_data[p1_idx], smooth_data[p2_idx]

            # محاسبه خط روند
            if p2_idx == p1_idx:
                return None

            slope = (p2_val - p1_val) / (p2_idx - p1_idx)
            intercept = p1_val - slope * p1_idx

            # بررسی شکست در داده‌های اخیر
            for i in range(p2_idx + 1, len(original_data)):
                trendline_val = slope * i + intercept
                current_val = original_data.iloc[i]
                margin = abs(current_val * 0.01)

                if is_resistance and current_val > trendline_val + margin:
                    return {
                        'type': 'dif_trendline_break_up',
                        'direction': 'bullish',
                        'index': dates_index[i],
                        'date': dates_index[i],
                        'score': self.pattern_scores.get('dif_trendline_break_up', 3.0),
                        'details': {
                            'break_value': float(current_val),
                            'trendline_value': float(trendline_val),
                            'break_percent': float((current_val - trendline_val) / trendline_val)
                        }
                    }

                elif not is_resistance and current_val < trendline_val - margin:
                    return {
                        'type': 'dif_trendline_break_down',
                        'direction': 'bearish',
                        'index': dates_index[i],
                        'date': dates_index[i],
                        'score': self.pattern_scores.get('dif_trendline_break_down', 3.0),
                        'details': {
                            'break_value': float(current_val),
                            'trendline_value': float(trendline_val),
                            'break_percent': float((trendline_val - current_val) / trendline_val)
                        }
                    }

            return None

        except Exception as e:
            logger.error(f"خطا در بررسی شکست خط روند: {e}")
            return None

    def _analyze_macd_histogram(self, hist: pd.Series, close: pd.Series,
                               dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تحلیل هیستوگرام MACD"""
        signals = []
        min_len = max(10, self.macd_hist_period)

        if len(hist) < min_len or len(close) != len(hist):
            return signals

        try:
            # یافتن قله‌ها و دره‌های هیستوگرام
            peaks, valleys = self.find_peaks_and_valleys(
                tuple(hist.values),
                distance=self.macd_peak_detection_settings['distance'],
                prominence_factor=self.macd_peak_detection_settings['prominence_factor']
            )

            # تحلیل انقباض (Shrinking)
            shrinking_signals = self._detect_histogram_shrinking(
                hist, peaks, valleys, dates_index
            )
            signals.extend(shrinking_signals)

            # واگرایی هیستوگرام
            if len(peaks) >= 2 or len(valleys) >= 2:
                hist_divergences = self._detect_histogram_divergence(
                    hist, close, peaks, valleys, dates_index
                )
                signals.extend(hist_divergences)

            # تحلیل بین‌های کشنده (Kill Bins)
            kill_bins = self._detect_kill_bins(hist, valleys, dates_index)
            signals.extend(kill_bins)

            # فیلتر سیگنال‌های اخیر
            if dates_index is not None and len(dates_index) > 0:
                last_date = dates_index[-1]
                recent_signals = [s for s in signals if s.get('date') == last_date]
                return recent_signals

            return signals

        except Exception as e:
            logger.error(f"خطا در تحلیل هیستوگرام MACD: {e}")
            return []

    def _detect_histogram_shrinking(self, hist: pd.Series, peaks: np.ndarray,
                                   valleys: np.ndarray, dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تشخیص انقباض هیستوگرام"""
        signals = []

        # Shrink Head (قله‌های مثبت کاهشی)
        for idx in peaks:
            if hist.iloc[idx] > 0:
                signals.append({
                    'type': 'macd_hist_shrink_head',
                    'direction': 'bearish',
                    'index': dates_index[idx],
                    'date': dates_index[idx],
                    'score': self.pattern_scores.get('macd_hist_shrink_head', 1.5),
                    'details': {
                        'hist_value': float(hist.iloc[idx])
                    }
                })

        # Pull Feet (دره‌های منفی افزایشی)
        for idx in valleys:
            if hist.iloc[idx] < 0:
                signals.append({
                    'type': 'macd_hist_pull_feet',
                    'direction': 'bullish',
                    'index': dates_index[idx],
                    'date': dates_index[idx],
                    'score': self.pattern_scores.get('macd_hist_pull_feet', 1.5),
                    'details': {
                        'hist_value': float(hist.iloc[idx])
                    }
                })

        return signals

    def _detect_histogram_divergence(self, hist: pd.Series, close: pd.Series,
                                   peaks: np.ndarray, valleys: np.ndarray,
                                   dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تشخیص واگرایی هیستوگرام"""
        signals = []

        # واگرایی نزولی (Top Divergence)
        if len(peaks) >= 2:
            p1_idx, p2_idx = peaks[-2], peaks[-1]
            p1_date, p2_date = dates_index[p1_idx], dates_index[p2_idx]

            if (hist.iloc[p2_idx] < hist.iloc[p1_idx] and
                close.loc[p2_date] > close.loc[p1_date]):
                signals.append({
                    'type': 'macd_hist_top_divergence',
                    'direction': 'bearish',
                    'index': p2_date,
                    'date': p2_date,
                    'score': self.pattern_scores.get('macd_hist_top_divergence', 3.8),
                    'details': {
                        'hist_p1': float(hist.iloc[p1_idx]),
                        'hist_p2': float(hist.iloc[p2_idx]),
                        'price_p1': float(close.loc[p1_date]),
                        'price_p2': float(close.loc[p2_date])
                    }
                })

        # واگرایی صعودی (Bottom Divergence)
        if len(valleys) >= 2:
            v1_idx, v2_idx = valleys[-2], valleys[-1]
            v1_date, v2_date = dates_index[v1_idx], dates_index[v2_idx]

            if (hist.iloc[v2_idx] > hist.iloc[v1_idx] and
                close.loc[v2_date] < close.loc[v1_date]):
                signals.append({
                    'type': 'macd_hist_bottom_divergence',
                    'direction': 'bullish',
                    'index': v2_date,
                    'date': v2_date,
                    'score': self.pattern_scores.get('macd_hist_bottom_divergence', 3.8),
                    'details': {
                        'hist_v1': float(hist.iloc[v1_idx]),
                        'hist_v2': float(hist.iloc[v2_idx]),
                        'price_v1': float(close.loc[v1_date]),
                        'price_v2': float(close.loc[v2_date])
                    }
                })

        return signals

    def _detect_kill_bins(self, hist: pd.Series, valleys: np.ndarray,
                        dates_index: pd.Index) -> List[Dict[str, Any]]:
        """تشخیص بین‌های کشنده"""
        signals = []

        if len(valleys) < 2:
            return signals

        for i in range(len(valleys) - 1):
            v1_idx, v2_idx = valleys[i], valleys[i + 1]

            # بررسی اینکه هر دو دره منفی باشند
            if hist.iloc[v1_idx] < 0 and hist.iloc[v2_idx] < 0:
                # بررسی هیستوگرام بین دو دره
                hist_between = hist.iloc[v1_idx:v2_idx + 1]

                # اگر تمام مقادیر منفی باشند
                if not hist_between.empty and hist_between.max() < 0:
                    signals.append({
                        'type': 'macd_hist_kill_long_bin',
                        'direction': 'bearish',
                        'index': dates_index[v2_idx],
                        'date': dates_index[v2_idx],
                        'score': self.pattern_scores.get('macd_hist_kill_long_bin', 2.0),
                        'details': {
                            'bin_start': int(v1_idx),
                            'bin_end': int(v2_idx),
                            'min_hist': float(hist_between.min())
                        }
                    })

        return signals

    def analyze_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل اندیکاتورهای مومنتوم - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'direction': 'neutral',
            'signals': [],
            'details': {}
        }

        required_len = 35
        if df is None or len(df) < required_len:
            results['status'] = 'insufficient_data'
            return results

        try:
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')

            # آماده‌سازی داده‌ها
            close_p = df['close'].values.astype(np.float64)
            high_p = df['high'].values.astype(np.float64)
            low_p = df['low'].values.astype(np.float64)

            # محاسبه اندیکاتورها با کش
            indicators = self._calculate_momentum_indicators(
                symbol, timeframe, high_p, low_p, close_p, df
            )

            if not indicators:
                results['status'] = 'calculation_error'
                return results

            # استخراج سیگنال‌ها
            momentum_signals = []

            # 1. تقاطع MACD
            macd_signals = self._detect_macd_crossovers(indicators)
            momentum_signals.extend(macd_signals)

            # 2. سیگنال‌های RSI
            rsi_signals = self._detect_rsi_signals(indicators)
            momentum_signals.extend(rsi_signals)

            # 3. سیگنال‌های استوکاستیک
            stoch_signals = self._detect_stochastic_signals(indicators)
            momentum_signals.extend(stoch_signals)

            # 4. سیگنال‌های MFI
            if indicators.get('mfi') is not None:
                mfi_signals = self._detect_mfi_signals(indicators)
                momentum_signals.extend(mfi_signals)

            # 5. واگرایی RSI
            close_s = pd.Series(close_p)
            rsi_s = pd.Series(indicators['rsi'])
            rsi_divergences = self._detect_divergence_generic(close_s, rsi_s, 'rsi')
            momentum_signals.extend(rsi_divergences)

            # محاسبه جهت کلی
            bullish_score = sum(
                s['score'] for s in momentum_signals
                if 'bullish' in s.get('direction', s.get('type', ''))
            )
            bearish_score = sum(
                s['score'] for s in momentum_signals
                if 'bearish' in s.get('direction', s.get('type', ''))
            )

            if bullish_score > bearish_score:
                results['direction'] = 'bullish'
            elif bearish_score > bullish_score:
                results['direction'] = 'bearish'

            # جزئیات نهایی
            results['signals'] = momentum_signals
            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)
            results['details'] = self._prepare_momentum_details(indicators)

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل اندیکاتورهای مومنتوم: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _calculate_momentum_indicators(self, symbol: str, timeframe: str,
                                     high: np.ndarray, low: np.ndarray,
                                     close: np.ndarray, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """محاسبه اندیکاتورهای مومنتوم با کش"""
        try:
            indicators = {}

            # MACD
            macd_key = self._cache_key(symbol, timeframe, 'MACD', (12, 26, 9))
            cached_macd = self._get_cached_indicator(macd_key)
            if cached_macd is not None:
                macd, macd_signal, macd_hist = cached_macd
            else:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                self._cache_indicator(macd_key, (macd, macd_signal, macd_hist))

            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist

            # RSI
            rsi_key = self._cache_key(symbol, timeframe, 'RSI', (14,))
            rsi = self._get_cached_indicator(rsi_key)
            if rsi is None:
                rsi = talib.RSI(close, timeperiod=14)
                self._cache_indicator(rsi_key, rsi)

            indicators['rsi'] = rsi

            # Stochastic
            stoch_key = self._cache_key(symbol, timeframe, 'STOCH', (14, 3, 3))
            cached_stoch = self._get_cached_indicator(stoch_key)
            if cached_stoch is not None:
                slowk, slowd = cached_stoch
            else:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                self._cache_indicator(stoch_key, (slowk, slowd))

            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd

            # MFI
            if 'volume' in df.columns:
                try:
                    volume = df['volume'].values.astype(np.float64)
                    mfi_key = self._cache_key(symbol, timeframe, 'MFI', (14,))
                    mfi = self._get_cached_indicator(mfi_key)
                    if mfi is None:
                        mfi = talib.MFI(high, low, close, volume, timeperiod=14)
                        self._cache_indicator(mfi_key, mfi)
                    indicators['mfi'] = mfi
                except:
                    indicators['mfi'] = None
            else:
                indicators['mfi'] = None

            # یافتن آخرین ایندکس معتبر
            last_valid_idx = self._find_last_valid_momentum_index(indicators)
            if last_valid_idx is None:
                return None

            indicators['last_valid_idx'] = last_valid_idx

            return indicators

        except Exception as e:
            logger.error(f"خطا در محاسبه اندیکاتورهای مومنتوم: {e}")
            return None

    def _find_last_valid_momentum_index(self, indicators: Dict[str, Any]) -> Optional[int]:
        """یافتن آخرین ایندکس معتبر برای اندیکاتورها"""
        arrays = [
            indicators['macd'],
            indicators['rsi'],
            indicators['stoch_k']
        ]

        if indicators.get('mfi') is not None:
            arrays.append(indicators['mfi'])

        # یافتن آخرین ایندکس که همه اندیکاتورها معتبر هستند
        for i in range(-1, -len(arrays[0]) - 1, -1):
            if all(not np.isnan(arr[i]) for arr in arrays):
                return i

        return None

    def _detect_macd_crossovers(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تشخیص تقاطع‌های MACD"""
        signals = []
        idx = indicators['last_valid_idx']

        if idx is None or idx <= -len(indicators['macd']):
            return signals

        curr_macd = indicators['macd'][idx]
        prev_macd = indicators['macd'][idx - 1]
        curr_sig = indicators['macd_signal'][idx]
        prev_sig = indicators['macd_signal'][idx - 1]

        # تقاطع صعودی
        if curr_macd > curr_sig and prev_macd <= prev_sig:
            signals.append({
                'type': 'macd_bullish_crossover',
                'direction': 'bullish',
                'score': self.pattern_scores.get('macd_bullish_crossover', 2.2)
            })

        # تقاطع نزولی
        elif curr_macd < curr_sig and prev_macd >= prev_sig:
            signals.append({
                'type': 'macd_bearish_crossover',
                'direction': 'bearish',
                'score': self.pattern_scores.get('macd_bearish_crossover', 2.2)
            })

        # تقاطع خط صفر
        if curr_macd > 0 and prev_macd <= 0:
            signals.append({
                'type': 'macd_bullish_zero_cross',
                'direction': 'bullish',
                'score': self.pattern_scores.get('macd_bullish_zero_cross', 1.8)
            })
        elif curr_macd < 0 and prev_macd >= 0:
            signals.append({
                'type': 'macd_bearish_zero_cross',
                'direction': 'bearish',
                'score': self.pattern_scores.get('macd_bearish_zero_cross', 1.8)
            })

        return signals

    def _detect_rsi_signals(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تشخیص سیگنال‌های RSI"""
        signals = []
        idx = indicators['last_valid_idx']

        if idx is None or idx <= -len(indicators['rsi']):
            return signals

        curr_rsi = indicators['rsi'][idx]
        prev_rsi = indicators['rsi'][idx - 1]

        # برگشت از اشباع خرید/فروش
        if curr_rsi < 30 and curr_rsi > prev_rsi:
            signals.append({
                'type': 'rsi_oversold_reversal',
                'direction': 'bullish',
                'score': self.pattern_scores.get('rsi_oversold_reversal', 2.3)
            })
        elif curr_rsi > 70 and curr_rsi < prev_rsi:
            signals.append({
                'type': 'rsi_overbought_reversal',
                'direction': 'bearish',
                'score': self.pattern_scores.get('rsi_overbought_reversal', 2.3)
            })

        # تقاطع سطح 50
        if curr_rsi > 50 and prev_rsi <= 50:
            signals.append({
                'type': 'rsi_bullish_50_cross',
                'direction': 'bullish',
                'score': self.pattern_scores.get('rsi_bullish_50_cross', 1.5)
            })
        elif curr_rsi < 50 and prev_rsi >= 50:
            signals.append({
                'type': 'rsi_bearish_50_cross',
                'direction': 'bearish',
                'score': self.pattern_scores.get('rsi_bearish_50_cross', 1.5)
            })

        return signals

    def _detect_stochastic_signals(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تشخیص سیگنال‌های استوکاستیک"""
        signals = []
        idx = indicators['last_valid_idx']

        if idx is None or idx <= -len(indicators['stoch_k']):
            return signals

        curr_k = indicators['stoch_k'][idx]
        prev_k = indicators['stoch_k'][idx - 1]
        curr_d = indicators['stoch_d'][idx]
        prev_d = indicators['stoch_d'][idx - 1]

        # تقاطع در اشباع خرید/فروش
        if curr_k < 20 and curr_d < 20 and curr_k > curr_d and prev_k <= prev_d:
            signals.append({
                'type': 'stochastic_oversold_bullish_cross',
                'direction': 'bullish',
                'score': self.pattern_scores.get('stochastic_oversold_bullish_cross', 2.5)
            })
        elif curr_k > 80 and curr_d > 80 and curr_k < curr_d and prev_k >= prev_d:
            signals.append({
                'type': 'stochastic_overbought_bearish_cross',
                'direction': 'bearish',
                'score': self.pattern_scores.get('stochastic_overbought_bearish_cross', 2.5)
            })

        return signals

    def _detect_mfi_signals(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تشخیص سیگنال‌های MFI"""
        signals = []

        if indicators.get('mfi') is None:
            return signals

        idx = indicators['last_valid_idx']
        if idx is None or idx <= -len(indicators['mfi']):
            return signals

        curr_mfi = indicators['mfi'][idx]
        prev_mfi = indicators['mfi'][idx - 1]

        # برگشت از اشباع خرید/فروش
        if curr_mfi < 20 and curr_mfi > prev_mfi:
            signals.append({
                'type': 'mfi_oversold_reversal',
                'direction': 'bullish',
                'score': self.pattern_scores.get('mfi_oversold_reversal', 2.4)
            })
        elif curr_mfi > 80 and curr_mfi < prev_mfi:
            signals.append({
                'type': 'mfi_overbought_reversal',
                'direction': 'bearish',
                'score': self.pattern_scores.get('mfi_overbought_reversal', 2.4)
            })

        return signals

    def _prepare_momentum_details(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """آماده‌سازی جزئیات اندیکاتورهای مومنتوم"""
        idx = indicators.get('last_valid_idx', -1)

        details = {}

        if idx is not None and idx > -len(indicators['rsi']):
            details['rsi'] = round(indicators['rsi'][idx], 2)
            details['macd'] = round(indicators['macd'][idx], 5)
            details['macd_signal'] = round(indicators['macd_signal'][idx], 5)
            details['stoch_k'] = round(indicators['stoch_k'][idx], 2)
            details['stoch_d'] = round(indicators['stoch_d'][idx], 2)

            if indicators.get('mfi') is not None:
                details['mfi'] = round(indicators['mfi'][idx], 2)

            # وضعیت اندیکاتورها
            details['rsi_condition'] = (
                'oversold' if details['rsi'] < 30 else
                'overbought' if details['rsi'] > 70 else 'neutral'
            )

            details['stoch_condition'] = (
                'oversold' if details['stoch_k'] < 20 and details['stoch_d'] < 20 else
                'overbought' if details['stoch_k'] > 80 and details['stoch_d'] > 80 else 'neutral'
            )

            if 'mfi' in details:
                details['mfi_condition'] = (
                    'oversold' if details['mfi'] < 20 else
                    'overbought' if details['mfi'] > 80 else 'neutral'
                )

        return details

    def detect_reversal_conditions(self, analysis_results: Dict[str, Dict[str, Any]],
                                 timeframe: str) -> Tuple[bool, float]:
        """تشخیص شرایط برگشتی - نسخه بهینه‌شده"""
        try:
            result = analysis_results.get(timeframe, {})
            if result.get('status') != 'ok':
                return False, 0.0

            is_reversal = False
            strength = 0.0

            # 1. واگرایی RSI/MACD
            divergence_strength = self._check_divergence_reversals(result)
            if divergence_strength > 0:
                strength += divergence_strength
                is_reversal = True

            # 2. اشباع خرید/فروش در خلاف روند
            oversold_overbought_strength = self._check_oversold_overbought_reversals(result)
            if oversold_overbought_strength > 0:
                strength += oversold_overbought_strength
                is_reversal = True

            # 3. الگوهای کندل برگشتی
            candlestick_strength = self._check_candlestick_reversals(result)
            if candlestick_strength > 0:
                strength += candlestick_strength
                is_reversal = True

            # 4. الگوهای هارمونیک
            harmonic_strength = self._check_harmonic_reversals(result)
            if harmonic_strength > 0:
                strength += harmonic_strength
                is_reversal = True

            # 5. سیگنال‌های کانال قیمت
            channel_strength = self._check_channel_reversals(result)
            if channel_strength > 0:
                strength += channel_strength
                is_reversal = True

            # 6. فیک‌اوت‌های حمایت/مقاومت
            sr_strength = self._check_sr_fakeouts(result)
            if sr_strength > 0:
                strength += sr_strength
                is_reversal = True

            # نرمالایز قدرت
            return is_reversal, min(1.0, strength)

        except Exception as e:
            logger.error(f"خطا در تشخیص شرایط برگشتی برای {timeframe}: {e}")
            return False, 0.0

    def _check_divergence_reversals(self, result: Dict[str, Any]) -> float:
        """بررسی واگرایی‌ها برای سیگنال برگشتی"""
        strength = 0.0

        # واگرایی در سیگنال‌های مومنتوم
        momentum_signals = result.get('momentum', {}).get('signals', [])

        for signal in momentum_signals:
            if 'divergence' in signal.get('type', ''):
                signal_strength = signal.get('strength', 0.5)
                if 'bullish' in signal.get('type', ''):
                    strength += 0.7 * signal_strength
                elif 'bearish' in signal.get('type', ''):
                    strength += 0.7 * signal_strength

        # واگرایی در MACD
        macd_signals = result.get('macd', {}).get('signals', [])

        for signal in macd_signals:
            if 'divergence' in signal.get('type', ''):
                strength += 0.6

        return strength

    def _check_oversold_overbought_reversals(self, result: Dict[str, Any]) -> float:
        """بررسی اشباع خرید/فروش در خلاف روند"""
        strength = 0.0

        momentum_data = result.get('momentum', {})
        trend_data = result.get('trend', {})

        rsi_condition = momentum_data.get('details', {}).get('rsi_condition', 'neutral')
        trend = trend_data.get('trend', 'neutral')

        # اشباع فروش در روند نزولی = احتمال برگشت صعودی
        if rsi_condition == 'oversold' and 'bearish' in trend:
            strength += 0.5

        # اشباع خرید در روند صعودی = احتمال برگشت نزولی
        elif rsi_condition == 'overbought' and 'bullish' in trend:
            strength += 0.5

        # بررسی MFI و Stochastic
        mfi_condition = momentum_data.get('details', {}).get('mfi_condition', 'neutral')
        stoch_condition = momentum_data.get('details', {}).get('stoch_condition', 'neutral')

        if mfi_condition == 'oversold' or stoch_condition == 'oversold':
            strength += 0.3
        elif mfi_condition == 'overbought' or stoch_condition == 'overbought':
            strength += 0.3

        return strength

    def _check_candlestick_reversals(self, result: Dict[str, Any]) -> float:
        """بررسی الگوهای کندل برگشتی"""
        strength = 0.0

        reversal_patterns = [
            'hammer', 'inverted_hammer', 'morning_star', 'evening_star',
            'bullish_engulfing', 'bearish_engulfing', 'dragonfly_doji',
            'gravestone_doji', 'piercing', 'dark_cloud_cover'
        ]

        pa_signals = result.get('price_action', {}).get('signals', [])

        for signal in pa_signals:
            signal_type = signal.get('type', '')
            if any(pattern in signal_type for pattern in reversal_patterns):
                pattern_score = signal.get('score', 0)
                strength += pattern_score / 3.0  # نرمالایز

        return strength

    def _check_harmonic_reversals(self, result: Dict[str, Any]) -> float:
        """بررسی الگوهای هارمونیک برگشتی"""
        strength = 0.0

        harmonic_patterns = result.get('harmonic_patterns', [])

        for pattern in harmonic_patterns:
            pattern_type = pattern.get('type', '')

            # الگوهای Butterfly و Crab قوی‌ترین سیگنال‌های برگشتی
            if 'butterfly' in pattern_type or 'crab' in pattern_type:
                pattern_confidence = pattern.get('confidence', 0.7)
                strength += 0.8 * pattern_confidence

            # سایر الگوها
            elif 'gartley' in pattern_type or 'bat' in pattern_type:
                pattern_confidence = pattern.get('confidence', 0.7)
                strength += 0.6 * pattern_confidence

        return strength

    def _check_channel_reversals(self, result: Dict[str, Any]) -> float:
        """بررسی سیگنال‌های کانال برای برگشت"""
        strength = 0.0

        channel_data = result.get('price_channels', {})
        channel_signal = channel_data.get('signal', {})

        if channel_signal:
            signal_type = channel_signal.get('type', '')

            # برگشت از کف/سقف کانال
            if signal_type == 'channel_bounce':
                signal_score = channel_signal.get('score', 0)
                strength += signal_score / 3.0  # نرمالایز

        return strength

    def _check_sr_fakeouts(self, result: Dict[str, Any]) -> float:
        """بررسی فیک‌اوت‌های حمایت/مقاومت"""
        strength = 0.0

        sr_details = result.get('support_resistance', {}).get('details', {})
        pa_details = result.get('price_action', {}).get('details', {})

        current_close = pa_details.get('close')

        if not current_close:
            return strength

        # بررسی شکست و بازگشت مقاومت
        broken_resistance = sr_details.get('broken_resistance')
        if broken_resistance:
            resist_price = (
                broken_resistance.get('price')
                if isinstance(broken_resistance, dict)
                else broken_resistance
            )

            if resist_price and abs(current_close - resist_price) / resist_price < 0.01:
                strength += 0.6

        # بررسی شکست و بازگشت حمایت
        broken_support = sr_details.get('broken_support')
        if broken_support:
            support_price = (
                broken_support.get('price')
                if isinstance(broken_support, dict)
                else broken_support
            )

            if support_price and abs(current_close - support_price) / support_price < 0.01:
                strength += 0.6

        return strength

    async def analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل پرایس اکشن - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'direction': 'neutral',
            'signals': [],
            'details': {},
            'atr': None
        }

        if df is None or len(df) < 20:
            results['status'] = 'insufficient_data'
            return results

        try:
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')

            # محاسبه ATR
            atr = self._calculate_atr(df, symbol, timeframe)
            results['atr'] = atr

            # تحلیل باندهای بولینگر
            bb_analysis = self._analyze_bollinger_bands(df, symbol, timeframe)
            results['details']['bollinger_bands'] = bb_analysis['details']
            results['signals'].extend(bb_analysis['signals'])

            # تشخیص الگوهای کندل
            candle_patterns = await self.detect_candlestick_patterns(df)
            results['details']['candle_patterns'] = candle_patterns

            # تحلیل حجم پیشرفته
            if 'volume' in df.columns:
                volume_analysis = self._analyze_advanced_volume(df)
                results['details']['volume_analysis'] = volume_analysis['details']
                results['signals'].extend(volume_analysis['signals'])

            # ترکیب سیگنال‌ها
            all_signals = results['signals'] + candle_patterns

            # محاسبه امتیازات
            bullish_score = sum(
                s.get('score', 0) for s in all_signals
                if s.get('direction') == 'bullish'
            )
            bearish_score = sum(
                s.get('score', 0) for s in all_signals
                if s.get('direction') == 'bearish'
            )

            results['signals'] = all_signals

            if bullish_score > bearish_score:
                results['direction'] = 'bullish'
            elif bearish_score > bullish_score:
                results['direction'] = 'bearish'

            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)

            # اطلاعات اضافی
            results['details']['close'] = float(df['close'].iloc[-1])
            results['details']['open'] = float(df['open'].iloc[-1])
            results['details']['high'] = float(df['high'].iloc[-1])
            results['details']['low'] = float(df['low'].iloc[-1])

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل پرایس اکشن: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _calculate_atr(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[float]:
        """محاسبه ATR با کش"""
        try:
            high_p = df['high'].values.astype(np.float64)
            low_p = df['low'].values.astype(np.float64)
            close_p = df['close'].values.astype(np.float64)

            atr_key = self._cache_key(symbol, timeframe, 'ATR', (14,))
            atr = self._get_cached_indicator(atr_key)

            if atr is None:
                atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
                self._cache_indicator(atr_key, atr)

            last_atr = atr[~np.isnan(atr)][-1] if not np.all(np.isnan(atr)) else 0

            return round(last_atr, 8) if last_atr > 0 else None

        except Exception as e:
            logger.error(f"خطا در محاسبه ATR: {e}")
            return None

    def _analyze_bollinger_bands(self, df: pd.DataFrame, symbol: str,
                               timeframe: str) -> Dict[str, Any]:
        """تحلیل باندهای بولینگر"""
        results = {'signals': [], 'details': {}}

        try:
            close_p = df['close'].values.astype(np.float64)

            # محاسبه باندها
            bb_key = self._cache_key(symbol, timeframe, 'BBANDS', (20, 2))
            cached_bb = self._get_cached_indicator(bb_key)

            if cached_bb is not None:
                upper, middle, lower = cached_bb
            else:
                upper, middle, lower = talib.BBANDS(
                    close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
                self._cache_indicator(bb_key, (upper, middle, lower))

            # تحلیل موقعیت قیمت
            current_close = df['close'].iloc[-1]
            current_upper = upper[-1]
            current_lower = lower[-1]
            current_middle = middle[-1]

            if not np.isnan(current_upper) and not np.isnan(current_lower):
                # موقعیت در باند
                bb_position = (
                    (current_close - current_lower) / (current_upper - current_lower)
                    if current_upper > current_lower else 0.5
                )

                # عرض باند
                bb_width = (
                    (current_upper - current_lower) / current_middle
                    if current_middle > 0 else 0
                )

                # تشخیص فشردگی باند
                bb_squeeze = False
                if len(df) >= 40:
                    prev_widths = []
                    for i in range(-20, -1):
                        if not np.isnan(upper[i]) and not np.isnan(lower[i]) and middle[i] > 0:
                            width = (upper[i] - lower[i]) / middle[i]
                            prev_widths.append(width)

                    if prev_widths:
                        avg_width = np.mean(prev_widths)
                        bb_squeeze = bb_width < avg_width * 0.8

                results['details'] = {
                    'upper': float(current_upper),
                    'middle': float(current_middle),
                    'lower': float(current_lower),
                    'position': float(bb_position),
                    'width': float(bb_width),
                    'squeeze': bb_squeeze
                }

                # تولید سیگنال‌ها
                if bb_squeeze:
                    results['signals'].append({
                        'type': 'bollinger_squeeze',
                        'direction': 'neutral',
                        'score': self.pattern_scores.get('bollinger_squeeze', 2.0)
                    })

                if current_close > current_upper:
                    results['signals'].append({
                        'type': 'bollinger_upper_break',
                        'direction': 'bullish',
                        'score': self.pattern_scores.get('bollinger_upper_break', 2.5)
                    })
                elif current_close < current_lower:
                    results['signals'].append({
                        'type': 'bollinger_lower_break',
                        'direction': 'bearish',
                        'score': self.pattern_scores.get('bollinger_lower_break', 2.5)
                    })

                # بازگشت از باندها
                if len(df) >= 2:
                    prev_close = df['close'].iloc[-2]

                    if prev_close > current_upper and current_close <= current_upper:
                        results['signals'].append({
                            'type': 'bollinger_upper_rejection',
                            'direction': 'bearish',
                            'score': self.pattern_scores.get('bollinger_upper_rejection', 2.0)
                        })
                    elif prev_close < current_lower and current_close >= current_lower:
                        results['signals'].append({
                            'type': 'bollinger_lower_bounce',
                            'direction': 'bullish',
                            'score': self.pattern_scores.get('bollinger_lower_bounce', 2.0)
                        })

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل باندهای بولینگر: {e}")
            return results

    def _analyze_advanced_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل پیشرفته حجم"""
        results = {'signals': [], 'details': {}}

        if 'volume' not in df.columns or len(df) < 30:
            return results

        try:
            volume = df['volume'].values
            close = df['close'].values

            # میانگین حجم
            avg_volume = np.mean(volume[-30:-1])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # تحلیل OBV (On Balance Volume)
            obv = talib.OBV(close, volume)
            obv_slope = (obv[-1] - obv[-10]) / 10 if len(obv) >= 10 else 0

            # تحلیل AD (Accumulation/Distribution)
            high = df['high'].values
            low = df['low'].values
            ad = talib.AD(high, low, close, volume)
            ad_slope = (ad[-1] - ad[-10]) / 10 if len(ad) >= 10 else 0

            results['details'] = {
                'current_volume': float(current_volume),
                'avg_volume': float(avg_volume),
                'volume_ratio': float(volume_ratio),
                'is_high_volume': volume_ratio > 1.5,
                'is_very_high_volume': volume_ratio > 2.5,
                'is_low_volume': volume_ratio < 0.5,
                'obv_trend': 'up' if obv_slope > 0 else 'down' if obv_slope < 0 else 'flat',
                'ad_trend': 'up' if ad_slope > 0 else 'down' if ad_slope < 0 else 'flat'
            }

            # سیگنال‌های حجم
            if volume_ratio > 2.5:
                candle_direction = 'bullish' if close[-1] > df['open'].iloc[-1] else 'bearish'

                results['signals'].append({
                    'type': f'high_volume_{candle_direction}',
                    'direction': candle_direction,
                    'score': self.pattern_scores.get(f'high_volume_{candle_direction}', 2.8)
                })

            # واگرایی حجم
            price_trend = 'up' if close[-1] > close[-10] else 'down'

            if price_trend == 'up' and obv_slope < 0:
                results['signals'].append({
                    'type': 'volume_price_divergence',
                    'direction': 'bearish',
                    'score': self.pattern_scores.get('volume_price_divergence', 2.5)
                })
            elif price_trend == 'down' and obv_slope > 0:
                results['signals'].append({
                    'type': 'volume_price_divergence',
                    'direction': 'bullish',
                    'score': self.pattern_scores.get('volume_price_divergence', 2.5)
                })

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل پیشرفته حجم: {e}")
            return results

    def calculate_risk_reward(self, symbol: str, direction: str, current_price: float,
                            analysis_results: Dict[str, Dict[str, Any]],
                            adapted_risk_config: Dict[str, Any]) -> Dict[str, float]:
        """محاسبه سطوح توقف ضرر و حد سود - نسخه بهینه‌شده"""
        # استفاده از یادگیری تطبیقی برای SL
        if self.adaptive_learning.enabled:
            base_sl_percent = adapted_risk_config.get('default_stop_loss_percent', self.base_default_sl_percent)
            tf = self._determine_primary_timeframe(analysis_results)

            if tf:
                adapted_sl_percent = self.adaptive_learning.get_adaptive_sl_percent(
                    base_sl_percent, symbol, tf, direction
                )
                adapted_risk_config['default_stop_loss_percent'] = adapted_sl_percent

        # پارامترهای ریسک
        default_sl_percent = adapted_risk_config.get('default_stop_loss_percent', self.base_default_sl_percent)
        preferred_rr = adapted_risk_config.get('preferred_risk_reward_ratio', self.base_preferred_risk_reward_ratio)
        min_rr = adapted_risk_config.get('min_risk_reward_ratio', self.base_min_risk_reward_ratio)

        try:
            # یافتن تایم‌فریم‌های معتبر
            valid_tfs = [tf for tf, res in analysis_results.items() if res.get('status') == 'ok']
            if not valid_tfs:
                raise ValueError("داده تحلیل معتبری برای محاسبه RR یافت نشد")

            # انتخاب بالاترین تایم‌فریم معتبر
            highest_tf = max(valid_tfs, key=lambda x: self.timeframe_weights.get(x, 0))

            # دریافت نتایج تحلیل
            pa_result = analysis_results[highest_tf].get('price_action', {})
            sr_result = analysis_results[highest_tf].get('support_resistance', {}).get('details', {})
            channel_result = analysis_results[highest_tf].get('price_channels', {})
            harmonic_patterns = analysis_results[highest_tf].get('harmonic_patterns', [])

            stop_loss = None
            take_profit = None
            calculation_method = "None"

            # دریافت ATR
            atr = pa_result.get('atr')
            atr = atr if atr and atr > 0 else current_price * 0.005

            # 1. اولویت اول: الگوهای هارمونیک
            if stop_loss is None and harmonic_patterns and direction in ['long', 'short']:
                harmonic_levels = self._calculate_harmonic_levels(
                    harmonic_patterns, direction, current_price
                )
                if harmonic_levels:
                    stop_loss = harmonic_levels['stop_loss']
                    take_profit = harmonic_levels['take_profit']
                    calculation_method = harmonic_levels['method']

            # 2. اولویت دوم: کانال‌های قیمت
            if stop_loss is None and channel_result.get('active_channel'):
                channel_levels = self._calculate_channel_levels(
                    channel_result['active_channel'], direction, current_price
                )
                if channel_levels:
                    stop_loss = channel_levels['stop_loss']
                    take_profit = channel_levels['take_profit']
                    calculation_method = channel_levels['method']

            # 3. اولویت سوم: سطوح حمایت/مقاومت
            if stop_loss is None:
                sr_levels = self._calculate_sr_levels(
                    sr_result, direction, current_price, atr
                )
                if sr_levels:
                    stop_loss = sr_levels['stop_loss']
                    calculation_method = sr_levels['method']

            # 4. استفاده از ATR اگر سطوح دور هستند
            if stop_loss is not None and atr > 0:
                sl_distance = abs(current_price - stop_loss)
                if sl_distance > atr * 3.0:
                    stop_loss = None  # سطح خیلی دور است

            # 5. محاسبه SL براساس ATR
            if stop_loss is None and atr > 0:
                sl_multiplier = adapted_risk_config.get('atr_trailing_multiplier', 2.0)
                stop_loss = self._calculate_atr_stop_loss(
                    current_price, direction, atr, sl_multiplier
                )
                calculation_method = f"ATR x{sl_multiplier}"

            # 6. SL درصدی به عنوان آخرین راه حل
            if stop_loss is None:
                stop_loss = self._calculate_percentage_stop_loss(
                    current_price, direction, default_sl_percent
                )
                calculation_method = f"Percentage {default_sl_percent}%"

            # بررسی حداقل فاصله SL
            stop_loss = self._ensure_minimum_sl_distance(
                stop_loss, current_price, direction, atr
            )

            # محاسبه فاصله ریسک نهایی
            risk_distance = abs(current_price - stop_loss)

            # محاسبه TP
            if take_profit is None:
                take_profit = self._calculate_take_profit(
                    current_price, direction, risk_distance,
                    preferred_rr, min_rr, sr_result
                )

            # محاسبه RR نهایی
            final_reward_distance = abs(take_profit - current_price)
            final_rr = final_reward_distance / risk_distance if risk_distance > 0 else 0

            # اطمینان از صحت مقادیر
            precision = 8

            return {
                'stop_loss': round(stop_loss, precision),
                'take_profit': round(take_profit, precision),
                'risk_reward_ratio': round(final_rr, 2),
                'risk_amount_per_unit': round(risk_distance, precision),
                'sl_method': calculation_method
            }

        except Exception as e:
            logger.error(f"خطا در محاسبه ریسک/ریوارد برای {symbol}: {e}", exc_info=True)

            # مقادیر پیش‌فرض در صورت خطا
            if direction == 'long':
                stop_loss = current_price * (1 - default_sl_percent / 100)
                take_profit = current_price * (1 + (default_sl_percent * min_rr) / 100)
            else:
                stop_loss = current_price * (1 + default_sl_percent / 100)
                take_profit = current_price * (1 - (default_sl_percent * min_rr) / 100)

            precision = 8
            return {
                'stop_loss': round(stop_loss, precision),
                'take_profit': round(take_profit, precision),
                'risk_reward_ratio': min_rr,
                'risk_amount_per_unit': round(current_price * (default_sl_percent / 100), precision),
                'sl_method': 'Error Fallback'
            }

    def _calculate_harmonic_levels(self, harmonic_patterns: List[Dict[str, Any]],
                                 direction: str, current_price: float) -> Optional[Dict[str, Any]]:
        """محاسبه سطوح SL/TP براساس الگوهای هارمونیک"""
        if not harmonic_patterns:
            return None

        # انتخاب بهترین الگو
        best_pattern = max(harmonic_patterns, key=lambda x: x.get('confidence', 0))
        pattern_type = best_pattern.get('type', '')
        pattern_direction = best_pattern.get('direction', '')

        # بررسی جهت
        if (direction == 'long' and pattern_direction != 'bullish') or \
           (direction == 'short' and pattern_direction != 'bearish'):
            return None

        # استفاده از منطقه PRZ
        prz = best_pattern.get('completion_zone', {})
        if not prz:
            return None

        if direction == 'long':
            stop_loss = prz.get('min', 0) * 0.99

            # هدف براساس نوع الگو
            if 'butterfly' in pattern_type or 'crab' in pattern_type:
                take_profit = current_price + (current_price - stop_loss) * 1.618
            else:
                # هدف به نقطه X
                x_price = best_pattern.get('points', {}).get('X', {}).get('price', 0)
                take_profit = x_price if x_price > current_price else current_price * 1.05
        else:  # short
            stop_loss = prz.get('max', 0) * 1.01

            if 'butterfly' in pattern_type or 'crab' in pattern_type:
                take_profit = current_price - (stop_loss - current_price) * 1.618
            else:
                x_price = best_pattern.get('points', {}).get('X', {}).get('price', 0)
                take_profit = x_price if x_price < current_price else current_price * 0.95

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'method': f"Harmonic_{pattern_type}"
        }

    def _calculate_channel_levels(self, channel: Dict[str, Any], direction: str,
                                current_price: float) -> Optional[Dict[str, Any]]:
        """محاسبه سطوح SL/TP براساس کانال قیمت"""
        upper_slope = channel.get('upper_slope', 0)
        upper_intercept = channel.get('upper_intercept', 0)
        lower_slope = channel.get('lower_slope', 0)
        lower_intercept = channel.get('lower_intercept', 0)

        # محاسبه سطوح فعلی کانال
        # فرض می‌کنیم ایندکس فعلی = طول داده‌ها - 1
        current_idx = 100  # این باید از داده واقعی گرفته شود

        upper_current = upper_slope * current_idx + upper_intercept
        lower_current = lower_slope * current_idx + lower_intercept

        if direction == 'long':
            stop_loss = lower_current * 0.99
            take_profit = upper_current * 0.99
        else:  # short
            stop_loss = upper_current * 1.01
            take_profit = lower_current * 1.01

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'method': f"Channel_{channel.get('direction', 'unknown')}"
        }

    def _calculate_sr_levels(self, sr_details: Dict[str, Any], direction: str,
                           current_price: float, atr: float) -> Optional[Dict[str, Any]]:
        """محاسبه سطح SL براساس حمایت/مقاومت"""
        if direction == 'long':
            nearest_support = sr_details.get('nearest_support')
            if nearest_support:
                support_price = (
                    nearest_support.get('price')
                    if isinstance(nearest_support, dict)
                    else nearest_support
                )

                if support_price and support_price < current_price:
                    return {
                        'stop_loss': support_price * 0.999,
                        'method': 'Support Level'
                    }
        else:  # short
            nearest_resistance = sr_details.get('nearest_resistance')
            if nearest_resistance:
                resist_price = (
                    nearest_resistance.get('price')
                    if isinstance(nearest_resistance, dict)
                    else nearest_resistance
                )

                if resist_price and resist_price > current_price:
                    return {
                        'stop_loss': resist_price * 1.001,
                        'method': 'Resistance Level'
                    }

        return None

    def _calculate_atr_stop_loss(self, current_price: float, direction: str,
                               atr: float, multiplier: float) -> float:
        """محاسبه SL براساس ATR"""
        if direction == 'long':
            return current_price - (atr * multiplier)
        else:  # short
            return current_price + (atr * multiplier)

    def _calculate_percentage_stop_loss(self, current_price: float, direction: str,
                                      sl_percent: float) -> float:
        """محاسبه SL درصدی"""
        if direction == 'long':
            return current_price * (1 - sl_percent / 100)
        else:  # short
            return current_price * (1 + sl_percent / 100)

    def _ensure_minimum_sl_distance(self, stop_loss: float, current_price: float,
                                  direction: str, atr: float) -> float:
        """اطمینان از حداقل فاصله SL"""
        min_distance = atr * 0.5 if atr > 0 else current_price * 0.001

        if direction == 'long':
            if (current_price - stop_loss) < min_distance:
                return current_price - min_distance
        else:  # short
            if (stop_loss - current_price) < min_distance:
                return current_price + min_distance

        return stop_loss

    def _calculate_take_profit(self, current_price: float, direction: str,
                             risk_distance: float, preferred_rr: float,
                             min_rr: float, sr_details: Dict[str, Any]) -> float:
        """محاسبه حد سود"""
        # محاسبه TP پایه
        reward_distance = risk_distance * preferred_rr
        reward_distance = max(reward_distance, current_price * 0.001)

        if direction == 'long':
            take_profit = current_price + reward_distance

            # تنظیم براساس مقاومت نزدیک
            nearest_resistance = sr_details.get('nearest_resistance')
            if nearest_resistance:
                resist_price = (
                    nearest_resistance.get('price')
                    if isinstance(nearest_resistance, dict)
                    else nearest_resistance
                )

                if resist_price and current_price < resist_price < take_profit:
                    if resist_price > current_price + (risk_distance * min_rr):
                        take_profit = resist_price * 0.999

        else:  # short
            take_profit = current_price - reward_distance

            # تنظیم براساس حمایت نزدیک
            nearest_support = sr_details.get('nearest_support')
            if nearest_support:
                support_price = (
                    nearest_support.get('price')
                    if isinstance(nearest_support, dict)
                    else nearest_support
                )

                if support_price and take_profit < support_price < current_price:
                    if support_price < current_price - (risk_distance * min_rr):
                        take_profit = support_price * 1.001

        return take_profit

    def _determine_primary_timeframe(self, analysis_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """تعیین تایم‌فریم اصلی از نتایج تحلیل"""
        valid_tfs = [tf for tf, res in analysis_results.items() if res.get('status') == 'ok']

        if not valid_tfs:
            return None

        # مرتب‌سازی براساس دقیقه (کوچکترین تایم‌فریم)
        sorted_tfs = sorted(valid_tfs, key=self._get_tf_minutes)

        return sorted_tfs[0] if sorted_tfs else None

    def _get_tf_minutes(self, timeframe: str) -> int:
        """تبدیل تایم‌فریم به دقیقه"""
        try:
            if 'm' in timeframe:
                return int(timeframe.replace('m', ''))
            elif 'h' in timeframe:
                return int(timeframe.replace('h', '')) * 60
            elif 'd' in timeframe:
                return int(timeframe.replace('d', '')) * 1440
            elif 'w' in timeframe:
                return int(timeframe.replace('w', '')) * 10080
            else:
                return 99999
        except ValueError:
            return 99999

    def analyze_higher_timeframe_structure(self, df: pd.DataFrame, tf: str,
                                         analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """تحلیل ساختار تایم‌فریم بالاتر - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'score': 1.0,
            'details': {},
            'signals': []
        }

        if not self.htf_enabled or df is None or tf is None:
            results['status'] = 'disabled_or_missing_data'
            return results

        try:
            # تعیین تایم‌فریم بالاتر
            htf = self._determine_higher_timeframe(tf, analysis_results)

            if not htf:
                results['status'] = 'no_higher_timeframe'
                return results

            htf_result = analysis_results.get(htf)
            if not htf_result or htf_result.get('status') != 'ok':
                results['status'] = 'missing_htf_analysis'
                return results

            # استخراج داده‌های تحلیل
            analysis_data = self._extract_htf_analysis_data(
                analysis_results[tf], htf_result, df
            )

            # محاسبه امتیاز ساختار
            structure_score = self._calculate_structure_score(analysis_data)

            # تولید سیگنال‌های HTF
            htf_signals = self._generate_htf_signals(analysis_data)

            # نتایج نهایی
            results['score'] = round(structure_score, 2)
            results['htf'] = htf
            results['aligned'] = analysis_data['trends_aligned']
            results['signals'] = htf_signals
            results['details'] = {
                'trends_aligned': analysis_data['trends_aligned'],
                'momentum_aligned': analysis_data['momentum_aligned'],
                'tf_trend': analysis_data['current_trend_dir'],
                'htf_trend': analysis_data['htf_trend_dir'],
                'tf_momentum': analysis_data['current_momentum_dir'],
                'htf_momentum': analysis_data['htf_momentum_dir'],
                'tf_phase': analysis_data['current_trend_phase'],
                'htf_phase': analysis_data['htf_trend_phase'],
                'at_htf_support': analysis_data['at_support_zone'],
                'at_htf_resistance': analysis_data['at_resistance_zone'],
                'htf_trend_strength': analysis_data['htf_trend_strength']
            }

            # اضافه کردن اطلاعات مناطق HTF
            if analysis_data['nearest_htf_support']:
                results['details']['nearest_htf_support'] = analysis_data['nearest_htf_support']
            if analysis_data['nearest_htf_resistance']:
                results['details']['nearest_htf_resistance'] = analysis_data['nearest_htf_resistance']

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل ساختار تایم‌فریم بالاتر: {e}", exc_info=True)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _determine_higher_timeframe(self, current_tf: str,
                                  analysis_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """تعیین تایم‌فریم بالاتر مناسب"""
        htf = None

        if self.htf_timeframe_method == 'next_higher':
            current_minutes = self._get_tf_minutes(current_tf)
            valid_tfs = [
                t for t in self.timeframes
                if self._get_tf_minutes(t) > current_minutes and t in analysis_results
            ]
            if valid_tfs:
                htf = min(valid_tfs, key=self._get_tf_minutes)

        elif self.htf_timeframe_method == 'fixed':
            if current_tf != self.htf_fixed_tf1 and self.htf_fixed_tf1 in analysis_results:
                htf = self.htf_fixed_tf1
            elif current_tf != self.htf_fixed_tf2 and self.htf_fixed_tf2 in analysis_results:
                htf = self.htf_fixed_tf2

        # اگر HTF پیدا نشد، بالاترین تایم‌فریم موجود
        if not htf:
            htf = max(
                self.timeframes,
                key=lambda x: self.timeframe_weights.get(x, 0)
            )
            if htf == current_tf or htf not in analysis_results:
                return None

        return htf

    def _extract_htf_analysis_data(self, current_result: Dict[str, Any],
                                 htf_result: Dict[str, Any],
                                 df: pd.DataFrame) -> Dict[str, Any]:
        """استخراج داده‌های تحلیل HTF"""
        current_price = df['close'].iloc[-1]

        # داده‌های روند
        current_trend = current_result.get('trend', {})
        htf_trend = htf_result.get('trend', {})
        current_trend_dir = current_trend.get('trend', 'neutral')
        htf_trend_dir = htf_trend.get('trend', 'neutral')

        # داده‌های مومنتوم
        current_momentum = current_result.get('momentum', {})
        htf_momentum = htf_result.get('momentum', {})

        # بررسی همراستایی
        trends_aligned = self._check_trend_alignment(current_trend_dir, htf_trend_dir)
        momentum_aligned = current_momentum.get('direction') == htf_momentum.get('direction')

        # مناطق حمایت/مقاومت HTF
        htf_sr = htf_result.get('support_resistance', {})
        htf_zones_data = self._analyze_htf_zones(htf_sr, current_price)

        return {
            'current_trend_dir': current_trend_dir,
            'htf_trend_dir': htf_trend_dir,
            'current_trend_phase': current_trend.get('phase', 'unknown'),
            'htf_trend_phase': htf_trend.get('phase', 'unknown'),
            'current_trend_strength': abs(current_trend.get('strength', 0)),
            'htf_trend_strength': abs(htf_trend.get('strength', 0)),
            'current_momentum_dir': current_momentum.get('direction', 'neutral'),
            'htf_momentum_dir': htf_momentum.get('direction', 'neutral'),
            'trends_aligned': trends_aligned,
            'momentum_aligned': momentum_aligned,
            'current_price': current_price,
            **htf_zones_data
        }

    def _check_trend_alignment(self, current_trend: str, htf_trend: str) -> bool:
        """بررسی همراستایی روندها"""
        return (
            ('bullish' in current_trend and 'bullish' in htf_trend) or
            ('bearish' in current_trend and 'bearish' in htf_trend)
        )

    def _analyze_htf_zones(self, htf_sr: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """تحلیل مناطق حمایت/مقاومت HTF"""
        result = {
            'nearest_htf_resistance': None,
            'nearest_htf_support': None,
            'at_resistance_zone': False,
            'at_support_zone': False,
            'price_above_support': False,
            'price_below_resistance': False
        }

        # مناطق مقاومت
        resistance_zones = htf_sr.get('resistance_zones', {}).get('zones', [])
        for zone in resistance_zones:
            if result['nearest_htf_resistance'] is None:
                result['nearest_htf_resistance'] = zone

            # بررسی قرارگیری در منطقه
            if abs(current_price - zone.get('center', 0)) <= zone.get('width', 0) / 2:
                result['at_resistance_zone'] = True
                break

        # مناطق حمایت
        support_zones = htf_sr.get('support_zones', {}).get('zones', [])
        for zone in support_zones:
            if result['nearest_htf_support'] is None:
                result['nearest_htf_support'] = zone

            # بررسی قرارگیری در منطقه
            if abs(current_price - zone.get('center', 0)) <= zone.get('width', 0) / 2:
                result['at_support_zone'] = True
                break

        # موقعیت نسبی
        if result['nearest_htf_support']:
            result['price_above_support'] = current_price > result['nearest_htf_support'].get('center', 0)

        if result['nearest_htf_resistance']:
            result['price_below_resistance'] = current_price < result['nearest_htf_resistance'].get('center', 0)

        return result

    def _calculate_structure_score(self, analysis_data: Dict[str, Any]) -> float:
        """محاسبه امتیاز ساختار HTF"""
        base_score = self.htf_score_config['base']
        score = base_score

        # تنظیم براساس همراستایی روند
        if analysis_data['trends_aligned']:
            score += self.htf_score_config['confirm_bonus']

            # جایزه اضافی براساس قدرت روند
            trend_strength_factor = min(
                analysis_data['current_trend_strength'],
                analysis_data['htf_trend_strength']
            ) / 3
            score *= (1 + self.htf_score_config['trend_bonus_mult'] * trend_strength_factor)
        else:
            score -= self.htf_score_config['contradict_penalty']

            # جریمه اضافی براساس قدرت تضاد
            trend_strength_factor = min(
                analysis_data['current_trend_strength'],
                analysis_data['htf_trend_strength']
            ) / 3
            score *= (1 - self.htf_score_config['trend_penalty_mult'] * trend_strength_factor)

        # تنظیم براساس همراستایی مومنتوم
        if analysis_data['momentum_aligned']:
            score *= 1.05
        else:
            score *= 0.95

        # تنظیم براساس موقعیت نسبت به مناطق HTF
        current_trend = analysis_data['current_trend_dir']

        if 'bullish' in current_trend:
            if analysis_data['price_above_support'] and analysis_data['price_below_resistance']:
                score *= 1.1
            if analysis_data['at_support_zone']:
                score *= 1.2
        elif 'bearish' in current_trend:
            if analysis_data['price_below_resistance'] and analysis_data['price_above_support']:
                score *= 1.1
            if analysis_data['at_resistance_zone']:
                score *= 1.2

        # محدود کردن امتیاز
        score = max(
            self.htf_score_config['min_score'],
            min(self.htf_score_config['max_score'], score)
        )

        return score

    def _generate_htf_signals(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """تولید سیگنال‌های HTF"""
        signals = []

        # سیگنال همراستایی قوی
        if (analysis_data['trends_aligned'] and
            analysis_data['momentum_aligned'] and
            analysis_data['htf_trend_strength'] >= 2):

            direction = 'bullish' if 'bullish' in analysis_data['htf_trend_dir'] else 'bearish'
            signals.append({
                'type': 'htf_strong_alignment',
                'direction': direction,
                'score': 3.5,
                'details': {
                    'htf_trend_strength': analysis_data['htf_trend_strength']
                }
            })

        # سیگنال برگشت از مناطق HTF
        if analysis_data['at_support_zone'] and 'bullish' in analysis_data['current_trend_dir']:
            signals.append({
                'type': 'htf_support_bounce',
                'direction': 'bullish',
                'score': 3.0
            })
        elif analysis_data['at_resistance_zone'] and 'bearish' in analysis_data['current_trend_dir']:
            signals.append({
                'type': 'htf_resistance_rejection',
                'direction': 'bearish',
                'score': 3.0
            })

        return signals

    def analyze_volatility_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل شرایط نوسان - نسخه بهینه‌شده"""
        results = {
            'status': 'ok',
            'score': 1.0,
            'condition': 'normal',
            'reject': False,
            'volatility_ratio': 1.0,
            'details': {},
            'signals': []
        }

        required_len = max(self.vol_atr_period, self.vol_atr_ma_period) + 10

        if not self.vol_enabled or df is None or len(df) < required_len:
            results['status'] = 'disabled_or_insufficient_data'
            return results

        try:
            # محاسبه ATR
            high_p = df['high'].values.astype(np.float64)
            low_p = df['low'].values.astype(np.float64)
            close_p = df['close'].values.astype(np.float64)

            atr = talib.ATR(high_p, low_p, close_p, timeperiod=self.vol_atr_period)

            # فیلتر مقادیر معتبر
            valid_atr = atr[~np.isnan(atr)]
            if len(valid_atr) < self.vol_atr_ma_period:
                results['status'] = 'insufficient_valid_data'
                return results

            # محاسبه ATR نرمالایز شده
            valid_close_p = close_p[-len(valid_atr):]
            atr_pct = (valid_atr / valid_close_p) * 100

            # میانگین متحرک ATR%
            if use_bottleneck:
                atr_pct_ma = bn.move_mean(atr_pct, window=self.vol_atr_ma_period, min_count=1)
            else:
                atr_pct_ma = pd.Series(atr_pct).rolling(
                    window=self.vol_atr_ma_period, min_periods=1
                ).mean().values

            # مقادیر فعلی
            current_atr_pct = atr_pct[-1]
            current_atr_pct_ma = atr_pct_ma[-1]

            # نسبت نوسان
            volatility_ratio = current_atr_pct / current_atr_pct_ma if current_atr_pct_ma > 0 else 1.0

            # تعیین شرایط نوسان
            vol_condition, vol_score = self._determine_volatility_condition(volatility_ratio)

            # بررسی رد سیگنال
            reject_due_to_extreme = (
                vol_condition == 'extreme' and self.vol_reject_extreme
            )

            # تحلیل‌های اضافی
            volatility_trend = self._analyze_volatility_trend(atr_pct_ma)
            volatility_breakout = self._detect_volatility_breakout(
                atr_pct, atr_pct_ma, volatility_ratio
            )

            # تولید سیگنال‌ها
            if volatility_breakout:
                results['signals'].append({
                    'type': 'volatility_breakout',
                    'direction': 'neutral',
                    'score': 2.0,
                    'details': {
                        'breakout_type': volatility_breakout
                    }
                })

            # نتایج نهایی
            results.update({
                'score': vol_score,
                'condition': vol_condition,
                'reject': reject_due_to_extreme,
                'volatility_ratio': round(volatility_ratio, 2),
                'details': {
                    'current_atr_pct': round(current_atr_pct, 3),
                    'average_atr_pct': round(current_atr_pct_ma, 3),
                    'raw_atr': round(valid_atr[-1], 5),
                    'volatility_trend': volatility_trend,
                    'historical_percentile': self._calculate_volatility_percentile(
                        current_atr_pct, atr_pct
                    )
                }
            })

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل شرایط نوسان: {e}", exc_info=True)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _determine_volatility_condition(self, ratio: float) -> Tuple[str, float]:
        """تعیین شرایط نوسان"""
        if ratio > self.vol_extreme_thresh:
            return 'extreme', self.vol_scores.get('extreme', 0.5)
        elif ratio > self.vol_high_thresh:
            return 'high', self.vol_scores.get('high', 0.8)
        elif ratio < self.vol_low_thresh:
            return 'low', self.vol_scores.get('low', 0.9)
        else:
            return 'normal', self.vol_scores.get('normal', 1.0)

    def _analyze_volatility_trend(self, atr_pct_ma: np.ndarray) -> str:
        """تحلیل روند نوسان"""
        if len(atr_pct_ma) < 10:
            return 'unknown'

        # محاسبه شیب
        recent_ma = atr_pct_ma[-10:]
        x = np.arange(len(recent_ma))
        slope, _ = np.polyfit(x, recent_ma, 1)

        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def _detect_volatility_breakout(self, atr_pct: np.ndarray,
                                   atr_pct_ma: np.ndarray,
                                   current_ratio: float) -> Optional[str]:
        """تشخیص شکست نوسان"""
        if len(atr_pct) < 20:
            return None

        # بررسی شکست صعودی
        if current_ratio > 1.5:
            # بررسی اینکه آیا از محدوده قبلی خارج شده
            recent_max = np.max(atr_pct[-20:-1])
            if atr_pct[-1] > recent_max * 1.2:
                return 'upward'

        # بررسی شکست نزولی
        elif current_ratio < 0.5:
            recent_min = np.min(atr_pct[-20:-1])
            if atr_pct[-1] < recent_min * 0.8:
                return 'downward'

        return None

    def _calculate_volatility_percentile(self, current_value: float,
                                       historical_values: np.ndarray) -> float:
        """محاسبه صدک نوسان"""
        if len(historical_values) < 20:
            return 50.0

        # محاسبه صدک
        percentile = stats.percentileofscore(historical_values, current_value)

        return round(percentile, 1)

    def _analyze_basic_momentum(self, rsi: pd.Series, slowk: np.ndarray,
                              slowd: np.ndarray, dif: pd.Series,
                              dea: pd.Series) -> Dict[str, Any]:
        """تحلیل پایه مومنتوم (سازگاری با نسخه قدیمی)"""
        results = {
            'status': 'ok',
            'direction': 'neutral',
            'signals': [],
            'details': {}
        }

        required_len = 2
        if rsi.empty or len(slowk) < required_len or dif.empty or dea.empty:
            results['status'] = 'insufficient_data'
            return results

        try:
            # مقادیر فعلی و قبلی
            curr_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            curr_k = slowk[-1]
            prev_k = slowk[-2]
            curr_d = slowd[-1]
            prev_d = slowd[-2]
            curr_dif = dif.iloc[-1]
            prev_dif = dif.iloc[-2]
            curr_dea = dea.iloc[-1]
            prev_dea = dea.iloc[-2]

            momentum_signals = []

            # تقاطع MACD
            if curr_dif > curr_dea and prev_dif <= prev_dea:
                momentum_signals.append({
                    'type': 'macd_bullish_crossover',
                    'score': self.pattern_scores.get('macd_bullish_crossover', 2.2)
                })
            elif curr_dif < curr_dea and prev_dif >= prev_dea:
                momentum_signals.append({
                    'type': 'macd_bearish_crossover',
                    'score': self.pattern_scores.get('macd_bearish_crossover', 2.2)
                })

            # سایر سیگنال‌ها...
            # (کد مشابه قبلی)

            results['signals'] = momentum_signals

            # محاسبه جهت کلی
            bullish_score = sum(s['score'] for s in momentum_signals if 'bullish' in s['type'])
            bearish_score = sum(s['score'] for s in momentum_signals if 'bearish' in s['type'])

            if bullish_score > bearish_score * 1.1:
                results['direction'] = 'bullish'
            elif bearish_score > bullish_score * 1.1:
                results['direction'] = 'bearish'

            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)
            results['details'] = {
                'rsi': round(curr_rsi, 2),
                'macd': round(curr_dif, 5),
                'macd_signal': round(curr_dea, 5),
                'stoch_k': round(curr_k, 2),
                'stoch_d': round(curr_d, 2)
            }

            return results

        except Exception as e:
            logger.error(f"خطا در تحلیل مومنتوم پایه: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    async def analyze_single_timeframe(self
