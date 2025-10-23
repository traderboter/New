"""
Backtest Engine V2 - موتور اصلی اجرای Backtest با SignalOrchestrator

🆕 نسخه 2.0 - تفاوت‌ها با نسخه قدیم:
=====================================
1. ✅ استفاده از SignalOrchestrator به جای SignalGenerator
2. ✅ معماری ماژولار signal_generation (10 Analyzers)
3. ✅ Context-Based Architecture
4. ✅ IndicatorCalculator مرکزی
5. ✅ API تک تایم‌فریم به جای چند تایم‌فریم

این ماژول همه کامپوننت‌ها را به هم متصل می‌کند و شبیه‌سازی را اجرا می‌کند.

Author: Adapted from backtest_engine.py for SignalOrchestrator
Date: 2025-10-23
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import os

# Backtest components
from backtest.csv_data_loader import CSVDataLoader
from backtest.historical_data_provider_v2 import HistoricalDataProvider, BacktestMarketDataFetcher
from backtest.time_simulator import TimeSimulator
from backtest.backtest_trade_manager import BacktestTradeManager, TradeDirection

# New signal generation system
from signal_generation.orchestrator import SignalOrchestrator
from signal_generation.shared.indicator_calculator import IndicatorCalculator
from signal_generation.signal_info import SignalInfo

logger = logging.getLogger(__name__)


class BacktestEngineV2:
    """
    موتور اصلی Backtest نسخه 2 که از SignalOrchestrator استفاده می‌کند
    
    🔑 تفاوت‌های کلیدی با نسخه قدیم:
    - از SignalOrchestrator استفاده می‌کند (به جای SignalGenerator)
    - API تک تایم‌فریم: generate_signal_for_symbol(symbol, timeframe)
    - نیاز به IndicatorCalculator دارد
    - با معماری ماژولار signal_generation کار می‌کند
    """

    def __init__(self, config: Dict):
        """
        مقداردهی اولیه BacktestEngineV2

        Args:
            config: تنظیمات کامل ربات
        """
        self.config = config
        self.backtest_config = config.get('backtest', {})

        # تنظیمات بازه زمانی
        start_date_config = self.backtest_config.get('start_date')

        # تشخیص 'auto' برای start_date
        if start_date_config == 'auto':
            self.start_date = None  # بعداً خودکار تعیین می‌شود
        else:
            self.start_date = self._parse_date(start_date_config)

        end_date_str = self.backtest_config.get('end_date', 'auto')
        self.end_date = None  # بعداً تنظیم می‌شود

        # تنظیمات دیگر
        self.symbols = self.backtest_config.get('symbols', [])
        self.initial_balance = self.backtest_config.get('initial_balance', 10000.0)
        self.step_timeframe = self.backtest_config.get('step_timeframe', '5m')
        self.process_interval = self.backtest_config.get('process_interval', 180)  # ثانیه
        
        # 🆕 تایم‌فریم اصلی برای تولید سیگنال
        self.signal_timeframe = config.get('signal_processing', {}).get('primary_timeframe', '1h')

        # کامپوننت‌های اصلی
        self.historical_provider: Optional[HistoricalDataProvider] = None
        self.data_fetcher: Optional[BacktestMarketDataFetcher] = None
        self.time_simulator: Optional[TimeSimulator] = None
        self.trade_manager: Optional[BacktestTradeManager] = None
        
        # 🆕 کامپوننت‌های جدید
        self.indicator_calculator: Optional[IndicatorCalculator] = None
        self.signal_orchestrator: Optional[SignalOrchestrator] = None

        # نتایج
        self.results = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'trades': [],
            'equity_curve': [],
            'statistics': {}
        }

        # پرچم‌ها
        self.is_running = False
        self.use_progress_bar = self.backtest_config.get('use_progress_bar', True)

        logger.info(f"BacktestEngineV2 initialized for {len(self.symbols)} symbols")

    def _parse_date(self, date_str: str) -> datetime:
        """تبدیل رشته تاریخ به datetime"""
        if isinstance(date_str, datetime):
            return date_str
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

    def _get_step_minutes(self, timeframe: str) -> int:
        """تبدیل تایم‌فریم به دقیقه"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return mapping.get(timeframe, 5)

    async def initialize(self):
        """راه‌اندازی تمام کامپوننت‌ها"""
        logger.info("Initializing Backtest Engine V2...")

        # 1. ایجاد HistoricalDataProvider (داده‌ها خودکار لود می‌شوند)
        self.historical_provider = HistoricalDataProvider(self.config)
        logger.info(f"✅ Historical data loaded for symbols: {self.symbols}")

        # 2. تعیین start_date و end_date
        if self.start_date is None or self.start_date == 'auto':
            # پیدا کردن اولین تاریخ موجود از داده‌های لود شده
            latest_start = None

            for symbol in self.symbols:
                timeframes = self.config.get('data_fetching', {}).get('timeframes', ['5m', '15m', '1h', '4h'])

                for tf in timeframes:
                    # دریافت بازه زمانی داده‌ها
                    date_range = self.historical_provider.get_data_range(symbol, tf)

                    if date_range:
                        start_date, end_date = date_range

                        if latest_start is None or start_date > latest_start:
                            latest_start = start_date

            if latest_start is None:
                raise ValueError("No data available for any timeframe!")

            self.start_date = latest_start
            logger.info(f"Auto-detected start date: {self.start_date}")

        if self.end_date is None or self.end_date == 'auto':
            # پیدا کردن آخرین تاریخ موجود از داده‌های لود شده
            earliest_end = None

            for symbol in self.symbols:
                timeframes = self.config.get('data_fetching', {}).get('timeframes', ['5m', '15m', '1h', '4h'])

                for tf in timeframes:
                    # دریافت بازه زمانی داده‌ها
                    date_range = self.historical_provider.get_data_range(symbol, tf)

                    if date_range:
                        start_date, end_date = date_range

                        if earliest_end is None or end_date < earliest_end:
                            earliest_end = end_date

            if earliest_end is None:
                raise ValueError("No data available for any timeframe!")

            self.end_date = earliest_end
            logger.info(f"Auto-detected end date: {self.end_date}")

        # نمایش دوره نهایی
        logger.info(f"📅 FINAL Backtest Period:")
        logger.info(f"   Start: {self.start_date}")
        logger.info(f"   End:   {self.end_date}")
        logger.info(f"   Duration: {self.end_date - self.start_date}")

        # 3. ایجاد BacktestMarketDataFetcher
        self.data_fetcher = BacktestMarketDataFetcher(self.historical_provider)

        # 4. ایجاد TimeSimulator
        step_minutes = self._get_step_minutes(self.step_timeframe)
        self.time_simulator = TimeSimulator(
            start_date=self.start_date,
            end_date=self.end_date,
            step_minutes=step_minutes
        )

        # 5. ایجاد TradeManager
        self.trade_manager = BacktestTradeManager(
            config=self.config,
            initial_balance=self.initial_balance
        )

        # 🆕 6. ایجاد IndicatorCalculator
        logger.info("Initializing IndicatorCalculator...")
        self.indicator_calculator = IndicatorCalculator(self.config)
        logger.info("✅ IndicatorCalculator initialized")

        # 🆕 7. ایجاد SignalOrchestrator
        logger.info("Initializing SignalOrchestrator...")

        try:
            # غیرفعال کردن adaptive learning در backtest
            if 'signal_generation' in self.config:
                if 'adaptive_learning' not in self.config['signal_generation']:
                    self.config['signal_generation']['adaptive_learning'] = {}
                self.config['signal_generation']['adaptive_learning']['enabled'] = False
                self.config['signal_generation']['use_adaptive_learning'] = False

            # ایجاد Orchestrator
            self.signal_orchestrator = SignalOrchestrator(
                config=self.config,
                market_data_fetcher=self.data_fetcher,
                indicator_calculator=self.indicator_calculator,
                trade_manager_callback=None  # در backtest نیاز نیست
            )

            # غیرفعال کردن circuit breaker و correlation برای backtest
            if hasattr(self.signal_orchestrator, 'circuit_breaker'):
                self.signal_orchestrator.circuit_breaker.enabled = False
                logger.info("✅ Circuit breaker disabled for backtest")

            if hasattr(self.signal_orchestrator, 'correlation_manager'):
                self.signal_orchestrator.correlation_manager.enabled = False
                logger.info("✅ Correlation manager disabled for backtest")

            # غیرفعال کردن timeframe score cache برای backtest
            # در backtest نیاز به محاسبات دقیق و تکرارپذیر داریم، نه سرعت
            if hasattr(self.signal_orchestrator, 'tf_score_cache'):
                self.signal_orchestrator.tf_score_cache.enabled = False
                logger.info("✅ Timeframe score cache disabled for backtest (ensures accurate calculations)")

            logger.info("✅ SignalOrchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SignalOrchestrator: {e}", exc_info=True)
            raise

        logger.info("✅ All components initialized successfully")

    def _print_backtest_summary(self):
        """نمایش خلاصه تنظیمات Backtest"""
        logger.info("=" * 60)
        logger.info("BACKTEST CONFIGURATION SUMMARY (V2)")
        logger.info("=" * 60)
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Duration: {self.end_date - self.start_date}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Initial Balance: {self.initial_balance:,.2f} USDT")
        logger.info(f"Step Timeframe: {self.step_timeframe}")
        logger.info(f"Signal Timeframe: {self.signal_timeframe}")
        logger.info(f"Process Interval: {self.process_interval}s")
        logger.info(f"Total Steps: {self.time_simulator.total_steps:,}")
        logger.info(f"Using: SignalOrchestrator (v2.0)")
        logger.info("=" * 60)

    async def run(self):
        """اجرای Backtest"""
        logger.info("🚀 Starting Backtest V2...")
        self.is_running = True
        self.results['start_time'] = datetime.now()

        # نمایش خلاصه
        self._print_backtest_summary()

        # Progress bar
        if self.use_progress_bar:
            pbar = tqdm(
                total=self.time_simulator.total_steps,
                desc="Backtest Progress",
                unit="step"
            )

        try:
            # حلقه اصلی
            while not self.time_simulator.is_finished():
                current_time = self.time_simulator.get_current_time()

                # به‌روزرسانی زمان در provider
                self.historical_provider.set_current_time(current_time)
                self.data_fetcher.set_current_time(current_time)

                # بررسی آیا باید پردازش کنیم
                should_process = self.time_simulator.should_process(self.process_interval)

                if should_process:
                    # پردازش همه نمادها
                    await self._process_all_symbols(current_time)

                # به‌روزرسانی معاملات باز
                await self._update_open_trades(current_time)

                # حرکت به گام بعدی
                self.time_simulator.step()

                # به‌روزرسانی progress bar
                if self.use_progress_bar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'Balance': f"{self.trade_manager.balance:.0f}",
                        'Trades': f"{self.trade_manager.stats['total_trades']}"
                    })

            if self.use_progress_bar:
                pbar.close()

            logger.info("✅ Backtest completed successfully")

        except Exception as e:
            logger.error(f"❌ Error during backtest: {e}", exc_info=True)
            raise

        finally:
            self.is_running = False
            self.results['end_time'] = datetime.now()
            self.results['duration'] = self.results['end_time'] - self.results['start_time']

            # جمع‌آوری نتایج
            await self._collect_results()

    async def _process_all_symbols(self, current_time: datetime):
        """پردازش تمام نمادها و تولید سیگنال"""
        for symbol in self.symbols:
            try:
                await self._process_symbol(symbol, current_time)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def _process_symbol(self, symbol: str, current_time: datetime):
        """
        🆕 پردازش یک نماد با SignalOrchestrator
        
        تفاوت با نسخه قدیم:
        - استفاده از generate_signal_for_symbol به جای analyze_symbol
        - API تک تایم‌فریم
        - SignalOrchestrator خودش داده را fetch می‌کند

        Args:
            symbol: نام نماد
            current_time: زمان فعلی
        """
        try:
            # 🆕 تولید سیگنال با Orchestrator (API جدید)
            # Orchestrator خودش داده را fetch و تحلیل می‌کند
            signal = await self.signal_orchestrator.generate_signal_for_symbol(
                symbol=symbol,
                timeframe=self.signal_timeframe
            )

            if not signal:
                logger.debug(f"No signal generated for {symbol} at {current_time}")
                return

            # بررسی نوع signal (باید SignalInfo باشد)
            if not isinstance(signal, SignalInfo):
                logger.warning(f"Invalid signal type for {symbol}: {type(signal)}")
                return

            # بررسی امتیاز حداقل
            min_score = self.config.get('signal_generation', {}).get('minimum_signal_score', 50)
            
            # استخراج score از SignalInfo
            score_value = 0
            if hasattr(signal, 'score'):
                score_obj = signal.score
                if hasattr(score_obj, 'final_score'):
                    score_value = score_obj.final_score
                elif isinstance(score_obj, (int, float)):
                    score_value = float(score_obj)

            if score_value < min_score:
                logger.debug(
                    f"Signal score too low for {symbol}: {score_value:.2f} < {min_score}"
                )
                return

            # بررسی معتبر بودن قیمت‌ها
            if not all([signal.entry_price > 0, signal.stop_loss > 0, signal.take_profit > 0]):
                logger.debug(
                    f"Invalid prices for {symbol}: entry={signal.entry_price}, "
                    f"sl={signal.stop_loss}, tp={signal.take_profit}"
                )
                return

            # باز کردن معامله
            logger.info(
                f"✨ Valid signal for {symbol}: {signal.direction.upper()} "
                f"@ {signal.entry_price:.2f} (score: {score_value:.2f}, "
                f"SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f})"
            )
            await self._open_trade_from_signal(signal, current_time)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    async def _open_trade_from_signal(self, signal: SignalInfo, current_time: datetime):
        """
        باز کردن معامله بر اساس سیگنال

        Args:
            signal: SignalInfo object
            current_time: زمان فعلی
        """
        try:
            symbol = signal.symbol
            direction = signal.direction.lower()

            if direction not in ['long', 'short']:
                logger.warning(f"Invalid direction for {symbol}: {direction}")
                return

            # دریافت قیمت‌ها از سیگنال
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit

            if not all([entry_price, stop_loss, take_profit]):
                logger.warning(f"Invalid signal prices for {symbol}")
                return

            # محاسبه حجم پوزیشن
            position_size = self._calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                direction=direction
            )

            if position_size <= 0:
                logger.debug(f"Position size is zero or negative for {symbol}")
                return

            # استخراج score
            score_value = 0
            if hasattr(signal, 'score'):
                score_obj = signal.score
                if hasattr(score_obj, 'final_score'):
                    score_value = score_obj.final_score
                elif isinstance(score_obj, (int, float)):
                    score_value = float(score_obj)

            # باز کردن معامله
            trade = self.trade_manager.open_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                entry_time=current_time,
                signal_score=score_value,
                timeframe=signal.timeframe,
                metadata=signal.metadata if hasattr(signal, 'metadata') else {}
            )

            if trade:
                logger.info(
                    f"✅ Trade opened: {symbol} {direction.upper()} @ {entry_price:.2f} "
                    f"(size: {position_size:.2f} USDT, score: {score_value:.1f})"
                )

        except Exception as e:
            logger.error(f"Error opening trade from signal: {e}", exc_info=True)

    def _calculate_position_size(self, entry_price: float, stop_loss: float,
                                 direction: str) -> float:
        """
        محاسبه حجم پوزیشن (USDT) بر اساس ریسک

        Returns:
            حجم پوزیشن به USDT
        """
        risk_config = self.config.get('risk_management', {})
        risk_percent = risk_config.get('max_risk_per_trade_percent', 2.0)
        balance = self.trade_manager.balance

        # محاسبه ریسک مجاز (USDT)
        risk_amount = balance * (risk_percent / 100)

        # محاسبه درصد فاصله SL
        if direction == 'long':
            sl_distance_percent = abs((entry_price - stop_loss) / entry_price)
        else:
            sl_distance_percent = abs((stop_loss - entry_price) / entry_price)

        if sl_distance_percent == 0 or sl_distance_percent > 0.5:
            logger.warning(f"Invalid SL distance: {sl_distance_percent:.2%}")
            return 0.0

        # محاسبه position بر اساس ریسک
        position_size = risk_amount / sl_distance_percent

        # محدود کردن به 500 USDT
        max_position = risk_config.get('max_position_size', 500)
        min_position = 100

        # محدود به 95% موجودی
        max_allowed = balance * 0.95

        position_size = max(min_position, min(position_size, max_position, max_allowed))

        logger.debug(
            f"Position sizing: risk={risk_amount:.2f}, sl_dist={sl_distance_percent:.2%}, "
            f"calculated={position_size:.2f}"
        )

        return position_size

    async def _update_open_trades(self, current_time: datetime):
        """به‌روزرسانی معاملات باز"""
        if not self.trade_manager.active_trades:
            return

        # دریافت قیمت‌های فعلی
        prices = {}
        for symbol in self.symbols:
            price = await self.data_fetcher.get_current_price(symbol)
            if price:
                prices[symbol] = price

        # به‌روزرسانی معاملات
        self.trade_manager.update_all_trades(prices, current_time)

    async def _collect_results(self):
        """جمع‌آوری نتایج نهایی"""
        logger.info("Collecting backtest results...")

        # آمار معاملات
        self.results['statistics'] = self.trade_manager.get_statistics()

        # تاریخچه معاملات
        self.results['trades'] = [
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'direction': t.direction.value,
                'entry_price': t.entry_price,
                'entry_time': t.entry_time.isoformat(),
                'exit_price': t.exit_price,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'position_size': t.position_size,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'realized_pnl': t.realized_pnl,
                'exit_reason': t.exit_reason.value if t.exit_reason else None,
                'duration': str(t.exit_time - t.entry_time) if t.exit_time else None,
                'mfe': t.max_favorable_excursion,
                'mae': t.max_adverse_excursion
            }
            for t in self.trade_manager.get_trade_history()
        ]

        # Equity Curve
        self.results['equity_curve'] = self.trade_manager.get_equity_curve()

        # نمایش نتایج
        self._print_results()

    def _print_results(self):
        """نمایش نتایج در کنسول"""
        stats = self.results['statistics']

        print("\n" + "=" * 70)
        print(" " * 20 + "BACKTEST RESULTS V2")
        print("=" * 70)

        print(f"\n📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"⏱️  Duration: {self.end_date - self.start_date}")
        print(f"🚀 Execution Time: {self.results['duration']}")

        print(f"\n💰 FINANCIAL SUMMARY")
        print(f"Initial Balance: {self.initial_balance:,.2f} USDT")
        print(f"Final Equity: {stats['current_equity']:,.2f} USDT")
        print(f"Total Return: {stats['total_return']:+.2f}%")
        print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")

        print(f"\n📊 TRADE STATISTICS")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Winning Trades: {stats['winning_trades']} ({stats['win_rate']:.1f}%)")
        print(f"Losing Trades: {stats['losing_trades']}")
        print(f"Win/Loss Ratio: {stats['profit_factor']:.2f}")

        print(f"\n💵 PROFIT/LOSS")
        print(f"Total Profit: {stats['total_profit']:,.2f} USDT")
        print(f"Total Loss: {stats['total_loss']:,.2f} USDT")
        print(f"Average Win: {stats['average_win']:,.2f} USDT")
        print(f"Average Loss: {stats['average_loss']:,.2f} USDT")
        print(f"Largest Win: {stats['largest_win']:,.2f} USDT")
        print(f"Largest Loss: {stats['largest_loss']:,.2f} USDT")

        print(f"\n💸 COSTS")
        print(f"Total Commission: {stats['total_commission']:,.2f} USDT")
        print(f"Total Slippage: {stats['total_slippage']:,.2f} USDT")

        print(f"\n🎯 STREAKS")
        print(f"Max Consecutive Wins: {stats['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {stats['max_consecutive_losses']}")

        print("\n" + "=" * 70 + "\n")

    async def save_results(self, output_dir: str = None):
        """
        ذخیره نتایج در فایل

        Args:
            output_dir: پوشه خروجی (پیش‌فرض: backtest_results)
        """
        if output_dir is None:
            output_dir = self.backtest_config.get('results_dir', 'backtest_results_v2')

        output_path = Path(output_dir)

        # ایجاد پوشه با timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_path / f"v2_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {run_dir}")

        # 1. ذخیره آمار کلی (JSON)
        stats_file = run_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(self.results['statistics'], f, indent=2)

        # 2. ذخیره معاملات (CSV)
        if self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_file = run_dir / 'trades.csv'
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved {len(trades_df)} trades to {trades_file}")

        # 3. ذخیره Equity Curve (CSV)
        if self.results['equity_curve']:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            equity_file = run_dir / 'equity_curve.csv'
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"Saved equity curve to {equity_file}")

        # 4. ذخیره تنظیمات (JSON)
        config_file = run_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

        # 5. ذخیره اطلاعات نسخه
        version_file = run_dir / 'version.txt'
        with open(version_file, 'w') as f:
            f.write("Backtest Engine Version 2.0\n")
            f.write("Using: SignalOrchestrator\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")

        logger.info(f"✅ All results saved to: {run_dir}")

        return run_dir


# تابع کمکی برای اجرای Backtest
async def run_backtest_v2(config_path: str = 'backtest/config_backtest_v2.yaml'):
    """
    اجرای Backtest V2 با فایل کانفیگ

    Args:
        config_path: مسیر فایل کانفیگ
    """
    import yaml

    if not os.path.isabs(config_path):
        # اگر config_path شامل 'backtest/' است، آن را حذف کن
        if config_path.startswith('backtest/'):
            config_path = config_path.replace('backtest/', '', 1)
        # حالا مسیر را نسبت به فایل فعلی بساز
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    # بارگذاری تنظیمات
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ایجاد و اجرای Engine
    engine = BacktestEngineV2(config)
    await engine.initialize()
    await engine.run()

    # ذخیره نتایج
    results_dir = await engine.save_results()

    return engine, results_dir
