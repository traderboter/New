# 📘 مستندات کامل روند کار main.py

## فهرست مطالب
1. [معرفی کلی](#معرفی-کلی)
2. [نقطه شروع برنامه](#نقطه-شروع-برنامه)
3. [گام‌های اجرا به تفصیل](#گامهای-اجرا-به-تفصیل)
4. [مثال‌های کاربردی](#مثالهای-کاربردی)
5. [نمودار جریان کار](#نمودار-جریان-کار)
6. [سناریوهای مختلف](#سناریوهای-مختلف)

---

## معرفی کلی

فایل `main.py` **نقطه ورودی اصلی** (Entry Point) یک ربات معاملاتی ارز دیجیتال پیشرفته است که وظیفه آن:

✅ **مدیریت چرخه حیات کامل ربات** - از شروع تا خاتمه
✅ **مدیریت تنظیمات** - بارگذاری، اعمال و نظارت بر تغییرات
✅ **راه‌اندازی سیستم لاگینگ** - ثبت رویدادها در فایل و کنسول
✅ **مدیریت سیگنال‌های سیستم** - خاموش شدن تمیز هنگام Ctrl+C
✅ **اجرای ربات اصلی** - فراخوانی کلاس `CryptoTradingBot`

---

## نقطه شروع برنامه

### بلاک اصلی اجرا (خطوط 366-402)

```python
if __name__ == "__main__":
    exit_code = 1  # کد خروج پیش‌فرض
    try:
        exit_code = asyncio.run(main())  # 👈 اجرای تابع اصلی async
        logger.info(f"برنامه با کد خروج {exit_code} به پایان رسید.")
    except KeyboardInterrupt:
        # کاربر Ctrl+C زد
        print("\nبرنامه با فشار کلید قطع شد")
        exit_code = 1
    except SystemExit as e:
        # خروج عادی از برنامه
        exit_code = e.code
    except Exception as e:
        # خطای غیرمنتظره
        print(f"\nخطای فاجعه‌بار: {e}")
        exit_code = 1
    finally:
        logging.shutdown()  # بستن تمیز هندلرهای لاگ
        sys.exit(exit_code)
```

**📌 نکته مهم**: برنامه از `asyncio.run(main())` استفاده می‌کند، یعنی تمام عملیات به صورت **ناهمزمان** (Asynchronous) اجرا می‌شوند.

---

## گام‌های اجرا به تفصیل

### گام 1️⃣: پردازش آرگومان‌های خط فرمان (خطوط 168-186)

زمانی که شما دستور زیر را اجرا می‌کنید:

```bash
python main.py -c config.yaml -v --symbols BTC/USDT,ETH/USDT --backup
```

برنامه این پارامترها را پردازش می‌کند:

| آرگومان | نوع | توضیح | مثال |
|---------|-----|-------|------|
| `-c, --config` | str | مسیر فایل تنظیمات | `config.yaml` |
| `-v, --verbose` | flag | فعال‌سازی لاگ DEBUG | فعال/غیرفعال |
| `--symbols` | str | نمادهای معاملاتی (با کاما جدا شده) | `BTC/USDT,ETH/USDT` |
| `--strategy` | str | انتخاب استراتژی خاص | `trend_following` |
| `--backup` | flag | ایجاد پشتیبان قبل از شروع | فعال/غیرفعال |
| `--no-trading` | flag | حالت شبیه‌سازی (بدون معامله واقعی) | فعال/غیرفعال |
| `--no-watch-config` | flag | غیرفعال کردن نظارت بر تغییرات تنظیمات | فعال/غیرفعال |
| `--update-config` | str | به‌روزرسانی بخشی از تنظیمات | `trading:{"mode":"live"}` |

**مثال پیشرفته**:
```bash
python main.py \
  --config config.yaml \
  --verbose \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --strategy ensemble \
  --backup \
  --update-config 'risk_management:{"max_risk_per_trade_percent":1.0}'
```

---

### گام 2️⃣: بارگذاری تنظیمات (خطوط 189-209)

```python
config = load_config(args.config)  # بارگذاری YAML یا JSON
config_last_modified = os.path.getmtime(args.config)  # ثبت زمان تغییر
```

**چه اتفاقی می‌افتد؟**

1. **بارگذاری فایل**:
   - اگر فرمت YAML باشد → استفاده از `yaml.safe_load()`
   - اگر فرمت JSON باشد → استفاده از `json.loads()`

2. **بررسی وجود**: اگر فایل وجود نداشته باشد → خطا و خروج

3. **ثبت زمان آخرین تغییر**: برای بررسی‌های بعدی

4. **ایجاد بخش `config_management`** (اگر وجود نداشته باشد):
   ```python
   config['config_management'] = {
       'auto_reload': True,           # بررسی خودکار تغییرات
       'check_interval_seconds': 30,  # هر 30 ثانیه بررسی
       'notify_changes': True,        # اعلان تغییرات
       'backup_before_update': True   # پشتیبان قبل از تغییر
   }
   ```

5. **ایجاد پوشه داده**:
   ```python
   data_dir = config.get('storage', {}).get('data_directory', 'data')
   ensure_directory(data_dir)  # اطمینان از وجود پوشه
   ```

---

### گام 3️⃣: به‌روزرسانی تنظیمات از خط فرمان (خطوط 212-248)

اگر از پارامتر `--update-config` استفاده کرده باشید:

**مثال**:
```bash
python main.py --update-config 'risk_management:{"max_risk_per_trade_percent":2.0}'
```

**عملیات**:
```python
# 1. تجزیه آرگومان
section = 'risk_management'
json_value = '{"max_risk_per_trade_percent":2.0}'

# 2. تبدیل به Python dict
section_value = json.loads(json_value)

# 3. به‌روزرسانی/ادغام
if isinstance(section_value, dict) and isinstance(config[section], dict):
    config[section].update(section_value)  # ادغام با مقادیر موجود
else:
    config[section] = section_value  # جایگزینی کامل

# 4. ذخیره در فایل
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, allow_unicode=True)
```

---

### گام 4️⃣: تنظیم سیستم لاگینگ (خطوط 86-143)

```python
setup_logging(config, args.verbose)
```

**چه اتفاقی می‌افتد؟**

1. **تعیین سطح لاگ**:
   ```python
   if args.verbose:
       log_level = logging.DEBUG  # جزئیات کامل
   else:
       log_level = logging.INFO   # اطلاعات عمومی
   ```

2. **پیکربندی فرمت**:
   ```
   2025-10-23 14:30:45 - crypto_trading_bot - INFO - ربات شروع به کار کرد
   ```

3. **تنظیم هندلرها**:

   **🖥️ هندلر کنسول** (همیشه فعال):
   ```python
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setLevel(log_level)
   root_logger.addHandler(console_handler)
   ```

   **📄 هندلر فایل** (اختیاری):
   ```python
   if config['logging']['file']:  # مثلاً 'logs/crypto_bot.log'
       if config['logging']['rotate']:  # چرخش فایل فعال باشد
           file_handler = RotatingFileHandler(
               'logs/crypto_bot.log',
               maxBytes=10*1024*1024,  # 10 مگابایت
               backupCount=5            # 5 فایل پشتیبان
           )
       else:
           file_handler = logging.FileHandler('logs/crypto_bot.log')

       root_logger.addHandler(file_handler)
   ```

**نتیجه**: همه لاگ‌ها هم در کنسول و هم در فایل نمایش داده می‌شوند.

---

### گام 5️⃣: پشتیبان‌گیری اولیه (خطوط 257-287)

اگر پارامتر `--backup` را استفاده کرده باشید:

```bash
python main.py --backup
```

**عملیات**:
```python
# 1. تعیین مسیر پشتیبان
backup_dir = 'data/backups'
backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # مثلاً: 20251023_143045

# 2. فایل‌های مهم برای پشتیبان‌گیری
data_files = [
    'config.yaml',                              # تنظیمات
    'data/trades.db',                           # دیتابیس معاملات
    'data/adaptive_learning_data.json',         # داده‌های یادگیری
    'data/correlation_data.json',               # داده‌های همبستگی
    'data/performance_metrics.json'             # متریک‌های عملکرد
]

# 3. ایجاد فایل ZIP
backup_file = f'data/backups/manual_backup_20251023_143045.zip'
with zipfile.ZipFile(backup_file, 'w') as zipf:
    for file_path in data_files:
        if os.path.exists(file_path):
            zipf.write(file_path, os.path.basename(file_path))
```

**خروجی**: فایل `manual_backup_20251023_143045.zip` در پوشه `data/backups/`

---

### گام 6️⃣: اعمال تنظیمات خط فرمان روی کانفیگ (خطوط 289-300)

#### 🔸 حالت شبیه‌سازی

```bash
python main.py --no-trading
```

```python
# تغییر در حافظه (فایل config تغییر نمی‌کند)
config['trading']['mode'] = 'simulation'
logger.info("معاملات واقعی غیرفعال شد (حالت شبیه‌سازی)")
```

#### 🔸 نمادهای معاملاتی

```bash
python main.py --symbols BTCUSDT,ETHUSDT,ADAUSDT
```

```python
symbols_list = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
config['exchange']['symbols'] = symbols_list
logger.info(f"نمادهای معاملاتی: {symbols_list}")
```

---

### گام 7️⃣: ایجاد نمونه ربات اصلی (خطوط 302-322)

```python
bot_instance = CryptoTradingBot(args.config)
```

**چه اتفاقی می‌افتد؟**

این خط کل ربات را راه‌اندازی می‌کند:

1. **بارگذاری ماژول‌های زیرساخت**:
   - `MarketDataFetcher` → دریافت داده‌های بازار
   - `SignalProcessor` → پردازش سیگنال‌ها
   - `TradeManager` → مدیریت معاملات

2. **راه‌اندازی سیستم‌های هوشمند**:
   - `SignalOrchestrator` → تولید سیگنال‌های معاملاتی
   - `MarketRegimeDetector` → تشخیص رژیم بازار
   - `AdaptiveLearningSystem` → یادگیری تطبیقی
   - `CorrelationManager` → مدیریت همبستگی نمادها
   - `EmergencyCircuitBreaker` → قطع‌کننده اضطراری

3. **راه‌اندازی سیستم‌های AI** (اگر فعال باشند):
   - `TradingBrainAI` → هوش مصنوعی معاملاتی
   - `MLSignalIntegration` → یکپارچه‌سازی سیگنال‌های ML

#### تنظیم استراتژی اولیه (خطوط 307-316)

```bash
python main.py --strategy trend_following
```

```python
if args.strategy and hasattr(bot_instance, 'strategy_manager'):
    strategy_manager = bot_instance.strategy_manager
    loop = asyncio.get_running_loop()
    loop.create_task(strategy_manager.switch_strategy('trend_following'))
```

---

### گام 8️⃣: تنظیم هندلرهای سیگنال سیستم (خطوط 324-333)

```python
# تنظیم گیرنده‌های سیگنال برای Ctrl+C و توقف سیستم
for sig in (signal.SIGINT, signal.SIGTERM):
    loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))
```

**هدف**: خاموش شدن تمیز هنگام دریافت سیگنال‌های:
- `SIGINT` → Ctrl+C
- `SIGTERM` → دستور kill

**تابع signal_handler (خطوط 147-160)**:
```python
def signal_handler(sig, frame):
    print(f"\nسیگنال {sig} دریافت شد، در حال خاموش شدن...")
    if bot_instance:
        bot_instance.stop()  # 👈 توقف تمیز ربات
```

---

### گام 9️⃣: اجرای حلقه اصلی ربات (خطوط 340-362)

```python
logger.info("در حال شروع حلقه اصلی ربات...")
try:
    success = await bot_instance.run()  # 👈 اجرای ربات
    bot_exit_code = 0 if success else 1
except KeyboardInterrupt:
    logger.info("KeyboardInterrupt دریافت شد")
    bot_instance.stop()
    bot_exit_code = 1
except Exception as e:
    logger.critical(f"خطای پیش‌بینی نشده: {e}")
    bot_instance.stop()
    bot_exit_code = 1
finally:
    logger.info("حلقه اجرای ربات به پایان رسید")

return bot_exit_code
```

**متد `bot_instance.run()` چه کار می‌کند؟**

این متد حلقه اصلی ربات را اجرا می‌کند:

```python
async def run(self):
    """حلقه اصلی ربات"""
    while self._running:
        try:
            # 1. دریافت داده‌های بازار
            market_data = await self.market_data_fetcher.fetch_all_symbols()

            # 2. تولید سیگنال‌های معاملاتی
            signals = await self.signal_processor.process_symbols(market_data)

            # 3. اعتبارسنجی سیگنال‌ها
            valid_signals = [s for s in signals if s.score >= min_score]

            # 4. اجرای معاملات
            for signal in valid_signals:
                await self.trade_manager.execute_trade(signal)

            # 5. مدیریت معاملات باز
            await self.trade_manager.update_open_trades()

            # 6. بررسی تغییرات تنظیمات
            if config_changed:
                await self.reload_config()

            # 7. صبر تا چرخه بعدی
            await asyncio.sleep(main_loop_interval)

        except Exception as e:
            logger.error(f"خطا در حلقه اصلی: {e}")

    return True  # موفقیت‌آمیز
```

**پارامتر کلیدی**: `main_loop_interval` (از `config.yaml`)
```yaml
core:
  main_loop_interval: 300  # هر 5 دقیقه (300 ثانیه)
```

---

## مثال‌های کاربردی

### مثال 1: اجرای ساده

```bash
python main.py
```

**نتیجه**:
- بارگذاری تنظیمات از `config.yaml` (پیش‌فرض)
- اجرا در حالتی که در فایل تنظیمات مشخص شده (simulation یا live)
- استفاده از نمادهای خودکار یا تعریف شده در `config.yaml`

**لاگ خروجی**:
```
2025-10-23 14:30:45 - __main__ - INFO - --- شروع اجرای ربات معاملاتی --- (نسخه: 2.0)
2025-10-23 14:30:45 - crypto_trading_bot - INFO - راه‌اندازی ربات با config: config.yaml
2025-10-23 14:30:46 - market_data_fetcher - INFO - اتصال به صرافی KuCoin
2025-10-23 14:30:47 - signal_processor - INFO - سیستم تولید سیگنال راه‌اندازی شد
2025-10-23 14:30:47 - __main__ - INFO - در حال شروع حلقه اصلی ربات...
```

---

### مثال 2: حالت توسعه (Development)

```bash
python main.py \
  --verbose \
  --no-trading \
  --symbols BTCUSDT,ETHUSDT \
  --backup
```

**نتیجه**:
- **Verbose**: لاگ‌های DEBUG (جزئیات کامل)
- **No-trading**: فقط شبیه‌سازی (بدون معامله واقعی)
- **Symbols**: فقط BTC و ETH
- **Backup**: پشتیبان قبل از شروع

**لاگ خروجی**:
```
2025-10-23 14:35:10 - __main__ - INFO - نسخه پشتیبان دستی در data/backups/manual_backup_20251023_143510.zip ایجاد شد
2025-10-23 14:35:10 - __main__ - INFO - معاملات واقعی غیرفعال شد (حالت شبیه‌سازی)
2025-10-23 14:35:10 - __main__ - INFO - نمادهای معاملاتی از خط فرمان: ['BTCUSDT', 'ETHUSDT']
2025-10-23 14:35:10 - __main__ - DEBUG - بارگذاری ماژول MarketDataFetcher...
2025-10-23 14:35:11 - __main__ - DEBUG - راه‌اندازی SignalProcessor...
```

---

### مثال 3: تغییر تنظیمات و اجرا

```bash
python main.py \
  --update-config 'risk_management:{"max_risk_per_trade_percent":1.0,"max_open_trades":5}' \
  --strategy ensemble
```

**نتیجه**:
- تنظیمات ریسک به 1% و حداکثر معاملات باز به 5 تغییر می‌کند
- تغییرات در `config.yaml` ذخیره می‌شود
- استراتژی ensemble انتخاب می‌شود

**لاگ خروجی**:
```
2025-10-23 14:40:15 - __main__ - INFO - بخش 'risk_management' با مقادیر جدید به‌روزرسانی شد
2025-10-23 14:40:15 - __main__ - INFO - تنظیمات به‌روزرسانی شده در config.yaml ذخیره شد
2025-10-23 14:40:16 - crypto_trading_bot - INFO - در حال تنظیم استراتژی اولیه به: ensemble
```

---

### مثال 4: اجرای Production

```bash
python main.py \
  --config production_config.yaml \
  --no-watch-config
```

**نتیجه**:
- استفاده از فایل تنظیمات خاص production
- غیرفعال کردن نظارت خودکار بر تنظیمات (برای پایداری)
- اجرا در حالت واقعی (اگر در فایل تنظیمات `mode: live` باشد)

---

## نمودار جریان کار

```
┌─────────────────────────────────────────────────────────────┐
│                    شروع: python main.py                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              if __name__ == "__main__":                     │
│                 asyncio.run(main())                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          async def main():                                  │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │  1️⃣  پردازش آرگومان‌های خط فرمان                 │     │
│  │     argparse → -c, -v, --symbols, --backup, ...  │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  2️⃣  بارگذاری تنظیمات                             │     │
│  │     load_config(config.yaml)                     │     │
│  │     ایجاد پوشه‌های لازم (data/)                   │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  3️⃣  به‌روزرسانی تنظیمات (اگر --update-config)    │     │
│  │     ادغام JSON با تنظیمات موجود                  │     │
│  │     ذخیره در فایل config.yaml                    │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  4️⃣  تنظیم سیستم لاگینگ                           │     │
│  │     setup_logging(config, verbose)               │     │
│  │     ├─ Console Handler                           │     │
│  │     └─ File Handler (با چرخش)                    │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  5️⃣  پشتیبان‌گیری (اگر --backup)                  │     │
│  │     ایجاد فایل ZIP از:                            │     │
│  │     • config.yaml                                │     │
│  │     • trades.db                                  │     │
│  │     • adaptive_learning_data.json                │     │
│  │     • correlation_data.json                      │     │
│  │     • performance_metrics.json                   │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  6️⃣  اعمال تنظیمات خط فرمان                       │     │
│  │     • --no-trading → mode='simulation'           │     │
│  │     • --symbols → تنظیم نمادها                    │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  7️⃣  ایجاد نمونه ربات                             │     │
│  │     bot_instance = CryptoTradingBot(config)      │     │
│  │     ├─ MarketDataFetcher                         │     │
│  │     ├─ SignalProcessor                           │     │
│  │     │   └─ SignalOrchestrator                    │     │
│  │     ├─ TradeManager                              │     │
│  │     ├─ MarketRegimeDetector                      │     │
│  │     ├─ AdaptiveLearningSystem                    │     │
│  │     ├─ CorrelationManager                        │     │
│  │     └─ EmergencyCircuitBreaker                   │     │
│  │                                                   │     │
│  │     • تنظیم استراتژی (اگر --strategy)            │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  8️⃣  تنظیم هندلرهای سیگنال سیستم                  │     │
│  │     SIGINT (Ctrl+C) → signal_handler             │     │
│  │     SIGTERM (kill)  → signal_handler             │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  9️⃣  اجرای حلقه اصلی ربات                         │     │
│  │     await bot_instance.run()                     │     │
│  │                                                   │     │
│  │  ┌─────────────────────────────────────────┐     │     │
│  │  │  while self._running:                   │     │     │
│  │  │                                         │     │     │
│  │  │    🔄 دریافت داده‌های بازار             │     │     │
│  │  │       ↓                                 │     │     │
│  │  │    📊 تحلیل و تولید سیگنال‌ها          │     │     │
│  │  │       ↓                                 │     │     │
│  │  │    ✅ اعتبارسنجی سیگنال‌ها              │     │     │
│  │  │       ↓                                 │     │     │
│  │  │    💰 اجرای معاملات معتبر              │     │     │
│  │  │       ↓                                 │     │     │
│  │  │    📈 به‌روزرسانی معاملات باز          │     │     │
│  │  │       ↓                                 │     │     │
│  │  │    🔧 بررسی تغییرات تنظیمات            │     │     │
│  │  │       ↓                                 │     │     │
│  │  │    ⏰ انتظار تا چرخه بعدی (300s)       │     │     │
│  │  │       ↓                                 │     │     │
│  │  │    └──────────┐                         │     │     │
│  │  │               └──(loop back)            │     │     │
│  │  └─────────────────────────────────────────┘     │     │
│  │                                                   │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │                                 │
│                          ▼                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  🔟 خاتمه                                          │     │
│  │     return exit_code (0=موفق، 1=خطا)              │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  بستن تمیز (finally)                        │
│                 logging.shutdown()                          │
│                 sys.exit(exit_code)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## سناریوهای مختلف

### سناریو 1: شروع سریع برای تست

```bash
# هدف: تست سریع سیستم بدون ریسک
python main.py --verbose --no-trading --symbols BTCUSDT
```

**چه اتفاقی می‌افتد**:
1. ✅ لاگ‌های DEBUG فعال می‌شود (جزئیات کامل)
2. ✅ حالت شبیه‌سازی (هیچ معامله واقعی انجام نمی‌شود)
3. ✅ فقط BTC تحلیل می‌شود (سریع‌تر)
4. ✅ می‌توانید سیگنال‌ها را ببینید بدون خطر مالی

---

### سناریو 2: اجرای Production

```bash
# هدف: اجرای واقعی با معاملات Live
python main.py \
  --config production_config.yaml \
  --no-watch-config
```

**تنظیمات مهم در `production_config.yaml`**:
```yaml
trading:
  mode: 'live'  # ⚠️ معاملات واقعی

risk_management:
  max_risk_per_trade_percent: 0.5  # ریسک محافظه‌کارانه
  max_open_trades: 3

logging:
  level: 'WARNING'  # فقط هشدارها و خطاها
  file: 'logs/production.log'
  rotate: True
```

**توصیه‌ها**:
- ✅ از `--no-watch-config` استفاده کنید (جلوگیری از تغییرات ناخواسته)
- ✅ ریسک را پایین نگه دارید
- ✅ فایل لاگ جداگانه داشته باشید
- ✅ قبل از اجرا، backup بگیرید

---

### سناریو 3: تحلیل گزارش عملکرد

```bash
# اجرای عادی
python main.py

# بعد از چند ساعت، مشاهده گزارش عملکرد
cat data/performance_metrics.json
```

**محتوای فایل `performance_metrics.json`**:
```json
{
  "daily_metrics": {
    "2025-10-23": {
      "signals_generated": 45,
      "trades_executed": 12,
      "successful_trades": 8,
      "failed_trades": 4,
      "total_profit_usdt": 125.50,
      "total_loss_usdt": 45.20,
      "win_rate": 0.67,
      "profit_factor": 2.78,
      "avg_return_per_trade": 6.69
    }
  },
  "system_stats": {
    "start_time": "2025-10-23T09:00:00",
    "uptime_seconds": 21600,
    "restart_count": 0,
    "error_count": 2
  }
}
```

---

### سناریو 4: تغییر تنظیمات در زمان اجرا

**فایل `config.yaml` قابلیت auto-reload دارد**:

1. ربات را اجرا کنید:
   ```bash
   python main.py
   ```

2. ربات هر 30 ثانیه فایل `config.yaml` را بررسی می‌کند

3. فایل `config.yaml` را ویرایش کنید:
   ```yaml
   signal_generation:
     minimum_signal_score: 50  # قبلاً 46 بود
   ```

4. ذخیره کنید

5. ربات خودکار تغییرات را تشخیص می‌دهد:
   ```
   2025-10-23 15:20:10 - config_manager - INFO - تغییرات در config.yaml تشخیص داده شد
   2025-10-23 15:20:10 - config_manager - INFO - ایجاد پشتیبان: config.yaml.backup_20251023_152010
   2025-10-23 15:20:10 - config_manager - INFO - تنظیمات جدید اعمال شد
   2025-10-23 15:20:10 - signal_processor - INFO - آستانه امتیاز سیگنال به 50 تغییر یافت
   ```

**⚠️ نکته**: برخی تنظیمات نیاز به restart دارند:
- کلیدهای API
- نوع بازار (futures/spot)
- تنظیمات Redis

---

### سناریو 5: اشکال‌زدایی مشکلات

```bash
# فعال کردن حداکثر جزئیات لاگ
python main.py --verbose --symbols ETHUSDT 2>&1 | tee debug.log
```

**خروجی** (در فایل `debug.log`):
```
2025-10-23 16:00:01 - market_data_fetcher - DEBUG - درخواست OHLCV برای ETHUSDT (1h)
2025-10-23 16:00:01 - exchange_client - DEBUG - GET https://api-futures.kucoin.com/api/v1/kline?symbol=ETHUSDT&granularity=60
2025-10-23 16:00:02 - market_data_fetcher - DEBUG - دریافت 500 کندل برای ETHUSDT
2025-10-23 16:00:02 - signal_processor - DEBUG - شروع تحلیل ETHUSDT با تایم‌فریم 1h
2025-10-23 16:00:03 - trend_analyzer - DEBUG - روند شناسایی شده: صعودی (قدرت: 2.3)
2025-10-23 16:00:03 - momentum_analyzer - DEBUG - RSI: 58.2, MACD: تقاطع صعودی
2025-10-23 16:00:04 - signal_scorer - DEBUG - امتیاز نهایی: 72.5 (آستانه: 60)
2025-10-23 16:00:04 - signal_validator - DEBUG - سیگنال معتبر است ✅
```

---

## توابع کمکی مهم

### 1. `ensure_directory(path)`

```python
def ensure_directory(path: str) -> bool:
    """اطمینان از وجود یک دایرکتوری"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"خطا در ایجاد دایرکتوری {path}: {e}")
        return False
```

**کاربرد**:
```python
ensure_directory('data/backups')  # ایجاد اگر وجود نداشته باشد
ensure_directory('logs')
ensure_directory('models/trading_brain')
```

---

### 2. `format_time(timestamp)`

```python
def format_time(timestamp: Optional[float] = None) -> str:
    """فرمت‌بندی زمان برای فایل‌نام"""
    timestamp = timestamp or time.time()
    return datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
```

**کاربرد**:
```python
backup_file = f"backup_{format_time()}.zip"
# خروجی: backup_20251023_143045.zip
```

---

### 3. `load_config(config_path)`

```python
def load_config(config_path: str) -> Dict[str, Any]:
    """بارگذاری فایل تنظیمات (YAML یا JSON)"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"فایل تنظیمات یافت نشد: {config_path}")

    if config_path.suffix in ('.yaml', '.yml'):
        return yaml.safe_load(config_path.read_text(encoding='utf-8'))
    elif config_path.suffix == '.json':
        return json.loads(config_path.read_text(encoding='utf-8'))
    else:
        raise ValueError(f"فرمت فایل پشتیبانی نمی‌شود: {config_path.suffix}")
```

**کاربرد**:
```python
config = load_config('config.yaml')      # ✅
config = load_config('config.json')      # ✅
config = load_config('config.txt')       # ❌ خطا
config = load_config('missing.yaml')     # ❌ فایل وجود ندارد
```

---

### 4. `check_config_changes(config_path)`

```python
def check_config_changes(config_path: str) -> bool:
    """بررسی تغییرات در فایل تنظیمات"""
    global config_last_modified

    current_mtime = os.path.getmtime(config_path)
    if current_mtime > config_last_modified:
        config_last_modified = current_mtime
        return True  # تغییر کرده
    return False  # تغییر نکرده
```

**کاربرد** (در حلقه اصلی ربات):
```python
while running:
    # ... کارهای دیگر ...

    if check_config_changes('config.yaml'):
        logger.info("تنظیمات تغییر کرده، در حال بارگذاری مجدد...")
        config = load_config('config.yaml')
        apply_new_config(config)

    await asyncio.sleep(30)  # بررسی هر 30 ثانیه
```

---

## مدیریت خطاها

### خطاهای رایج و راه‌حل‌ها

#### ❌ خطا 1: فایل تنظیمات یافت نشد

```
FileNotFoundError: فایل تنظیمات یافت نشد: config.yaml
```

**راه‌حل**:
```bash
# ایجاد فایل config.yaml از نمونه
cp config.example.yaml config.yaml

# یا مشخص کردن مسیر دیگر
python main.py --config /path/to/my_config.yaml
```

---

#### ❌ خطا 2: فرمت JSON نامعتبر در --update-config

```
json.JSONDecodeError: مقدار JSON نامعتبر
```

**راه‌حل**:
```bash
# ❌ اشتباه
python main.py --update-config 'trading:{mode:live}'

# ✅ صحیح
python main.py --update-config 'trading:{"mode":"live"}'
```

---

#### ❌ خطا 3: خطای اتصال به صرافی

```
ConnectionError: خطا در اتصال به KuCoin API
```

**راه‌حل**:
1. بررسی اتصال اینترنت
2. بررسی کلیدهای API در `config.yaml`:
   ```yaml
   exchange:
     api_key: 'YOUR_API_KEY'
     api_secret: 'YOUR_API_SECRET'
     api_passphrase: 'YOUR_PASSPHRASE'
   ```
3. بررسی محدودیت‌های IP در صرافی

---

#### ❌ خطا 4: خطای Redis

```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**راه‌حل**:
```bash
# روش 1: نصب و راه‌اندازی Redis
sudo apt-get install redis-server
sudo systemctl start redis

# روش 2: غیرفعال کردن Redis
# ویرایش config.yaml:
data_fetching:
  cache:
    use_redis: False  # استفاده از کش حافظه
```

---

## نکات مهم امنیتی

### 🔐 محافظت از کلیدهای API

**❌ هرگز این کار را نکنید**:
```bash
# کلیدهای API در خط فرمان (در history ذخیره می‌شود!)
python main.py --update-config 'exchange:{"api_key":"YOUR_SECRET_KEY"}'
```

**✅ روش صحیح**:
```bash
# 1. استفاده از متغیرهای محیطی
export KUCOIN_API_KEY="your_key"
export KUCOIN_API_SECRET="your_secret"
export KUCOIN_API_PASSPHRASE="your_passphrase"

# 2. خواندن از متغیرها در کد
```

یا در `config.yaml`:
```yaml
exchange:
  api_key: ${KUCOIN_API_KEY}
  api_secret: ${KUCOIN_API_SECRET}
  api_passphrase: ${KUCOIN_API_PASSPHRASE}
```

---

## خلاصه کارهای main.py

| مرحله | عملیات | نتیجه |
|-------|---------|-------|
| **1** | پردازش آرگومان‌های CLI | تعیین تنظیمات اجرا |
| **2** | بارگذاری config.yaml | بارگذاری تنظیمات اصلی |
| **3** | به‌روزرسانی تنظیمات | اعمال تغییرات از CLI |
| **4** | راه‌اندازی لاگینگ | ثبت رویدادها در فایل و کنسول |
| **5** | پشتیبان‌گیری | ایجاد نسخه امن از داده‌ها |
| **6** | اعمال تنظیمات CLI | تنظیم حالت و نمادها |
| **7** | ایجاد ربات | راه‌اندازی کامل سیستم |
| **8** | تنظیم Signal Handlers | مدیریت Ctrl+C |
| **9** | اجرای حلقه اصلی | معاملات و تحلیل مداوم |
| **10** | خاتمه تمیز | بستن اتصالات و ذخیره داده‌ها |

---

## جمع‌بندی

**`main.py` = دروازه ورودی + مدیر سیستم**

این فایل:
- 🚀 برنامه را راه‌اندازی می‌کند
- ⚙️ تنظیمات را مدیریت می‌کند
- 📝 سیستم لاگ را راه‌اندازی می‌کند
- 💾 از داده‌ها پشتیبان می‌گیرد
- 🤖 ربات اصلی را اجرا می‌کند
- 🛑 خاموش شدن تمیز را تضمین می‌کند

**بدون این فایل، ربات نمی‌تواند کار کند!**

---

**📅 آخرین به‌روزرسانی**: 2025-10-23
**✍️ نویسنده**: مستندسازی خودکار برای پروژه Crypto Trading Bot
