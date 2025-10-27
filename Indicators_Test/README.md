# Indicators Test Suite

این پوشه شامل تست‌های جامع برای تمام اندیکاتورهای تکنیکال است.

## ساختار تست‌ها

- `test_utils.py` - توابع کمکی برای ایجاد داده‌های تست
- `test_sma.py` - تست‌های Simple Moving Average
- `test_ema.py` - تست‌های Exponential Moving Average
- `test_rsi.py` - تست‌های Relative Strength Index
- `test_macd.py` - تست‌های MACD
- `test_bollinger_bands.py` - تست‌های Bollinger Bands
- `test_atr.py` - تست‌های Average True Range
- `test_stochastic.py` - تست‌های Stochastic Oscillator
- `test_obv.py` - تست‌های On-Balance Volume
- `test_indicator_orchestrator.py` - تست‌های Orchestrator

## اجرای تست‌ها

### اجرای همه تست‌ها:
```bash
pytest Indicators_Test/ -v
```

### اجرای تست یک اندیکاتور خاص:
```bash
pytest Indicators_Test/test_rsi.py -v
```

### اجرای با coverage report:
```bash
pytest Indicators_Test/ --cov=signal_generation.analyzers.indicators --cov-report=html
```

## نیازمندی‌ها

```bash
pip install pytest pandas numpy
```

## انواع تست‌ها

### ۱. تست‌های اساسی
- بررسی initialization صحیح
- بررسی محاسبات پایه
- بررسی ستون‌های خروجی

### ۲. تست‌های صحت محاسبات
- مقایسه با مقادیر دستی
- بررسی فرمول‌های ریاضی
- مقایسه با مقادیر معتبر

### ۳. تست‌های Edge Cases
- داده‌های خالی
- داده‌های ناکافی
- ستون‌های گم‌شده
- مقادیر NaN و Inf
- ولوم صفر یا منفی

### ۴. تست‌های رفتاری
- روند صعودی
- روند نزولی
- قیمت ثابت
- نوسانات بالا

### ۵. تست‌های عملکرد
- Caching
- محاسبات متعدد
- مدیریت خطا

## پوشش تست

هر فایل تست شامل:
- ✅ حداقل 15-20 تست
- ✅ تست تمام متدهای عمومی
- ✅ تست edge cases
- ✅ تست با داده‌های واقعی
- ✅ تست مدیریت خطا

## مثال استفاده

```python
from Indicators_Test.test_utils import create_sample_ohlcv_data
from signal_generation.analyzers.indicators.rsi import RSIIndicator

# ایجاد داده تست
df = create_sample_ohlcv_data(num_rows=100)

# ایجاد اندیکاتور
rsi = RSIIndicator({'rsi_period': 14})

# محاسبه
result_df = rsi.calculate_safe(df)

# استخراج مقادیر
latest_rsi = rsi.get_latest_value(result_df, 'rsi')
print(f"Latest RSI: {latest_rsi}")
```

## نکات مهم

1. **داده‌های تست**: از `test_utils.py` برای ایجاد داده‌های تست استفاده کنید
2. **Assertions**: از توابع کمکی مثل `assert_column_exists` و `assert_no_inf` استفاده کنید
3. **پوشش کامل**: هر تست باید یک جنبه خاص را بررسی کند
4. **مستندسازی**: هر تست باید docstring واضح داشته باشد

## نتایج بررسی کدها

کدهای اندیکاتورها **به خوبی نوشته شده‌اند** با امتیاز کلی ⭐⭐⭐⭐ (4 از 5).

نقاط قوت:
- معماری تمیز با کلاس پایه
- مدیریت خطای جامع
- Validation کامل
- پشتیبانی از Caching
- مستندسازی عالی

نکات قابل بهبود:
- بهینه‌سازی حلقه در EMA
- افزایش پوشش تست
