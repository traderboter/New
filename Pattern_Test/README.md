# 🧪 Pattern Test - تست الگوهای کندلی

این فولدر شامل ابزارهایی برای تست الگوهای کندلی روی داده‌های تاریخی است.

## 📁 ساختار فولدر

```
Pattern_Test/
├── test_doji_simple.py        # تست الگوی Doji
├── Charts/                     # نمودارهای تولید شده
├── doji_detections_*.json     # نتایج تشخیص الگو
└── README.md                   # این فایل
```

## 🚀 نحوه استفاده

### پیش‌نیازها

```bash
# نصب کتابخانه‌های مورد نیاز
pip install pandas matplotlib
```

### اجرای تست Doji

```bash
cd /home/user/New/Pattern_Test
python3 test_doji_simple.py
```

## 📊 خروجی‌ها

### 1. نمودارهای کندلی (Charts/)
برای هر الگوی تشخیص داده شده، یک نمودار PNG تولید می‌شود که شامل:
- 50 کندل قبلی + کندل فعلی
- علامت‌گذاری کندل تشخیص داده شده (دایره آبی)
- اطلاعات الگو (confidence, direction, location)

### 2. فایل JSON نتایج
اطلاعات تمام الگوهای تشخیص داده شده در فایل JSON ذخیره می‌شود:
- ایندکس کندل
- timestamp
- OHLCV
- تایم‌فریم

## 🔧 تنظیمات

می‌توانید در فایل `test_doji_simple.py` موارد زیر را تغییر دهید:

```python
# انتخاب تایم‌فریم
timeframe = '5min'  # یا '15min', '1hour', '4hour'

# تعداد کندل‌های قبلی برای نمایش
lookback = 50

# شروع تست از کندل چندم
start_from = 100
```

## 📝 نمونه خروجی

```
🧪 تست الگوی Doji روی داده‌های تاریخی BTC/USDT
================================================================================
✅ DojiPatternTester initialized
   📂 Data directory: /home/user/New/historical/BTC-USDT
   📂 Output directory: /home/user/New/Pattern_Test
   📊 Charts directory: /home/user/New/Pattern_Test/Charts

📊 تایم‌فریم انتخاب شده: 5min

📖 در حال خواندن /home/user/New/historical/BTC-USDT/5min.csv...
   ✅ 2500 کندل بارگذاری شد
   📅 از 2024-11-23 02:10:00 تا 2024-12-01 10:35:00

🔍 شروع تست کندل به کندل از کندل 100...

🎯 الگوی Doji پیدا شد!
   📍 کندل 523: 2024-11-25 08:25:00
   💰 OHLC: O=95871.20 H=95898.30 L=95835.00 C=95871.20
   ⭐ Confidence: 75.0%
   📊 Location: current
   🔍 Candles ago: 0
   💾 نمودار ذخیره شد: doji_5min_candle_523_2024-11-25_0825.png

...

📊 نتایج:
   🔍 تعداد کندل‌های بررسی شده: 2400
   ✅ تعداد الگوهای پیدا شده: 15

💾 نتایج در /home/user/New/Pattern_Test/doji_detections_5min.json ذخیره شد

================================================================================
✅ تست با موفقیت انجام شد!
📊 15 الگوی Doji پیدا شد
📁 نمودارها در /home/user/New/Pattern_Test/Charts ذخیره شدند
================================================================================
```

## 🎯 چگونه کار می‌کند؟

1. **بارگذاری داده‌ها**: فایل CSV تایم‌فریم مورد نظر بارگذاری می‌شود
2. **حلقه کندل به کندل**: از کندل 100 شروع کرده و هر بار یک کندل جلو می‌رود
3. **تشخیص الگو**: برای هر کندل، الگوی Doji با کد واقعی تشخیص داده می‌شود
4. **رسم نمودار**: اگر الگو پیدا شد، نمودار 50 کندل قبلی + کندل فعلی رسم می‌شود
5. **ذخیره نتایج**: اطلاعات در JSON و نمودارها در PNG ذخیره می‌شوند

## ⚙️ تست الگوهای دیگر

برای افزودن تست الگوهای دیگر (مثل Hammer, Engulfing و غیره):
1. فایل جدید بسازید (مثل `test_hammer_simple.py`)
2. import pattern detector را تغییر دهید
3. نام‌ها و فایل‌های خروجی را مطابق بدهید

مثال:
```python
from signal_generation.analyzers.patterns.candlestick.hammer import HammerPattern

self.pattern_detector = HammerPattern()
```

## 📌 نکات مهم

- هر بار اجرای برنامه، چارت‌های قبلی پاک می‌شوند
- تشخیص الگو با کد واقعی `signal_generation` انجام می‌شود
- نتایج برای تحلیل و debugging مفید هستند
- می‌توانید تایم‌فریم‌های مختلف را تست کنید

## 🐛 عیب‌یابی

### خطا: ModuleNotFoundError: No module named 'pandas'
```bash
pip install pandas
```

### خطا: ModuleNotFoundError: No module named 'matplotlib'
```bash
pip install matplotlib
```

### خطا: فایل CSV پیدا نشد
بررسی کنید که فایل‌های CSV در مسیر زیر وجود دارند:
```
/home/user/New/historical/BTC-USDT/
├── 5min.csv
├── 15min.csv
├── 1hour.csv
└── 4hour.csv
```
