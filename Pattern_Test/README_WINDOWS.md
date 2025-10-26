# 🪟 راهنمای اجرا در Windows - تست الگوهای کندلی

این راهنما نحوه اجرای تست‌های الگوی کندلی در Windows را توضیح می‌دهد.

## 📋 پیش‌نیازها

### 1. Python 3.8+
بررسی کنید که Python نصب شده باشد:
```cmd
python --version
```

اگر نصب نیست، از [python.org](https://www.python.org/downloads/) دانلود کنید.

### 2. نصب کتابخانه‌های مورد نیاز

#### روش 1: استفاده از requirements.txt (توصیه می‌شود)
```cmd
cd C:\Users\trade\Documents\PythonProject\New\Pattern_Test
pip install -r requirements.txt
```

#### روش 2: نصب دستی
```cmd
pip install pandas
pip install matplotlib
pip install numpy
```

**نکته مهم برای TA-Lib:**
TA-Lib نیاز به نصب ویژه در Windows دارد:

```cmd
# دانلود فایل wheel مناسب از:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

# سپس نصب (مثال برای Python 3.10 و 64-bit):
pip install TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl
```

### 3. ساختار فولدر

مطمئن شوید که ساختار زیر را دارید:
```
C:\Users\trade\Documents\PythonProject\New\
├── Pattern_Test\
│   ├── test_doji_windows.py    ← فایل اصلی
│   ├── requirements.txt
│   ├── README_WINDOWS.md       ← این فایل
│   └── Charts\                  ← خودکار ساخته می‌شود
├── historical\
│   └── BTC-USDT\
│       ├── 5min.csv
│       ├── 15min.csv
│       ├── 1hour.csv
│       └── 4hour.csv
└── signal_generation\
    └── analyzers\
        └── patterns\
            └── candlestick\
                └── doji.py
```

## 🚀 نحوه اجرا

### روش 1: از PyCharm

1. باز کردن فایل `test_doji_windows.py` در PyCharm
2. **کلیک راست** روی فایل
3. انتخاب **"Run 'test_doji_windows'"** (نه "Run with pytest"!)
4. منتظر بمانید تا تست تمام شود

### روش 2: از Command Prompt

```cmd
cd C:\Users\trade\Documents\PythonProject\New
python Pattern_Test\test_doji_windows.py
```

### روش 3: از Terminal در PyCharm

1. باز کردن Terminal در PyCharm (Alt+F12)
2. اجرای دستور:
```cmd
python Pattern_Test\test_doji_windows.py
```

## 📊 خروجی انتظاری

```
================================================================================
🧪 تست الگوی Doji روی داده‌های تاریخی BTC/USDT - Windows
================================================================================
📁 Project root: C:\Users\trade\Documents\PythonProject\New
✅ DojiPattern imported successfully
✅ DojiPatternTester initialized
   📂 Data directory: C:\Users\trade\Documents\PythonProject\New\historical\BTC-USDT
   📊 Charts directory: C:\Users\trade\Documents\PythonProject\New\Pattern_Test\Charts

📊 تایم‌فریم انتخاب شده: 5min

📖 در حال خواندن C:\Users\trade\Documents\PythonProject\New\historical\BTC-USDT\5min.csv...
   ✅ 2500 کندل بارگذاری شد
   📅 از 2024-11-23 02:10:00 تا 2024-12-01 10:35:00

🔍 شروع تست کندل به کندل از کندل 100...
   ⏳ پیشرفت: 0.0% (100/2500)
   ⏳ پیشرفت: 4.2% (200/2500)

🎯 الگوی Doji #1 پیدا شد!
   📍 کندل 523: 2024-11-25 08:25:00
   💰 OHLC: O=95871.20 H=95898.30 L=95835.00 C=95871.20
   ⭐ Confidence: 75.0%
   📊 Location: current
   🔍 Candles ago: 0
   💾 نمودار ذخیره شد: doji_5min_candle_523_2024-11-25_0825.png

...

📊 نتایج نهایی:
   🔍 تعداد کندل‌های بررسی شده: 2400
   ✅ تعداد الگوهای پیدا شده: 15

💾 نتایج در doji_detections_5min.json ذخیره شد

================================================================================
✅ تست با موفقیت انجام شد!
📊 15 الگوی Doji پیدا شد
📁 نمودارها در C:\...\Pattern_Test\Charts ذخیره شدند
================================================================================

📋 نمونه نتایج (5 اولی):
   1. کندل 523: 2024-11-25 08:25:00 - C=95871.20
   2. کندل 784: 2024-11-25 12:30:00 - C=96234.50
   ...

⏸️  Press Enter to exit...
```

## 🔧 حل مشکلات رایج

### ❌ خطا: `ModuleNotFoundError: No module named 'signal_generation'`

**علت:** مسیر پروژه اشتباه است

**حل:**
1. مطمئن شوید که فایل `test_doji_windows.py` در پوشه `Pattern_Test` است
2. ساختار پوشه‌ها را بررسی کنید
3. مطمئن شوید که پوشه `signal_generation` در کنار `Pattern_Test` وجود دارد

### ❌ خطا: `FileNotFoundError: فایل ...5min.csv پیدا نشد`

**علت:** فایل‌های CSV در مسیر درست نیستند

**حل:**
1. بررسی کنید که پوشه `historical/BTC-USDT/` وجود دارد
2. بررسی کنید که فایل‌های CSV در آن پوشه هستند
3. نام فایل‌ها باید دقیقاً این‌ها باشد:
   - `5min.csv`
   - `15min.csv`
   - `1hour.csv`
   - `4hour.csv`

### ❌ خطا: `No module named 'pandas'`

**حل:**
```cmd
pip install pandas matplotlib numpy
```

### ❌ PyCharm سعی می‌کند با pytest اجرا کند

**حل:**
1. کلیک راست روی فایل
2. انتخاب **"Modify Run Configuration..."**
3. در قسمت **"Target"**، انتخاب **"Script path"** به جای "pytest"
4. مسیر فایل را وارد کنید: `C:\Users\trade\Documents\PythonProject\New\Pattern_Test\test_doji_windows.py`
5. OK و سپس Run

یا ساده‌تر: **Terminal را باز کنید و دستور python را اجرا کنید**

### ⚠️ نمودار ذخیره نمی‌شود

**علت:** matplotlib یا مشکل permission

**حل:**
```cmd
pip install matplotlib --upgrade
```

و مطمئن شوید که پوشه `Charts` قابل نوشتن است (read-only نباشد)

## ⚙️ تنظیمات

می‌توانید در فایل `test_doji_windows.py` موارد زیر را تغییر دهید:

```python
# خط 308 تقریباً
timeframe = '5min'    # یا '15min', '1hour', '4hour'

# خط 317 تقریباً
lookback = 50         # تعداد کندل‌های قبلی برای نمایش
start_from = 100      # شروع تست از کندل چندم
```

## 📁 فایل‌های خروجی

بعد از اجرا، این فایل‌ها ایجاد می‌شوند:

```
Pattern_Test\
├── Charts\
│   ├── doji_5min_candle_523_2024-11-25_0825.png
│   ├── doji_5min_candle_784_2024-11-25_1230.png
│   └── ...
└── doji_detections_5min.json
```

### نمودارها (PNG)
- 50 کندل قبلی + کندل فعلی
- علامت‌گذاری کندل تشخیص داده شده (دایره آبی)
- اطلاعات الگو در گوشه

### فایل JSON
```json
[
  {
    "index": 523,
    "timestamp": "2024-11-25 08:25:00",
    "open": 95871.2,
    "high": 95898.3,
    "low": 95835.0,
    "close": 95871.2,
    "volume": 12345.67,
    "timeframe": "5min"
  }
]
```

## 🎯 مثال کامل اجرا

```cmd
# 1. رفتن به پوشه پروژه
cd C:\Users\trade\Documents\PythonProject\New

# 2. بررسی وجود فایل‌های CSV
dir historical\BTC-USDT\*.csv

# 3. نصب کتابخانه‌ها (اولین بار)
pip install -r Pattern_Test\requirements.txt

# 4. اجرای تست
python Pattern_Test\test_doji_windows.py

# 5. بررسی نتایج
dir Pattern_Test\Charts\*.png
type Pattern_Test\doji_detections_5min.json
```

## 💡 نکات مهم

1. **حتماً از Terminal/CMD استفاده کنید**، نه از pytest runner در PyCharm
2. **مسیر کاری**: باید در ریشه پروژه باشید (`New\`)
3. **فایل‌های CSV**: باید در `historical/BTC-USDT/` باشند
4. **هر بار اجرا**: چارت‌های قبلی پاک می‌شوند
5. **زمان اجرا**: بسته به تعداد کندل‌ها، ممکن است چند دقیقه طول بکشد

## 🐛 Debug

اگر مشکل دارید، این اطلاعات را بررسی کنید:

```python
# در خط اول خروجی باید ببینید:
📁 Project root: C:\Users\trade\Documents\PythonProject\New

# اگر مسیر اشتباه است، یعنی ساختار پوشه‌ها درست نیست
```

## ✅ چک‌لیست قبل از اجرا

- [ ] Python 3.8+ نصب شده
- [ ] pandas, matplotlib نصب شده
- [ ] فایل‌های CSV در `historical/BTC-USDT/` موجود هستند
- [ ] پوشه `signal_generation` موجود است
- [ ] در Terminal/CMD هستید (نه pytest)
- [ ] در ریشه پروژه هستید (`New\`)

---

**سوال یا مشکل دارید؟**
- خروجی کامل را کپی کنید
- ساختار پوشه‌ها را بررسی کنید
- مسیر project_root را در خروجی چک کنید
