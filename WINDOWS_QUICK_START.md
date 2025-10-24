# 🪟 راهنمای سریع اجرا در Windows

## 🚀 روش آسان (با فایل Batch)

### گام 1: مطمئن شوید Python و dependencies نصب شده‌اند

باز کردن Command Prompt در مسیر پروژه:
```cmd
cd C:\Users\trade\Documents\PythonProject\New
```

فعال کردن venv (اگر قبلاً ایجاد کرده‌اید):
```cmd
venv\Scripts\activate
```

نصب dependencies (فقط بار اول):
```cmd
pip install pandas numpy scipy TA-Lib
```

### گام 2: اجرای تست با فایل Batch

Double-click روی `run_test_pattern.bat` یا از Command Prompt:

```cmd
run_test_pattern.bat doji
```

یا با الگوهای دیگر:
```cmd
run_test_pattern.bat hammer
run_test_pattern.bat "shooting star"
run_test_pattern.bat engulfing
```

---

## 📝 روش Command Line مستقیم

اگر ترجیح می‌دهید command را خودتان بنویسید:

```cmd
cd C:\Users\trade\Documents\PythonProject\New
venv\Scripts\activate
python signal_generation\tests\test_pattern.py --pattern doji --data-dir historical\BTC-USDT
```

---

## 🎯 الگوهایی که می‌توانید تست کنید

```cmd
run_test_pattern.bat doji
run_test_pattern.bat hammer
run_test_pattern.bat "inverted hammer"
run_test_pattern.bat engulfing
run_test_pattern.bat "morning star"
run_test_pattern.bat "piercing line"
run_test_pattern.bat "three white soldiers"
run_test_pattern.bat "shooting star"
run_test_pattern.bat "hanging man"
run_test_pattern.bat "evening star"
run_test_pattern.bat "dark cloud cover"
run_test_pattern.bat "three black crows"
run_test_pattern.bat harami
run_test_pattern.bat "double top bottom"
run_test_pattern.bat triangle
run_test_pattern.bat wedge
```

---

## 🐛 حل مشکلات رایج

### خطا: 'python' is not recognized

**حل:** Python نصب نیست یا به PATH اضافه نشده.

1. Python را از [python.org](https://www.python.org/downloads/) نصب کنید
2. حتماً گزینه "Add Python to PATH" را تیک بزنید

### خطا: ModuleNotFoundError

**حل 1:** مطمئن شوید در مسیر صحیح هستید:
```cmd
cd C:\Users\trade\Documents\PythonProject\New
```

**حل 2:** PYTHONPATH را set کنید:
```cmd
set PYTHONPATH=C:\Users\trade\Documents\PythonProject\New
```

**حل 3:** از فایل batch استفاده کنید (خودکار PYTHONPATH را set می‌کند)

### خطا: Data not found

**حل:** مطمئن شوید فایل‌های CSV در مسیر درست هستند:
```
New\
└── historical\
    └── BTC-USDT\
        ├── 5m.csv
        ├── 15m.csv
        ├── 1h.csv
        └── 4h.csv
```

---

## 💡 اجرا در PyCharm Terminal

در PyCharm:
1. Alt+F12 برای باز کردن Terminal
2. دستور را بنویسید:
   ```cmd
   run_test_pattern.bat doji
   ```

یا:
```cmd
python signal_generation\tests\test_pattern.py --pattern doji --data-dir historical\BTC-USDT
```

---

## 📊 خروجی نمونه

```
===================================================
Pattern Test Runner
===================================================

Current directory: C:\Users\trade\Documents\PythonProject\New

Activating virtual environment...

===================================================
Testing pattern: doji
===================================================

================================================================================
🎯 Pattern Testing: DOJI
================================================================================
Data Directory: historical\BTC-USDT
Timeframes: 5m, 15m, 1h, 4h

================================================================================
🔍 Testing DOJI on 5m
================================================================================
✓ Loaded 35040 candles
  Period: 2024-01-01 00:00:00 to 2024-12-31 23:55:00

✓ Pattern registered: DojiPattern

🔎 Scanning for doji patterns...
✓ Found 892 doji patterns

📊 Pattern Detections (892):
...
```

---

## ✅ شروع کنید!

```cmd
run_test_pattern.bat doji
```

نتیجه را به من بگویید! 🚀
