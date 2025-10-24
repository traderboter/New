# 🪟 راهنمای اجرا در Windows

## ⚠️ مشکل شما

فایل `test_pattern.py` یک اسکریپت command-line است و با PyCharm Test Runner اجرا نمی‌شود.

---

## ✅ راه حل 1: اجرا از Command Line (توصیه می‌شود)

### گام 1: باز کردن Command Prompt یا Terminal در PyCharm

در PyCharm:
- Alt+F12 (Terminal در پایین)
- یا View → Tool Windows → Terminal

### گام 2: رفتن به مسیر پروژه

```cmd
cd C:\Users\trade\Documents\PythonProject\New
```

### گام 3: فعال کردن Virtual Environment

```cmd
venv\Scripts\activate
```

### گام 4: اجرای تست

```cmd
python signal_generation/tests/test_pattern.py --pattern doji --data-dir historical/BTC-USDT
```

---

## ✅ راه حل 2: تنظیم PYTHONPATH در PyCharm

### روش A: در Run Configuration

1. Right-click روی `test_pattern.py`
2. Modify Run Configuration
3. در قسمت **Environment Variables** کلیک کنید
4. اضافه کنید:
   ```
   Name: PYTHONPATH
   Value: C:\Users\trade\Documents\PythonProject\New
   ```
5. در قسمت **Parameters** بنویسید:
   ```
   --pattern doji --data-dir historical/BTC-USDT
   ```
6. OK → سپس Run کنید

### روش B: با Script Parameters

1. Right-click روی `test_pattern.py`
2. Run 'test_pattern'
3. Edit Configurations
4. Script parameters: `--pattern doji --data-dir historical/BTC-USDT`
5. Working directory: `C:\Users\trade\Documents\PythonProject\New`
6. OK

---

## ✅ راه حل 3: فایل Batch برای اجرای آسان

محتوای این فایل را ذخیره کنید به عنوان `run_test.bat` در مسیر `New\`:

```batch
@echo off
cd /d "C:\Users\trade\Documents\PythonProject\New"
call venv\Scripts\activate
python signal_generation/tests/test_pattern.py --pattern %1 --data-dir historical/BTC-USDT
pause
```

سپس double-click کنید یا از CMD:
```cmd
run_test.bat doji
run_test.bat hammer
```

---

## 🎯 دستور کامل برای تست

```cmd
# رفتن به مسیر پروژه
cd C:\Users\trade\Documents\PythonProject\New

# فعال کردن venv
venv\Scripts\activate

# تست الگوی Doji
python signal_generation\tests\test_pattern.py --pattern doji --data-dir historical\BTC-USDT

# تست الگوی Hammer
python signal_generation\tests\test_pattern.py --pattern hammer --data-dir historical\BTC-USDT

# تست الگوی Engulfing
python signal_generation\tests\test_pattern.py --pattern engulfing --data-dir historical\BTC-USDT
```

---

## 🐛 اگر هنوز مشکل دارید

### خطا: ModuleNotFoundError: No module named 'signal_generation'

```cmd
# چک کنید که در مسیر صحیح هستید:
cd
# باید چاپ شود: C:\Users\trade\Documents\PythonProject\New

# چک کنید که venv فعال است:
where python
# باید نشان دهد: ...\New\venv\Scripts\python.exe

# اگر نیاز بود، PYTHONPATH را set کنید:
set PYTHONPATH=C:\Users\trade\Documents\PythonProject\New
```

### خطا: FileNotFoundError: Data not found

```cmd
# چک کنید که فایل‌های داده وجود دارند:
dir historical\BTC-USDT
# باید ببینید: 5m.csv, 15m.csv, 1h.csv, 4h.csv
```

---

## 📝 مثال کامل اجرا

```cmd
C:\Users\trade\Documents\PythonProject\New>venv\Scripts\activate

(venv) C:\Users\trade\Documents\PythonProject\New>python signal_generation\tests\test_pattern.py --pattern doji --data-dir historical\BTC-USDT

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
...
```

---

## 🔧 نصب Dependencies (اگر لازم بود)

```cmd
venv\Scripts\activate
pip install pandas numpy scipy TA-Lib
```

---

## 💡 نکته مهم

این فایل **نه** یک unittest است، بلکه یک **CLI tool** است که باید با arguments اجرا شود:

❌ **اشتباه:** اجرا به عنوان test در PyCharm
✅ **درست:** اجرا از Terminal با `--pattern` argument

---

## 📞 اگر همچنان مشکل دارید

Screenshot از:
1. Terminal که در آن command را اجرا می‌کنید
2. خطایی که می‌بینید

بفرستید تا دقیق‌تر کمک کنم.
