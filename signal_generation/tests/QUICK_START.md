# 🚀 شروع سریع تست الگوها

## 📁 ساختار داده

مطمئن شوید داده‌های BTC-USDT در این مسیر وجود دارد:
```
New/historical/BTC-USDT/
├── 5m.csv
├── 15m.csv
├── 1h.csv
└── 4h.csv
```

---

## ⚡ تست یک الگو (توصیه شده برای شروع)

### مثال 1: تست الگوی Doji
```bash
cd /home/user/New
python signal_generation/tests/test_pattern.py --pattern doji
```

### مثال 2: تست الگوی Hammer
```bash
python signal_generation/tests/test_pattern.py --pattern hammer
```

### مثال 3: تست الگوی Engulfing
```bash
python signal_generation/tests/test_pattern.py --pattern engulfing
```

---

## 📝 خروجی نمونه

```
================================================================================
🔍 Testing DOJI on 1h
================================================================================
✓ Loaded 8760 candles
  Period: 2024-01-01 00:00:00 to 2024-12-31 23:00:00

✓ Pattern registered: DojiPattern

🔎 Scanning for doji patterns...
✓ Found 234 doji patterns

📊 Pattern Detections (234):
================================================================================

  #1. Doji
     Direction: reversal
     Strength: 1/3
     Confidence: 0.85

     Context (index 1250):
     Date                 Open       High       Low        Close      Pattern
     ---------------------------------------------------------------------------
     2024-02-15 08:00     50234.50   50456.20   50123.40   50345.60
     2024-02-15 09:00     50345.60   50567.80   50234.50   50456.20
     2024-02-15 10:00     50456.20   50678.90   50345.60   50567.80   <<<
     2024-02-15 11:00     50567.80   50789.00   50456.20   50678.90

📋 Summary Report - DOJI
================================================================================

Timeframe    Candles      Detections      Rate         Status
--------------------------------------------------------------------------------
5m           35040        892             2.546%       ✓ OK
15m          11680        298             2.551%       ✓ OK
1h           8760         234             2.671%       ✓ OK
4h           2190         58              2.648%       ✓ OK
--------------------------------------------------------------------------------
TOTAL        57670        1482            2.569%
```

---

## 📊 لیست الگوها برای تست

### 🟢 Candlestick Patterns (16 الگو):

**Bullish:**
1. `hammer` - چکش
2. `inverted hammer` - چکش وارونه
3. `engulfing` - پوشش دهنده
4. `morning star` - ستاره صبحگاهی
5. `piercing line` - خط نافذ
6. `three white soldiers` - سه سرباز سفید
7. `morning doji star` - ستاره دوجی صبحگاهی

**Bearish:**
8. `shooting star` - ستاره دنباله‌دار
9. `hanging man` - مرد به دار آویخته
10. `evening star` - ستاره عصرگاهی
11. `dark cloud cover` - پوشش ابر تیره
12. `three black crows` - سه کلاغ سیاه
13. `evening doji star` - ستاره دوجی عصرگاهی

**Reversal:**
14. `doji` - دوجی
15. `harami` - هارامی
16. `harami cross` - هارامی متقاطع

### 📈 Chart Patterns (4 الگو):

17. `double top bottom` - سقف/کف دوقلو
18. `head shoulders` - سر و شانه
19. `triangle` - مثلث
20. `wedge` - گوه

---

## 🎯 پیشنهاد ترتیب تست

### مرحله 1: الگوهای ساده (شروع کنید با اینها)
```bash
python signal_generation/tests/test_pattern.py --pattern doji
python signal_generation/tests/test_pattern.py --pattern hammer
python signal_generation/tests/test_pattern.py --pattern "shooting star"
```

### مرحله 2: الگوهای دو کندلی
```bash
python signal_generation/tests/test_pattern.py --pattern engulfing
python signal_generation/tests/test_pattern.py --pattern harami
python signal_generation/tests/test_pattern.py --pattern "piercing line"
python signal_generation/tests/test_pattern.py --pattern "dark cloud cover"
```

### مرحله 3: الگوهای سه کندلی
```bash
python signal_generation/tests/test_pattern.py --pattern "morning star"
python signal_generation/tests/test_pattern.py --pattern "evening star"
python signal_generation/tests/test_pattern.py --pattern "three white soldiers"
python signal_generation/tests/test_pattern.py --pattern "three black crows"
```

### مرحله 4: Chart patterns
```bash
python signal_generation/tests/test_pattern.py --pattern "double top bottom"
python signal_generation/tests/test_pattern.py --pattern triangle
python signal_generation/tests/test_pattern.py --pattern wedge
python signal_generation/tests/test_pattern.py --pattern "head shoulders"
```

---

## 🔍 چه چیزی را بررسی کنید

### ✅ الگو صحیح است:
- تعداد تشخیص منطقی است (نه صفر، نه خیلی زیاد)
- Context کندل‌ها درست است
- در موقعیت مناسب تشخیص داده شده

### ⚠️ نیاز به بررسی:
- هیچ تشخیصی نداریم (0 detection)
- تشخیص خیلی زیاد (>10%)
- Context نامناسب

### ❌ مشکل دارد:
- Error در تشخیص
- تشخیص در جای اشتباه
- کندل‌های نادرست

---

## 💡 نکات مهم

1. **یک الگو در هر مرحله**: تمرکز روی یک الگو، بررسی کامل، سپس الگوی بعدی

2. **بررسی نتایج**:
   - نگاه کنید به context کندل‌ها
   - ببینید آیا منطقی است؟
   - با TradingView مقایسه کنید

3. **ثبت مشکلات**:
   - اگر مشکلی دیدید، یادداشت کنید
   - نام الگو + تایم‌فریم + توضیح مشکل

4. **صبور باشید**:
   - تست دقیق زمان می‌برد
   - هر الگو را کامل بررسی کنید
   - عجله نکنید

---

## 🐛 اگر مشکلی بود

### مشکل 1: فایل داده پیدا نمی‌شود
```
Error: File not found: historical/BTC-USDT/5m.csv
```
**حل**: مطمئن شوید در مسیر `/home/user/New` هستید

### مشکل 2: الگو پیدا نمی‌شود
```
Error: Pattern not found: xyz
```
**حل**: نام الگو را از لیست بالا چک کنید

### مشکل 3: Import error
```
ModuleNotFoundError: No module named 'signal_generation'
```
**حل**: PYTHONPATH را set کنید:
```bash
export PYTHONPATH=/home/user/New:$PYTHONPATH
python signal_generation/tests/test_pattern.py --pattern doji
```

---

## 📞 بعد از تست

نتایج را با این فرمت به من بدهید:

```
الگو: Doji
تایم‌فریم‌ها: 5m, 15m, 1h, 4h
نرخ تشخیص: 2.5%, 2.5%, 2.7%, 2.6%
وضعیت: ✅ OK / ⚠️ Warning / ❌ Error
یادداشت: (اگر مشکلی بود توضیح دهید)
```

---

## ✨ شروع کنید!

```bash
cd /home/user/New
python signal_generation/tests/test_pattern.py --pattern doji
```

**موفق باشید! 🚀**
