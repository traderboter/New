# 📋 راهنمای تست الگوها - Pattern Testing Guide

این راهنما برای تست منظم و مرحله‌ای تمام الگوهای تشخیص شده است.

---

## 🎯 هدف

تست تک‌تک الگوها روی داده‌های واقعی BTC-USDT در 4 تایم‌فریم (5m, 15m, 1h, 4h) برای:
- ✅ اطمینان از تشخیص صحیح
- ✅ بررسی دقت و نرخ تشخیص
- ✅ شناسایی مشکلات احتمالی
- ✅ بهبود پارامترها

---

## 🚀 نحوه استفاده

### 1. تست یک الگوی خاص:

```bash
cd /home/user/New

# تست الگوی Doji
python signal_generation/tests/test_pattern.py --pattern doji --data-dir historical/BTC-USDT

# تست الگوی Hammer
python signal_generation/tests/test_pattern.py --pattern hammer --data-dir historical/BTC-USDT

# تست الگوی Engulfing
python signal_generation/tests/test_pattern.py --pattern engulfing --data-dir historical/BTC-USDT
```

### 2. پارامترها:

```bash
--pattern <name>      # نام الگو (ضروری)
--data-dir <path>     # مسیر داده‌ها (پیش‌فرض: historical/BTC-USDT)
```

---

## 📊 خروجی تست

هر تست شامل:

### 1. **جزئیات تشخیص** برای هر تایم‌فریم:
```
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
     Metadata: {...}

     Context (index 1250):
     Date                 Open       High       Low        Close      Pattern
     ---------------------------------------------------------------------------
     2024-02-15 08:00     50234.50   50456.20   50123.40   50345.60
     2024-02-15 09:00     50345.60   50567.80   50234.50   50456.20
     2024-02-15 10:00     50456.20   50678.90   50345.60   50567.80   <<<
     2024-02-15 11:00     50567.80   50789.00   50456.20   50678.90
```

### 2. **خلاصه کلی**:
```
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

💡 Analysis:
  ✓ Pattern detected successfully across timeframes
  Total: 1482 detections in 57670 candles
```

---

## 📝 چک‌لیست تست الگوها

### 🟢 Candlestick Patterns (16 الگو)

#### Bullish Patterns:
- [ ] **Hammer**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern hammer
  ```
  - انتظار: تشخیص در پایین روند نزولی
  - سایه پایینی بلند، بدنه کوچک

- [ ] **Inverted Hammer**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "inverted hammer"
  ```
  - انتظار: تشخیص در پایین روند نزولی
  - سایه بالایی بلند، بدنه کوچک

- [ ] **Engulfing** (Bullish)
  ```bash
  python signal_generation/tests/test_pattern.py --pattern engulfing
  ```
  - انتظار: کندل سبز بزرگ که کندل قبلی را می‌پوشاند

- [ ] **Morning Star**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "morning star"
  ```
  - انتظار: 3 کندل، الگوی بازگشت صعودی

- [ ] **Piercing Line**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "piercing line"
  ```
  - انتظار: 2 کندل، بازگشت صعودی

- [ ] **Three White Soldiers**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "three white soldiers"
  ```
  - انتظار: 3 کندل صعودی متوالی

- [ ] **Morning Doji Star**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "morning doji star"
  ```
  - انتظار: 3 کندل با doji در وسط

#### Bearish Patterns:
- [ ] **Shooting Star**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "shooting star"
  ```
  - انتظار: در بالای روند صعودی
  - سایه بالایی بلند

- [ ] **Hanging Man**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "hanging man"
  ```
  - انتظار: در بالای روند صعودی
  - سایه پایینی بلند

- [ ] **Evening Star**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "evening star"
  ```
  - انتظار: 3 کندل، بازگشت نزولی

- [ ] **Dark Cloud Cover**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "dark cloud cover"
  ```
  - انتظار: 2 کندل، بازگشت نزولی

- [ ] **Three Black Crows**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "three black crows"
  ```
  - انتظار: 3 کندل نزولی متوالی

- [ ] **Evening Doji Star**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "evening doji star"
  ```
  - انتظار: 3 کندل با doji در وسط

#### Reversal Patterns:
- [ ] **Doji**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern doji
  ```
  - انتظار: open ≈ close
  - نشان‌دهنده تردید بازار

- [ ] **Harami**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern harami
  ```
  - انتظار: کندل کوچک داخل بدنه کندل قبلی

- [ ] **Harami Cross**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "harami cross"
  ```
  - انتظار: doji داخل بدنه کندل قبلی

---

### 🟢 Chart Patterns (4 الگو)

- [ ] **Double Top/Bottom**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "double top bottom"
  ```
  - انتظار: دو قله یا دو دره در سطح مشابه

- [ ] **Head and Shoulders**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern "head shoulders"
  ```
  - انتظار: 3 قله، وسطی بلندتر (سر و شانه)

- [ ] **Triangle**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern triangle
  ```
  - انتظار: خطوط همگرا (صعودی، نزولی، متقارن)

- [ ] **Wedge**
  ```bash
  python signal_generation/tests/test_pattern.py --pattern wedge
  ```
  - انتظار: خطوط همگرا صعودی یا نزولی

---

## 🔍 معیارهای ارزیابی

### ✅ الگو صحیح است اگر:

1. **تشخیص در شرایط مناسب**:
   - Hammer/Inverted Hammer → در پایین روند نزولی
   - Shooting Star/Hanging Man → در بالای روند صعودی
   - Engulfing → در نقاط بازگشت

2. **نرخ تشخیص منطقی**:
   - الگوهای معمولی (Doji): 2-5%
   - الگوهای نادر (Three White Soldiers): 0.1-1%
   - الگوهای Chart: 0.05-0.5%

3. **Context صحیح**:
   - کندل‌های قبل و بعد منطقی هستند
   - الگو در موقعیت مناسب قیمت

### ⚠️ مشکلات احتمالی:

1. **تشخیص ندادن** (0 detection):
   - پارامترها خیلی strict
   - مشکل در کد تشخیص
   - الگو در این دوره وجود ندارد

2. **تشخیص بیش از حد** (>10%):
   - پارامترها خیلی loose
   - False positives زیاد
   - نیاز به بهبود دقت

3. **تشخیص نامناسب**:
   - الگو در context اشتباه تشخیص داده شده
   - نیاز به اضافه شدن context checking

---

## 📊 ثبت نتایج

برای هر الگو، نتایج را در جدول زیر ثبت کنید:

| الگو | 5m | 15m | 1h | 4h | وضعیت | یادداشت |
|------|----|----|----|----|-------|---------|
| Doji | ✅ 2.5% | ✅ 2.5% | ✅ 2.7% | ✅ 2.6% | ✅ OK | - |
| Hammer | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

### علائم وضعیت:
- ✅ **OK**: تشخیص صحیح، نرخ منطقی
- ⚠️ **Warning**: نرخ پایین یا بالا، نیاز به بررسی
- ❌ **Error**: مشکل در تشخیص، نیاز به رفع باگ

---

## 🐛 رفع مشکل

### اگر الگویی تشخیص داده نمی‌شود:

1. **بررسی کد الگو**:
   ```bash
   # نگاه کنید به فایل الگو
   cat signal_generation/analyzers/patterns/candlestick/doji.py
   ```

2. **بررسی پارامترها**:
   - threshold ها خیلی strict نیست؟
   - شرایط تشخیص منطقی است؟

3. **تست با داده‌های دستی**:
   - یک کندل doji معمولی را دستی بسازید
   - ببینید آیا تشخیص می‌دهد

### اگر تشخیص بیش از حد است:

1. **بررسی شرایط**:
   - آیا context checking وجود دارد؟
   - آیا threshold کافی است؟

2. **بهبود دقت**:
   - اضافه کردن شرط trend
   - اضافه کردن شرط volume
   - تنظیم threshold ها

---

## 💡 نکات مهم

### 1. تست منظم:
- یک الگو در هر مرحله
- نتایج را ثبت کنید
- قبل از رفتن به الگوی بعدی، مشکلات را رفع کنید

### 2. مقایسه با منابع معتبر:
- نمونه‌های واقعی از TradingView
- کتاب‌های تحلیل تکنیکال
- نظرات متخصصین

### 3. داده‌های تست:
- داده‌های واقعی BTC-USDT
- دوره زمانی کافی (حداقل 1 سال)
- چند تایم‌فریم مختلف

---

## 📧 گزارش مشکل

اگر مشکلی یافتید:

1. نام الگو
2. تایم‌فریم
3. نتایج تست (نرخ تشخیص)
4. نمونه‌های اشتباه (اگر وجود دارد)
5. خروجی کامل تست

---

## ✅ پس از تست موفق

وقتی همه الگوها تست شدند:
- [ ] همه 16 الگوی candlestick ✅
- [ ] همه 4 الگوی chart ✅
- [ ] نرخ تشخیص منطقی
- [ ] Context صحیح
- [ ] آماده برای production

---

**تاریخ**: 2025-10-24
**نسخه**: 1.0
**وضعیت**: آماده برای تست
