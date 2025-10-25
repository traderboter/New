# TA-Lib Pattern Lookback Requirements - کشفیات جامع

تاریخ: 2025-10-25
هدف: تعیین تعداد کندل‌های مورد نیاز برای **تمام** الگوهای TA-Lib

---

## 📊 خلاصه نتایج

**تست انجام شده:** 16 الگو
**داده:** 10,543 کندل BTC/USDT 1-hour

### نتیجه کلیدی: ⚠️

> **هیچ الگویی با ۱ کندل کار نمی‌کند!**
>
> **همه** الگوها نیاز به کندل‌های قبلی دارند تا TA-Lib بتواند آنها را تشخیص دهد.

---

## 📋 جدول کامل نتایج

| الگو | تعداد Detection | نرخ | حداقل Lookback | کل کندل لازم | وضعیت |
|------|----------------|-----|----------------|--------------|-------|
| **Shooting Star** | 75 | 0.71% | **11** | **12** | ✅ Fixed |
| **Hammer** | 277 | 2.63% | **11** | **12** | ✅ Fixed |
| **Inverted Hammer** | 59 | 0.56% | **11** | **12** | ⚠️ نیاز به Fix |
| **Hanging Man** | 187 | 1.77% | **11** | **12** | ⚠️ نیاز به Fix |
| **Doji** | 1,485 | 14.09% | **10** | **11** | ⚠️ نیاز به Fix |
| **Engulfing** | 1,714 | 16.26% | **2** | **3** | ⚠️ نیاز به Fix |
| **Piercing Line** | 3 | 0.03% | **11** | **12** | ⚠️ نیاز به Fix |
| **Dark Cloud Cover** | 3 | 0.03% | **11** | **12** | ⚠️ نیاز به Fix |
| **Harami** | 765 | 7.26% | **11** | **12** | ⚠️ نیاز به Fix |
| **Harami Cross** | 147 | 1.39% | **11** | **12** | ⚠️ نیاز به Fix |
| **Morning Star** | 40 | 0.38% | **12** | **13** | ⚠️ نیاز به Fix |
| **Evening Star** | 49 | 0.46% | **12** | **13** | ⚠️ نیاز به Fix |
| **Morning Doji Star** | 12 | 0.11% | **12** | **13** | ⚠️ نیاز به Fix |
| **Evening Doji Star** | 12 | 0.11% | **12** | **13** | ⚠️ نیاز به Fix |
| **3 White Soldiers** | 9 | 0.09% | **12** | **13** | ⚠️ نیاز به Fix |
| **3 Black Crows** | 1 | 0.01% | **15** | **16** | ⚠️ نیاز به Fix |

---

## 🔬 دسته‌بندی بر اساس تعداد کندل

### گروه 1: کمترین نیاز (3 کندل)
```
✅ Engulfing: 3 candles (lookback=2)
   - Detection rate: 16.26% (بیشترین!)
   - الگوی دو کندلی
```

### گروه 2: نیاز متوسط (11 کندل)
```
✅ Doji: 11 candles (lookback=10)
   - Detection rate: 14.09% (دومین بیشترین!)
```

### گروه 3: نیاز استاندارد (12 کندل)
```
الگوهای تک کندلی و دو کندلی:
✅ Shooting Star: 12 candles (lookback=11) - FIXED ✅
✅ Hammer: 12 candles (lookback=11) - FIXED ✅
⚠️ Inverted Hammer: 12 candles (lookback=11)
⚠️ Hanging Man: 12 candles (lookback=11)
⚠️ Piercing Line: 12 candles (lookback=11)
⚠️ Dark Cloud Cover: 12 candles (lookback=11)
⚠️ Harami: 12 candles (lookback=11)
⚠️ Harami Cross: 12 candles (lookback=11)
```

### گروه 4: نیاز بیشتر (13 کندل)
```
الگوهای سه کندلی (Star patterns):
⚠️ Morning Star: 13 candles (lookback=12)
⚠️ Evening Star: 13 candles (lookback=12)
⚠️ Morning Doji Star: 13 candles (lookback=12)
⚠️ Evening Doji Star: 13 candles (lookback=12)
⚠️ 3 White Soldiers: 13 candles (lookback=12)
```

### گروه 5: بیشترین نیاز (16 کندل)
```
⚠️ 3 Black Crows: 16 candles (lookback=15)
   - Detection rate: 0.01% (نادرترین!)
```

---

## 💡 یافته‌های مهم

### ✅ الگوهای اصلاح شده (2 مورد):
1. **Shooting Star** ✅ - v2.0.0 (Fixed: 2025-10-25)
2. **Hammer** ✅ - v2.0.0 (Fixed: 2025-10-25)

### ⚠️ الگوهای نیاز به اصلاح (14 مورد):

**الویت VERY HIGH (Detection rate بالا):**
1. **Engulfing** - 16.26% detection rate, نیاز به 3 کندل
2. **Doji** - 14.09% detection rate, نیاز به 11 کندل
3. **Harami** - 7.26% detection rate, نیاز به 12 کندل

**الویت HIGH:**
4. **Hanging Man** - 1.77% detection rate, نیاز به 12 کندل
5. **Harami Cross** - 1.39% detection rate, نیاز به 12 کندل
6. **Inverted Hammer** - 0.56% detection rate, نیاز به 12 کندل

**الویت MEDIUM:**
7. **Evening Star** - 0.46% detection rate, نیاز به 13 کندل
8. **Morning Star** - 0.38% detection rate, نیاز به 13 کندل

**الویت LOW (detection rate خیلی کم):**
9. **Morning Doji Star** - 0.11% detection rate, نیاز به 13 کندل
10. **Evening Doji Star** - 0.11% detection rate, نیاز به 13 کندل
11. **3 White Soldiers** - 0.09% detection rate, نیاز به 13 کندل
12. **Piercing Line** - 0.03% detection rate, نیاز به 12 کندل
13. **Dark Cloud Cover** - 0.03% detection rate, نیاز به 12 کندل
14. **3 Black Crows** - 0.01% detection rate, نیاز به 16 کندل

---

## 🔧 راه‌حل: چگونه باید اصلاح کنیم؟

### الگوی کلی (برای همه الگوها):

#### ❌ کد فعلی (احتمالاً اشتباه):
```python
def detect(self, df: pd.DataFrame, ...) -> bool:
    if not self._validate_dataframe(df):
        return False

    # فقط کندل آخر - اشتباه!
    df_tail = df.tail(1)
    pattern = talib.CDLXXX(
        df_tail['open'].values,
        df_tail['high'].values,
        df_tail['low'].values,
        df_tail['close'].values
    )
    return pattern[-1] != 0
```

#### ✅ کد درست:
```python
def detect(self, df: pd.DataFrame, ...) -> bool:
    if not self._validate_dataframe(df):
        return False

    # چک تعداد کندل مورد نیاز (بسته به الگو)
    MIN_CANDLES = 12  # یا 11، 13، 16 بسته به الگو
    if len(df) < MIN_CANDLES:
        return False

    # دادن کندل‌های کافی به TA-Lib
    df_tail = df.tail(100)  # یا df بدون tail

    pattern = talib.CDLXXX(
        df_tail['open'].values,
        df_tail['high'].values,
        df_tail['low'].values,
        df_tail['close'].values
    )

    # فقط کندل آخر را بررسی می‌کنیم
    return pattern[-1] != 0
```

### تعداد کندل مورد نیاز برای هر الگو:

```python
PATTERN_MIN_CANDLES = {
    'CDLENGULFING': 3,           # ✅ کمترین
    'CDLDOJI': 11,

    # الگوهای 12 کندلی (اکثریت)
    'CDLSHOOTINGSTAR': 12,       # ✅ Fixed
    'CDLHAMMER': 12,             # ✅ Fixed
    'CDLINVERTEDHAMMER': 12,
    'CDLHANGINGMAN': 12,
    'CDLPIERCING': 12,
    'CDLDARKCLOUDCOVER': 12,
    'CDLHARAMI': 12,
    'CDLHARAMICROSS': 12,

    # الگوهای 13 کندلی (Star patterns)
    'CDLMORNINGSTAR': 13,
    'CDLEVENINGSTAR': 13,
    'CDLMORNINGDOJISTAR': 13,
    'CDLEVENINGDOJISTAR': 13,
    'CDL3WHITESOLDIERS': 13,

    # بیشترین
    'CDL3BLACKCROWS': 16,        # ✅ بیشترین نیاز
}
```

---

## 📈 آمار جالب

### Detection Rate:
```
بیشترین detection:
1. Engulfing: 16.26%
2. Doji: 14.09%
3. Harami: 7.26%

کمترین detection:
1. 3 Black Crows: 0.01%
2. Piercing Line: 0.03%
3. Dark Cloud Cover: 0.03%
```

### ارتباط بین Complexity و Lookback:
```
Simple patterns (1-2 candles): 11-12 candles lookback
Complex patterns (3 candles): 13-16 candles lookback

Exception: Engulfing (2-candle) فقط 3 کندل نیاز دارد!
```

---

## 🎯 پلان اصلاح

### مرحله 1: Fix High-Priority Patterns (3 الگو)
```
☐ Engulfing (16.26% - بیشترین impact)
☐ Doji (14.09%)
☐ Harami (7.26%)
```

### مرحله 2: Fix Medium-Priority Patterns (3 الگو)
```
☐ Hanging Man (1.77%)
☐ Harami Cross (1.39%)
☐ Inverted Hammer (0.56%)
```

### مرحله 3: Fix Star Patterns (4 الگو)
```
☐ Evening Star (0.46%)
☐ Morning Star (0.38%)
☐ Morning Doji Star (0.11%)
☐ Evening Doji Star (0.11%)
```

### مرحله 4: Fix Rare Patterns (4 الگو)
```
☐ 3 White Soldiers (0.09%)
☐ Piercing Line (0.03%)
☐ Dark Cloud Cover (0.03%)
☐ 3 Black Crows (0.01%)
```

---

## ✅ چک‌لیست اصلاح هر الگو

برای هر pattern فایل:

1. ☐ بررسی کد فعلی `detect()` method
2. ☐ تعیین `MIN_CANDLES` مناسب
3. ☐ اضافه کردن `if len(df) < MIN_CANDLES: return False`
4. ☐ تغییر `df.tail(1)` به `df.tail(100)`
5. ☐ تست با BTC data
6. ☐ بررسی detection rate (باید با جدول بالا مطابقت داشته باشد)
7. ☐ Update version number
8. ☐ Update docstring
9. ☐ Commit

---

## 🚀 مرحله بعدی

**آیا می‌خواهید:**

1. ✅ شروع اصلاح از الگوی با بیشترین detection rate؟ (Engulfing)
2. ✅ یک‌به‌یک همه الگوها را اصلاح کنیم؟
3. ✅ یک اسکریپت automation بسازیم؟

---

## 📝 نتیجه‌گیری

> **کشف کلیدی:**
>
> مشکل در **همه** الگوهای TA-Lib وجود داشت، نه فقط Shooting Star و Hammer!
>
> TA-Lib به کندل‌های قبلی برای context نیاز دارد تا بتواند الگوها را با دقت تشخیص دهد.
>
> با اصلاح همه الگوها، detection rate از **~0%** به **نرخ واقعی** افزایش می‌یابد!

**تعداد الگوهای نیاز به اصلاح:** 14
**قبل از fix:** احتمالاً 0 detection برای همه
**بعد از fix:** مجموع ~4,500 detection در 10,543 کندل (42.7%)

---

**ایجاد شده توسط:** test_all_patterns_lookback.py
**تاریخ:** 2025-10-25
**داده تست:** BTC/USDT 1-hour (10,543 candles)
