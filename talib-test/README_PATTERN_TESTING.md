# راهنمای تست الگوهای کندلی 🔬

این راهنما نحوه تست minimum candles requirement برای همه الگوهای کندلی را توضیح می‌دهد.

## 📋 وضعیت فعلی الگوها

### ✅ الگوهای تست شده با Recency Scoring v3.0.0:
1. **Hammer** - min_candles: 12, lookback: 11
2. **Shooting Star** - min_candles: 12, lookback: 11
3. **Engulfing** - min_candles: 3, lookback: 2
4. **Doji** - min_candles: 11, lookback: 10
5. **Morning Star** - min_candles: 13, lookback: 12
6. **Inverted Hammer** - min_candles: 12, lookback: 11

### ❓ الگوهای نیاز به تست (10 الگو):
1. **Dark Cloud Cover** - TA-Lib: CDLDARKCLOUDCOVER
2. **Evening Star** - TA-Lib: CDLEVENINGSTAR
3. **Evening Doji Star** - TA-Lib: CDLEVENINGDOJISTAR
4. **Harami** - TA-Lib: CDLHARAMI
5. **Harami Cross** - TA-Lib: CDLHARAMICROSS
6. **Hanging Man** - TA-Lib: CDLHANGINGMAN
7. **Piercing Line** - TA-Lib: CDLPIERCING
8. **Morning Doji Star** - TA-Lib: CDLMORNINGDOJISTAR
9. **Three White Soldiers** - TA-Lib: CDL3WHITESOLDIERS
10. **Three Black Crows** - TA-Lib: CDL3BLACKCROWS

---

## 🚀 نحوه اجرای تست‌ها

### روش 1: تست تک‌تک الگوها

برای تست یک الگو به صورت جداگانه:

```bash
cd /home/user/New/talib-test

# ویرایش فایل و تغییر الگو
# PATTERN_TO_TEST = "DARKCLOUDCOVER"  # یا هر الگوی دیگر

python3 test_pattern_with_real_data.py
```

**الگوهای قابل تست:**
- `DARKCLOUDCOVER` - Dark Cloud Cover
- `EVENINGSTAR` - Evening Star
- `EVENINGDOJISTAR` - Evening Doji Star
- `HARAMI` - Harami
- `HARAMICROSS` - Harami Cross
- `HANGINGMAN` - Hanging Man
- `PIERCINGLINE` - Piercing Line
- `MORNINGDOJISTAR` - Morning Doji Star
- `THREEWHITESOLDIERS` - Three White Soldiers
- `THREEBLACKCROWS` - Three Black Crows

### روش 2: تست همه الگوها به صورت خودکار

```bash
cd /home/user/New/talib-test
python3 test_all_patterns_batch.py
```

این اسکریپت:
- همه 16 الگو را تست می‌کند
- نتایج را در `pattern_test_results.json` ذخیره می‌کند
- یک گزارش خلاصه نمایش می‌دهد

---

## 📊 نحوه تفسیر نتایج

بعد از اجرای تست، برای هر الگو این اطلاعات به دست می‌آید:

```
✅ حداقل lookback: 11
✅ حداقل کل کندل: 12
💡 این الگو به 11 کندل قبلی نیاز دارد
```

### مثال:
اگر یک الگو نیاز به **minimum 13 candles** دارد:
- `min_candles = 13` → حداقل تعداد کندل برای TA-Lib
- `lookback_window = 12` → تعداد کندل قبلی (13 - 1 = 12)

---

## 🔧 پیاده‌سازی Recency Scoring برای الگوهای جدید

بعد از تست هر الگو و پیدا کردن `min_candles` و `lookback_window`:

### گام 1: به‌روزرسانی version به 3.0.0

```python
PATTERN_VERSION = "3.0.0"
```

### گام 2: تغییر متد `detect()`

```python
def detect(self, df, ...):
    # Reset detection cache
    self._last_detection_candles_ago = None

    # Check minimum candles
    if len(df) < MIN_CANDLES:  # از نتایج تست
        return False

    try:
        # Run TA-Lib
        result = talib.CDLPATTERN(...)

        # NEW v3.0.0: Check last N candles
        lookback = min(self.lookback_window, len(result))

        for i in range(lookback):
            idx = -(i + 1)
            if result[idx] != 0:
                # Additional checks if needed (trend, etc.)

                # Store position
                self._last_detection_candles_ago = i
                return True

        return False
    except:
        return False
```

### گام 3: به‌روزرسانی `_get_detection_details()`

```python
def _get_detection_details(self, df):
    # Get detection position
    candles_ago = getattr(self, '_last_detection_candles_ago', 0)
    if candles_ago is None:
        candles_ago = 0

    # Get recency multiplier
    if candles_ago < len(self.recency_multipliers):
        recency_multiplier = self.recency_multipliers[candles_ago]
    else:
        recency_multiplier = 0.0

    # Get detected candle
    candle_idx = -(candles_ago + 1)
    detected_candle = df.iloc[candle_idx]

    # Calculate base confidence
    base_confidence = ...  # معمولاً 0.7-0.85

    # Adjust with recency
    adjusted_confidence = min(base_confidence * recency_multiplier, 0.95)

    return {
        'location': 'current' if candles_ago == 0 else 'recent',
        'candles_ago': candles_ago,
        'recency_multiplier': recency_multiplier,
        'confidence': adjusted_confidence,
        'metadata': {
            ...
            'recency_info': {
                'candles_ago': candles_ago,
                'multiplier': recency_multiplier,
                'lookback_window': self.lookback_window,
                'base_confidence': base_confidence,
                'adjusted_confidence': adjusted_confidence
            }
        }
    }
```

---

## 📝 نکات مهم

### 1. الگوهای بدون TA-Lib

برخی الگوها از TA-Lib استفاده نمی‌کنند (مثل Doji که detector دستی دارد).
برای این الگوها:
- نیازی به تست TA-Lib minimum candles نیست
- باید lookback_window را به صورت دستی تعیین کنیم
- معمولاً 5-10 کندل کافی است

### 2. الگوهای 3-کندلی

الگوهای مثل Morning Star که 3 کندل دارند:
- Index calculation دقیق‌تر است
- باید از `candle_idx - 2` برای کندل اول استفاده کنیم

```python
# Morning Star: 3-candle pattern
candle_idx = -(candles_ago + 1)  # کندل سوم (completion)
first_candle = df.iloc[candle_idx - 2]   # کندل اول
star_candle = df.iloc[candle_idx - 1]    # کندل دوم (star)
last_candle = df.iloc[candle_idx]        # کندل سوم
```

### 3. پیشنهاد Recency Multipliers

بر اساس قدرت الگو:

**الگوهای قوی (decay کند):**
```python
recency_multipliers = [1.0, 0.95, 0.85, 0.75, 0.6, 0.4]
```
مثال: Engulfing, Morning Star, Evening Star, Three White Soldiers

**الگوهای متوسط (decay معمولی):**
```python
recency_multipliers = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
```
مثال: Hammer, Shooting Star, Inverted Hammer, Piercing Line

**الگوهای ضعیف (decay سریع):**
```python
recency_multipliers = [1.0, 0.7, 0.5, 0.3, 0.15, 0.05]
```
مثال: Doji, Harami

---

## 📂 ساختار فایل‌ها

```
talib-test/
├── test_pattern_with_real_data.py      # تست تک‌تک (دستی)
├── test_all_patterns_batch.py          # تست همه (خودکار)
├── pattern_test_results.json           # نتایج تست (خروجی)
└── README_PATTERN_TESTING.md           # این راهنما
```

---

## 🎯 چک‌لیست کامل

برای هر الگوی جدید:

- [ ] تست الگو با `test_pattern_with_real_data.py`
- [ ] یادداشت `min_candles` و `lookback_window`
- [ ] به‌روزرسانی version به 3.0.0
- [ ] پیاده‌سازی multi-candle lookback در `detect()`
- [ ] به‌روزرسانی `_get_detection_details()` با recency info
- [ ] انتخاب `recency_multipliers` مناسب
- [ ] تست با داده واقعی
- [ ] Commit و Push

---

## 🚨 نیازمندی‌ها

برای اجرای تست‌ها نیاز است:

1. **TA-Lib** نصب باشد:
   ```bash
   pip install TA-Lib
   ```

2. **داده BTC** موجود باشد:
   ```
   /home/user/New/historical/BTC-USDT/1hour.csv
   ```

3. **Python packages:**
   ```bash
   pip install pandas numpy
   ```

---

تاریخ: 2025-10-25
نسخه: 1.0.0
