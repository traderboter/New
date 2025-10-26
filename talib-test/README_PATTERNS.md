# راهنمای تست الگوهای کندل استیک

این پوشه شامل اسکریپت‌های تست برای 26 الگوی کندل استیک است.

## فایل‌های موجود

### 1. `test_pattern_with_real_data.py`
**تست تک الگو با داده واقعی**

این اسکریپت یک الگوی خاص را با داده واقعی BTC تست می‌کند.

**نحوه استفاده:**
```bash
# 1. الگوی مورد نظر را در خط 23 فایل تنظیم کنید:
PATTERN_TO_TEST = "MARUBOZU"  # یا هر الگوی دیگر

# 2. اجرای تست:
python3 test_pattern_with_real_data.py
```

**الگوهای قابل تست:**
- الگوهای قدیمی (16 الگو): HAMMER, DOJI, ENGULFING, SHOOTINGSTAR, و غیره
- الگوهای Phase 1 (5 الگو): MARUBOZU, DRAGONFLYDOJI, GRAVESTONEDOJI, SPINNINGTOP, LONGLEGGEDDOJI
- الگوهای Phase 2 (5 الگو): THREEINSIDE, THREEOUTSIDE, BELTHOLD, THREEMETHODS, MATHOLD

### 2. `test_all_patterns.py`
**تست خودکار همه 26 الگو**

این اسکریپت به صورت خودکار همه الگوها را تست می‌کند و گزارش کامل ارائه می‌دهد.

**نحوه استفاده:**
```bash
python3 test_all_patterns.py
```

**خروجی:**
- تعداد detections برای هر الگو
- حداقل lookback مورد نیاز
- گروه‌بندی بر اساس دسته (قدیمی، Phase 1، Phase 2)
- جدول کامل نتایج
- آمار کلی

### 3. `test_pattern_interactive.py`
**تست تعاملی با منوی انتخاب**

این اسکریپت یک منوی تعاملی ارائه می‌دهد که می‌توانید الگوی دلخواه را انتخاب کنید.

**نحوه استفاده:**
```bash
python3 test_pattern_interactive.py
```

**ویژگی‌ها:**
- منوی تعاملی برای انتخاب الگو
- گروه‌بندی الگوها بر اساس دسته
- تست کامل الگوی انتخابی
- امکان تست چندین الگو به صورت پی در پی

## الگوهای موجود (26 الگو)

### الگوهای قدیمی (16 الگو)
1. **Hammer** - Bullish reversal
2. **Inverted Hammer** - Bullish reversal
3. **Shooting Star** - Bearish reversal
4. **Hanging Man** - Bearish reversal
5. **Doji** - Indecision
6. **Engulfing** - Reversal (both directions)
7. **Harami** - Reversal (both directions)
8. **Harami Cross** - Reversal (both directions)
9. **Morning Star** - Bullish reversal
10. **Evening Star** - Bearish reversal
11. **Morning Doji Star** - Bullish reversal
12. **Evening Doji Star** - Bearish reversal
13. **Piercing Line** - Bullish reversal
14. **Dark Cloud Cover** - Bearish reversal
15. **Three White Soldiers** - Strong bullish continuation
16. **Three Black Crows** - Strong bearish continuation

### الگوهای Phase 1 - قدرتمند (5 الگو)
17. **Marubozu** - Strong continuation/reversal با کندل بدون سایه
18. **Dragonfly Doji** - Bullish reversal با سایه پایینی بلند
19. **Gravestone Doji** - Bearish reversal با سایه بالایی بلند
20. **Spinning Top** - Indecision با بدنه کوچک
21. **Long-Legged Doji** - Strong indecision با سایه‌های بلند

### الگوهای Phase 2 - ادامه و تایید (5 الگو)
22. **Three Inside Up/Down** - Harami + تایید (3-candle reversal)
23. **Three Outside Up/Down** - Engulfing + تایید (3-candle reversal)
24. **Belt Hold** - Strong single-candle reversal
25. **Rising/Falling Three Methods** - 5-candle continuation
26. **Mat Hold** - Bullish continuation با gap (5-candle)

## مثال‌های استفاده

### مثال 1: تست الگوی Marubozu
```bash
# روش 1: ویرایش test_pattern_with_real_data.py
# در خط 23 بنویسید: PATTERN_TO_TEST = "MARUBOZU"
python3 test_pattern_with_real_data.py

# روش 2: استفاده از اسکریپت تعاملی
python3 test_pattern_interactive.py
# سپس عدد الگوی Marubozu را انتخاب کنید
```

### مثال 2: تست همه الگوها
```bash
python3 test_all_patterns.py
```

این دستور همه 26 الگو را تست می‌کند و گزارش کامل ارائه می‌دهد.

### مثال 3: مقایسه الگوهای جدید با قدیمی
```bash
python3 test_all_patterns.py | grep -E "(Phase 1|Phase 2)"
```

## نکات مهم

1. **داده مورد نیاز:**
   - همه اسکریپت‌ها به فایل `../historical/BTC-USDT/1hour.csv` نیاز دارند
   - اطمینان حاصل کنید که این فایل موجود است

2. **وابستگی‌ها:**
   - pandas
   - numpy
   - talib (TA-Lib)

3. **حداقل Lookback:**
   - هر الگو به تعداد مشخصی کندل قبلی نیاز دارد
   - اسکریپت‌ها به صورت خودکار این مقدار را محاسبه می‌کنند
   - الگوهای تک کندلی (مثل Hammer): lookback = 0
   - الگوهای چند کندلی (مثل Three Inside): lookback >= 2

4. **نتایج:**
   - تعداد detections نشان می‌دهد الگو چقدر در داده واقعی پیدا می‌شود
   - الگوهایی با detection = 0 در داده فعلی یافت نشده‌اند (اما کد درست است)

## خروجی نمونه

```
🔬 تست خودکار همه الگوهای کندل استیک
================================================================================

📊 بارگذاری داده BTC...
✅ 8760 کندل بارگذاری شد

🔍 شروع تست الگوها...
--------------------------------------------------------------------------------
  Testing: Hammer                        (قدیمی    )... ✅   45 detections, min_lookback=0
  Testing: Marubozu                      (Phase 1  )... ✅  128 detections, min_lookback=0
  Testing: Three Inside Up/Down          (Phase 2  )... ✅   23 detections, min_lookback=2
  ...

📊 خلاصه نتایج
================================================================================

📁 Phase 1:
--------------------------------------------------------------------------------
  کل الگوها: 5
  موفق (با detection): 5
  میانگین min_lookback: 0.0

📁 Phase 2:
--------------------------------------------------------------------------------
  کل الگوها: 5
  موفق (با detection): 4
  میانگین min_lookback: 2.5
```

## پشتیبانی

برای سوالات یا مشکلات، لطفا به مستندات TALib مراجعه کنید:
https://ta-lib.org/

---

**تاریخ به‌روزرسانی:** 2025-10-26
**نسخه:** 2.0 (شامل 26 الگو)
