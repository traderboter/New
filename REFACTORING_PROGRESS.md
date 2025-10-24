# 🔄 Signal Generation Refactoring Progress

## 📋 هدف پروژه
باز طراحی سیستم `signal_generation` به نحوی که هر الگو و اندیکاتور فایل مخصوص به خودش را داشته باشد و با orchestrator به درستی فراخوانی شود.

## 🎯 معماری جدید

```
signal_generation/
├── orchestrator.py                    # مدیر اصلی (بدون تغییر)
│
├── analyzers/
│   ├── base_analyzer.py              # کلاس پایه
│   │
│   ├── patterns/                     # 📁 الگوها
│   │   ├── __init__.py
│   │   ├── pattern_orchestrator.py  # Orchestrator الگوها
│   │   ├── base_pattern.py          # کلاس پایه برای الگوها
│   │   │
│   │   ├── candlestick/              # 📁 الگوهای کندل‌استیک (16 الگو)
│   │   │   ├── __init__.py
│   │   │   ├── hammer.py
│   │   │   ├── inverted_hammer.py
│   │   │   ├── engulfing.py
│   │   │   ├── morning_star.py
│   │   │   ├── piercing_line.py
│   │   │   ├── three_white_soldiers.py
│   │   │   ├── morning_doji_star.py
│   │   │   ├── shooting_star.py
│   │   │   ├── hanging_man.py
│   │   │   ├── evening_star.py
│   │   │   ├── dark_cloud_cover.py
│   │   │   ├── three_black_crows.py
│   │   │   ├── evening_doji_star.py
│   │   │   ├── doji.py
│   │   │   ├── harami.py
│   │   │   └── harami_cross.py
│   │   │
│   │   └── chart/                    # 📁 الگوهای چارتی (4 الگو)
│   │       ├── __init__.py
│   │       ├── double_top_bottom.py
│   │       ├── head_shoulders.py
│   │       ├── triangle.py
│   │       └── wedge.py
│   │
│   ├── indicators/                   # 📁 اندیکاتورها (8 اندیکاتور)
│   │   ├── __init__.py
│   │   ├── indicator_orchestrator.py # Orchestrator اندیکاتورها
│   │   ├── base_indicator.py        # کلاس پایه
│   │   ├── ema.py
│   │   ├── sma.py
│   │   ├── rsi.py
│   │   ├── macd.py
│   │   ├── atr.py
│   │   ├── bollinger_bands.py
│   │   ├── stochastic.py
│   │   └── obv.py
│   │
│   ├── trend_analyzer.py             # بدون تغییر
│   ├── momentum_analyzer.py          # بدون تغییر
│   ├── volume_analyzer.py            # بدون تغییر
│   ├── sr_analyzer.py                # بدون تغییر
│   ├── volatility_analyzer.py        # بدون تغییر
│   ├── harmonic_analyzer.py          # بدون تغییر
│   ├── channel_analyzer.py           # بدون تغییر
│   ├── cyclical_analyzer.py          # بدون تغییر
│   └── htf_analyzer.py               # بدون تغییر
│
└── ... (بقیه بدون تغییر)
```

---

## ✅ مراحل تکمیل شده

### Phase 1: تحلیل و طراحی ✅
- [x] بررسی ساختار فعلی signal_generation
- [x] شناسایی لیست کامل الگوها (16 candlestick + 4 chart patterns)
- [x] شناسایی لیست کامل اندیکاتورها (8 indicators)
- [x] طراحی معماری جدید با orchestrator pattern
- [x] ایجاد فایل مستندات پیشرفت (این فایل)

### Phase 2: ساختار پایه ✅
- [x] ایجاد ساختار پوشه‌های patterns
  - [x] `analyzers/patterns/`
  - [x] `analyzers/patterns/candlestick/`
  - [x] `analyzers/patterns/chart/`
- [x] ایجاد ساختار پوشه‌های indicators
  - [x] `analyzers/indicators/`
- [x] ایجاد فایل‌های `__init__.py` برای تمام پوشه‌ها

### Phase 3: کلاس‌های پایه ✅
- [x] پیاده‌سازی `base_pattern.py` (کلاس پایه برای الگوها)
- [x] پیاده‌سازی `base_indicator.py` (کلاس پایه برای اندیکاتورها)

### Phase 4: Orchestrators ✅
- [x] پیاده‌سازی `pattern_orchestrator.py`
- [x] پیاده‌سازی `indicator_orchestrator.py`

### Phase 5: استخراج الگوهای Candlestick - Batch 1 ✅
- [x] `hammer.py`
- [x] `inverted_hammer.py`
- [x] `engulfing.py`
- [x] `morning_star.py`

---

## 🔄 مراحل در حال انجام

### Phase 5: استخراج الگوهای Candlestick - Batch 2 (12 الگو باقی‌مانده)
- [ ] `piercing_line.py`
- [ ] `three_white_soldiers.py`
- [ ] `morning_doji_star.py`
- [ ] `shooting_star.py`
- [ ] `hanging_man.py`
- [ ] `evening_star.py`
- [ ] `dark_cloud_cover.py`
- [ ] `three_black_crows.py`
- [ ] `evening_doji_star.py`
- [ ] `doji.py`
- [ ] `harami.py`
- [ ] `harami_cross.py`

### Phase 6: استخراج الگوهای Chart (4 فایل)
- [ ] `double_top_bottom.py`
- [ ] `head_shoulders.py`
- [ ] `triangle.py`
- [ ] `wedge.py`

### Phase 7: استخراج اندیکاتورها (8 فایل)
- [ ] `ema.py`
- [ ] `sma.py`
- [ ] `rsi.py`
- [ ] `macd.py`
- [ ] `atr.py`
- [ ] `bollinger_bands.py`
- [ ] `stochastic.py`
- [ ] `obv.py`

### Phase 8: یکپارچه‌سازی
- [ ] به‌روزرسانی `pattern_analyzer.py` برای استفاده از pattern orchestrator
- [ ] به‌روزرسانی `shared/indicator_calculator.py` برای استفاده از indicator orchestrator
- [ ] به‌روزرسانی `__init__.py` فایل‌ها برای exports صحیح

### Phase 9: تست و اعتبارسنجی
- [ ] تست pattern orchestrator
- [ ] تست indicator orchestrator
- [ ] تست pattern_analyzer با ساختار جدید
- [ ] تست indicator_calculator با ساختار جدید
- [ ] تست کامل pipeline
- [ ] مقایسه نتایج با نسخه قبلی

### Phase 10: مستندات
- [ ] به‌روزرسانی README.md
- [ ] به‌روزرسانی docstrings
- [ ] ایجاد مستندات استفاده

---

## 📊 آمار پیشرفت

| مرحله | تعداد کل | تکمیل شده | باقی‌مانده | درصد پیشرفت |
|-------|----------|-----------|-----------|--------------|
| Phase 1: تحلیل و طراحی | 5 | 5 | 0 | 100% ✅ |
| Phase 2: ساختار پایه | 5 | 5 | 0 | 100% ✅ |
| Phase 3: کلاس‌های پایه | 2 | 2 | 0 | 100% ✅ |
| Phase 4: Orchestrators | 2 | 2 | 0 | 100% ✅ |
| Phase 5: الگوهای Candlestick | 16 | 4 | 12 | 25% 🔄 |
| Phase 6: الگوهای Chart | 4 | 0 | 4 | 0% |
| Phase 7: اندیکاتورها | 8 | 0 | 8 | 0% |
| Phase 8: یکپارچه‌سازی | 3 | 0 | 3 | 0% |
| Phase 9: تست | 6 | 0 | 6 | 0% |
| Phase 10: مستندات | 3 | 0 | 3 | 0% |
| **جمع کل** | **54** | **18** | **36** | **33.3%** |

---

## 🎯 مزایای معماری جدید

### 1. Separation of Concerns
- هر الگو/اندیکاتور مسئولیت واضح و مشخص دارد
- کد تمیزتر و قابل فهم‌تر

### 2. Easy Testing
- تست هر الگو به صورت مستقل و جداگانه
- Unit tests ساده‌تر و سریع‌تر

### 3. Maintainability
- نگهداری کد آسان‌تر
- اضافه کردن الگوهای جدید بدون تغییر کدهای موجود

### 4. Reusability
- امکان استفاده مجدد از الگوها در جاهای مختلف
- کد DRY (Don't Repeat Yourself)

### 5. Clear Structure
- ساختار مشخص و منطقی
- Developer-friendly

### 6. Performance
- امکان lazy loading
- بهینه‌سازی آسان‌تر

---

## 🔧 توجهات فنی

### نکات مهم برای پیاده‌سازی:

1. **کلاس پایه Pattern**:
   - متد `detect()` برای شناسایی الگو
   - متد `calculate_strength()` برای محاسبه قدرت
   - متد `get_info()` برای بازگشت اطلاعات

2. **کلاس پایه Indicator**:
   - متد `calculate()` برای محاسبه اندیکاتور
   - متد `validate()` برای اعتبارسنجی ورودی
   - متد `get_values()` برای بازگشت مقادیر

3. **Pattern Orchestrator**:
   - لود تمام الگوها
   - فراخوانی `detect()` برای هر الگو
   - جمع‌آوری و ترکیب نتایج

4. **Indicator Orchestrator**:
   - لود تمام اندیکاتورها
   - فراخوانی `calculate()` برای هر اندیکاتور
   - کش کردن نتایج

### Backward Compatibility:
- حفظ API موجود تا جایی که ممکن است
- تست‌های رگرسیون برای اطمینان از عدم شکست عملکردهای قبلی

---

## 📅 Timeline تخمینی

- **Phase 2-3**: ساختار پایه و کلاس‌ها → 1 ساعت
- **Phase 4**: Orchestrators → 2 ساعت
- **Phase 5**: الگوهای Candlestick → 4 ساعت
- **Phase 6**: الگوهای Chart → 2 ساعت
- **Phase 7**: اندیکاتورها → 3 ساعت
- **Phase 8**: یکپارچه‌سازی → 2 ساعت
- **Phase 9**: تست → 3 ساعت
- **Phase 10**: مستندات → 1 ساعت

**جمع کل تخمینی**: ~18 ساعت

---

## 📝 یادداشت‌ها

### تاریخچه تغییرات:
- **2025-10-24 14:30**: پیشرفت قابل توجه (33.3%)
  - ✅ ایجاد کامل ساختار پوشه‌ها
  - ✅ پیاده‌سازی کلاس‌های پایه (BasePattern, BaseIndicator)
  - ✅ پیاده‌سازی Orchestrators (PatternOrchestrator, IndicatorOrchestrator)
  - ✅ پیاده‌سازی 4 الگوی candlestick (Hammer, Inverted Hammer, Engulfing, Morning Star)

- **2025-10-24 12:00**: شروع پروژه refactoring
  - ایجاد فایل REFACTORING_PROGRESS.md
  - تحلیل ساختار فعلی
  - طراحی معماری جدید

---

## ⚠️ ریسک‌ها و چالش‌ها

1. **Breaking Changes**: احتمال شکست کدهای موجود
   - راه حل: حفظ backward compatibility

2. **Performance**: احتمال کاهش سرعت به دلیل overhead
   - راه حل: profiling و بهینه‌سازی

3. **Complexity**: افزایش تعداد فایل‌ها
   - راه حل: مستندات دقیق و ساختار واضح

---

## 🚀 مرحله بعدی

**الان**: شروع Phase 2 - ایجاد ساختار پوشه‌ها و کلاس‌های پایه

**آخرین به‌روزرسانی**: 2025-10-24 12:00 UTC
