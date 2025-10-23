# Signal Generation Module (Refactored)

## 📁 ساختار پوشه‌ها

```
signal_generation/
├── __init__.py                 # نقطه ورود اصلی
├── orchestrator.py             # SignalGenerator (هماهنگ‌کننده)
├── context.py                  # AnalysisContext (حافظه مشترک)
│
├── analyzers/                  # تحلیلگرها
│   ├── __init__.py
│   ├── base_analyzer.py        # کلاس پایه
│   ├── trend_analyzer.py       # تحلیل روند
│   ├── momentum_analyzer.py    # تحلیل مومنتوم
│   ├── volume_analyzer.py      # تحلیل حجم
│   ├── pattern_analyzer.py     # الگوهای قیمتی
│   ├── sr_analyzer.py          # حمایت/مقاومت
│   └── ...                     # سایر تحلیلگرها
│
├── processing/                 # پردازش سیگنال
│   ├── __init__.py
│   ├── signal_scorer.py        # امتیازدهی
│   └── signal_validator.py     # اعتبارسنجی
│
├── shared/                     # ابزارهای مشترک
│   ├── __init__.py
│   ├── indicator_calculator.py # محاسبه اندیکاتورها
│   ├── data_models.py          # مدل‌های داده
│   └── utils.py                # توابع کمکی
│
└── systems/                    # سیستم‌های پشتیبان
    ├── __init__.py
    ├── market_regime_detector.py
    ├── adaptive_learning_system.py
    ├── correlation_manager.py
    └── emergency_circuit_breaker.py
```

## 🎯 نحوه استفاده

```python
from signal_generation import SignalGenerator

# ایجاد نمونه
signal_generator = SignalGenerator(config)

# تحلیل
signal = await signal_generator.analyze_symbol(symbol, timeframes_data)
```

## 📊 وضعیت توسعه

- [x] Phase 0: آماده‌سازی
- [x] Phase 1: ایجاد زیرساخت ← **در حال انجام**
- [ ] Phase 2: IndicatorCalculator
- [ ] Phase 3: Analyzers
- [ ] Phase 4: Signal Processing
- [ ] Phase 5: Orchestrator
- [ ] Phase 6: Testing
- [ ] Phase 7: Deployment

## 📝 یادداشت‌ها

این ساختار در حال توسعه است. برای اطلاعات بیشتر به `REFACTORING_ROADMAP.md` مراجعه کنید.
