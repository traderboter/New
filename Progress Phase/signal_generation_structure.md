# 🏗️ ساختار واقعی Signal Generation v2.0

## 📂 نمای کلی ساختار

```
signal_generation/                          (فولدر اصلی)
├── 📄 README.md                            (مستندات)
├── 📄 __init__.py                          (99 lines)
│
├── 📁 analyzers/                           (تحلیل‌گرها - 11 فایل)
│   ├── 📄 __init__.py                      (27 lines)
│   ├── 📄 base_analyzer.py                 (88 lines)   - کلاس پایه
│   ├── 📄 channel_analyzer.py              (134 lines)  - تحلیل کانال‌ها
│   ├── 📄 cyclical_analyzer.py             (156 lines)  - تحلیل چرخه‌ای
│   ├── 📄 harmonic_analyzer.py             (273 lines)  - الگوهای هارمونیک
│   ├── 📄 htf_analyzer.py                  (267 lines)  - تایم‌فریم بالاتر
│   ├── 📄 momentum_analyzer.py             (602 lines)  - مومنتوم
│   ├── 📄 pattern_analyzer.py              (665 lines)  - الگوهای کندل
│   ├── 📄 sr_analyzer.py                   (598 lines)  - حمایت/مقاومت
│   ├── 📄 trend_analyzer.py                (480 lines)  - روند
│   ├── 📄 volatility_analyzer.py           (454 lines)  - نوسان
│   └── 📄 volume_analyzer.py               (504 lines)  - حجم
│
├── 📄 context.py                           (161 lines)  - AnalysisContext
├── 📄 orchestrator.py                      (729 lines)  - SignalOrchestrator
│
├── 📁 shared/                              (کامپوننت‌های مشترک - 3 فایل)
│   ├── 📄 __init__.py                      (9 lines)
│   ├── 📄 data_models.py                   (471 lines)  - مدل‌های داده
│   └── 📄 indicator_calculator.py          (484 lines)  - محاسبه اندیکاتورها
│
├── 📄 signal_info.py                       (275 lines)  - SignalInfo + SignalRejection
├── 📄 signal_score.py                      (231 lines)  - SignalScore
├── 📄 signal_scorer.py                     (579 lines)  - SignalScorer
├── 📄 signal_validator.py                  (470 lines)  - SignalValidator
│
├── 📁 systems/                             (سیستم‌های کمکی - 5 فایل)
│   ├── 📄 __init__.py                      (25 lines)
│   ├── 📄 adaptive_learning_system.py      (424 lines)  - یادگیری تطبیقی
│   ├── 📄 correlation_manager.py           (333 lines)  - مدیریت همبستگی
│   ├── 📄 emergency_circuit_breaker.py     (276 lines)  - مدار ایمنی
│   └── 📄 market_regime_detector.py        (327 lines)  - تشخیص رژیم بازار
│
└── 📁 examples/                            (نمونه‌های استفاده - 2 فایل)
    ├── 📄 orchestrator_example.py          (314 lines)
    └── 📄 phase4_integration_example.py    (363 lines)
```


## 🎯 نقشه کامپوننت‌ها و مسئولیت‌ها

### 1️⃣ **Core Layer (لایه هسته)**
```
context.py                    → AnalysisContext
   ↓
   مرکز اشتراک داده بین تمام Analyzer ها
   ذخیره DataFrame enriched + نتایج تحلیل‌ها
```

```
orchestrator.py               → SignalOrchestrator + OrchestratorStats
   ↓
   مدیر اصلی pipeline
   هماهنگ‌کننده تمام Analyzer ها
   کنترل جریان از ابتدا تا انتها
```

---

### 2️⃣ **Analyzers Layer (لایه تحلیل‌گرها)**
```
analyzers/
├── base_analyzer.py          → BaseAnalyzer (کلاس پایه - Abstract)
│   └── تمام Analyzer ها از این ارث‌بری می‌کنند
│
├── trend_analyzer.py         → TrendAnalyzer
│   └── 🔍 تشخیص روند (bullish/bearish/neutral)
│   └── محاسبه: direction, strength, phase
│
├── momentum_analyzer.py      → MomentumAnalyzer (Context-Aware! 🧠)
│   └── 📈 MACD, RSI, Stochastic, Divergence
│   └── می‌خواند: context.get_result('trend')
│
├── volume_analyzer.py        → VolumeAnalyzer (Context-Aware! 🧠)
│   └── 📦 Volume, OBV, Volume Ratio
│   └── می‌خواند: trend + momentum
│
├── pattern_analyzer.py       → PatternAnalyzer
│   └── 🎨 الگوهای کندل (60+ patterns)
│   └── Hammer, Engulfing, Doji, Morning Star...
│
├── sr_analyzer.py            → SRAnalyzer
│   └── 🎯 حمایت و مقاومت (Support/Resistance)
│   └── شناسایی breakout و breakdown
│
├── volatility_analyzer.py    → VolatilityAnalyzer
│   └── 📊 ATR, Bollinger Bands, نوسانات
│
├── harmonic_analyzer.py      → HarmonicAnalyzer
│   └── 🎵 الگوهای هارمونیک (Gartley, Butterfly, Bat)
│
├── channel_analyzer.py       → ChannelAnalyzer
│   └── 📐 کانال‌های قیمتی
│
├── cyclical_analyzer.py      → CyclicalAnalyzer
│   └── 🔄 الگوهای چرخه‌ای
│
└── htf_analyzer.py           → HTFAnalyzer
    └── 🔭 تحلیل تایم‌فریم بالاتر (HTF = Higher TimeFrame)
```

---

### 3️⃣ **Shared Layer (لایه مشترک)**
```
shared/
├── indicator_calculator.py   → IndicatorCalculator
│   └── ⚡ محاسبه تمام اندیکاتورها یکبار
│   └── EMA, SMA, RSI, MACD, ATR, Bollinger, Stochastic, OBV
│   └── در context تزریق می‌شود
│
└── data_models.py            → SignalScore, SignalInfo, TradeResult
    └── 📋 مدل‌های داده استاندارد
```

---

### 4️⃣ **Scoring & Validation Layer (لایه امتیازدهی و اعتبارسنجی)**
```
signal_score.py               → SignalScore (مدل داده)
signal_scorer.py              → SignalScorer (محاسبه امتیاز)
   ↓
   ترکیب نتایج تمام Analyzer ها
   وزن‌دهی: Trend(30%), Momentum(25%), Volume(15%)...
   ضرایب تایید: Volume Bonus, Trend Alignment

signal_info.py                → SignalInfo + SignalRejection
   ↓
   شیء نهایی سیگنال

signal_validator.py           → SignalValidator
   ↓
   ✅ بررسی Score >= 60
   ✅ بررسی R/R >= 2
   ✅ Circuit Breaker
   ✅ Correlation
   ✅ Portfolio Exposure
```

---

### 5️⃣ **Systems Layer (سیستم‌های کمکی)**
```
systems/
├── adaptive_learning_system.py   → AdaptiveLearningSystem
│   └── 🧠 یادگیری از معاملات گذشته
│   └── بهبود پارامترها
│
├── correlation_manager.py        → CorrelationManager
│   └── 🔗 مدیریت همبستگی نمادها
│   └── جلوگیری از معاملات همبسته
│
├── emergency_circuit_breaker.py  → EmergencyCircuitBreaker
│   └── 🚨 مدار ایمنی اضطراری
│   └── توقف معاملات در شرایط بحرانی
│
└── market_regime_detector.py     → MarketRegimeDetector
    └── 🎭 تشخیص رژیم بازار
    └── Trending, Ranging, Volatile...
```

---

## 🔄 جریان کامل پردازش (Pipeline Flow)

```
1️⃣  INPUT: DataFrame (OHLCV)
      ↓
2️⃣  AnalysisContext.create()
      ↓
3️⃣  IndicatorCalculator.calculate_all(context)
      ↓ (DataFrame enriched شد)
4️⃣  SignalOrchestrator.process()
      ├─→ TrendAnalyzer.analyze(context)        → context.add_result('trend', ...)
      ├─→ MomentumAnalyzer.analyze(context)     → context.add_result('momentum', ...)
      ├─→ VolumeAnalyzer.analyze(context)       → context.add_result('volume', ...)
      ├─→ PatternAnalyzer.analyze(context)      → context.add_result('patterns', ...)
      ├─→ SRAnalyzer.analyze(context)           → context.add_result('sr_levels', ...)
      ├─→ VolatilityAnalyzer.analyze(context)   → context.add_result('volatility', ...)
      ├─→ HarmonicAnalyzer.analyze(context)     → context.add_result('harmonic', ...)
      ├─→ ChannelAnalyzer.analyze(context)      → context.add_result('channels', ...)
      ├─→ CyclicalAnalyzer.analyze(context)     → context.add_result('cyclical', ...)
      └─→ HTFAnalyzer.analyze(context)          → context.add_result('htf', ...)
      ↓ (Context حالا پر از نتایج است)
5️⃣  SignalScorer.calculate_score(context)
      ↓ (Score محاسبه شد)
6️⃣  SignalValidator.validate(signal, context)
      ↓ (اعتبارسنجی شد)
7️⃣  OUTPUT: SignalInfo or SignalRejection
```

---

## 🔑 کلیدی‌ترین تفاوت با کد قدیم

### ❌ کد قدیم (signal_generator.py - 5806 خط!)
```python
class SignalGenerator:
    def analyze_symbol(self, df):
        # همه چیز در یک جا!
        # 5806 خط کد در یک فایل
        # God Object Anti-Pattern
```

### ✅ کد جدید (signal_generation/ - 25 فایل)
```python
# معماری ماژولار و تمیز
signal_generation/
├── analyzers/           (11 Analyzer مستقل)
├── shared/              (کامپوننت‌های مشترک)
├── systems/             (سیستم‌های کمکی)
└── orchestrator.py      (مدیر اصلی)

# هر Analyzer:
- مستقل است (Independent)
- تست‌پذیر است (Testable)
- Context-Aware است (هوشمند)
- کمتر از 700 خط کد دارد
```
