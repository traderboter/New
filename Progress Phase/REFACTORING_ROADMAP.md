# 🗺️ نقشه راه Refactoring سیستم Signal Generator

> **آخرین بروزرسانی:** 2025-10-15  
> **هدف:** تبدیل `signal_generator.py` به معماری Orchestrator با Context-Based Analysis

---

## 📋 فهرست مطالب

1. [نمای کلی پروژه](#نمای-کلی-پروژه)
2. [معماری هدف](#معماری-هدف)
3. [مراحل اجرایی (Phases)](#مراحل-اجرایی)
4. [چک‌لیست پیشرفت](#چک‌لیست-پیشرفت)
5. [جزئیات هر فاز](#جزئیات-هر-فاز)
6. [راهنمای تست](#راهنمای-تست)
7. [نکات مهم](#نکات-مهم)

---

## 🎯 نمای کلی پروژه

### مشکل فعلی:
- فایل `signal_generator.py` بیش از **3000+ خط** کد دارد
- یک کلاس God Object است (تمام مسئولیت‌ها در یک جا)
- محاسبات تکراری اندیکاتورها
- ماژول‌ها از نتایج یکدیگر بی‌خبرند
- تست و نگهداری سخت است

### راه‌حل:
✅ معماری **Orchestrator-Based** با **Context-Aware Analysis**  
✅ تقسیم به ماژول‌های کوچک و مستقل  
✅ محاسبه یکبار اندیکاتورها با `IndicatorCalculator`  
✅ اشتراک اطلاعات بین ماژول‌ها با `AnalysisContext`  

### نتیجه نهایی:
- 🎯 کد تمیز و قابل نگهداری
- ⚡ عملکرد بهتر (حذف محاسبات تکراری)
- 🧪 تست‌پذیری بالا
- 🔧 قابلیت توسعه آسان
- 🤝 همکاری بین ماژول‌ها

---

## 🏗️ معماری هدف

### ساختار نهایی پوشه‌ها:

```
📁 signal_generation/
│
├── 📄 __init__.py                          # نقطه ورود اصلی
├── 📄 orchestrator.py                      # SignalGenerator (هماهنگ‌کننده اصلی)
├── 📄 context.py                           # AnalysisContext (حافظه مشترک)
│
├── 📁 analyzers/                           # ماژول‌های تحلیلگر
│   ├── 📄 __init__.py
│   ├── 📄 base_analyzer.py                 # کلاس پایه مشترک
│   ├── 📄 trend_analyzer.py                # تحلیل روند
│   ├── 📄 momentum_analyzer.py             # تحلیل مومنتوم (RSI, MACD, واگرایی)
│   ├── 📄 volume_analyzer.py               # تحلیل حجم معاملات
│   ├── 📄 pattern_analyzer.py              # الگوهای کندلی + چارتی
│   ├── 📄 sr_analyzer.py                   # حمایت و مقاومت
│   ├── 📄 harmonic_analyzer.py             # الگوهای هارمونیک
│   ├── 📄 channel_analyzer.py              # کانال‌های قیمتی
│   ├── 📄 cyclical_analyzer.py             # الگوهای چرخه‌ای
│   ├── 📄 volatility_analyzer.py           # تحلیل نوسان
│   └── 📄 htf_analyzer.py                  # تایم‌فریم بالاتر
│
├── 📁 processing/                          # پردازش نهایی سیگنال
│   ├── 📄 __init__.py
│   ├── 📄 signal_scorer.py                 # امتیازدهی هوشمند
│   └── 📄 signal_validator.py              # اعتبارسنجی و فیلترها
│
├── 📁 shared/                              # ابزارهای مشترک
│   ├── 📄 __init__.py
│   ├── 📄 indicator_calculator.py          # محاسبه همه اندیکاتورها
│   ├── 📄 data_models.py                   # مدل‌های داده (SignalInfo, etc.)
│   └── 📄 utils.py                         # توابع کمکی
│
└── 📁 systems/                             # سیستم‌های پشتیبان (موجود)
    ├── 📄 __init__.py
    ├── 📄 market_regime_detector.py        # تشخیص رژیم بازار
    ├── 📄 adaptive_learning_system.py      # یادگیری تطبیقی
    ├── 📄 correlation_manager.py           # مدیریت همبستگی
    └── 📄 emergency_circuit_breaker.py     # قطع‌کننده اضطراری
```

### جریان کاری (Pipeline):

```
┌─────────────────────────────────────────────────────────────┐
│  SignalGenerator (Orchestrator)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Create Empty Context                              │
│  AnalysisContext(symbol, timeframe, dataframe)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Calculate Indicators (ONE TIME)                   │
│  IndicatorCalculator.calculate_all(context)                 │
│  → EMA, RSI, MACD, ATR, Stochastic, Volume SMA, etc.       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Run Analyzers (Sequential with Context)           │
│                                                              │
│  TrendAnalyzer.analyze(context)                             │
│    → context.add_result('trend', {...})                     │
│                                                              │
│  MomentumAnalyzer.analyze(context)                          │
│    → Can read: context.get_result('trend')                  │
│    → context.add_result('momentum', {...})                  │
│                                                              │
│  VolumeAnalyzer.analyze(context)                            │
│    → context.add_result('volume', {...})                    │
│                                                              │
│  PatternAnalyzer.analyze(context)                           │
│    → Uses trend info for better scoring                     │
│    → context.add_result('patterns', {...})                  │
│                                                              │
│  [... other analyzers ...]                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Score Signal                                      │
│  SignalScorer.calculate_score(context)                      │
│  → Combines all results into weighted score                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: Validate Signal                                   │
│  SignalValidator.validate(signal, context)                  │
│  → Risk/Reward check                                        │
│  → Correlation management                                   │
│  → Circuit breaker check                                    │
│  → Portfolio exposure check                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                   ✅ Final Signal or ❌ Rejected
```

---

## 📊 مراحل اجرایی (Phases)

### ✅ **PHASE 0: آماده‌سازی** [COMPLETED: ❌]
- [ ] 0.1. بکاپ کامل کد فعلی
- [ ] 0.2. ایجاد شاخه Git جدید: `feature/signal-generator-refactoring`
- [ ] 0.3. مستندسازی توابع مهم فعلی
- [ ] 0.4. نوشتن تست‌های ادغام برای رفتار فعلی (Baseline Tests)

---

### 🔄 **PHASE 1: ایجاد زیرساخت اصلی** [COMPLETED: ❌]

#### 1.1. ایجاد ساختار پوشه‌ها
```bash
mkdir -p signal_generation/analyzers
mkdir -p signal_generation/processing
mkdir -p signal_generation/shared
mkdir -p signal_generation/systems
```

#### 1.2. ایجاد فایل‌های پایه
- [ ] `signal_generation/__init__.py`
- [ ] `signal_generation/context.py`
- [ ] `signal_generation/analyzers/base_analyzer.py`
- [ ] `signal_generation/shared/data_models.py`

#### 1.3. پیاده‌سازی AnalysisContext
```python
# context.py
class AnalysisContext:
    - symbol: str
    - timeframe: str
    - df: pd.DataFrame (enriched with indicators)
    - results: Dict[str, Any]
    - metadata: Dict[str, Any]
    
    Methods:
    - add_result(analyzer_name, result)
    - get_result(analyzer_name)
    - has_result(analyzer_name)
    - get_all_results()
```

#### 1.4. پیاده‌سازی BaseAnalyzer
```python
# analyzers/base_analyzer.py
class BaseAnalyzer(ABC):
    - config: Dict
    - enabled: bool
    
    Methods:
    - @abstractmethod analyze(context: AnalysisContext)
    - _check_enabled()
    - _validate_context(context)
```

**تحویل Phase 1:**
- ✅ ساختار پوشه‌ها ایجاد شده
- ✅ کلاس‌های پایه نوشته و تست شده
- ✅ مستندات API برای کلاس‌های پایه

---

### 🔄 **PHASE 2: ایجاد IndicatorCalculator** [COMPLETED: ❌]

#### 2.1. طراحی IndicatorCalculator
- [ ] شناسایی تمام اندیکاتورهای مورد استفاده در کد فعلی
- [ ] طراحی API برای محاسبه دسته‌جمعی
- [ ] تعریف کنفیگ اندیکاتورها

#### 2.2. پیاده‌سازی
```python
# shared/indicator_calculator.py
class IndicatorCalculator:
    Methods:
    - calculate_all(context: AnalysisContext)
    - calculate_moving_averages(df)
    - calculate_oscillators(df)
    - calculate_volatility_indicators(df)
    - calculate_volume_indicators(df)
    - calculate_momentum_indicators(df)
```

#### 2.3. اندیکاتورهای مورد نیاز:
- [ ] Moving Averages: EMA(20, 50, 200), SMA(20, 50, 200)
- [ ] Oscillators: RSI(14), Stochastic, MACD
- [ ] Volatility: ATR(14), Bollinger Bands
- [ ] Volume: Volume SMA(20), OBV
- [ ] Momentum: Rate of Change (ROC)

#### 2.4. تست‌ها
- [ ] تست محاسبه صحیح هر اندیکاتور
- [ ] تست عملکرد (Performance Test)
- [ ] مقایسه با نتایج کد قدیمی

**تحویل Phase 2:**
- ✅ IndicatorCalculator کامل و تست شده
- ✅ تمام اندیکاتورهای لازم محاسبه می‌شوند
- ✅ عملکرد بهتر از کد قدیمی (بدون تکرار)

---

### 🔄 **PHASE 3: پیاده‌سازی Analyzers (به ترتیب اولویت)** [COMPLETED: ❌]

#### 3.1. TrendAnalyzer ⭐ (اولویت بالا)
- [ ] استخراج کد از `detect_trend()` فعلی
- [ ] پیاده‌سازی در کلاس جدید
- [ ] تست مقایسه‌ای با کد قدیمی
- [ ] مستندسازی

**خروجی:**
```python
context.results['trend'] = {
    'direction': 'bullish' | 'bearish' | 'sideways',
    'strength': float (0-3),
    'phase': 'starting' | 'continuing' | 'weakening',
    'ema_alignment': bool,
    'price_position': str
}
```

#### 3.2. MomentumAnalyzer ⭐ (اولویت بالا)
- [ ] استخراج کد از `analyze_momentum_indicators()` فعلی
- [ ] افزودن تشخیص واگرایی
- [ ] تست و مستندسازی

**خروجی:**
```python
context.results['momentum'] = {
    'rsi': float,
    'rsi_signal': 'overbought' | 'oversold' | 'neutral',
    'stochastic': Dict,
    'macd': Dict,
    'divergence': Dict | None,
    'overall': 'strong_bullish' | 'bullish' | 'neutral' | 'bearish' | 'strong_bearish'
}
```

#### 3.3. VolumeAnalyzer ⭐ (اولویت بالا)
- [ ] استخراج کد از `analyze_volume_trend()` فعلی
- [ ] تست و مستندسازی

**خروجی:**
```python
context.results['volume'] = {
    'is_confirmed_by_volume': bool,
    'volume_ratio': float,
    'volume_trend': 'increasing' | 'decreasing' | 'stable',
    'breakout_volume': bool
}
```

#### 3.4. PatternAnalyzer ⭐⭐ (اولویت متوسط)
- [ ] استخراج کد از `analyze_price_action()` فعلی
- [ ] ترکیب الگوهای کندلی و چارتی
- [ ] **استفاده از context برای امتیازدهی هوشمند**
- [ ] تست و مستندسازی

**خروجی:**
```python
context.results['patterns'] = {
    'candlestick_patterns': List[Dict],
    'chart_patterns': List[Dict],
    'pattern_strength': float,
    'alignment_with_trend': bool  # استفاده از context.get_result('trend')
}
```

#### 3.5. SRAnalyzer (Support/Resistance) ⭐⭐ (اولویت متوسط)
- [ ] استخراج کد از `detect_support_resistance()` فعلی
- [ ] تست و مستندسازی

#### 3.6. HarmonicAnalyzer ⭐⭐⭐ (اولویت پایین)
- [ ] استخراج کد از `detect_harmonic_patterns()` فعلی
- [ ] تست و مستندسازی

#### 3.7. ChannelAnalyzer ⭐⭐⭐ (اولویت پایین)
- [ ] استخراج کد از `detect_price_channels()` فعلی
- [ ] تست و مستندسازی

#### 3.8. CyclicalAnalyzer ⭐⭐⭐ (اولویت پایین)
- [ ] استخراج کد از `detect_cyclical_patterns()` فعلی
- [ ] تست و مستندسازی

#### 3.9. VolatilityAnalyzer ⭐⭐ (اولویت متوسط)
- [ ] استخراج کد از `analyze_volatility_conditions()` فعلی
- [ ] تست و مستندسازی

#### 3.10. HTFAnalyzer (Higher Timeframe) ⭐⭐ (اولویت متوسط)
- [ ] استخراج کد از `analyze_higher_timeframe_structure()` فعلی
- [ ] تست و مستندسازی

**تحویل Phase 3:**
- ✅ تمام Analyzers پیاده‌سازی شده
- ✅ هر Analyzer تست واحد دارد
- ✅ Analyzers از Context برای تصمیم‌گیری هوشمند استفاده می‌کنند

---

### 🔄 **PHASE 4: پیاده‌سازی Signal Processing** [COMPLETED: ❌]

#### 4.1. SignalScorer
```python
# processing/signal_scorer.py
class SignalScorer:
    Methods:
    - calculate_score(context: AnalysisContext) -> float
    - calculate_base_score(context)
    - apply_timeframe_weight(score, timeframe)
    - apply_confluence_bonus(context)
    - apply_trend_alignment(context)
    - calculate_final_score() -> SignalScore
```

#### 4.2. فرمول امتیازدهی
```
Final Score = Base Score 
            × Timeframe Weight
            × Trend Alignment Factor
            × Volume Confirmation Factor
            × Pattern Quality Factor
            × HTF Structure Factor
            × Volatility Factor
            + Confluence Bonus
```

#### 4.3. SignalValidator
```python
# processing/signal_validator.py
class SignalValidator:
    Methods:
    - validate(signal: SignalInfo, context: AnalysisContext) -> Optional[SignalInfo]
    - check_risk_reward(signal)
    - check_circuit_breaker(signal)
    - check_correlation_safety(signal)
    - check_volatility_limits(signal)
    - apply_adaptive_thresholds(signal)
```

**تحویل Phase 4:**
- ✅ امتیازدهی هوشمند پیاده‌سازی شده
- ✅ تمام فیلترها و اعتبارسنجی‌ها موجود است
- ✅ تست‌های کامل برای scoring و validation

---

### 🔄 **PHASE 5: پیاده‌سازی Orchestrator** [COMPLETED: ❌]

#### 5.1. SignalGenerator (Orchestrator)
```python
# orchestrator.py
class SignalGenerator:
    def __init__(self, config):
        # Initialize all components
        self.indicator_calculator = IndicatorCalculator(config)
        
        # Core analyzers
        self.analyzers = [
            TrendAnalyzer(config),
            MomentumAnalyzer(config),
            VolumeAnalyzer(config),
            PatternAnalyzer(config),
            SRAnalyzer(config),
            # ... other analyzers
        ]
        
        # Processing
        self.scorer = SignalScorer(config)
        self.validator = SignalValidator(config)
        
        # Systems (existing)
        self.regime_detector = MarketRegimeDetector(config)
        self.adaptive_learning = AdaptiveLearningSystem(config)
        self.correlation_manager = CorrelationManager(config)
        self.circuit_breaker = EmergencyCircuitBreaker(config)
    
    async def analyze_symbol(self, symbol, timeframes_data):
        # Main orchestration logic
```

#### 5.2. جریان اصلی
- [ ] پیاده‌سازی حلقه اصلی analyze_symbol
- [ ] مدیریت Multi-timeframe
- [ ] ترکیب نتایج از تایم‌فریم‌های مختلف
- [ ] مدیریت خطا و Exception Handling

#### 5.3. یکپارچه‌سازی با سیستم‌های موجود
- [ ] MarketRegimeDetector
- [ ] AdaptiveLearningSystem
- [ ] CorrelationManager
- [ ] EmergencyCircuitBreaker

**تحویل Phase 5:**
- ✅ Orchestrator کامل
- ✅ تمام ماژول‌ها یکپارچه شده‌اند
- ✅ جریان کاری کامل تست شده

---

### 🔄 **PHASE 6: Migration & Testing** [COMPLETED: ❌]

#### 6.1. تست یکپارچگی
- [ ] تست کامل با داده‌های واقعی
- [ ] مقایسه نتایج با کد قدیمی
- [ ] اطمینان از یکسان بودن رفتار (Regression Test)

#### 6.2. تست عملکرد
- [ ] اندازه‌گیری زمان اجرا
- [ ] اندازه‌گیری مصرف حافظه
- [ ] مقایسه با کد قدیمی

#### 6.3. یکپارچه‌سازی با SignalProcessor
- [ ] به‌روزرسانی `signal_processor.py`
- [ ] تست با `crypto_trading_bot.py`

#### 6.4. مستندسازی
- [ ] API Documentation
- [ ] Architecture Diagrams
- [ ] Usage Examples
- [ ] Migration Guide

**تحویل Phase 6:**
- ✅ سیستم کامل تست شده
- ✅ عملکرد بهتر از کد قدیمی
- ✅ مستندات کامل

---

### 🔄 **PHASE 7: Deployment & Cleanup** [COMPLETED: ❌]

#### 7.1. آماده‌سازی Production
- [ ] کد قدیمی را Deprecate کنید
- [ ] فایل `signal_generator.py` قدیمی را به `signal_generator_legacy.py` تغییر نام دهید
- [ ] Import ها را به‌روزرسانی کنید

#### 7.2. Monitoring
- [ ] اضافه کردن Logging مناسب
- [ ] اضافه کردن Metrics
- [ ] تنظیم Alerts

#### 7.3. پاکسازی
- [ ] حذف کدهای مرده (Dead Code)
- [ ] حذف Import های غیرضروری
- [ ] بهینه‌سازی نهایی

**تحویل Phase 7:**
- ✅ سیستم در Production
- ✅ Monitoring فعال
- ✅ کد قدیمی حذف/آرشیو شده

---

## ✅ چک‌لیست پیشرفت کلی

### Phase 0: Preparation
- [ ] بکاپ کد فعلی
- [ ] ایجاد شاخه Git
- [ ] مستندسازی رفتار فعلی
- [ ] نوشتن Baseline Tests

### Phase 1: Infrastructure
- [ ] ساختار پوشه‌ها
- [ ] AnalysisContext
- [ ] BaseAnalyzer
- [ ] Data Models

### Phase 2: Indicator Calculator
- [ ] طراحی IndicatorCalculator
- [ ] پیاده‌سازی تمام اندیکاتورها
- [ ] تست عملکرد

### Phase 3: Analyzers (10 ماژول)
- [ ] TrendAnalyzer
- [ ] MomentumAnalyzer
- [ ] VolumeAnalyzer
- [ ] PatternAnalyzer
- [ ] SRAnalyzer
- [ ] HarmonicAnalyzer
- [ ] ChannelAnalyzer
- [ ] CyclicalAnalyzer
- [ ] VolatilityAnalyzer
- [ ] HTFAnalyzer

### Phase 4: Signal Processing
- [ ] SignalScorer
- [ ] SignalValidator

### Phase 5: Orchestrator
- [ ] SignalGenerator (Orchestrator)
- [ ] یکپارچه‌سازی با سیستم‌های موجود

### Phase 6: Testing
- [ ] تست یکپارچگی
- [ ] تست عملکرد
- [ ] مستندسازی

### Phase 7: Deployment
- [ ] Production Deployment
- [ ] Monitoring Setup
- [ ] Cleanup

---

## 🧪 راهنمای تست

### تست واحد (Unit Tests)
هر Analyzer باید تست‌های زیر را داشته باشد:

```python
# tests/analyzers/test_trend_analyzer.py

def test_trend_analyzer_bullish_trend():
    """تست تشخیص روند صعودی"""
    # Arrange: داده‌های تست با روند صعودی
    # Act: اجرای analyzer
    # Assert: بررسی نتیجه

def test_trend_analyzer_with_context():
    """تست استفاده از Context"""
    # بررسی اضافه شدن نتایج به Context

def test_trend_analyzer_disabled():
    """تست حالت غیرفعال"""
    # بررسی عدم اجرا در صورت غیرفعال بودن
```

### تست یکپارچگی (Integration Tests)
```python
def test_full_pipeline():
    """تست کامل Pipeline"""
    # از IndicatorCalculator تا Signal نهایی
    
def test_context_sharing():
    """تست اشتراک اطلاعات بین Analyzers"""
    # بررسی استفاده Analyzer B از نتایج Analyzer A
```

### تست رگرسیون (Regression Tests)
```python
def test_compare_with_legacy():
    """مقایسه نتایج با کد قدیمی"""
    # بررسی یکسان بودن رفتار
```

---

## 💡 نکات مهم

### ⚠️ نکات توسعه:

1. **تست اول، کد بعد (TDD)**
   - قبل از نوشتن Analyzer، تست بنویسید

2. **Commit های کوچک و منظم**
   - هر Analyzer یک commit جداگانه
   - پیام‌های واضح و توصیفی

3. **مستندات همزمان با کد**
   - Docstring برای همه کلاس‌ها و متدها
   - مثال‌های استفاده

4. **Code Review**
   - هر Phase باید Review شود قبل از شروع Phase بعدی

5. **Performance Profiling**
   - استفاده از `cProfile` برای شناسایی bottleneck ها

### 🔥 Best Practices:

```python
# ✅ GOOD: استفاده از Context
class TrendAnalyzer(BaseAnalyzer):
    def analyze(self, context: AnalysisContext):
        df = context.df
        # از indicators از پیش محاسبه شده استفاده کن
        ema_20 = df['ema_20'].iloc[-1]
        # ...
        context.add_result('trend', result)

# ❌ BAD: محاسبه مجدد
class TrendAnalyzer(BaseAnalyzer):
    def analyze(self, context: AnalysisContext):
        df = context.df
        # محاسبه دوباره! (تکراری و کند)
        ema_20 = ta.ema(df['close'], 20).iloc[-1]
```

```python
# ✅ GOOD: استفاده از نتایج دیگران
class PatternAnalyzer(BaseAnalyzer):
    def analyze(self, context: AnalysisContext):
        # بخوان از Context
        trend = context.get_result('trend')
        
        # تصمیم هوشمند
        if trend and trend['direction'] == 'bullish':
            score *= 1.5  # امتیاز بیشتر

# ❌ BAD: بی‌خبری از دیگران
class PatternAnalyzer(BaseAnalyzer):
    def analyze(self, context: AnalysisContext):
        # همیشه امتیاز ثابت (بدون توجه به Context)
        score = 10
```

### 📝 الگوی نامگذاری:

- **Classes**: `PascalCase` - `TrendAnalyzer`, `SignalScorer`
- **Methods**: `snake_case` - `analyze()`, `calculate_score()`
- **Constants**: `UPPER_SNAKE_CASE` - `MAX_SCORE`, `MIN_THRESHOLD`
- **Files**: `snake_case.py` - `trend_analyzer.py`, `signal_scorer.py`

---

## 📅 تایم‌لاین تخمینی

| Phase | تخمین زمان | اولویت |
|-------|------------|--------|
| Phase 0 | 1-2 روز | بالا ⭐⭐⭐ |
| Phase 1 | 2-3 روز | بالا ⭐⭐⭐ |
| Phase 2 | 3-4 روز | بالا ⭐⭐⭐ |
| Phase 3 | 10-14 روز | بالا ⭐⭐⭐ |
| Phase 4 | 4-5 روز | متوسط ⭐⭐ |
| Phase 5 | 3-4 روز | بالا ⭐⭐⭐ |
| Phase 6 | 5-7 روز | بالا ⭐⭐⭐ |
| Phase 7 | 2-3 روز | متوسط ⭐⭐ |
| **جمع کل** | **30-42 روز** | |

💡 **توصیه**: این کار را به صورت Incremental انجام دهید. نیازی نیست همه را یکجا پیاده‌سازی کنید.

---

## 🎯 معیارهای موفقیت

پس از اتمام Refactoring، باید این معیارها برآورده شوند:

### کیفیت کد:
- ✅ هیچ فایلی بیش از 500 خط کد نداشته باشد
- ✅ Coverage تست‌ها بالای 80% باشد
- ✅ هیچ Code Smell یا Anti-pattern وجود نداشته باشد

### عملکرد:
- ✅ سرعت تحلیل حداقل 20% بهتر از قبل
- ✅ مصرف حافظه کمتر یا مساوی قبل
- ✅ بدون Memory Leak

### نگهداری:
- ✅ افزودن Analyzer جدید کمتر از 2 ساعت زمان ببرد
- ✅ تغییر منطق یک Analyzer سایر قسمت‌ها را تحت تأثیر قرار ندهد
- ✅ مستندات کامل و به‌روز باشد

### رفتار:
- ✅ تمام تست‌های Regression پاس شوند
- ✅ نتایج یکسان با کد قدیمی (در 95%+ موارد)

---

## 📚 منابع مفید

### الگوهای طراحی:
- **Strategy Pattern**: برای Analyzers
- **Chain of Responsibility**: برای Pipeline
- **Factory Pattern**: برای ساخت Analyzers
- **Observer Pattern**: برای اطلاع از تغییرات

### کتابخانه‌های مفید:
- `pandas-ta`: برای اندیکاتورهای تکنیکال
- `pytest`: برای تست
- `pytest-cov`: برای Coverage
- `pytest-asyncio`: برای تست کدهای async

---

## 🔄 نسخه‌بندی این سند

| نسخه | تاریخ | تغییرات |
|------|-------|---------|
| 1.0 | 2025-10-15 | نسخه اولیه نقشه راه |

---

## 🤝 مشارکت

برای هر پیشنهاد یا سوال، به این سند مراجعه کنید و وضعیت Checklist ها را به‌روز نگه دارید.

---

**🎉 موفق باشید!**

این یک پروژه Refactoring بزرگ است، اما با این نقشه راه می‌توانید گام به گام پیش بروید.
