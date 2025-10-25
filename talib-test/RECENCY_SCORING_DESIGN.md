# Pattern Recency Scoring System - طرح پیاده‌سازی

تاریخ: 2025-10-25
هدف: امتیازدهی بر اساس تازگی الگو (چند کندل پیش شکل گرفته)

---

## 🎯 هدف

**مشکل فعلی:**
- فقط آخرین کندل را چک می‌کنیم
- اگر الگو در 2-3 کندل قبل باشد، از دست می‌رود
- مثال: Hammer در 2 کندل قبل → امتیاز صفر!

**راه‌حل:**
- چک کردن N کندل آخر (مثلاً 5 کندل)
- هر کندل یک ضریب دارد بر اساس فاصله‌اش از حال
- الگوهای تازه‌تر امتیاز بیشتر می‌گیرند

---

## 📊 سیستم امتیازدهی

### ضرایب پیش‌فرض (قابل تنظیم):

| Candles Ago | Multiplier | امتیاز نسبی |
|-------------|------------|-------------|
| 0 (آخرین) | 1.0 | 100% |
| 1 | 0.9 | 90% |
| 2 | 0.8 | 80% |
| 3 | 0.7 | 70% |
| 4 | 0.6 | 60% |
| 5 | 0.5 | 50% |

**مثال:**
- Hammer در کندل آخر → ضریب = 1.0 → امتیاز کامل
- Hammer در 2 کندل قبل → ضریب = 0.8 → 80% امتیاز
- Doji در 4 کندل قبل → ضریب = 0.6 → 60% امتیاز

---

## 🔧 تغییرات مورد نیاز

### 1. Config Updates

برای هر الگو در `config.json`:

```json
{
  "patterns": {
    "hammer": {
      "enabled": true,
      "weight": 3,
      "min_candles": 12,
      "lookback_window": 5,
      "recency_multipliers": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    },
    "shooting_star": {
      "enabled": true,
      "weight": 3,
      "min_candles": 12,
      "lookback_window": 5,
      "recency_multipliers": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    },
    "engulfing": {
      "enabled": true,
      "weight": 4,
      "min_candles": 3,
      "lookback_window": 5,
      "recency_multipliers": [1.0, 0.95, 0.85, 0.7, 0.5, 0.3]
    },
    "doji": {
      "enabled": true,
      "weight": 1,
      "min_candles": 11,
      "lookback_window": 5,
      "recency_multipliers": [1.0, 0.7, 0.5, 0.3, 0.1, 0.05]
    }
  }
}
```

**توضیحات:**
- `min_candles`: حداقل کندل برای detection (از تست‌ها)
- `lookback_window`: چند کندل آخر را چک کنیم (معمولاً 5)
- `recency_multipliers`: ضریب هر کندل (index 0 = آخرین)

**چرا multipliers متفاوت؟**
- **Engulfing**: الگوی قوی → حتی 2 کندل قبل هم خوب است (0.85)
- **Doji**: الگوی ضعیف → فقط همین الان مهم است (0.7, 0.5, 0.3, ...)

---

### 2. BasePattern Changes

```python
class BasePattern:
    def __init__(self, config: Dict[str, Any] = None):
        # ... existing code ...

        # Recency scoring parameters
        pattern_name_lower = self._get_pattern_name().lower().replace(' ', '_')
        pattern_config = config.get('patterns', {}).get(pattern_name_lower, {})

        self.lookback_window = pattern_config.get('lookback_window', 5)
        self.recency_multipliers = pattern_config.get(
            'recency_multipliers',
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # default
        )

    def detect(self, df: pd.DataFrame, ...) -> bool:
        """
        Detect pattern in last N candles (lookback_window).

        Returns:
            bool: True if pattern detected in any of the last N candles
        """
        # Implementation will vary per pattern
        pass

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detection details including recency information.

        Returns:
            {
                'location': 'recent',  # or 'current'
                'candles_ago': 2,      # 0-5
                'recency_multiplier': 0.8,  # based on candles_ago
                'confidence': 0.85,
                'metadata': {...}
            }
        """
        pass
```

---

### 3. Pattern Implementation Example (Hammer)

```python
class HammerPattern(BasePattern):
    def detect(self, df: pd.DataFrame, ...) -> bool:
        """
        Detect Hammer in last N candles.

        Strategy:
        1. Check minimum candles (12 for Hammer)
        2. Run TA-Lib on full data
        3. Check last N candles (lookback_window)
        4. Return True if found in any
        """
        if not self._validate_dataframe(df):
            return False

        # Minimum candles check
        MIN_CANDLES = 12
        if len(df) < MIN_CANDLES:
            return False

        try:
            # Run TA-Lib on full data
            result = talib.CDLHAMMER(
                df['open'].values,
                df['high'].values,
                df['low'].values,
                df['close'].values
            )

            # Check last N candles
            lookback = min(self.lookback_window, len(result))

            for i in range(lookback):
                idx = -(i + 1)  # -1, -2, -3, ...
                if result[idx] != 0:
                    # Found! Store for later
                    self._last_detection_candles_ago = i
                    return True

            return False

        except Exception as e:
            return False

    def _get_detection_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detection details with recency scoring.
        """
        if len(df) == 0:
            return super()._get_detection_details(df)

        # Find which candle has the pattern
        candles_ago = getattr(self, '_last_detection_candles_ago', 0)

        # Get recency multiplier
        if candles_ago < len(self.recency_multipliers):
            recency_multiplier = self.recency_multipliers[candles_ago]
        else:
            recency_multiplier = 0.0  # Too old

        # Get the actual candle
        candle_idx = -(candles_ago + 1)
        candle = df.iloc[candle_idx]

        # Calculate quality metrics for this candle
        quality_metrics = self._calculate_quality_metrics(candle, df)

        # Calculate confidence with recency
        base_confidence = 0.75  # base for Hammer
        confidence = min(
            base_confidence * recency_multiplier,
            0.95
        )

        return {
            'location': 'current' if candles_ago == 0 else 'recent',
            'candles_ago': candles_ago,
            'recency_multiplier': recency_multiplier,
            'confidence': confidence,
            'metadata': {
                **quality_metrics,
                'recency_info': {
                    'candles_ago': candles_ago,
                    'multiplier': recency_multiplier,
                    'lookback_window': self.lookback_window
                }
            }
        }
```

---

### 4. Scoring System Integration

```python
class PatternAnalyzer:
    def calculate_score(self, pattern_result: Dict) -> float:
        """
        Calculate final score with recency multiplier.

        Before:
          score = base_weight * confidence

        After:
          score = base_weight * confidence * recency_multiplier
        """
        base_weight = pattern_result['weight']
        confidence = pattern_result['details']['confidence']
        recency_multiplier = pattern_result['details'].get('recency_multiplier', 1.0)

        # Final score
        score = base_weight * confidence * recency_multiplier

        return score
```

---

## 📈 مثال‌های عملی

### مثال 1: Hammer در کندل آخر

```
Input:
  - Pattern: Hammer
  - Candles ago: 0
  - Base confidence: 0.85

Calculation:
  - Recency multiplier: 1.0 (index 0)
  - Final confidence: 0.85 * 1.0 = 0.85
  - Score: 3 (weight) * 0.85 * 1.0 = 2.55

Output:
  {
    'pattern': 'Hammer',
    'candles_ago': 0,
    'recency_multiplier': 1.0,
    'confidence': 0.85,
    'score': 2.55
  }
```

### مثال 2: Hammer در 2 کندل قبل

```
Input:
  - Pattern: Hammer
  - Candles ago: 2
  - Base confidence: 0.85

Calculation:
  - Recency multiplier: 0.8 (index 2)
  - Final confidence: 0.85 * 0.8 = 0.68
  - Score: 3 (weight) * 0.68 * 0.8 = 1.63

Output:
  {
    'pattern': 'Hammer',
    'candles_ago': 2,
    'recency_multiplier': 0.8,
    'confidence': 0.68,
    'score': 1.63
  }
```

### مثال 3: Doji در 4 کندل قبل

```
Input:
  - Pattern: Doji
  - Candles ago: 4
  - Base confidence: 0.70

Calculation:
  - Recency multiplier: 0.1 (Doji has aggressive decay)
  - Final confidence: 0.70 * 0.1 = 0.07
  - Score: 1 (weight) * 0.07 * 0.1 = 0.007

Output:
  {
    'pattern': 'Doji',
    'candles_ago': 4,
    'recency_multiplier': 0.1,
    'confidence': 0.07,
    'score': 0.007
  }
```

---

## 🎲 تنظیم پارامترها (Tuning)

### چگونه بفهمیم ضرایب درست هستند؟

1. **Backtesting:**
   - تست با داده‌های تاریخی
   - مقایسه نتایج معاملات با ضرایب مختلف

2. **Analysis:**
   ```
   اگر دیدیم:
     - Hammer در 1 کندل قبل → موفقیت 85% (خوب!)
     - Hammer در 3 کندل قبل → موفقیت 40% (ضعیف)

   پس:
     - multipliers[1] = 0.9 ✅ (خوب است)
     - multipliers[3] = 0.7 → کم کنیم به 0.4
   ```

3. **Pattern-specific tuning:**
   ```python
   # الگوهای قوی - decay کند
   "engulfing": [1.0, 0.95, 0.85, 0.7, 0.5, 0.3]

   # الگوهای متوسط - decay معمولی
   "hammer": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

   # الگوهای ضعیف - decay سریع
   "doji": [1.0, 0.7, 0.5, 0.3, 0.1, 0.05]
   ```

---

## 📊 Logging & Debugging

### Log Format:

```
[PATTERN] Hammer detected at candles_ago=2, multiplier=0.8, score=1.63
  - Base confidence: 0.85
  - Recency multiplier: 0.8
  - Final confidence: 0.68
  - Weight: 3
  - Final score: 1.63
```

### Metadata in Signal:

```python
{
  'timestamp': '2025-10-25 12:00:00',
  'patterns': [
    {
      'name': 'Hammer',
      'candles_ago': 2,
      'recency_multiplier': 0.8,
      'confidence': 0.68,
      'score': 1.63,
      'metadata': {
        'body_size': 389.10,
        'lower_shadow': 548.40,
        'recency_info': {
          'candles_ago': 2,
          'multiplier': 0.8,
          'lookback_window': 5
        }
      }
    }
  ],
  'total_score': 1.63
}
```

---

## ✅ Checklist پیاده‌سازی

### Phase 1: Config & Base
- [ ] اضافه کردن `lookback_window` به config
- [ ] اضافه کردن `recency_multipliers` به config (برای هر الگو)
- [ ] اضافه کردن به `BasePattern.__init__`
- [ ] تست config loading

### Phase 2: Pattern Detection
- [ ] تغییر `detect()` برای چک کردن N کندل آخر
- [ ] ذخیره `candles_ago` در instance variable
- [ ] تست با Hammer
- [ ] تست با Engulfing
- [ ] تست با بقیه الگوها

### Phase 3: Scoring
- [ ] تغییر `_get_detection_details()` برای برگرداندن recency info
- [ ] تغییر scoring system برای اعمال multiplier
- [ ] تست محاسبات

### Phase 4: Testing & Validation
- [ ] Unit tests برای recency scoring
- [ ] Integration tests با داده واقعی
- [ ] Backtesting با multipliers مختلف

### Phase 5: Monitoring & Tuning
- [ ] اضافه کردن logging
- [ ] Dashboard برای نمایش candles_ago
- [ ] تحلیل نتایج و تنظیم multipliers

---

## 🎯 نتایج تست‌های شما

بر اساس نتایج:

| Pattern | Min Candles | Lookback | پیشنهاد Multipliers |
|---------|-------------|----------|-------------------|
| Engulfing | 3 | 2 | [1.0, 0.95, 0.85] |
| Hammer | 12 | 11 | [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] |
| Shooting Star | 12 | 11 | [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] |
| Doji | 11 | 10 | [1.0, 0.7, 0.5, 0.3, 0.1, 0.05] |
| Morning Star | 13 | 12 | [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] |
| Inverted Hammer | 12 | 11 | [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] |

---

## 🚀 مزایا

1. **گرفتن الگوهای اخیر:**
   - قبلاً: فقط کندل آخر
   - حالا: 5 کندل آخر

2. **امتیازدهی هوشمند:**
   - الگوهای تازه‌تر → امتیاز بیشتر
   - الگوهای قدیمی‌تر → امتیاز کمتر

3. **قابل تنظیم:**
   - هر الگو multipliers خودش را دارد
   - با backtesting بهینه می‌کنیم

4. **Transparency:**
   - می‌دانیم چه الگویی در چه زمانی پیدا شده
   - تحلیل معاملات راحت‌تر می‌شود

---

**نوشته شده توسط:** Development Team
**تاریخ:** 2025-10-25
**وضعیت:** Ready for Implementation
