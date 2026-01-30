# Multi-Agent Interview Coach — Демо (5 минут)

## Что это?

**AI-powered симулятор технического собеседования** с несколькими специализированными LLM-агентами, которые совместно проводят адаптивное интервью с кандидатом на русском языке.

---

## Архитектура: Multi-Agent System

Система реализует паттерн **hidden reflection** — агенты общаются между собой внутренне, прежде чем ответить кандидату.

### Основные агенты

```
┌─────────────────────────────────────────────────────────────┐
│                    InterviewerAgent                         │
│  Задает вопросы, адаптирует сложность, ведет диалог        │
└─────────────────────────────────────────────────────────────┘
                            ↓ получает анализ
┌─────────────────────────────────────────────────────────────┐
│                     ObserverAgent                           │
│  Анализирует ответы за кулисами:                           │
│  • Оценивает качество ответа (excellent/good/poor)         │
│  • Определяет уверенность кандидата (high/medium/low)      │
│  • Детектирует галлюцинации (Python 4.0, несуществующие    │
│    фичи)                                                    │
│  • Рекомендует действие (continue/increase_difficulty/      │
│    correct_gently)                                          │
│  • Отслеживает покрытые темы                               │
└─────────────────────────────────────────────────────────────┘
                            ↓ по завершении
┌─────────────────────────────────────────────────────────────┐
│              FeedbackGeneratorAgent                         │
│  Генерирует итоговый фидбек:                               │
│  • Hiring verdict (hire/maybe/no_hire)                     │
│  • Assessed grade (Junior/Middle/Senior)                   │
│  • Confirmed skills & knowledge gaps                       │
│  • Soft skills assessment (clarity, honesty, engagement)   │
│  • Personalized learning roadmap                           │
└─────────────────────────────────────────────────────────────┘
```

### Как они взаимодействуют?

**На каждом шаге интервью:**

1. **Кандидат** отвечает на вопрос
2. **ObserverAgent** анализирует ответ → выдает `ObserverAnalysis` (structured output)
3. **InterviewerAgent** получает анализ → использует его для формирования следующего вопроса
4. Внутренние мысли агентов логируются, но **не показываются кандидату**

**Это создает контекстную осведомленность** — каждый агент помнит всю историю диалога и адаптирует свое поведение.

**Пример:**
- Кандидат говорит: "Я бы использовал Python 4.0 для async/await"
- **Observer** детектирует галлюцинацию (Python 4.0 не существует)
- **Interviewer** мягко корректирует: "Хочу уточнить: async/await был добавлен в Python 3.5, а Python 4.0 пока не выпущен. Можете рассказать подробнее?"

---

## Structured Output — Гарантия Надежности

Все агенты используют **Pydantic models + LLM structured output** для гарантированной валидации данных.

### Примеры моделей

**ObserverAnalysis:**
```python
class ObserverAnalysis(BaseModel):
    answer_quality: Literal["excellent", "good", "poor", "off_topic"]
    confidence_level: Literal["high", "medium", "low"]
    hallucination_detected: bool
    recommended_action: Literal["continue", "increase_difficulty",
                                "decrease_difficulty", "correct_gently"]
    topics_covered: List[str]
    key_observations: List[str]
```

**FinalFeedback:**
```python
class FinalFeedback(BaseModel):
    hiring_recommendation: HiringRecommendation  # enum
    assessed_grade: Grade  # enum
    confidence_score: int  # 0-100
    confirmed_skills: List[SkillAssessment]
    knowledge_gaps: List[SkillAssessment]
    soft_skills: SoftSkillsAssessment
    roadmap: List[str]
```

**Преимущества:**
- ✅ Никаких ошибок парсинга — LLM гарантированно выдает валидный JSON
- ✅ Type safety на всех этапах
- ✅ Автоматическая валидация полей (enum constraints, ranges)
- ✅ Простая интеграция с кодом

---

## Multi-Model Feedback Generation

Система поддерживает **мульти-модельную генерацию фидбека** для уменьшения bias и повышения качества оценки.

### Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluator Models                         │
│  (Независимо оценивают кандидата)                          │
│                                                             │
│  Model 1: google/gemini-2.0-flash → Independent Feedback   │
│  Model 2: anthropic/claude-3.5-sonnet → Independent Feedback│
└─────────────────────────────────────────────────────────────┘
                            ↓ оба фидбека передаются
┌─────────────────────────────────────────────────────────────┐
│                   Aggregator Model                          │
│  Model 3: openai/gpt-4o                                    │
│                                                             │
│  Анализирует оба фидбека:                                  │
│  • Находит области консенсуса (высокая уверенность)       │
│  • Анализирует расхождения (принимает взвешенное решение) │
│  • Синтезирует финальный сбалансированный фидбек          │
└─────────────────────────────────────────────────────────────┘
```

### Настройка

```bash
# .env
FEEDBACK_MODE=multi_model

# Evaluators (минимум 2)
FEEDBACK_EVALUATOR_MODELS=google/gemini-2.0-flash,anthropic/claude-3.5-sonnet

# Aggregator (1 модель для синтеза)
FEEDBACK_AGGREGATOR_MODEL=openai/gpt-4o
```

### Преимущества
- **Reduced Bias:** Разные модели → минимизация индивидуальных bias
- **Improved Quality:** Разнообразие перспектив → более сбалансированная оценка
- **Higher Confidence:** Согласие между моделями → усиленная уверенность
- **Better Coverage:** Каждая модель может заметить то, что пропустили другие

### Trade-off
- **Стоимость:** ~3x ($0.16 vs $0.05 за фидбек)
- **Latency:** Последовательное выполнение 3 LLM вызовов

→ Используется для high-stakes интервью, где качество критично

---

## Evaluation System — Тестирование Агентов

Система включает **полный offline evaluation pipeline** для тестирования агентов с разными версиями промптов.

### 1. Генерация Датасетов (`generate_dataset.py`)

Использует LLM для создания синтетических тестовых кейсов.

**Два режима:**

#### a) Single-Agent Mode
Генерирует тест-кейсы для отдельного агента:

```bash
# Генерировать 10 кейсов для ObserverAgent
python scripts/generate_dataset.py --agent-name observer --num-cases 10
```

Генерируются кейсы с разными сценариями:
- Хороший ответ с высокой уверенностью
- Плохой ответ с низкой уверенностью
- Галлюцинации (Python 4.0, несуществующие API)
- Off-topic ответы
- Ответы про неправильный framework (Django вместо Rails)

#### b) Full-Session Mode
Генерирует **полные сессии интервью** с имитацией реального диалога:

```bash
# Генерировать 5 полных сессий по 8 раундов
python scripts/generate_dataset.py --mode full-session --num-cases 5 --turns-per-session 8
```

**Процесс:**
1. **LLM генерирует профиль кандидата** (имя, должность, grade, сильные/слабые стороны, личностные черты)
2. **Симулирует интервью:**
   - Interviewer задает вопрос
   - Interviewee отвечает (с учетом своего профиля)
   - Observer анализирует ответ
   - Interviewer использует анализ для следующего вопроса
   - Цикл повторяется 8 раз
3. **Генерирует финальный фидбек**

**Ключевая фича:** Использует **structured output** для всех генераций:
- `IntervieweeProfile` (Pydantic)
- `IntervieweeResponseModel` (Pydantic)
- `ObserverAnalysisPydantic` (Pydantic)
- `InterviewerResponseModel` (Pydantic)
- `FinalFeedbackPydantic` (Pydantic)

**Результат:** Датасеты загружаются в **Langfuse** для дальнейшей оценки.

---

### 2. Оценка Агентов (`evaluate_agent.py`)

Проводит **offline evaluation** агентов с использованием **LLM-as-a-judge**.

```bash
# Оценить ObserverAgent с версией промпта 2
python scripts/evaluate_agent.py --agent-name observer --prompt-version 2

# Оценить InterviewerAgent с latest промптом
python scripts/evaluate_agent.py --agent-name interviewer --prompt-version latest
```

**Процесс:**

1. **Загружает датасет** из Langfuse (full_interview_sessions)
2. **Извлекает тест-кейсы** для конкретного агента из полных сессий:
   - Observer: 1 кейс на каждый ответ кандидата
   - Interviewer: 1 кейс на каждый ответ интервьюера
   - FeedbackGenerator: 1 кейс на всю сессию
3. **Запускает агента** с указанной версией промпта
4. **LLM-as-a-judge оценивает** выход агента с помощью structured output:

**Примеры метрик:**

**ObserverAgent:**
```python
class ObserverEvaluationScore(BaseModel):
    overall_score: float  # 0.0-1.0
    quality_assessment_score: float
    hallucination_detection_score: float
    recommended_action_score: float
    reasoning_quality_score: float
    comment: str
```

**InterviewerAgent:**
```python
class InterviewerEvaluationScore(BaseModel):
    overall_score: float
    relevance_score: float
    difficulty_appropriateness_score: float
    tone_professionalism_score: float
    topic_coverage_score: float
    comment: str
```

5. **Сохраняет результаты** в Langfuse (трейсы для каждого кейса + summary trace)

**Workflow:**
```bash
# 1. Создать baseline с текущей продакшн версией
python scripts/evaluate_agent.py --agent-name observer --prompt-version 1

# 2. Изменить промпт в Langfuse (создастся версия 2)

# 3. Протестировать новую версию
python scripts/evaluate_agent.py --agent-name observer --prompt-version 2

# 4. Сравнить результаты в Langfuse UI
# 5. Если лучше → деплоить новую версию
```

---

## Ключевые Фичи

### 1. Adaptive Difficulty
Observer анализирует качество ответов → рекомендует увеличить/уменьшить сложность → Interviewer применяет

### 2. Hallucination Detection
Observer специально обучен детектировать ложные технические утверждения (Python 4.0, несуществующие API)

### 3. Topic Tracking
Observer отслеживает покрытые темы → Interviewer избегает повторений

### 4. Centralized Prompt Management (Langfuse)
- Промпты хранятся в Langfuse с версионированием
- Можно A/B тестировать разные версии промптов
- Fallback на локальные файлы, если Langfuse недоступен

### 5. Observability (Langfuse Tracing)
- Все вызовы LLM трейсятся в Langfuse
- Можно отследить весь flow агентов
- Метрики и аналитика для оптимизации

### 6. Comprehensive Logging
- `logs/app.log` — все события (DEBUG+)
- `logs/errors.log` — только ошибки (ERROR+)
- `logs/interview_*.json` — полные транскрипты интервью

---

## Технологический Стек

- **Python 3.11+**
- **LLM Provider:** OpenRouter (доступ к Claude, GPT, Gemini и др.)
- **Structured Output:** Pydantic models
- **Observability:** Langfuse (tracing, prompt management, datasets)
- **Core Libraries:**
  - `pydantic` для валидации данных
  - `anthropic` SDK для structured outputs
  - `langfuse` для tracing и evaluation
  - `python-dotenv` для конфигурации

---

## Демо Команды

```bash
# 1. Запустить интервью в интерактивном режиме
python main.py

# 2. Запустить из скрипта
python main.py example_script.txt

# 3. Сгенерировать тестовый датасет
python scripts/generate_dataset.py --mode full-session --num-cases 5 --turns-per-session 8

# 4. Оценить агента
python scripts/evaluate_agent.py --agent-name observer --prompt-version latest

# 5. Загрузить промпты в Langfuse
python scripts/upload_prompts_to_langfuse.py
```

---

## Результаты

- ✅ **Адаптивное интервью** с учетом уровня кандидата
- ✅ **Детекция галлюцинаций** с мягкой коррекцией
- ✅ **Structured outputs** — 0 ошибок парсинга
- ✅ **Multi-model feedback** — сниженный bias, повышенное качество
- ✅ **Полная оценка системы** — LLM-as-a-judge с версионированием промптов
- ✅ **Production-ready** — логирование, tracing, error handling

---

## Архитектура Файловой Системы

```
ai-interviewer/
├── src/
│   ├── agents/              # Специализированные агенты
│   │   ├── interviewer.py
│   │   ├── observer.py
│   │   ├── feedback_generator.py
│   │   └── multi_model_feedback_generator.py
│   ├── core/
│   │   ├── orchestrator.py  # Координирует всех агентов
│   │   ├── llm_client.py    # Абстракция LLM с structured output
│   │   └── models.py        # Pydantic models
│   └── utils/
│       ├── prompt_loader.py # Загрузка промптов из Langfuse/локально
│       ├── logger.py        # Логирование интервью
│       └── tracing.py       # Langfuse tracing
├── prompts/                 # Локальные промпты (fallback)
│   ├── interviewer/
│   ├── observer/
│   ├── feedback_generator/
│   └── evaluation/          # Промпты для генерации/оценки датасетов
├── scripts/
│   ├── generate_dataset.py  # Генерация синтетических датасетов
│   ├── evaluate_agent.py    # Offline evaluation с LLM-as-judge
│   └── upload_prompts_to_langfuse.py
├── logs/                    # Логи и транскрипты интервью
└── main.py                  # Entry point
```
