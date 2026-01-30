# Langfuse Integration - Tier 1 MVP Implementation Guide

## Overview

This document provides step-by-step instructions for integrating Langfuse observability into the Multi-Agent Interview Coach system. This is the **Tier 1 MVP** implementation that provides basic tracing with minimal code changes.

**Estimated Time:** 1-2 hours
**Complexity:** Low
**Impact:** Complete visibility into all LLM calls and interview sessions

---

## What You'll Achieve

After this integration:
- ✅ Every interview will be tracked as a trace in Langfuse
- ✅ All 11 LLM call sites automatically instrumented
- ✅ Session metadata (candidate info, position, grade) captured
- ✅ Token usage and costs tracked per interview
- ✅ Zero changes to agent code required

---

## Implementation Steps

### Step 1: Create Tracing Utility Module

**Create new file:** `src/utils/tracing.py`

```python
"""
Langfuse tracing utilities for interview observability.
"""
import os
from typing import Optional
from langfuse import Langfuse
from langfuse import observe, langfuse_context

# Initialize Langfuse client (will be None if credentials not provided)
_langfuse_client: Optional[Langfuse] = None

def initialize_langfuse() -> Optional[Langfuse]:
    """
    Initialize Langfuse client with credentials from environment.
    Returns None if credentials are not configured.
    """
    global _langfuse_client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("⚠️  Langfuse credentials not found. Tracing disabled.")
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        print("✅ Langfuse tracing enabled")
        return _langfuse_client
    except Exception as e:
        print(f"⚠️  Failed to initialize Langfuse: {e}")
        return None

def get_langfuse_client() -> Optional[Langfuse]:
    """Get the initialized Langfuse client."""
    return _langfuse_client

def is_tracing_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    return _langfuse_client is not None
```

---

### Step 2: Modify LLM Client Base Class

**File to edit:** `src/core/llm_client.py`

Add Langfuse import at the top:
```python
from langfuse import observe
```

Find the `BaseLLMClient` abstract class and add the `@observe` decorator to both methods:

**Before:**
```python
@abstractmethod
def generate(self, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """Generate text response."""
    pass

@abstractmethod
def generate_json(self, system_prompt: str, user_message: str, response_model, temperature: float = 0.7):
    """Generate structured JSON response."""
    pass
```

**After:**
```python
@observe(as_type="generation")
@abstractmethod
def generate(self, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """Generate text response."""
    pass

@observe(as_type="generation")
@abstractmethod
def generate_json(self, system_prompt: str, user_message: str, response_model, temperature: float = 0.7):
    """Generate structured JSON response."""
    pass
```

**Note:** The decorator on abstract methods will automatically apply to all concrete implementations (OpenAIClient, AnthropicClient, OpenRouterClient).

---

### Step 3: Modify Orchestrator to Track Sessions

**File to edit:** `src/core/orchestrator.py`

#### 3a. Add imports at the top:
```python
from langfuse import observe, langfuse_context
from src.utils.tracing import initialize_langfuse, is_tracing_enabled
```

#### 3b. Initialize Langfuse in `__init__` method

Find the `__init__` method (around line 26) and add initialization after the logger setup:

**Add after line 37 (after `self.logger = Logger()`):**
```python
# Initialize Langfuse tracing
self.langfuse_client = initialize_langfuse()
```

#### 3c. Add trace decorator to `run_interview` method

Find the `run_interview` method (around line 41) and add the decorator:

**Before:**
```python
def run_interview(self, candidate_responses: Optional[list[str]] = None) -> InterviewSession:
    """Run the interview with optional scripted responses."""
```

**After:**
```python
@observe(name="interview_session")
def run_interview(self, candidate_responses: Optional[list[str]] = None) -> InterviewSession:
    """Run the interview with optional scripted responses."""
```

#### 3d. Add session metadata at the start of `run_interview`

**Add right after the method begins (after line 43 - after docstring):**
```python
# Track session metadata in Langfuse
if is_tracing_enabled():
    langfuse_context.update_current_trace(
        name=f"Interview: {self.candidate_info.name}",
        user_id=self.candidate_info.name,
        metadata={
            "position": self.candidate_info.position,
            "target_grade": self.candidate_info.grade,
            "experience": self.candidate_info.experience,
            "mode": "scripted" if candidate_responses else "interactive"
        },
        tags=["interview", self.candidate_info.grade, self.candidate_info.position]
    )
```

#### 3e. Update trace with final results

Find the end of `run_interview` method (around line 134, before the return statement) and add:

**Add before `return session` (around line 134):**
```python
# Update trace with final results
if is_tracing_enabled() and final_feedback:
    langfuse_context.update_current_observation(
        output={
            "verdict": final_feedback.verdict,
            "grade": final_feedback.grade,
            "total_turns": len(session.turns),
            "hiring_recommendation": final_feedback.hiring_recommendation
        },
        metadata={
            "confirmed_skills": final_feedback.confirmed_skills,
            "skill_gaps": final_feedback.skill_gaps,
            "topics_covered": final_feedback.topics_covered
        }
    )
```

---

### Step 4: Optional - Add Spans to Agent Methods

If you want more granular visibility, add `@observe` decorators to key agent methods:

#### InterviewerAgent (`src/agents/interviewer.py`)

Add import:
```python
from langfuse import observe
```

Add decorators to these methods:
```python
@observe(name="generate_greeting")
def generate_greeting(self) -> str:
    # ... existing code

@observe(name="generate_first_question")
def generate_first_question(self) -> str:
    # ... existing code

@observe(name="generate_interviewer_response")
def generate_response(self, candidate_response: str, observer_analysis: ObserverAnalysis) -> str:
    # ... existing code
```

#### ObserverAgent (`src/agents/observer.py`)

Add import:
```python
from langfuse import observe
```

Add decorator:
```python
@observe(name="analyze_candidate_response")
def analyze_response(self, candidate_response: str) -> ObserverAnalysis:
    # ... existing code
```

**After the analysis, add metadata capture (before the return statement):**
```python
# Capture analysis metadata for observability
if analysis:
    from langfuse import langfuse_context
    langfuse_context.update_current_observation(
        metadata={
            "quality": analysis.answer_quality,
            "hallucination": analysis.hallucination_detected,
            "recommended_action": analysis.recommended_action,
            "difficulty_adjustment": analysis.difficulty_adjustment,
            "off_topic": analysis.off_topic
        }
    )
```

#### FeedbackGeneratorAgent (`src/agents/feedback_generator.py`)

Add import:
```python
from langfuse import observe
```

Add decorator:
```python
@observe(name="generate_final_feedback")
def generate_feedback(self, conversation_history: list[dict], candidate_info) -> FinalFeedback:
    # ... existing code
```

---

## Testing the Integration

### Test 1: Run Interactive Interview
```bash
python main.py
```

Conduct a short interview (2-3 turns), then say "stop interview".

### Test 2: Run Scripted Interview
```bash
python main.py example_script.txt
```
