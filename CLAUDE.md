# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent Interview Coach is an AI-powered technical interview simulator built with Python. The system uses multiple specialized LLM agents that collaborate to conduct adaptive technical interviews with candidates.

## Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add OPENROUTER_API_KEY to .env
# Also set OPENROUTER_MODEL (optional, defaults to anthropic/claude-3.5-sonnet)
# Optionally add Langfuse credentials for tracing: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
```

### Running the Application
```bash
# Interactive mode (prompts for candidate info)
python main.py

# Scripted mode (runs interview from a script file)
python main.py example_script.txt
```

### Development
No test suite or linting configured currently.

### Prompt Management with Langfuse
The system supports managing prompts in Langfuse for centralized prompt versioning and editing:

```bash
# Upload all local prompts to Langfuse
python scripts/upload_prompts_to_langfuse.py
```

**How it works:**
- By default, prompts are loaded from local files in `prompts/` directory
- If Langfuse credentials are configured, the system will fetch prompts from Langfuse first
- Falls back to local files if Langfuse is unavailable or prompt doesn't exist
- Prompts are cached in memory for performance

**Naming convention:**
- Local files: `prompts/{agent_type}/{prompt_name}.txt`
- Langfuse prompts: `{agent_type}_{prompt_name}`
- Example: `prompts/interviewer/system.txt` → `interviewer_system` in Langfuse

**Available prompts:**
- `interviewer_system` - System prompt for interviewer agent
- `interviewer_greeting` - Template for initial greeting
- `interviewer_response` - Template for interviewer responses
- `observer_system` - System prompt for observer agent
- `observer_analysis` - Template for response analysis
- `feedback_generator_system` - System prompt for feedback generator
- `feedback_generator_feedback` - Template for final feedback

## Multi-Agent Architecture

The system implements a **hidden reflection** pattern where agents communicate internally before responding to the candidate:

### Agent Flow
1. **InterviewerAgent** (`src/agents/interviewer.py`) - Conducts the interview
   - Generates greeting and questions
   - Responds to candidate answers
   - Adapts difficulty based on performance
   - Handles candidate questions about the role

2. **ObserverAgent** (`src/agents/observer.py`) - Analyzes responses behind the scenes
   - Assesses answer quality and confidence
   - Detects hallucinations (false technical claims)
   - Identifies off-topic responses
   - Recommends difficulty adjustments
   - Tracks covered topics to avoid repetition
   - Guides the Interviewer on next actions

3. **FeedbackGeneratorAgent** (`src/agents/feedback_generator.py`) - Creates final report
   - Generates comprehensive feedback when interview ends
   - Assesses grade level and hiring recommendation
   - Provides technical review with confirmed skills and gaps
   - Evaluates soft skills (clarity, honesty, engagement)
   - Creates personalized learning roadmap

### Communication Pattern
On each turn:
1. Candidate provides response
2. Observer analyzes response → produces `ObserverAnalysis`
3. Interviewer receives analysis → generates response using Observer's guidance
4. Internal thoughts logged (not shown to candidate)

This creates **context awareness** - each agent maintains conversation history and adapts their behavior accordingly.

## Core Components

### Orchestrator (`src/core/orchestrator.py`)
The `InterviewOrchestrator` class coordinates all agents and manages interview lifecycle:
- Initializes agents with shared LLM client
- Maintains interview state (candidate info, turn count, active status)
- Routes messages between agents
- Handles stop commands (`стоп игра`, `stop interview`)
- Triggers feedback generation on interview completion

### LLM Abstraction (`src/core/llm_client.py`)
LLM client abstraction using OpenRouter:
- `BaseLLMClient` - Abstract interface with `generate()`, `generate_json()`, and `generate_structured()` methods
- `OpenRouterClient` - Uses `anthropic/claude-3.5-sonnet` by default, supports any OpenRouter model
- Configure model via OPENROUTER_MODEL environment variable

### Data Models (`src/core/models.py`)
Structured with dataclasses:
- `CandidateInfo` - Name, position, target grade, experience
- `ObserverAnalysis` - Answer quality, hallucination flags, recommended actions
- `FinalFeedback` - Verdict, skills assessment, soft skills, roadmap
- `InterviewSession` - Complete log with turns and feedback

### Utilities

**Interview Session Logging** (`src/utils/logger.py`)
Saves complete interview transcripts to `logs/` as JSON files including:
- Candidate metadata
- All conversation turns with visible and internal messages
- Final feedback with structured assessment

**Application Logging** (`src/utils/app_logger.py`)
Comprehensive logging system for debugging and error tracking:
- Creates `logs/app.log` with all events (DEBUG level+)
- Creates `logs/errors.log` with only errors and exceptions (ERROR level+)
- Rotating file handlers (10MB max, 5 backups each)
- Logs all LLM API calls, exceptions, and critical operations
- Console output limited to WARNING+ for clean UX
- Automatic initialization on module import
- See `LOGGING.md` for detailed usage guide

**Tracing** (`src/utils/tracing.py`)
Optional Langfuse integration for observability:
- Initializes Langfuse client with credentials from environment variables
- Provides `@observe` decorator for function tracing
- Falls back gracefully if credentials not configured
- Helps monitor agent behavior and performance

**Prompt Loader** (`src/utils/prompt_loader.py`)
Utility for loading and managing system prompts for agents:
- Fetches prompts from Langfuse if available (for centralized management)
- Falls back to local files in `prompts/` directory
- Caches prompts in memory for performance
- Naming: `{agent_type}_{prompt_name}` in Langfuse (e.g., `interviewer_system`)

## Key Design Principles

### Role Specialization
Each agent has a distinct responsibility and system prompt. Agents don't overlap in function.

### Adaptive Difficulty
Observer analyzes performance → recommends difficulty adjustment → Interviewer applies it in next question. Difficulty levels: easy, medium, hard.

### Hallucination Detection
Observer is specifically trained to detect false technical claims (e.g., "Python 4.0", non-existent features). When detected, Interviewer politely corrects the candidate.

### Topic Tracking
Observer maintains a list of covered topics. Interviewer explicitly avoids asking about topics already thoroughly discussed.

### Stop Commands
Interview ends when candidate says stop phrases (detected via regex in `orchestrator.py:148-155`), triggering feedback generation. Supported patterns:
- `стоп (игра|интервью)?` - Russian: "stop (game|interview)?"
- `stop (game|interview)?` - English equivalent
- `давай фидбэк` - Russian: "give feedback"
- `give feedback` - English equivalent
- `завершить` - Russian: "finish/end"
- `end interview` - English equivalent

## Script File Format

For scripted mode (`python main.py script.txt`):
```
NAME: Candidate Name
POSITION: Backend Developer
GRADE: Junior
EXPERIENCE: Brief description
---
First candidate response
Second candidate response
стоп игра
```

Lines before `---` are metadata (key: value format). Lines after are candidate responses (one per line).
