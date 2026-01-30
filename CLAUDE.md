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
# Add one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY to .env
# For OpenRouter, also set OPENROUTER_MODEL (optional, defaults to anthropic/claude-3.5-sonnet)
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
Factory pattern for multi-provider support:
- `BaseLLMClient` - Abstract interface with `generate()` and `generate_json()` methods
- `OpenAIClient` - Uses `gpt-4o-mini` by default
- `AnthropicClient` - Uses `claude-3-5-sonnet-20241022` by default
- `OpenRouterClient` - Uses `anthropic/claude-3.5-sonnet` by default, supports any OpenRouter model
- Provider auto-selected based on available API key (priority: OpenRouter → OpenAI → Anthropic)

### Data Models (`src/core/models.py`)
Structured with dataclasses:
- `CandidateInfo` - Name, position, target grade, experience
- `ObserverAnalysis` - Answer quality, hallucination flags, recommended actions
- `FinalFeedback` - Verdict, skills assessment, soft skills, roadmap
- `InterviewSession` - Complete log with turns and feedback

### Logging (`src/utils/logger.py`)
Saves complete interview transcripts to `logs/` as JSON files including:
- Candidate metadata
- All conversation turns with visible and internal messages
- Final feedback with structured assessment

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
Interview ends when candidate says stop phrases (detected via regex in `orchestrator.py:122-132`), triggering feedback generation.

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
