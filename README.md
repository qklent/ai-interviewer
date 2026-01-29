# Multi-Agent Interview Coach

AI-powered technical interview simulator with multiple specialized agents.

## Architecture

The system consists of three specialized agents:

1. **Interviewer Agent** - Conducts the interview, asks questions, adapts difficulty
2. **Observer Agent** - Analyzes responses, detects hallucinations, guides Interviewer
3. **Feedback Generator Agent** - Creates comprehensive final feedback

### System Properties

- **Role Specialization**: Each agent has a distinct responsibility
- **Hidden Reflection**: Agents communicate internally before responding
- **Context Awareness**: Maintains conversation history, avoids repetition
- **Adaptability**: Question difficulty adjusts based on performance
- **Robustness**: Handles off-topic responses and hallucinations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-interviewer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API key (OpenAI or Anthropic)
```

## Usage

### Interactive Mode

```bash
python main.py
```

Follow the prompts to:
1. Enter candidate information (name, position, grade, experience)
2. Respond to interview questions
3. Say "стоп игра" or "stop interview" to end and get feedback

### Scripted Mode

Create a script file with the following format:

```
NAME: Alex
POSITION: Backend Developer
GRADE: Junior
EXPERIENCE: Django pet projects, basic SQL
---
First response here
Second response here
стоп игра
```

Run with:
```bash
python main.py script.txt
```

## Output

Interview logs are saved to `logs/` directory in JSON format:

```json
{
  "participant_name": "Alex",
  "position": "Backend Developer",
  "target_grade": "Junior",
  "turns": [
    {
      "turn_id": 1,
      "agent_visible_message": "...",
      "user_message": "...",
      "internal_thoughts": "[Observer]: ... | [Interviewer]: ..."
    }
  ],
  "final_feedback": {
    "verdict": {...},
    "technical_review": {...},
    "soft_skills": {...},
    "roadmap": {...}
  }
}
```

## Features

### Hallucination Detection

The Observer agent identifies factually incorrect statements:
- Non-existent technologies or versions (e.g., "Python 4.0")
- Made-up features or concepts

### Adaptive Difficulty

Questions adjust based on performance:
- Excellent answers → harder questions
- Struggling → simpler questions or hints

### Comprehensive Feedback

Final feedback includes:
- **Verdict**: Grade assessment, hiring recommendation, confidence score
- **Technical Review**: Confirmed skills and knowledge gaps with correct answers
- **Soft Skills**: Clarity, honesty, engagement scores
- **Roadmap**: Topics to improve and suggested resources

## Supported LLM Providers

- OpenAI (GPT-4, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet)

## License

MIT
