#!/usr/bin/env python3
"""Multi-Agent Interview Coach - CLI Interface."""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.core.orchestrator import InterviewOrchestrator


def print_banner():
    """Print the application banner."""
    print("=" * 60)
    print("       MULTI-AGENT INTERVIEW COACH")
    print("       Technical Interview Simulator")
    print("=" * 60)
    print()


def get_candidate_info() -> tuple[str, str, str, str]:
    """Get candidate information from user input."""
    print("Please provide the following information:")
    print("-" * 40)

    name = input("Candidate Name: ").strip()
    if not name:
        name = "Candidate"

    position = input("Position (e.g., Backend Developer): ").strip()
    if not position:
        position = "Software Developer"

    grade = input("Target Grade (Junior/Middle/Senior): ").strip()
    if grade.lower() not in ["junior", "middle", "senior"]:
        print(f"Invalid grade '{grade}', defaulting to Junior")
        grade = "Junior"

    experience = input("Brief experience description: ").strip()
    if not experience:
        experience = "No experience provided"

    return name, position, grade, experience


def run_interactive_interview():
    """Run an interactive interview session."""
    print_banner()

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: No API key found!")
        print("Please set one of the following environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENROUTER_API_KEY")
        print()
        print("Example:")
        print("  export OPENROUTER_API_KEY=your-key-here")
        print("  export OPENROUTER_MODEL=anthropic/claude-3.5-sonnet")
        sys.exit(1)

    # Determine which provider to use
    if os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        print(f"Using OPENROUTER as LLM provider with model: {model}")
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
        print(f"Using OPENAI as LLM provider")
    else:
        provider = "anthropic"
        print(f"Using ANTHROPIC as LLM provider")
    print()

    # Get candidate info
    name, position, grade, experience = get_candidate_info()
    print()

    # Initialize orchestrator
    try:
        orchestrator = InterviewOrchestrator(
            llm_provider=provider,
            output_dir="logs",
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        sys.exit(1)

    # Start interview
    print("=" * 60)
    print("       INTERVIEW STARTING")
    print("=" * 60)
    print()
    print("Commands:")
    print("  - Type your responses normally")
    print("  - Say 'стоп игра' or 'stop interview' to end and get feedback")
    print("  - Press Ctrl+C to exit without feedback")
    print()
    print("-" * 60)

    try:
        greeting = orchestrator.start_interview(name, position, grade, experience)
        print(f"\nInterviewer: {greeting}\n")

        while orchestrator.is_interview_active():
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                response, ended = orchestrator.process_response(user_input)
                print(f"\nInterviewer: {response}\n")

                if ended:
                    break

            except KeyboardInterrupt:
                print("\n\nInterview interrupted by user.")
                break

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nThank you for using Multi-Agent Interview Coach!")


def run_scripted_interview(script_file: str):
    """Run an interview using a script file.

    Script file format (one line per turn):
    NAME: <name>
    POSITION: <position>
    GRADE: <grade>
    EXPERIENCE: <experience>
    ---
    <response 1>
    <response 2>
    ...
    """
    print_banner()

    if not os.path.exists(script_file):
        print(f"ERROR: Script file not found: {script_file}")
        sys.exit(1)

    with open(script_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse script
    if "---" not in content:
        print("ERROR: Invalid script format. Missing '---' separator.")
        sys.exit(1)

    header, responses_text = content.split("---", 1)

    # Parse header
    info = {}
    for line in header.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip().upper()] = value.strip()

    name = info.get("NAME", "Candidate")
    position = info.get("POSITION", "Software Developer")
    grade = info.get("GRADE", "Junior")
    experience = info.get("EXPERIENCE", "")

    # Parse responses
    responses = [r.strip() for r in responses_text.strip().split("\n") if r.strip()]

    print(f"Running scripted interview for: {name}")
    print(f"Position: {position}, Grade: {grade}")
    print(f"Responses to process: {len(responses)}")
    print()

    # Determine provider
    if os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
    else:
        provider = "anthropic"

    # Initialize and run
    orchestrator = InterviewOrchestrator(llm_provider=provider, output_dir="logs")

    print("=" * 60)
    greeting = orchestrator.start_interview(name, position, grade, experience)
    print(f"\nInterviewer: {greeting}\n")

    for i, response in enumerate(responses, 1):
        print(f"You [{i}]: {response}\n")
        reply, ended = orchestrator.process_response(response)
        print(f"Interviewer: {reply}\n")
        print("-" * 40)

        if ended:
            break

    print("\nScript completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Script mode
        run_scripted_interview(sys.argv[1])
    else:
        # Interactive mode
        run_interactive_interview()
