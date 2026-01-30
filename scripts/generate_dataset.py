#!/usr/bin/env python3
"""
Dataset Generator Script

This script uses LLM to generate synthetic test cases for agent evaluation.

Usage:
    # Generate test cases for a single agent
    python scripts/generate_dataset.py --agent-name observer --num-cases 10
    python scripts/generate_dataset.py --agent-name interviewer --num-cases 5
    python scripts/generate_dataset.py --agent-name feedback_generator --num-cases 3

    # Generate complete interview sessions with all agents
    python scripts/generate_dataset.py --mode full-session --num-cases 5 --turns-per-session 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from src.core.llm_client import get_llm_client


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test dataset for agent evaluation"
    )
    parser.add_argument(
        "--mode",
        default="single-agent",
        choices=["single-agent", "full-session"],
        help="Generation mode: single-agent for individual agent tests, full-session for complete interviews (default: single-agent)"
    )
    parser.add_argument(
        "--agent-name",
        choices=["observer", "interviewer", "feedback_generator", "interviewee_profile", "interviewee_response"],
        help="Name of the agent to generate test cases for (required for single-agent mode)"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of test cases or interview sessions to generate (default: 10)"
    )
    parser.add_argument(
        "--turns-per-session",
        type=int,
        default=8,
        help="Number of conversation turns per session (for full-session mode, default: 8)"
    )
    parser.add_argument(
        "--dataset-name",
        help="Custom dataset name (default: {agent_name}_evaluation or full_interview_sessions)"
    )
    parser.add_argument(
        "--output-file",
        help="Save generated cases to JSON file (optional)"
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == "single-agent" and not args.agent_name:
        parser.error("--agent-name is required for single-agent mode")

    return args


def load_generator_prompt(agent_name: str, num_cases: int) -> str:
    """Load the dataset generator prompt for the agent."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "evaluation" / f"{agent_name}_dataset_generator.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Generator prompt not found: {prompt_path}")

    prompt = prompt_path.read_text()
    # Replace placeholder with actual number
    return prompt.replace("{num_cases}", str(num_cases))


def generate_test_cases(agent_name: str, num_cases: int) -> List[Dict]:
    """
    Generate test cases using LLM.

    Args:
        agent_name: Name of the agent
        num_cases: Number of test cases to generate

    Returns:
        List of test case dictionaries
    """
    llm_client = get_llm_client_instance()
    generator_prompt = load_generator_prompt(agent_name, num_cases)

    print(f"Generating {num_cases} test cases for {agent_name}...")
    print("This may take a minute...\n")

    try:
        # Generate cases as JSON
        # The generator_prompt is the system prompt, we need a user prompt too
        user_prompt = "Generate the test cases as specified in the instructions. Return them as a JSON object with a 'test_cases' key containing an array of test cases."
        result = llm_client.generate_json(generator_prompt, user_prompt)

        # The result should be a list of test cases
        if isinstance(result, list):
            test_cases = result
        elif isinstance(result, dict) and "test_cases" in result:
            test_cases = result["test_cases"]
        else:
            raise ValueError(f"Unexpected result format: {type(result)}")

        print(f"✓ Generated {len(test_cases)} test cases\n")
        return test_cases

    except Exception as e:
        print(f"✗ Error generating test cases: {e}")
        raise


def upload_to_langfuse(dataset_name: str, test_cases: List[Dict], agent_name: str):
    """
    Upload test cases to Langfuse as a dataset.

    Args:
        dataset_name: Name of the dataset
        test_cases: List of test cases
        agent_name: Name of the agent
    """
    langfuse = Langfuse()

    print(f"Uploading to Langfuse dataset '{dataset_name}'...")

    # Create or get dataset
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description=f"Evaluation dataset for {agent_name} agent",
            metadata={
                "agent": agent_name,
                "num_cases": len(test_cases),
                "generated": "true"
            }
        )
        print(f"✓ Created dataset '{dataset_name}'")
    except Exception as e:
        print(f"Note: Dataset may already exist: {e}")

    # Upload each test case
    success_count = 0
    for i, test_case in enumerate(test_cases):
        try:
            # Transform test case structure for Langfuse
            # If test case already has "input" key, use it directly
            if "input" in test_case:
                input_data = test_case["input"]
            else:
                # Otherwise, package all fields except expected_output as input
                input_data = {k: v for k, v in test_case.items() if k != "expected_output"}

            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=input_data,
                expected_output=test_case.get("expected_output"),
                metadata=test_case.get("metadata", {})
            )
            success_count += 1
            print(f"  ✓ Uploaded test case {i+1}/{len(test_cases)}")
        except Exception as e:
            print(f"  ✗ Error uploading test case {i+1}: {e}")

    print(f"\n✓ Successfully uploaded {success_count}/{len(test_cases)} test cases to Langfuse\n")


def save_to_file(output_file: str, test_cases: List[Dict]):
    """Save test cases to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"✓ Saved test cases to {output_file}\n")


def get_llm_client_instance():
    """Get configured LLM client."""
    if os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        return get_llm_client(provider, model=model)
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
        return get_llm_client(provider)
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"
        return get_llm_client(provider)
    else:
        raise ValueError(
            "No API key found! Please set one of: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
        )


def generate_interviewee_profile(llm_client) -> Dict:
    """Generate a single interviewee profile."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "evaluation" / "interviewee_profile_generator.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Generator prompt not found: {prompt_path}")

    system_prompt = prompt_path.read_text()
    user_prompt = "Generate a single interviewee profile as specified. Return it as a JSON object."

    profile = llm_client.generate_json(system_prompt, user_prompt)
    return profile


def generate_interviewee_response(llm_client, profile: Dict, question: str, conversation_history: List[Dict]) -> str:
    """Generate interviewee response based on profile and context."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "evaluation" / "interviewee_response_generator.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Generator prompt not found: {prompt_path}")

    system_prompt = prompt_path.read_text()

    user_prompt = f"""Generate the interviewee's response to the following question, considering their profile and conversation history:

Profile: {json.dumps(profile, indent=2)}

Question: {question}

Conversation history:
{json.dumps(conversation_history, indent=2)}

Return the response as a JSON object with a 'response' key."""

    result = llm_client.generate_json(system_prompt, user_prompt)

    if isinstance(result, dict) and "response" in result:
        return result["response"]
    elif isinstance(result, str):
        return result
    else:
        return str(result)


def generate_observer_analysis(llm_client, question: str, response: str, profile: Dict) -> Dict:
    """Generate observer analysis for a candidate response."""
    system_prompt = """You are an ObserverAgent that analyzes candidate responses in technical interviews.

Analyze the candidate's response and provide structured feedback including:
- quality: excellent/good/poor/off_topic
- confidence: high/medium/low
- has_hallucination: true/false (detects false technical claims)
- hallucination_details: explanation if has_hallucination is true
- recommended_action: continue/increase_difficulty/decrease_difficulty/correct_gently/ask_clarifying_question
- reasoning: explanation of the assessment
- covered_topics: list of technical topics discussed in this response

Return analysis as a JSON object."""

    user_prompt = f"""Question: {question}

Candidate response: {response}

Context - Candidate profile:
{json.dumps(profile, indent=2)}

Provide your analysis:"""

    return llm_client.generate_json(system_prompt, user_prompt)


def generate_interviewer_response(llm_client, profile: Dict, conversation_history: List[Dict],
                                  observer_analysis: Dict, covered_topics: List[str]) -> str:
    """Generate interviewer's next question or response."""
    system_prompt = """You are an InterviewerAgent conducting a technical interview.

Based on the conversation history and the ObserverAgent's analysis, generate your next response.
This could be:
- Acknowledging the answer and asking a follow-up question
- Correcting hallucinations gently
- Adjusting difficulty based on performance
- Moving to a new topic

Avoid topics that have already been thoroughly covered.

Return your response as a JSON object with a 'response' key."""

    user_prompt = f"""Candidate profile:
{json.dumps(profile, indent=2)}

Conversation history:
{json.dumps(conversation_history, indent=2)}

Observer's analysis of last response:
{json.dumps(observer_analysis, indent=2)}

Topics already covered: {', '.join(covered_topics) if covered_topics else 'None yet'}

Generate your response:"""

    result = llm_client.generate_json(system_prompt, user_prompt)

    if isinstance(result, dict) and "response" in result:
        return result["response"]
    elif isinstance(result, str):
        return result
    else:
        return str(result)


def generate_full_session(llm_client, turns: int = 8) -> Dict:
    """Generate a complete interview session with all agents."""
    print("  Generating interviewee profile...")
    profile = generate_interviewee_profile(llm_client)

    conversation_history = []
    covered_topics = []
    observer_analyses = []

    # Initial greeting
    print("  Generating initial greeting...")
    greeting = f"Hello! I'm your interviewer today. I see you're applying for the {profile.get('position', 'position')} role at {profile.get('grade', 'grade')} level. Let me start with a question: Can you tell me about your experience with {profile.get('position', 'this role')}?"

    conversation_history.append({
        "role": "interviewer",
        "content": greeting
    })

    # Generate conversation turns
    for turn in range(turns):
        print(f"  Generating turn {turn + 1}/{turns}...")

        # Interviewee responds
        current_question = conversation_history[-1]["content"]
        interviewee_response = generate_interviewee_response(
            llm_client, profile, current_question, conversation_history
        )

        conversation_history.append({
            "role": "interviewee",
            "content": interviewee_response
        })

        # Observer analyzes
        observer_analysis = generate_observer_analysis(
            llm_client, current_question, interviewee_response, profile
        )
        observer_analyses.append(observer_analysis)

        # Update covered topics
        if "covered_topics" in observer_analysis:
            for topic in observer_analysis["covered_topics"]:
                if topic not in covered_topics:
                    covered_topics.append(topic)

        # Interviewer responds (if not last turn)
        if turn < turns - 1:
            interviewer_response = generate_interviewer_response(
                llm_client, profile, conversation_history, observer_analysis, covered_topics
            )

            conversation_history.append({
                "role": "interviewer",
                "content": interviewer_response
            })

    # Generate final feedback
    print("  Generating final feedback...")
    system_prompt = """You are a FeedbackGeneratorAgent. Generate comprehensive final feedback for the interview.

Include:
- verdict: hire/maybe/no_hire
- overall_grade: assessed grade level
- technical_skills: {confirmed: [], gaps: []}
- soft_skills: {strengths: [], areas_for_improvement: []}
- recommendation: hiring recommendation text
- learning_roadmap: personalized suggestions

Return as a JSON object."""

    user_prompt = f"""Candidate profile:
{json.dumps(profile, indent=2)}

Conversation:
{json.dumps(conversation_history, indent=2)}

Observer analyses:
{json.dumps(observer_analyses, indent=2)}

Generate final feedback:"""

    final_feedback = llm_client.generate_json(system_prompt, user_prompt)

    return {
        "profile": profile,
        "conversation_history": conversation_history,
        "observer_analyses": observer_analyses,
        "covered_topics": covered_topics,
        "final_feedback": final_feedback
    }


def generate_full_sessions(num_sessions: int, turns_per_session: int) -> List[Dict]:
    """Generate multiple complete interview sessions."""
    llm_client = get_llm_client_instance()

    sessions = []

    print(f"Generating {num_sessions} complete interview sessions...")
    print(f"Each session will have {turns_per_session} conversation turns.\n")

    for i in range(num_sessions):
        print(f"Session {i + 1}/{num_sessions}:")
        try:
            session = generate_full_session(llm_client, turns_per_session)
            sessions.append(session)
            print(f"✓ Session {i + 1} complete\n")
        except Exception as e:
            print(f"✗ Error generating session {i + 1}: {e}\n")

    return sessions


def main():
    args = parse_args()

    # Determine dataset name
    if args.mode == "full-session":
        dataset_name = args.dataset_name or "full_interview_sessions"
    else:
        dataset_name = args.dataset_name or f"{args.agent_name}_evaluation"

    print(f"\n{'='*80}")
    print("Dataset Generation")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")

    if args.mode == "single-agent":
        print(f"Agent: {args.agent_name}")
        print(f"Number of cases: {args.num_cases}")
    else:
        print(f"Number of sessions: {args.num_cases}")
        print(f"Turns per session: {args.turns_per_session}")

    print(f"Dataset name: {dataset_name}")
    print(f"{'='*80}\n")

    # Generate data based on mode
    try:
        if args.mode == "full-session":
            # Generate complete interview sessions
            test_cases = generate_full_sessions(args.num_cases, args.turns_per_session)
        else:
            # Generate test cases for a single agent
            test_cases = generate_test_cases(args.agent_name, args.num_cases)
    except Exception as e:
        print(f"\nFailed to generate data: {e}")
        sys.exit(1)

    # Save to file if requested
    if args.output_file:
        save_to_file(args.output_file, test_cases)

    # Upload to Langfuse
    try:
        upload_to_langfuse(dataset_name, test_cases, args.agent_name if args.mode == "single-agent" else "full_sessions")
    except Exception as e:
        print(f"\nFailed to upload to Langfuse: {e}")
        print("Data was generated but not uploaded.")
        sys.exit(1)

    print(f"{'='*80}")
    print("✓ Dataset generation complete!")
    print(f"{'='*80}\n")

    if args.mode == "full-session":
        print("Next steps:")
        print("  1. Review the sessions in Langfuse UI or output file")
        print("  2. Use sessions for end-to-end evaluation or training")
    else:
        print("Next steps:")
        print("  1. Review the dataset in Langfuse UI")
        print("  2. Run evaluation:")
        print(f"     python scripts/evaluate_agent.py --agent-name {args.agent_name} --prompt-version latest")
    print()


if __name__ == "__main__":
    main()
