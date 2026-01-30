#!/usr/bin/env python3
"""
Offline Evaluation Script for Multi-Agent Interview System

This script evaluates agents using a unified dataset of full interview sessions.
It extracts agent-specific test cases from complete sessions dynamically.

Usage:
    python scripts/evaluate_agent.py --agent-name observer --prompt-version latest
    python scripts/evaluate_agent.py --agent-name interviewer --prompt-version 2
    python scripts/evaluate_agent.py --agent-name feedback_generator --prompt-version 3

    # Use custom dataset (default: full_interview_sessions)
    python scripts/evaluate_agent.py --agent-name observer --prompt-version latest --dataset-name my_dataset
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langfuse import Langfuse, Evaluation
from pydantic import BaseModel, Field
from src.core.llm_client import get_llm_client
from src.agents.observer import ObserverAgent
from src.agents.interviewer import InterviewerAgent
from src.agents.feedback_generator import FeedbackGeneratorAgent
from src.core.models import CandidateInfo, ObserverAnalysis
from src.utils.prompt_loader import load_prompt


# ============================================================================
# Pydantic Models for Evaluation Scores
# ============================================================================


class ObserverEvaluationScore(BaseModel):
    """Evaluation score for ObserverAgent output."""

    overall_score: float = Field(
        ge=0.0, le=1.0, description="Overall score from 0.0 to 1.0"
    )
    quality_assessment_score: float = Field(
        ge=0.0, le=1.0, description="How well quality was assessed"
    )
    hallucination_detection_score: float = Field(
        ge=0.0, le=1.0, description="Accuracy of hallucination detection"
    )
    recommended_action_score: float = Field(
        ge=0.0, le=1.0, description="Appropriateness of recommended action"
    )
    reasoning_quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality of reasoning provided"
    )
    comment: str = Field(description="Detailed explanation of the evaluation")


class InterviewerEvaluationScore(BaseModel):
    """Evaluation score for InterviewerAgent output."""

    overall_score: float = Field(
        ge=0.0, le=1.0, description="Overall score from 0.0 to 1.0"
    )
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Relevance to context and analysis"
    )
    difficulty_appropriateness_score: float = Field(
        ge=0.0, le=1.0, description="Appropriate difficulty level"
    )
    tone_professionalism_score: float = Field(
        ge=0.0, le=1.0, description="Professional and encouraging tone"
    )
    topic_coverage_score: float = Field(
        ge=0.0, le=1.0, description="Avoids repetition, explores new topics"
    )
    comment: str = Field(description="Detailed explanation of the evaluation")


class FeedbackGeneratorEvaluationScore(BaseModel):
    """Evaluation score for FeedbackGeneratorAgent output."""

    overall_score: float = Field(
        ge=0.0, le=1.0, description="Overall score from 0.0 to 1.0"
    )
    verdict_accuracy_score: float = Field(
        ge=0.0, le=1.0, description="Accuracy of hiring verdict"
    )
    technical_assessment_score: float = Field(
        ge=0.0, le=1.0, description="Quality of technical skills assessment"
    )
    soft_skills_assessment_score: float = Field(
        ge=0.0, le=1.0, description="Quality of soft skills assessment"
    )
    roadmap_quality_score: float = Field(
        ge=0.0, le=1.0, description="Usefulness of learning roadmap"
    )
    comment: str = Field(description="Detailed explanation of the evaluation")


# ============================================================================
# Command Line Parsing
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate agent performance with specific prompt version"
    )
    parser.add_argument(
        "--agent-name",
        required=True,
        choices=["observer", "interviewer", "feedback_generator"],
        help="Name of the agent to evaluate",
    )
    parser.add_argument(
        "--prompt-version",
        required=True,
        help="Prompt version to test (use 'latest' or specific version number)",
    )
    parser.add_argument(
        "--dataset-name",
        help="Override default dataset name (default: full_interview_sessions)",
    )
    parser.add_argument("--run-name", help="Custom name for this evaluation run")

    return parser.parse_args()


def load_evaluator_prompt(agent_name: str) -> str:
    """Load the LLM-as-judge evaluator prompt for the agent."""
    prompt_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "evaluation"
        / f"{agent_name}_evaluator.txt"
    )

    if not prompt_path.exists():
        raise FileNotFoundError(f"Evaluator prompt not found: {prompt_path}")

    return prompt_path.read_text()


def get_prompt_content(
    langfuse: Langfuse, agent_name: str, prompt_type: str, version: str
) -> str:
    """
    Get prompt content from Langfuse for specified version.

    Args:
        langfuse: Langfuse client
        agent_name: Name of the agent (observer/interviewer/feedback_generator)
        prompt_type: Type of prompt (system/greeting/response/etc)
        version: "latest" or specific version number

    Returns:
        Prompt content as string
    """
    prompt_name = f"{agent_name}_{prompt_type}"

    try:
        if version == "latest":
            # Explicitly use "latest" label instead of default "production"
            prompt = langfuse.get_prompt(prompt_name, label="latest")
        else:
            prompt = langfuse.get_prompt(prompt_name, version=int(version))

        return prompt.prompt
    except Exception as e:
        print(f"Warning: Could not load prompt '{prompt_name}' version {version} from Langfuse: {e}")
        print("Falling back to local file...")
        return load_prompt(agent_name, prompt_type)


# ============================================================================
# Data Extraction from Full Sessions
# ============================================================================


def transform_conversation_history(raw_history: List[Dict]) -> List[Dict]:
    """
    Transform conversation history from dataset format to agent format.

    Dataset format: [{"role": "interviewer"/"interviewee", "content": "..."}]
    Agent format: [{"agent_message": "...", "user_message": "..."}]

    Args:
        raw_history: Conversation history in dataset format

    Returns:
        Conversation history in agent format
    """
    transformed = []
    for i in range(0, len(raw_history) - 1, 2):
        if i + 1 < len(raw_history):
            if raw_history[i]["role"] == "interviewer":
                transformed.append({
                    "agent_message": raw_history[i]["content"],
                    "user_message": raw_history[i + 1]["content"]
                })
    return transformed


def extract_observer_test_cases(session: Dict) -> List[Dict]:
    """
    Extract individual observer test cases from a full session.

    Creates one test case per conversation turn where the interviewee responded.

    Args:
        session: Full interview session with profile, conversation_history, observer_analyses, etc.

    Returns:
        List of test cases with input/expected_output for ObserverAgent
    """
    test_cases = []
    conversation_history = session.get("conversation_history", [])
    observer_analyses = session.get("observer_analyses", [])

    # Track cumulative covered topics up to each turn
    cumulative_topics = []

    # Find all interviewee responses and match them with observer analyses
    turn_idx = 0
    for i in range(len(conversation_history) - 1):
        if conversation_history[i]["role"] == "interviewer":
            # Check if next message is from interviewee
            if i + 1 < len(conversation_history) and conversation_history[i + 1]["role"] == "interviewee":
                question = conversation_history[i]["content"]
                candidate_response = conversation_history[i + 1]["content"]

                # Get observer analysis for this turn (if available)
                if turn_idx < len(observer_analyses):
                    expected_analysis = observer_analyses[turn_idx]

                    # Update cumulative topics with new topics from this analysis
                    if "covered_topics" in expected_analysis:
                        for topic in expected_analysis["covered_topics"]:
                            if topic not in cumulative_topics:
                                cumulative_topics.append(topic)

                    test_cases.append({
                        "input": {
                            "question": question,
                            "candidate_response": candidate_response,
                            "conversation_history": conversation_history[:i+2],  # Up to current turn
                            "covered_topics": cumulative_topics.copy(),
                        },
                        "expected_output": expected_analysis
                    })
                    turn_idx += 1

    return test_cases


def extract_interviewer_test_cases(session: Dict) -> List[Dict]:
    """
    Extract individual interviewer test cases from a full session.

    Creates one test case per interviewer response (excluding initial greeting).

    Args:
        session: Full interview session

    Returns:
        List of test cases with input/expected_output for InterviewerAgent
    """
    test_cases = []
    conversation_history = session.get("conversation_history", [])
    observer_analyses = session.get("observer_analyses", [])
    profile = session.get("profile", {})

    # Track cumulative covered topics
    cumulative_topics = []

    # Find all interviewer responses after the initial greeting
    analysis_idx = 0
    for i in range(len(conversation_history)):
        if conversation_history[i]["role"] == "interviewer" and i > 0:
            # This is an interviewer response after a candidate answer
            # Get the observer analysis for the previous candidate response
            if analysis_idx < len(observer_analyses):
                observer_analysis = observer_analyses[analysis_idx]

                # Update cumulative topics
                if "covered_topics" in observer_analysis:
                    for topic in observer_analysis["covered_topics"]:
                        if topic not in cumulative_topics:
                            cumulative_topics.append(topic)

                test_cases.append({
                    "input": {
                        "position": profile.get("position", "Unknown"),
                        "grade": profile.get("grade", "Unknown"),
                        "conversation_history": conversation_history[:i],  # Up to but not including this response
                        "observer_analysis": observer_analysis,
                        "covered_topics": cumulative_topics.copy(),
                    },
                    "expected_output": {
                        "quality_criteria": {
                            "note": "No explicit expected output available from full session",
                            "actual_response": conversation_history[i]["content"]
                        }
                    }
                })
                analysis_idx += 1

    return test_cases


def extract_feedback_generator_test_case(session: Dict) -> Dict:
    """
    Extract feedback generator test case from a full session.

    Uses the entire session to create a single test case.

    Args:
        session: Full interview session

    Returns:
        Single test case with input/expected_output for FeedbackGeneratorAgent
    """
    profile = session.get("profile", {})

    return {
        "input": {
            "position": profile.get("position", "Unknown"),
            "target_grade": profile.get("grade", "Unknown"),
            "conversation_history": session.get("conversation_history", []),
            "observer_analyses": session.get("observer_analyses", []),
        },
        "expected_output": session.get("final_feedback", {})
    }


# ============================================================================
# Observer Agent Evaluation
# ============================================================================


def evaluate_observer_agent(
    item: Dict, langfuse: Langfuse, prompt_version: str
) -> Dict:
    """
    Run ObserverAgent with specified prompt version and return analysis.

    Args:
        item: Dataset item with input and expected_output
        langfuse: Langfuse client
        prompt_version: Prompt version to use

    Returns:
        Dictionary with ObserverAgent's analysis
    """
    llm_client = get_llm_client()
    observer = ObserverAgent(llm_client)

    # Override system prompt with specified version
    system_prompt = get_prompt_content(langfuse, "observer", "system", prompt_version)
    observer.system_prompt = system_prompt

    # Extract input
    question = item["input"]["question"]
    candidate_response = item["input"]["candidate_response"]
    raw_conversation_history = item["input"].get("conversation_history", [])

    # Transform conversation history to agent format
    conversation_history = transform_conversation_history(raw_conversation_history)

    # Create a minimal candidate info (observer needs this but doesn't use much of it)
    candidate_info = CandidateInfo(
        name="Test Candidate",
        position="Software Engineer",
        target_grade="Middle",
        experience="N/A",
    )

    # Run analysis
    analysis = observer.analyze_response(
        candidate_info=candidate_info,
        conversation_history=conversation_history,
        current_question=question,
        candidate_response=candidate_response,
    )

    # Convert to dict for evaluation (map actual model fields to dataset format)
    return {
        "quality": analysis.answer_quality,
        "confidence": analysis.confidence_level,
        "has_hallucination": analysis.hallucination_detected,
        "hallucination_details": ", ".join(analysis.key_observations) if analysis.key_observations else None,
        "recommended_action": analysis.recommended_action,
        "reasoning": ", ".join(analysis.key_observations) if analysis.key_observations else "",
        "covered_topics": analysis.topics_covered,
    }


def create_observer_evaluator(langfuse: Langfuse):
    """Create LLM-as-a-judge evaluator for ObserverAgent."""
    evaluator_prompt = load_evaluator_prompt("observer")
    llm_client = get_llm_client()

    def observer_evaluator(
        *, input: Dict, output: Dict, expected_output: Dict, **kwargs
    ) -> Evaluation:
        """Evaluate ObserverAgent output using LLM-as-judge."""

        # Build prompt for evaluator
        eval_input = f"""
QUESTION: {input["question"]}

CANDIDATE RESPONSE: {input["candidate_response"]}

ACTUAL OUTPUT (ObserverAgent's analysis):
{json.dumps(output, indent=2)}

EXPECTED OUTPUT (Correct analysis):
{json.dumps(expected_output, indent=2)}

{evaluator_prompt}
"""

        try:
            # Get evaluation from LLM using structured output
            result = llm_client.generate_structured(
                system_prompt="You are an expert evaluator for ObserverAgent outputs. Provide structured evaluation scores.",
                user_prompt=eval_input,
                response_format=ObserverEvaluationScore,
            )

            return Evaluation(
                name="observer_llm_judge",
                value=result.overall_score,
                comment=result.comment,
                metadata=result.model_dump(),  # Store all individual scores
            )
        except Exception as e:
            return Evaluation(
                name="observer_llm_judge",
                value=0.0,
                comment=f"Evaluation failed: {str(e)}",
            )

    return observer_evaluator


# ============================================================================
# Interviewer Agent Evaluation
# ============================================================================


def evaluate_interviewer_agent(
    item: Dict, langfuse: Langfuse, prompt_version: str
) -> Dict:
    """
    Run InterviewerAgent with specified prompt version and return response.
    """
    try:
        print("    → Getting LLM client...")
        llm_client = get_llm_client()

        print("    → Creating candidate info...")
        # Create candidate info from item
        candidate_info = CandidateInfo(
            name="Test Candidate",
            position=item["input"]["position"],
            target_grade=item["input"]["grade"],
            experience="N/A",
        )

        print("    → Creating InterviewerAgent...")
        interviewer = InterviewerAgent(llm_client)

        print("    → Loading system prompt...")
        # Override system prompt
        system_prompt = get_prompt_content(
            langfuse, "interviewer", "system", prompt_version
        )
        interviewer.system_prompt = system_prompt

        print("    → Extracting conversation history...")
        # Extract input
        raw_conversation_history = item["input"]["conversation_history"]
        observer_analysis_data = item["input"]["observer_analysis"]

        # Transform conversation history from dataset format (role/content) to agent format (agent_message/user_message)
        conversation_history = transform_conversation_history(raw_conversation_history)

        print("    → Extracting current question and response...")
        # Extract current question and candidate response from raw conversation history
        current_question = ""
        candidate_response = ""
        if len(raw_conversation_history) >= 2:
            # Find the last interviewer question and candidate response pair
            for i in range(len(raw_conversation_history) - 1, -1, -1):
                if raw_conversation_history[i]["role"] == "interviewee":
                    candidate_response = raw_conversation_history[i]["content"]
                    # Find the previous interviewer question
                    for j in range(i - 1, -1, -1):
                        if raw_conversation_history[j]["role"] == "interviewer":
                            current_question = raw_conversation_history[j]["content"]
                            break
                    break

        print("    → Creating ObserverAnalysis object...")
        # Convert observer_analysis dict to ObserverAnalysis object
        # Map from dataset format to actual ObserverAnalysis model
        observer_analysis = ObserverAnalysis(
            answer_quality=observer_analysis_data.get("quality", "good"),
            confidence_level=observer_analysis_data.get("confidence", "medium"),
            factual_accuracy=not observer_analysis_data.get("has_hallucination", False),
            hallucination_detected=observer_analysis_data.get("has_hallucination", False),
            off_topic=observer_analysis_data.get("quality") == "off_topic",
            candidate_question_detected=False,
            recommended_action=observer_analysis_data.get("recommended_action", "continue"),
            difficulty_adjustment="maintain",
            topics_covered=observer_analysis_data.get("covered_topics", []),
            key_observations=[observer_analysis_data.get("reasoning", "")],
        )

        print("    → Calling generate_response...")
        # Generate response
        response, internal = interviewer.generate_response(
            candidate_info=candidate_info,
            conversation_history=conversation_history,
            current_question=current_question,
            candidate_response=candidate_response,
            observer_analysis=observer_analysis,
        )

        print("    → Response generated successfully")
        return {"response": response, "internal_thoughts": internal}
    except Exception as e:
        import traceback
        print(f"    ✗ Error in evaluate_interviewer_agent: {e}")
        print("    ✗ Detailed traceback:")
        traceback.print_exc()
        raise


def create_interviewer_evaluator(langfuse: Langfuse):
    """Create LLM-as-a-judge evaluator for InterviewerAgent."""
    evaluator_prompt = load_evaluator_prompt("interviewer")
    llm_client = get_llm_client()

    def interviewer_evaluator(
        *, input: Dict, output: Dict, expected_output: Dict, **kwargs
    ) -> Evaluation:
        """Evaluate InterviewerAgent output using LLM-as-judge."""

        eval_input = f"""
POSITION: {input["position"]}
GRADE: {input["grade"]}

CONVERSATION HISTORY:
{json.dumps(input["conversation_history"], indent=2)}

OBSERVER ANALYSIS:
{json.dumps(input["observer_analysis"], indent=2)}

COVERED TOPICS: {", ".join(input["covered_topics"])}

ACTUAL OUTPUT (Interviewer's response):
{output["response"]}

EXPECTED OUTPUT CRITERIA:
{json.dumps(expected_output.get("quality_criteria", {}), indent=2)}

{evaluator_prompt}
"""

        try:
            # Get evaluation from LLM using structured output
            result = llm_client.generate_structured(
                system_prompt="You are an expert evaluator for InterviewerAgent outputs. Provide structured evaluation scores.",
                user_prompt=eval_input,
                response_format=InterviewerEvaluationScore,
            )

            return Evaluation(
                name="interviewer_llm_judge",
                value=result.overall_score,
                comment=result.comment,
                metadata=result.model_dump(),
            )
        except Exception as e:
            return Evaluation(
                name="interviewer_llm_judge",
                value=0.0,
                comment=f"Evaluation failed: {str(e)}",
            )

    return interviewer_evaluator


# ============================================================================
# Feedback Generator Agent Evaluation
# ============================================================================


def evaluate_feedback_generator_agent(
    item: Dict, langfuse: Langfuse, prompt_version: str
) -> Dict:
    """
    Run FeedbackGeneratorAgent with specified prompt version and return feedback.
    """
    llm_client = get_llm_client()

    # Create candidate info
    candidate_info = CandidateInfo(
        name="Test Candidate",
        position=item["input"]["position"],
        target_grade=item["input"]["target_grade"],
        experience="N/A",
    )

    feedback_gen = FeedbackGeneratorAgent(llm_client)

    # Override system prompt
    system_prompt = get_prompt_content(
        langfuse, "feedback_generator", "system", prompt_version
    )
    feedback_gen.system_prompt = system_prompt

    # Extract input
    raw_conversation_history = item["input"]["conversation_history"]

    # Transform conversation history to agent format
    conversation_history = transform_conversation_history(raw_conversation_history)

    # Generate feedback (no longer needs observer_analyses parameter)
    feedback, _ = feedback_gen.generate_feedback(
        candidate_info=candidate_info,
        conversation_history=conversation_history,
    )

    # Convert to dict (map actual model fields to dataset format)
    # Note: Grade and HiringRecommendation are str enums, so we can convert them directly to str
    return {
        "verdict": str(feedback.hiring_recommendation),
        "assessed_grade": str(feedback.assessed_grade),
        "grade_reasoning": f"Confidence: {feedback.confidence_score}%",
        "confirmed_skills": [{"topic": s.topic, "status": s.status, "details": s.details} for s in feedback.confirmed_skills],
        "knowledge_gaps": [{"topic": s.topic, "status": s.status, "details": s.details} for s in feedback.knowledge_gaps],
        "concerning_patterns": [],  # Not in new model
        "soft_skills": {
            "clarity": feedback.soft_skills.clarity if feedback.soft_skills else 5,
            "clarity_notes": feedback.soft_skills.clarity_notes if feedback.soft_skills else "",
            "honesty": feedback.soft_skills.honesty if feedback.soft_skills else 5,
            "honesty_notes": feedback.soft_skills.honesty_notes if feedback.soft_skills else "",
            "engagement": feedback.soft_skills.engagement if feedback.soft_skills else 5,
            "engagement_notes": feedback.soft_skills.engagement_notes if feedback.soft_skills else "",
        },
        "learning_roadmap": feedback.roadmap if feedback.roadmap else feedback.topics_to_improve,
    }


def create_feedback_generator_evaluator(langfuse: Langfuse):  # noqa: ARG001
    """Create LLM-as-a-judge evaluator for FeedbackGeneratorAgent."""
    evaluator_prompt = load_evaluator_prompt("feedback_generator")
    llm_client = get_llm_client()

    def feedback_evaluator(
        *, input: Dict, output: Dict, expected_output: Dict, **kwargs  # noqa: ARG001
    ) -> Evaluation:
        """Evaluate FeedbackGeneratorAgent output using LLM-as-judge."""

        eval_input = f"""
POSITION: {input["position"]}
TARGET GRADE: {input["target_grade"]}

CONVERSATION HISTORY:
{json.dumps(input["conversation_history"], indent=2)}

OBSERVER ANALYSES:
{json.dumps(input["observer_analyses"], indent=2)}

ACTUAL OUTPUT (FeedbackGenerator's feedback):
{json.dumps(output, indent=2)}

EXPECTED OUTPUT (Correct feedback):
{json.dumps(expected_output, indent=2)}

{evaluator_prompt}
"""

        try:
            # Get evaluation from LLM using structured output
            result = llm_client.generate_structured(
                system_prompt="You are an expert evaluator for FeedbackGeneratorAgent outputs. Provide structured evaluation scores.",
                user_prompt=eval_input,
                response_format=FeedbackGeneratorEvaluationScore,
            )

            return Evaluation(
                name="feedback_generator_llm_judge",
                value=result.overall_score,
                comment=result.comment,
                metadata=result.model_dump(),
            )
        except Exception as e:
            return Evaluation(
                name="feedback_generator_llm_judge",
                value=0.0,
                comment=f"Evaluation failed: {str(e)}",
            )

    return feedback_evaluator


# ============================================================================
# Main Evaluation Runner
# ============================================================================


def run_evaluation(
    agent_name: str, prompt_version: str, dataset_name: str = None, run_name: str = None
):
    """
    Run offline evaluation for specified agent and prompt version.

    This function extracts agent-specific test cases from full interview sessions
    and evaluates the agent against them.

    Args:
        agent_name: Name of agent to evaluate
        prompt_version: Prompt version to test
        dataset_name: Optional dataset name override
        run_name: Optional custom run name
    """
    langfuse = Langfuse()

    # Determine dataset name (default to unified dataset)
    if dataset_name is None:
        dataset_name = "full_interview_sessions"

    # Determine run name
    if run_name is None:
        run_name = f"{agent_name}_v{prompt_version}"

    print(f"\n{'=' * 80}")
    print("Starting Evaluation")
    print(f"{'=' * 80}")
    print(f"Agent: {agent_name}")
    print(f"Prompt Version: {prompt_version}")
    print(f"Dataset: {dataset_name}")
    print(f"Run Name: {run_name}")
    print(f"{'=' * 80}\n")

    # Get dataset
    try:
        dataset = langfuse.get_dataset(dataset_name)
        dataset_items = list(dataset.items)
        print(f"✓ Loaded dataset '{dataset_name}' with {len(dataset_items)} sessions\n")
    except Exception as e:
        print(f"✗ Error loading dataset '{dataset_name}': {e}")
        print("\nTo create a dataset, use:")
        print("  python scripts/generate_dataset.py --mode full-session --num-cases 5")
        return

    # Extract agent-specific test cases from full sessions
    print(f"Extracting {agent_name} test cases from full sessions...\n")
    extracted_test_cases = []

    for session_item in dataset_items:
        # The session data is in session_item.input
        session = session_item.input

        if agent_name == "observer":
            cases = extract_observer_test_cases(session)
            extracted_test_cases.extend(cases)
        elif agent_name == "interviewer":
            cases = extract_interviewer_test_cases(session)
            extracted_test_cases.extend(cases)
        elif agent_name == "feedback_generator":
            case = extract_feedback_generator_test_case(session)
            extracted_test_cases.append(case)
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    print(f"✓ Extracted {len(extracted_test_cases)} test cases for {agent_name}\n")

    if len(extracted_test_cases) == 0:
        print("✗ No test cases extracted. Check dataset structure.")
        return

    # Select task and evaluator based on agent
    def observer_task(item):
        return evaluate_observer_agent(item, langfuse, prompt_version)

    def interviewer_task(item):
        return evaluate_interviewer_agent(item, langfuse, prompt_version)

    def feedback_generator_task(item):
        return evaluate_feedback_generator_agent(item, langfuse, prompt_version)

    if agent_name == "observer":
        task_fn = observer_task
        evaluator = create_observer_evaluator(langfuse)
    elif agent_name == "interviewer":
        task_fn = interviewer_task
        evaluator = create_interviewer_evaluator(langfuse)
    elif agent_name == "feedback_generator":
        task_fn = feedback_generator_task
        evaluator = create_feedback_generator_evaluator(langfuse)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # Run experiment
    print("Running experiment... This may take a few minutes.\n")

    # We need to manually iterate through test cases
    results = []
    for i, test_case in enumerate(extracted_test_cases):
        print(f"Evaluating test case {i + 1}/{len(extracted_test_cases)}...")
        try:
            # Create a simple object that mimics Langfuse dataset item structure
            class TestCaseItem:
                def __init__(self, tc):
                    self.input = tc["input"]
                    self.expected_output = tc.get("expected_output")

                def __getitem__(self, key):
                    # Allow dict-style access for backwards compatibility
                    if key == "input":
                        return self.input
                    elif key == "expected_output":
                        return self.expected_output
                    raise KeyError(key)

            item = TestCaseItem(test_case)

            # Run task
            print(f"  → Running {agent_name} agent...")
            output = task_fn(item)
            print(f"  ✓ Agent execution complete")

            # Run evaluator
            print(f"  → Running LLM-as-judge evaluator...")
            evaluation = evaluator(
                input=item.input,
                output=output,
                expected_output=item.expected_output if item.expected_output else {}
            )
            print(f"  ✓ Evaluation complete (score: {evaluation.value:.3f})")

            results.append({
                "test_case": i + 1,
                "score": evaluation.value,
                "comment": evaluation.comment,
                "metadata": evaluation.metadata if hasattr(evaluation, 'metadata') else {}
            })
        except Exception as e:
            import traceback
            print(f"  ✗ Error on test case {i + 1}: {e}")
            print(f"  ✗ Traceback:")
            traceback.print_exc()
            results.append({
                "test_case": i + 1,
                "score": 0.0,
                "comment": f"Error: {str(e)}",
                "metadata": {}
            })

    # Calculate summary statistics
    scores = [r["score"] for r in results if r["score"] is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Print results
    print(f"\n{'=' * 80}")
    print("Evaluation Results")
    print(f"{'=' * 80}\n")
    print(f"Total test cases: {len(results)}")
    print(f"Average score: {avg_score:.3f}")
    print("\nPer-case results:")
    for result in results:
        print(f"  Case {result['test_case']}: {result['score']:.3f} - {result['comment'][:100]}...")

    print(f"\n{'=' * 80}")
    print("✓ Evaluation complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    args = parse_args()

    run_evaluation(
        agent_name=args.agent_name,
        prompt_version=args.prompt_version,
        dataset_name=args.dataset_name,
        run_name=args.run_name,
    )
