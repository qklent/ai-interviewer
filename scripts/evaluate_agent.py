#!/usr/bin/env python3
"""
Offline Evaluation Script for Multi-Agent Interview System

Usage:
    python scripts/evaluate_agent.py --agent-name observer --prompt-version latest
    python scripts/evaluate_agent.py --agent-name interviewer --prompt-version 2
    python scripts/evaluate_agent.py --agent-name feedback_generator --prompt-version 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langfuse import Langfuse, Evaluation
from src.core.llm_client import create_llm_client
from src.agents.observer import ObserverAgent
from src.agents.interviewer import InterviewerAgent
from src.agents.feedback_generator import FeedbackGeneratorAgent
from src.core.models import CandidateInfo, ObserverAnalysis
from src.utils.prompt_loader import load_prompt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate agent performance with specific prompt version"
    )
    parser.add_argument(
        "--agent-name",
        required=True,
        choices=["observer", "interviewer", "feedback_generator"],
        help="Name of the agent to evaluate"
    )
    parser.add_argument(
        "--prompt-version",
        required=True,
        help="Prompt version to test (use 'latest' or specific version number)"
    )
    parser.add_argument(
        "--dataset-name",
        help="Override default dataset name (default: {agent_name}_evaluation)"
    )
    parser.add_argument(
        "--run-name",
        help="Custom name for this evaluation run"
    )

    return parser.parse_args()


def load_evaluator_prompt(agent_name: str) -> str:
    """Load the LLM-as-judge evaluator prompt for the agent."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "evaluation" / f"{agent_name}_evaluator.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Evaluator prompt not found: {prompt_path}")

    return prompt_path.read_text()


def get_prompt_content(langfuse: Langfuse, agent_name: str, prompt_type: str, version: str) -> str:
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
            prompt = langfuse.get_prompt(prompt_name)
        else:
            prompt = langfuse.get_prompt(prompt_name, version=int(version))

        return prompt.prompt
    except Exception as e:
        print(f"Warning: Could not load prompt '{prompt_name}' version {version} from Langfuse: {e}")
        print(f"Falling back to local file...")
        return load_prompt(agent_name, prompt_type)


# ============================================================================
# Observer Agent Evaluation
# ============================================================================

def evaluate_observer_agent(item: Dict, langfuse: Langfuse, prompt_version: str) -> Dict:
    """
    Run ObserverAgent with specified prompt version and return analysis.

    Args:
        item: Dataset item with input and expected_output
        langfuse: Langfuse client
        prompt_version: Prompt version to use

    Returns:
        Dictionary with ObserverAgent's analysis
    """
    llm_client = create_llm_client()
    observer = ObserverAgent(llm_client)

    # Override system prompt with specified version
    system_prompt = get_prompt_content(langfuse, "observer", "system", prompt_version)
    observer.system_prompt = system_prompt

    # Extract input
    question = item["input"]["question"]
    candidate_response = item["input"]["candidate_response"]
    conversation_history = item["input"].get("conversation_history", [])
    covered_topics = item["input"].get("covered_topics", [])

    # Run analysis
    analysis = observer.analyze_response(
        question=question,
        response=candidate_response,
        conversation_history=conversation_history,
        covered_topics=covered_topics
    )

    # Convert to dict for evaluation
    return {
        "quality": analysis.quality,
        "confidence": analysis.confidence,
        "has_hallucination": analysis.has_hallucination,
        "hallucination_details": analysis.hallucination_details,
        "recommended_action": analysis.recommended_action,
        "reasoning": analysis.reasoning,
        "covered_topics": analysis.covered_topics
    }


def create_observer_evaluator(langfuse: Langfuse):
    """Create LLM-as-a-judge evaluator for ObserverAgent."""
    evaluator_prompt = load_evaluator_prompt("observer")
    llm_client = create_llm_client()

    def observer_evaluator(*, input: Dict, output: Dict, expected_output: Dict, **kwargs) -> Evaluation:
        """Evaluate ObserverAgent output using LLM-as-judge."""

        # Build prompt for evaluator
        eval_input = f"""
QUESTION: {input['question']}

CANDIDATE RESPONSE: {input['candidate_response']}

ACTUAL OUTPUT (ObserverAgent's analysis):
{json.dumps(output, indent=2)}

EXPECTED OUTPUT (Correct analysis):
{json.dumps(expected_output, indent=2)}

{evaluator_prompt}
"""

        try:
            # Get evaluation from LLM
            result = llm_client.generate_json(eval_input)

            return Evaluation(
                name="observer_llm_judge",
                value=result["overall_score"],
                comment=result["comment"],
                metadata=result  # Store all individual scores
            )
        except Exception as e:
            return Evaluation(
                name="observer_llm_judge",
                value=0.0,
                comment=f"Evaluation failed: {str(e)}"
            )

    return observer_evaluator


# ============================================================================
# Interviewer Agent Evaluation
# ============================================================================

def evaluate_interviewer_agent(item: Dict, langfuse: Langfuse, prompt_version: str) -> Dict:
    """
    Run InterviewerAgent with specified prompt version and return response.
    """
    llm_client = create_llm_client()

    # Create candidate info from item
    candidate_info = CandidateInfo(
        name="Test Candidate",
        position=item["input"]["position"],
        target_grade=item["input"]["grade"],
        years_of_experience="N/A"
    )

    interviewer = InterviewerAgent(llm_client, candidate_info)

    # Override system prompt
    system_prompt = get_prompt_content(langfuse, "interviewer", "system", prompt_version)
    interviewer.system_prompt = system_prompt

    # Extract input
    conversation_history = item["input"]["conversation_history"]
    observer_analysis_data = item["input"]["observer_analysis"]
    covered_topics = item["input"]["covered_topics"]

    # Convert observer_analysis dict to ObserverAnalysis object
    observer_analysis = ObserverAnalysis(
        quality=observer_analysis_data["quality"],
        confidence=observer_analysis_data["confidence"],
        has_hallucination=observer_analysis_data.get("has_hallucination", False),
        hallucination_details=observer_analysis_data.get("hallucination_details"),
        recommended_action=observer_analysis_data["recommended_action"],
        reasoning=observer_analysis_data["reasoning"],
        covered_topics=observer_analysis_data.get("covered_topics", [])
    )

    # Generate response
    response = interviewer.generate_response(
        conversation_history=conversation_history,
        observer_analysis=observer_analysis,
        covered_topics=covered_topics
    )

    return {
        "response": response["visible"],
        "internal_thoughts": response["internal"]
    }


def create_interviewer_evaluator(langfuse: Langfuse):
    """Create LLM-as-a-judge evaluator for InterviewerAgent."""
    evaluator_prompt = load_evaluator_prompt("interviewer")
    llm_client = create_llm_client()

    def interviewer_evaluator(*, input: Dict, output: Dict, expected_output: Dict, **kwargs) -> Evaluation:
        """Evaluate InterviewerAgent output using LLM-as-judge."""

        eval_input = f"""
POSITION: {input['position']}
GRADE: {input['grade']}

CONVERSATION HISTORY:
{json.dumps(input['conversation_history'], indent=2)}

OBSERVER ANALYSIS:
{json.dumps(input['observer_analysis'], indent=2)}

COVERED TOPICS: {', '.join(input['covered_topics'])}

ACTUAL OUTPUT (Interviewer's response):
{output['response']}

EXPECTED OUTPUT CRITERIA:
{json.dumps(expected_output.get('quality_criteria', {}), indent=2)}

{evaluator_prompt}
"""

        try:
            result = llm_client.generate_json(eval_input)

            return Evaluation(
                name="interviewer_llm_judge",
                value=result["overall_score"],
                comment=result["comment"],
                metadata=result
            )
        except Exception as e:
            return Evaluation(
                name="interviewer_llm_judge",
                value=0.0,
                comment=f"Evaluation failed: {str(e)}"
            )

    return interviewer_evaluator


# ============================================================================
# Feedback Generator Agent Evaluation
# ============================================================================

def evaluate_feedback_generator_agent(item: Dict, langfuse: Langfuse, prompt_version: str) -> Dict:
    """
    Run FeedbackGeneratorAgent with specified prompt version and return feedback.
    """
    llm_client = create_llm_client()

    # Create candidate info
    candidate_info = CandidateInfo(
        name="Test Candidate",
        position=item["input"]["position"],
        target_grade=item["input"]["target_grade"],
        years_of_experience="N/A"
    )

    feedback_gen = FeedbackGeneratorAgent(llm_client, candidate_info)

    # Override system prompt
    system_prompt = get_prompt_content(langfuse, "feedback_generator", "system", prompt_version)
    feedback_gen.system_prompt = system_prompt

    # Extract input
    conversation_history = item["input"]["conversation_history"]

    # Convert observer analyses to ObserverAnalysis objects
    observer_analyses = []
    for analysis_data in item["input"]["observer_analyses"]:
        observer_analyses.append(ObserverAnalysis(
            quality=analysis_data["quality"],
            confidence=analysis_data["confidence"],
            has_hallucination=analysis_data.get("has_hallucination", False),
            hallucination_details=analysis_data.get("hallucination_details"),
            recommended_action=analysis_data["recommended_action"],
            reasoning=analysis_data["reasoning"],
            covered_topics=analysis_data.get("covered_topics", [])
        ))

    # Generate feedback
    feedback = feedback_gen.generate_feedback(
        conversation_history=conversation_history,
        observer_analyses=observer_analyses
    )

    # Convert to dict
    return {
        "verdict": feedback.verdict,
        "assessed_grade": feedback.assessed_grade,
        "grade_reasoning": feedback.grade_reasoning,
        "confirmed_skills": feedback.confirmed_skills,
        "knowledge_gaps": feedback.knowledge_gaps,
        "concerning_patterns": feedback.concerning_patterns,
        "soft_skills": feedback.soft_skills,
        "learning_roadmap": feedback.learning_roadmap
    }


def create_feedback_generator_evaluator(langfuse: Langfuse):
    """Create LLM-as-a-judge evaluator for FeedbackGeneratorAgent."""
    evaluator_prompt = load_evaluator_prompt("feedback_generator")
    llm_client = create_llm_client()

    def feedback_evaluator(*, input: Dict, output: Dict, expected_output: Dict, **kwargs) -> Evaluation:
        """Evaluate FeedbackGeneratorAgent output using LLM-as-judge."""

        eval_input = f"""
POSITION: {input['position']}
TARGET GRADE: {input['target_grade']}

CONVERSATION HISTORY:
{json.dumps(input['conversation_history'], indent=2)}

OBSERVER ANALYSES:
{json.dumps(input['observer_analyses'], indent=2)}

ACTUAL OUTPUT (FeedbackGenerator's feedback):
{json.dumps(output, indent=2)}

EXPECTED OUTPUT (Correct feedback):
{json.dumps(expected_output, indent=2)}

{evaluator_prompt}
"""

        try:
            result = llm_client.generate_json(eval_input)

            return Evaluation(
                name="feedback_generator_llm_judge",
                value=result["overall_score"],
                comment=result["comment"],
                metadata=result
            )
        except Exception as e:
            return Evaluation(
                name="feedback_generator_llm_judge",
                value=0.0,
                comment=f"Evaluation failed: {str(e)}"
            )

    return feedback_evaluator


# ============================================================================
# Main Evaluation Runner
# ============================================================================

def run_evaluation(agent_name: str, prompt_version: str, dataset_name: str = None, run_name: str = None):
    """
    Run offline evaluation for specified agent and prompt version.

    Args:
        agent_name: Name of agent to evaluate
        prompt_version: Prompt version to test
        dataset_name: Optional dataset name override
        run_name: Optional custom run name
    """
    langfuse = Langfuse()

    # Determine dataset name
    if dataset_name is None:
        dataset_name = f"{agent_name}_evaluation"

    # Determine run name
    if run_name is None:
        run_name = f"{agent_name}_v{prompt_version}"

    print(f"\n{'='*80}")
    print(f"Starting Evaluation")
    print(f"{'='*80}")
    print(f"Agent: {agent_name}")
    print(f"Prompt Version: {prompt_version}")
    print(f"Dataset: {dataset_name}")
    print(f"Run Name: {run_name}")
    print(f"{'='*80}\n")

    # Get dataset
    try:
        dataset = langfuse.get_dataset(dataset_name)
        print(f"✓ Loaded dataset '{dataset_name}' with {len(list(dataset.items))} items\n")
    except Exception as e:
        print(f"✗ Error loading dataset '{dataset_name}': {e}")
        print(f"\nTo create a dataset, use:")
        print(f"  python scripts/generate_dataset.py --agent-name {agent_name} --num-cases 10")
        return

    # Select task and evaluator based on agent
    if agent_name == "observer":
        task_fn = lambda item: evaluate_observer_agent(item, langfuse, prompt_version)
        evaluator = create_observer_evaluator(langfuse)
    elif agent_name == "interviewer":
        task_fn = lambda item: evaluate_interviewer_agent(item, langfuse, prompt_version)
        evaluator = create_interviewer_evaluator(langfuse)
    elif agent_name == "feedback_generator":
        task_fn = lambda item: evaluate_feedback_generator_agent(item, langfuse, prompt_version)
        evaluator = create_feedback_generator_evaluator(langfuse)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # Run experiment
    print("Running experiment... This may take a few minutes.\n")

    result = langfuse.run_experiment(
        name=run_name,
        description=f"Evaluating {agent_name} with prompt version {prompt_version}",
        data=dataset,
        task=task_fn,
        evaluators=[evaluator]
    )

    # Print results
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}\n")
    print(result.format())
    print(f"\n{'='*80}")
    print(f"✓ Evaluation complete! View detailed results in Langfuse UI.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    args = parse_args()

    run_evaluation(
        agent_name=args.agent_name,
        prompt_version=args.prompt_version,
        dataset_name=args.dataset_name,
        run_name=args.run_name
    )
