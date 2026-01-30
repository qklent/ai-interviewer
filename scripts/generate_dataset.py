#!/usr/bin/env python3
"""
Dataset Generator Script

This script uses LLM to generate synthetic test cases for agent evaluation.

Usage:
    python scripts/generate_dataset.py --agent-name observer --num-cases 10
    python scripts/generate_dataset.py --agent-name interviewer --num-cases 5
    python scripts/generate_dataset.py --agent-name feedback_generator --num-cases 3
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
        "--agent-name",
        required=True,
        choices=["observer", "interviewer", "feedback_generator"],
        help="Name of the agent to generate test cases for"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of test cases to generate (default: 10)"
    )
    parser.add_argument(
        "--dataset-name",
        help="Custom dataset name (default: {agent_name}_evaluation)"
    )
    parser.add_argument(
        "--output-file",
        help="Save generated cases to JSON file (optional)"
    )

    return parser.parse_args()


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
    # Determine which provider to use (same logic as main.py)
    if os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        llm_client = get_llm_client(provider, model=model)
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
        llm_client = get_llm_client(provider)
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"
        llm_client = get_llm_client(provider)
    else:
        raise ValueError(
            "No API key found! Please set one of: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
        )

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


def main():
    args = parse_args()

    # Determine dataset name
    dataset_name = args.dataset_name or f"{args.agent_name}_evaluation"

    print(f"\n{'='*80}")
    print("Dataset Generation")
    print(f"{'='*80}")
    print(f"Agent: {args.agent_name}")
    print(f"Number of cases: {args.num_cases}")
    print(f"Dataset name: {dataset_name}")
    print(f"{'='*80}\n")

    # Generate test cases
    try:
        test_cases = generate_test_cases(args.agent_name, args.num_cases)
    except Exception as e:
        print(f"\nFailed to generate test cases: {e}")
        sys.exit(1)

    # Save to file if requested
    if args.output_file:
        save_to_file(args.output_file, test_cases)

    # Upload to Langfuse
    try:
        upload_to_langfuse(dataset_name, test_cases, args.agent_name)
    except Exception as e:
        print(f"\nFailed to upload to Langfuse: {e}")
        print("Test cases were generated but not uploaded.")
        sys.exit(1)

    print(f"{'='*80}")
    print("✓ Dataset generation complete!")
    print(f"{'='*80}\n")
    print(f"Next steps:")
    print(f"  1. Review the dataset in Langfuse UI")
    print(f"  2. Run evaluation:")
    print(f"     python scripts/evaluate_agent.py --agent-name {args.agent_name} --prompt-version latest")
    print()


if __name__ == "__main__":
    main()
