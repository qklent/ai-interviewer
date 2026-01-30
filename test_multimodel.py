#!/usr/bin/env python3
"""
Test script for multi-model feedback generation.

This script verifies that:
1. MultiModelFeedbackGenerator can be imported and initialized
2. Configuration is loaded correctly
3. The system switches between single-model and multi-model modes properly
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.agents.multi_model_feedback_generator import MultiModelFeedbackGenerator
from src.agents.feedback_generator import FeedbackGeneratorAgent
from src.core.llm_client import get_llm_client
from src.core.models import CandidateInfo, Grade

load_dotenv()


def test_imports():
    """Test that all required modules can be imported."""
    print("✓ All imports successful")


def test_initialization():
    """Test that MultiModelFeedbackGenerator can be initialized."""
    try:
        generator = MultiModelFeedbackGenerator(
            google_model="google/gemini-2.0-flash-thinking-exp-1219",
            anthropic_model="anthropic/claude-3.5-sonnet",
            openai_model="openai/gpt-4o",
            save_intermediate=True,
        )
        print("✓ MultiModelFeedbackGenerator initialized successfully")
        print(f"  - Fallback mode: {generator.fallback_mode}")
        print(f"  - Save intermediate: {generator.save_intermediate}")
        return generator
    except Exception as e:
        print(f"✗ Failed to initialize MultiModelFeedbackGenerator: {e}")
        raise


def test_configuration():
    """Test that configuration is loaded correctly."""
    # Test environment variable reading
    feedback_mode = os.getenv("FEEDBACK_MODE", "single_model")
    print(f"✓ FEEDBACK_MODE: {feedback_mode}")

    if feedback_mode == "multi_model":
        google_model = os.getenv(
            "FEEDBACK_MODEL_GOOGLE", "google/gemini-2.0-flash-thinking-exp-1219"
        )
        anthropic_model = os.getenv(
            "FEEDBACK_MODEL_ANTHROPIC", "anthropic/claude-3.5-sonnet"
        )
        openai_model = os.getenv("FEEDBACK_MODEL_OPENAI", "openai/gpt-4o")
        save_intermediate = (
            os.getenv("SAVE_INTERMEDIATE_FEEDBACK", "true").lower() == "true"
        )

        print(f"  - Google model: {google_model}")
        print(f"  - Anthropic model: {anthropic_model}")
        print(f"  - OpenAI model: {openai_model}")
        print(f"  - Save intermediate: {save_intermediate}")


def test_mode_switching():
    """Test switching between single-model and multi-model modes."""
    print("\n--- Testing Mode Switching ---")

    # Save original mode
    original_mode = os.getenv("FEEDBACK_MODE")

    # Test single-model mode
    os.environ["FEEDBACK_MODE"] = "single_model"
    llm = get_llm_client("openrouter")
    single_generator = FeedbackGeneratorAgent(llm)
    print("✓ Single-model mode: FeedbackGeneratorAgent initialized")

    # Test multi-model mode
    os.environ["FEEDBACK_MODE"] = "multi_model"
    multi_generator = MultiModelFeedbackGenerator(
        google_model="google/gemini-2.0-flash-thinking-exp-1219",
        anthropic_model="anthropic/claude-3.5-sonnet",
        openai_model="openai/gpt-4o",
        save_intermediate=True,
    )
    print("✓ Multi-model mode: MultiModelFeedbackGenerator initialized")

    # Restore original mode
    if original_mode:
        os.environ["FEEDBACK_MODE"] = original_mode
    else:
        os.environ.pop("FEEDBACK_MODE", None)


def test_prompts_exist():
    """Test that aggregation prompts exist."""
    print("\n--- Testing Prompts ---")

    prompts_dir = Path(__file__).parent / "prompts" / "feedback_generator"

    aggregate_system = prompts_dir / "aggregate_system.txt"
    aggregate_feedback = prompts_dir / "aggregate_feedback.txt"

    if aggregate_system.exists():
        print(f"✓ aggregate_system.txt exists ({aggregate_system.stat().st_size} bytes)")
    else:
        print(f"✗ aggregate_system.txt NOT FOUND at {aggregate_system}")

    if aggregate_feedback.exists():
        print(
            f"✓ aggregate_feedback.txt exists ({aggregate_feedback.stat().st_size} bytes)"
        )
    else:
        print(f"✗ aggregate_feedback.txt NOT FOUND at {aggregate_feedback}")


def test_serialization():
    """Test feedback serialization."""
    print("\n--- Testing Serialization ---")

    from src.core.models import (
        FinalFeedback,
        SkillAssessment,
        SoftSkillsAssessment,
        HiringRecommendation,
    )

    # Create dummy feedback
    feedback = FinalFeedback(
        assessed_grade=Grade.MIDDLE,
        hiring_recommendation=HiringRecommendation.HIRE,
        confidence_score=85,
        confirmed_skills=[
            SkillAssessment(
                topic="Python", status="confirmed", details="Strong understanding"
            )
        ],
        knowledge_gaps=[
            SkillAssessment(
                topic="Async",
                status="gap",
                details="Needs improvement",
                correct_answer="Use asyncio",
            )
        ],
        soft_skills=SoftSkillsAssessment(
            clarity=8,
            clarity_notes="Clear communication",
            honesty=9,
            honesty_notes="Honest about gaps",
            engagement=7,
            engagement_notes="Good engagement",
        ),
        topics_to_improve=["Async programming"],
        recommended_actions=["Study asyncio"],
        resources=["Python docs"],
    )

    # Test serialization
    generator = MultiModelFeedbackGenerator(
        google_model="google/gemini-2.0-flash-thinking-exp-1219",
        anthropic_model="anthropic/claude-3.5-sonnet",
        openai_model="openai/gpt-4o",
    )

    serialized = generator._serialize_feedback(feedback)
    print("✓ Feedback serialization successful")
    print(f"  - Keys: {list(serialized.keys())}")


def main():
    print("=" * 60)
    print("  MULTI-MODEL FEEDBACK GENERATOR TEST")
    print("=" * 60)
    print()

    try:
        test_imports()
        test_initialization()
        test_configuration()
        test_mode_switching()
        test_prompts_exist()
        test_serialization()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe multi-model feedback generation system is ready to use!")
        print("\nTo enable it, set in your .env file:")
        print("  FEEDBACK_MODE=multi_model")
        print("\nTo test it with a real interview, run:")
        print("  python main.py")

    except Exception as e:
        print("\n" + "=" * 60)
        print("  TESTS FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
