"""Multi-Model Feedback Generator - uses multiple LLMs for diverse evaluation."""

import os
from typing import Optional

from langfuse import observe
from src.core.llm_client import get_llm_client, BaseLLMClient
from src.core.models import CandidateInfo, FinalFeedback
from src.core.schemas import FeedbackSchema
from src.agents.feedback_generator import FeedbackGeneratorAgent
from src.utils.prompt_loader import load_prompt
from src.utils.app_logger import get_logger

logger = get_logger(__name__)


class MultiModelFeedbackGenerator:
    """Generates feedback using multiple models for reduced bias and improved quality.

    Architecture:
    - Model 1 (Google): Generate independent feedback
    - Model 2 (Anthropic): Generate independent feedback
    - Model 3 (OpenAI): Aggregate both feedbacks into final synthesis
    """

    def __init__(
        self,
        google_model: str,
        anthropic_model: str,
        openai_model: str,
        save_intermediate: bool = True,
    ):
        """Initialize multi-model feedback generator.

        Args:
            google_model: Google model identifier (e.g., google/gemini-2.0-flash-thinking-exp-1219)
            anthropic_model: Anthropic model identifier (e.g., anthropic/claude-3.5-sonnet)
            openai_model: OpenAI model identifier (e.g., openai/gpt-4o)
            save_intermediate: Whether to save intermediate feedback results
        """
        logger.info(
            f"Initializing MultiModelFeedbackGenerator: "
            f"google={google_model}, anthropic={anthropic_model}, openai={openai_model}"
        )

        # Create three separate LLM clients
        self.google_llm = get_llm_client("openrouter", model=google_model)
        self.anthropic_llm = get_llm_client("openrouter", model=anthropic_model)
        self.openai_llm = get_llm_client("openrouter", model=openai_model)

        # Create feedback generator instances for Models 1 & 2
        self.google_generator = FeedbackGeneratorAgent(self.google_llm)
        self.anthropic_generator = FeedbackGeneratorAgent(self.anthropic_llm)

        self.save_intermediate = save_intermediate
        self.fallback_mode = os.getenv("MULTIMODEL_FALLBACK_MODE", "partial")

        logger.info(
            f"MultiModelFeedbackGenerator initialized with fallback_mode={self.fallback_mode}"
        )

    @observe(name="MultiModel: Generate Feedback")
    def generate_feedback(
        self,
        candidate_info: CandidateInfo,
        conversation_history: list[dict],
    ) -> tuple[FinalFeedback, str, Optional[dict]]:
        """Generate feedback using multi-model approach.

        Returns:
            Tuple of (final_feedback, formatted_string, intermediate_results)
            - intermediate_results is dict with models 1 & 2 outputs (if save_intermediate=True)
        """
        logger.info("Starting multi-model feedback generation")

        # Stage 1 & 2: Generate independent feedbacks
        feedback_1, formatted_1, feedback_2, formatted_2 = (
            self._generate_independent_feedbacks(candidate_info, conversation_history)
        )

        # Handle failure cases
        if feedback_1 is None and feedback_2 is None:
            logger.error("Both evaluation models failed, falling back to single-model")
            return self._fallback_single_model(candidate_info, conversation_history)

        if feedback_1 is None or feedback_2 is None:
            logger.warning("One model failed, handling partial failure")
            return self._handle_partial_failure(
                feedback_1, formatted_1, feedback_2, formatted_2
            )

        # Stage 3: Aggregate with Model 3
        try:
            final_feedback, formatted_final = self._aggregate_feedbacks(
                candidate_info, feedback_1, feedback_2
            )
            logger.info("Multi-model feedback generation completed successfully")
        except Exception as e:
            logger.error(f"Aggregation failed: {e}, falling back to Anthropic feedback")
            # Use Anthropic (Model 2) as fallback - typically most balanced
            final_feedback = feedback_2
            formatted_final = formatted_2

        # Build intermediate results if requested
        intermediate = None
        if self.save_intermediate:
            intermediate = {
                "google_feedback": self._serialize_feedback(feedback_1),
                "anthropic_feedback": self._serialize_feedback(feedback_2),
                "google_formatted": formatted_1,
                "anthropic_formatted": formatted_2,
            }

        return final_feedback, formatted_final, intermediate

    @observe(name="MultiModel: Independent Feedbacks")
    def _generate_independent_feedbacks(
        self,
        candidate_info: CandidateInfo,
        conversation_history: list[dict],
    ) -> tuple[
        Optional[FinalFeedback],
        Optional[str],
        Optional[FinalFeedback],
        Optional[str],
    ]:
        """Generate independent feedbacks from Models 1 and 2.

        Returns:
            Tuple of (feedback_1, formatted_1, feedback_2, formatted_2)
        """
        feedback_1 = None
        formatted_1 = None
        feedback_2 = None
        formatted_2 = None

        # Generate Google feedback (Model 1)
        try:
            logger.info("Generating Google (Model 1) feedback...")
            feedback_1, formatted_1 = self.google_generator.generate_feedback(
                candidate_info, conversation_history
            )
            logger.info("Google feedback generated successfully")
        except Exception as e:
            logger.error(f"Google feedback generation failed: {e}", exc_info=True)

        # Generate Anthropic feedback (Model 2)
        try:
            logger.info("Generating Anthropic (Model 2) feedback...")
            feedback_2, formatted_2 = self.anthropic_generator.generate_feedback(
                candidate_info, conversation_history
            )
            logger.info("Anthropic feedback generated successfully")
        except Exception as e:
            logger.error(f"Anthropic feedback generation failed: {e}", exc_info=True)

        return feedback_1, formatted_1, feedback_2, formatted_2

    @observe(name="MultiModel: Aggregate Feedbacks")
    def _aggregate_feedbacks(
        self,
        candidate_info: CandidateInfo,
        feedback_1: FinalFeedback,
        feedback_2: FinalFeedback,
    ) -> tuple[FinalFeedback, str]:
        """Aggregate two independent feedbacks using Model 3.

        Args:
            candidate_info: Candidate information
            feedback_1: Feedback from Google (Model 1)
            feedback_2: Feedback from Anthropic (Model 2)

        Returns:
            Tuple of (aggregated_feedback, formatted_string)
        """
        logger.info("Starting feedback aggregation with OpenAI (Model 3)")

        # Serialize feedbacks to JSON for Model 3
        feedback_1_dict = self._serialize_feedback(feedback_1)
        feedback_2_dict = self._serialize_feedback(feedback_2)

        # Load aggregation prompts
        aggregate_system, system_metadata = load_prompt(
            "feedback_generator", "aggregate_system"
        )
        aggregate_template, template_metadata = load_prompt(
            "feedback_generator", "aggregate_feedback"
        )

        # Format aggregation prompt
        import json

        aggregate_prompt = aggregate_template.format(
            name=candidate_info.name,
            position=candidate_info.position,
            target_grade=candidate_info.target_grade.value,
            experience=candidate_info.experience,
            feedback_1=json.dumps(feedback_1_dict, indent=2, ensure_ascii=False),
            feedback_2=json.dumps(feedback_2_dict, indent=2, ensure_ascii=False),
        )

        # Generate aggregated feedback with Model 3
        response = self.openai_llm.generate_structured(
            system_prompt=aggregate_system,
            user_prompt=aggregate_prompt,
            response_format=FeedbackSchema,
            temperature=0.3,
            max_tokens=3000,
            prompt_metadata=system_metadata,
        )

        # Convert schema response to FinalFeedback (reuse logic from FeedbackGeneratorAgent)
        final_feedback = self._convert_schema_to_feedback(response)

        # Format for display
        formatted = self._format_feedback_for_display(final_feedback, response)

        logger.info("Feedback aggregation completed successfully")
        return final_feedback, formatted

    def _handle_partial_failure(
        self,
        feedback_1: Optional[FinalFeedback],
        formatted_1: Optional[str],
        feedback_2: Optional[FinalFeedback],
        formatted_2: Optional[str],
    ) -> tuple[FinalFeedback, str, Optional[dict]]:
        """Handle case where one model succeeded and one failed.

        Returns single successful feedback (skips aggregation).
        """
        if self.fallback_mode == "fail":
            logger.error("Fallback mode is 'fail', raising exception")
            raise Exception("One model failed and fallback_mode=fail")

        if self.fallback_mode == "single_model":
            logger.warning("Fallback mode is 'single_model', using default model")
            return self._fallback_single_model(None, None)

        # Default: partial mode - use whichever succeeded
        successful_feedback = feedback_1 or feedback_2
        successful_formatted = formatted_1 or formatted_2

        logger.warning(
            f"Using {'Google' if feedback_1 else 'Anthropic'} feedback (no aggregation)"
        )

        intermediate = None
        if self.save_intermediate:
            intermediate = {
                "google_feedback": self._serialize_feedback(feedback_1) if feedback_1 else None,
                "anthropic_feedback": self._serialize_feedback(feedback_2) if feedback_2 else None,
                "google_formatted": formatted_1,
                "anthropic_formatted": formatted_2,
                "note": "Partial failure - only one model succeeded",
            }

        return successful_feedback, successful_formatted, intermediate

    def _fallback_single_model(
        self,
        candidate_info: Optional[CandidateInfo],
        conversation_history: Optional[list[dict]],
    ) -> tuple[FinalFeedback, str, None]:
        """Complete fallback to single-model approach using default model.

        This is called when both Models 1 & 2 fail, or when fallback_mode=single_model.
        """
        logger.warning("Falling back to single-model feedback generation")

        # Use default model (from environment or anthropic/claude-3.5-sonnet)
        default_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        fallback_llm = get_llm_client("openrouter", model=default_model)
        fallback_generator = FeedbackGeneratorAgent(fallback_llm)

        feedback, formatted = fallback_generator.generate_feedback(
            candidate_info, conversation_history
        )

        logger.info(f"Fallback feedback generated with model: {default_model}")
        return feedback, formatted, None

    def _serialize_feedback(self, feedback: FinalFeedback) -> dict:
        """Serialize FinalFeedback to dictionary for aggregation."""
        return {
            "verdict": {
                "assessed_grade": feedback.assessed_grade.value,
                "hiring_recommendation": feedback.hiring_recommendation.value,
                "confidence_score": feedback.confidence_score,
            },
            "technical_review": {
                "confirmed_skills": [
                    {
                        "topic": s.topic,
                        "details": s.details,
                    }
                    for s in feedback.confirmed_skills
                ],
                "knowledge_gaps": [
                    {
                        "topic": g.topic,
                        "details": g.details,
                        "correct_answer": g.correct_answer,
                    }
                    for g in feedback.knowledge_gaps
                ],
            },
            "soft_skills": {
                "clarity": {
                    "score": feedback.soft_skills.clarity,
                    "notes": feedback.soft_skills.clarity_notes,
                },
                "honesty": {
                    "score": feedback.soft_skills.honesty,
                    "notes": feedback.soft_skills.honesty_notes,
                },
                "engagement": {
                    "score": feedback.soft_skills.engagement,
                    "notes": feedback.soft_skills.engagement_notes,
                },
            }
            if feedback.soft_skills
            else None,
            "roadmap": {
                "topics_to_improve": feedback.topics_to_improve,
                "recommended_actions": feedback.recommended_actions,
                "resources": feedback.resources,
            },
        }

    def _convert_schema_to_feedback(self, response: FeedbackSchema) -> FinalFeedback:
        """Convert FeedbackSchema to FinalFeedback object.

        Reuses conversion logic from FeedbackGeneratorAgent.
        """
        from src.core.models import (
            Grade,
            HiringRecommendation,
            SkillAssessment,
            SoftSkillsAssessment,
        )

        # Map grade string to enum
        grade_map = {
            "Junior": Grade.JUNIOR,
            "Middle": Grade.MIDDLE,
            "Senior": Grade.SENIOR,
        }
        assessed_grade = grade_map.get(response.verdict.assessed_grade, Grade.JUNIOR)

        # Map hiring recommendation to enum
        hire_map = {
            "No Hire": HiringRecommendation.NO_HIRE,
            "Hire": HiringRecommendation.HIRE,
            "Strong Hire": HiringRecommendation.STRONG_HIRE,
        }
        hiring_rec = hire_map.get(
            response.verdict.hiring_recommendation, HiringRecommendation.NO_HIRE
        )

        # Build confirmed skills
        confirmed_skills = [
            SkillAssessment(
                topic=skill.topic,
                status="confirmed",
                details=skill.details,
            )
            for skill in response.technical_review.confirmed_skills
        ]

        # Build knowledge gaps
        knowledge_gaps = [
            SkillAssessment(
                topic=gap.topic,
                status="gap",
                details=gap.details,
                correct_answer=gap.correct_answer,
            )
            for gap in response.technical_review.knowledge_gaps
        ]

        # Build soft skills assessment
        soft_skills = SoftSkillsAssessment(
            clarity=response.soft_skills.clarity.score,
            clarity_notes=response.soft_skills.clarity.notes,
            honesty=response.soft_skills.honesty.score,
            honesty_notes=response.soft_skills.honesty.notes,
            engagement=response.soft_skills.engagement.score,
            engagement_notes=response.soft_skills.engagement.notes,
        )

        # Build final feedback object
        return FinalFeedback(
            assessed_grade=assessed_grade,
            hiring_recommendation=hiring_rec,
            confidence_score=response.verdict.confidence_score,
            confirmed_skills=confirmed_skills,
            knowledge_gaps=knowledge_gaps,
            soft_skills=soft_skills,
            topics_to_improve=response.roadmap.topics_to_improve,
            recommended_actions=response.roadmap.specific_recommendations,
            roadmap=response.roadmap.topics_to_improve
            + response.roadmap.specific_recommendations,
            resources=response.roadmap.resources,
        )

    def _format_feedback_for_display(
        self, feedback: FinalFeedback, response: FeedbackSchema
    ) -> str:
        """Format the feedback for human-readable display.

        Reuses formatting logic from FeedbackGeneratorAgent.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("           INTERVIEW FEEDBACK REPORT")
        lines.append("          (Multi-Model Evaluation)")
        lines.append("=" * 60)
        lines.append("")

        # Verdict section
        lines.append("A. VERDICT")
        lines.append("-" * 40)
        lines.append(f"   Assessed Grade: {feedback.assessed_grade.value}")
        lines.append(
            f"   Hiring Recommendation: {feedback.hiring_recommendation.value}"
        )
        lines.append(f"   Confidence Score: {feedback.confidence_score}%")

        if response.verdict.summary:
            lines.append(f"   Summary: {response.verdict.summary}")
        lines.append("")

        # Technical Review section
        lines.append("B. TECHNICAL REVIEW")
        lines.append("-" * 40)

        lines.append("   Confirmed Skills:")
        if feedback.confirmed_skills:
            for skill in feedback.confirmed_skills:
                lines.append(f"   [+] {skill.topic}")
                lines.append(f"       {skill.details}")
        else:
            lines.append("   No confirmed skills recorded.")
        lines.append("")

        lines.append("   Knowledge Gaps:")
        if feedback.knowledge_gaps:
            for gap in feedback.knowledge_gaps:
                lines.append(f"   [-] {gap.topic}")
                lines.append(f"       Issue: {gap.details}")
                if gap.correct_answer:
                    lines.append(f"       Correct Answer: {gap.correct_answer}")
        else:
            lines.append("   No significant gaps identified.")
        lines.append("")

        # Soft Skills section
        lines.append("C. SOFT SKILLS & COMMUNICATION")
        lines.append("-" * 40)
        if feedback.soft_skills:
            lines.append(f"   Clarity: {feedback.soft_skills.clarity}/10")
            lines.append(f"       {feedback.soft_skills.clarity_notes}")
            lines.append(f"   Honesty: {feedback.soft_skills.honesty}/10")
            lines.append(f"       {feedback.soft_skills.honesty_notes}")
            lines.append(f"   Engagement: {feedback.soft_skills.engagement}/10")
            lines.append(f"       {feedback.soft_skills.engagement_notes}")
        lines.append("")

        # Roadmap section
        lines.append("D. PERSONAL ROADMAP (Next Steps)")
        lines.append("-" * 40)

        if feedback.topics_to_improve:
            lines.append("   Topics to Improve:")
            for topic in feedback.topics_to_improve:
                lines.append(f"   - {topic}")
            lines.append("")

        if feedback.recommended_actions:
            lines.append("   Recommended Actions:")
            for i, action in enumerate(feedback.recommended_actions, 1):
                lines.append(f"   {i}. {action}")
        else:
            lines.append("   No specific recommendations.")
        lines.append("")

        if feedback.resources:
            lines.append("   Suggested Resources:")
            for resource in feedback.resources:
                lines.append(f"   - {resource}")
        lines.append("")

        # Notable moments
        if response.notable_moments.strengths or response.notable_moments.concerns:
            lines.append("E. NOTABLE MOMENTS")
            lines.append("-" * 40)
            if response.notable_moments.strengths:
                lines.append("   Strengths:")
                for s in response.notable_moments.strengths:
                    lines.append(f"   [+] {s}")
            if response.notable_moments.concerns:
                lines.append("   Concerns:")
                for c in response.notable_moments.concerns:
                    lines.append(f"   [-] {c}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("         END OF FEEDBACK REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)
