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
    - Multiple evaluator models: Generate independent feedback evaluations
    - Single aggregator model: Synthesizes all evaluations into final feedback
    """

    def __init__(
        self,
        evaluator_models: list[str],
        aggregator_model: str,
        save_intermediate: bool = True,
    ):
        """Initialize multi-model feedback generator.

        Args:
            evaluator_models: List of model identifiers for independent evaluations
                            (e.g., ['google/gemini-2.0-flash', 'anthropic/claude-3.5-sonnet'])
            aggregator_model: Model identifier for aggregating evaluations
                            (e.g., 'openai/gpt-4o')
            save_intermediate: Whether to save intermediate feedback results
        """
        if not evaluator_models or len(evaluator_models) < 2:
            raise ValueError("At least 2 evaluator models are required")

        logger.info(
            f"Initializing MultiModelFeedbackGenerator: "
            f"evaluators={evaluator_models}, aggregator={aggregator_model}"
        )

        # Create LLM clients and generators for each evaluator model
        self.evaluator_models = evaluator_models
        self.evaluator_llms = [
            get_llm_client("openrouter", model=model) for model in evaluator_models
        ]
        self.evaluator_generators = [
            FeedbackGeneratorAgent(llm) for llm in self.evaluator_llms
        ]

        # Create aggregator LLM client
        self.aggregator_model = aggregator_model
        self.aggregator_llm = get_llm_client("openrouter", model=aggregator_model)

        self.save_intermediate = save_intermediate
        self.fallback_mode = os.getenv("MULTIMODEL_FALLBACK_MODE", "partial")

        logger.info(
            f"MultiModelFeedbackGenerator initialized with {len(evaluator_models)} evaluators "
            f"and fallback_mode={self.fallback_mode}"
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
            - intermediate_results is dict with all evaluator outputs (if save_intermediate=True)
        """
        logger.info("Starting multi-model feedback generation")

        # Stage 1: Generate independent feedbacks from all evaluators
        evaluator_results = self._generate_independent_feedbacks(
            candidate_info, conversation_history
        )

        # Filter out failed evaluations
        successful_results = [
            (feedback, formatted, model)
            for feedback, formatted, model in evaluator_results
            if feedback is not None
        ]

        # Handle failure cases
        if len(successful_results) == 0:
            logger.error("All evaluation models failed, falling back to single-model")
            return self._fallback_single_model(candidate_info, conversation_history)

        if len(successful_results) == 1:
            logger.warning("Only one model succeeded, using single evaluation")
            return self._handle_single_success(successful_results[0], evaluator_results)

        # Stage 2: Aggregate successful feedbacks
        try:
            final_feedback, formatted_final = self._aggregate_feedbacks(
                candidate_info, successful_results
            )
            logger.info("Multi-model feedback generation completed successfully")
        except Exception as e:
            logger.error(f"Aggregation failed: {e}, falling back to first successful feedback")
            # Use first successful evaluation as fallback
            final_feedback = successful_results[0][0]
            formatted_final = successful_results[0][1]

        # Build intermediate results if requested
        intermediate = None
        if self.save_intermediate:
            intermediate = {}
            for i, (feedback, formatted, model) in enumerate(evaluator_results):
                intermediate[f"evaluator_{i}_model"] = model
                intermediate[f"evaluator_{i}_feedback"] = (
                    self._serialize_feedback(feedback) if feedback else None
                )
                intermediate[f"evaluator_{i}_formatted"] = formatted

        return final_feedback, formatted_final, intermediate

    @observe(name="MultiModel: Independent Feedbacks")
    def _generate_independent_feedbacks(
        self,
        candidate_info: CandidateInfo,
        conversation_history: list[dict],
    ) -> list[tuple[Optional[FinalFeedback], Optional[str], str]]:
        """Generate independent feedbacks from all evaluator models.

        Returns:
            List of tuples: [(feedback, formatted_string, model_name), ...]
        """
        results = []

        for i, (generator, model) in enumerate(
            zip(self.evaluator_generators, self.evaluator_models)
        ):
            feedback = None
            formatted = None

            try:
                logger.info(f"Generating feedback from evaluator {i+1} ({model})...")
                feedback, formatted = generator.generate_feedback(
                    candidate_info, conversation_history
                )
                logger.info(f"Evaluator {i+1} ({model}) feedback generated successfully")
            except Exception as e:
                logger.error(
                    f"Evaluator {i+1} ({model}) feedback generation failed: {e}",
                    exc_info=True,
                    extra={
                        "model": model,
                        "evaluator_index": i,
                        "candidate": candidate_info.name,
                        "error_type": type(e).__name__,
                    },
                )

            results.append((feedback, formatted, model))

        return results

    @observe(name="MultiModel: Aggregate Feedbacks")
    def _aggregate_feedbacks(
        self,
        candidate_info: CandidateInfo,
        successful_results: list[tuple[FinalFeedback, str, str]],
    ) -> tuple[FinalFeedback, str]:
        """Aggregate multiple independent feedbacks using aggregator model.

        Args:
            candidate_info: Candidate information
            successful_results: List of (feedback, formatted, model_name) tuples

        Returns:
            Tuple of (aggregated_feedback, formatted_string)
        """
        logger.info(
            f"Starting feedback aggregation with {self.aggregator_model} "
            f"({len(successful_results)} feedbacks to aggregate)"
        )

        # Serialize all feedbacks to JSON for aggregator
        import json

        feedbacks_dict = [
            {
                "model": model,
                "feedback": self._serialize_feedback(feedback),
            }
            for feedback, _, model in successful_results
        ]

        # Load aggregation prompts
        aggregate_system, system_metadata = load_prompt(
            "feedback_generator", "aggregate_system"
        )
        aggregate_template, template_metadata = load_prompt(
            "feedback_generator", "aggregate_feedback"
        )

        # Format all feedbacks as readable sections
        feedbacks_sections = []
        for i, item in enumerate(feedbacks_dict, 1):
            section = f"### EVALUATOR {i} ({item['model']}) ASSESSMENT\n\n"
            section += json.dumps(item['feedback'], indent=2, ensure_ascii=False)
            feedbacks_sections.append(section)

        all_feedbacks_str = "\n\n".join(feedbacks_sections)

        # Format aggregation prompt with all feedbacks
        aggregate_prompt = aggregate_template.format(
            name=candidate_info.name,
            position=candidate_info.position,
            target_grade=candidate_info.target_grade.value,
            experience=candidate_info.experience,
            all_feedbacks=all_feedbacks_str,
            num_feedbacks=len(successful_results),
            # Backward compatibility: provide first two feedbacks if template uses them
            feedback_1=json.dumps(feedbacks_dict[0]["feedback"], indent=2, ensure_ascii=False)
            if len(feedbacks_dict) > 0
            else "{}",
            feedback_2=json.dumps(feedbacks_dict[1]["feedback"], indent=2, ensure_ascii=False)
            if len(feedbacks_dict) > 1
            else "{}",
        )

        # Generate aggregated feedback with aggregator model
        response = self.aggregator_llm.generate_structured(
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

    def _handle_single_success(
        self,
        successful_result: tuple[FinalFeedback, str, str],
        all_results: list[tuple[Optional[FinalFeedback], Optional[str], str]],
    ) -> tuple[FinalFeedback, str, Optional[dict]]:
        """Handle case where only one model succeeded.

        Returns single successful feedback (skips aggregation).
        """
        if self.fallback_mode == "fail":
            logger.error("Fallback mode is 'fail', raising exception")
            raise Exception("Only one model succeeded and fallback_mode=fail")

        if self.fallback_mode == "single_model":
            logger.warning("Fallback mode is 'single_model', using default model")
            return self._fallback_single_model(None, None)

        # Default: partial mode - use the successful evaluation
        feedback, formatted, model = successful_result

        logger.warning(f"Using {model} feedback (no aggregation - only 1 succeeded)")

        intermediate = None
        if self.save_intermediate:
            intermediate = {}
            for i, (fb, fmt, mdl) in enumerate(all_results):
                intermediate[f"evaluator_{i}_model"] = mdl
                intermediate[f"evaluator_{i}_feedback"] = (
                    self._serialize_feedback(fb) if fb else None
                )
                intermediate[f"evaluator_{i}_formatted"] = fmt
            intermediate["note"] = "Only one evaluator succeeded - no aggregation performed"

        return feedback, formatted, intermediate

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
