"""Feedback Generator Agent - creates comprehensive final feedback."""

from typing import Optional

from langfuse import observe
from src.core.llm_client import BaseLLMClient
from src.core.models import (
    FinalFeedback,
    SkillAssessment,
    SoftSkillsAssessment,
    CandidateInfo,
    Grade,
    HiringRecommendation,
)
from src.core.schemas import FeedbackSchema
from src.utils.prompt_loader import load_prompt


# Load prompts from files
FEEDBACK_SYSTEM_PROMPT, FEEDBACK_SYSTEM_METADATA = load_prompt("feedback_generator", "system")
FEEDBACK_PROMPT, FEEDBACK_PROMPT_METADATA = load_prompt("feedback_generator", "feedback")


class FeedbackGeneratorAgent:
    """Agent that generates comprehensive final feedback."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    @observe(name="FeedbackGenerator: Generate Final Feedback")
    def generate_feedback(
        self,
        candidate_info: CandidateInfo,
        conversation_history: list[dict],
    ) -> tuple[FinalFeedback, str]:
        """Generate final feedback for the interview.

        Returns:
            Tuple of (FinalFeedback object, formatted feedback string for display)
        """
        # Format transcript
        transcript = ""
        for turn in conversation_history:
            transcript += f"Interviewer: {turn['agent_message']}\n"
            transcript += f"Candidate: {turn['user_message']}\n\n"

        if not transcript:
            transcript = "No conversation recorded."

        prompt = FEEDBACK_PROMPT.format(
            name=candidate_info.name,
            position=candidate_info.position,
            target_grade=candidate_info.target_grade.value,
            experience=candidate_info.experience,
            transcript=transcript,
        )

        # Use structured outputs with Pydantic schema
        response = self.llm.generate_structured(
            system_prompt=FEEDBACK_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_format=FeedbackSchema,
            temperature=0.3,
            max_tokens=3000,
            prompt_metadata=FEEDBACK_SYSTEM_METADATA,
        )

        # Map grade string to enum
        grade_map = {
            "Junior": Grade.JUNIOR,
            "Middle": Grade.MIDDLE,
            "Senior": Grade.SENIOR,
        }
        assessed_grade = grade_map.get(
            response.verdict.assessed_grade, Grade.JUNIOR
        )

        # Map hiring recommendation to enum
        hire_map = {
            "No Hire": HiringRecommendation.NO_HIRE,
            "Hire": HiringRecommendation.HIRE,
            "Strong Hire": HiringRecommendation.STRONG_HIRE,
        }
        hiring_rec = hire_map.get(
            response.verdict.hiring_recommendation,
            HiringRecommendation.NO_HIRE,
        )

        # Build confirmed skills from structured response
        confirmed_skills = [
            SkillAssessment(
                topic=skill.topic,
                status="confirmed",
                details=skill.details,
            )
            for skill in response.technical_review.confirmed_skills
        ]

        # Build knowledge gaps from structured response
        knowledge_gaps = [
            SkillAssessment(
                topic=gap.topic,
                status="gap",
                details=gap.details,
                correct_answer=gap.correct_answer,
            )
            for gap in response.technical_review.knowledge_gaps
        ]

        # Build soft skills assessment from structured response
        soft_skills = SoftSkillsAssessment(
            clarity=response.soft_skills.clarity.score,
            clarity_notes=response.soft_skills.clarity.notes,
            honesty=response.soft_skills.honesty.score,
            honesty_notes=response.soft_skills.honesty.notes,
            engagement=response.soft_skills.engagement.score,
            engagement_notes=response.soft_skills.engagement.notes,
        )

        # Build final feedback object
        feedback = FinalFeedback(
            assessed_grade=assessed_grade,
            hiring_recommendation=hiring_rec,
            confidence_score=response.verdict.confidence_score,
            confirmed_skills=confirmed_skills,
            knowledge_gaps=knowledge_gaps,
            soft_skills=soft_skills,
            topics_to_improve=response.roadmap.topics_to_improve,
            recommended_actions=response.roadmap.specific_recommendations,
            roadmap=response.roadmap.topics_to_improve + response.roadmap.specific_recommendations,  # Deprecated
            resources=response.roadmap.resources,
        )

        # Format for display
        formatted = self._format_feedback_for_display(feedback, response)

        return feedback, formatted

    def _format_feedback_for_display(
        self, feedback: FinalFeedback, response: FeedbackSchema
    ) -> str:
        """Format the feedback for human-readable display."""
        lines = []
        lines.append("=" * 60)
        lines.append("           INTERVIEW FEEDBACK REPORT")
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

        # Get summary from structured response
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

        # Topics to improve
        if feedback.topics_to_improve:
            lines.append("   Topics to Improve:")
            for topic in feedback.topics_to_improve:
                lines.append(f"   - {topic}")
            lines.append("")

        # Recommended actions
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

        # Notable moments from structured response
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
