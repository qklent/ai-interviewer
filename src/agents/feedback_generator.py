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
from src.utils.prompt_loader import load_prompt


# Load prompts from files
FEEDBACK_SYSTEM_PROMPT, FEEDBACK_SYSTEM_METADATA = load_prompt("feedback_generator", "system")
FEEDBACK_PROMPT, FEEDBACK_PROMPT_METADATA = load_prompt("feedback_generator", "feedback")


class FeedbackGeneratorAgent:
    """Agent that generates comprehensive final feedback."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    @observe(name="generate_final_feedback")
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

        response = self.llm.generate_json(
            system_prompt=FEEDBACK_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=3000,
            prompt_metadata=FEEDBACK_SYSTEM_METADATA,
        )

        # Parse the response into FinalFeedback object
        verdict = response.get("verdict", {})
        tech_review = response.get("technical_review", {})
        soft_skills_data = response.get("soft_skills", {})
        roadmap_data = response.get("roadmap", {})

        # Map grade string to enum
        grade_map = {
            "Junior": Grade.JUNIOR,
            "Middle": Grade.MIDDLE,
            "Senior": Grade.SENIOR,
        }
        assessed_grade = grade_map.get(
            verdict.get("assessed_grade", "Junior"), Grade.JUNIOR
        )

        # Map hiring recommendation to enum
        hire_map = {
            "No Hire": HiringRecommendation.NO_HIRE,
            "Hire": HiringRecommendation.HIRE,
            "Strong Hire": HiringRecommendation.STRONG_HIRE,
        }
        hiring_rec = hire_map.get(
            verdict.get("hiring_recommendation", "No Hire"),
            HiringRecommendation.NO_HIRE,
        )

        # Build confirmed skills
        confirmed_skills = [
            SkillAssessment(
                topic=skill.get("topic", "Unknown"),
                status="confirmed",
                details=skill.get("details", ""),
            )
            for skill in tech_review.get("confirmed_skills", [])
        ]

        # Build knowledge gaps
        knowledge_gaps = [
            SkillAssessment(
                topic=gap.get("topic", "Unknown"),
                status="gap",
                details=gap.get("details", ""),
                correct_answer=gap.get("correct_answer"),
            )
            for gap in tech_review.get("knowledge_gaps", [])
        ]

        # Build soft skills assessment
        clarity_data = soft_skills_data.get("clarity", {})
        honesty_data = soft_skills_data.get("honesty", {})
        engagement_data = soft_skills_data.get("engagement", {})

        soft_skills = SoftSkillsAssessment(
            clarity=clarity_data.get("score", 5),
            clarity_notes=clarity_data.get("notes", ""),
            honesty=honesty_data.get("score", 5),
            honesty_notes=honesty_data.get("notes", ""),
            engagement=engagement_data.get("score", 5),
            engagement_notes=engagement_data.get("notes", ""),
        )

        # Build final feedback object
        feedback = FinalFeedback(
            assessed_grade=assessed_grade,
            hiring_recommendation=hiring_rec,
            confidence_score=verdict.get("confidence_score", 50),
            confirmed_skills=confirmed_skills,
            knowledge_gaps=knowledge_gaps,
            soft_skills=soft_skills,
            topics_to_improve=roadmap_data.get("topics_to_improve", []),
            recommended_actions=roadmap_data.get("specific_recommendations", []),
            roadmap=roadmap_data.get("topics_to_improve", [])  # Deprecated
            + roadmap_data.get("specific_recommendations", []),
            resources=roadmap_data.get("resources", []),
        )

        # Format for display
        formatted = self._format_feedback_for_display(feedback, response)

        return feedback, formatted

    def _format_feedback_for_display(
        self, feedback: FinalFeedback, raw_response: dict
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

        verdict = raw_response.get("verdict", {})
        if verdict.get("summary"):
            lines.append(f"   Summary: {verdict['summary']}")
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

        # Notable moments
        notable = raw_response.get("notable_moments", {})
        if notable.get("strengths") or notable.get("concerns"):
            lines.append("E. NOTABLE MOMENTS")
            lines.append("-" * 40)
            if notable.get("strengths"):
                lines.append("   Strengths:")
                for s in notable["strengths"]:
                    lines.append(f"   [+] {s}")
            if notable.get("concerns"):
                lines.append("   Concerns:")
                for c in notable["concerns"]:
                    lines.append(f"   [-] {c}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("         END OF FEEDBACK REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)
