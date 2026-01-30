"""Interview session logger for saving sessions to JSON."""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.core.models import (
    InterviewSession,
    Turn,
    FinalFeedback,
    SkillAssessment,
    SoftSkillsAssessment,
    Grade,
    HiringRecommendation,
)


class InterviewLogger:
    """Logger for saving interview sessions to JSON files."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[InterviewSession] = None

    def start_session(
        self,
        participant_name: str,
        position: str,
        target_grade: Grade,
        experience: str,
    ) -> None:
        """Initialize a new interview session."""
        self.session = InterviewSession(
            participant_name=participant_name,
            position=position,
            target_grade=target_grade,
            experience=experience,
            turns=[],
            final_feedback=None,
        )

    def add_turn(
        self,
        agent_visible_message: str,
        user_message: str,
        internal_thoughts: str,
    ) -> int:
        """Add a conversation turn to the session."""
        if self.session is None:
            raise ValueError("Session not started. Call start_session() first.")

        turn_id = len(self.session.turns) + 1
        turn = Turn(
            turn_id=turn_id,
            agent_visible_message=agent_visible_message,
            user_message=user_message,
            internal_thoughts=internal_thoughts,
        )
        self.session.turns.append(turn)
        return turn_id

    def set_final_feedback(self, feedback: FinalFeedback) -> None:
        """Set the final feedback for the session."""
        if self.session is None:
            raise ValueError("Session not started. Call start_session() first.")
        self.session.final_feedback = feedback
        self.session.ended_at = datetime.now().isoformat()

    def set_intermediate_feedbacks(self, intermediate: dict) -> None:
        """Store intermediate feedback results from multi-model generation."""
        if self.session is None:
            raise ValueError("Session not started. Call start_session() first.")
        self.session.intermediate_feedbacks = intermediate

    def _serialize_feedback(self, feedback: FinalFeedback) -> dict:
        """Serialize FinalFeedback to a dictionary."""
        result = {
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
                        "topic": s.topic,
                        "details": s.details,
                        "correct_answer": s.correct_answer,
                    }
                    for s in feedback.knowledge_gaps
                ],
            },
            "soft_skills": None,
            "roadmap": {
                "topics_to_improve": feedback.topics_to_improve,
                "recommended_actions": feedback.recommended_actions,
                "resources": feedback.resources,
            },
        }

        if feedback.soft_skills:
            result["soft_skills"] = {
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

        return result

    def save(self, filename: Optional[str] = None) -> str:
        """Save the session to a JSON file."""
        if self.session is None:
            raise ValueError("Session not started. Call start_session() first.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.session.participant_name.replace(" ", "_")
            filename = f"interview_{safe_name}_{timestamp}.json"

        filepath = self.output_dir / filename

        # Build the output dictionary
        output = {
            "participant_name": self.session.participant_name,
            "position": self.session.position,
            "target_grade": self.session.target_grade.value,
            "experience": self.session.experience,
            "started_at": self.session.started_at,
            "ended_at": self.session.ended_at,
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "agent_visible_message": turn.agent_visible_message,
                    "user_message": turn.user_message,
                    "internal_thoughts": turn.internal_thoughts,
                }
                for turn in self.session.turns
            ],
            "final_feedback": (
                self._serialize_feedback(self.session.final_feedback)
                if self.session.final_feedback
                else None
            ),
            "intermediate_feedbacks": self.session.intermediate_feedbacks,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return str(filepath)

    def get_conversation_history(self) -> list[dict]:
        """Get the conversation history for context."""
        if self.session is None:
            return []

        return [
            {
                "turn_id": turn.turn_id,
                "agent_message": turn.agent_visible_message,
                "user_message": turn.user_message,
            }
            for turn in self.session.turns
        ]

    def get_all_user_messages(self) -> list[str]:
        """Get all user messages from the session."""
        if self.session is None:
            return []
        return [turn.user_message for turn in self.session.turns]

    def get_topics_discussed(self) -> list[str]:
        """Extract topics from internal thoughts."""
        if self.session is None:
            return []

        topics = []
        for turn in self.session.turns:
            # Simple extraction - in real system would parse more carefully
            if "topic:" in turn.internal_thoughts.lower():
                # Extract topic mentions
                pass
        return topics
