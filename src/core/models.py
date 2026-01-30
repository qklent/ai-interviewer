"""Data models for the interview system."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Grade(str, Enum):
    JUNIOR = "Junior"
    MIDDLE = "Middle"
    SENIOR = "Senior"


class HiringRecommendation(str, Enum):
    NO_HIRE = "No Hire"
    HIRE = "Hire"
    STRONG_HIRE = "Strong Hire"


@dataclass
class CandidateInfo:
    """Information about the candidate."""
    name: str
    position: str
    target_grade: Grade
    experience: str


@dataclass
class Turn:
    """A single turn in the interview conversation."""
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str


@dataclass
class SkillAssessment:
    """Assessment of a specific skill/topic."""
    topic: str
    status: str  # "confirmed" or "gap"
    details: str
    correct_answer: Optional[str] = None  # For gaps - what the correct answer should be


@dataclass
class SoftSkillsAssessment:
    """Assessment of soft skills."""
    clarity: int  # 1-10
    clarity_notes: str
    honesty: int  # 1-10
    honesty_notes: str
    engagement: int  # 1-10
    engagement_notes: str


@dataclass
class FinalFeedback:
    """Final feedback structure."""
    # Verdict
    assessed_grade: Grade
    hiring_recommendation: HiringRecommendation
    confidence_score: int  # 0-100

    # Hard Skills
    confirmed_skills: list[SkillAssessment] = field(default_factory=list)
    knowledge_gaps: list[SkillAssessment] = field(default_factory=list)

    # Soft Skills
    soft_skills: Optional[SoftSkillsAssessment] = None

    # Roadmap
    topics_to_improve: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    roadmap: list[str] = field(default_factory=list)  # Deprecated, kept for backward compat


@dataclass
class InterviewSession:
    """Complete interview session data."""
    participant_name: str
    position: str
    target_grade: Grade
    experience: str
    turns: list[Turn] = field(default_factory=list)
    final_feedback: Optional[FinalFeedback] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    intermediate_feedbacks: Optional[dict] = None  # For multi-model feedback


@dataclass
class ObserverAnalysis:
    """Analysis from the Observer agent."""
    answer_quality: str  # "excellent", "good", "partial", "poor", "incorrect"
    confidence_level: str  # "high", "medium", "low"
    factual_accuracy: bool
    hallucination_detected: bool
    off_topic: bool
    candidate_question_detected: bool
    candidate_question: Optional[str] = None
    key_observations: list[str] = field(default_factory=list)
    recommended_action: str = ""  # Instruction for Interviewer
    difficulty_adjustment: str = "maintain"  # "increase", "decrease", "maintain"
    topics_covered: list[str] = field(default_factory=list)


@dataclass
class InterviewerDecision:
    """Decision from the Interviewer agent."""
    next_question: str
    question_difficulty: str  # "easy", "medium", "hard"
    topic: str
    rationale: str
    should_give_hint: bool = False
    hint: Optional[str] = None
    response_to_candidate_question: Optional[str] = None
