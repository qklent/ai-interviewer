"""Pydantic schemas for structured outputs from LLM agents."""

from typing import Optional
from pydantic import BaseModel, Field


# Observer Schemas
class ObserverAnalysisSchema(BaseModel):
    """Schema for Observer agent's analysis output."""

    answer_quality: str = Field(
        description="Quality of the candidate's answer: excellent, good, partial, poor, or incorrect"
    )
    confidence_level: str = Field(
        description="Confidence level of the candidate: high, medium, or low"
    )
    factual_accuracy: bool = Field(
        description="Whether the answer is factually accurate"
    )
    hallucination_detected: bool = Field(
        description="True ONLY if candidate makes FALSE TECHNICAL CLAIMS (not for wrong answers or admitting 'I don't know')"
    )
    hallucination_details: Optional[str] = Field(
        default=None,
        description="Description of the hallucination if detected"
    )
    off_topic: bool = Field(
        description="Whether the candidate went off-topic or answered the wrong question"
    )
    off_topic_details: Optional[str] = Field(
        default=None,
        description="Description if candidate went off-topic"
    )
    candidate_question_detected: bool = Field(
        description="Whether the candidate asked a question"
    )
    candidate_question: Optional[str] = Field(
        default=None,
        description="The question asked by the candidate, if any"
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="Key observations about the candidate's response"
    )
    recommended_action: str = Field(
        description="Specific instruction for the Interviewer on what to do next"
    )
    difficulty_adjustment: str = Field(
        description="Difficulty adjustment recommendation: increase, decrease, or maintain"
    )
    topics_covered_in_this_response: list[str] = Field(
        default_factory=list,
        description="Topics covered in this specific response"
    )
    correct_information: Optional[str] = Field(
        default=None,
        description="If hallucination detected, provide the correct facts here"
    )


# Interviewer Schemas
class InterviewerGreetingSchema(BaseModel):
    """Schema for Interviewer's greeting output."""

    greeting: str = Field(
        description="The greeting message to welcome the candidate"
    )
    rationale: str = Field(
        description="Internal reasoning for why this greeting approach was chosen"
    )


class InterviewerResponseSchema(BaseModel):
    """Schema for Interviewer's response output."""

    response: str = Field(
        description="The complete response to show the candidate"
    )
    next_question: Optional[str] = Field(
        default=None,
        description="The technical question included in the response, if any"
    )
    question_difficulty: str = Field(
        default="medium",
        description="Difficulty level of the question: easy, medium, or hard"
    )
    topic: str = Field(
        description="The topic of the question"
    )
    rationale: str = Field(
        description="Internal reasoning (not shown to candidate)"
    )
    addressed_candidate_question: bool = Field(
        default=False,
        description="Whether a candidate's question was addressed"
    )
    corrected_misinformation: bool = Field(
        default=False,
        description="Whether misinformation was corrected"
    )


class InterviewerFirstQuestionSchema(BaseModel):
    """Schema for Interviewer's first question output."""

    response: str = Field(
        description="Acknowledgment of the candidate's intro plus the question"
    )
    question: str = Field(
        description="Just the technical question part"
    )
    topic: str = Field(
        description="Topic of the question"
    )
    difficulty: str = Field(
        description="Difficulty level: easy, medium, or hard"
    )
    rationale: str = Field(
        description="Why this question was chosen"
    )


# Feedback Generator Schemas
class SkillAssessmentSchema(BaseModel):
    """Schema for skill assessment."""

    topic: str = Field(
        description="The skill or topic being assessed"
    )
    details: str = Field(
        description="Details about what the candidate demonstrated or didn't know"
    )
    correct_answer: Optional[str] = Field(
        default=None,
        description="For knowledge gaps - the correct answer they should know"
    )


class SoftSkillScoreSchema(BaseModel):
    """Schema for individual soft skill score."""

    score: int = Field(
        ge=1,
        le=10,
        description="Score from 1 to 10"
    )
    notes: str = Field(
        description="Detailed notes about the score"
    )


class SoftSkillsSchema(BaseModel):
    """Schema for soft skills assessment."""

    clarity: SoftSkillScoreSchema = Field(
        description="How well the candidate explained their thoughts"
    )
    honesty: SoftSkillScoreSchema = Field(
        description="Whether they admitted when they didn't know vs making things up"
    )
    engagement: SoftSkillScoreSchema = Field(
        description="Whether they asked good questions and showed interest"
    )


class VerdictSchema(BaseModel):
    """Schema for interview verdict."""

    assessed_grade: str = Field(
        description="Assessed grade level: Junior, Middle, or Senior"
    )
    hiring_recommendation: str = Field(
        description="Hiring recommendation: No Hire, Hire, or Strong Hire"
    )
    confidence_score: int = Field(
        ge=0,
        le=100,
        description="Confidence score from 0 to 100"
    )
    summary: str = Field(
        description="1-2 sentence summary of the decision"
    )


class TechnicalReviewSchema(BaseModel):
    """Schema for technical review."""

    confirmed_skills: list[SkillAssessmentSchema] = Field(
        default_factory=list,
        description="Skills the candidate demonstrated well"
    )
    knowledge_gaps: list[SkillAssessmentSchema] = Field(
        default_factory=list,
        description="Areas where the candidate had gaps or errors"
    )


class RoadmapSchema(BaseModel):
    """Schema for learning roadmap."""

    topics_to_improve: list[str] = Field(
        default_factory=list,
        description="List of topics the candidate should improve"
    )
    specific_recommendations: list[str] = Field(
        default_factory=list,
        description="Specific actionable recommendations"
    )
    resources: list[str] = Field(
        default_factory=list,
        description="Optional resource links or suggestions"
    )


class NotableMomentsSchema(BaseModel):
    """Schema for notable moments in the interview."""

    strengths: list[str] = Field(
        default_factory=list,
        description="Specific strong moments from the interview"
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Specific concerning moments"
    )


class FeedbackSchema(BaseModel):
    """Schema for complete feedback output."""

    verdict: VerdictSchema = Field(
        description="Interview verdict and hiring recommendation"
    )
    technical_review: TechnicalReviewSchema = Field(
        description="Technical skills assessment"
    )
    soft_skills: SoftSkillsSchema = Field(
        description="Soft skills and communication assessment"
    )
    roadmap: RoadmapSchema = Field(
        description="Personalized learning roadmap"
    )
    notable_moments: NotableMomentsSchema = Field(
        default_factory=NotableMomentsSchema,
        description="Notable strengths and concerns"
    )
