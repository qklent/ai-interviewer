"""Interviewer Agent - conducts the technical interview."""

from typing import Optional

from langfuse import observe
from src.core.llm_client import BaseLLMClient
from src.core.models import (
    InterviewerDecision,
    ObserverAnalysis,
    CandidateInfo,
    Grade,
)
from src.utils.prompt_loader import load_prompt


# Load prompts from files
INTERVIEWER_SYSTEM_PROMPT, INTERVIEWER_SYSTEM_METADATA = load_prompt("interviewer", "system")
INTERVIEWER_GREETING_PROMPT, INTERVIEWER_GREETING_METADATA = load_prompt("interviewer", "greeting")
INTERVIEWER_RESPONSE_PROMPT, INTERVIEWER_RESPONSE_METADATA = load_prompt("interviewer", "response")


class InterviewerAgent:
    """Interviewer agent that conducts the technical interview."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.current_difficulty = "medium"
        self.questions_asked: list[str] = []

    @observe(name="Interviewer: Generate Greeting")
    def generate_greeting(self, candidate_info: CandidateInfo) -> tuple[str, str]:
        """Generate an opening greeting for the interview.

        Returns:
            Tuple of (greeting message, internal rationale)
        """
        prompt = INTERVIEWER_GREETING_PROMPT.format(
            name=candidate_info.name,
            position=candidate_info.position,
            target_grade=candidate_info.target_grade.value,
            experience=candidate_info.experience,
        )

        response = self.llm.generate_json(
            system_prompt=INTERVIEWER_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.7,
            prompt_metadata=INTERVIEWER_SYSTEM_METADATA,
        )

        greeting = response.get("greeting", "Hello! Please tell me about yourself.")
        rationale = response.get("rationale", "Standard greeting")

        return greeting, f"[Interviewer]: {rationale}"

    @observe(name="Interviewer: Generate Response")
    def generate_response(
        self,
        candidate_info: CandidateInfo,
        conversation_history: list[dict],
        current_question: str,
        candidate_response: str,
        observer_analysis: ObserverAnalysis,
    ) -> tuple[str, str]:
        """Generate a response based on the candidate's answer and Observer's analysis.

        Returns:
            Tuple of (response message, internal rationale)
        """
        # Format conversation history
        history_text = ""
        for turn in conversation_history[:-1] if conversation_history else []:
            history_text += f"Interviewer: {turn['agent_message']}\n"
            history_text += f"Candidate: {turn['user_message']}\n\n"

        if not history_text:
            history_text = "This is the beginning of the interview."

        # Format topics covered
        topics_text = (
            ", ".join(observer_analysis.topics_covered)
            if observer_analysis.topics_covered
            else "None yet"
        )

        # Format observations
        observations_text = (
            "; ".join(observer_analysis.key_observations)
            if observer_analysis.key_observations
            else "No specific observations"
        )

        prompt = INTERVIEWER_RESPONSE_PROMPT.format(
            name=candidate_info.name,
            position=candidate_info.position,
            target_grade=candidate_info.target_grade.value,
            experience=candidate_info.experience,
            conversation_history=history_text,
            current_question=current_question,
            candidate_response=candidate_response,
            answer_quality=observer_analysis.answer_quality,
            factual_accuracy=observer_analysis.factual_accuracy,
            hallucination_detected=observer_analysis.hallucination_detected,
            off_topic=observer_analysis.off_topic,
            candidate_question_detected=observer_analysis.candidate_question_detected,
            candidate_question=observer_analysis.candidate_question or "None",
            observations=observations_text,
            recommended_action=observer_analysis.recommended_action,
            difficulty_adjustment=observer_analysis.difficulty_adjustment,
            topics_covered=topics_text,
        )

        response = self.llm.generate_json(
            system_prompt=INTERVIEWER_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.7,
            prompt_metadata=INTERVIEWER_SYSTEM_METADATA,
        )

        message = response.get("response", "Let's continue with the next question.")
        rationale = response.get("rationale", "Continuing interview")
        topic = response.get("topic", "general")
        difficulty = response.get("question_difficulty", "medium")

        # Update internal state
        self.current_difficulty = difficulty
        if response.get("next_question"):
            self.questions_asked.append(response["next_question"])

        # Build internal thoughts string
        internal_parts = [f"[Interviewer]: {rationale}"]
        if response.get("addressed_candidate_question"):
            internal_parts.append(
                "[Interviewer]: Addressed candidate's question before continuing."
            )
        if response.get("corrected_misinformation"):
            internal_parts.append(
                "[Interviewer]: Corrected factual error in candidate's response."
            )
        internal_parts.append(
            f"[Interviewer]: Topic: {topic}, Difficulty: {difficulty}"
        )

        internal_thoughts = " | ".join(internal_parts)

        return message, internal_thoughts

    @observe(name="Interviewer: Generate First Question")
    def generate_first_question(
        self,
        candidate_info: CandidateInfo,
        candidate_intro: str,
    ) -> tuple[str, str]:
        """Generate the first technical question based on candidate's introduction.

        Returns:
            Tuple of (question message, internal rationale)
        """
        prompt = f"""Based on the candidate's introduction, generate an appropriate first technical question.

CANDIDATE INFORMATION:
- Name: {candidate_info.name}
- Position: {candidate_info.position}
- Target Grade: {candidate_info.target_grade.value}
- Experience: {candidate_info.experience}

CANDIDATE'S INTRODUCTION:
{candidate_intro}

Generate a question that:
1. Is appropriate for their stated experience level
2. Relates to technologies they mentioned (if any)
3. Starts with fundamentals before going deeper
4. Is clear and specific

Return a JSON object:
{{
    "response": "your acknowledgment of their intro + the question",
    "question": "just the technical question part",
    "topic": "topic of the question",
    "difficulty": "easy|medium|hard",
    "rationale": "why you chose this question"
}}"""

        response = self.llm.generate_json(
            system_prompt=INTERVIEWER_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.7,
            prompt_metadata=INTERVIEWER_SYSTEM_METADATA,
        )

        message = response.get(
            "response", "Great! Let me ask you a technical question."
        )
        rationale = response.get("rationale", "Starting with basics")
        topic = response.get("topic", "fundamentals")
        difficulty = response.get("difficulty", "easy")

        self.current_difficulty = difficulty
        if response.get("question"):
            self.questions_asked.append(response["question"])

        internal = (
            f"[Interviewer]: {rationale} | Topic: {topic}, Difficulty: {difficulty}"
        )

        return message, internal

    def get_current_difficulty(self) -> str:
        """Get the current question difficulty level."""
        return self.current_difficulty

    def reset(self) -> None:
        """Reset the interviewer for a new interview."""
        self.current_difficulty = "medium"
        self.questions_asked = []
