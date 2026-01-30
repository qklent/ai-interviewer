"""Observer Agent - analyzes candidate responses and guides the Interviewer."""

from typing import Optional

from langfuse import observe, get_client
from src.core.llm_client import BaseLLMClient
from src.core.models import ObserverAnalysis, CandidateInfo, Grade
from src.utils.prompt_loader import load_prompt
from src.utils.tracing import is_tracing_enabled


# Load prompts from files
OBSERVER_SYSTEM_PROMPT, OBSERVER_SYSTEM_METADATA = load_prompt("observer", "system")
OBSERVER_ANALYSIS_PROMPT, OBSERVER_ANALYSIS_METADATA = load_prompt("observer", "analysis")


class ObserverAgent:
    """Observer agent that analyzes candidate responses."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.topics_covered: list[str] = []

    @observe(name="analyze_candidate_response")
    def analyze_response(
        self,
        candidate_info: CandidateInfo,
        conversation_history: list[dict],
        current_question: str,
        candidate_response: str,
    ) -> ObserverAnalysis:
        """Analyze a candidate's response and provide guidance."""

        # Format conversation history
        history_text = ""
        for turn in conversation_history:
            history_text += f"Interviewer: {turn['agent_message']}\n"
            history_text += f"Candidate: {turn['user_message']}\n\n"

        if not history_text:
            history_text = "This is the first turn of the interview."

        # Format topics covered
        topics_text = (
            ", ".join(self.topics_covered) if self.topics_covered else "None yet"
        )

        prompt = OBSERVER_ANALYSIS_PROMPT.format(
            name=candidate_info.name,
            position=candidate_info.position,
            target_grade=candidate_info.target_grade.value,
            experience=candidate_info.experience,
            conversation_history=history_text,
            current_question=current_question,
            candidate_response=candidate_response,
            topics_covered=topics_text,
        )

        response = self.llm.generate_json(
            system_prompt=OBSERVER_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.3,
            prompt_metadata=OBSERVER_SYSTEM_METADATA,
        )

        # Update topics covered
        new_topics = response.get("topics_covered_in_this_response", [])
        for topic in new_topics:
            if topic not in self.topics_covered:
                self.topics_covered.append(topic)

        # Build analysis object
        analysis = ObserverAnalysis(
            answer_quality=response.get("answer_quality", "partial"),
            confidence_level=response.get("confidence_level", "medium"),
            factual_accuracy=response.get("factual_accuracy", True),
            hallucination_detected=response.get("hallucination_detected", False),
            off_topic=response.get("off_topic", False),
            candidate_question_detected=response.get(
                "candidate_question_detected", False
            ),
            candidate_question=response.get("candidate_question"),
            key_observations=response.get("key_observations", []),
            recommended_action=response.get("recommended_action", ""),
            difficulty_adjustment=response.get("difficulty_adjustment", "maintain"),
            topics_covered=self.topics_covered.copy(),
        )

        # Capture analysis metadata for observability
        if analysis and is_tracing_enabled():
            langfuse = get_client()
            langfuse.update_current_span(
                metadata={
                    "quality": analysis.answer_quality,
                    "hallucination": analysis.hallucination_detected,
                    "recommended_action": analysis.recommended_action,
                    "difficulty_adjustment": analysis.difficulty_adjustment,
                    "off_topic": analysis.off_topic,
                }
            )

        return analysis

    def format_internal_thoughts(self, analysis: ObserverAnalysis) -> str:
        """Format the analysis as internal thoughts for logging."""
        thoughts = []

        # Quality assessment
        thoughts.append(f"[Observer]: Answer quality: {analysis.answer_quality}")
        thoughts.append(f"[Observer]: Confidence level: {analysis.confidence_level}")

        # Hallucination check
        if analysis.hallucination_detected:
            thoughts.append(
                "[Observer]: WARNING - Hallucination detected! Candidate made false technical claims."
            )

        # Off-topic check
        if analysis.off_topic and not analysis.hallucination_detected:
            thoughts.append(
                "[Observer]: Candidate went off-topic or answered wrong question, need to redirect."
            )
        elif analysis.off_topic and analysis.hallucination_detected:
            thoughts.append(
                "[Observer]: Candidate went off-topic with hallucinated information."
            )

        # Candidate question
        if analysis.candidate_question_detected:
            thoughts.append(
                f"[Observer]: Candidate asked a question: '{analysis.candidate_question}' - Interviewer should address it."
            )

        # Key observations
        for obs in analysis.key_observations:
            thoughts.append(f"[Observer]: {obs}")

        # Recommendation
        thoughts.append(f"[Observer -> Interviewer]: {analysis.recommended_action}")

        # Difficulty adjustment
        if analysis.difficulty_adjustment != "maintain":
            thoughts.append(
                f"[Observer]: Recommend to {analysis.difficulty_adjustment} question difficulty."
            )

        return " | ".join(thoughts)

    def get_topics_covered(self) -> list[str]:
        """Get list of topics already covered in the interview."""
        return self.topics_covered.copy()

    def reset(self) -> None:
        """Reset the observer for a new interview."""
        self.topics_covered = []
