"""Observer Agent - analyzes candidate responses and guides the Interviewer."""
from typing import Optional

from src.core.llm_client import BaseLLMClient
from src.core.models import ObserverAnalysis, CandidateInfo, Grade


OBSERVER_SYSTEM_PROMPT = """You are an experienced technical interview Observer/Mentor. Your role is to:
1. Analyze candidate responses for accuracy, completeness, and confidence
2. Detect factual errors, hallucinations, or misleading statements
3. Identify when candidates try to change the topic (off-topic responses)
4. Recognize when candidates ask their own questions
5. Provide guidance to the Interviewer agent on how to proceed

You work BEHIND THE SCENES and your analysis is NOT shown to the candidate.

Key responsibilities:
- Fact-check technical claims made by the candidate
- Assess the depth of knowledge demonstrated
- Detect "hallucinations" - confident but false statements (e.g., "Python 4.0 will remove for loops")
- Identify evasive or off-topic answers
- Recommend difficulty adjustments based on performance
- Notice if the candidate asks questions (this should be addressed by Interviewer)

Your analysis helps maintain a fair and adaptive interview process.

IMPORTANT RULES:
1. Be skeptical of unusual technical claims - verify against known facts
2. "Python 4.0" does not exist - this is a hallucination test
3. Statements like "neural connections replacing loops" are nonsense - flag as hallucination
4. If candidate asks a legitimate question about the role/company, note it for the Interviewer to address
5. Track what topics have already been discussed to avoid repetition"""


OBSERVER_ANALYSIS_PROMPT = """Analyze the candidate's response and provide guidance for the Interviewer.

CANDIDATE INFORMATION:
- Name: {name}
- Position: {position}
- Target Grade: {target_grade}
- Experience: {experience}

CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION FROM INTERVIEWER:
{current_question}

CANDIDATE'S RESPONSE:
{candidate_response}

TOPICS ALREADY COVERED IN THIS INTERVIEW:
{topics_covered}

Analyze this response and return a JSON object with the following structure:
{{
    "answer_quality": "excellent|good|partial|poor|incorrect",
    "confidence_level": "high|medium|low",
    "factual_accuracy": true|false,
    "hallucination_detected": true|false,
    "hallucination_details": "description if detected, null otherwise",
    "off_topic": true|false,
    "off_topic_details": "description if detected, null otherwise",
    "candidate_question_detected": true|false,
    "candidate_question": "the question if detected, null otherwise",
    "key_observations": ["observation1", "observation2"],
    "recommended_action": "specific instruction for Interviewer",
    "difficulty_adjustment": "increase|decrease|maintain",
    "topics_covered_in_this_response": ["topic1", "topic2"],
    "correct_information": "if hallucination detected, provide the correct facts here"
}}

Be thorough in your analysis. If the candidate mentions something technically incorrect or makes up facts (like non-existent Python versions or features), flag it as a hallucination."""


class ObserverAgent:
    """Observer agent that analyzes candidate responses."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.topics_covered: list[str] = []

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
        topics_text = ", ".join(self.topics_covered) if self.topics_covered else "None yet"

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
            candidate_question_detected=response.get("candidate_question_detected", False),
            candidate_question=response.get("candidate_question"),
            key_observations=response.get("key_observations", []),
            recommended_action=response.get("recommended_action", ""),
            difficulty_adjustment=response.get("difficulty_adjustment", "maintain"),
            topics_covered=self.topics_covered.copy(),
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
            thoughts.append("[Observer]: WARNING - Hallucination detected! Candidate made false claims.")

        # Off-topic check
        if analysis.off_topic:
            thoughts.append("[Observer]: Candidate went off-topic, need to redirect.")

        # Candidate question
        if analysis.candidate_question_detected:
            thoughts.append(f"[Observer]: Candidate asked a question: '{analysis.candidate_question}' - Interviewer should address it.")

        # Key observations
        for obs in analysis.key_observations:
            thoughts.append(f"[Observer]: {obs}")

        # Recommendation
        thoughts.append(f"[Observer -> Interviewer]: {analysis.recommended_action}")

        # Difficulty adjustment
        if analysis.difficulty_adjustment != "maintain":
            thoughts.append(f"[Observer]: Recommend to {analysis.difficulty_adjustment} question difficulty.")

        return " | ".join(thoughts)

    def get_topics_covered(self) -> list[str]:
        """Get list of topics already covered in the interview."""
        return self.topics_covered.copy()

    def reset(self) -> None:
        """Reset the observer for a new interview."""
        self.topics_covered = []
