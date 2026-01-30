"""Interview Orchestrator - coordinates agents and manages interview flow."""

import re
from typing import Optional

from langfuse import observe, get_client
from src.core.llm_client import BaseLLMClient, get_llm_client
from src.core.models import CandidateInfo, Grade, FinalFeedback
from src.agents.interviewer import InterviewerAgent
from src.agents.observer import ObserverAgent
from src.agents.feedback_generator import FeedbackGeneratorAgent
from src.utils.logger import InterviewLogger
from src.utils.tracing import initialize_langfuse, is_tracing_enabled


class InterviewOrchestrator:
    """Orchestrates the multi-agent interview process."""

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        output_dir: str = "logs",
    ):
        """Initialize the orchestrator with agents.

        Args:
            llm_provider: LLM provider to use ("openai" or "anthropic")
            llm_model: Optional model override
            output_dir: Directory for interview logs
        """
        # Initialize LLM client
        kwargs = {}
        if llm_model:
            kwargs["model"] = llm_model
        self.llm = get_llm_client(llm_provider, **kwargs)

        # Initialize agents
        self.interviewer = InterviewerAgent(self.llm)
        self.observer = ObserverAgent(self.llm)
        self.feedback_generator = FeedbackGeneratorAgent(self.llm)

        # Initialize logger
        self.logger = InterviewLogger(output_dir)

        # Initialize Langfuse tracing
        self.langfuse_client = initialize_langfuse()

        # Interview state
        self.candidate_info: Optional[CandidateInfo] = None
        self.is_active = False
        self.current_question = ""
        self.turn_count = 0
        self.is_first_response = True

    @observe(name="interview_session")
    def start_interview(
        self,
        name: str,
        position: str,
        grade: str,
        experience: str,
    ) -> str:
        """Start a new interview session.

        Args:
            name: Candidate's name
            position: Target position
            grade: Target grade (Junior/Middle/Senior)
            experience: Description of candidate's experience

        Returns:
            Opening greeting from the interviewer
        """
        # Track session metadata in Langfuse
        if is_tracing_enabled():
            langfuse = get_client()
            langfuse.update_current_span(
                name=f"Interview: {name}",
                user_id=name,
                metadata={
                    "position": position,
                    "target_grade": grade,
                    "experience": experience,
                    "mode": "interactive",
                },
                tags=["interview", grade, position],
            )

        # Parse grade
        grade_map = {
            "junior": Grade.JUNIOR,
            "middle": Grade.MIDDLE,
            "senior": Grade.SENIOR,
        }
        target_grade = grade_map.get(grade.lower(), Grade.JUNIOR)

        # Create candidate info
        self.candidate_info = CandidateInfo(
            name=name,
            position=position,
            target_grade=target_grade,
            experience=experience,
        )

        # Reset agents
        self.interviewer.reset()
        self.observer.reset()

        # Initialize logger
        self.logger.start_session(
            participant_name=name,
            position=position,
            target_grade=target_grade,
            experience=experience,
        )

        # Generate greeting
        greeting, internal_thoughts = self.interviewer.generate_greeting(
            self.candidate_info
        )

        # Log the opening (no user message yet)
        self.current_question = greeting
        self.is_active = True
        self.turn_count = 0
        self.is_first_response = True

        return greeting

    def process_response(self, user_message: str) -> tuple[str, bool]:
        """Process a candidate's response and generate the next interviewer message.

        Args:
            user_message: The candidate's message

        Returns:
            Tuple of (interviewer's response, is_interview_ended)
        """
        if not self.is_active or not self.candidate_info:
            return "Interview not started. Please start an interview first.", False

        # Check for stop command
        stop_patterns = [
            r"стоп\s*(игра|интервью)?",
            r"stop\s*(game|interview)?",
            r"давай\s*фидбэк",
            r"give\s*feedback",
            r"завершить",
            r"end\s*interview",
        ]
        for pattern in stop_patterns:
            if re.search(pattern, user_message.lower()):
                return self._end_interview(user_message)

        self.turn_count += 1

        # Get conversation history
        history = self.logger.get_conversation_history()

        # Handle first response (after greeting) differently
        if self.is_first_response:
            self.is_first_response = False

            # This is the candidate's introduction
            response, interviewer_thoughts = self.interviewer.generate_first_question(
                self.candidate_info,
                user_message,
            )

            # Observer analyzes the introduction
            observer_analysis = self.observer.analyze_response(
                self.candidate_info,
                history,
                self.current_question,
                user_message,
            )
            observer_thoughts = self.observer.format_internal_thoughts(
                observer_analysis
            )

            # Combine internal thoughts
            internal_thoughts = f"{observer_thoughts} | {interviewer_thoughts}"

            # Log the turn
            self.logger.add_turn(
                agent_visible_message=self.current_question,
                user_message=user_message,
                internal_thoughts=internal_thoughts,
            )

            self.current_question = response
            return response, False

        # Regular turn: Observer analyzes first, then Interviewer responds

        # Step 1: Observer analyzes the candidate's response
        observer_analysis = self.observer.analyze_response(
            self.candidate_info,
            history,
            self.current_question,
            user_message,
        )
        observer_thoughts = self.observer.format_internal_thoughts(observer_analysis)

        # Step 2: Interviewer generates response based on Observer's analysis
        response, interviewer_thoughts = self.interviewer.generate_response(
            self.candidate_info,
            history,
            self.current_question,
            user_message,
            observer_analysis,
        )

        # Combine internal thoughts
        internal_thoughts = f"{observer_thoughts} | {interviewer_thoughts}"

        # Log the turn
        self.logger.add_turn(
            agent_visible_message=self.current_question,
            user_message=user_message,
            internal_thoughts=internal_thoughts,
        )

        self.current_question = response
        return response, False

    def _end_interview(self, final_message: str) -> tuple[str, bool]:
        """End the interview and generate feedback.

        Args:
            final_message: The candidate's final message

        Returns:
            Tuple of (feedback report, True indicating interview ended)
        """
        if not self.candidate_info:
            return "No interview in progress.", True

        # Log final turn if there's a current question
        if self.current_question:
            self.logger.add_turn(
                agent_visible_message=self.current_question,
                user_message=final_message,
                internal_thoughts="[Observer]: Interview end requested by candidate. [Interviewer]: Generating final feedback.",
            )

        # Generate feedback
        history = self.logger.get_conversation_history()
        feedback, formatted_feedback = self.feedback_generator.generate_feedback(
            self.candidate_info,
            history,
        )

        # Set feedback in logger
        self.logger.set_final_feedback(feedback)

        # Update trace with final results
        if is_tracing_enabled() and feedback:
            langfuse = get_client()
            langfuse.update_current_span(
                output={
                    "verdict": feedback.verdict,
                    "grade": feedback.grade.value
                    if hasattr(feedback.grade, "value")
                    else str(feedback.grade),
                    "total_turns": self.turn_count,
                    "hiring_recommendation": feedback.hiring_recommendation,
                },
                metadata={
                    "confirmed_skills": feedback.confirmed_skills,
                    "skill_gaps": feedback.skill_gaps,
                    "topics_covered": feedback.topics_covered
                    if hasattr(feedback, "topics_covered")
                    else [],
                },
            )

        # Save the log
        log_path = self.logger.save()

        self.is_active = False

        return formatted_feedback + f"\n\nInterview log saved to: {log_path}", True

    def get_log_path(self) -> Optional[str]:
        """Get the path to the saved log file."""
        if self.logger.session:
            return self.logger.save()
        return None

    def is_interview_active(self) -> bool:
        """Check if an interview is currently active."""
        return self.is_active
