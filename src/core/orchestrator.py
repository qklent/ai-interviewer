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
from src.utils.prompt_loader import _loader as prompt_loader
from src.utils.app_logger import get_logger

logger = get_logger(__name__)


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
        logger.info(f"Initializing InterviewOrchestrator with provider={llm_provider}, model={llm_model}")

        try:
            # Initialize LLM client
            kwargs = {}
            if llm_model:
                kwargs["model"] = llm_model
            self.llm = get_llm_client(llm_provider, **kwargs)
            logger.info(f"LLM client initialized successfully: {type(self.llm).__name__}")

            # Initialize agents
            self.interviewer = InterviewerAgent(self.llm)
            self.observer = ObserverAgent(self.llm)
            self.feedback_generator = FeedbackGeneratorAgent(self.llm)
            logger.info("All agents initialized successfully")

            # Initialize logger
            self.logger = InterviewLogger(output_dir)

            # Initialize Langfuse tracing
            self.langfuse_client = initialize_langfuse()

            # Refresh prompt loader to use Langfuse if available
            prompt_loader.refresh_langfuse_status()

            # Interview state
            self.candidate_info: Optional[CandidateInfo] = None
            self.is_active = False
            self.current_question = ""
            self.turn_count = 0
            self.is_first_response = True

        except Exception as e:
            logger.exception(f"Failed to initialize orchestrator: {e}")
            raise

    @observe(name="Orchestrator: Interview Session", metadata={"agent": "orchestrator"})
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
        logger.info(f"Starting interview for {name} - Position: {position}, Grade: {grade}")

        try:
            # Track session metadata in Langfuse
            if is_tracing_enabled():
                try:
                    langfuse = get_client()
                    # Update trace-level attributes
                    langfuse.update_current_trace(
                        user_id=name,
                        tags=["interview", grade, position],
                    )
                    # Update span-level attributes
                    langfuse.update_current_span(
                        name=f"Interview: {name}",
                        metadata={
                            "position": position,
                            "target_grade": grade,
                            "experience": experience,
                            "mode": "interactive",
                        },
                    )
                except Exception as e:
                    logger.warning(f"Could not update Langfuse trace/span: {e}")
                    print(f"⚠️  Could not update Langfuse trace/span: {e}")

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
            logger.debug("Agents reset successfully")

            # Initialize logger
            self.logger.start_session(
                participant_name=name,
                position=position,
                target_grade=target_grade,
                experience=experience,
            )

            # Generate greeting
            logger.debug("Generating greeting...")
            greeting, internal_thoughts = self.interviewer.generate_greeting(
                self.candidate_info
            )
            logger.debug(f"Greeting generated successfully (length: {len(greeting)})")

            # Log the opening (no user message yet)
            self.current_question = greeting
            self.is_active = True
            self.turn_count = 0
            self.is_first_response = True

            logger.info("Interview started successfully")
            return greeting

        except Exception as e:
            logger.exception(f"Failed to start interview: {e}")
            raise

    @observe(name="Orchestrator: Process Response", metadata={"agent": "orchestrator"})
    def process_response(self, user_message: str) -> tuple[str, bool]:
        """Process a candidate's response and generate the next interviewer message.

        Args:
            user_message: The candidate's message

        Returns:
            Tuple of (interviewer's response, is_interview_ended)
        """
        logger.debug(f"Processing response (turn {self.turn_count + 1}): {user_message[:100]}...")

        try:
            if not self.is_active or not self.candidate_info:
                logger.warning("Attempted to process response without active interview")
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
                    logger.info(f"Stop command detected: {pattern}")
                    return self._end_interview(user_message)

            self.turn_count += 1

            # Get conversation history
            history = self.logger.get_conversation_history()

            # Handle first response (after greeting) differently
            if self.is_first_response:
                logger.debug("Processing first response after greeting")
                self.is_first_response = False

                # This is the candidate's introduction
                response, interviewer_thoughts = self.interviewer.generate_first_question(
                    self.candidate_info,
                    user_message,
                )
                logger.debug(f"First question generated (length: {len(response)})")

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
                logger.debug("Observer analysis completed for first response")

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
            logger.debug("Processing regular turn")

            # Step 1: Observer analyzes the candidate's response
            observer_analysis = self.observer.analyze_response(
                self.candidate_info,
                history,
                self.current_question,
                user_message,
            )
            observer_thoughts = self.observer.format_internal_thoughts(observer_analysis)
            logger.debug(f"Observer analysis: quality={observer_analysis.answer_quality}, hallucination={observer_analysis.hallucination_detected}")

            # Step 2: Interviewer generates response based on Observer's analysis
            response, interviewer_thoughts = self.interviewer.generate_response(
                self.candidate_info,
                history,
                self.current_question,
                user_message,
                observer_analysis,
            )
            logger.debug(f"Interviewer response generated (length: {len(response)})")

            # Combine internal thoughts
            internal_thoughts = f"{observer_thoughts} | {interviewer_thoughts}"

            # Log the turn
            self.logger.add_turn(
                agent_visible_message=self.current_question,
                user_message=user_message,
                internal_thoughts=internal_thoughts,
            )

            self.current_question = response
            logger.info(f"Turn {self.turn_count} processed successfully")
            return response, False

        except Exception as e:
            logger.exception(f"Error processing response: {e}")
            # Try to save partial interview state
            try:
                self.logger.save()
                logger.info("Partial interview state saved after error")
            except Exception as save_error:
                logger.exception(f"Failed to save partial interview state: {save_error}")
            raise

    def _end_interview(self, final_message: str) -> tuple[str, bool]:
        """End the interview and generate feedback.

        Args:
            final_message: The candidate's final message

        Returns:
            Tuple of (feedback report, True indicating interview ended)
        """
        logger.info(f"Ending interview (total turns: {self.turn_count})")

        try:
            if not self.candidate_info:
                logger.warning("Attempted to end interview with no candidate info")
                return "No interview in progress.", True

            # Log final turn if there's a current question
            if self.current_question:
                self.logger.add_turn(
                    agent_visible_message=self.current_question,
                    user_message=final_message,
                    internal_thoughts="[Observer]: Interview end requested by candidate. [Interviewer]: Generating final feedback.",
                )

            # Generate feedback
            logger.debug("Generating final feedback...")
            history = self.logger.get_conversation_history()
            feedback, formatted_feedback = self.feedback_generator.generate_feedback(
                self.candidate_info,
                history,
            )
            logger.debug("Final feedback generated successfully")

            # Set feedback in logger
            self.logger.set_final_feedback(feedback)

            # Update trace with final results
            if is_tracing_enabled() and feedback:
                try:
                    langfuse = get_client()
                    langfuse.update_current_span(
                        output={
                            "assessed_grade": feedback.assessed_grade.value
                            if hasattr(feedback.assessed_grade, "value")
                            else str(feedback.assessed_grade),
                            "confidence_score": feedback.confidence_score,
                            "total_turns": self.turn_count,
                            "hiring_recommendation": feedback.hiring_recommendation.value
                            if hasattr(feedback.hiring_recommendation, "value")
                            else str(feedback.hiring_recommendation),
                        },
                        metadata={
                            "confirmed_skills": [
                                {"topic": s.topic, "status": s.status, "details": s.details}
                                for s in feedback.confirmed_skills
                            ],
                            "knowledge_gaps": [
                                {"topic": s.topic, "status": s.status, "details": s.details}
                                for s in feedback.knowledge_gaps
                            ],
                        },
                    )
                except Exception as e:
                    logger.warning(f"Could not update Langfuse span: {e}")
                    print(f"⚠️  Could not update Langfuse span: {e}")

            # Save the log
            log_path = self.logger.save()
            logger.info(f"Interview log saved to: {log_path}")

            self.is_active = False
            logger.info("Interview ended successfully")

            return formatted_feedback + f"\n\nInterview log saved to: {log_path}", True

        except Exception as e:
            logger.exception(f"Error ending interview: {e}")
            # Try to save what we can
            try:
                log_path = self.logger.save()
                logger.info(f"Partial interview log saved to: {log_path}")
            except Exception as save_error:
                logger.exception(f"Failed to save interview log: {save_error}")
            raise

    def get_log_path(self) -> Optional[str]:
        """Get the path to the saved log file."""
        if self.logger.session:
            return self.logger.save()
        return None

    def is_interview_active(self) -> bool:
        """Check if an interview is currently active."""
        return self.is_active
