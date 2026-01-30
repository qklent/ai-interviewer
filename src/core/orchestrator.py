"""Interview Orchestrator - coordinates agents and manages interview flow."""

import os
import re
import uuid
from datetime import datetime
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
        llm_provider: str = "openrouter",
        llm_model: Optional[str] = None,
        output_dir: str = "logs",
    ):
        """Initialize the orchestrator with agents.

        Args:
            llm_provider: LLM provider to use (currently only "openrouter" is supported)
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

            # Initialize feedback generator based on mode
            feedback_mode = os.getenv("FEEDBACK_MODE", "single_model")
            logger.info(f"Feedback mode: {feedback_mode}")

            if feedback_mode == "multi_model":
                from src.agents.multi_model_feedback_generator import MultiModelFeedbackGenerator

                # Parse evaluator models from comma-separated list
                evaluator_models_str = os.getenv(
                    "FEEDBACK_EVALUATOR_MODELS",
                    "google/gemini-2.0-flash-thinking-exp-1219,anthropic/claude-3.5-sonnet"
                )
                evaluator_models = [m.strip() for m in evaluator_models_str.split(",")]

                # Get aggregator model
                aggregator_model = os.getenv("FEEDBACK_AGGREGATOR_MODEL", "openai/gpt-4o")

                self.feedback_generator = MultiModelFeedbackGenerator(
                    evaluator_models=evaluator_models,
                    aggregator_model=aggregator_model,
                    save_intermediate=os.getenv("SAVE_INTERMEDIATE_FEEDBACK", "true").lower() == "true",
                )
                logger.info("Initialized MultiModelFeedbackGenerator")
            else:
                self.feedback_generator = FeedbackGeneratorAgent(self.llm)
                logger.info("Initialized single-model FeedbackGeneratorAgent")

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
            self.session_id: Optional[str] = None  # Langfuse session ID for grouping traces

        except Exception as e:
            logger.exception(f"Failed to initialize orchestrator: {e}")
            raise

    @observe(name="Orchestrator: Interview Session")
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
            # Generate unique session ID for this interview
            # Format: interview_{timestamp}_{uuid}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"interview_{timestamp}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Created session ID: {self.session_id}")

            # Track session metadata in Langfuse
            if is_tracing_enabled():
                try:
                    langfuse = get_client()
                    # Update trace-level attributes with session_id
                    langfuse.update_current_trace(
                        session_id=self.session_id,  # Group all traces by session
                        user_id=name,
                        tags=["interview", grade, position],
                        metadata={
                            "candidate_name": name,
                            "position": position,
                            "target_grade": grade,
                            "experience": experience,
                            "interview_start": datetime.now().isoformat(),
                        },
                    )
                    # Update span-level attributes
                    langfuse.update_current_span(
                        name=f"Interview Start: {name}",
                        metadata={
                            "position": position,
                            "target_grade": grade,
                            "experience": experience,
                            "mode": "interactive",
                        },
                    )
                    logger.info(f"Langfuse session created: {self.session_id}")
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

    @observe(name="Orchestrator: Process Response")
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

            # Tag this trace with the session_id
            if is_tracing_enabled() and self.session_id:
                try:
                    langfuse = get_client()
                    langfuse.update_current_trace(
                        session_id=self.session_id,
                        metadata={"turn": self.turn_count + 1},
                    )
                except Exception as e:
                    logger.debug(f"Could not update trace session_id: {e}")

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
            result = self.feedback_generator.generate_feedback(
                self.candidate_info,
                history,
            )

            # Handle both single-model (2-tuple) and multi-model (3-tuple) returns
            if isinstance(result, tuple) and len(result) == 3:
                feedback, formatted_feedback, intermediate = result
                if intermediate:
                    logger.debug("Storing intermediate feedbacks from multi-model generation")
                    self.logger.set_intermediate_feedbacks(intermediate)
            else:
                feedback, formatted_feedback = result
                intermediate = None

            logger.debug("Final feedback generated successfully")

            # Set feedback in logger
            self.logger.set_final_feedback(feedback)

            # Update trace with final results
            if is_tracing_enabled() and feedback:
                try:
                    langfuse = get_client()
                    # Tag with session_id and add final outcome
                    langfuse.update_current_trace(
                        session_id=self.session_id,
                        metadata={
                            "interview_end": datetime.now().isoformat(),
                            "total_turns": self.turn_count,
                            "outcome": "completed",
                        },
                    )
                    langfuse.update_current_span(
                        name=f"Interview End: {self.candidate_info.name}",
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
                    logger.info(f"Langfuse session finalized: {self.session_id}")
                except Exception as e:
                    logger.warning(f"Could not update Langfuse span: {e}")
                    print(f"⚠️  Could not update Langfuse span: {e}")

            # Save the log
            log_path = self.logger.save()
            logger.info(f"Interview log saved to: {log_path}")

            # Flush Langfuse to ensure all traces are sent
            if is_tracing_enabled():
                try:
                    langfuse = get_client()
                    langfuse.flush()
                    logger.debug("Langfuse traces flushed successfully")
                except Exception as e:
                    logger.debug(f"Could not flush Langfuse: {e}")

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
