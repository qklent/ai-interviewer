"""LLM client abstraction for multiple providers."""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, TypeVar, Type
from langfuse import get_client
from src.utils.app_logger import get_logger
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.utils.prompt_loader import PromptMetadata

T = TypeVar("T", bound=BaseModel)


def _get_retry_decorator():
    """Create a retry decorator with configurable max attempts from environment."""
    max_retries = int(os.getenv("MAX_LLM_RETRIES", "3"))

    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(initial=1, max=60),
        retry=(
            retry_if_exception_type(APIConnectionError) |
            retry_if_exception_type(RateLimitError) |
            retry_if_exception_type(APITimeoutError) |
            retry_if_exception_type(json.JSONDecodeError) |
            retry_if_exception_type(APIError)
        ),
        before_sleep=before_sleep_log(logger, logger.level, exc_info=True),
        reraise=True,
    )


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_metadata: Optional metadata for linking to Langfuse prompts
        """
        pass

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> dict:
        """Generate a JSON response from the LLM.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_metadata: Optional metadata for linking to Langfuse prompts
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Type[T],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> T:
        """Generate a structured response from the LLM using Pydantic models.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            response_format: Pydantic model class for structured output
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_metadata: Optional metadata for linking to Langfuse prompts

        Returns:
            Parsed Pydantic model instance
        """
        pass


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client (OpenAI-compatible interface)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.model = model or os.getenv(
            "OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"
        )

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Create retry decorator with configured max retries
        self._retry = _get_retry_decorator()

        logger.info(f"OpenRouterClient initialized with model: {self.model}")

    def _create_chat_completion(self, messages, temperature, max_tokens, response_format=None):
        """Internal method to create chat completion with retry logic."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        return self._retry(lambda: self.client.chat.completions.create(**kwargs))()

    def _parse_chat_completion(self, messages, temperature, max_tokens, response_format):
        """Internal method to parse chat completion with retry logic."""
        return self._retry(lambda: self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        ))()

    def _parse_json_with_retry(self, content: str) -> dict:
        """Parse JSON with retry logic."""
        return self._retry(lambda: json.loads(content))()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> str:
        logger.debug(f"OpenRouter generate: model={self.model}, temp={temperature}, max_tokens={max_tokens}")

        try:
            langfuse = get_client()

            # Fetch the prompt from Langfuse if metadata is available
            prompt = None
            if prompt_metadata:
                try:
                    prompt = langfuse.get_prompt(
                        prompt_metadata.name,
                        version=prompt_metadata.version
                    )
                except Exception as e:
                    logger.debug(f"Could not fetch prompt from Langfuse: {e}")

            # Create a generation with the prompt linked
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="llm-generation",
                prompt=prompt
            ) as generation:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                response = self._create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = response.choices[0].message.content or ""
                logger.debug(f"OpenRouter response received (length: {len(content)})")

                # Update the generation with output
                generation.update(output=content)

                return content

        except Exception as e:
            logger.exception(f"OpenRouter API call failed: {e}")
            raise

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> dict:
        logger.debug(f"OpenRouter generate_json: model={self.model}, temp={temperature}, max_tokens={max_tokens}")

        try:
            langfuse = get_client()

            # Fetch the prompt from Langfuse if metadata is available
            prompt = None
            if prompt_metadata:
                try:
                    prompt = langfuse.get_prompt(
                        prompt_metadata.name,
                        version=prompt_metadata.version
                    )
                except Exception as e:
                    logger.debug(f"Could not fetch prompt from Langfuse: {e}")

            # Create a generation with the prompt linked
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="llm-generation",
                prompt=prompt
            ) as generation:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                response = self._create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )

                # Check response validity
                if not response.choices or len(response.choices) == 0:
                    logger.error(f"No choices in response from OpenRouter (model={self.model})")
                    logger.error(f"Full response: {response}")
                    raise ValueError("Empty response from OpenRouter API")

                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Log finish reason if not 'stop'
                if finish_reason != "stop":
                    logger.warning(
                        f"Unusual finish_reason: {finish_reason} (model={self.model}). "
                        f"This may indicate truncation or filtering."
                    )

                content = message.content or "{}"

                # Log if content is empty or suspicious
                if not content or not content.strip():
                    logger.error(f"Empty content received from OpenRouter (model={self.model})")
                    logger.error(f"Finish reason: {finish_reason}")
                    logger.error(f"Message object: {message}")
                    logger.error(f"Full response: {response}")
                    content = "{}"  # Force to valid empty JSON

                result = self._parse_json_with_retry(content)

                logger.debug(f"OpenRouter JSON response received with {len(result)} keys")

                # Update the generation with output
                generation.update(output=result)

                return result

        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse JSON from OpenRouter response: {e}")
            logger.error(f"Content that failed to parse (first 500 chars): {content[:500] if 'content' in locals() else 'N/A'}")
            logger.error(f"Content length: {len(content) if 'content' in locals() else 'N/A'}")
            logger.error(f"Content repr: {repr(content) if 'content' in locals() else 'N/A'}")
            raise
        except Exception as e:
            logger.exception(f"OpenRouter API call failed: {e}")
            raise

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Type[T],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> T:
        """Generate structured output for OpenRouter.

        Tries to use OpenAI's parse() method first, falls back to json_object mode.
        """
        logger.debug(f"OpenRouter generate_structured: model={self.model}, response_format={response_format.__name__}")

        langfuse = get_client()

        # Fetch the prompt from Langfuse if metadata is available
        prompt = None
        if prompt_metadata:
            try:
                prompt = langfuse.get_prompt(
                    prompt_metadata.name,
                    version=prompt_metadata.version
                )
            except Exception:
                pass

        # Try using the beta parse API (like OpenAI)
        try:
            logger.debug(f"OpenRouter attempting native parse() API with response_format={response_format.__name__}")

            with langfuse.start_as_current_observation(
                as_type="generation",
                name="llm-generation",
                prompt=prompt
            ) as generation:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                response = self._parse_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

                parsed_output = response.choices[0].message.parsed

                # Update the generation with output
                generation.update(output=parsed_output.model_dump() if parsed_output else None)

                logger.debug(f"Successfully used native parse() API for {response_format.__name__}")
                return parsed_output

        except Exception as e:
            # Parse API not supported, fall back to json_object mode with schema in prompt
            logger.debug(f"Parse API not supported ({e}), falling back to JSON mode (response_format={{\"type\": \"json_object\"}}) for {response_format.__name__}")

            # Get JSON schema from Pydantic model
            schema = response_format.model_json_schema()
            schema_str = json.dumps(schema, indent=2)

            # Add schema to user prompt
            enhanced_user_prompt = f"""{user_prompt}

IMPORTANT: Respond with ONLY a JSON data instance that matches this schema structure.
Do NOT include the schema definition itself in your response.

Schema for reference:
{schema_str}

Your response must be a valid JSON object containing the actual data values for the required fields listed in the schema.
Example: If the schema requires a "response" field of type string, return {{"response": "your actual text here"}}, NOT the schema definition."""

            # Use generate_json and then validate with Pydantic
            json_result = self.generate_json(
                system_prompt=system_prompt,
                user_prompt=enhanced_user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_metadata=prompt_metadata,
            )

            # Parse JSON into Pydantic model
            try:
                parsed_output = response_format.model_validate(json_result)
                logger.debug(f"Successfully parsed response into {response_format.__name__}")
                return parsed_output
            except Exception as validation_error:
                logger.exception(f"Failed to parse response into {response_format.__name__}: {validation_error}")
                raise


def get_llm_client(provider: str = "openrouter", **kwargs) -> BaseLLMClient:
    """Factory function to get an LLM client.

    Args:
        provider: LLM provider to use (currently only "openrouter" is supported)
        **kwargs: Additional arguments to pass to the client constructor

    Returns:
        Initialized LLM client instance
    """
    if provider != "openrouter":
        raise ValueError(
            f"Unknown provider: {provider}. Only 'openrouter' is supported."
        )

    return OpenRouterClient(**kwargs)
