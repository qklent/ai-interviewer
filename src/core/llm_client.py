"""LLM client abstraction for multiple providers."""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, TypeVar, Type
from langfuse import get_client
from src.utils.app_logger import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.utils.prompt_loader import PromptMetadata

T = TypeVar("T", bound=BaseModel)


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


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logger.info(f"OpenAIClient initialized with model: {model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> str:
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
                pass  # Silently fail if prompt not found

        # Create a generation with the prompt linked
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-generation",
            prompt=prompt
        ) as generation:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content or ""

            # Update the generation with output
            generation.update(output=content)

            return content

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> dict:
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
                pass  # Silently fail if prompt not found

        # Create a generation with the prompt linked
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-generation",
            prompt=prompt
        ) as generation:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            result = json.loads(content)

            # Update the generation with output
            generation.update(output=result)

            return result

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Type[T],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> T:
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
                pass  # Silently fail if prompt not found

        # Create a generation with the prompt linked
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-generation",
            prompt=prompt
        ) as generation:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            parsed_output = response.choices[0].message.parsed

            # Update the generation with output
            generation.update(output=parsed_output.model_dump() if parsed_output else None)

            return parsed_output


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"
    ):
        import anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment variables")
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        logger.info(f"AnthropicClient initialized with model: {model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> str:
        logger.debug(f"Anthropic generate: model={self.model}, max_tokens={max_tokens}")

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
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                content = response.content[0].text
                logger.debug(f"Anthropic response received (length: {len(content)})")

                # Update the generation with output
                generation.update(output=content)

                return content

        except Exception as e:
            logger.exception(f"Anthropic API call failed: {e}")
            raise

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> dict:
        logger.debug(f"Anthropic generate_json: model={self.model}, max_tokens={max_tokens}")

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
                json_system = (
                    system_prompt
                    + "\n\nIMPORTANT: You must respond with valid JSON only, no other text."
                )
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=json_system,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                content = response.content[0].text

                # Try to extract JSON if there's extra text
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.debug(f"Response not pure JSON, extracting (error: {e})")
                    logger.debug(f"Raw content preview (first 500 chars): {content[:500]}")
                    # Try to find JSON in the response
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start != -1 and end > start:
                        extracted = content[start:end]
                        result = json.loads(extracted)
                        logger.debug(f"JSON extraction successful (stripped {start} leading + {len(content) - end} trailing chars)")
                    else:
                        logger.error(f"Could not extract JSON from response: {content[:200]}")
                        raise

                logger.debug(f"Anthropic JSON response received with {len(result)} keys")

                # Update the generation with output
                generation.update(output=result)

                return result

        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse JSON from Anthropic response: {e}")
            raise
        except Exception as e:
            logger.exception(f"Anthropic API call failed: {e}")
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
        """Generate structured output for Anthropic.

        Since Anthropic doesn't have native structured outputs like OpenAI,
        we use generate_json and parse it into the Pydantic model.
        """
        logger.debug(f"Anthropic generate_structured: model={self.model}, response_format={response_format.__name__}")

        # Get JSON schema from Pydantic model
        schema = response_format.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Add schema to user prompt
        enhanced_user_prompt = f"""{user_prompt}

IMPORTANT: Respond with a JSON object that follows this exact schema:
{schema_str}

Make sure to include ALL required fields in your response."""

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
        except Exception as e:
            logger.exception(f"Failed to parse response into {response_format.__name__}: {e}")
            raise


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
        logger.info(f"OpenRouterClient initialized with model: {self.model}")

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
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
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
                # Add JSON instruction to system prompt
                json_system = (
                    system_prompt
                    + "\n\nIMPORTANT: You must respond with valid JSON only, no other text."
                )

                # Try to use JSON mode if supported by the model
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": json_system},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except Exception as json_mode_error:
                    # Fallback if model doesn't support JSON mode
                    logger.debug(f"JSON mode not supported, falling back: {json_mode_error}")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": json_system},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                content = response.choices[0].message.content or "{}"

                # Try to extract JSON if there's extra text
                try:
                    result = json.loads(content)
                    logger.debug(f"OpenRouter JSON response received with {len(result)} keys")
                except json.JSONDecodeError as e:
                    logger.debug(f"Response not pure JSON, extracting (error: {e})")
                    logger.debug(f"Raw content preview (first 500 chars): {content[:500]}")

                    # Try to find and parse the first complete JSON object
                    start = content.find("{")
                    if start == -1:
                        logger.error(f"No JSON object found in response: {content[:200]}")
                        raise

                    # Use JSONDecoder to parse from the start position
                    # This properly handles nested objects and stops at the right place
                    try:
                        decoder = json.JSONDecoder()
                        result, end_idx = decoder.raw_decode(content, start)
                        logger.debug(f"JSON extraction successful (stripped {start} leading + {len(content) - end_idx} trailing chars)")
                        logger.debug(f"OpenRouter JSON response received with {len(result)} keys")
                    except json.JSONDecodeError as e2:
                        logger.error(f"Could not extract JSON from response: {e2}")
                        logger.error(f"Content preview: {content[:500]}")
                        raise

                # Update the generation with output
                generation.update(output=result)

                return result

        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse JSON from OpenRouter response: {e}")
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

        OpenRouter supports OpenAI-compatible interface, but may not support
        beta.chat.completions.parse. Fall back to generate_json + validation.
        """
        logger.debug(f"OpenRouter generate_structured: model={self.model}, response_format={response_format.__name__}")

        # Get JSON schema from Pydantic model
        schema = response_format.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Add schema to user prompt
        enhanced_user_prompt = f"""{user_prompt}

IMPORTANT: Respond with a JSON object that follows this exact schema:
{schema_str}

Make sure to include ALL required fields in your response."""

        # Use generate_json and then validate with Pydantic
        # This is more reliable for OpenRouter than assuming parse() support
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
        except Exception as e:
            logger.exception(f"Failed to parse response into {response_format.__name__}: {e}")
            raise


def get_llm_client(provider: str = "openai", **kwargs) -> BaseLLMClient:
    """Factory function to get an LLM client."""
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "openrouter": OpenRouterClient,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(providers.keys())}"
        )

    return providers[provider](**kwargs)
