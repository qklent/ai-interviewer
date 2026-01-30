"""LLM client abstraction for multiple providers."""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from langfuse import observe
from langfuse.decorators import langfuse_context

if TYPE_CHECKING:
    from src.utils.prompt_loader import PromptMetadata


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @observe(as_type="generation")
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

    @observe(as_type="generation")
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


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> str:
        # Link prompt to current trace if metadata is available
        if prompt_metadata:
            try:
                langfuse_context.update_current_observation(
                    prompt={"name": prompt_metadata.name, "version": prompt_metadata.version}
                )
            except Exception:
                pass  # Silently fail if tracing not available

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> dict:
        # Link prompt to current trace if metadata is available
        if prompt_metadata:
            try:
                langfuse_context.update_current_observation(
                    prompt={"name": prompt_metadata.name, "version": prompt_metadata.version}
                )
            except Exception:
                pass  # Silently fail if tracing not available

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
        return json.loads(content)


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"
    ):
        import anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> str:
        # Link prompt to current trace if metadata is available
        if prompt_metadata:
            try:
                langfuse_context.update_current_observation(
                    prompt={"name": prompt_metadata.name, "version": prompt_metadata.version}
                )
            except Exception:
                pass  # Silently fail if tracing not available

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> dict:
        # Link prompt to current trace if metadata is available
        if prompt_metadata:
            try:
                langfuse_context.update_current_observation(
                    prompt={"name": prompt_metadata.name, "version": prompt_metadata.version}
                )
            except Exception:
                pass  # Silently fail if tracing not available

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
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
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
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.model = model or os.getenv(
            "OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"
        )

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> str:
        # Link prompt to current trace if metadata is available
        if prompt_metadata:
            try:
                langfuse_context.update_current_observation(
                    prompt={"name": prompt_metadata.name, "version": prompt_metadata.version}
                )
            except Exception:
                pass  # Silently fail if tracing not available

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        prompt_metadata: Optional["PromptMetadata"] = None,
    ) -> dict:
        # Link prompt to current trace if metadata is available
        if prompt_metadata:
            try:
                langfuse_context.update_current_observation(
                    prompt={"name": prompt_metadata.name, "version": prompt_metadata.version}
                )
            except Exception:
                pass  # Silently fail if tracing not available

        # Add JSON instruction to system prompt
        json_system = (
            system_prompt
            + "\n\nIMPORTANT: You must respond with valid JSON only, no other text."
        )

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
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
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
