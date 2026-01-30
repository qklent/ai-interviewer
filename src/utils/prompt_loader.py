"""Utility for loading prompts from Langfuse or local files."""
import os
from pathlib import Path
from typing import Dict, Optional
from src.utils.tracing import get_langfuse_client, is_tracing_enabled


class PromptLoader:
    """Loads and caches prompt templates from Langfuse or fallback to local files."""

    def __init__(self):
        # Find the prompts directory relative to the project root (for fallback)
        self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        self._cache: Dict[str, str] = {}
        self._langfuse_available = is_tracing_enabled()

    def load(self, agent_type: str, prompt_name: str) -> str:
        """Load a prompt template from Langfuse or file.

        Priority:
        1. Check cache
        2. Try fetching from Langfuse (if enabled)
        3. Fall back to local file

        Args:
            agent_type: The agent type (e.g., 'interviewer', 'observer', 'feedback_generator')
            prompt_name: The prompt name (e.g., 'system', 'greeting')

        Returns:
            The prompt content as a string

        Raises:
            FileNotFoundError: If the prompt is not found in Langfuse or local files
        """
        cache_key = f"{agent_type}/{prompt_name}"

        # Return from cache if available
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try loading from Langfuse first
        content = self._load_from_langfuse(agent_type, prompt_name)

        # Fall back to local file if Langfuse fails
        if content is None:
            content = self._load_from_file(agent_type, prompt_name)

        # Cache and return
        self._cache[cache_key] = content
        return content

    def _load_from_langfuse(self, agent_type: str, prompt_name: str) -> Optional[str]:
        """Try to load prompt from Langfuse.

        Args:
            agent_type: The agent type
            prompt_name: The prompt name

        Returns:
            Prompt content if found in Langfuse, None otherwise
        """
        if not self._langfuse_available:
            return None

        try:
            langfuse = get_langfuse_client()
            if langfuse is None:
                return None

            # Langfuse prompt naming: {agent_type}_{prompt_name}
            # e.g., "interviewer_system", "observer_analysis"
            langfuse_prompt_name = f"{agent_type}_{prompt_name}"

            prompt = langfuse.get_prompt(langfuse_prompt_name)

            # Extract prompt text from the Langfuse prompt object
            # Langfuse prompts can have different formats
            if hasattr(prompt, 'prompt'):
                return prompt.prompt
            elif hasattr(prompt, 'compile'):
                # If it's a compiled prompt, call compile() without variables
                return prompt.compile()
            else:
                print(f"⚠️  Langfuse prompt '{langfuse_prompt_name}' has unexpected format")
                return None

        except Exception as e:
            # Silently fail and fall back to file - this is expected when prompt doesn't exist
            return None

    def _load_from_file(self, agent_type: str, prompt_name: str) -> str:
        """Load prompt from local file.

        Args:
            agent_type: The agent type
            prompt_name: The prompt name

        Returns:
            Prompt content as string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        prompt_path = self.prompts_dir / agent_type / f"{prompt_name}.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt not found in Langfuse or local files.\n"
                f"Langfuse name: {agent_type}_{prompt_name}\n"
                f"File path: prompts/{agent_type}/{prompt_name}.txt"
            )

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    def clear_cache(self) -> None:
        """Clear the prompt cache. Useful for development/testing."""
        self._cache.clear()

    def refresh_langfuse_status(self) -> None:
        """Refresh the Langfuse availability status. Call after initializing Langfuse."""
        self._langfuse_available = is_tracing_enabled()
        self.clear_cache()  # Clear cache to reload prompts from Langfuse


# Global instance for convenience
_loader = PromptLoader()


def load_prompt(agent_type: str, prompt_name: str) -> str:
    """Convenience function to load a prompt using the global loader.

    Args:
        agent_type: The agent type (e.g., 'interviewer', 'observer', 'feedback_generator')
        prompt_name: The prompt filename without extension (e.g., 'system', 'greeting')

    Returns:
        The prompt content as a string
    """
    return _loader.load(agent_type, prompt_name)
