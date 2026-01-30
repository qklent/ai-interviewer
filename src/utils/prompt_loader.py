"""Utility for loading prompts from files."""
import os
from pathlib import Path
from typing import Dict


class PromptLoader:
    """Loads and caches prompt templates from the prompts directory."""

    def __init__(self):
        # Find the prompts directory relative to the project root
        self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        self._cache: Dict[str, str] = {}

    def load(self, agent_type: str, prompt_name: str) -> str:
        """Load a prompt template from a file.

        Args:
            agent_type: The agent type (e.g., 'interviewer', 'observer', 'feedback_generator')
            prompt_name: The prompt filename without extension (e.g., 'system', 'greeting')

        Returns:
            The prompt content as a string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        cache_key = f"{agent_type}/{prompt_name}"

        # Return from cache if available
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load from file
        prompt_path = self.prompts_dir / agent_type / f"{prompt_name}.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}\n"
                f"Expected: prompts/{agent_type}/{prompt_name}.txt"
            )

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Cache and return
        self._cache[cache_key] = content
        return content

    def clear_cache(self) -> None:
        """Clear the prompt cache. Useful for development/testing."""
        self._cache.clear()


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
