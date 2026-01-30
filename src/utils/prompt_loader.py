"""Utility for loading prompts from Langfuse or local files."""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from src.utils.tracing import get_langfuse_client, is_tracing_enabled


@dataclass
class PromptMetadata:
    """Metadata about a prompt for linking to traces."""
    name: str
    version: Optional[int] = None
    config: Optional[dict] = None


class PromptLoader:
    """Loads and caches prompt templates from Langfuse or fallback to local files."""

    def __init__(self):
        # Find the prompts directory relative to the project root (for fallback)
        self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        self._cache: Dict[str, Tuple[str, Optional[PromptMetadata]]] = {}
        self._langfuse_available = is_tracing_enabled()

    def load(self, agent_type: str, prompt_name: str) -> Tuple[str, Optional[PromptMetadata]]:
        """Load a prompt template from Langfuse or file.

        Priority:
        1. Check cache
        2. Try fetching from Langfuse (if enabled)
        3. Fall back to local file

        Args:
            agent_type: The agent type (e.g., 'interviewer', 'observer', 'feedback_generator')
            prompt_name: The prompt name (e.g., 'system', 'greeting')

        Returns:
            Tuple of (prompt content, metadata). Metadata is None for local files.

        Raises:
            FileNotFoundError: If the prompt is not found in Langfuse or local files
        """
        cache_key = f"{agent_type}/{prompt_name}"

        # Return from cache if available
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try loading from Langfuse first
        content, metadata = self._load_from_langfuse(agent_type, prompt_name)

        # Fall back to local file if Langfuse fails
        if content is None:
            content = self._load_from_file(agent_type, prompt_name)
            metadata = None

        # Cache and return
        self._cache[cache_key] = (content, metadata)
        return content, metadata

    def _load_from_langfuse(self, agent_type: str, prompt_name: str) -> Tuple[Optional[str], Optional[PromptMetadata]]:
        """Try to load prompt from Langfuse.

        Args:
            agent_type: The agent type
            prompt_name: The prompt name

        Returns:
            Tuple of (prompt content, metadata) if found in Langfuse, (None, None) otherwise
        """
        if not self._langfuse_available:
            return None, None

        try:
            langfuse = get_langfuse_client()
            if langfuse is None:
                return None, None

            # Langfuse prompt naming: {agent_type}_{prompt_name}
            # e.g., "interviewer_system", "observer_analysis"
            langfuse_prompt_name = f"{agent_type}_{prompt_name}"

            prompt = langfuse.get_prompt(langfuse_prompt_name)

            # Create metadata for linking to traces
            metadata = PromptMetadata(
                name=langfuse_prompt_name,
                version=prompt.version if hasattr(prompt, 'version') else None,
                config=prompt.config if hasattr(prompt, 'config') else None
            )

            # Extract prompt text from the Langfuse prompt object
            # Langfuse prompts can have different formats
            content = None
            if hasattr(prompt, 'prompt'):
                content = prompt.prompt
            elif hasattr(prompt, 'compile'):
                # If it's a compiled prompt, call compile() without variables
                content = prompt.compile()
            else:
                print(f"⚠️  Langfuse prompt '{langfuse_prompt_name}' has unexpected format")
                return None, None

            return content, metadata

        except Exception as e:
            # Silently fail and fall back to file - this is expected when prompt doesn't exist
            return None, None

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


def load_prompt(agent_type: str, prompt_name: str) -> Tuple[str, Optional[PromptMetadata]]:
    """Convenience function to load a prompt using the global loader.

    Args:
        agent_type: The agent type (e.g., 'interviewer', 'observer', 'feedback_generator')
        prompt_name: The prompt filename without extension (e.g., 'system', 'greeting')

    Returns:
        Tuple of (prompt content, metadata). Metadata is None for local files.
    """
    return _loader.load(agent_type, prompt_name)
