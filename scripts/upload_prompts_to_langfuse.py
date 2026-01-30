#!/usr/bin/env python3
"""
Upload local prompts to Langfuse prompt management.

This script reads all prompt files from the prompts/ directory and uploads them
to Langfuse with the naming convention: {agent_type}_{prompt_name}

Example:
  prompts/interviewer/system.txt -> "interviewer_system" in Langfuse
  prompts/observer/analysis.txt -> "observer_analysis" in Langfuse

Requirements:
  - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST must be set in .env
  - All prompts in prompts/ directory will be uploaded
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()


def upload_prompts():
    """Upload all prompts from local files to Langfuse."""

    # Check for credentials
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("ERROR: Langfuse credentials not found in environment variables.")
        print("Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file")
        sys.exit(1)

    # Initialize Langfuse
    try:
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        print(f"✓ Connected to Langfuse at {host}")
    except Exception as e:
        print(f"ERROR: Failed to connect to Langfuse: {e}")
        sys.exit(1)

    # Find all prompt files
    prompts_dir = Path(__file__).parent.parent / "prompts"

    if not prompts_dir.exists():
        print(f"ERROR: Prompts directory not found: {prompts_dir}")
        sys.exit(1)

    prompt_files = list(prompts_dir.glob("**/*.txt"))

    if not prompt_files:
        print(f"ERROR: No .txt files found in {prompts_dir}")
        sys.exit(1)

    print(f"\nFound {len(prompt_files)} prompt files to upload:")
    print("-" * 60)

    uploaded = 0
    failed = 0

    for prompt_file in prompt_files:
        # Extract agent_type and prompt_name from path
        # e.g., prompts/interviewer/system.txt -> agent_type="interviewer", prompt_name="system"
        relative_path = prompt_file.relative_to(prompts_dir)
        parts = relative_path.parts

        if len(parts) != 2:
            print(f"⚠ Skipping {prompt_file} - unexpected directory structure")
            continue

        agent_type = parts[0]
        prompt_name = parts[1].replace(".txt", "")

        # Langfuse prompt name: {agent_type}_{prompt_name}
        langfuse_name = f"{agent_type}_{prompt_name}"

        # Read prompt content
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"✗ Failed to read {prompt_file}: {e}")
            failed += 1
            continue

        # Upload to Langfuse
        try:
            langfuse.create_prompt(
                name=langfuse_name,
                prompt=content,
                type="text",
                labels=[agent_type, "ai-interviewer"],
                tags=[agent_type, prompt_name]
            )
            print(f"✓ Uploaded: {langfuse_name} ({len(content)} chars)")
            uploaded += 1
        except Exception as e:
            print(f"✗ Failed to upload {langfuse_name}: {e}")
            failed += 1

    print("-" * 60)
    print(f"\nResults:")
    print(f"  Uploaded: {uploaded}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(prompt_files)}")

    if uploaded > 0:
        print(f"\n✓ Successfully uploaded {uploaded} prompts to Langfuse!")
        print(f"\nYou can view and manage them at: {host}")
        print("\nPrompt naming convention:")
        print("  Local: prompts/{agent_type}/{prompt_name}.txt")
        print("  Langfuse: {agent_type}_{prompt_name}")

    if failed > 0:
        print(f"\n⚠ Warning: {failed} prompts failed to upload. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("  UPLOAD PROMPTS TO LANGFUSE")
    print("=" * 60)
    print()
    upload_prompts()
