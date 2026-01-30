#!/usr/bin/env python3
"""Test script to verify logging works correctly."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.app_logger import get_logger

# Get logger instance
logger = get_logger(__name__)

def test_basic_logging():
    """Test basic logging functionality."""
    logger.info("Testing basic logging")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

def test_exception_logging():
    """Test exception logging."""
    logger.info("Testing exception logging")

    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.exception(f"Caught exception during test: {e}")
        logger.info("Exception logged successfully")

def test_nested_operations():
    """Test logging in nested operations."""
    logger.info("Starting nested operation test")

    try:
        logger.debug("Step 1: Initialization")
        data = {"key": "value"}

        logger.debug(f"Step 2: Processing data with {len(data)} items")

        # Simulate API call
        logger.debug("Step 3: Simulating API call")

        logger.info("Nested operation completed successfully")

    except Exception as e:
        logger.exception(f"Nested operation failed: {e}")

if __name__ == "__main__":
    print("Testing logging system...")
    print("=" * 60)

    test_basic_logging()
    print("✓ Basic logging test completed")

    test_exception_logging()
    print("✓ Exception logging test completed")

    test_nested_operations()
    print("✓ Nested operations test completed")

    print("=" * 60)
    print("\nLogging test completed!")
    print("\nCheck the following files:")
    print("  - logs/app.log (should contain all messages)")
    print("  - logs/errors.log (should contain only the error message)")
    print("\nYou can view them with:")
    print("  cat logs/app.log")
    print("  cat logs/errors.log")
