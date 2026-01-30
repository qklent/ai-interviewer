"""Application-wide logger for debugging and error tracking."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_app_logger(
    log_dir: str = "logs",
    log_level: int = logging.DEBUG,
) -> logging.Logger:
    """Set up application-wide logging with file and console handlers.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: DEBUG)

    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("ai_interviewer")
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(levelname)s: %(message)s",
    )

    # File handler for all logs (rotating, max 10MB, keep 5 backups)
    all_logs_file = log_path / "app.log"
    all_handler = RotatingFileHandler(
        all_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(detailed_formatter)

    # File handler for errors only (rotating, max 10MB, keep 5 backups)
    error_logs_file = log_path / "errors.log"
    error_handler = RotatingFileHandler(
        error_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Console handler for INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(all_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    # Log startup
    logger.info("=" * 60)
    logger.info(f"Application logger initialized at {datetime.now().isoformat()}")
    logger.info(f"Log directory: {log_path.absolute()}")
    logger.info("=" * 60)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"ai_interviewer.{name}")
    return logging.getLogger("ai_interviewer")


# Initialize the main logger on module import
_main_logger = setup_app_logger()
