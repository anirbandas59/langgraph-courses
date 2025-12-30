"""Logging configuration and utilities for the Adaptive RAG workflow."""

import logging
import os
import warnings

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", suppress_warnings: bool = True) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        suppress_warnings: If True, suppress common library warnings
    """
    if suppress_warnings:
        # Suppress common warnings from dependencies
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", module="langchain_openai")
        warnings.filterwarnings("ignore", module="langchain_tavily")

        # Set USER_AGENT to avoid warnings
        if "USER_AGENT" not in os.environ:
            os.environ["USER_AGENT"] = "AdaptiveRAG/1.0"

    # Configure root logger with RichHandler
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
