"""Structlog configuration.

Call ``configure_logging()`` once at process start (main.py / CLI entry point).
Every module then just does ``log = structlog.get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Wire structlog to emit structured JSON to stdout.

    Args:
        level: Standard logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    # add_logger_name requires a stdlib logger (.name attribute); omit it here
    # because we use PrintLoggerFactory.  Module context is visible from the
    # structlog.get_logger(__name__) call in each module.
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
