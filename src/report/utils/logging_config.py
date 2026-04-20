"""Logging configuration using Rich for beautiful console output."""

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with Rich handler."""
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        ],
    )
    # Suppress noisy third-party loggers
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
