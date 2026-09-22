"""Central logging helper.

Uso:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("...")
"""

from __future__ import annotations

import logging
import os
import sys

_configured = False


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root handler once (idempotent). Respects LOG_LEVEL env."""
    global _configured
    if _configured:
        return
    env_level = os.environ.get("LOG_LEVEL", level)
    logging.basicConfig(
        level=env_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring base configuration exists."""
    configure_logging()
    return logging.getLogger(name)
