"""Production logging setup for scripts and API workers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import sys

from utils.config import settings


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "process=%(process)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = settings.logs_dir / "app.log"
MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_CONFIGURED = False


def configure_logging() -> None:
    """Configure console and rotating file handlers once for the whole process."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)

    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        external_logger = logging.getLogger(logger_name)
        external_logger.handlers.clear()
        external_logger.propagate = True
        external_logger.setLevel(LOG_LEVEL)

    for noisy_logger_name in ("httpx", "matplotlib", "cmdstanpy", "prophet"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = True
    return logger
