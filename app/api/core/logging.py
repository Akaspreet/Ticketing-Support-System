"""
Logging configuration for the application.
"""
import sys
import logging
from pathlib import Path
from loguru import logger                       # pylint: disable=import-error

from app.api.core.config import settings


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging and redirects to loguru.
    """
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging():
    """Configure logging with loguru."""
    # Remove default handlers
    logger.remove()

    # Add console output handler with appropriate format for the environment
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Determine log level from environment
    log_level = "DEBUG" if settings.DEBUG else "INFO"

    # Add stderr handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        enqueue=True
    )

    # Create logs directory if not exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Add file handler for non-development environments
    if settings.ENVIRONMENT != "development":
        logger.add(
            "logs/app.log",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            format=log_format,
            level=log_level,
            enqueue=True
        )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # List of loggers to configure
    loggers = [
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.access"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("fastapi"),
        logging.getLogger("elasticsearch"),
        logging.getLogger("pymongo"),
        logging.getLogger("langchain")
    ]

    # Apply configuration to all loggers
    for log in loggers:
        log.handlers = [InterceptHandler()]
        log.propagate = False


# Export logger instance
app_logger = logger
