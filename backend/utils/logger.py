"""Structured logging with loguru."""

from pathlib import Path

from loguru import logger

from backend.utils.config import get_settings


def setup_logging() -> None:
    """Configure loguru: console + rotating predictions log under logs dir."""
    settings = get_settings()
    log_dir: Path = settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = log_dir / "predictions.log"
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    logger.add(
        predictions_path,
        rotation="10 MB",
        retention=5,
        level="INFO",
        format="{time:ISO} | {level} | {message}",
        enqueue=True,
    )
