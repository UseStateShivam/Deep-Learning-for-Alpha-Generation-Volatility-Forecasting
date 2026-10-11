"""
Logging Configuration
Provides structured, standardized logging across all modules in the alpha_volatility package.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "alpha_volatility", level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a logger instance with consistent formatting.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional file path to write log output.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
