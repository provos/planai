import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional


def setup_logging(
    logs_dir: str = "logs",
    logs_prefix="general_log",
    logger_name: Optional[str] = None,
    level: Optional[int] = logging.INFO,
) -> logging.Logger:
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{logs_prefix}_{timestamp}.log")

    logger = logging.getLogger(name=logger_name)
    logger.handlers = []
    logger.setLevel(level)

    # Create file handler but don't create the file until the first emit
    file_handler = logging.FileHandler(log_file, mode="w", delay=True)
    file_handler.setLevel(level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


@contextmanager
def measure_time():
    start_time = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        end_time = time.perf_counter()
        result["elapsed_time"] = (end_time - start_time) * 1000
