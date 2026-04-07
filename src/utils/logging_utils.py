import logging
import time
from pathlib import Path


def get_logger(name=None, log_dir="./logs"):
    """Set up logging to stdout and to ./logs/log_YYYYMMDD_HHMMSS.log."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"log_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger(name)
    logger.log_file = log_file
    return logger
