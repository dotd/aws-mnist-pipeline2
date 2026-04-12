import logging
import sys
import time
from pathlib import Path


class _StderrToLogger:
    """Redirect stderr writes to both original stderr and a log file."""

    def __init__(self, log_file_handler, original_stderr):
        self.log_file_handler = log_file_handler
        self.original_stderr = original_stderr

    def write(self, msg):
        self.original_stderr.write(msg)
        if msg.strip():
            self.log_file_handler.stream.write(msg)
            self.log_file_handler.stream.flush()

    def flush(self):
        self.original_stderr.flush()
        self.log_file_handler.stream.flush()


def get_logger(name=None, log_dir="./logs"):
    """Set up logging to stdout and to ./logs/log_YYYYMMDD_HHMMSS.log.
    Also captures stderr (e.g. wandb output) to the same log file."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"log_{time.strftime('%Y%m%d_%H%M%S')}.log"

    file_handler = logging.FileHandler(log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            file_handler,
        ],
    )

    # Capture stderr (wandb prints its summary there) to the log file
    sys.stderr = _StderrToLogger(file_handler, sys.stderr)

    logger = logging.getLogger(name)
    logger.log_file = log_file
    return logger
