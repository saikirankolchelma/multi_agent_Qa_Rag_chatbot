import logging
from config import config

def setup_logging():
    logging.basicConfig(
        level=config.LOG_LEVEL.upper(),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

# Get a logger instance for global use if needed, after setup
logger = logging.getLogger(__name__)