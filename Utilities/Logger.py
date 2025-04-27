import logging
import sys
import os
import re
from colorama import Fore, Style, init
import multiprocessing
from queue import Empty

# Initialize colorama (auto-reset for Windows support)
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """ Custom formatter for colored logging with path and number highlighting. """
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    PATH_COLOR = Fore.BLUE      # Color for file/folder paths
    NUMBER_COLOR = Fore.WHITE  # Color for numbers

    PATH_REGEX = re.compile(r'\b(?:[A-Za-z]:\\|/)?(?:[\w.-]+[\\/])*[\w.-]+\.\w+\b')
    NUMBER_REGEX = re.compile(r'\b\d+(\.\d+)?\b')  # Matches integers and decimals

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)  # Default to white
        log_message = super().format(record)

        # Highlight file/folder paths
        log_message = self.PATH_REGEX.sub(lambda m: f"{self.PATH_COLOR}{m.group(0)}{log_color}", log_message)

        # Highlight numbers
        log_message = self.NUMBER_REGEX.sub(lambda m: f"{self.NUMBER_COLOR}{m.group(0)}{log_color}", log_message)

        return f"{log_color}{log_message}{Style.RESET_ALL}"


class WorkerIDFilter(logging.Filter):
    """Custom filter to add shortened worker ID to log records."""
    def filter(self, record):
        # Extract just the worker number from the 'ForkPoolWorker-x' format
        worker_id = re.sub(r'ForkPoolWorker-(\d+)', r'Worker \1', multiprocessing.current_process().name)
        record.worker_id = worker_id  # Add simplified worker name (e.g., Worker 1, Worker 2)
        return record


def logger(name, log_file, level=logging.DEBUG):
    """
    Sets up a logger to log messages to a file and console with colored output.
    :param name: Name of the logger.
    :param log_file: The file to store logs.
    :param level: Logging level (default: DEBUG).
    :return: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate log entries if multiple instances are created
    if logger.hasHandlers():
        return logger

    log_directory = os.path.dirname(log_file)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        logger.debug(f"Directory '{log_directory}' created.")

    # File handler (plain text logs)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(worker_id)s - %(levelname)s - %(message)s'))

    # Console handler (colored logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(worker_id)s - %(levelname)s - %(message)s'))

    # Add the filter to include worker_id in all log messages
    logger.addFilter(WorkerIDFilter())

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Custom log method with skip_lines option
    def custom_info(message, *args, skip_lines=False, **kwargs):
        if skip_lines:
            print("\n" * 4, end="")  # Print two blank lines before logging
        logger._log(logging.INFO, message, args, **kwargs)

    # Replace the default info method with custom_info
    logger.info = custom_info

    return logger