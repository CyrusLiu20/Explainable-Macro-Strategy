import logging
import sys
import os
from colorama import Fore, Style, init

# Initialize colorama (auto-reset for Windows support)
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """ Custom formatter for colored logging """
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)  # Default to white
        log_message = super().format(record)
        return f"{log_color}{log_message}{Style.RESET_ALL}"

def scraper_logger(name, log_file, level=logging.DEBUG):
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
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler (colored logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
