import logging
from termcolor import colored

class ColouredFormatter(logging.Formatter):
    COLOURS = {
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
        'DEBUG': 'blue',
        'INFO': 'green'
    }

    def format(self, record):
        log_message = super().format(record)
        return colored(log_message, self.COLOURS.get(record.levelname))


def configure_logging():
    logging.basicConfig(level=logging.INFO)

    # Set up colorized formatter
    formatter = ColouredFormatter('%(asctime)s - %(levelname)s - %(message)s')

      # Apply the formatter to the root logger

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # Apply the formatter to the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)