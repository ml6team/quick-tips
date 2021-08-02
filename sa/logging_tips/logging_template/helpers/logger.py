"""
This file contains the logging setup.
"""

import logging
import sys

APP_LOGGER_NAME = 'MyLoggerApp'

def setup_applevel_logger(logger_name=APP_LOGGER_NAME, file_name=None):
    """..."""
    logger = logging.getLogger(logger_name)
    syslog = logging.StreamHandler()

    format = '%(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format)
    syslog.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(sh)

    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(module_name):
    """..."""
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)
