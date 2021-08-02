"""
Custom module
"""

import logger

log = logger.get_logger(__name__)


def multiply(num1, num2):  # multiply two numbers
    """..."""
    log.debug('Executing multiply.')
    return num1 * num2
