"""
File containing the entrypoint of the script.
"""

from helpers import logger
log = logger.setup_applevel_logger(file_name = 'app_debug.log')

from helpers import module


def run():
    """..."""
    log.debug('Calling module function.')
    module.multiply(5, 2)
    log.info('Finished.')


if __name__ == '__main__':
    run()
