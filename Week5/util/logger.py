import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_file='app.log', log_level=logging.INFO, max_bytes=5*1024*1024, backup_count=5):
    """
    Set up a logger that writes to a file and the console with rotation.
    
    Args:
        log_file (str): Path to the log file (default: 'app.log').
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        max_bytes (int): Maximum size of the log file in bytes before rotation (default: 5MB).
        backup_count (int): Number of backup log files to keep (default: 5).
    
    Returns:
        logging.Logger: Configured logger object.
    """
    # Create a logger
    logger = logging.getLogger('ActionClassificationLogger')
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times if the logger is called more than once
    if logger.handlers:
        return logger

    # Define the log format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,  # Rotate when file size exceeds max_bytes
        backupCount=backup_count  # Keep this many backup files
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Example usage
if __name__ == '__main__':
    # Set up the logger
    logger = setup_logger(
        log_file='action_classification.log',
        log_level=logging.INFO,
        max_bytes=5*1024*1024,  # 5MB
        backup_count=5
    )

    # Test the logger
    logger.debug('This is a debug message (not shown with INFO level)')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')

    # Example with an exception
    try:
        1 / 0
    except ZeroDivisionError as e:
        logger.exception('An error occurred: %s', str(e))