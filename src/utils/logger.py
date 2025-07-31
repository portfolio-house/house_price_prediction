import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir="logs"):
    """Configure comprehensive logging system"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(f"{log_dir}/{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)
    
    # Formatters
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger