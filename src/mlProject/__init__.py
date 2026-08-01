import os
import sys
import logging
from pythonjsonlogger import jsonlogger

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

def _setup_logging():
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger as well to capture Flask/Gunicorn logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_filepath)
    stream_handler = logging.StreamHandler(sys.stdout)
    
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(module)s %(message)s'
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    return logging.getLogger("mlProjectLogger")

logger = _setup_logging()
