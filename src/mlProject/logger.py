import os
import sys
import logging
from logging.handlers import RotatingFileHandler

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Using RotatingFileHandler to manage log size (max 5MB per file, keep 3 backups)
file_handler = RotatingFileHandler(log_filepath, maxBytes=5*1024*1024, backupCount=3)
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level= logging.INFO,
    format= logging_str,
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger("mlProjectLogger")
