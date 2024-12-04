import sys
from pathlib import Path
import logging
import sys

def setup_logger(log_file):
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('')
    log = logging.getLogger('logger')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(log_file, encoding='utf-8')
    streamHandler = logging.StreamHandler(sys.stderr)

    fileHandler.setLevel(logging.DEBUG)
    streamHandler.setLevel(logging.DEBUG)
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    return log

def flush_logger():
    logger = logging.getLogger("logger")
    if logger.handlers:
        for handler in logger.handlers:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.stream.flush()