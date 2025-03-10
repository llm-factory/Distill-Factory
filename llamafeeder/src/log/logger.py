import sys
from pathlib import Path
import logging
import sys
from datetime import datetime

class Logger():
    def __init__(self):
        self.name = None
        self.logger = self.setup_logger()
    def getName(self):
        return self.name
    def setup_logger(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"log/logger_{current_time}.log"    
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('')
        log = logging.getLogger('logger')
        self.name = log_file

        log.handlers.clear()
        log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileHandler = logging.FileHandler(log_file, encoding='utf-8')
        streamHandler = logging.StreamHandler(sys.stderr)

        fileHandler.setLevel(logging.DEBUG)
        streamHandler.setLevel(logging.INFO)
        
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)
        
        log.addHandler(fileHandler)
        log.addHandler(streamHandler)
        log.propagate = False
        return log

    def flush_logger():
        logger = logging.getLogger("logger")
        if logger.handlers:
            for handler in logger.handlers:
                handler.flush()
                if isinstance(handler, logging.FileHandler):
                    handler.stream.flush()
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)