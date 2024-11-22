import sys
from pathlib import Path
import logging
import logging
import sys

class Logger(logging.Logger):
    def __init__(self, filename):
        super().__init__(filename)
        self.filename = filename
        self.terminal = sys.stdout
        with open(self.filename, "w", encoding="utf-8") as log_file:
            log_file.write("")

    def write(self, message):
        self.terminal.write(message) 
        self.terminal.flush()         
        with open(self.filename, "a", encoding="utf-8") as log_file:
            log_file.write(message)
            log_file.flush()
        # print("file is closed")
        

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return False

    

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r", encoding="utf-8") as f:
        return f.read()

def flush():
    with open("output.log", "w", encoding="utf-8") as f:
        f.write("")