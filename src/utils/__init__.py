from .generic import *
from .file_utils import *


class Logger:
    def __init__(self, pipe, log_path: str):
        self.pipe = pipe
        self.log = open(log_path, 'w')

    def write(self, message: str):
        self.pipe.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.pipe.flush()
        self.log.flush()