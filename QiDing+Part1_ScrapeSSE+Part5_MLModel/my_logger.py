# -*- coding: utf-8 -*-

import sys
import logging


class Logger(logging.Logger):
    INFO = logging.INFO
    WARNING = logging.WARNING
    DEBUG = logging.DEBUG
    ERROR = logging.ERROR

    def __init__(self, name='log', path=None, add_info='', level=logging.INFO, console=True):
        super(Logger, self).__init__(name=name)
        self.level = level
        self.formatter = logging.Formatter('%(asctime)s' + ' ' + add_info + '%(levelname)s %(message)s')
        self.log_path_list = []

        if path is not None:
            self.log_path_list.append(path)
            file_log = logging.FileHandler(path)
            file_log.setFormatter(self.formatter)
            self.addHandler(file_log)

        if console:
            console = logging.StreamHandler(stream=sys.stdout)
            console.setLevel(self.level)
            console.setFormatter(self.formatter)
            self.addHandler(console)

    def add_path(self, log_path, level=None):
        if log_path is not None:
            file_log = logging.FileHandler(log_path)
            if level is None:
                level = self.level
            file_log.setLevel(level)
            file_log.setFormatter(self.formatter)
            self.addHandler(file_log)


this_log = Logger()
