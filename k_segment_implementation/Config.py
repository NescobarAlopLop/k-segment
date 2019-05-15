#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging as log
import sys


class Config:
    __slots__ = "debug_level", "debug_stream"

    def __init__(self):
        self.debug_level = log.DEBUG
        self.debug_stream = sys.stderr

    def __repr__(self):
        return "Config({}, {}".format(self.debug_level, self.debug_stream)
