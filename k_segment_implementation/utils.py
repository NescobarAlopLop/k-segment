#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from random import randint
from random import seed

import sys
import inspect


def generate_data_array(size):
    array = []
    seed()
    for _ in range(size):
        array.append(randint(0, 255))
    return array


def this_file_name():
    return inspect.currentframe().f_code.co_filename


def this_func_name():
    return sys._getframe().f_back.f_code.co_name


