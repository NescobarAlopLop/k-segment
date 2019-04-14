#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from random import randint
from random import seed


def generate_data_array(size):
    array = []
    seed()
    for _ in range(size):
        array.append(randint(0, 255))
    return array

