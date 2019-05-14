#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging as log
import numpy as np
import sys
import inspect

from typing import List, Optional
try:
    from utils import this_file_name, this_func_name
except ImportError:
    from k_segment_implementation.utils import this_file_name, this_func_name

log.basicConfig(stream=sys.stderr, level=log.INFO)


class Dividers:
    # __slots__ = "cost", "begin", "end", "approx_coefficients", "sub_divs"

    def __init__(self, begin: int, end: int, sub_divs: Optional[List["Dividers"]] = None,
                 cost: Optional[float] = None, approx_coefficients: Optional[List[int]] = None,
                 repr_idx: Optional[List[int]] = None):
        self.cost = cost
        self.begin = begin
        self.end = end
        self.approx_coefficients = approx_coefficients
        self.sub_divs = sub_divs
        self.repr_idx = repr_idx

    def __lt__(self, other):
        if self.cost is None:
            print("found your error case")
        try:
            res = self.cost < other.cost
            return res
        except ValueError as e:
            print(e)
        # return self.cost < other.cost

    def __repr__(self):
        return "Dividers({}, {}, {}, {}, {}, {})".format(self.cost, self.begin, self.end,
                                                         self.approx_coefficients, self.sub_divs, self.repr_idx)

    def update_cost(self, cost):
        log.debug('{}, {}, {}'.format(this_file_name(), this_func_name(), inspect.currentframe().f_lineno))
        self.cost = cost


def get_dividers_point_pairs_for_drawing(dividers: List["Dividers"]):
    log.debug('{}, {}, {}'.format(this_file_name(), this_func_name(), inspect.currentframe().f_lineno))
    result = []
    try:
        for y in dividers:
            # TODO remove?
            if y.sub_divs is not None and type(y.sub_divs) != 'NoneType':
                for x in y.sub_divs:
                    # x.begin y.begin x.begin y.end
                    # x.end y.begin  x.end y.end
                    x1y1 = (x.begin, y.begin) # x.begin y.begin
                    x1y2 = (x.begin, y.end) #  x.begin y.end
                    x2y1 = (x.end, y.begin) #  x.end y.begin
                    x2y2 = (x.end, y.end) #  x.end y.end

                    line1 = [x1y1, x1y2]
                    line2 = [x2y1, x2y2]
                    line3 = [x1y1, x2y1]
                    line4 = [x1y2, x2y2]

                    result.extend([line1, line2, line3, line4])
    except TypeError as e:
        print(e)
    return result
