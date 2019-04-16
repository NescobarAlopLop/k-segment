#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging as log
import numpy as np
import sys
import inspect

from typing import List, Optional
from utils import this_file_name, this_func_name
# from k_segment_coreset.utils_seg import calc_best_fit_line_polyfit, sqrd_dist_sum

log.basicConfig(stream=sys.stderr, level=log.DEBUG)


class Dividers:
    __slots__ = "cost", "begin", "end", "sub_divs"

    def __init__(self, begin: int, end: int, sub_divs: Optional[List["Dividers"]] = None,
                 cost: Optional[float] = None):
        self.cost = cost
        self.begin = begin
        self.end = end
        self.sub_divs = sub_divs

    def __repr__(self):
        return "Dividers({}, {}, {})".format(self.begin, self.end, self.sub_divs)

    def update_cost(self, cost):
        self.cost = cost


class Partitioner:
    def __init__(self, data, k,  depth=0, dividers=None):
        self.raw_data = data
        self.raw_data_shape = self.raw_data.shape
        self.size = self.count_elements(self.raw_data)
        self.depth = depth
        self.k = self.check_in_bounds('k', k, 2, self.size / 12 + 1)
        self.enumerated_data = list(enumerate(self.serialize_2d_data(data)))
        self.dividers = dividers if dividers is not None else self.__init_dividers(self.k)
        self.output = []

    def __init_dividers(self, k):
        log.debug('{}, {}, {}'.format(this_file_name(), this_func_name(), inspect.currentframe().f_lineno))
        indices = []
        if self.depth == 0:
            return self.__do_initial_axis_partition(k, 0)
        for dim in range(self.depth + 1):
            partitioned_dimension_indices = self.__do_initial_axis_partition(k, dim)
            indices.append(partitioned_dimension_indices)
        indices = indices[::-1]
        for dim in range(len(indices) - 1):
            for i in range(len(indices[dim + 1])):
                indices[dim + 1][i].sub_divs = indices[dim]
        return indices[-1]

    def __do_initial_axis_partition(self, k, axis):
        log.debug('{}, {}, {}'.format(this_file_name(), this_func_name(), inspect.currentframe().f_lineno))
        indices = np.arange(self.raw_data_shape[axis])
        temp = np.array_split(indices, 4 * k)
        dividers = []
        for i in temp:
            dividers.append(Dividers(i[0], i[-1]))
        return dividers

    def get_segment(self, depth, index):

    def get_dividers_point_pairs_for_drawing(self):
        pass
        # TODO list of pairs
        # ((,),(,))

    def get_output(self):
        return self.output

    @staticmethod
    def check_in_bounds(var_name, var, lower, upper):
        if var < lower or var > upper:
            log.debug('check_in_bounds({name}): {name} is {var}'.format(name=var_name,var=var))
            print("Current k: ", var)
            raise ValueError(
                'Bad {name} value. {name} must satisfy inequality:' +
                ' ( {name} > {lower} and {name} < {upper} )'.format(name=var_name, lower=lower, upper=upper))
        return var

    @staticmethod
    def count_elements(data):
        size = 1
        for dimension in np.shape(data):
            size *= dimension
        return size

    @staticmethod
    def serialize_2d_data(data):
        array_1d = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                array_1d.append(data[i][j])
        return array_1d

    @staticmethod
    def __file():
        return inspect.currentframe().f_code.co_filename

    @staticmethod
    def __func():
        return inspect.currentframe().f_back.f_code.co_name

