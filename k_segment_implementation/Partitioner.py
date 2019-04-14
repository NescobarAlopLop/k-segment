#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging as log
import numpy as np
import sys

from collections import namedtuple
from typing import List, Optional

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
        return "Dividers({}, {}, {}".format(self.begin, self.end, self.sub_divs)

    def update_cost(self, cost):
        self.cost = cost


class Partitioner:
    def __init__(self, data, depth=0):
        self.raw_data = data
        self.raw_data_shape = self.raw_data.shape
        self.size = self.count_elements(self.raw_data)
        self.depth = depth
        self.enumerated_data = list(enumerate(self.serialize_2d_data(data)))

        self.dividers = None
        self.output = []

    def partition(self, k, axis=0):
        if k < 2 or k > self.size / 12 + 1:
            raise ValueError('Bad k number. k must satisfy inequality: (k > 2 and k < n/12+1)')
        indices = range(self.raw_data_shape[axis])
        temp = np.array_split(indices, 4 * k, axis)
        dividers = []
        for i in temp:
            dividers.append(Dividers(i[0][0], i[-1][0]))
            print(dividers)
            # FIXME
            # self.partitioned_data.append(item)
        # return self.partitioned_data

    # def partition(self, k, axis=0):
    #     if k < 2 or k > self.size / 12 + 1:
    #         # print("k: ", k, self.size)
    #         raise ValueError('Bad k number. k must satisfy inequality: (k > 2 and k < n/12+1)')
    #     self.partitioned_data = []
    #     temp = np.array_split(self.enumerated_data, 4 * k, axis)
    #     for i in temp:
    #         item = self.Item(-1, i, [])
    #         self.partitioned_data.append(item)
    #     return self.partitioned_data

    def get_dividers(self):
        pass
        # TODO list of pairs
        # ((,),(,))

    # def calculate_total_cost(self):
    #     costs = []
    #     for p in self.partitioned_data:
    #         costs.append(p.c_i)
    #     g_i = calc_best_fit_line_polyfit(np.array(costs))
    #     return sqrd_dist_sum(np.array(costs), g_i)

    # def update_cost_for_group(self, i, cost):
    #     item = self.Item(cost, self.partitioned_data[i].P_i, self.partitioned_data[i].g_i)
    #     self.remove_item(i, self.depth)
    #     self.partitioned_data.append(item)
    #     self.partitioned_data.sort(key=lambda x: x.cost)
    #
    # def calculate_cost_for_group(self, index):
    #     print("calculate_cost_for_group(self, index): debug me (make recursively retrieval of data)")
    #     data = []
    #     for p in self.partitioned_data[index].P_i:
    #         print("p: ", p)
    #         data.append(self.partitioned_data[index].P_i[1])
    #     g_i = calc_best_fit_line_polyfit(np.array(data))
    #     return sqrd_dist_sum(np.array(data), g_i)

    # # iter_tools
    # # update dividers
    # def remove_item(self, index, depth):
    #     print("remove_item(self, index): debug me (make recursively retrieval of data)")
    #     if depth < 0:
    #         return
    #     self.output.append(self.partitioned_data[index])
    #     if depth == 0:
    #         for p in self.partitioned_data[index].P_i:
    #             p_for_deletion = tuple(p)
    #             self.enumerated_data.remove(p_for_deletion)
    #         del self.partitioned_data[index]
    #     else:
    #         for i in enumerate(self.partitioned_data[index].P_i):
    #             self.remove_item(i, depth - 1)

    def get_output(self):
        return self.output

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

