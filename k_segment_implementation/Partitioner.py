#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from collections import namedtuple

import numpy as np

from k_segment_coreset.utils_seg import calc_best_fit_line_polyfit, sqrd_dist_sum

from typing import List, Optional


class Divs():
    def __init__(self, b: int, e: int, sub_divs: Optional[List["Divs"]] = None):
        self.b = b
        self.e = e
        self.sub_divs = sub_divs


class Partitioner:
    def __init__(self, data, depth=0):
        self.Item = namedtuple('Item', ['cost', 'P_i', 'g_i'])
        self.raw_data = data
        self.raw_data_shape = self.raw_data.shape
        self.enumerated_data = list(enumerate(data.reshape()))
        #self.enumerated_data = np.reshape(self.enumerated_data, self.raw_data_shape)
        self.size = self.enumerated_data.size
        # print("list(enumerate(data)): ", list(enumerate(data)))
        self.partitioned_data = []
        self.output = []

    def partition(self, k, axis=0):
        if k < 2 or k > self.size/12 + 1:
            print("k: ", k, self.size)
            raise ValueError('Bad k number. k must satisfy inequality: (k > 2 and k < n/12+1)')
        self.partitioned_data = []
        temp = np.array_split(self.enumerated_data, 4 * k, axis)
        for i in temp:
            item = self.Item(-1, i, [])
            self.partitioned_data.append(item)
        return self.partitioned_data

    def get_cost_for_group(self, index):
        return self.partitioned_data[index].c_i

    def update_cost_for_group(self, i, cost):
        item = self.Item(cost, self.partitioned_data[i].P_i, self.partitioned_data[i].g_i)
        self.remove_item(i)
        self.add_item(item)

    def calculate_cost_for_group(self, index):
        print("calculate_cost_for_group(self, index): debug me (make recursively retrieval of data)")
        data = []
        for p in self.partitioned_data[index].P_i:
            print("p: ", p)
            data.append(self.partitioned_data[index].P_i[1])
        g_i = calc_best_fit_line_polyfit(np.array(data))
        return sqrd_dist_sum(np.array(data), g_i)

    # iter_tools
    #
    def remove_item(self, index):
        print("remove_item(self, index): debug me (make recursively retrieval of data)")
        self.output.append(self.partitioned_data[index])
        for p in self.partitioned_data[index].P_i:
            p_for_deletion = tuple(p)
            self.enumerated_data.remove(p_for_deletion)
        del self.partitioned_data[index]

    def add_item(self, item):
        self.partitioned_data.append(item)
        self.partitioned_data.sort(key=lambda x: x.cost)

    def get_output(self):
        return self.output

    def calculate_size(self, data):
        size = 1
        for dimension in np.shape(data):
            size *= dimension
        return size

    def get_segment(self, index):
        return self.partitioned_data[index]
