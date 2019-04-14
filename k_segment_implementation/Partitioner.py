#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from collections import namedtuple
from numpy import array_split


class Partitioner:
    def __init__(self, data):
        self.Item = namedtuple('Item', ['cost', 'P_i', 'g_i'])
        self.size = len(data)
        self.enumerated_data = list(enumerate(data))
        #print("list(enumerate(data)): ", list(enumerate(data)))
        self.partitioned_data = []
        self.output = []

    def partition(self, k):
        if k < 2 or k > self.size/12 + 1:
            raise ValueError('Bad k number. k must satisfy inequality: (k > 2 and k < n/12+1)')
        self.partitioned_data = []
        temp = array_split(self.enumerated_data, 4 * k)
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

    def update_approximation_coefficients_for_group(self, i, g):
        self.partitioned_data[i].g_i = g

    def remove_item(self, index):
        self.output.append(self.partitioned_data[index])
        for p in self.partitioned_data[index].P_i:
            p_for_deletion = tuple(p)
            self.enumerated_data.remove(p_for_deletion)
        del self.partitioned_data[index]

    def add_item(self, item):
        self.partitioned_data.append(item)
        self.partitioned_data.sort(key=lambda x: x.cost)
