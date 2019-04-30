#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys

import numpy as np
import heapq
from utils import generate_data_array
from Dividers import Dividers
from k_segment_coreset.utils_seg import calc_best_fit_line_polyfit, sqrd_dist_sum


def bicriteria(data, k, depth, cost_function, approximation_function):
    if data.shape[1] != 1 and depth > 0:
        output = []
        data_indices = np.arange(data.shape[0])
        costs = []
        while len(data_indices) >= 2 * k + 1:
            print("len(data_indices): ", len(data_indices))
            row_indices = np.array_split(data_indices, 4 * k)
            tmp = []
            print("row indices: ", row_indices)
            for indices in row_indices:
                print("data: ", data)
                print("data[indices, :].T: ", data[indices, :].T)
                g_i, c_i, sub_divs = bicriteria(data[indices, :].T, k, depth - 1, cost_function, approximation_function)
                print("g_i, c_i: ", g_i, c_i)
                begin = indices[0]
                end = indices[-1]
                dividers = Dividers(begin, end, sub_divs, c_i, g_i)
                tmp.append(dividers)

            n_best = heapq.nsmallest(k + 1, tmp)
            output.extend(n_best)

            for divider in output:
                print("output: ", output)
                print("divider: ", divider)
                print("AAAAAAAAAAAA")
                if divider.begin == divider.end:
                    if divider.begin in data_indices:
                        print("DDDDDDDDDDDDD")
                        data_indices = np.delete(data_indices, divider.begin)
                    continue

                for i in range(divider.begin, divider.end):
                    print("BBBBBBBBBBBBB")
                    if i in data_indices:
                        print("CCCCCCCCCCCCC")
                        data_indices = np.delete(data_indices, i)

            print("data_indices: ", data_indices)
        for item in output:
            costs.append(item.cost)
        print("costs: ", costs)
        g_i = approximation_function(costs)
        c_i = cost_function(costs, g_i)

        return g_i, c_i, output
    else:
        # means = data.mean(axis=0)
        # print(means)
        g_i = approximation_function(data.reshape((data.shape[1], data.shape[0])))
        c_i = cost_function(data.reshape((data.shape[1], data.shape[0])), g_i)
        return g_i, c_i, None


def main(argv):

    size = 256
    k = 2
    depth = 1

    data = generate_data_array(size)
    data = np.reshape(data, (8, 32))
    bicriteria(data, k, depth, sqrd_dist_sum, calc_best_fit_line_polyfit)

    print(data)

    return 0


if __name__ == "__main__":
    print("Python version: {}".format(sys.version))
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)
