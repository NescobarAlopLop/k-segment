#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
from scipy import misc
import numpy as np
import heapq
try:
    from utils import generate_data_array
except ImportError:
    from k_segment_implementation.utils import generate_data_array
from Dividers import Dividers
from k_segment_coreset.utils_seg import calc_best_fit_line_polyfit, sqrd_dist_sum


def bicriteria(data, k, depth, cost_function, approximation_function):
    if data.shape[1] != 1 and depth > 0:
        output = []
        data_indices = np.arange(data.shape[0])
        costs = []
        while len(data_indices) >= 2 * k + 1:
            print("len(data_indices): ", len(data_indices))
            row_indices = np.array_split(data_indices, 3 * k)
            tmp = []
            print("row indices: ", row_indices)
            for indices in row_indices:
                print("data: ", data)
                print("data[indices, :].T: ", data[indices, :].T)
                data = np.column_stack((np.arange(1, len(data) + 1), data[:]))
                g_i, c_i, sub_divs = bicriteria(data[indices, :].T, k, depth - 1, cost_function, approximation_function)
                print("g_i, c_i: ", g_i, c_i)
                begin = indices[0]
                end = indices[-1]
                dividers = Dividers(begin, end, sub_divs, c_i, g_i, indices)
                tmp.append(dividers)

            n_best = heapq.nsmallest(k + 1, tmp)
            output.extend(n_best)

            values_to_remove = []
            for div in n_best:
                values_to_remove.extend(div.repr_idx)
            rows_to_remove = np.nonzero(np.asarray(values_to_remove)[:, None] == data_indices)[1]

            data_indices = np.delete(data_indices, rows_to_remove, axis=0)
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
        g_i = approximation_function(data.T)
        # g_i = approximation_function(data.reshape((data.shape[1], data.shape[0])))
        c_i = cost_function(data.T, g_i)
        return g_i, c_i, None


def main(argv):

    size = 256
    k = 2
    depth = 1

    data = generate_data_array(size)
    data = np.reshape(data, (8, 32))
    data2 = np.asarray([10, 10, 10, 11, 10, 11, 10, 14,
                       10, 10, 10, 12, 10, 15, 10, 12,
                       10, 13, 10, 16, 10, 10, 10, 13])
    data2 = misc.imread('/home/ge/magneton/input_data/tomato_100_67.JPG')
    data2 = np.mean(data2, axis=2)
    data2 = np.column_stack((np.arange(1, len(data2) + 1), data2[:]))
    # data2 = np.vstack((data2,data2))
    _, _, divs = bicriteria(data2, k, depth, sqrd_dist_sum, calc_best_fit_line_polyfit)
    print('#' * 60)
    for div in divs:
        print(div.get_dividers_point_pairs_for_drawing())
    print(data)

    return 0


if __name__ == "__main__":
    print("Python version: {}".format(sys.version))
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)
