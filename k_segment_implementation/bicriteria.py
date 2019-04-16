#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys

import numpy as np
import heapq
from utils import generate_data_array
from Dividers import Dividers
from k_segment_coreset.utils_seg import calc_best_fit_line_polyfit, sqrd_dist_sum


def bicriteria(data, k, depth, cost_function, approximation_function):
    output = []
    if data.shape[1] != 1 and depth > 0:
        data_indices = np.arange(data.shape[0])
        while len(data_indices) >= 2 * k + 1:
            row_indices = np.array_split(data_indices, 4 * k)
            tmp = []
            for indices in row_indices:
                g_i, c_i, sub_divs = bicriteria(data[indices, :].T, k, depth - 1, cost_function, approximation_function)
                begin = indices[0]
                end = indices[-1]
                dividers = Dividers(begin, end, sub_divs, c_i, g_i)
                tmp.append(dividers)
            output.append(heapq.nsmallest(k + 1, tmp))
            # FIXME
            # need to update data_indices, otherwise the loop is infinite
    else:
        means = data.mean(axis=0)
        g_i = approximation_function(means)
        c_i = cost_function(means, g_i)

        return g_i, c_i, None


def main(argv):
    # size = argv[1]
    # k = argv[2]

    size = 256
    k = 2

    data = generate_data_array(size)
    data = np.reshape(data, (8, 32))
    print(data)

    return 0


if __name__ == "__main__":
    print("Python version: {}".format(sys.version))
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)
