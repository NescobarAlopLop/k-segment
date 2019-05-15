#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import heapq

try:
    from utils_seg import best_fit_line_and_cost
except ImportError:
    from k_segment_coreset.utils_seg import best_fit_line_and_cost

from Dividers import Dividers


def bicriteria(data, k, depth, cost_function):
    if data.shape[1] >= 1 and depth > 0:
        output = []
        data_indices = np.arange(data.shape[0])
        costs = []
        while len(data_indices) >= 2 * k + 1:
            row_indices = np.array_split(data_indices, 3 * k)
            tmp = []
            for indices in row_indices:
                if len(indices) > 0:
                    g_i, c_i, sub_divs = bicriteria(data[indices, :].T, k, depth - 1, cost_function)
                    begin = indices[0]
                    end = indices[-1]
                    dividers = Dividers(begin, end, sub_divs, c_i, g_i, indices)
                    tmp.append(dividers)

            n_best = heapq.nsmallest(k + 1, tmp)
            output.extend(n_best)

            values_to_remove = []
            for div in n_best:
                values_to_remove.extend(div.interval_inner_indices)
            rows_to_remove = np.nonzero(np.asarray(values_to_remove)[:, None] == data_indices)[1]

            data_indices = np.delete(data_indices, rows_to_remove, axis=0)

        for item in output:
            costs.append(item.cost)

        g_i, c_i = cost_function(data[data_indices, :].T)
        return g_i, c_i, output
    else:
        g_i, c_i = cost_function(data.T)
        return g_i, c_i, None
