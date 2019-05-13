import unittest

import numpy as np

import utils_seg


# import cProfile


class UtilsTest(unittest.TestCase):
    # cProfile.run('re.compile("test_coreset_merging")')

    def test_svd1(self):
        data1 = np.array([
            [1.05, 2.12],
            [2.18, 4.22],
            [3.31, 6.35],
            [4.42, 9.38],
            [5.5, 10.49],
        ])
        idxs = np.arange(len(data1))
        coeff, cost = utils_seg.best_fit_line_and_cost(data1, idxs)
        print("coeff: {}, cost: {}".format(coeff, cost))
        utils_seg.plot_data_vs_svd_line_3d(data1, coeff, -2, 9)

    def test_svd2(self):
        data1 = np.array([
            [10, 60],
            [20, 70],
            [30, 80],
            [40, 90],
            [50, 100],
        ])
        idxs = np.arange(len(data1))
        coeff, cost = utils_seg.best_fit_line_and_cost(data1, idxs)
        print("coeff: {}, cost: {}".format(coeff, cost))
        utils_seg.plot_data_vs_svd_line_3d(data1, coeff, 10, 50)
