import numpy as np
import ksegment
import CoresetKSeg
from CoresetKSeg import CoresetKSeg
import unittest
import matplotlib.pyplot as plt
from utils_seg import load_csv_into_dataframe, visualize_2d
from k_segment_implementation.Partitioner import Partitioner
from utils_seg import calc_best_fit_line_polyfit, sqrd_dist_sum, gen_synthetic_graph


def load_csv_file(path, n_rows=None):
    # rv = np.genfromtxt(path, delimiter=',')
    rv = np.loadtxt(path, float, delimiter=',', skiprows=1)
    if n_rows:
        rv = rv[:n_rows]
    return rv


class PartitionerTest(unittest.TestCase):
    def test_foo(self):
        d = gen_synthetic_graph(110, 5, dim=2)
        plt.scatter(d[:, 0], d[:, 1], s=3)
        self.assertTrue(True)
        plt.show()


    def test_init(self):

        data = gen_synthetic_graph(n=200, k=1, deviation=0, dim=1)
        # data = np.arange(200)
        data = np.column_stack((np.arange(1, len(data) + 1), data[:]))

        plt.scatter(data[:, 0], data[:, 1], s=3)
        plt.show()

        g_i = calc_best_fit_line_polyfit(np.array(data))
        cost = sqrd_dist_sum(np.array(data), g_i)

        print("{:0.1f}".format(cost))

        p = Partitioner(data)
