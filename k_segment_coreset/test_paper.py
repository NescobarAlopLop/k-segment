import numpy as np
from utils_seg import gen_synthetic_graph
import ksegment
import CoresetKSeg
from CoresetKSeg import CoresetKSeg
import unittest
import matplotlib.pyplot as plt
from utils_seg import load_csv_into_dataframe, visualize_2d


def load_csv_file(path, n_rows=None):
    # rv = np.genfromtxt(path, delimiter=',')
    rv = np.loadtxt(path, float, delimiter=',', skiprows=1)
    if n_rows:
        rv = rv[:n_rows]
    return rv


class KSegmentTest(unittest.TestCase):
    # cProfile.run('re.compile("test_coreset_merging")')
    def test_foo(self):
        d = gen_synthetic_graph(110, 5, dim=2)
        plt.scatter(d[:, 0], d[:, 1], s=3)
        self.assertTrue(True)
        # plt.show()

    def test_basic_demo_synth(self, n=500, k=5, epsilon=0.5, show_fig=False):
        data = gen_synthetic_graph(n, k, 1)
        data = np.column_stack((np.arange(1, len(data) + 1), data[:]))

        coreset = CoresetKSeg(data, k, epsilon, add_index_col=False)
        print("original data len\t{}\ncoreset points len:\t{}".format(len(data), len(coreset)))
        print(len(data), len(coreset))
        self.assertGreater(len(data), len(coreset))

    def test_loop(self):
        n = 800
        k = 10
        for i in range(100, n, 400):
            for j in range(3, k, 3):
                self.test_basic_demo_synth(n=i, k=j)

    def test_small_test_case(self):
        data = np.asarray([10, 10, 10, 11, 10, 11, 10, 14,
                           10, 10, 10, 12, 10, 15, 10, 12,
                           10, 13, 10, 16, 10, 10, 10, 13])
        data = np.column_stack((np.arange(1, len(data) + 1), data[:]))

        k = 4
        eps = 0.5
        k_eps_coreset = CoresetKSeg(data, k, eps, add_index_col=False)
        self.assertGreater(len(data), len(k_eps_coreset))
        visualize_2d(data, k_eps_coreset.coreset, k, eps, show=False)

    def test_coreset_k_seg_init(self):
        data = load_csv_into_dataframe("/home/ge/k-segment/datasets/KO_no_date.csv").values[600:800]
        data = np.column_stack((np.arange(1, len(data) + 1), data[:]))

        k = 4
        eps = 0.3
        k_eps_coreset = CoresetKSeg(data, k, eps, add_index_col=False)
        self.assertGreater(len(data), len(k_eps_coreset))

        visualize_2d(data, k_eps_coreset.coreset, k, eps, show=False)

    def test_coreset_k_seg_call(self):
        data = load_csv_into_dataframe("/home/ge/k-segment/datasets/KO_no_date.csv").values[600:800]
        data = np.column_stack((np.arange(1, len(data) + 1), data[:]))

        k = 4
        eps = 0.3
        k_eps_coreset = CoresetKSeg(data, k, eps, add_index_col=False)
        k_eps_coreset(data)
        self.assertGreater(len(data), len(k_eps_coreset))
        visualize_2d(data, k_eps_coreset.coreset, k, eps, show=False)
