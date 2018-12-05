import numpy as np
from utils import load_csv_file, visualize_2d, gen_synthetic_graph
import ksegment
import CoresetKSeg
import unittest
import matplotlib.pyplot as plt


class KSegmentTest(unittest.TestCase):
    @staticmethod
    def test_foo():
        """
        visual test for synthetic graph generation
        """
        d = gen_synthetic_graph(n=400, k=7, deviation=1, max_diff=15)
        plt.scatter(np.arange(len(d)), d, s=3)
        plt.show()

    def test_from_file(self, path=None, n=None, b=0, k=10, eps=0.1, show=True):
        """
        :param path:    path to file
        :type path:     str
        :param n:       number of point to take starting at b
        :type n:        int
        :param b:       take points starting from line b
        :type b:        int
        :param k:       number of segments
        :type k:        int
        :param eps:     epsilon error
        :type eps:      float
        :param show:    show plots
        :type show:     bool
        :return:
        """
        if not path:
            data = gen_synthetic_graph(n=400, k=k, dim=1, deviation=0.1, max_diff=3)
            # data = gen_synthetic_graph(n=400, k=k, dim=1, deviation=1, max_diff=2)
        else:
            data = load_csv_file(path)
        if b > len(data):
            b = 0
        if not n:
            n = len(data)
        else:
            n = b + n
        if n > len(data):
            n = len(data)
        print("using points {} to {}".format(b, b+n))
        data = data[b:n, :]

        p = np.column_stack((np.arange(1, len(data) + 1), data[:]))

        coreset = CoresetKSeg.build_coreset(p, k, eps)
        dividers = ksegment.coreset_k_segment(coreset, k)
        coreset_points = ksegment.get_coreset_points(coreset)

        print("original data len\t{}\ncoreset points len:\t{}".format(len(p), len(coreset_points)))
        # self.assertGreater(len(p), len(coreset_points))

        print(coreset_points)
        # pass data input as 2d, first col is time, 2nd is average on all data cols
        p = np.column_stack((p[:, 0], np.linalg.norm(p[:, 1:], axis=1)))

        visualize_2d(p, coreset, k, eps, show=show)

    def test_KO_stock(self):
        """
        run k-segmentation coreset on CocaCola stock price
        """
        self.test_from_file("../datasets/KO_no_date.csv", n=600, b=540, k=8, show=False)

    def test_chunk_num_1(self):
        """
        run k-segmentation coreset on iotshield input test
        """
        self.test_from_file("../datasets/chunk_num_1.csv", n=100, b=100, show=False)

    def test_basic_demo_synth(self):
        """
        run k-segmentation on random data with default values
        """
        self.test_from_file(show=True)

    def test_loop(self):
        """
        generate graphs for random data
        """
        n = 1200
        k = 10
        for i in range(500, n, 200):
            for j in range(3, k, 2):
                self.test_from_file(n=i, k=j, show=False)

    def test_sanity_check(self, show=True):
        """
        :param path:    path to file
        :type path:     str
        :param n:       number of point to take starting at b
        :type n:        int
        :param b:       take points starting from line b
        :type b:        int
        :param k:       number of segments
        :type k:        int
        :param eps:     epsilon error
        :type eps:      float
        :param show:    show plots
        :type show:     bool
        :return:
        """
        n = 110
        dim = 1
        one = 100 * np.ones((n, 1)) - np.random.randn(n).reshape((n, 1))/250
        two = 200 * np.ones((n, 1)) - np.random.randn(n).reshape((n, 1))/250
        three = 300 * np.ones((n, 1)) - np.random.randn(n).reshape((n, 1))/150
        arr = np.row_stack((one, two, three))
        p = np.column_stack((np.arange(1, len(arr) + 1), arr[:]))

        k = 3
        eps = 0.4
        coreset = CoresetKSeg.build_coreset(p, k, eps)
        dividers = ksegment.coreset_k_segment(coreset, k)
        coreset_points = ksegment.get_coreset_points(coreset)

        print("original data len\t{}\ncoreset points len:\t{}".format(len(p), len(coreset_points)))
        # self.assertGreater(len(p), len(coreset_points))
        print("coreset size has to be: O(dk/eps^2) = {}".format((dim * k) / eps ** 2))
        # pass data input as 2d, first col is time, 2nd is average on all data cols
        p = np.column_stack((p[:, 0], np.linalg.norm(p[:, 1:], axis=1)))

        # visualize_2d(p, dividers, len(coreset_points), coreset_points, show=show)
        visualize_2d(p, coreset, k, eps, show=show)
