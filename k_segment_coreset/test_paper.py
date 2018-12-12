import numpy as np
import utils_seg
from utils_seg import gen_synthetic_graph
import ksegment
import CoresetKSeg
import unittest
import matplotlib.pyplot as plt
# import cProfile


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
        # plt.show()

    def test_basic_demo_synth(self, n=1000, k=5, epsilon=0.5):
        # k = 11
        # epsilon = 0.2
        # n = 200
        data = gen_synthetic_graph(n, k, 1)
        p = np.c_[np.mgrid[1:len(data) + 1], data]

        coreset = CoresetKSeg.build_coreset(points=p, k=k, eps=epsilon)
        dividers = ksegment.coreset_k_segment(coreset, k)
        coreset_points = ksegment.get_coreset_points(coreset)
        print("original data len\t{}\ncoreset points len:\t{}".format(len(p), len(coreset_points)))
        utils_seg.visualize_2d(p, dividers, len(coreset), coreset_points,
                               show=True
                               )     # Uncomment to see results
        print (len(p), len(coreset_points))
        self.assertGreater(len(p), len(coreset_points))

    def test_loop(self):
        n = 800
        k = 10
        for i in range(100, n, 400):
            for j in range(3, k, 3):
                self.test_basic_demo_synth(n=i, k=j)

    @staticmethod
    def test_bicriteria2():
        dim = 1
        n = 12
        k = 2
        n, m = 4 * k + 2, 5
        data = np.arange(n * m).reshape((n, m))
        p = np.c_[np.mgrid[1:data.shape[0] + 1], data]
        print("test input data\n{}".format(p))
        f = [0.0] * len(p)
        b = CoresetKSeg.bicriteria(data, k, f)
        b2 = CoresetKSeg.bicriteria2(data, k)

        print(b, b2)

    def test_bicriteria_small_test_case(self):
        dim = 1
        points = np.asarray([10, 10, 10, 11, 10, 11, 10, 14,
                             10, 10, 10, 12, 10, 15, 10, 12,
                             10, 13, 10, 16, 10, 10, 10, 13])
        k = 10
        eps = 0.3
        n = 5000
        # points = gen_synthetic_graph(n, k, dim, deviation=2, max_diff=7)
        points = np.c_[np.mgrid[1:len(points) + 1], points]
        # points = np.c_[np.mgrid[0:n], points]

        n = len(points)
        f = [0.0] * n
        bi_crit = CoresetKSeg.bicriteria(points, k, f, mul=1)
        print(f, sum(f))
        bi_crit2 = CoresetKSeg.bicriteria2(points, k, mul=4)

        sigma = (eps ** 2 * bi_crit) / (100 * k * np.log2(len(points)))
        coreset = CoresetKSeg.BalancedPartition(points, eps, sigma, False)

        print("bicritiria estimate:")
        print(bi_crit, bi_crit2, sum(f))
        print("balanced partition len: {} /original {} = 1 / {}".format(len(coreset), len(points),
                                                                       len(points) / len(coreset)))
        print("coreset size percent: {:.2f}%".format(100 * len(coreset)/float(len(points))))
        # coreset_points = ksegment.get_coreset_points(coreset)
        # dividers = ksegment.coreset_k_segment(coreset, k)
        # coreset size has to be: O(dk/eps^2)
        # utils_seg.visualize_2d(points, dividers, len(coreset), coreset_points, show=True)
        utils_seg.visualize_2d(points, coreset, k, eps, show=True)
        # print(f)

    def test_coreset_size_over_k(self):
        dim = 1
        k = 3
        eps = 0.00
        n = 8000
        points = gen_synthetic_graph(n, k, dim, deviation=1, max_diff=30)
        points = np.c_[np.mgrid[1:len(points) + 1], points]

        n = len(points)
        f = [0.0] * n
        bi_crit = CoresetKSeg.bicriteria(points, k, f, mul=4)
        print("bicritiria estimate:")
        print("sum from function: ", bi_crit)
        print("sum over f:", sum(f))

        sigma = (eps ** 2 * bi_crit) / (100 * k * np.log2(len(points)))
        coreset = CoresetKSeg.BalancedPartition(points, eps, sigma, False)

        print("balanced partition len: {} /original {} = 1 / {}".format(len(coreset), len(points),
                                                                        len(points) / len(coreset)))
        print("coreset size percent: {:.2f}%".format(100 * len(coreset) / float(len(points))))
        # coreset_points = ksegment.get_coreset_points(coreset)
        # dividers = ksegment.coreset_k_segment(coreset, k)
        print("coreset size has to be: O(dk/eps^2) = {}".format(dim * k / eps ** 2))
        plt.scatter(points[:, 0], points[:, 1], s=3)
        plt.show()
        # utils.visualize_2d(points, dividers, len(coreset), coreset_points, show=True)
        # print(f)
