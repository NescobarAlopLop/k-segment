import unittest

import matplotlib.pyplot as plt
import numpy as np

import CoresetKSeg
import ksegment
import test
from utils_seg import load_csv_into_dataframe, visualize_2d, gen_synthetic_graph


class KSegmentTest(unittest.TestCase):
    @staticmethod
    def test_foo():
        """
        visual test for synthetic graph generation
        """
        d = gen_synthetic_graph(n=400, k=7, deviation=1, max_diff=15)
        plt.scatter(np.arange(len(d)), d, s=3)
        plt.show()

    def test_from_file(self, path: str = None,
                       n: int = None, b: int = 0, k: int = 10,
                       eps: float = 0.8, show: bool = False) -> None:
        """
        :param path:    path to file
        :param n:       number of point to take starting at b
        :param b:       take points starting from line b
        :param k:       number of segments
        :param eps:     epsilon error
        :param show:    show plots
        """
        if path is None:
            data = gen_synthetic_graph(n=400, k=k, dim=1, deviation=0.01, max_diff=3)
            # data = gen_synthetic_graph(n=400, k=k, dim=1, deviation=1, max_diff=2)
        else:
            data = load_csv_into_dataframe(path).values
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
        print("coreset size has to be: O(k) · (log n / eps^2 ) = {}".format(k * (np.log2(n) / eps ** 2)))

        coreset = CoresetKSeg.CoresetKSeg.compute_coreset(p, k, eps)
        coreset_points = ksegment.get_coreset_points(coreset)

        print("original data len\t{}\ncoreset points len:\t{}".format(len(p), len(coreset_points)))
        self.assertGreaterEqual(len(p), len(coreset_points))

        visualize_2d(p, coreset, k, eps, show=show)

    def test_KO_stock(self):
        """
        run k-segmentation coreset on CocaCola stock price
        """
        self.test_from_file("../datasets/KO_no_date.csv", n=100, b=0, k=8, eps=0.4, show=False)

    def test_chunk_num_1(self):
        """
        run k-segmentation coreset on iotshield input test
        """
        self.test_from_file("../datasets/chunk_num_1.csv", n=200, b=100, k=8, eps=0.6, show=False)

    def test_basic_demo_synth(self):
        """
        run k-segmentation on random data with default values
        """
        self.test_from_file(n=12000, k=1000, show=False)

    def test_loop(self):
        """
        generate graphs for random data
        """
        n = 1200
        k = 10
        for i in range(500, n, 300):
            for j in range(3, k, 3):
                self.test_from_file(n=i, k=j, show=False)

    def test_sanity_check(self, show: bool = False) -> None:
        n = 200

        p = test.example3(n)
        p = np.column_stack((np.arange(1, len(p) + 1), p[:]))
        k = 3
        eps = 0.1

        coreset = CoresetKSeg.CoresetKSeg.compute_coreset(p, k, eps)

        print("coreset size has to be: O(k) · (log n / eps^2 ) = {}".format(k * (np.log2(n) / eps ** 2)))
        visualize_2d(p, coreset, k, eps, show=show)

    # def test_class_and_static_functions(self):
    #     # TODO: add proper self.assertEqual
    #     points = load_csv_into_dataframe("/home/ge/k-segment/datasets/bicriteria test case - coreset.csv").values[:, 1]
    #     points = np.column_stack((np.arange(1, len(points) + 1), points[:]))
    #     k = 4
    #     eps = 0.2
    #     coreset = CoresetKSeg.CoresetKSeg.compute_coreset(points, k, eps)
    #     visualize_2d(points, coreset, k, eps, show=False)
    #
    #     coreset2 = CoresetKSeg.CoresetKSeg.compute_coreset(coreset, k, eps, is_coreset=True)
    #     visualize_2d(ksegment.get_coreset_points(coreset), coreset2, k, eps, show=False)
    #
    #     coreset_class = CoresetKSeg.CoresetKSeg(k, eps, weights=None)
    #     coreset_class.compute(points)
    #     visualize_2d(points, coreset_class.k_eps_coreset, k, eps, show=False)
    #     coreset_class2 = coreset_class.compute_coreset(coreset_class.k_eps_coreset, k, eps, is_coreset=True)
    #     visualize_2d(ksegment.get_coreset_points(coreset_class.k_eps_coreset), coreset_class2, k, eps, show=False)

    # def test_compare_spark_shuffle_map_to_singlethread(self):
    #     # points = load_csv_into_dataframe("/home/ge/k-segment/datasets/KO_no_date.csv").values
    #     points = load_csv_into_dataframe("/home/ge/k-segment/datasets/bicriteria test case - coreset.csv").values[:, 1]
    #     points = np.column_stack((np.arange(1, len(points) + 1), points[:]))
    #     k = 4
    #     eps = 0.2
    #
    #     from pyspark import SparkContext, SparkConf
    #     conf = SparkConf().setMaster('local[*]').setAppName('Test')
    #     # Set scheduler to FAIR:
    #     # http://spark.apache.org/docs/latest/job-scheduling.html#scheduling-within-an-application
    #     conf.set('spark.scheduler.mode', 'FAIR')
    #     sc = SparkContext(conf=conf)
    #     points_rdd = sc.parallelize(points, numSlices=2)
    #     points_rdd_1 = points_rdd.map(lambda x: CoresetKSeg.CoresetKSeg.compute_coreset(x, k, eps))
    #     coll = points_rdd_1.collect()
    #     coll.__len__()
    #     # visualize_2d(points, coreset, k, eps, show=True)
    #     # points_rdd = sc.parallelize(points, k).glom()
    #     # coresets_rdd_collected = points_rdd.map(
    #     #       lambda x: CoresetKSeg.build_coreset_on_pyspark(np.asarray(x), k, eps))\
    #     #       .reduce(print)
    #     # print(coresets_rdd_collected)
    #     # print(coresets_rdd_collected)
    #     # sc.stop()
    #     pass


if __name__ == '__main__':
    unittest.main()
