import numpy as np
import unittest
import matplotlib.pyplot as plt
import CoresetKSeg
import ksegment
from utils_seg import load_csv_into_dataframe, visualize_2d, gen_synthetic_graph, calc_cost_dividers
import test


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
        if not path:
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

        coreset = CoresetKSeg.build_coreset(p, k, eps)
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
        self.test_from_file(show=False)

    def test_loop(self):
        """
        generate graphs for random data
        """
        n = 1200
        k = 10
        for i in range(500, n, 200):
            for j in range(3, k, 2):
                self.test_from_file(n=i, k=j, show=False)

    def test_sanity_check(self, show: bool = False) -> None:
        n = 200

        p = test.example3(n)
        p = np.column_stack((np.arange(1, len(p) + 1), p[:]))
        k = 3
        eps = 0.5

        coreset = CoresetKSeg.build_coreset(p, k, eps)

        print("coreset size has to be: O(k) · (log n / eps^2 ) = {}".format(k * (np.log2(n) / eps ** 2)))
        visualize_2d(p, coreset, k, eps, show=show)

    def test_both_bicrit(self):
        n = 110
        p = test.example4(n)
        p = np.column_stack((np.arange(1, len(p) + 1), p[:]))
        k = 3
        f = []
        bicritiria_cost = CoresetKSeg.bicriteria(p, k, f)
        bicritiria_cost2 = CoresetKSeg.bicriteria2(p, k)
        bicritiria_orig = CoresetKSeg.bicriteria_orig(p, k)
        print("Bicritiria estimate:\n"
              "{:<.2f} bi\n"
              "{:<.2f} bi2\n"
              "{:<.2f} bi_orig\n"
              "{:<.2f} f".format(bicritiria_cost, bicritiria_cost2, bicritiria_orig, sum(f)))
        real_cost = calc_cost_dividers(p, ksegment.k_segment(p, k))
        print("real cost: ", real_cost)
        self.assertGreaterEqual(real_cost, bicritiria_cost)
        self.assertGreaterEqual(real_cost, bicritiria_cost2)
        self.assertGreaterEqual(real_cost, bicritiria_orig)

    def test_compare_spark_shuffle_map_to_singlethread(self):
        points = load_csv_into_dataframe("/home/ge/k-segment/datasets/KO_no_date.csv").values
        points = np.column_stack((np.arange(1, len(points) + 1), points[:]))
        k = 5
        eps = 0.4
        coreset = CoresetKSeg.build_coreset(points, k, eps)
        coreset2 = CoresetKSeg.build_coreset(coreset, k, eps, True)
        # from pyspark import SparkContext, SparkConf
        # conf = SparkConf().setMaster('local[*]').setAppName('Test')
        # # Set scheduler to FAIR: http://spark.apache.org/docs/latest/job-scheduling.html#scheduling-within-an-application
        # conf.set('spark.scheduler.mode', 'FAIR')
        # sc = SparkContext(conf=conf)
        # points_rdd_1 = points_rdd(lambda x: CoresetKSeg.build_coreset_on_pyspark(x, k, eps))
        # visualize_2d(points, coreset, k, eps, show=True)
        # points_rdd = sc.parallelize(points, k).glom()
        # coresets_rdd_collected = points_rdd.map(lambda x: CoresetKSeg.build_coreset_on_pyspark(np.asarray(x), k, eps))\
        #     .reduce(print)
        # print(coresets_rdd_collected)
        # print(coresets_rdd_collected)
        # sc.stop()
        pass


if __name__ == '__main__':
    unittest.main()
