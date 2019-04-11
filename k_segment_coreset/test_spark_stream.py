import warnings

import numpy as np
from pyspark import SparkContext

import ksegment
from CoresetKSeg import CoresetKSeg
from utils_seg import gen_synthetic_graph

warnings.filterwarnings("ignore")


def main():

    num_of_points = 500
    chunk_size = 100
    k = 5
    eps = 0.4
    points = gen_synthetic_graph(n=num_of_points + 1, k=k, dim=1, deviation=0.01, max_diff=3)
    points = np.column_stack((np.arange(1, len(points) + 1), points[:]))
    aggregated_for_rdd = []

    for i in range(0, len(points), chunk_size):
        aggregated_for_rdd.append(points[i:i + chunk_size])

    sc = SparkContext()
    # data = sc.parallelize(aggregated_for_rdd)
    data = sc.parallelize(aggregated_for_rdd, 4)

    def func(x):
        print(x)
    all_coresets = data.map(lambda x: CoresetKSeg.compute_coreset(x, k, eps)).collect()
    # all_coresets = data.mapPartitions(lambda x: func(x))
    sc.stop()
    # tmp = []
    # for t in all_coresets:
    #     tmp += t
    # dividers = ksegment.coreset_k_segment(tmp, k)
    # print(all_coresets)
    # print(dividers)


if __name__ == "__main__":
    main()
