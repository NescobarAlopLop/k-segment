import numpy as np
import utils
import ksegment
import Coreset
import unittest
import matplotlib.pyplot as plt
import pandas as pd
# import cProfile


def load_csv_file(path):
    df = pd.read_csv(
        filepath_or_buffer=path,
        skiprows=1,
        encoding='utf8',
        sep=',',
        engine='python'
    )
    return df.values


def gen_synthetic_graph(n, k, dim=1, deviation=20, max_diff=40):
    """
    generates synthetic graph with noise
    :param n: number of data points
    :param k: number of segments
    :param dim: number of dimensions per point
    :param deviation: deviation of noise
    :param max_diff: maximum difference in value between first and last point of a segment
    :return: ndarrray of n data points of dimension d, visually split into k segments
    """
    sizes_of_subsets = np.ceil(np.random.dirichlet(np.ones(k), size=1) * n).astype(int)[0]
    n = int(sum(sizes_of_subsets))
    data = np.zeros(shape=(n, dim))
    # if (-1) ** np.random.randint(1,3, size=1)[0] > 0:
    for d in range(dim):
        stop_idx = 0
        stop_val = 0
        for size in sizes_of_subsets:
            start_idx = stop_idx
            stop_idx += size
            start_val = stop_val
            stop_val = np.random.randint(1, max_diff, size=1)[0]
            line = np.linspace(start_val, stop_val, size)
            noise = np.random.normal(0, deviation, stop_idx - start_idx)
            data[start_idx:stop_idx, d] = line + noise

    return data


class KSegmentTest(unittest.TestCase):
    # cProfile.run('re.compile("test_coreset_merging")')
    @staticmethod
    def test_foo():
        d = gen_synthetic_graph(n=400, k=7, deviation=1, max_diff=15)
        plt.scatter(np.arange(len(d)), d, s=3)
        plt.show()

    def test_from_file(self, path=None, n=None, b=0, k=3, eps=0.2, show=False):
        if not path:
            data = gen_synthetic_graph(n=200, k=k, dim=2, deviation=2, max_diff=30)
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
        coreset = Coreset.build_coreset(p, k, eps)
        dividers = ksegment.coreset_k_segment(coreset, k)
        # pass data input as 2d, first col is time, 2nd is average on all data cols
        p = np.column_stack((p[:, 0], np.linalg.norm(p[:, 1:], axis=1)))

        utils.visualize_2d(p, dividers, len(coreset), show=show)

    def test_KO(self):
        self.test_from_file("../datasets/KO_no_date.csv", n=600, b=940, k=8, show=False)

    def test_chunk_num_1(self):
        self.test_from_file("../datasets/chunk_num_1.csv", n=100, b=100, show=False)

    def test_basic_demo_synth(self):
        self.test_from_file(show=True)

    def test_loop(self):
        n = 1200
        k = 10
        for i in range(500, n, 200):
            for j in range(3, k, 2):
                self.test_from_file(n=i, k=j)
