import numpy as np
import utils
import ksegment
import Coreset
import unittest
import matplotlib.pyplot as plt
# import cProfile


def load_csv_file(path, n_rows=None):
    # rv = np.genfromtxt(path, delimiter=',')
    rv = np.loadtxt(path, float, delimiter=',', skiprows=1)
    if n_rows:
        rv = rv[:n_rows]
    return rv


def gen_synthetic_graph(n, k, dim=1, deviation=5):
    sizes_of_subsets = np.ceil(np.random.dirichlet(np.ones(k), size=1) * n)
    n = int(sum(sum(sizes_of_subsets)))
    data = np.zeros(shape=(n, dim + 1))
    # data = np.zeros(shape=(n, dim))
    data[:,0] = np.arange(0, n)
    start = 0
    up = True
    for s in sizes_of_subsets:
        for i in s:
            stop = int(start + i)
            # print("start {}, stop {}".format(start, stop))
            for d in range(dim):
                # if (-1) ** np.random.randint(1,3, size=1)[0] > 0:
                if up:
                    data[start:stop, dim] = np.arange(start, stop, 1) + np.random.normal(0, deviation, stop - start)
                else:
                    data[start:stop, dim] = \
                        abs(np.arange(start, start - (stop - start), -1)) + np.random.normal(0, deviation, stop - start)
            start = stop
            up = not up
    return data


class KSegmentTest(unittest.TestCase):
    # cProfile.run('re.compile("test_coreset_merging")')
    def test_foo(self):
        d = gen_synthetic_graph(110, 5)
        plt.scatter(d[:, 0], d[:, 1], s=3)
        # plt.show()


    def test_basic_demo_synth(self, n=200, k=11, epsilon=0.3):
        # k = 11
        # epsilon = 0.2
        # n = 200
        data = gen_synthetic_graph(n, k, 1)[:, 1:]
        p = np.c_[np.mgrid[1:len(data) + 1], data]

        coreset = Coreset.build_coreset(p, k, epsilon)
        dividers = ksegment.coreset_k_segment(coreset, k)
        utils.visualize_2d(p, dividers, len(coreset),
                           show=False
                           )     # Uncomment to see results

    def test_loop(self):
        n = 800
        k = 10
        for i in range(100, n, 400):
            for j in range(3, k, 3):
                self.test_basic_demo_synth(n=i, k=j)

    def test_bicriteria2(self):
        dim = 1
        n = 12
        k = 2
        n, m = 4 * k + 2,50
        data = np.arange(n * m).reshape((n, m))
        p = np.c_[np.mgrid[1:data.shape[0] + 1], data]
        print("test input data\n{}".format(p))

        b = Coreset.bicriteria(data, k)
        b2 = Coreset.bicriteria2(data, k)

        print(b, b2)