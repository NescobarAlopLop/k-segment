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


def generate_input_file(n):
    data = example1(n)
    np.savetxt('input.csv', data, '%.5f', delimiter=' ')


def gen_synthetic_graph(n, k, dim=1, deviation=5, h=100):
    sizes_of_subsets = np.ceil(np.random.dirichlet(np.ones(k), size=1) * n).astype(int)[0]
    n = int(sum(sizes_of_subsets))
    data = np.zeros(shape=(n, dim + 1))
    # if (-1) ** np.random.randint(1,3, size=1)[0] > 0:
    data[:, 0] = np.arange(n)
    stop_idx = 0
    stop_val = 0
    for d in range(dim):
        for s in sizes_of_subsets:
            start_idx = stop_idx
            stop_idx += s
            start_val = stop_val
            stop_val = np.random.randint(1, h, size=1)[0]
            data[start_idx:stop_idx, dim] = np.linspace(start_val, stop_val, s) +\
                                            np.random.normal(0, deviation, stop_idx - start_idx)
    return data


class KSegmentTest(unittest.TestCase):
    # cProfile.run('re.compile("test_coreset_merging")')
    def test_foo(self):
        d = gen_synthetic_graph(n=700, k=7, deviation=3)
        plt.scatter(d[:, 0], d[:, 1], s=3)
        plt.show()

    def test_basic_demo(self):
        # generate points
        k = 3
        epsilon = 0.5
        n = 600
        generate_input_file(n)
        data = np.genfromtxt("input.csv", delimiter=" ")
        p = np.c_[np.mgrid[1:n + 1], data]

        coreset = Coreset.build_coreset(p, k, epsilon)
        dividers = ksegment.coreset_k_segment(coreset, k)
        utils.visualize_3d(p, dividers)     # Uncomment to see results


    # def test_KO(self):
    #     # generate points
    #     k = 3
    #     epsilon = 0.5
    #     n = 600
    #     generate_input_file(n)
    #     data = load_csv_file("/home/ge/k-segment/spark_coreset-master/datasets/KO_no_date.csv", n_rows=200)
    #     p = np.c_[np.mgrid[1:n + 1], data.T]
    #
    #     coreset = Coreset.build_coreset(p, k, epsilon)
    #     dividers = ksegment.coreset_k_segment(coreset, k)
    #     utils.visualize_2d(p, dividers, len(coreset))     # Uncomment to see results

    def test_basic_demo_synth(self, n=100, k=5, epsilon=0.1, show=False):
        # k = 11
        # epsilon = 0.2
        # n = 200
        data = gen_synthetic_graph(n, k, 1)[:, 1:]
        p = np.c_[np.mgrid[1:len(data) + 1], data]

        coreset = Coreset.build_coreset(p, k, epsilon)
        dividers = ksegment.coreset_k_segment(coreset, k)
        utils.visualize_2d(p, dividers, len(coreset), show=show)     # Uncomment to see results

    def test_loop(self):
        n = 1200
        k = 10
        for i in range(500, n, 200):
            for j in range(3, k, 2):
                self.test_basic_demo_synth(n=i, k=j)

    def test_bicritiria(self):
        n = 300
        k = 4
        data = example1(n)

        p = np.c_[np.mgrid[1:n + 1], data]

        bicritiria_cost2 = Coreset.bicriteria2(p, k)
        print("Bicritiria estimate: ", bicritiria_cost2)
        real_cost = utils.calc_cost_dividers(p, ksegment.k_segment(p, k))
        print("real cost: ", real_cost)
        self.assertGreaterEqual(real_cost, bicritiria_cost2)


    def test_calc_best_fit_line_weighted(self):
        data = np.array([[1, 3.2627812, -3.1364346],
                         [2, 3.4707861, -3.28776192],
                         [3, 3.67879099, -3.43908923]])
        w = [1.0, 1.0, 1.0]
        best_fit_line = utils.calc_best_fit_line_polyfit(data, w)
        print(best_fit_line)

    def test_calc_sqr_dist_weighted(self):
        data = np.array([[1, 1],
                         [2, 3],
                         [3, 4],
                         [4, 4]])
        w = [1, 0, 0, 1]
        best_fit_line_cost_weighted = utils.best_fit_line_cost_weighted(data, w)
        print(best_fit_line_cost_weighted)

    def test_Piecewise_coreset(self):
        n = 600
        w = Coreset.PiecewiseCoreset(n, 0.01)
        self.assertAlmostEqual(n, sum(w), delta=n/100)


def random_data(n, dimension):
    return np.random.random_integers(0, 100, (n, dimension))


# 3 straight lines with noise
# choose N that divides by 6
def example1(n):
    x1 = np.mgrid[1:9:2 * n / 6j]
    y1 = np.mgrid[-5:3:2 * n / 6j]
    x2 = np.mgrid[23:90:n / 2j]
    y2 = np.mgrid[43:0:n / 2j]
    x3 = np.mgrid[80:60:n / 6j]
    y3 = np.mgrid[90:100:n / 6j]

    x = np.r_[x1, x2, x3]
    y = np.r_[y1, y2, y3]
    x += np.random.normal(size=x.shape) * 3
    y += np.random.normal(size=y.shape) * 3
    return np.c_[x, y]


# random
def example2():
    x1 = np.mgrid[1:9:100j]
    y1 = np.mgrid[-5:3:100j]
    x1 += np.random.normal(size=x1.shape) * 4
    return np.c_[x1, y1]
