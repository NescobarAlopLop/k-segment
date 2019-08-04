import cProfile
import os
import sys
from datetime import datetime
from io import StringIO
from pstats import Stats
from time import time
from typing import Union, Optional

import imageio as io
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np

# from utils_seg import best_fit_line_and_cost

cp = cProfile.Profile()


class KsegmentOptException(Exception):
    pass


def load_img_as_greyscale(path: str):
    return imageio.imread(path, as_gray=False, pilmode="RGB")


def binarize_img(img: Union[np.ndarray, io.core.util.Array], threshold: int = 140) -> Union[
    np.ndarray, io.core.util.Array]:
    return 1 * (img > threshold)


def mean_squared_distance(arr):
    """
    Computes the MSE of arr points to mean(arr)
    :param arr: array of points in any dimension
    :return: MSE between each point and mean of all points
    """
    mse = 0.0
    for p in arr:
        mse += np.linalg.norm(np.mean(arr) - p) ** 2
    return mse


def one_center_cost(arr):
    """
    Computes the OCC of arr points
    :param arr: array of points in any dimension
    :return: OCC between each point
    """
    bin_arr = binarize_img(arr)
    tuples = np.where(bin_arr == 1)
    first_point = np.ndarray(tuples[0][0], tuples[1][0])
    # for i, j in zip(tuples[0], tuples[1]):
    #     np.linalg.norm(first_point - (i,j))
    np.linalg.norm(first_point - np.column_stack(tuples[:]), axis=)


class KMean(object):
    def __init__(self, mat, k, cost_func=mean_squared_distance):
        self.mat = mat
        self.mat_rows, self.mat_cols = mat.shape[0], mat.shape[1]
        if 1 <= k <= min(self.mat_rows, self.mat_cols):
            self.k = k
        else:
            raise KsegmentOptException("k={}, but has to be: {} < k <= {}, defined by shape of input"
                                       .format(k, 1, min(self.mat_rows, self.mat_cols)))
        self.cost_func = cost_func
        dt = np.dtype([('weight', np.float32), ('path', np.uint32, (k + 1,))])
        self.d_ij_path = np.zeros((self.mat_rows + 1, self.mat_rows + 1), dtype=dt)
        self.horizontal_dividers = []
        self.total_weight = 0.0
        # self.q = Queue(maxsize=0)

    def __repr__(self):
        return '{}'.format(self.mat)

    def best_sum_of_variances(self):
        if self.k == 1:
            self.total_weight = self.cost_func(self.mat)
            self.horizontal_dividers = [0, self.mat_rows]
        else:
            for x, y in ((i, j) for i in range(0, self.d_ij_path.shape[0])
                         for j in range(i + 1, self.d_ij_path.shape[0])):
                self.d_ij_path[x, y] = self.best_segment(self.compute_distance_matrix(x, y))
            self.total_weight, self.horizontal_dividers = self.best_segment(self.d_ij_path)
        return self.total_weight, self.horizontal_dividers

    def best_segment(self, d):
        """
            Input: distance matrix  D \in [0,\infty)^{n\times n} between nodes
            Output: Value F(k,n) of k shortest path from node 1 to node n
                    and the dividers indices that provide this cost
            1. F(1,1:n)=D(1,1:n)
            2. For m=2 to k-1
            2.1 -For i=1 to n
            2.2 -- F(m,i)= \min_j (F(m-1,j)+D(j,i))

            3. Return path(dividers), \min_j F(k-1,j)+D(j,n), iterations, F
        """
        n = d.shape[1]
        _F = d.copy()  # 1. F(1,1:n)=D(1,1:n)
        _P = np.zeros(shape=(self.k, n), dtype=int)
        try:
            # 2. For m=2 to k-1   k last one is not used so k-1
            # 2.1 -For i=1 to n
            # 2.2 -- F(m,i)= \min_j (F(m-1,j)+D(j,i))
            for z, x in ((m, i) for m in range(1, self.k) for i in range(2, n)):
                # index_weight = {}
                # for y in range(z, x):
                #     index_weight[y] = _F[z - 1, y]['weight'] + d[y, x]['weight']
                # if len(index_weight) > 0:
                #     _P[z, x], _F[z, x] = min(index_weight.items(), key=lambda val: val[1])
                # 2 lines after the comment interchange 5 lines above it
                # and make the code run slower, although less function calls
                min_weight = _F[z - 1:z, :]['weight'] + d[:, x:x + 1]['weight'].flatten()
                _P[z, x], _F[z, x] = np.argmin(min_weight), min_weight.min()

        except IndexError:
            assert KsegmentOptException('index out of range')

        index_weight = {}
        for j in range(self.k - 1, n - 1):  # 3. Return \min_j F(k-1,j)+D(j,n)
            index_weight[j] = _F[self.k - 1 - 1, j]['weight'] + d[j, n - 1]['weight']
        _P[self.k - 1, n - 1] = min(index_weight, key=index_weight.get)
        # 1 line under the comment interchange 3 lines above the comment and makes code run slower
        # _P[self.k - 1, n - 1] = \
        # np.argmin(_F[self.k - 1 - 1:self.k - 1, self.k - 1:n - 1]['weight'] \
        # + d[self.k - 1:n - 1, n - 1:n]['weight'].flatten())
        # cp.disable()
        # in total profiling: numpy way: 22525 function calls in 0.024 seconds; 112494 function calls in 0.125 seconds
        #                    for loop way: 25568 function calls in 0.006 seconds; 239538 function calls in 0.057 seconds
        return _F[self.k - 1 - 1, _P[self.k - 1, n - 1]]['weight'] + d[_P[self.k - 1, n - 1], n - 1]['weight'], \
               self.compute_path(_P, n)

    def compute_path(self, p, n):
        goal = n - 1
        list_len = self.k + 1
        shortest_path = [0 for _ in range(list_len)]
        for i in range(0, self.k):
            shortest_path[self.k - i] = goal
            goal = p[self.k - i - 1, goal]
        return shortest_path

    def compute_distance_matrix(self, i, j):
        dt = np.dtype([('weight', np.float32)])
        d_ab = np.zeros((self.mat_cols + 1, self.mat_cols + 1), dtype=dt)
        # no need to calc jump of 1, since distance of one element from itself is zero
        for x, y in ((a, b) for a in range(self.mat_cols + 1) for b in range(a + 1, self.mat_cols + 1)):
            d_ab[x, y] = self.cost_func(self.mat[i:j, x:y])

        return d_ab


def timer(func):
    """
    timer func used as decorator to check runtime
    :param func: any params required by a function it is called on
    :return: function it was called on
    """

    def f(*args, **kwargs):
        time_before = time()
        rv = func(*args, **kwargs)
        time_after = time()
        print("elapsed: {}".format(time_after - time_before))
        return rv

    return f


def plot_results(w_class, show_fig=False, img_path: str = None):
    offset = 0.0
    fig, ax = plt.subplots()

    ax.set_ylim(bottom=-offset, top=w_class.mat_rows - offset)
    ax.set_xlim(left=-offset, right=w_class.mat_cols - offset)

    if type(img_path) is str:
        img = imageio.imread(img_path)
        ax.imshow(img, extent=[-offset, w_class.mat_cols + offset, - offset, w_class.mat_rows + offset])
    else:
        ax.imshow(w_class.mat, extent=[-offset, w_class.mat_cols + offset, - offset, w_class.mat_rows + offset])

    plt.xticks([x for x in range(0, w_class.mat_cols, w_class.k)])
    plt.yticks([x for x in range(0, w_class.mat_rows, w_class.k)])
    # plt.gca().invert_yaxis()

    horizontal_lines = sorted(w_class.horizontal_dividers)
    for divider in horizontal_lines:
        line = lines.Line2D([0, w_class.mat_cols], [w_class.mat_rows - divider, w_class.mat_rows - divider],
                            lw=2, color='r', axes=ax)
        ax.add_line(line)

    for l, m in zip(horizontal_lines[:-1], horizontal_lines[1:]):
        for div in w_class.d_ij_path[l, m]['path']:
            vertical_separation = lines.Line2D([div, div], [w_class.mat_rows - l, w_class.mat_rows - m],
                                               lw=2, color='r', linestyle='--')
            ax.add_line(vertical_separation)

    plt.grid(False)
    plt.title('optimal segmentation, k=' + str(w_class.k))
    # plt.text(0.5, 0.5, str(w_class.k), horizontalalignment='center',
    #          verticalalignment = 'center', transform = ax.transAxes)

    basename = "log_im"
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    k = "k=" + str(w_class.k)
    filename = "_".join([basename, d, k])  # e.g. 'log_im_120508_171442'
    output_fig_path = os.path.join(os.path.dirname(__file__), 'output_figures')
    if not os.path.exists(output_fig_path):
        os.makedirs(output_fig_path)

    output_fig_path = os.path.join(output_fig_path, filename)
    print("saving in {}".format(output_fig_path))
    plt.savefig(output_fig_path)
    if show_fig:
        plt.show()
    return output_fig_path


# TODO: save figure in separate func:
# def save_plot_as_image(fig, path=''):


@timer
def main(in_data=None, k=4, show_fig=True):
    data = np.array([
        [10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50, 50],
        [10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50, 50],
        [40, 40, 40, 40, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        [40, 40, 40, 40, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        [40, 40, 40, 40, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        [40, 40, 40, 40, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 30, 30, 30, 40, 40, 40, 40, 40],
        [50, 50, 50, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 30, 30, 30, 40, 40, 40, 40, 40],
        [50, 50, 50, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 30, 30, 30, 40, 40, 40, 40, 40],
        [20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 50, 30, 30, 30],
        [20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 50, 30, 30, 30],
        [20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 50, 30, 30, 30],
        [20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 50, 30, 30, 30],
        [20, 20, 20, 20, 40, 40, 40, 40, 40, 40, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 50, 30, 30, 30],
    ]) * 5

    # if type(in_data) is type('str'):
    #     data = imageio.imread(in_data)
    #     data = np.array(data).mean(axis=2)
    # if type(in_data) is type(data):
    #     data = in_data

    w_class = KMean(mat=data, k=k)
    w_class.best_sum_of_variances()

    print('class mat weight', w_class.total_weight, w_class.horizontal_dividers)

    path_to_fig = plot_results(w_class, show_fig=show_fig, img_path=in_data)
    return path_to_fig


if __name__ == '__main__':
    cp.enable()

    file_path = None
    k = None
    print(sys.argv)
    if len(sys.argv) >= 3:
        file_path = sys.argv[1]
        k = int(sys.argv[2])
        out_fig = main(in_data=file_path, k=k)

    elif len(sys.argv) >= 2:
        file_path = sys.argv[1]
        out_fig = main(in_data=file_path)

    else:
        out_fig = main()

    cp.disable()

    s = StringIO()
    ps = Stats(cp, stream=s).sort_stats('tottime')
    ps.print_stats()

    pre, ext = os.path.splitext(out_fig)
    with open(pre + '.txt', 'w+') as f:
        f.write(s.getvalue())

    cp.dump_stats(pre)
