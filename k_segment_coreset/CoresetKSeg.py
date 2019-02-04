import numpy as np
import math
try:
    import utils_seg
    import ksegment
except ImportError:
    from k_segment_coreset import utils_seg
    from k_segment_coreset import ksegment
from typing import Union, List, Optional


class OneSegCoreset:
    __slots__ = ['repPoints', 'weight', 'SVt']

    def __init__(self, repPoints, weight, SVt):
        self.repPoints = repPoints
        self.weight = weight
        self.SVt = SVt


class SegmentCoreset:
    __slots__ = ['C', 'g', 'b', 'e']

    def __init__(self, coreset: OneSegCoreset, g: np.ndarray, b: int, e: int) -> None:
        self.C = coreset  # 1-segment coreset
        self.g = g  # best line
        self.b = b  # coreset beginning index
        self.e = e  # coreset ending index

    def __repr__(self):
        return "OneSegmentCoreset {} - {} {}".format(self.b, self.e, self.C.repPoints)

    def __str__(self):
        return "\nOneSegmentCoreset {} - {}\n{}".format(self.b, self.e, self.C.repPoints)


class CoresetKSeg(object):
    def __init__(self, k: int, eps: float, weights=None) -> None:
        self.k = k
        if eps < 0 or eps > 1:
            raise Exception('CoresetKSeg eps error value has to be 0 < eps <= 1')
        self.eps = eps
        self.is_coreset = False
        self.weights = weights
        self.k_eps_coreset = []

    def __call__(self, data):
        self.k_eps_coreset = self.compute_coreset(data, self.k, self.eps, self.f, self.is_coreset)
        return self.k_eps_coreset

    def compute(self, data_points: Union[np.ndarray, List[SegmentCoreset]]):
        if type(data_points) is not np.ndarray:
            self.is_coreset = True
        self.k_eps_coreset = self.compute_coreset(data_points, self.k, self.eps, is_coreset=self.is_coreset)
        return self.k_eps_coreset, compute_piecewise_coreset(len(self), self.eps)

    @staticmethod
    def compute_coreset(data, k: int, eps: float, f: list = None, is_coreset: bool = False) -> List[SegmentCoreset]:
        h = CoresetKSeg.compute_bicriteria(data, k, f, is_coreset=is_coreset)
        # sigma is calculated according to the formula in the paper
        try:
            sigma = (eps ** 2 * h) / (100 * k * np.log2(len(data)))
            return CoresetKSeg.balanced_partition(data, eps, sigma, is_coreset)
        except TypeError as e:
            print('in compute_coreset error: {}'.format(e))

    @staticmethod
    def compute_bicriteria(points: Union[np.ndarray, List[OneSegCoreset]],
                           k: int, f: list = None, mul: int = 4, is_coreset: bool = False) -> float:
        """
        this fucntion computes compute_bicriteria estimation of data complexity according to algorithm 2
        :param points:  input data points
        :param k:       number of segments to split the data into
        :param f:       TODO: future use for f* function
        :param mul:     not a magic number, proven value from paper, edit only for debug
        :param is_coreset: if True then input data is a coreset
        :return:    compute_bicriteria estimation
        """
        if len(points) <= (mul * k + 1):
            # TODO: f: = a 1 - segment mean of P
            return one_seg_cost(points, is_coreset)
        chunk_size = int(math.ceil(len(points) / (mul * k)))
        # one_seg_res will  hold segment starting index and result (squared distance sum)
        one_seg_res = []
        # we partition the signal into 4 k sub - intervals or:
        # partition to mul*k segments and call 1-segment for each
        for start_idx in range(0, len(points), chunk_size):
            partition_set = one_seg_cost(points[start_idx:start_idx + chunk_size], is_coreset)
            one_seg_res.append((partition_set, start_idx))
        # TODO: switch to max heap and test performance
        one_seg_res = sorted(one_seg_res, key=lambda one_res: one_res[0])
        # cost = the distances of the min k+1 segments
        cost = 0
        # sum distances of k+1 min segments and make a list of points to delete from P to get P \ Q from the algo
        rows_to_delete = []
        for start_idx in range(k + 1):
            cost += one_seg_res[start_idx][0]
            rows_to_delete += range(one_seg_res[start_idx][1], one_seg_res[start_idx][1] + chunk_size)
        points = np.delete(points, rows_to_delete, axis=0)
        return cost + CoresetKSeg.compute_bicriteria(points, k, f, mul, is_coreset)

    @staticmethod
    def balanced_partition(points: Union[np.ndarray, List[OneSegCoreset]], eps: float, bicritiria_est: float, is_coreset=False) -> List[SegmentCoreset]:
        Q = []
        D = []
        # add arbitrary item to list
        dimensions = points[0].C.repPoints.shape[1] if is_coreset else points.shape[1]
        if is_coreset:
            points.append(points[0])  # arbitrary coreset n+1
        else:
            points = np.vstack((points, np.zeros(dimensions)))  # arbitrary point n+1
        n = len(points)
        for i in range(0, n):
            Q.append(points[i])
            cost = one_seg_cost(np.asarray(Q), is_coreset)
            # if current number of points can be turned into a coreset - 3 conditions :
            # 1) cost passed threshold
            # 2) number of points to be packaged greater than dimensions + 1
            # 3) number of points left greater then dimensions + 1 (so they could be packaged later)
            if cost > bicritiria_est \
                    and (is_coreset or (len(Q) > dimensions + 1 and dimensions + 1 <= n - 1 - i)) \
                    or i == n - 1:
                if is_coreset and len(Q) == 1:
                    if i != n - 1:
                        D.append(Q[0])
                        Q = []
                    continue
                T = Q[:-1]
                C = compute_one_segment_coreset(T, is_coreset)
                g = utils_seg.calc_best_fit_line_polyfit(compute_one_segment_coreset(np.asarray(T), is_coreset).repPoints)
                if is_coreset:
                    b = T[0].b
                    e = T[-1].e
                else:
                    b = T[0][0]  # signal index of first item in T
                    e = T[-1][0]  # signal index of last item in T
                D.append(SegmentCoreset(C, g, int(b), int(e)))
                Q = [Q[-1]]
        return D

    def compute_dividers(self) -> np.ndarray:
        return ksegment.coreset_k_segment(self.k_eps_coreset, self.k)

    @staticmethod
    def add_index_col_to_data(data: np.ndarray) -> np.ndarray:
        return np.column_stack((np.arange(1, len(data) + 1), data[:]))

    @property
    def coreset(self):
        return self.k_eps_coreset

    def __len__(self):
        return len(self.k_eps_coreset)

    def __iter__(self):
        self.last = 0
        return self

    def __next__(self):
        self.last += 1
        if self.last >= len(self.coreset):
            raise StopIteration
        return self.coreset[self.last]

    def __repr__(self):
        return "KSegCoreset size = {}, for k = {}, eps = {:<2}".format(len(self.coreset), self.k, self.eps)

    def __str__(self):
        pass

    def __add__(self, other):
        pass


def one_seg_cost(points, is_coreset=False):
    if is_coreset:
        one_segment_coreset = compute_one_segment_coreset(points, is_coreset)
        return utils_seg.cost_best_fit_line_to_points(one_segment_coreset.repPoints, is_coreset) * one_segment_coreset.weight
    else:
        return utils_seg.cost_best_fit_line_to_points(points, is_coreset)


def compute_one_segment_coreset(P, is_coreset=False):
    if len(P) < 2:
        return P[0].C
    if is_coreset:
        svt_to_stack = []
        for one_seg_coreset in P:
            svt_to_stack.append(one_seg_coreset.C.SVt)
        X = np.vstack(svt_to_stack)
    else:
        # add 1's to the first column
        X = np.insert(P, 0, values=1, axis=1)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    # reshape S
    S = np.diag(s)
    # calculate SV
    SVt = np.dot(S, V)
    u = SVt[:, 0]   # u is leftmost column of SVt
    w = (np.linalg.norm(u) ** 2) / X.shape[1]
    q = np.identity(X.shape[1])     # q - temporary matrix to build an identity matrix with leftmost column - u
    try:
        q[:, 0] = u / np.linalg.norm(u)
    except Exception as e:
        print("exception: {}".format(e))
        print("iscoreset:", is_coreset, "P", P, "u:", u, "q:", q)
    Q = np.linalg.qr(q)[0]      # QR decomposition returns in Q what is requested
    if np.allclose(Q[:, 0], -q[:, 0]):
        Q = -Q
    # assert matrix is as expected
    assert (np.allclose(Q[:, 0], q[:, 0]))
    # calculate Y
    y = np.identity(X.shape[1])  # y - temporary matrix to build an identity matrix with leftmost column
    y_left_col = math.sqrt(w) / np.linalg.norm(u)
    y[:, 0] = y_left_col  # set y's first column to be sqrt of w divided by u's normal
    # compute Y with QR decompression - first column will not change - it is already normalized
    Y = np.linalg.qr(y)[0]
    if np.allclose(Y[:, 0], -y[:, 0]):
        Y = -Y
    # assert matrix is as expected
    assert (np.allclose(Y[:, 0], y[:, 0]))
    YQtSVt = np.dot(np.dot(Y, Q.T), SVt)
    YQtSVt /= math.sqrt(w)
    # set B to the d+1 rightmost columns
    B = YQtSVt[:, 1:]
    # return [B, w, SVt]
    return OneSegCoreset(repPoints=B, weight=w, SVt=SVt)


def compute_piecewise_coreset(n, eps):
    # TODO: provide proper s
    def s(index, points_number):
        return max(4.0 / float(index), 4.0 / (points_number - index + 1))
    eps = eps / np.log2(n)
    s_arr = [s(i, n) for i in range(1, n + 1)]
    t = sum(s_arr)
    B = []
    b_list = []
    W = np.zeros(n)
    for i in range(1, n + 1):
        b = math.ceil(sum(s_arr[0:i]) / (t * eps))
        if b not in b_list:
            B.append(i)
        b_list.append(b)
    for j in B:
        I = [i + 1 for i, b in enumerate(b_list) if b == b_list[j - 1]]
        W[j - 1] = (1. / s_arr[j - 1]) * sum([s_arr[i - 1] for i in I])
    return W
