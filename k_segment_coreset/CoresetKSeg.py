import numpy as np
import math
import utils
import ksegment


class OneSegCoreset:
    def __init__(self, repPoints, weight, SVt):
        self.repPoints = repPoints
        self.weight = weight
        self.SVt = SVt


class SegmentCoreset:
    def __init__(self, coreset, g, b, e):
        self.C = coreset  # 1-segment coreset
        self.g = g  # best line
        self.b = b  # coreset beginning index
        self.e = e  # coreset ending index

    def __repr__(self):
        return "OneSegmentCoreset " + str(self.b) + "-" + str(self.e) + "\n" + str(self.C.repPoints) + "\n"


class CoresetKSeg(object):
    def __init__(self, k, eps, weights=None):
        self.k = k
        self.eps = eps
        self.coreset = None
        self.dividers = None
        self.iscoreset = False
        if weights is not None:
            self.iscoreset = True
            self.weights = weights

    def __len__(self):
        return len(self.coreset)

    def compute(self, data_points):
        data_points = np.column_stack((np.arange(1, len(data_points) + 1), data_points[:]))
        self.coreset = build_coreset(data_points, self.k, self.eps)
        self.dividers = ksegment.coreset_k_segment(self.coreset, self.k)
        return self.coreset, self.weights

# def bicriteria(points, k, is_coreset=False):
#     if len(points) <= (4 * k + 1):
#         return one_seg_cost(points, is_coreset)
#     m = int(math.floor(len(points) / (4 * k)))
#     i = 0
#     j = m
#     # one_seg_res will  hold segment starting index and result (squared distance sum)
#     one_seg_res = []
#     # partition to 4k segments and call 1-segment for each
#     while i < len(points):
#         partition_set = one_seg_cost(points[i:j], is_coreset)
#         one_seg_res.append((partition_set, int(i)))
#         i += m
#         j += m
#     # sort result
#     one_seg_res = sorted(one_seg_res, key=lambda res: res[0])
#     # res = the distances of the min k+1 segments
#     res = 0
#     # sum distances of k+1 min segments and make a list of point to delete from P to get P \ Q from the algorithm
#     rows_to_delete = []
#     for i in range(k + 1):
#         res += one_seg_res[i][0]
#         for j in range(m):
#             rows_to_delete.append(one_seg_res[i][1] + j)
#     points = np.delete(points, rows_to_delete, axis=0)
#     c = bicriteria(points, k, is_coreset)
#     if type(res) != type(c):
#         print c
#     return res + c


def bicriteria(points, k, f, mul=4, is_coreset=False):
    """
    :param points:      input dataset of points
    :param k:           number of segments
    :param is_coreset:
    :param f: float array
    :param mul: int
    :return:            cost c a
    """
    if len(points) < (mul * k + 1):
        # for p in points:
        #     f[p[0]] = 0
        return 0  # TODO changes
    chunk_size = int(math.floor(len(points) / (mul * k)))
    # one_seg_res will  hold segment starting index and result (squared distance sum)
    one_seg_res = []
    # partition to mul*k segments and call 1-segment for each
    for start_idx in range(0, len(points), chunk_size):
        partition_set = one_seg_cost(points[start_idx:start_idx+chunk_size], is_coreset)
        one_seg_res.append((partition_set, start_idx, int(points[start_idx][0])))
    # TODO: switch to max heap and test performance
    one_seg_res = sorted(one_seg_res, key=lambda one_res: one_res[0])
    # res = the distances of the min k+1 segments
    cost = 0
    # sum distances of k+1 min segments and make a list of points to delete from P to get P \ Q from the algo'
    rows_to_delete = []
    for start_idx in range(k + 1):
        cost += one_seg_res[start_idx][0]
        f[one_seg_res[start_idx][2]:one_seg_res[start_idx][2]+chunk_size] = [round(one_seg_res[start_idx][0], 2)] * chunk_size
        rows_to_delete += range(one_seg_res[start_idx][1], one_seg_res[start_idx][1] + chunk_size)
    points = np.delete(points, rows_to_delete, axis=0)
    return cost + bicriteria(points, k, f, mul, is_coreset)


def bicriteria2(points, k, mul=4, is_coreset=False):
    """
    :param points:      input dataset of points
    :param k:           number of segments
    :param mul: int
    :param is_coreset:
    :return:            cost c a
    """
    if len(points) <= (mul * k + 1):
        return 0
    m = int(math.floor(len(points) / (mul * k)))
    i = 0
    j = m
    # one_seg_res will  hold segment starting index and result (squred distance sum)
    one_seg_res = []
    # partition to 4k segments and call 1-segment for each
    while i < len(points):
        partition_set = one_seg_cost(points[i:j], is_coreset)
        # partition_set = bicriteria(points[i:j], k, is_coreset)
        one_seg_res.append((partition_set, int(i)))
        i += m
        j += m
    # sort result
    one_seg_res = sorted(one_seg_res, key=lambda res: res[0])
    # res = the distances of the min k+1 segments
    cost = 0
    # sum distances of k+1 min segments and make a list of points to delete from P to get P \ Q from the algo'
    rows_to_delete = []
    for i in range(k + 1):
        cost += one_seg_res[i][0]
        for j in range(m):
            rows_to_delete.append(one_seg_res[i][1] + j)
    points = np.delete(points, rows_to_delete, axis=0)
    return cost + bicriteria2(points, k, mul, is_coreset)


def BalancedPartition(P, a, bicritiriaEst, is_coreset=False):
    Q = []
    D = []
    points = P
    # add arbitrary item to list
    dimensions = points[0].C.repPoints.shape[1] if is_coreset else points.shape[1]
    if is_coreset:
        points.append(P[0])  # arbitrary coreset n+1
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
        if cost > bicritiriaEst \
                and (is_coreset or (len(Q) > dimensions + 1 and dimensions + 1 <= n - 1 - i)) \
                or i == n - 1:
            if is_coreset and len(Q) == 1:
                if i != n - 1:
                    D.append(Q[0])
                    Q = []
                continue
            T = Q[:-1]
            C = OneSegmentCorset(T, is_coreset)
            g = utils.calc_best_fit_line_polyfit(OneSegmentCorset(np.asarray(T), is_coreset).repPoints)
            if is_coreset:
                b = T[0].b
                e = T[-1].e
            else:
                b = T[0][0]     # signal index of first item in T
                e = T[-1][0]    # signal index of last item in T
            D.append(SegmentCoreset(C, g, b, e))
            Q = [Q[-1]]
    return D


def build_coreset(points, k, eps, is_coreset=False):
    f = [float] * (len(points) + 1)

    h = bicriteria(points, k, f, is_coreset=is_coreset)
    print("bicritiria estimate:", h)

    sigma = (eps ** 2 * h) / (100 * k * np.log2(len(points)))
    return BalancedPartition(points, eps, sigma, is_coreset)


def one_seg_cost(points, is_coreset=False):
    if is_coreset:
        one_segment_coreset = OneSegmentCorset(points, is_coreset)
        return utils.cost_best_fit_line_to_points(one_segment_coreset.repPoints, is_coreset) * one_segment_coreset.weight
    else:
        return utils.cost_best_fit_line_to_points(points, is_coreset)


def OneSegmentCorset(P, is_coreset=False):
    if len(P) < 2:
        return P[0].C
    if is_coreset:
        svt_to_stack = []
        for oneSegCoreset in P:
            svt_to_stack.append(oneSegCoreset.C.SVt)
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
    except:
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


def PiecewiseCoreset(n, eps):
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
