import numpy as np
import mpl_toolkits.mplot3d as m3d
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


def cost_best_fit_line_to_points(points, is_coreset=False):
    best_fit_line = calc_best_fit_line_polyfit(points, is_coreset)
    return sqrd_dist_sum(points, best_fit_line)


def best_fit_line_cost_weighted(points, weight, is_coreset=False):
    best_fit_line = calc_best_fit_line_polyfit(points, weight, is_coreset)
    return sqrd_dist_sum_weighted(points, best_fit_line, weight)


def calc_best_fit_line(points):
    """
    calc_best_fit_line -
        input - set of points
        return- matrix [dX2] such as that for each column C[i] -
            C[i,0] is the slope
            C[i,1] is the intercept with the i-th dimensional axis
    """
    try:
        n = len(points)
        time_array = points[:, 0]
        ones = np.vstack([time_array, np.ones(n)]).T
        data = points[:, 1:]
        return np.linalg.lstsq(ones, data, rcond=None)[0]
    except Exception as e:
        print("error in calc_best_fit_line {}\nerror: {}".format(points, e))


def calc_best_fit_line_polyfit(points, weight=False, is_coreset=False):
    if type(weight) == bool:
        weight = [1] * len(points)
        if weight:
            is_coreset = True
    try:
        time_array = points[:, 0]
        data = points[:, 1:]
        return np.polyfit(time_array, data, 1, w=weight)
    except Exception as e:
        print("error in calc_best_fit_line polyfit \nis coreset: {}\nerror: {}"
              .format(is_coreset.__str__(), e))


def sqrd_dist_sum(points, line):
    try:
        time_array = points[:, 0]
        tmp = np.vstack([time_array, np.ones(len(time_array))]).T
        data = points[:, 1:]
        projected_points = np.dot(tmp, line)
        return ((projected_points - data) ** 2).sum(axis=None)
    except Exception as e:
        print("error in sqrd_dist_sum: {}".format(e))


def sqrd_dist_sum_weighted(points, line, w):
    try:
        time_array = points[:, 0]
        ones = np.vstack([time_array, np.ones(len(time_array))]).T
        data = points[:, 1:]
        projected_points = np.dot(ones, line)
        norm_vector = np.apply_along_axis(np.linalg.norm, axis=1, arr=data - projected_points)
        squared_norm_distances = np.square(norm_vector)
        return sum(squared_norm_distances * (w ** 2))
    except Exception as e:
        print("error in sqrd_dist_sum: {}".format(e))


def pt_on_line(x, line):
    coordinates = [x]
    for i in range(len(line[0])):
        coordinates.append(line[0, int(i)] * x + line[1, int(i)])
    return coordinates


def calc_cost_dividers(points, dividers):
    cost = 0.0
    for i in range(len(dividers) - 1):
        segment = points[int(dividers[i]) - 1: int(dividers[i + 1]), :]
        cost += sqrd_dist_sum(segment, calc_best_fit_line_polyfit(segment))
    return cost


def lines_from_dividers(points, dividers):
    lines = []
    for i in range(len(dividers) - 1):
        segment = points[int(dividers[i]) - 1:int(dividers[i + 1]), :]
        lines.append(calc_best_fit_line_polyfit(segment))
    return lines


def visualize_3d(points, dividers):
    line_pts_list = []
    all_sgmnt_sqrd_dist_sum = 0
    for i in range(len(dividers) - 1):
        line_start_arr_index = dividers[i] - 1
        line_end_arr_index = dividers[i + 1] - 1 if i != len(dividers) - 2 else dividers[i + 1]
        segment = points[int(line_start_arr_index):int(line_end_arr_index), :]
        best_fit_line = calc_best_fit_line_polyfit(segment)
        line_pts_list.append([pt_on_line(dividers[i], best_fit_line),
                              pt_on_line(dividers[i + 1] - (1 if i != len(dividers) - 2 else 0), best_fit_line)])
        all_sgmnt_sqrd_dist_sum += sqrd_dist_sum(segment, best_fit_line)

    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*points.T)
    for line in line_pts_list:
        lint_pts_arr = np.asarray(line)
        ax.plot3D(*lint_pts_arr.T, label='line ')

    ax.set_xlabel('time axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('x2 axis')

    plt.show()


# def visualize_2d(points, dividers, coreset_size, coreset_points, show=False):
def visualize_2d(points, coreset, k, eps, show=False):
    import ksegment
    line_pts_list = []
    all_sgmnt_sqrd_dist_sum = 0
    dividers = ksegment.coreset_k_segment(coreset, k)
    for i in range(len(dividers) - 1):
        line_start_arr_index = dividers[i] - 1
        line_end_arr_index = dividers[i + 1] - 1 if i != len(dividers) - 2 else dividers[i + 1]
        segment = points[int(line_start_arr_index):int(line_end_arr_index), :]
        best_fit_line = calc_best_fit_line_polyfit(segment)
        line_pts_list.append([pt_on_line(dividers[i], best_fit_line),
                              pt_on_line(dividers[i + 1] - (1 if i != len(dividers) - 2 else 0), best_fit_line)])
        all_sgmnt_sqrd_dist_sum += sqrd_dist_sum(segment, best_fit_line)

    plt.figure(figsize=(19, 9))
    plt.scatter(points[:, 0], points[:, 1], s=3)
    coreset_points = ksegment.get_coreset_points(coreset)
    plt.scatter(coreset_points[:, 0], coreset_points[:, 1], s=20, c='r')
    # i = 0
    # for c in coreset:
    #     line_pts_array = np.asarray(c.g)
    #     plt.plot(c.g, label='[{}] b = {}, e = {}'.format(i, c.b, c.e))
    #     # plt.plot(*line_pts_array.T, label='[{}] b = {}, e = {}'.format(i, c.b, c.e))
    #     i += 1
    i = 0
    for line in line_pts_list:
        lint_pts_arr = np.asarray(line)
        plt.plot(*lint_pts_arr.T, label=str(i))
        i += 1
    plt.suptitle('data size {}, coreset size {}, k = {}, error = {}% mse for all points = {:.3f}'
                 .format(len(points), len(coreset_points), len(line_pts_list), eps * 100, all_sgmnt_sqrd_dist_sum))
    plt.legend()
    print("saving image: {:%Y_%m_%d_%s}.png".format(datetime.now()))
    plt.savefig("results/{:%Y_%m_%d_%s}".format(datetime.now()))
    if show:
        plt.show()
    plt.clf()


def is_unitary(m):
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))


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


def load_csv_file(path):
    df = pd.read_csv(
        filepath_or_buffer=path,
        skiprows=1,
        encoding='utf8',
        sep=',',
        engine='python'
    )
    return df.values
