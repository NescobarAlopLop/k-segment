import numpy as np
import mpl_toolkits.mplot3d as m3d
import matplotlib.pyplot as plt


def best_fit_line_cost(points, is_coreset=False):
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
        coordinates.append(line[0, i] * x + line[1, i])
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
        segment = points[line_start_arr_index:line_end_arr_index, :]
        best_fit_line = calc_best_fit_line_polyfit(segment)
        line_pts_list.append([pt_on_line(dividers[i], best_fit_line),
                              pt_on_line(dividers[i + 1] - (1 if i != len(dividers) - 2 else 0), best_fit_line)])
        all_sgmnt_sqrd_dist_sum += sqrd_dist_sum(segment, best_fit_line)

    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*points.T)
    for line in line_pts_list:
        lint_pts_arr = np.asarray(line)
        ax.plot3D(*lint_pts_arr.T)

    ax.set_xlabel('time axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('x2 axis')

    plt.show()


def is_unitary(m):
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))
