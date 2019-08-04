from typing import Union
import imageio as io
import numpy as np


def binarize_img(img: Union[np.ndarray, io.core.util.Array], threshold: int = 140) -> Union[
    np.ndarray, io.core.util.Array]:
    return 1 * (img > threshold)


def get_D(points):
    return np.linalg.norm(points[:, np.newaxis] - points, axis=2)


def calc_cost(arr, k, partition):
    my_partition = partition[0]


def one_center_cost(arr):
    """
    Computes the OCC of arr points
    :param arr: array of points in any dimension
    :return: OCC between each point
    """
    bin_arr = binarize_img(arr)
    points = np.column_stack(np.where(bin_arr == 1)[:])
    first_point = np.array([points[0]])
    # for i, j in zip(tuples[0], tuples[1]):
    #     np.linalg.norm(first_point - (i,j))
    norms = np.linalg.norm(first_point - points, axis=1)
    index_of_farthest_point = np.where(norms == max(norms))
    farthest_point = np.array([points[index_of_farthest_point]])
    return max(norms)


def main():
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
    bin_arr = binarize_img(data)
    points = np.column_stack(np.where(bin_arr == 1)[:])
    distances = get_D(points)


if __name__ == "__main__":
    main()
