import unittest
import imageio
import numpy as np
from k_segment_implementation.bicriteria import bicriteria
from utils_seg import best_fit_line_and_cost, calc_best_fit_line_polyfit
from Dividers import get_dividers_point_pairs_for_drawing
import matplotlib.pyplot as plt


class UtilsTest(unittest.TestCase):
    @staticmethod
    def test_bicriteria_one_dim_data():
        k = 2
        depth = 1
        data1 = np.asarray([
            10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13
            ])[:, np.newaxis]
        _, _, divs = bicriteria(data1, k, depth, best_fit_line_and_cost, calc_best_fit_line_polyfit)
        points = get_dividers_point_pairs_for_drawing(divs)
        plt.figure()  # (figsize=(16, 9), dpi=200)
        plt.grid(True)

        for line in points:
            print(line)
            x = [line[0][0], line[1][0]]
            y = [line[0][1], line[1][1]]
            plt.plot(x,y)
        plt.show()

    @staticmethod
    def test_bicriteria_two_dim_data():
        k = 2
        depth = 2
        data2 = [[10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 [10, 13, 10, 16, 10, 10, 10, 12, 10, 15, 10, 12, 10, 10, 10, 13, 10, 10, 10, 11, 10, 11, 10, 14],
                 [10, 13, 10, 16, 10, 10, 10, 12, 10, 15, 10, 12, 10, 10, 10, 13, 10, 10, 10, 11, 10, 11, 10, 14],
                 [10, 13, 10, 16, 10, 10, 10, 12, 10, 15, 10, 12, 10, 10, 10, 13, 10, 10, 10, 11, 10, 11, 10, 14],
                 [10, 13, 10, 16, 10, 10, 10, 12, 10, 15, 10, 12, 10, 10, 10, 13, 10, 10, 10, 11, 10, 11, 10, 14],
                 [10, 10, 10, 13, 10, 15, 10, 12, 10, 10, 10, 12, 10, 10, 10, 11, 10, 11, 10, 14, 10, 13, 10, 16],
                 [10, 10, 10, 13, 10, 15, 10, 12, 10, 10, 10, 12, 10, 10, 10, 11, 10, 11, 10, 14, 10, 13, 10, 16],
                 [10, 10, 10, 13, 10, 15, 10, 12, 10, 10, 10, 12, 10, 10, 10, 11, 10, 11, 10, 14, 10, 13, 10, 16],
                 [10, 10, 10, 13, 10, 15, 10, 12, 10, 10, 10, 12, 10, 10, 10, 11, 10, 11, 10, 14, 10, 13, 10, 16],
                 [10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 10, 10, 11, 10, 11, 10, 14, 10, 10, 10, 12, 10, 15, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 [10, 11, 10, 14, 10, 10, 10, 11, 10, 10, 10, 12, 10, 13, 10, 16, 10, 10, 10, 13, 10, 15, 10, 12],
                 ]
        data2 = np.array(data2)

        _, _, divs = bicriteria(data2, k, depth, best_fit_line_and_cost, calc_best_fit_line_polyfit)
        points = get_dividers_point_pairs_for_drawing(divs)
        plt.figure()  # (figsize=(16, 9), dpi=200)
        plt.grid(True)

        plt.imshow(data2)
        for line in points:
            # print(line)
            x = [line[0][0], line[1][0]]
            y = [line[0][1], line[1][1]]
            plt.plot(x,y, linewidth=4.4, color='red')
        plt.show()

    @staticmethod
    def test_with_img():
        img_file = imageio.imread('/home/ge/k-segment/k_segment_implementation/colour_grid.png')
        img_file = np.mean(img_file, axis=2)
        k = 2
        depth = 2

        data2 = np.asarray(img_file)

        _, _, divs = bicriteria(data2, k, depth, best_fit_line_and_cost, calc_best_fit_line_polyfit)
        points = get_dividers_point_pairs_for_drawing(divs)
        plt.figure()  # (figsize=(16, 9), dpi=200)
        plt.grid(True)

        plt.imshow(data2)
        for line in points:
            # print(line)
            x = [line[0][0], line[1][0]]
            y = [line[0][1], line[1][1]]
            plt.plot(x,y, linewidth=1, color='red')
        plt.show()

    @staticmethod
    def test_with_banana():
        img_file = imageio.imread('/home/ge/k-segment/datasets/segmentation/2018-04-09_BA_tree_12.JPG')
        flat_img = np.mean(img_file, axis=2)
        k = 3
        depth = 2

        data2 = np.asarray(flat_img)

        _, _, divs = bicriteria(data2, k, depth, best_fit_line_and_cost, calc_best_fit_line_polyfit)
        points = get_dividers_point_pairs_for_drawing(divs)
        plt.figure()  # (figsize=(16, 9), dpi=200)
        plt.grid(True)

        plt.imshow(img_file)
        for line in points:
            # print(line)
            x = [line[0][0], line[1][0]]
            y = [line[0][1], line[1][1]]
            plt.plot(x,y, linewidth=1, color='red')
        plt.show()

    @staticmethod
    def test_with_bar_code():
        img_file = imageio.imread('/home/ge/k-segment/datasets/segmentation/bar_code.png')
        flat_img = np.mean(img_file, axis=2)
        k = 2
        depth = 2

        data2 = np.asarray(flat_img)

        _, _, divs = bicriteria(data2, k, depth, best_fit_line_and_cost, calc_best_fit_line_polyfit)
        points = get_dividers_point_pairs_for_drawing(divs)
        plt.figure()  # (figsize=(16, 9), dpi=200)
        plt.grid(True)

        plt.imshow(img_file)
        for line in points:
            # print(line)
            x = [line[0][0], line[1][0]]
            y = [line[0][1], line[1][1]]
            plt.plot(x, y, linewidth=1, color='red')
        plt.show()

    @staticmethod
    def test_with_bar_code_1():
        img_file = imageio.imread('/home/ge/k-segment/datasets/segmentation/bar_code.png', )
        flat_img = np.mean(img_file, axis=2)
        k = 1
        depth = 2

        data2 = np.asarray(flat_img)

        _, _, divs = bicriteria(data2, k, depth, best_fit_line_and_cost, calc_best_fit_line_polyfit)
        points = get_dividers_point_pairs_for_drawing(divs)
        plt.figure()  # (figsize=(16, 9), dpi=200)
        plt.grid(True)

        plt.imshow(img_file)
        for line in points:
            # print(line)
            x = [line[0][0], line[1][0]]
            y = [line[0][1], line[1][1]]
            plt.plot(x, y, linewidth=1, color='red')
        plt.show()
