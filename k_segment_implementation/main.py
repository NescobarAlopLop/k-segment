#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

try:
    from utils_seg import best_fit_line_and_cost
except ImportError:
    from k_segment_coreset.utils_seg import best_fit_line_and_cost

from bicriteria import bicriteria
from Dividers import get_dividers_point_pairs_for_drawing


def main():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("-f", "--filename", required=True, help="File name of the image")

    argument_parser.add_argument("-k", required=True, type=int, help="k number for segmentation")

    argument_parser.add_argument("-d", "--depth", required=False, default=2, type=int,
                                 help="Depth of segmentation (default: 2)")

    argument_parser.add_argument("-w", "--width", required=False, default=1, type=float,
                                 help="Width of the line for drawing segment borders (can be float)")

    argument_parser.add_argument("-c", "--color", required=False, default='red', type=str,
                                 help="Color of the line for drawing segment borders (default: red)")

    argument_parser.print_help()

    args = vars(argument_parser.parse_args())
    print("Starting with arguments: {}".format(args))

    img_file = misc.imread(args['filename'])
    flat_img = np.mean(img_file, axis=2)

    data2 = np.asarray(flat_img)

    _, _, divs = bicriteria(data2, args['k'], args['depth'], best_fit_line_and_cost)
    points = get_dividers_point_pairs_for_drawing(divs)
    plt.figure()  # (figsize=(16, 9), dpi=200)
    plt.grid(True)

    plt.imshow(img_file)
    for line in points:
        x = [line[0][0], line[1][0]]
        y = [line[0][1], line[1][1]]
        plt.plot(x, y, linewidth=args['width'], color=args['color'])
    plt.show()

    return 0


if __name__ == "__main__":
    print("Python version: {}".format(sys.version))
    ret = main()
    if ret is not None:
        sys.exit(ret)
