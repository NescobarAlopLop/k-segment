#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys

import numpy as np
from utils import generate_data_array
from Partitioner import Partitioner


def main(argv):

    # size = argv[1]
    # k = argv[2]

    size = 64
    k = 4

    data = generate_data_array(size)
    print(data)
    data = np.reshape(data, (8, 8))
    partitioner = Partitioner(data, 0)
    partitioned_data = partitioner.partition(k, 0)

    partitioner.update_cost_for_group(2, 1)
    partitioner.update_cost_for_group(4, 10)

    #print(partitioned_data)
    partitioner.remove_item(2)
    #print(partitioned_data)
    #print(partitioner.enumerated_data)

    partitioner = Partitioner(data, 0)
    partitioner.partition(k, 0)
    partitioner.calculate_cost_for_group(1)
    return 0


if __name__ == "__main__":
    print("Python version: {}".format(sys.version))
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)

