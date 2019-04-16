#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys

import numpy as np
from utils import generate_data_array
from Partitioner import Partitioner


def main(argv):

    # size = argv[1]
    # k = argv[2]

    size = 256
    k = 2

    data = generate_data_array(size)
    data = np.reshape(data, (8, 32))
    print(data)
    #partitioned_data = partitioner.partition(k, 1)

    #partitioner.update_cost_for_group(2, 1)
    #partitioner.update_cost_for_group(4, 10)

    #print(partitioned_data)
    #artitioner.remove_item(2)
    #print(partitioned_data)
    #print(partitioner.enumerated_data)

    partitioner = Partitioner(data, k, depth=1)
    print(partitioner.dividers)
    #partitioner.partition(k, 0)
    #partitioner.calculate_cost_for_group(1)
    return 0


if __name__ == "__main__":
    print("Python version: {}".format(sys.version))
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)

