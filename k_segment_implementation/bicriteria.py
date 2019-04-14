#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys

from utils import generate_data_array
from Partitioner import Partitioner


def main(argv):

    # size = argv[1]
    # k = argv[2]

    size = 16
    k = 2

    data = generate_data_array(size)
    partitioner = Partitioner(data)
    partitioned_data = partitioner.partition(k)

    partitioner.update_cost_for_group(2, 1)
    partitioner.update_cost_for_group(4, 10)

    #print(partitioned_data)
    partitioner.remove_item(2)
    #print(partitioned_data)
    #print(partitioner.enumerated_data)
    return 0


if __name__ == "__main__":
    print("Python version: {}".format(sys.version))
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)

