__author__ = 'Anton'


"""
coreset vs. uniform graph plotting
"""

import numpy as np
from test_framework import TestCoreset


x_1 = 10000        # 9900000
y_1 = 10000        # 9950000
x_2 = 10000      # 100000
y_2 = 10000     # 10000000


def random_uniform_points(n, m):
    """
    Create uniform n points from range 0 to m.
    For example:
    n = 100000 m=10000000 will give good results
    """
    x = np.random.randint(0, m, n)
    y = np.random.randint(0, m, n)
    p = np.hstack((y, x)).reshape(n, 2)
    return p.astype(np.float64)


def two_clusters_test(n, m):
    """
    Create 2 clusters, first is size n, second is m
    """
    x = np.random.randint(0, 10, n)
    y = np.random.randint(0, 10, n)
    p = np.hstack((y, x)).reshape(n, 2)

    x = np.random.randint(x, y, m)
    y = np.random.randint(x, y, m)
    p1 = np.hstack((y, x)).reshape(m, 2)
    p = np.vstack((p, p1))
    return p.astype(np.float64)


points = random_uniform_points(x_2, y_2)
# p = two_clusters_test(30, 1000000)
k = 5
test = TestCoreset(points)
test_range = range(100, 5500, 500)
trails = 10
"""
No tree.
"""
# test.run_test(k, test_range, trails, cset_kmeans_trials=10)

"""
Using `stream.m`:
"""
test.run_test(k, test_range, trails, tree=True, num_chunks=4)
