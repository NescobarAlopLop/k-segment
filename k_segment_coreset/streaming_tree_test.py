import numpy as np
from k_segment_coreset.CoresetKSeg import CoresetKSeg
from collections import namedtuple
from threading import Thread, Event
import CoresetKSeg
from stack import Stack
import utils_seg
import random
from utils_seg import visualize_2d
# from utils_seg import gen_synthetic_graph, load_csv_into_dataframe
# import pandas as pd
# from pyspark.streaming import StreamingContext
# import unittest
# import sys
# import copy
# from pyspark import SparkContext
# import ksegment
# from typing import Union


StackItem = namedtuple("StackItem", "coreset level")
WeightedPointSet = namedtuple("WeightedPointSet", "points weights")


class CoresetStreamer(Thread):

    def __init__(self, coreset_alg, sample_size: int, eps: float, k: int, streaming_context):
        super().__init__(name="coreset stream thread")
        self.coreset_alg = coreset_alg
        self.sample_size = sample_size
        self.last_leaf = []
        self.eps = eps
        self.k = k
        self.streaming_context = streaming_context
        self._stop_event = Event()
        self.stack = Stack()
        self.leaf_size = sample_size

    def _add_leaf(self, points, weights):
        if weights is None:
            weights = np.ones((points.shape[0])).ravel()
        self._insert_into_tree(WeightedPointSet(points, weights))

    def _merge(self, pset1: WeightedPointSet, pset2: WeightedPointSet):
        try:
            if type(pset1.points) is np.ndarray:
                points = np.vstack([pset1.points, pset2.points])
            else:
                points = pset1.points + pset2.points
        except ValueError as e:
            print(e)
            raise e
        weights = np.hstack([pset1.weights, pset2.weights])
        # points = pset1.points + pset2.points
        # weights = pset1.weights + pset2.weights
        cset = self.coreset_alg(k=self.k, eps=self.eps, weights=weights)
        coreset, weights = cset.compute(data_points=points)
        return WeightedPointSet(coreset, weights)

    def _is_correct_level(self, level):
        if self.stack.is_empty():
            return True
        elif self.stack.top().level > level:
            return True
        elif self.stack.top().level == level:
            return False
        else:
            raise Exception("New level should be smaller")

    def _insert_into_tree(self, coreset):
        level = 1
        while not self._is_correct_level(level):
            last = self.stack.pop()
            coreset = self._merge(last.coreset, coreset)
            level += 1
        self.stack.push(StackItem(coreset, level))

    def add_points(self, points):
        """Add a set of points to the stream.

        If the set is larger than leaf_size, it is split
        into several sets and a coreset is constructed on each set.
        """
        # TODO: maybe tay into account leaf_size or maybe external chunk size is enough
        # for split in np.array_split(points, self.leaf_size):
        self._add_leaf(points, None)

    def get_unified_coreset(self):
        solution = None
        while not self.stack.is_empty():
            coreset = self.stack.pop().coreset
            if solution is None:
                solution = coreset
            else:
                solution = self._merge(solution, coreset)
        return solution.points, solution.weights

    def __str__(self):
        return '{}'.format(self.stack)


def batch(iterable_data, batch_size: int = 10, random_size_chunks: bool = False):
    data_len = len(iterable_data)
    chunk_start = batch_size
    min_batch_size = min(20, batch_size)
    max_batch_size = max(50, batch_size)
    current_chunk_size = batch_size
    while chunk_start < data_len:
        if random_size_chunks:
            current_chunk_size = random.randint(min_batch_size, max_batch_size)
        yield iterable_data[chunk_start:min(chunk_start + current_chunk_size, data_len)]
        chunk_start += current_chunk_size


def main(path: str, col: int = 0):
    points = utils_seg.load_csv_into_dataframe(path).values[:, col]
    points = np.column_stack((np.arange(1, len(points) + 1), points[:]))
    k = 4
    eps = 0.3
    stream = CoresetStreamer(CoresetKSeg.CoresetKSeg, sample_size=100, eps=eps, k=k, streaming_context=None)
    for chunk in batch(points, batch_size=70, random_size_chunks=False):
        print(len(chunk))
        stream.add_points(chunk)
    p_cset, w_cset = stream.get_unified_coreset()
    print(p_cset, w_cset)
    visualize_2d(points, p_cset, k, eps, show=True)
    print(stream)


if __name__ == '__main__':
    # file_path = '/home/ge/k-segment/datasets/KO_no_date.csv'
    file_path = '/home/ge/k-segment/datasets/bicriteria test case - coreset.csv'
    main(file_path)
    # if len(sys.argv) != 3:
    #     print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
    #     sys.exit(-1)
    # sc = SparkContext(appName="PythonStreamingNetworkWordCount")
    # ssc = StreamingContext(sc, batchDuration=1)
    # streamer = CoresetStreamer(None, 100, 0.4, 4, ssc)
    # try:
    #     streamer.start()
    # except (KeyboardInterrupt, EOFError, Exception) as e:
    #     print("#"*80)
    #     print(e)
    #     streamer.join()
    #     ssc.stop(stopSparkContext=True, stopGraceFully=True)
    # ssc.awaitTerminationOrTimeout(timeout=100)
