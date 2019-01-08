import numpy as np
import CoresetKSeg
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import pandas as pd
from collections import namedtuple
from utils_seg import gen_synthetic_graph, load_csv_into_dataframe
import queue
import unittest
import sys
from threading import Thread, Event
import CoresetKSeg

from stack import Stack

StackItem = namedtuple("StackItem", "coreset level")
WeightedPointSet = namedtuple("WeightedPointSet", "points weights")


def point_accumulator(path=None):
    if path is None:
        points = pd.DataFrame(gen_synthetic_graph(1200, 5))
    else:
        points = load_csv_into_dataframe(path)
    print(points)

    return points


class Stream(Thread):

    def __init__(self, coreset_alg, leaf_size: int, eps: float, k: int, streaming_context):
        super().__init__(name="coreset stream thread")
        self.coreset_alg = coreset_alg
        self.leaf_size = leaf_size
        self.last_leaf = []
        self.eps = eps
        self.stack = Stack()
        self.k = k
        self.streaming_context = streaming_context
        # self.host = stream_host_address
        # self.port = stream_host_port
        # self.lines = self.streaming_context.socketTextStream(stream_host_address, stream_host_port)
        self._stop_event = Event()
        self.tree = queue.Queue()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        # sc = SparkContext(appName="PythonStreamingNetworkWordCount")
        # ssc = StreamingContext(sc, 1)
        lines = self.streaming_context.socketTextStream("localhost", 9990)
        # lines = self.streaming_context.textFileStream(".")
        flat_lines = lines.flatMap(lambda line: line.split(","))
        maped_lines = flat_lines.map(lambda word: (word, 1))
        counted_lines = maped_lines.reduceByKey(lambda a, b: a + b)
        counted_lines.pprint()
        try:
            self.streaming_context.start()
            # self.streaming_context.awaitTerminationOrTimeout(timeout=19)
        except (KeyboardInterrupt, EOFError, Exception) as e:
            print("#" * 80)
            print(e)
            raise KeyboardInterrupt
#     def _merge(self, pset1, pset2):
#         points = np.vstack([pset1.points, pset2.points])
#         weights = np.hstack([pset1.weights, pset2.weights])
#         cset = self.coreset_alg(k=self.k, eps=self.eps, weights=weights)
#         coreset, weights = cset.compute(data_points=points)
#         return WeightedPointSet(coreset, weights)
#
#     def _add_leaf(self, points, weights):
#         if weights is None:
#             weights = np.ones((points.shape[0])).ravel()
#         self._insert_into_tree(WeightedPointSet(points, weights))
#
#     def _is_correct_level(self, level):
#         if self.stack.is_empty():
#             return True
#         elif self.stack.top().level > level:
#             return True
#         elif self.stack.top().level == level:
#             return False
#         else:
#             raise Exception("New level should be smaller")
#
#     def _insert_into_tree(self, coreset):
#         level = 1
#         while not self._is_correct_level(level):
#             last = self.stack.pop()
#             coreset = self._merge(last.coreset, coreset)
#             level += 1
#         self.stack.push(StackItem(coreset, level))
#
#     def add_points(self, points):
#         """Add a set of points to the stream.
#
#         If the set is larger than leaf_size, it is split
#         into several sets and a coreset is constructed on each set.
#         """
#
#         for split in np.array_split(points, self.leaf_size):
#             self._add_leaf(split, None)
#         # for i in range(len(points)):
#         #     s
#
#     def get_unified_coreset(self):
#         solution = None
#         while not self.stack.is_empty():
#             coreset = self.stack.pop().coreset
#             if solution is None:
#                 solution = coreset
#             else:
#                 solution = self._merge(solution, coreset)
#         return solution.points, solution.weights


def batch(iterable_data, chunk_size=1):
    l = len(iterable_data)
    for ndx in range(0, l, chunk_size):
        yield iterable_data[ndx:min(ndx + chunk_size, l)]


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
    #     sys.exit(-1)
    sc = SparkContext(appName="PythonStreamingNetworkWordCount")
    ssc = StreamingContext(sc, batchDuration=1)
    streamer = Stream(None, 100, 0.4, 4, ssc)
    try:
        streamer.start()
    except (KeyboardInterrupt, EOFError, Exception) as e:
        print("#"*80)
        print(e)
        streamer.join()
        ssc.stop(stopSparkContext=True, stopGraceFully=True)
    ssc.awaitTerminationOrTimeout(timeout=10)
