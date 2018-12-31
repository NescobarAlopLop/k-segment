# import sys
# from operator import add
#
# from pyspark import SparkContext
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print(sys.stderr, "Usage: wordcount <master> <file>")
#         exit(-1)
#     sc = SparkContext(sys.argv[1], "PythonWordCount")
#     lines = sc.textFile(sys.argv[2], 1)
#     counts = lines.flatMap(lambda x: x.split(' ')) \
#                   .map(lambda x: (x, 1)) \
#                   .reduceByKey(add)
#     output = counts.collect()
#     for (word, count) in output:
#         print("%s: %i" % (word, count))


import threading
from pyspark import SparkContext, SparkConf
import numpy as np


def task(sc, i):
  print(sc.parallelize(np.random.randint(0, 100000, 10000)).count())
  # a = sum(np.random.randint(0, 100000, 10000000))
  # print(a)

def run_multiple_jobs():
  conf = SparkConf().setMaster('local[*]').setAppName('appname')
  # Set scheduler to FAIR: http://spark.apache.org/docs/latest/job-scheduling.html#scheduling-within-an-application
  conf.set('spark.scheduler.mode', 'FAIR')
  sc = SparkContext(conf=conf)

  for i in range(6):
    t = threading.Thread(target=task, args=(sc, i))
    t.start()
    print('spark task', i, 'has started')


run_multiple_jobs()
