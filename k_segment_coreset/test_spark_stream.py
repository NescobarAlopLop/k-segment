from CoresetKSeg import build_coreset_for_pyspark
from utils_seg import load_csv_file, gen_synthetic_graph
import numpy as np
from pyspark import SparkContext


sc = SparkContext()

num_of_points = 50000
chunk_size = 10000
k = 5
eps = 0.4
points = gen_synthetic_graph(n=num_of_points + 1, k=k, dim=1, deviation=0.01, max_diff=3)
points = np.column_stack((np.arange(1, len(points) + 1), points[:]))
aggregated_for_rdd = []


for i in range(0, len(points), chunk_size):
    aggregated_for_rdd.append(points[i:i + chunk_size])

data = sc.parallelize(aggregated_for_rdd)

all_coresets = data.map(lambda x: build_coreset_for_pyspark(x, k, eps)).collect()
sc.stop()

print(all_coresets)
print(len(all_coresets))
