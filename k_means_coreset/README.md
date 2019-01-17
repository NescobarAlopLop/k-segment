# k-segment-2d
The goal of this project is to extend the ability of k-segment coreset algorithm to be able to provide compression
for 2 dimensional data.

## How to:
To start working with this project please have a look at tests first.
I feel like my predecessors did quite a good job and they are pretty much self explanatory.

The original work was done here [https://github.com/vkhakham/k-segment](https://github.com/vkhakham/k-segment)  


# spark_coreset
Spark Framework for running coresets

# running instructions for windows
create environment variable:
https://blogs.msdn.microsoft.com/arsen/2016/02/09/resolving-spark-1-6-0-java-lang-nullpointerexception-not-found-value-sqlcontext-error-when-running-spark-shell-on-windows-10-64-bit/

from run_k-segment_coreset folder:
	bash -c "./../go_k_segment_coreset.sh" && %SPARK_BIN%\spark-submit.cmd --master local[4] tree.py