import sys
import findspark
if ("-localhost" in sys.argv):
    findspark.init("/u/cs451/packages/spark")
import pyspark
from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext(appName="ConvertToSequenceFile")

    input_dir_path = sys.argv[1]
    output_path = sys.argv[2]
    num_partitions = sys.argv[3]

    # Delete output_path if it already exists
    fs = (sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration()))
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(output_path), True)

    sc.binaryFiles(input_dir_path, int(num_partitions)).saveAsSequenceFile(output_path)