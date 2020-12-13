import sys
import findspark
if ("-localhost" in sys.argv):
    findspark.init("/u/cs451/packages/spark")
import pyspark
from pyspark import SparkContext
import numpy as np
import cv2
from pyspark.sql import SQLContext, Row

def find_closest_cluster(centroids,x):
    '''
    Calcuates the min sqaured Euclidean distance between a data point and 
    each of the centroids
        centroids: [int, [float]]
        x: ([float]) 
    '''
    min_dist = np.inf
    closest_cluster_id = 0
    for c in centroids:
        d = np.linalg.norm(c[1] - x)**2
        if (d < min_dist):
            min_dist = d
            closest_cluster_id = c[0]
    return (closest_cluster_id,1)

if __name__ == "__main__":
    sc = SparkContext(appName="Search")
    sqlContext = SQLContext(sc)

    try:
        input_image_path = sys.argv[1]
        input_clusters_path = sys.argv[2] # This is the path to the parquet file, if your parquet file is at "clusteringParquetOutput/Iteration-00009", enter that, not "clusteringParquetOutput"
        output_path = sys.argv[3]
    except:
        print("Usage: Search.py <input_image_path> <input_clusters_path> <output_path>; Example: Search.py deps/train2014_10images/COCO_train2014_000000100777.jpg val2014_clusters_3centroids_10iterations/Iteration-00009 SearchResults")

    # Extract SIFT descriptors of Image
    imgfile_imgbytes = sc.binaryFiles(input_image_path)
    imgfilename, imgbytes = imgfile_imgbytes.collect()[0]
    nparr = np.frombuffer(bytes(imgbytes), np.uint8)
    img = cv2.imdecode(nparr, 0)
    extractor = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = extractor.detectAndCompute(img, None)
    features_rdd = sc.parallelize(descriptors.tolist()).map(lambda x: (np.array(x)))

    # Load cluster information
    clusters = sqlContext.read.parquet(input_clusters_path).rdd.map(lambda x: (x['cluster_id'], x['centroid']))
    clusters_broadcast = sc.broadcast(clusters.collect())

    # Find closest cluster for each descriptor, then reduce and sort by most descriptors for each cluster
    results = features_rdd.map(lambda x: find_closest_cluster(clusters_broadcast.value, x))
    results_sorted = results.reduceByKey(lambda x, y: x + y).map(lambda x: (x[1], x[0])).sortByKey(False).map(lambda x: (x[1], x[0]))

    # Delete path_path if it already exists
    fs = (sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration()))
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(output_path), True)
    
    results_sorted.coalesce(1).saveAsTextFile(output_path)
    