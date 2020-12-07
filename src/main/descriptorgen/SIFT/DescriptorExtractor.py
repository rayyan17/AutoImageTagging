import logging
import io
import sys
import os
import findspark
if ("-localhost" in sys.argv):
    findspark.init("/u/cs451/packages/spark")
import cv2
import numpy as np
# TODO don't need below line
np.set_printoptions(threshold=sys.maxsize)
import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

# Base Reference: https://samos-it.com/posts/computer-vision-opencv-sift-surf-kmeans-on-spark.html

def extract_opencv_features():

    def extract_opencv_features_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.frombuffer(bytes(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, 0)
            # print("type(img)" + str(type(img)))
            # if feature_name in ["surf", "SURF"]:
            #     extractor = cv2.SURF()
            # elif feature_name in ["sift", "SIFT"]:
            #     extractor = cv2.SIFT()
            
            # cv2.xfeatures2d.SIFT_create() for SIFT and cv2.ORB_create() for ORB work
            extractor = cv2.xfeatures2d.SIFT_create()

            kp, descriptors = extractor.detectAndCompute(img, None)
            # print("kp type = {}, descriptors type = {}".format(type(kp), type(descriptors)))

            return [(imgfilename, descriptors)]
        except Exception as e:
            logging.exception(e)
            return []

    return extract_opencv_features_nested

if __name__ == "__main__":
    sc = SparkContext(appName="DescriptorExtractor")
    sqlContext = SQLContext(sc)

    image_seqfile_path = sys.argv[1]
    output_path = sys.argv[2]
    num_partitions = sys.argv[3]

    images = sc.sequenceFile(image_seqfile_path, minSplits=int(num_partitions))
    iCount = images.count()
    print("images count = {}".format(iCount))

    features = images.flatMap(extract_opencv_features())
    features = features.filter(lambda x: x[1] is not None)
    # features.map(lambda x: (x[0], x[1].tolist())).saveAsTextFile(output_path)
    features = features.map(lambda x: (Row(fileName=x[0], features=x[1].tolist())))
    featuresSchema = sqlContext.createDataFrame(features)
    featuresSchema.registerTempTable("images")

    # Delete output_path if it already exists
    fs = (sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration()))
    fs.delete(sc._jvm.org.apache.hadoop.fs.Path(output_path), True)

    featuresSchema.write.parquet(output_path)

#sc.binaryFiles("deps/train2014_10images").map(lambda x: x[0]).count()

#sc.binaryFiles("deps/train2014_10images").saveAsSequenceFile("10ImagesSequenceFile_2")