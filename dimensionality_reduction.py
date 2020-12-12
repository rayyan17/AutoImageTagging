import numpy as np
from pyspark import SparkContext
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext


def addFileName(l, fname):
    l.append(fname)
    l.extend(l[:128])
    return(l[128:])


if __name__ == '__main__':
    sc = SparkContext(appName="PCAModeling")
    sqlContext = SQLContext(sc)

    input_parquet_path = "/user/ylalwani/CS651FinalProject/val2014DescriptorParquet"

    features_rdd = sqlContext.read.parquet(input_parquet_path).rdd.map(lambda x: [addFileName(l, x['fileName']) for l in x['features']]).flatMap(lambda x: x).map(lambda x: (str(x[0]), np.array(x[1:])))
    row_vectors = features_rdd.map(lambda t: (t[0], Vectors.dense(t[1])))

    df = sqlContext.createDataFrame(row_vectors, ["imgid", "features"])
    pca = PCA(k=2, inputCol="features")
    pca.setOutputCol("pca_features")

    model = pca.fit(df.select("features"))
    model.save("/user/ylalwani/CS651FinalProject/pca-model-full")
