import os
from collections import Counter
from functools import partial
from functools import reduce

import pandas as pd
from pyspark import SparkContext
from pyspark.ml.feature import PCAModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, FloatType

get_cluster_size = udf(lambda x: len(x), IntegerType())
transform_to_vec = udf(lambda x: Vectors.dense(x), VectorUDT())
get_component1 = udf(lambda x: x.toArray().tolist()[0], FloatType())
get_component2 = udf(lambda x: x.toArray().tolist()[1], FloatType())


def unionAll(dfs):
    return reduce(DataFrame.unionAll, dfs)


def get_cluster_purity(images_p, b_mapping):
    total_c = list()
    for img in images_p:
        img_c = b_mapping.value[img.split("/")[-1]]
        total_c += img_c

    N = len(images_p)
    cntr = Counter(total_c)
    purity_score = 0
    u_c = 0
    for k, v in cntr.items():
        purity_score += v/N
        u_c += 1

    return purity_score/u_c


def create_cluster_df(abs_path, broadcast_mapping, sqlContext, spark_session):
    fs = spark_session._jvm.org.apache.hadoop.fs.FileSystem.get(spark_session._jsc.hadoopConfiguration())
    list_status = fs.listStatus(spark_session._jvm.org.apache.hadoop.fs.Path(abs_path))
    results = [file.getPath().getName() for file in list_status]

    pca_trained_model = PCAModel.load("/user/ylalwani/CS651FinalProject/pca-model-full")
    calculate_cluster_purity = udf(partial(get_cluster_purity, b_mapping=broadcast_mapping), FloatType())

    df_visual_list = list()
    df_analysis_list = list()
    iterr = 0
    for file in results:
        iter_file_path = os.path.join(abs_path, file)
        df_out = sqlContext.read.parquet(iter_file_path)
        df_out = df_out.withColumn("cluster_size", get_cluster_size(df_out.images))
        df_out = df_out.withColumn("features", transform_to_vec(df_out.centroid))

        df_out = df_out.withColumn("cluster_purity", calculate_cluster_purity(df_out.images))

        df_out = pca_trained_model.transform(df_out)
        df_out = df_out.withColumn("component1", get_component1(df_out.pca_features))
        df_out = df_out.withColumn("component2", get_component2(df_out.pca_features))

        df_visual = df_out.select(df_out.cluster_id, df_out.component1, df_out.component2, df_out.cluster_size)
        df_visual = df_visual.withColumn("iteration", udf(lambda x: iterr)(df_out.cluster_id))
        df_visual_list.append(df_visual)

        df_analysis = df_out.select(df_out.cluster_id, df_out.loss, df_out.cluster_purity)
        df_analysis = df_analysis.withColumn("iteration", udf(lambda x: iterr)(df_out.cluster_id))
        df_analysis_list.append(df_analysis)

        iterr += 1

    unionAll(df_visual_list).write.parquet("cluster_visual2.parquet", mode="overwrite")
    unionAll(df_analysis_list).write.parquet("cluster_analysis2.parquet", mode="overwrite")

    return unionAll(df_visual_list)


if __name__ == '__main__':

    sc = SparkContext(appName="PCAModeling")
    sqlContext = SQLContext(sc)
    spark_session = SparkSession(sc)

    annotation_file_path = "/user/ylalwani/CS651FinalProject/image_super_cat_2014train.csv"
    mapping = dict(sc.textFile(annotation_file_path).map(lambda x: [x.split(",")[0], x.split(",")[1:]]).collect())
    broadcast_mapping = sc.broadcast(mapping)

    clustering_output_path = "/user/ylalwani/CS651FinalProject/train2014_10imagesClusteringParquetOutput/"

    create_cluster_df(clustering_output_path, broadcast_mapping, sqlContext, spark_session)
    # df.write.parquet("cluster_visual2.parquet", mode="overwrite")
    # df =

# /user/ylalwani/CS651FinalProject/image_super_cat_2014val.csv
# sqlContext.read.parquet("cluster_visual2.parquet").toPandas().to_csv("cluster_visual.csv", index=False)
