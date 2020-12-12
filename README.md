# CS651-Image-Search

## Initial Setup:

### Virtual Environments

Need  to make use of a virtual environment in order allow for packages not installed on all cluster nodes to be made available.

Navigate to your project directory and setup you virtual environment via:

```bash
pip3 install virtualenv
virtualenv --copies my_env
```

 Activate into the virtual environment  via:

```
source my_env/bin/activate
```

Upgrade you pip to the newest version

```
pip install -U pip
```

Install all required  packages decencies

```
pip install opencv-python
```

The environment logs which libraries you have installed. Can produce this log so that others may load them into their own virtual environment. Produce the log via:

```
pip freeze > requirements.txt
```

If you want to load someone else requirements into your environment do so via:

```
pip install -r requirements.txt
```

When you done working in you virtual environment exit it via:

```
deactivate
```

### Running Scripts In Local  Mode:

The setup for running in local vs on the cluster requires changing the configuration of Environment Variables which automatically called by a spark context initalization

When in local these  Environment Variables can be set automatically within the script and the context created for use via:

```python
import findspark
findspark.init("/u/cs451/packages/spark")
import pyspark
from pyspark import SparkContext
sc = pyspark.SparkContext(appName="local-script-run")

```

### Running Scripts On Cluster:

Need to use spark-submit command to access cluster resources.  Also need to submit with your script an archive of the virtual environment so that it can be distributed to the different nodes.

Create a tar of your virtual environment via:

```
tar -hzcf venv.tar.gz my_env/*
```

When you submit you script to the cluster  make sure you are not still activated within the virtual environment then use the following to include the virtual environment in the spark-submit:

```
PYSPARK_PYTHON=./env/bin/python \ 
SPARK_HOME=/u/cs451/packages/spark \
spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/bin/python \
--master yarn-cluster \
--archives venv.tar.gz#env \
my_file.py 
```

If main imports multiple scripts then we can include them in the submit via:

```
PYSPARK_PYTHON=./env/bin/python \ 
SPARK_HOME=/u/cs451/packages/spark \
spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/bin/python \
--master yarn-cluster \
--archives venv.tar.gz#env \
--py-files script1.py, script2.py
main.py 
```

Note: When using the above command, ensure that when you extract your tarball, the `lib` and `bin` folders are created, and not an intermediate directory (`venv/lib`, `venv/bin`). If you run into errors, obtain the yarn logs for your run via `yarn logs -applicationId application_xxxxxxxxxxxxx_xxxxx > yarnlogs.txt`, and confirm that the PYSPARK_PYTHON environment variable and the imports within the script are pointing to the correct locations after extraction.

**NOTE:** When submitting to cluster you cannot have  `findspark.init()`   called  in a script or it will throw an error.  I use an optimal command line argument  `-localhost`  for my scripts which specifies to call   `findspark.init() ` in my scripts otherwise its not called.

### Reading Files From Spark:

Any files that are read via  `sc.textFile(<filepath>)`  within our scripts need to be stored on HDFS and 

In the clustering.py main file  I added the  test data used in the `if __name__ == "__main__"`  portion   to my HDFS via. 

```
-hdfs dfs -put featureTestData.txt
```

**NOTE:** It's possible for us to create a shared directory on HDFS so that we could all work from the same data. We should do this!

### HDFS Working Directory

The HDFS directory, `/user/ylalwani/CS651FinalProject`, has been allowed public access to act as a working directory for this project. The [val2014](https://cocodataset.org/#download) dataset has been added to it.


## Pipeline

### Converting Images to a SequenceFile

Run [src/main/descriptorgen/SIFT/ConvertToSequenceFile.py](https://github.com/rayyan17/CS651-Auto-Image-Captioning/blob/main/src/main/descriptorgen/SIFT/ConvertToSequenceFile.py) to convert a directory of images to a SequenceFile. The script takes 3 arguments: Input Directory of Images, Output Location, Number Of Partitions.

```
PYSPARK_PYTHON=./env/bin/python SPARK_HOME=/u/cs451/packages/spark spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/bin/python --master yarn-cluster --archives yash_env2/yash_env2.zip#env --num-executors 4 --executor-cores 4 --executor-memory 24G --driver-memory 2g src/main/descriptorgen/SIFT/ConvertToSequenceFile.py CS651FinalProject/val2014 CS651FinalProject/val2014SeqFile 32
```

### Generating SIFT descriptors

Run [src/main/descriptorgen/SIFT/DescriptorExtractor.py](https://github.com/rayyan17/CS651-Auto-Image-Captioning/blob/main/src/main/descriptorgen/SIFT/DescriptorExtractor.py) to generate SIFT descriptors for each image in a SequenceFile, and save them in a table as a Parquet file, with columns `fileName` (str) and `features` (list of descriptors, each descriptor is a list of 128 floats). The script takes 3 arguments: Input Sequence File, Output Directory, Number of Partitions. The virtual environment passed will need to include cv2 (`pip install opencv-contrib-python-headless`).

```
PYSPARK_PYTHON=./env/bin/python SPARK_HOME=/u/cs451/packages/spark spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/bin/python --master yarn-cluster --archives yash_env2/yash_env2.zip#env --num-executors 4 --executor-cores 4 --executor-memory 24G --driver-memory 2g src/main/descriptorgen/SIFT/DescriptorExtractor.py CS651FinalProject/val2014SeqFile CS651FinalProject/val2014DescriptorParquet 32
```

### Creating Clusters

Run [clustering.py](https://github.com/rayyan17/CS651-Auto-Image-Captioning/blob/main/clustering.py) to generate clusters after training on the SIFT descriptors. Note that KMeans is trained on the features themselves, so instead of passing an image to the model, we flatten each image's `features` and pass descriptors to the model. After each iteration, a Parquet file is created containing columns `cluster_id` (int), `centroid` (list), `images` (list) and `loss` (float). The script takes 3 arguments: Input Parquet File containing Descriptors, Number of Clusters, Output Directory for Parquet file for each Iteration.

```
PYSPARK_PYTHON=./env/bin/python SPARK_HOME=/u/cs451/packages/spark spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/bin/python --master yarn-cluster --archives yash_env2/yash_env2.zip#env --num-executors 4 --executor-cores 4 --executor-memory 24G --driver-memory 2g clustering.py CS651FinalProject/val2014DescriptorParquet 3 val2014_clusters_3centroids_10iterations
```

### Search

Run [Search.py](https://github.com/rayyan17/CS651-Auto-Image-Captioning/blob/main/Search.py) to find the clusters an image belongs to. The result is a list of clusters that would contain descriptors from the input image, sorted by most descriptors per cluster, printed to stdout. The script takes 2 arguments: Input Image File, Input Parquet File with Cluster Information


(On Linux)
```
spark-submit Search.py deps/train2014_10images/COCO_train2014_000000100777.jpg val2014_clusters_3centroids_10iterations/Iteration-00009 -localhost
```

TODO: Replace with command for Datasci, save output as textFile