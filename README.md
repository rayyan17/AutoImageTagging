# CS651-Image-Search

### Initial Setup:

##### Virtual Environments

Need to make use of a virtual environment in order allow for packages not installed on all cluster nodes to be made available.

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

##### Running Scripts In Local  Mode:

The setup for running in local vs on the cluster requires changing the configuration of Environment Variables which automatically called by a spark context initalization

When in local these  Environment Variables can be set automatically within the script and the context created for use via:

```python
import findspark
findspark.init("/u/cs451/packages/spark")
import pyspark
from pyspark import SparkContext
sc = pyspark.SparkContext(appName="local-script-run")

```

##### Running Scripts On Cluster:

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

**NOTE:** When submitting to cluster you cannot have  `findspark.init()`   called  in a script or it will throw an error.  I use an optimal command line argument  `-localhost`  for my scripts which specifies to call   `findspark.init() ` in my scripts otherwise its not called.



##### Reading Files From Spark:

Any files that are read via  `sc.textFile(<filepath>)`  within our scripts need to be stored on HDFS and 

In the clustering.py main file  I added the  test data used in the `if __name__ == "__main__"`  portion   to my HDFS via. 

```
-hdfs dfs -put featureTestData.txt
```

**NOTE:** It's possible for us to create a shared directory on HDFS so that we could all work from the same data. We should do this!

