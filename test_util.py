import sys
# import findspark
# if ("cluster" not in sys.argv):
#     findspark.init("/u/cs451/packages/spark")
import unittest
import logging
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession


# Reference: https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8
class PySparkTest(unittest.TestCase):
    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)
    
    @classmethod
    def create_testing_pyspark_context(cls):
        
        return pyspark.SparkContext(appName="testing-pyspark-context")

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession.builder
            .appName('testing-pyspark-session')
            .enableHiveSupport()
            .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.sc = cls.create_testing_pyspark_context()
        #cls.spark = cls.create_testing_pyspark_session()

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()
        #cls.spark.stop()
