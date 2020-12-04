import sys
import findspark
findspark.init("/u/cs451/packages/spark")
from test_util import PySparkTest
import clustering
import numpy as np
import unittest
import warnings

class TestClusteringInitalization(PySparkTest):
    def setUp(self):
       warnings.simplefilter("ignore", ResourceWarning)
       self.features_rdd =  self.sc.textFile("featureTestData.txt").\
       map(lambda x: x.split(" ")).map(lambda x:np.array([float(c) for c in x])).map(lambda x:(int(x[0]),x[1:]))
    
    def test_init_centroids_length(self):
        init_centroids = clustering.init_centroids(data=self.features_rdd,k=3,sc=self.sc)
        self.assertEqual(len(init_centroids),3)
    
class TestClusteringUpdate(PySparkTest):
    def setUp(self):
       warnings.simplefilter("ignore", ResourceWarning)
       self.features_rdd = self.sc.textFile("featureTestData.txt").\
       map(lambda x: x.split(" ")).map(lambda x:np.array([float(c) for c in x])).map(lambda x:(int(x[0]),x[1:]))
       self.init_centroids = clustering.init_centroids(data=self.features_rdd,k=3,sc=self.sc)
       
    def test_cluster_assignment_not_none(self):
        centroids_broadcast = self.sc.broadcast(self.init_centroids)
        assignments = self.features_rdd.map(lambda x: clustering.cluster_assignment(centroids_broadcast.value,x))
        self.assertIsNotNone(assignments)
        
    def test_cluster_assignment_no_empty_cluster(self):
        centroids_broadcast = self.sc.broadcast(self.init_centroids)
        assignments = self.features_rdd.map(lambda x: clustering.cluster_assignment(centroids_broadcast.value,x)).countByKey()
        self.assertEqual(len(assignments),3)
    
    def test_cluster_centroids_average_calc(self):
        centroids_broadcast = self.sc.broadcast(self.init_centroids)
        centroids_update = self.features_rdd.map(lambda x: clustering.cluster_assignment(centroids_broadcast.value,x)).\
            map(lambda x: (x[0],(x[1][1],1))).reduceByKey(lambda x,y: (x[0] + y[0], x[1] + y[1])).\
            map(lambda x: (x[0],x[1][0]/x[1][1])).collect()
        self.assertEqual(len(centroids_update),3)
    
    def test_loss_calculation(self):
        centroids_broadcast = self.sc.broadcast(self.init_centroids)
        centroids_update = self.features_rdd.map(lambda x: clustering.cluster_assignment(centroids_broadcast.value,x)).\
            map(lambda x: (x[0],(x[1][1],1))).reduceByKey(lambda x,y: (x[0] + y[0], x[1] + y[1])).\
            map(lambda x: (x[0],x[1][0]/x[1][1]))
        loss = centroids_update.\
            map(lambda x: np.linalg.norm(centroids_broadcast.value[x[0]] - x[1])**2).reduce(lambda x,y: x + y)
        print("LOSS: " + str(loss))
        self.assertIsNotNone(loss)
    def  test_cluster_centroids_update(self):
        centroids = self.init_centroids
        centroids_broadcast = self.sc.broadcast(centroids)
        centroids_update = self.features_rdd.map(lambda x: clustering.cluster_assignment(centroids_broadcast.value,x)).\
            map(lambda x: (x[0],(x[1][1],1))).reduceByKey(lambda x,y: (x[0] + y[0], x[1] + y[1])).\
            map(lambda x: (x[0],x[1][0]/x[1][1]))
        centroids = centroids_update.collect()
        centroids_broadcast.destroy()
        print(centroids)
        self.assertIsNotNone(centroids)
        
   

if __name__ == '__main__':
    unittest.main()