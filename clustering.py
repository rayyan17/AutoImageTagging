import sys
import findspark
if ("-localhost" in sys.argv):
    findspark.init("/u/cs451/packages/spark")
import pyspark
from pyspark import SparkContext
import numpy as np

def min_distance_squared(centroids,x):
    '''
    Calcuates the min sqaured Euclidean distance between a data point and 
    each of the centroids
        centroids: [[float]]
        x: (int,[float]) 
    '''
    min_dist = np.inf
    for c in centroids:
        d = np.linalg.norm(c - x[1])**2
        if (d < min_dist): min_dist = d   
    return (x,min_dist)

def stochastic_reduction(x,y):
    '''
    Facilities sampling a data point where each points probability is proportional
    to its squared distance to its closest center. 
        x: ((int,[float]), float)
        y: ((int,[float]), float)
    return: ((int,[float]), float)

    '''
    total_dist = x[1] + y[1]
    px = x[1]/total_dist
    u = np.random.uniform(low=0,high=1,size=1)
    if (u[0] <= px ): return (x[0],total_dist) 
    else: return (y[0],total_dist)




def init_centroids(data,k,sc,plus=True,seed=None):
    '''
    Produces the inital k cluster centroids   
        data: An RDD of [(int,[float])] 
        k: int
        plus: boolean
    return: [[float]]
    '''
    # K-means++ initalization
    if (plus is True):
        # Sample one data point randomly based on uniform distribtion and extract feature vector 
        centroids = [data.takeSample(withReplacement=False, num=1, seed=seed)[0][1]]
        while len(centroids) < k:
            # Makes selected centroids avaliable to each node
            centroids_broadcast = sc.broadcast(centroids)
            # Compute  each points min distance squared to a cluster center
            # And sample new point c from set with probabilties of points proportional to thier min distance squared 
            c = data.map(lambda x: min_distance_squared(centroids_broadcast.value, x)).\
                reduce(lambda x,y: stochastic_reduction(x,y))
            # Add new sampled point to centroid intalisation
            centroids.append(c[0][1])
            # Remove previous centroids from memorey of each node
            centroids_broadcast.destroy()
        
    # Traditonal K-means Initalization
    else:
        # Sample k points from dataset uniformly with out replacement 
        centroids = data.takeSample(withReplacement=False, num=k, seed=seed).map(lambda x: x[1]).collect()

    return centroids

def cluster_assignment(centroids,x):
    min_dist = np.inf
    for i in range(len(centroids)):
        d = np.linalg.norm(centroids[i] - x[1])**2
        if (d < min_dist): 
            cluster_id = i
            min_dist = d  
    return (cluster_id,x)

def k_means(data,k,sc,plus=True,seed=None,maxiter=1e3):
    centroids = init_centroids(data,k,sc)
    t = 0
    while t < maxiter:
        centroids_broadcast = sc.broadcast(centroids)
        centroids_update = data.map(lambda x: cluster_assignment(centroids_broadcast.value,x)).\
            map(lambda x: (x[0],(x[1][1],1))).\
            reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1])).\
            map(lambda x: (x[0],x[1][0]/ x[1][1]))
        loss = centroids_update.\
            map(lambda x: np.linalg.norm(centroids_broadcast.value[x[0]] - x[1])**2).reduce(lambda x,y: x + y)
        print("Iteration "+ str(t) +": Loss=" + str(loss))
        if (loss == float(0)):
            break
        centroids = [x[1] for x in centroids_update.collect()]
        centroids_broadcast.destroy()
        t+=1
    
    
    return {str(i):centroids[i] for i in range(k)}
    return centroids


if __name__ == "__main__":   
    sc = pyspark.SparkContext(appName="clustering")
    print(sc)
    features_rdd = sc.textFile("featureTestData.txt").\
    map(lambda x: x.split(" ")).map(lambda x:np.array([float(c) for c in x])).map(lambda x:(int(x[0]),x[1:]))
    centroids = k_means(data=features_rdd,k=3,sc=sc)
    print(centroids)
    

    
    
    	
    

