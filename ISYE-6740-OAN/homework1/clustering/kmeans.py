import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class KMeans:
    """[Class containing implementation of KMeans algorithm. A type of clustering model that selects centroids
     based on a point comprising of the average distances across the points of a cluster, determined by the selected
     distance metric]
    """
    def __init__(self, k: int, img: np.ndarray):
        """[Constructor of the class used to instantiate the K-centroids object with a k number of centroids
        and a img array]

        Arguments:
            k {int} -- [Numbers of clusters to be chosen]
            img {np.ndarray} -- [Array of RGB coordinates for an image]
        """
        self.k = k
        self.img = img

    def _calculate_distance(self, input_array: np.ndarray, metric:str):
        """[Computes the distance for the provided metric for the clustering algorithm]

        Arguments:
            metric {str} -- [Metric to use in the calculation (i.e. Euclidean, Manhattan, etc.)]

        Raises:
            NotImplementedError: [Raised when a metric calculation hasn't been implemented yet]
        """
        try:
            if metric.upper() in 'EUCLIDEAN':
                dist = np.linalg.norm(input_array,ord=2,axis=1)
            elif metric.upper() in 'MANHATTAN':
                dist = np.linalg.norm(input_array,ord=1,axis=1)
            else:
                raise NotImplementedError('This distance metric: {0} has not been implemented yet'.format(metric))
            return dist
        except Exception as e:
            logger.error("Error computing the distance for the input array")
            raise(e)

    def _compute_initial_centroids(self):
        """[This method computes the initial centroids to be used to the k-centroids algorithm]

        Returns:
            [reshaped] -- [Reshaped 2-dimensional numpy array]
            [centroids] -- [Array containing initial centroids/centers for the algorithm]
        """
        try:
            logger.info("Computing the initial centroids for the K-centroids algorithm")
            #Reshape the current matrix to a 2D Representation of RBG coordinates
            reshaped = np.reshape(self.img,(self.img.shape[0] * self.img.shape[1],self.img.shape[2]))

            #Determine the numbers of rows and columns in the newly shaped array
            n_rows, n_cols = reshaped.shape
            centroids = np.zeros((self.k,n_cols))

            #Create initial centroids based on random values
            centroids = np.random.uniform(np.min(np.min(reshaped,axis=0)),100,centroids.shape)/100
            return reshaped, centroids

        except Exception as e:
            logger.error("Error running the process to compute the intial means / centroids of the existing clusters")
            raise(e)

    def _calculate_distance_to_centroids(self, X: np.ndarray, centroids: np.ndarray, metric: str):
        """[Methods traverses through each point and determines which medoid/cluster it belongs to]

        Arguments:
            X {np.ndarray} -- [2-D array of observations containing RGB coordinates]
            centroids {np.ndarray} -- [Array containing initial centroids/centers for the algorithm]
            metric {str} -- [Metric to use in the calculation (i.e. Euclidean, Manhattan, etc.)]
        """
        try:
            logger.info("Calculating the distance for an array of observations to the centroids")
            #Initialize the variables for number of observations and number of clusters
            m = len(X)
            k = len(centroids)
            #Initialize an empty numpy matrix to calculate the distance between each observation and the 1 to k centroids
            dist_matrix = np.empty((m, k))
            #Traverse through each observation and calculate the distance
            for index in range(m):
                distance = self._calculate_distance(X[index,:]-centroids,metric)
                dist_matrix[index,:] = distance
                if metric.upper() in 'EUCLIDEAN':
                    dist_matrix[index,:] = distance**2
            return dist_matrix

        except Exception as e:
            logger.error("Error running the process to compute the intial means / centroids of the existing clusters")
            raise(e)

    def _update_centroids(self, dist_matrix: np.ndarray, X: np.ndarray, centroids: np.ndarray, metric: str):
        """[Methods traverses through each point and determines which medoid/cluster it belongs to]

        Arguments:"
            dist_matrix {np.ndarray} -- [description]
            X {np.ndarray} -- [2-D array of observations containing RGB coordinates]
            centroids {np.ndarray} -- [Array containing initial centroids/centers for the algorithm]
            metric {str} -- [Metric to use in the calculation (i.e. Euclidean, Manhattan, etc.)]
        """
        try:
            logger.info("Now updating the centroids to the centroids with the new mean value for each cluster")
            #Distance matrix containing the distance between each observation and the centroids
            cluster_assignments = np.argmin(dist_matrix, axis=1)
            #Instantiate new centroids to be calculated
            new_centroids = np.zeros(centroids.shape)

            #Iterate through each cluster of the k clusters
            for idx,k in enumerate(tqdm(np.unique(cluster_assignments))):
                logger.info("Now updating cluster {0} out of {1} clusters".format(str(k+1),str(len(np.unique(cluster_assignments)))))
                #For each cluster filter the observations of the cluster by the cluster assignment matrix
                cluster_observations = X[np.where(cluster_assignments==k)]
                #Add the new means of the cluster in the centroid array
                new_centroids[k] = np.mean(cluster_observations,axis=0)
            return new_centroids, cluster_assignments

        except Exception as e:
            raise(e)

    def _convergence(self, old_centroids, new_centroids):
        """[Determine convergence by determining if the medoid values on the previous iteration
        are the same as the existing iteration]

        Arguments:
            old_centroids {[type]} -- [description]
            new_centroids {[type]} -- [description]
        """
        return set([tuple(x) for x in np.around(old_centroids,decimals=5)]) == set([tuple(x) for x in np.around(new_centroids,decimals=5)])

    def k_means(self, iterations: int=5):
        """[Main method that will run the full k-centroids algorithm, which will run either
        until the max iterations is hit or convergence for the centroids]

        Keyword Arguments:
            iterations {int} -- [Number of iterations to recompute the centroids] (default: {5})
        """
        #Reshape the image data to be a 2D instead of 3D array and initialize the centroids
        reshaped, centroids = self._compute_initial_centroids()
        
        convergence = False
        iteration = 1
        while(not convergence) and (iteration <= iterations):
            logger.info("Running the {0} iteration of kmeans".format(str(iteration)))
            #Make copy of centroids so they can be later compared to update centroids to measure convergence
            prev_centroids = centroids.copy()

            #Calculate the distance to the of each obversations to the centroids     
            dist_matrix = self._calculate_distance_to_centroids(reshaped, centroids, 'euclidean')

            #Based on the cluster assignment, iterate through the observations in each cluster and update the centroids
            centroids, cluster_assignments = self._update_centroids(dist_matrix,reshaped,centroids,'euclidean')                
            
            #Determine if the old vs the new centroids have converged
            convergence = self._convergence(prev_centroids, centroids)
            if convergence:
                logger.info("The algorithm converged at {0} iteration of kmeans".format(str(iteration)))
                return centroids,cluster_assignments,iteration
            elif iteration == iterations:
                logger.info("The algorithm has ran through the max iterations")
                return centroids,cluster_assignments,iteration
            else:
                iteration +=1