import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class KMedoids:
    """[Class containing implementation of K-Medoids algorithm. A type of clustering model that differs
    from K-Means based on it's initial choice of centroids to a known point rather than an averaged point.]
    """

    def __init__(self, k: int, img: object):
        """[Constructor of the class used to instantiate the K-Medoids object with a k number of centroids
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
            NotImplementedError: [description]
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

    def _compute_initial_medoids(self):
        """[This method computes the initial centroids to be used to the k-medoids algorithm]

        Returns:
            [reshaped] -- [Reshaped 2-dimensional numpy array]
            [medoids] -- [Array containing initial medoids/centers for the algorithm]
        """
        try:
            logger.info("Computing the initial centroids for the K-Medoids algorithm")
            #Reshape the current matrix to a 2D Representation of RBG coordinates
            reshaped = np.reshape(self.img,(self.img.shape[0] * self.img.shape[1],self.img.shape[2]))

            #Determine the numbers of rows and columns in the newly shaped array
            n_rows, n_cols = reshaped.shape

            #Create empty np array that is number of clusters by number of columns
            medoids = np.zeros((self.k,n_cols))

            #Iterate through each index and initialize the centroids array with a random observation from the
            for index in range(self.k):
                random_int = int(np.random.random(1)*n_rows)
                medoids[index] = reshaped[random_int]

            return reshaped, medoids

        except Exception as e:
            logger.error("Error running the process to compute the intial means / centroids of the existing clusters")
            raise(e)

    def _calculate_distance_to_medoids(self, X: np.ndarray, medoids: np.ndarray, metric: str):
        """[Methods traverses through each point and determines which medoid/cluster it belongs to]

        Arguments:
            X {np.ndarray} -- [2-D array of observations containing RGB coordinates]
            medoids {np.ndarray} -- [Array containing initial medoids/centers for the algorithm]
            metric {str} -- [Metric to use in the calculation (i.e. Euclidean, Manhattan, etc.)]
        """
        try:
            logger.info("Calculating the distance for an array of observations to the medoids")
            #Initialize the variables for number of observations and number of clusters
            m = len(X)
            k = len(medoids)
            #Initialize an empty numpy matrix to calculate the distance between each observation and the 1 to k medoids
            dist_matrix = np.empty((m, k))
            #Traverse through each observation and calculate the distance
            for index in range(m):
                distance = self._calculate_distance(X[index,:]-medoids,metric)
                dist_matrix[index,:] = distance
                if metric.upper() in 'EUCLIDEAN':
                    dist_matrix[index,:] = distance**2
            return dist_matrix

        except Exception as e:
            logger.error("Error running the process to compute the intial means / centroids of the existing clusters")
            raise(e)

    def _update_medoids(self, dist_matrix: np.ndarray, X: np.ndarray, medoids: np.ndarray, metric: str):
        """[Methods traverses through each point and determines which medoid/cluster it belongs to]

        Arguments:"
            dist_matrix {np.ndarray} -- [description]
            X {np.ndarray} -- [2-D array of observations containing RGB coordinates]
            medoids {np.ndarray} -- [Array containing initial medoids/centers for the algorithm]
            metric {str} -- [Metric to use in the calculation (i.e. Euclidean, Manhattan, etc.)]
        """
        try:
            logger.info("Now updating the medoids to the medoids with the argmin sum of errors")
            #Distance matrix containing the distance between each observation and the medoids
            cluster_assignments = np.argmin(dist_matrix, axis=1)
            #Instantiate new medoids to be calculated
            new_medoids = np.zeros(medoids.shape)

            #Iterate through each cluster of the k clusters
            for k in np.unique(cluster_assignments):
                logger.info("Now updating cluster {0} out of {1} clusters".format(str(k+1),str(len(np.unique(cluster_assignments)))))
                #For each cluster filter the observations of the cluster by the cluster assignment matrix
                cluster_observations = X[np.where(cluster_assignments==k)]
                #Take sample of the dataset
                sample_idx = np.random.randint(len(cluster_observations), size=int(len(cluster_observations)/10))
                sample_observations = cluster_observations[sample_idx,:]
                #Measure the sum of distances of the cluster's existing medoids to all other observations in the cluster
                medoids_cluster_distance = sum(self._calculate_distance(sample_observations-medoids[k],metric))
                #Iterate through each 
                sum_distances_cluster = []
                cluster_len = len(sample_observations)
                for idx, x in enumerate(tqdm(sample_observations)):
                    if idx < cluster_len:
                        #Measure the sum of distances of this observation to all other observations in the cluster
                        sum_distances_x = sum(self._calculate_distance(sample_observations-x,metric))
                        #Compare if the sum of distances of all observations in the cluster to this point is < then the medoid distance
                        sum_distances_cluster.append(sum_distances_x)
                #Convert the list into a numpy array and determine the smallest value, compare that against the medoids sum of distances, 
                #If the sum of distances of the observation is less than the medoids, then swap the medoid with the observation
                if np.amin(np.array(sum_distances_cluster)) < medoids_cluster_distance:
                    new_medoids[k] = sample_observations[np.argmin(np.array(sum_distances_cluster))]

                else:
                    new_medoids[k] = medoids[k]
            return new_medoids, cluster_assignments

        except Exception as e:
            raise(e)

    def _convergence(self, old_medoids, new_medoids):
        """[Determine convergence by determining if the medoid values on the previous iteration
        are the same as the existing iteration]

        Arguments:
            old_medoids {[type]} -- [description]
            new_medoids {[type]} -- [description]
        """
        return set([tuple(x) for x in np.around(old_medoids,decimals=5)]) == set([tuple(x) for x in np.around(new_medoids,decimals=5)])

    def k_medoids(self, iterations: int=5):
        """[Main method that will run the full k-medoids algorithm, which will run either
        until the max iterations is hit or convergence for the medoids]

        Keyword Arguments:
            iterations {int} -- [Number of iterations to recompute the medoids] (default: {5})
        """
        #Reshape the image data to be a 2D instead of 3D array and initialize the medoids
        reshaped, medoids = self._compute_initial_medoids()
        
        convergence = False
        iteration = 1
        while(not convergence) and (iteration <= iterations):
            logger.info("Running the {0} iteration of kmedoids".format(str(iteration)))
            #Make copy of medoids so they can be later compared to update medoids to measure convergence
            prev_medoids = medoids.copy()

            #Calculate the distance to the of each obversations to the medoids     
            dist_matrix = self._calculate_distance_to_medoids(reshaped, medoids, 'euclidean')

            #Based on the cluster assignment, iterate through the observations in each cluster and update the medoids
            medoids, cluster_assignments = self._update_medoids(dist_matrix,reshaped,medoids,'euclidean')                
            
            #Determine if the old vs the new medoids have converged
            convergence = self._convergence(prev_medoids,medoids)
            if convergence:
                logger.info("The algorithm converged at {0} iteration of kmedoids".format(str(iteration)))
                return medoids,cluster_assignments,iteration
            elif iteration == iterations:
                logger.info("The algorithm has ran through the max iterations")
                return medoids,cluster_assignments,iteration
            else:
                iteration +=1