import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import ndarray
import logging
from clustering.kmeans import KMeans
from clustering.kmedoids import KMedoids
import csv
import os

logger = logging.getLogger(__name__)

class ClusteringController:
    """[Class containing the controller logic to determine which clustering algorithm to use]
    """
    def __init__(self, k: int, img: object, algorithm: str, iterations: int, img_name: str):
        """[Constructor of the class used to instantiate the K-Medoids object with a k number of centroids
        and a img array]

        Arguments:
            k {int} -- [Numbers of clusters to be chosen]
            img {object} -- [Original image that is being used for clustering]
            algorithm {str} -- [Choice of algorithm: kmeans, kmedoids or both]
            iterations {int} -- [Number of iterations to run the algorithm or until it converges]
            img_name {str} -- [Name of image]
        """
        self.k = k
        self.img = img
        self.algorithm = algorithm
        self.kmedoids = KMedoids(self.k, self.img)
        self.kmeans = KMeans(self.k, self.img)
        self.iterations = iterations
        self.img_name = img_name

    def run_clustering_algorithm(self):
        """[Creates the class for the corresponding algorithm and runs the results to 
        get the final medoid/or centroids and cluster assignments]

        Returns:
            [type] -- [Medoids/centroids depending on the algorithm chosen and the cluster assignment
            for each observation / labels]
        """        
        try:
            if self.algorithm.upper() in 'KMEDOIDS':
                points, labels, iteration = self.kmedoids.k_medoids(iterations=self.iterations)
            elif self.algorithm.upper() in 'KMEANS':
                points, labels, iteration = self.kmeans.k_means(iterations=self.iterations)
            else:
                raise NotImplementedError('This algorithm has not yet been implemented yet')
            return points, labels, iteration

        except Exception as e:
            raise(e)

    def save_image(self, points: ndarray, labels: ndarray, iteration: int): 
        """[Reconstructs the image from the clustering medoids or centroids
        and the cluster assignments then saves the image]

        Arguments:
            points {ndarray} -- [Centroids or medoids]
            labels {ndarray} -- [Cluster assignments/labels for each observation]
            iteration {int} -- [Iteration number the algorithm converged]
        """
        try:

            # recovering the compressed image by 
            # assigning each pixel to its corresponding centroid. 
            final_medoids = np.array(points) 
            recovered = final_medoids[labels.astype(int), :] 
            
            # getting back the 3d matrix (row, col, rgb(3)) 
            recovered = np.reshape(recovered, (self.img.shape[0], self.img.shape[1], self.img.shape[2]))
        
            # plotting the compressed image. 
            plt.imshow(recovered) 
            filename = 'compressed_{0}_{1}_iter{2}_{3}.png'.format(self.algorithm,str(self.k),str(self.iterations),self.img_name.split('/')[-1].split('.')[0])
            results_list = [self.algorithm, self.k, iteration, self.iterations,filename]

            #Write results of clustering run to the csv file    
            if not os.path.exists('./clustering_results.csv'):
                csv_header = ['Algorithm Name','Number of Clusters (K)','Number of Iterations Ran','Total Iterations','Image Name']
                with open('./clustering_results.csv', 'w',newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(csv_header)
                    wr.writerow(results_list)
            else:
                with open('./clustering_results.csv', 'a',newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(results_list)

            # saving the compressed image. 
            plt.imsave(filename,recovered)

        except Exception as e:
            raise(e)