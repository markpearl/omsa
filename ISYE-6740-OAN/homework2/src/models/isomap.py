import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from numpy import ndarray, random, sqrt
import logging
import csv
import os
import math
import networkx as nx
from tqdm import tqdm, tnrange
import PIL.Image as pilim
from scipy.ndimage import rotate

logger = logging.getLogger(__name__)

class ISOMap:

    def __init__(self, distance_metric: str, data: object):
        """[Constructor for isomap algorithm]

        Args:
            distance_metric (str): [description]
            data (object): [description]
        """
        self.distance_metric = distance_metric
        self.data = data

    def _calculate_distance(self, input_array: np.ndarray, metric:str):
        """[Computes the distance for the provided metric for the clustering algorithm]

        Arguments:
            input_array {np.ndarray} -- [Input array to measure the distance between observations]
            metric {str} -- [Metric to use in the calculation (i.e. Euclidean, Manhattan, etc.)]

        Raises:
            NotImplementedError: [description]
        """
        try:
            if metric.upper() in 'EUCLIDEAN':
                dist = np.linalg.norm(input_array,ord=2)
            elif metric.upper() in 'MANHATTAN':
                dist = np.linalg.norm(input_array,ord=1)
            else:
                raise NotImplementedError('This distance metric: {0} has not been implemented yet'.format(metric))
            return dist
        except Exception as e:
            logger.error("Error computing the distance for the input array")
            raise(e)

    def _calculate_adjacency_matrix(self, epsilon:int):
        """[This method calculates the distance between each image to create the similarity graph for the isomap algorithm]  
        
        Arguments:
            epsilon {int} -- [Threshold used to determine how many neighbors each given node will have]        
        """      
        try:  
            #Convert shape of faces data to pandas dataframe and reshape to transpose
            pdf = pd.DataFrame(self.data['images']).T
            
            #Calculate distance matrix for each image observations to form nearest neighbors
            dist_matrix = np.zeros((pdf.shape[0],pdf.shape[0]))

            logger.info("Now calculating the weighted/distance matrix for each image in the graph and calculating the nearest neighbors")
            for idx, row in tqdm(pdf.iterrows(),total=pdf.shape[0]):
                for sub_idx, sub_row in pdf.iterrows():
                    #Iterate through each observation
                    dist_matrix[idx][sub_idx] = self._calculate_distance(pdf.iloc[idx]-pdf.iloc[sub_idx],'euclidean')
                    #If the current index for the matrix is in the first minimum 100 observations then grab 101 nodes (i.e 101 - current observation = 100)
                if idx in np.argpartition(dist_matrix[idx],epsilon+1):
                    np.put(dist_matrix[idx],np.argpartition(dist_matrix[idx],epsilon+1)[epsilon+1:],[999999])
                else:
                    np.put(dist_matrix[idx],np.argpartition(dist_matrix[idx],epsilon)[epsilon:],[999999])
            
            return dist_matrix, pdf
        
        except Exception as e:
            logger.error("Error calculating the weighted graph for the nearest neighbors")
            raise(e)

    def _compute_similarity_matrix(self,W):
        """[Generate Graph and Obtain Matrix D from weight matrix W defining the weight on the edge between each pair of nodes. Note that you can assign sufficiently large weights to non-existing edges.]

        Args:
            W ([type]): [Weighted matrix corresponding to all of the images as nodes, and their corresponding distances to all other observations.]

        Returns:
            [D]: [Distance matrix for each node]
        """
        try:
            n = np.shape(W)[0]
            Graph = nx.DiGraph()
            logger.info('Now computing the similarity graph based on the nearest neighbors weight graph')
            list_graph_shape = list(range(n))
            for i in tqdm(list_graph_shape):
                for j in list_graph_shape:
                    Graph.add_weighted_edges_from([(i,j,min(W[i,j], W[j,i]))])

            logger.info("Now calculating the shortest distance path for each observations using the Dijkstra algorithm")
            res = dict(nx.all_pairs_dijkstra_path_length(Graph))
            D = np.zeros([n,n])
            for i in range(n):
                for j in range(n):
                    D[i,j] = res[i][j]
            return D
        except Exception as e:
            logger.error("Error computing the similarity graph based on the weighted nearest neighbors graph")
            raise(e)

    def _compute_z_matrix(self, D, k):
        """[Method computes the centered matrix and the principal componets for the Weights image matrix]

        Args:
            W ([type]): [W weighted matrix for the calculation]
            k : Number of principal components to select
        """                    
        try:
            #Compute the D matrix
            D = (D + D.T)/2
            #Comput the ones matrix used as an input to calculate the centered matrix H
            n = D.shape[1]
            ones = np.ones([n,1])
            H = np.eye(n) - 1/n*ones.dot(ones.T)
            #Calculate the Centered matrix
            C = -H.dot(D**2).dot(H)/(2*n)
            eig_val, eig_vec = np.linalg.eig(C)

            #Select the 
            index = np.argsort(-eig_val)
            Z = eig_vec[:,index[0:k]].dot(np.diag(np.sqrt(eig_val[index[0:2]])))
            Z_pdf = pd.DataFrame(Z, columns=['Component 1', 'Component 2'])
            return Z_pdf

        except Exception as e:
            raise(e)

    def _display_network_graph(self, Z: object, pdf, sample: int):
        """[Displays networkx graph containing the node and edge relationships for the given graph G]

        Args:
            Z (object): [Object containing the Z eigenvectors to display]
            pdf {DataFrame} : [Pandas dataframe containing the original images]
            sample (int): [Number of nodes/images in the graph to display]
        """ 
        try:
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            ax = fig.add_subplot(111)
            num_images, num_pixels = pdf.shape
            pixels_per_dimension = int(math.sqrt(num_pixels))
            ax.set_title('L1 Norm 2D Components from Isomap of Facial Images')
            if self.distance_metric in 'euclidean':
                ax.set_title('L2 Norm 2D Components from Isomap of Facial Images')
            ax.set_xlabel('Component: 1')
            ax.set_ylabel('Component: 2')

            # Show 40 of the images ont the plot
            x_size = (max(Z['Component 1']) - min(Z['Component 1'])) * 0.08
            y_size = (max(Z['Component 2']) - min(Z['Component 2'])) * 0.08
            for i in range(sample):
                img_num = np.random.randint(0, num_images)
                x0 = Z.loc[img_num, 'Component 1'] - (x_size / 2.)
                y0 = Z.loc[img_num, 'Component 2'] - (y_size / 2.)
                x1 = Z.loc[img_num, 'Component 1'] + (x_size / 2.)
                y1 = Z.loc[img_num, 'Component 2'] + (y_size / 2.)
                img = rotate(pdf.iloc[img_num,:].values.reshape(pixels_per_dimension, pixels_per_dimension),-90)
                ax.imshow(img, aspect='auto', cmap=plt.cm.gray, extent=(x0, x1, y0, y1))

            # Show 2D components plot
            ax.scatter(Z['Component 1'], Z['Component 2'], marker='.',alpha=0.7)

            ax.set_ylabel('Up-Down Pose')
            ax.set_xlabel('Right-Left Pose')
            return plt

        except Exception as e:
            logger.error("Error displaying the networkx graph")
            raise(e)