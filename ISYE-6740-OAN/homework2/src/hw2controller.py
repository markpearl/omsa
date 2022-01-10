import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as pilim
from numpy import ndarray
import logging
import csv
import os
import scipy.io as sio
from scipy.ndimage import rotate
from src.models.isomap import ISOMap
from src.models.density_estimation import Density_Estimation
from src.models.gmm_model import GMM
import pandas as pd

logger = logging.getLogger(__name__)

class HW2Controller:

    def __init__(self):
        """[Constructor for controller class for HW2]
        """
        self.metrics = ['euclidean','manhattan']

    def plot_isomap_results(self,figure,adj_matrix,simm_matrix,pdf,distance_metric):
        """[Used to plot the results of the ISOMAP algorithm and save them to the outputs folder]

        Args:
            figure ([type]): [Figure for isomap faces]
            adj_matrix ([type]): [Adjacency matrix]
            simm_matrix ([type]): [Similarity matrix]
            pdf: Images pandas dataframe storing numpy representation for each image
            distance_metric ([type]): [Distance metric used]
        """
        try:
            #Plot 3 faces from the pandas dataframe and show the result
            if 'euclidean' in distance_metric:
                image_idxs = np.random.randint(range(3),len(adj_matrix)) 
                cmap = cm.get_cmap('copper_r')
                for index in image_idxs:
                    greyscale = cmap(rotate(pdf.iloc[index,:].values.reshape(64, 64), -90),bytes=True)
                    im = pilim.fromarray(greyscale, mode='RGBA')
                    im.save(f'./reports/faces_{index}.png')
            
            #Plot the adjacency and similiarity matrices and save the outputs to reports
            fig, ax = plt.subplots(1, 2, figsize=(5, 2))
            ax[0].imshow(adj_matrix,cmap=plt.get_cmap('gray'))
            ax[1].imshow(simm_matrix,cmap=plt.get_cmap('gray'))
            plt.title('L1-Norm Isomap Matrices')
            if 'euclidean' in distance_metric:
                plt.title('L2-Norm Isomap Matrices')
            ax[0].set_title('Adjacency Matrix')
            ax[1].set_title('Similarity Matrix')
            fig.savefig('./reports/{0}_matrices.png'.format(distance_metric))
            fig.show()

            #Save output for the isomap implementation
            figure.savefig('./reports/{0}_isomap_result.png'.format(distance_metric))
            figure.show()
        except Exception as e:
            raise(e)  

    def run_isomap(self):
        """[This method runs the ISOMAP algorithm on the following faces data]
        """       
        try: 
            for distance_metric in self.metrics:
                #Read input data
                iso_data = sio.loadmat('./data/isomap.mat')
                #Create ISOMap class and compute adjancey matrix
                iso_map = ISOMap(distance_metric,iso_data)
                adjacency_matrix, pdf = iso_map._calculate_adjacency_matrix(epsilon=100)
                similarity_matrix = iso_map._compute_similarity_matrix(adjacency_matrix)
                z = iso_map._compute_z_matrix(similarity_matrix,2)
                faces_plot = iso_map._display_network_graph(z,pdf,40)
                self.plot_isomap_results(faces_plot,adjacency_matrix,similarity_matrix,pdf,distance_metric)

        except Exception as e:
            raise(e)

    def run_density_estimation(self):
        try:
            #Read input data
            pol_csv = pd.read_csv("./data/n90pol.csv")
            #Create the class variable
            density_estimation = Density_Estimation(pol_csv)
            #Set the number of bins and compute the 2D Histogram for part a
            num_bins = [10,15,20]
            ranges, dataset = density_estimation._calculate_histogram(num_bins)
            #Plot the contour plots
            density_estimation.contour_plots(ranges,dataset)
            #Calculate the colnames to iterate through for the conditional probability plots
            col_names = [col for col in pol_csv.columns if col not in 'orientation']
            density_estimation._calculate_conditional_probabilities(dataset,ranges,col_names)

        except Exception as e:
            raise(e)


    def run_gmm_model(self):
        try:
            #Read the input image digits data
            data  = np.loadtxt("./data/data.dat")
            labels = np.loadtxt("./data/label.dat")
            #Create the GMM class variable and run the methods
            gmm = GMM(data, labels, 2, 30, 100)
            #Display two images from the dataset
            gmm.display_images(data[:, 0],'2image')
            gmm.display_images(data[:, -1],'6image')


        except Exception as e:
            raise(e)