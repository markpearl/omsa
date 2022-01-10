import pandas as pd
import matplotlib as plt
import numpy as np
import scipy.misc as spm
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.spatial.distance import cdist
from sklearn import cluster
from skimage import color,io
from skimage.filters import threshold_otsu,threshold_multiotsu,prewitt,sobel
from skimage.data import camera
from skimage.util import compare_images
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from skimage import color, io
from scipy.ndimage.filters import convolve
import logging

logger = logging.getLogger(__name__)

class CannyAlgorithm:
    def __init__(self):
        #Input gaussian filter to smooth the input image
        self.h = 1/1115* np.array([[1,4,7,10,7,4,1],
                                  [4,12,26,33,26,12,4],
                                  [7,26,55,71,55,26,7],
                                  [10,33,71,91,71,33,10],
                                  [7,26,55,71,55,26,7],
                                  [4,12,26,33,26,12,4],
                                  [1,4,7,10,7,4,1]])
        #Load grayscale image                                  
        self.grayscale = color.rgb2gray(io.imread('./horse1-2.jpg'))*255

    def gaussian_filter(self):
        """[Apply gaussian filter to image for smoothing purposes and add 0 padding to matrix]
        """
        logger.info("Creating smoothed matrix from applying gaussian filter")        
        return np.pad(convolve(self.grayscale,self.h), ((1,1),(1,1)),'constant')

    def compute_partial_derivatives(self, s_i1_j: float, s_i_j: float, s_i1_j1: float, s_i_j1:float):
        """[Compute partial derivate for pixel with respect to x]

        Args:
            si_j1 (float): [Pixel directly to the east in image matrix]
            s_i_j (float): [Current pixel]
            s_i1_j1 (float): [Pixel directly in south-east coordinate of image in relation to pixel]
            s_i1_j (float): [Pixel directly south of pixel image matrix]
        """
        deriv_x = 1/2*(s_i1_j-s_i_j+s_i1_j1-s_i_j1)
        deriv_y = 1/2*(s_i_j1-s_i_j+s_i1_j1-s_i1_j)
        return deriv_x, deriv_y
    
    def compute_gradients(self, S: np.ndarray):
        """[Iterate through each pixel and calculate the gradient direction and magnitute]

        Args:
            S (np.ndarray): [Smoothed image S representing the output of the grayscale image after
            gaussian filter has been applied]
        """
        logger.info("Computing the gradient magnitude matrices G and theta, respectively")
        #Create gradient magnitute array
        G = np.zeros(S.shape)
        #Create theta for gradient direction
        theta = np.zeros(S.shape)
        #Iterate through each pixel and calculate the magnitute and direction
        for i in range(1,S.shape[0]-1):
            for j in range(1,S.shape[1]-1):
                s_i_j = S[i,j]
                s_i1_j1 = S[i+1,j+1]
                s_i1_j = S[i+1,j]
                s_i_j1 = S[i,j+1]
                #Compute partial derivative for wrt to x and y
                x,y = self.compute_partial_derivatives(s_i1_j, s_i_j, s_i1_j1, s_i_j1)
                #Compute gradient magnitude
                G[i,j] = np.sqrt((x**2)+(y**2))
                #Compute gradient direction
                theta[i,j] = np.arctan2(x,y)
        plt.imsave('q3_G.png', G, cmap=plt.cm.gray)
        plt.imsave('q3_theta.png', theta, cmap=plt.cm.gray)                     
        return G, theta


    def compute_nonmaximal_suppresion(self, G: np.ndarray, theta: np.ndarray):
        """[Nonmaximal suppression method]

        Args:
            G (np.ndarray): [Gradient magnitude matrix]
            theta (np.ndarray): [Gradient direction matrix]
        """        
        #Create gradient magnitute array
        phi = np.zeros(theta.shape)
        #Iterate through each pixel and calculate the magnitute and direction
        logger.info("Computing non-max suppression phi output")
        for i in range(1,theta.shape[0]-1):
            for j in range(1,theta.shape[1]-1):
                #Iterate through each row in theta
                if (-1/8*np.pi < theta[i,j] <= 1/8*np.pi):
                    i_1 = (i,j-1)
                    i_2 = (i,j+1)
                elif (1/8*np.pi < theta[i,j] <= 3/8*np.pi):
                    i_1 = (i+1,j-1)
                    i_2 = (i-1,j+1)
                elif (-3/8*np.pi < theta[i,j] <= -1/8*np.pi):
                    i_1 = (i-1,j-1)
                    i_2 = (i+1,j+1)
                elif ((3/8*np.pi < theta[i,j] <= 1/2*np.pi) or (-1/2*np.pi < theta[i,j] <= -3/8*np.pi)):
                    i_1 = (i-1,j)
                    i_2 = (i+1,j)  
                elif G[i,j] >= G[i_1] and G[i,j] >= G[i_2]:
                    phi[i,j]=G[i,j]
        plt.imsave('q3_phi.png', phi, cmap=plt.cm.gray)                               
        return phi

    def hysterisis(self, non_max_supress: np.ndarray):
        """[Run threshold with hysterisis to mitigate streaking]

        Args:
            G (np.ndarray): [description]
            theta (np.ndarray): [description]
        """        
        #Create gradient magnitute array
        logger.info("Computing the hysterisis step and outputting the final edge image.")
        edge_image = np.zeros(non_max_supress.shape)
        tau_1, tau_2 = 3, 8
        
        #Iterate through each pixel and calculate the magnitute and direction
        for i in range(1,non_max_supress.shape[0]-1):
            for j in range(1,non_max_supress.shape[1]-1):

                if non_max_supress[i,j] >= tau_2 and edge_image[i,j]==0:
                    edge_image[i,j]=1
                elif non_max_supress[i,j] >= tau_1 and edge_image[i,j]==0:
                    neighbors = []
                    neighbors.append(edge_image[i-1,j-1])
                    neighbors.append(edge_image[i-1,j])
                    neighbors.append(edge_image[i-1,j+1])
                    neighbors.append(edge_image[i,j+1])
                    neighbors.append(edge_image[i+1,j+1])
                    neighbors.append(edge_image[i+1,j])
                    neighbors.append(edge_image[i+1,j-1])
                    neighbors.append(edge_image[i,j-1])

                    if 1 in neighbors:
                        edge_image[i,j]=1
        plt.imsave('q3_edge_image.png', edge_image, cmap=plt.cm.gray)                
        return edge_image   
        

if __name__ in "__main__":
    logging.basicConfig(level=logging.INFO)
    canny = CannyAlgorithm()
    #Apply gaussian filter to input image in order to smooth
    smooth_image = canny.gaussian_filter()
    #Compute gradient direction and magnitute vectors
    G, theta = canny.compute_gradients(smooth_image)
    #Non-max suppression
    non_max_suppression = canny.compute_nonmaximal_suppresion(G, theta) 
    #Calculate hysterisis
    edge_image = canny.hysterisis(non_max_suppression)