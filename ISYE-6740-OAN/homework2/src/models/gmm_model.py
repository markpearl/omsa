import arrow
import numpy as np 
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, data, label, C, iterations=100, r=50):
        """[summary]

        Args:
            data ([type]): [Data containing the images of 2 and 6's, respectively]
            label ([type]): [Data containing the labels of each image]
            C ([type]): [Covariance matrix]
            iterations (int): [Number of iterations to run the model for]
            r (int): [low-rank approximation parameter)
        """        
        self.data      = data
        self.label     = label
        self.C         = C
        self.iterations   = iterations
        self.r         = r
        self.log_likelihood   = []

    def display_images(self, img_row, img_name):
        """[This method retrieves a ndarray represesntation of an image and displays
        its reshaped contents]

        Args:
            img_row ([ndarray]): [Numpy ndarray with image data to display]
            img_name (str): [Determines if it's a 2 or 6 image]
        """        
        try:

            imdata   = np.reshape(img_row, newshape=(28, 28),order='F')
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            ax.imshow(imdata)
            plt.savefig('./outputs/{0}_img.png'.format(img_name))
            plt.close()
        except Exception as e:
            raise(e)