import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
import math
from sklearn.neighbors import KernelDensity
import logging

logger = logging.getLogger(__name__)

class Density_Estimation:

    def __init__(self, data: object):
        self.data = data

    def _calculate_histogram(self, bins: list):
        """[Histogram calculates the distribution for each variable and returns the sequence
        of each variable in its associated bin]

        Arg:
            bin {list} [List of number of bins to use for the histogram plot]
        Returns:
            [plt, ranges]: [Return the histogram and the range for the variables]
        """
        try:
            #Get the unique values for the orientation variable to form the range
            ranges = list(set(self.data['orientation']))
            #Iterate through the range and create the dataset variable, containing the frequency
            #each time an orientation value occurs
            dataset = []
            for i in range(len(ranges)):
                dataset.append(self.data[self.data['orientation'] == ranges[i]])

            # plot the histogram
            x = self.data['amygdala']
            y = self.data['acc']
            # get the conditional probability for each variable in the variable_name list and show the plot
            col = len(bins)
            row = 1
            fig, ax = plt.subplots(figsize = (30,10), ncols = col, nrows = row)
            for i in range(len(bins)):
                ax[i].set_title("2-D Histogram for Amygdala by Acc: {0} bins".format(str(bins[i])))
                ax[i].set_xlabel('Amygdala X')
                ax[i].set_ylabel('Acc Y')
                ax[i].hist2d(x, y, bins=(bins[i], bins[i]), cmap=plt.cm.copper)
            plt.savefig('./outputs/conditional_prob.png')
            return ranges,dataset

        except Exception as e:
            raise(e)

    def contour_plots(self, ranges: list, dataset: list):
        """[Plot the contour plots for each permutation of the acc and amy]

        Args:
            ranges ([list]): [Contains the unique values for the orientation variable to iterate through]
            dataset ([list]): [Dataset containing the frequency for each orientation value]
        """         
        try:
            col = math.ceil(np.sqrt(len(ranges)))
            row = math.ceil(np.sqrt(len(ranges)))
            #Comput the number of rows and cols to get the dimensions for the number of sublots of show
            fig, ax= plt.subplots(figsize = (10,10), ncols = col, nrows = row)
            #Iterate through the length of ranges to get the contour plot for each distinct value of the orientation 
            #Variable
            for i in range(len(ranges)):
                #Set the title for the Orientation Value
                ax[int(i/col)][i%col].set_title("Orientation Value " + str(ranges[i]))
                #Utilize the seaborn library to calculate the contour plot for the given orientation value
                #and assigned it to the subplot with the axis variable
                sns.kdeplot(dataset[i]["amygdala"], dataset[i]["acc"], ax = ax[int(i/col)][i%col])
            plt.savefig('./outputs/kde_contour_plot.png')
        except Exception as e:
            raise(e)

    def _calculate_conditional_probabilities(self,dataset: list,ranges: list,variable_names: list):
        """[Calculate the conditional probabilities for each variable in the variable_name list]

        Args:
            ranges ([list]): [Contains the unique values for the orientation variable to iterate through]
            dataset ([list]): [Dataset containing the frequency for each orientation value]
            variable_names ([list]): [Dataset containing the frequency for each orientation value]            
        """
        try:
            #Calculate the conditional probablity across each dataset / orientation distinct value
            prob_each_ori = np.zeros(4)
            for i in range(4):
                prob_each_ori[i] = len(dataset[i])/len(self.data)

            # get the conditional probability for each variable in the variable_name list and show the plot
            col = len(variable_names)
            row = 1
            fig, ax = plt.subplots(figsize = (20,10), ncols = col, nrows = row)
            for i in range(len(variable_names)):
                for j in range(len(ranges)):
                    ax[i].set_title("Conditional Probablity Plot: " + str(variable_names[i]))
                    sns.kdeplot(dataset[j][variable_names[i]],label = 'Orientation ' + str(ranges[j]),ax = ax[i])
            return plt.savefig('./outputs/conditional_prob.png') 
        except Exception as e:
            raise(e)