import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import csv
import os
from src.models.train_model import TrainModel
from src.models.predict_model import PredictModel
from src.data.make_dataset import read_data
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
from matplotlib.colors import ListedColormap
import plotly.figure_factory as FF
import chart_studio.plotly as py

logger = logging.getLogger(__name__)

class HW5Controller:
    
    def __init__(self, datasets: str,model_names:str):
        """[Constructor class for the controller which will either train or predict a given model 
        based on the dataset name retrieved from the run script.]

        Args:
            model_names (str): [Name of models/classifiers to be used]
            datasets (str): [This dataset string determine which dataset to import for the question]
            num_features (str): [Number of features to use when preparing the X observations / features vectors]
        """        
        self.datasets = datasets
        self.model_names = model_names
        
    def run_model(self):
        """[Call method to train the given model selected]
        """
        try:
            #Initialize dataframe to store the model accuracy
            columns = ['Model_Name','Model_Accuracy','Dataset']
            df = pd.DataFrame([],columns=columns)
            #Initialize X and Y data for the dataset
            X,Y = read_data(self.datasets,None)
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
            #Only run the model on all feature for the digits dataset
            cart_results = []
            forest_results = []
            for model in self.model_names:
                #Initialize train model class
                logger.info('Now training the {0} model for the {1} dataset.'.format(model,self.datasets))
                trained_model = TrainModel(X_train,Y_train,X_test,Y_test)
                
                if model in 'cart':
                    cart_results = getattr(trained_model,f'_train_{model}')(depth=60)
                else:
                    forest_results = getattr(trained_model,f'_train_{model}')(depth=60)
                
            self.plot_results(cart_results,forest_results,model)



        except Exception as e:
            logger.error("Error running the controller method to fit/train models and retrieve predictions")
            raise(e)

    def plot_results(self,cart_results:list,forest_results:list,model_name:str):
        """[Plot the decision boundary for the fitted model]

        Args:
            cart_results (list) : [List containg roc auc score for cart]
            forest_results (list) : [List containg roc auc score for random forest]
            model_name (str): [Model name to plot the decision boundary result]
        """
        try:
            plt.figure()
            plt.plot(cart_results,label='CART / Decision Tree')
            plt.plot(forest_results,label='Random Forest')
            plt.title("Max Depth versus AUC")
            plt.xlabel("Max Depth for Each Tree for {0}".format(model_name))
            plt.ylabel("AUC Score for {0}".format(model_name))
            plt.legend()
            plt.savefig('./reports/figures/rf_cart_auc.png')

        except Exception as e:
            logger.error('Error plotting the decision boundary for the following model')
            raise(e)