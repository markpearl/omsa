import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import csv
import os
from src.models.train_model import TrainModel
from src.models.predict_model import PredictModel
from src.features.feature import FeatureEngineering
from src.data.make_dataset import read_data
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
from matplotlib.colors import ListedColormap
import plotly.figure_factory as FF
import chart_studio.plotly as py 

logger = logging.getLogger(__name__)

class Controller:
    
    def __init__(self, dataset: str,model_names:str,time_col_names:list):
        """[Constructor class for the controller which will either train or predict a given model 
        based on the dataset name retrieved from the run script.]

        Args:
            dataset (str): [This dataset string determine which dataset to import for the question]
            model_names (str): [Name of models/classifiers to be used]
            time_col_names (list): [List representing the time column names]
        """        
        self.X,self.Y = read_data(dataset)
        self.model_names = model_names
        self.time_col_names = time_col_names

    def run_feature_processing(self, ordinal_encoding:bool, one_hot_ecoding:bool, corr_features:bool):
        """[Method executed to run any feature engineering or extraction on these existing training datasets]

        Args:
            ordinal_encoding (bool): [Boolean variable with True/False indicator is this step will be run]
            one_hot_ecoding (bool): [Boolean variable with True/False indicator is this step will be run]
            corr_features (bool): [Boolean variable with True/False indicator is this step will be run]
        """  
        try:
            #Create Feature engineering class to run feature engineering and extraction operations on our training dataset
            feature_proc = FeatureEngineering(self.X)
            
            #Replace missing and NaN values contained in the data
            X = feature_proc._replace_nan_values(self.X)

            #Iterate through the ordinal column names and conduct ordinal encoding
            if ordinal_encoding:
                X = feature_proc._ordinal_ecoding(X)
            #Conduct one-hot encoding on our categorical features
            if one_hot_ecoding:
                X = feature_proc._one_hot_ecoding(X)
            #Select the most highlighly correlated features with the target variable
            if corr_features:
                correlation_threshold = 0.4
                X = feature_proc._extract_correlated_features(X,correlation_threshold,'SalePrice','Id')
            return X

        except Exception as e:
            logger.error("Error running the feature processing step")
            raise(e)

    def clean_dataset(self,df: object):
        """[Clean pandas dataframe of any NaN, Inf values]

        Args:
            df ([object]): [Input pandas dataframe]

        Returns:
            [df]: [Cleansed pandas dataframe]
        """
        try:
            df.dropna(inplace=True)
            indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
            return df[indices_to_keep].astype(np.float64)
        except Exception as e:
            raise(e)
        
    def run_model(self, X: object, Y: object):
        """[Call method to train the given model selected]
        """
        try:
            #Initialize dataframe to store the model accuracy
            columns = ['Model_Name','Model_Score','RMSE_Value']
            df = pd.DataFrame([],columns=columns)
            
            #Create the X and Y dataset by splitting SalePrice dependent variable into it's own dataframe
            Y = X.SalePrice
            X = X.drop('SalePrice',axis=1)

            #Iterate through the combinations of each model and num_features in the dataset to use
            for model in self.model_names:
                trained_model = TrainModel(X,Y)
                #Retrieve the model object and parameters for training
                model_config = trained_model._retrieve_parameter_grid(model)
                model_score, rmse = trained_model._train_model(model_config[model],model_config['model_params'],X,Y,5,5)
                data = {'Model_Name':model,'Model_Score':model_score,'RMSE_Value':rmse}
                df = df.append(data,ignore_index=True)
            #Store results to csv output
            df.to_csv('./src/data/model_output.csv',mode='w')

        except Exception as e:
            logger.error("Error running the controller method to fit/train models and retrieve predictions")
            raise(e)

    def plot_decision_boundary(self,X:object,Y:object,fit_model:object,model_name:str):
        """[Plot the decision boundary for the fitted model]

        Args:
            X (object): [X observations of the dataset]
            Y (object): [Y observations for the target variable]
            fit_model (object): [Fit/trained model to run a prediction on]
            model_name (str): [Model name to plot the decision boundary result]
        """
        try:
            #Step size in the grid mesh for the plot
            h = .02 

            # Create color maps
            cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
            cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            X = np.c_[X['X_0'],X['X_1']]
            Y = Y.values.ravel()
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = fit_model.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                        edgecolor='k', s=20)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("2-Class classification {0}".format(model_name))
            plt.savefig('./reports/figures/{0}_img.png'.format(model_name))

        except Exception as e:
            logger.error('Error plotting the decision boundary for the following model')
            raise(e)