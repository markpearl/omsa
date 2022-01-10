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

class HW3Controller:
    
    def __init__(self, datasets: str,num_features:str,model_names:str):
        """[Constructor class for the controller which will either train or predict a given model 
        based on the dataset name retrieved from the run script.]

        Args:
            model_names (str): [Name of models/classifiers to be used]
            datasets (str): [This dataset string determine which dataset to import for the question]
            num_features (str): [Number of features to use when preparing the X observations / features vectors]
        """        
        self.datasets = datasets
        self.num_features = num_features
        self.model_names = model_names
        
    def run_model(self):
        """[Call method to train the given model selected]
        """
        try:
            #Initialize dataframe to store the model accuracy
            columns = ['Model_Name','Model_Accuracy','Num_Features','Dataset']
            df = pd.DataFrame([],columns=columns)
            
            #Iterate through the combinations of each model and num_features in the dataset to use
            for dataset, num_features in list(itertools.product(self.datasets,self.num_features)):
                #Initialize X and Y data for the dataset
                X,Y = read_data(dataset,num_features)
                X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
                #Only run the model on all feature for the digits dataset
                if 'digits' in dataset and '2' in num_features:
                    #Store results to csv output
                    df.to_csv('./src/data/model_output.csv',mode='w')
                    break
                for model in self.model_names:
                    #Initialize train model class
                    logger.info('Now training the {0} model for the {1} dataset for: {2} number of features.'.format(model,dataset,num_features))
                    trained_model = TrainModel(X_train,Y_train.values.ravel())
                    #Train the model name and receive the fitted model object
                    fit_model = getattr(trained_model,f'_train_{model}')()
                    #If 2 features included, plot and save the decision boundary result
                    if '2' in num_features:
                        self.plot_decision_boundary(X,Y,fit_model,model)                        
                    predict_model = PredictModel(X_test)
                    y_pred = predict_model._predict_model(fit_model)
                    #Record the model accuracy
                    accuracy = sum(y_pred.ravel() == (np.array(Y_test).ravel()))/Y_test.shape[0]
                    data = {'Model_Name':model,'Model_Accuracy':accuracy,'Num_Features':num_features,'Dataset':dataset}
                    df = df.append(data,ignore_index=True)

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