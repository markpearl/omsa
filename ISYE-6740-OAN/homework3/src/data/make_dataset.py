# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def read_data(dataset_name:str,num_features:str):
    """[Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).]

    Args:
        dataset_name (str): [Dataset name for the homework question which is either divorce or digit classification]
        num_features (str): [Referencing the model run parameter for the number of features to use]
    Returns:
        [X,Y]: [X datacontaining the feature vectors, Y containing the target variable]
    """
    try:
        logger.info('Reading in raw dataset: {0} for the homework assignment'.format(dataset_name))
        #Load dataframe and header file containing column descriptions in the header file
        if 'divorce' in dataset_name and num_features in 'All':
            df = pd.read_csv('./src/data/marriage.csv',header=None,prefix='X_')
            df = df.rename(columns = {'X_54':'Y'})
            X = df[[col for col in df.columns if col not in 'Y']]        
            Y = df[[col for col in df.columns if col in 'Y']]  
        elif 'divorce' in dataset_name and num_features in '2':
            df = pd.read_csv('./src/data/marriage.csv',header=None,prefix='X_')
            df = df.rename(columns = {'X_54':'Y'})
            X = df.iloc[:,0:2]
            Y = df[[col for col in df.columns if col in 'Y']]  
        else:
            #Read the input image digits data
            X  = np.loadtxt("./src/data/data.dat")
            X = pd.DataFrame(X.reshape(X.shape[1],X.shape[0]))
            Y = np.loadtxt("./src/data/label.dat")
            Y = pd.DataFrame(Y).replace(2,0).replace(6,1)
        
        return X,Y

    except Exception as e:
        logger.error("Error instantiating dataset for: {_")
        raise(e)