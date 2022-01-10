# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import csv 

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
        if 'spam' in dataset_name:
            data = []
            with open('./src/data/spambase.data') as cf:
                readcsv = csv.reader(cf, delimiter=',')
                for row in readcsv:
                    data.append(row)       
            data = np.array(data).astype(np.float)

        X = data[:, :-1]
        Y = data[:, -1]
        return X,Y

    except Exception as e:
        logger.error("Error instantiating dataset for: {_")
        raise(e)