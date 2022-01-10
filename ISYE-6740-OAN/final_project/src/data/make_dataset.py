# -*- coding: utf-8 -*-

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def read_data(dataset_name:str):
    """[Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).]

    Args:
        dataset_name (str): [Dataset name for the homework question which is either divorce or digit classification]
    Returns:
        [X,Y]: [X data containing our training data, Y containing the target variable]
    """
    try:
        logger.info('Reading in raw dataset: {0} for final project'.format(dataset_name))
        #Load dataframe and header file containing column descriptions in the header file
        X = pd.read_csv('./data/external/train.csv')
        Y = pd.read_csv('./data/external/test.csv')
        return X,Y

    except Exception as e:
        logger.error("Error instantiating dataset for: {0}".format(dataset_name))
        raise(e)