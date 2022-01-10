import logging.config
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

class FeatureEngineering: 

    def __init__(self, X: object):
        """[Constructor for the TrainModel class for various models to fit]

        Args:
            X (object): [X observations of the dataset containing our training data]
        """
        self.X = X
    
    def _extract_time_features(self, old_col_names: list, unit: str):
        """[Method to work with time related data to extract time difference
        calculations]

        Args:
            old_col_name (list): [List of old column names to apply the transformation]
            unit (str): [Unit of time for column (i.e. Years, Months, Weeks, Etc.)]
        """
        try:
            logger.info("Calculating new features related to the time difference between current day and each time feature")
            #Create temporary column for the current year
            self.X['current_year'] = datetime.now().year
            
            for old_name in old_col_names:
                #If the unit is year, then calculate the difference from the current year to the passed year
                if unit in 'Year':
                    self.X[old_name+'_Diff'] = self.X['current_year'] - self.X[old_name]
                else:
                    raise NotImplementedError('This metric has not been implemented as of yet')
            
                self.X = self.X.drop(columns=[old_name])
            
            #Drop the temporary column and the old column as the calculation is complete
            self.X = self.X.drop(columns=['current_year'])
            return self.X
        except Exception as e:
            logger.error("Error when calculation the time difference for the following column : {0}".format(old_name))
            raise(e)

    def _extract_correlated_features(self, X: object, threshold: float, dependent_column:str, exclusion_cols:str):
        """[summary]

        Args:
            X (object): [X ]
            threshold (float): [Threshold of the absolute value of the correlation score we're looking for]
            dependent_column (str): [Target variable we're assessing the correlation score against all other columns]
            exclusion_cols (str): [Exclusion columns to be be added back in once filtering occurs]
        """
        """[This method select the features that have a correlation score higher than the provided threshold]

        Args:
            X (object): [X observations of the dataset containing our training data]
        """        
        try:
            #Dependent columns for the correlation calculation, adding Id as this is the joining factor and needed for kaggle submission
            #Correlations object
            correlations = X.corr()
            #Create correlations dataframe containing the most correlated column
            corr_df = pd.DataFrame(correlations['SalePrice']).reset_index().rename(columns={'index':'Correlated_Column'})
            corr_df = corr_df.loc[corr_df['SalePrice'].abs()>threshold]
            corr_cols = [col for col in corr_df['Correlated_Column'].values.tolist() if col not in dependent_column]
            corr_cols = corr_cols.append(exclusion_cols)
            return X[[corr_cols]]

        except Exception as e:
            logger.error("Error running the process to determine the most correlated features with the target variable")
            raise(e)

    def _replace_nan_values(self, X: object):
        """[This method iterates through our Nan values and either replace or imputes the missing value]

        Args:
            X (object): [Training observations to apply the transformation]
        """      
        try:
            logger.info("Replacing NaN values across the numerical and categorical features")
            # Columns containing categorical features where the Nan values just means that it's missing
            cat_cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
                        'GarageQual','GarageCond','GarageFinish','GarageType',
                        'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']

            # replace 'NaN' with 'None' in these columns
            for col in cat_cols_fillna:
                X[col].fillna('None',inplace=True)

            #Numerical columns that will fill with a value of 0
            numeric_cols_fillna = ['MasVnrArea','BsmtFullBath','BsmtHalfBath','BsmtFinSF1',
            'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageArea','GarageCars']

            # replace 'NaN' with 0 in these columns
            for col in numeric_cols_fillna:
                X[col].fillna(0,inplace=True)
            
            #Impute the GarageYrBlt variable with the YearBuilt value
            X.loc[X.GarageYrBlt.isnull(),'GarageYrBlt'] = X.loc[X.GarageYrBlt.isnull(),'YearBuilt']
            #Impute the missing values for the remaining variables LotFrontage, Electrical
            X = X.fillna(X.mean())
            X = X.drop(columns=['Electrical'])
            return X

        except Exception as e:
            raise(e)


    def _ordinal_ecoding(self, X: object):
        """[This method iterates through our ordinal columns and provides them with a numerical value based on the level]

        Args:
            X (object): [Training observations to apply the transformation]
        """        
        try:
            logger.info("Calculating ordinal columns with Ordinal Encoding")
            #Provide the grouping for the ordinal data contained in the dataframe
            lot_ord_rank = {'Reg': 4 , 'IR1': 3, 'IR2': 2, 'IR3': 1}
            util_ord_rank = {'AllPub': 4 , 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1}
            slope_ord_rank = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
            expo_ord_rank = {'Ex': 5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
            bsmt_exposure_ord_rank = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
            bsmt_fin_ord_rank = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
            electrical_ord_rank = {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1}
            functional_ord_rank = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
            finish_ord_rank = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
            paved_ord_rank = {'Y': 2, 'P': 1, 'N': 0}
            fence_ord_rank = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}

            X_ord = X.replace({'LotShape': lot_ord_rank,
                         'Utilities': util_ord_rank,
                         'LandSlope': slope_ord_rank,
                         'ExterQual': expo_ord_rank,
                         'ExterCond': expo_ord_rank,
                         'BsmtQual': expo_ord_rank,
                         'BsmtCond': expo_ord_rank,
                         'BsmtExposure': bsmt_exposure_ord_rank,
                         'BsmtFinType1': bsmt_fin_ord_rank,
                         'BsmtFinType2': bsmt_fin_ord_rank,
                         'HeatingQC': expo_ord_rank,
                         'Electrical': electrical_ord_rank,
                         'KitchenQual': expo_ord_rank,
                         'Functional': functional_ord_rank,
                         'FireplaceQu': expo_ord_rank,
                         'GarageFinish': finish_ord_rank,
                         'GarageQual': expo_ord_rank,
                         'GarageCond': expo_ord_rank,
                         'PavedDrive': paved_ord_rank,
                         'PoolQC': expo_ord_rank,
                         'Fence': fence_ord_rank})
            return X_ord

        except Exception as e:
            logger.error("Error with ordinal encoding process")
            raise(e)

    def _one_hot_ecoding(self, X: object):
        """[This method creates dummy variables for our categorical features that require encoding]

        Args:
            X (object): [Training observations to apply the transformation]
        """        
        logger.info("Conducting one-hot encoding on categorical features")
        try:
            #Use the get_dummies libray to retrieve dummy variables for the categorical data
            return pd.get_dummies(X)

        except Exception as e:
            raise(e)