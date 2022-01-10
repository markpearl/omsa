import logging.config
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TrainModel: 

    def __init__(self, X: object, Y: object):
        """[Constructor for the TrainModel class for various models to fit]

        Args:
            X (object): [X observations of the dataset]
            Y (object): [Y observations for the target variable]
        """
        self.X_train, self.Y_train = X, Y

    def _retrieve_parameter_grid(self, model_name:str):
        """[Retrieves parameters to use for the given model]

        Args:
            model_name (str): [Name of the model]
        """
        try:
            logger.info("Retrieving model config parameters for the: {0} model".format(model_name))
            model_dict = {}
            #Model params for RandomForest
            if model_name in 'RandomForest':
                param_grid = {'n_estimators':[100,150,200],
                'max_features':[25,50,75],
                'min_samples_split':[2,4,6]}
                model = RandomForestRegressor()

            #Model params for ElasticNet
            elif model_name in 'ElasticNet':
                param_grid = {'alpha': np.arange(1e-4,1e-3,1e-4),
                'l1_ratio': np.arange(0.1,1.0,0.1),
                'max_iter':[100000],
                'tol':[10]}
                model = ElasticNet()
            
            #Model params for Ridge Regression
            elif model_name in 'Ridge':
                param_grid = {'alpha': np.arange(0.25,6,0.25),'tol':[10]}
                model = Ridge()
            #Model params for Lasso Regression
            elif model_name in 'Lasso':
                param_grid =  {'alpha': np.arange(1e-4,1e-3,4e-5),'tol':[10]}
                model = Lasso()
            else:
                raise NotImplementedError('The model provided doesn"t have a parameter grid')
            model_dict[model_name] = model
            model_dict['model_params'] = param_grid
            return model_dict

        except Exception as e:
            logger.error("Error retrieving grid parameters")
            raise(e)

    def rmse(self,y_true, y_pred):
        """[Calculate rmse metric]

        Args:
            y_true ([type]): [Actual values of the y variable]
            y_pred ([type]): [Predicted values of the y variable]

        Returns:
            [rmse]: [Return the rmse calculation]
        """
        diff = y_pred - y_true
        sum_sq = sum(diff**2)    
        n = len(y_pred)   
        rmse = np.sqrt(sum_sq/n)
        return rmse

    def _train_model(self,model:object, param_grid: dict, X:object, y:object, splits=5, repeats=5):
        """[This is the training method used for the model we're working with]

        Args:
            model (str): [Model object]
            param_grid (list): [Parameter grid containing hyperparameter values]
            X (object): [Traning data observations]
            y (object): [Test data observations]
            splits (int, optional): [description]. Defaults to 5.
            repeats (int, optional): [description]. Defaults to 5.
        """
        try:
             #Create scoring metric to validate the model for rmse
            rmse_score = make_scorer(self.rmse, greater_is_better=False)
            # Cross validation class variable
            rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
            
            # Check if grid search parameters were passed
            if len(param_grid)>0:
                logger.info("Running GridSearch cross-validation to find optimal hyperparameter values")
                # setup grid search parameters
                gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                                    scoring=rmse_score,
                                    verbose=1, return_train_score=True)

                #Fit the model and search grid for optimal value
                gsearch.fit(X,y)
                # extract best model from the grid
                model = gsearch.best_estimator_        
                best_idx = gsearch.best_index_

                # get cv-scores for best model
                grid_results = pd.DataFrame(gsearch.cv_results_)       
                cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
                cv_std = grid_results.loc[best_idx,'std_test_score']

            else:
                grid_results = []
                cv_results = cross_val_score(model, X, y, scoring=rmse_score, cv=rkfold)
                cv_mean = abs(np.mean(cv_results))
                cv_std = np.std(cv_results)
            
            # Combine the mean and standard deviation score to calculate the cv score
            cross_validation_score = pd.Series({'mean':cv_mean,'std':cv_std})

            logger.info("Running prediction against target variable")
            # predict y using the fitted model
            y_pred = model.predict(X)
            
            logger.info("Retrieving Model Score and RMSE values")
            # print stats on model performance   
            model_score = model.score(X,y) 
            rmse = self.rmse(y, y_pred)
            return model_score, rmse

        except Exception as e:
            logger.error("Error running the training process on the model")
            raise(e)