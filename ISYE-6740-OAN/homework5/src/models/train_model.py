import logging.config
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

class TrainModel: 

    def __init__(self, X_train: object, Y_train: object,X_test: object, Y_test: object):
        """[Constructor for the TrainModel class for various models to fit]

        Args:
            X_train (object): [X observations of the train dataset]
            Y_train (object): [Y observations for the target variable/ train]
            X_test (object): [X observations of the test dataset]
            Y_test (object): [Y observations for the target variable/ test]
        """
        self.X_train, self.Y_train,self.X_test,self.Y_test = X_train, Y_train, X_test, Y_test
    
    def _train_cart(self,depth:int):
        """[Model to train the gaussian naive bayes algorithm]
        """
        try:
            logger.info("Now training the cart model")
            # plot the tree##
            ctree = tree.DecisionTreeClassifier().fit(self.X_train, self.Y_train)
            plt.figure() 
            tree.plot_tree(ctree, max_depth=3, filled=True)
            model_results = []
            for depth in range(1,depth): 
                ctree = tree.DecisionTreeClassifier(max_depth = depth).fit(self.X_train, self.Y_train)
                prediction = ctree.predict(self.X_test)
                model_results.append(roc_auc_score(self.Y_test, prediction))  
            return model_results
                
            
        except Exception as e:
            logger.error("Error training the cart algorithm")
            raise(e)

    def _train_forest(self, depth:int):
        """[Model to train the k-nearest neighbors algorithm]
        """
        try:
            logger.info("Now training the random forest regressor model")
            model_results = []
            for depth in range(1,depth):
                cforest = RandomForestClassifier(max_depth = depth).fit(self.X_train, self.Y_train)
                prediction = cforest.predict(self.X_test)
                model_results.append(roc_auc_score(self.Y_test, prediction))  
            return model_results

        except Exception as e:
            logger.error("Error training the random forest")
            raise(e)