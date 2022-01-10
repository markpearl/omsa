import logging.config
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

class PredictModel: 

    def __init__(self, X: object):
        """[Constructor for the Predict class for various models to run prediction against test dataset]

        Args:
            X (object): [X test observations of the dataset]
        """
        self.X_test = X
    
    def _predict_model(self, fit_model:object):
        """[Model to train the gaussian naive bayes algorithm]

        Args:
            fit_model (object): [Fit/trained model to run a prediction on]

        Returns:
            [y_pred]: [Returns prediction results for the model on the y-test data]
        """
        try:
            logger.info("Now running prediction for the provided fitted model")
            y_pred = fit_model.predict(self.X_test)
            return y_pred

        except Exception as e:
            logger.error("Error running prediction for the provided trained/fitted model")
            raise(e)