import logging.config
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class TrainModel: 

    def __init__(self, X: object, Y: object):
        """[Constructor for the TrainModel class for various models to fit]

        Args:
            X (object): [X observations of the dataset]
            Y (object): [Y observations for the target variable]
        """
        self.X_train, self.Y_train = X, Y
    
    def _train_gnb(self):
        """[Model to train the gaussian naive bayes algorithm]
        """
        try:
            logger.info("Training the Gaussian Naive Bayes algorithm")
            gnb = GaussianNB()
            gnb.fit(self.X_train, self.Y_train)
            return gnb 
            
        except Exception as e:
            logger.error("Error training the gaussian naive classifier")
            raise(e)

    def _train_knn(self):
        """[Model to train the k-nearest neighbors algorithm]
        """
        try:
            logger.info("Training the KNN algorithm")
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(self.X_train, self.Y_train)
            return knn

        except Exception as e:
            logger.error("Error training the gaussian naive classifier")
            raise(e)

    def _train_logistic(self):
        """[Model to train the logistic regression algorithm]
        """
        try:
            logger.info("Training the Logistic Regression algorithm")
            clf = LogisticRegression(random_state=0,max_iter=300)
            clf.fit(self.X_train, self.Y_train)
            return clf

        except Exception as e:
            logger.error("Error training the gaussian naive classifier")
            raise(e)