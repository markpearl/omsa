import logging.config
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)

class TrainModel: 

    def __init__(self, X: object, Y: object):
        """[Constructor for the TrainModel class for various models to fit]

        Args:
            X (object): [X observations of the dataset]
            Y (object): [Y observations for the target variable]
        """
        self.X_train, self.Y_train = X, Y
    
    def _train_gnb(self,params:object):
        """[Model to train the gaussian naive bayes algorithm]

        Args:
            params (object): [Parameters to use for the corresponding trained model]

        Returns:
            [clf]: [Trained model sklearn object]
        """
        try:
            logger.info("Training the Gaussian Naive Bayes algorithm")
            gnb = GaussianNB()
            gnb.fit(self.X_train, self.Y_train)
            return gnb 
            
        except Exception as e:
            logger.error("Error training the gaussian naive classifier")
            raise(e)

    def _train_knn(self,params:object):
        """[Model to train the k nearest neighbors algorithm]

        Args:
            params (object): [Parameters to use for the corresponding trained model]

        Returns:
            [clf]: [Trained model sklearn object]
        """
        try:
            logger.info("Training the KNN algorithm")
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(self.X_train, self.Y_train)
            return knn

        except Exception as e:
            logger.error("Error training the gaussian naive classifier")
            raise(e)

    def _train_logistic(self,params:object):
        """[Model to train the logistic regression algorithm]

        Args:
            params (object): [Parameters to use for the corresponding trained model]

        Returns:
            [clf]: [Trained model sklearn object]
        """
        try:
            logger.info("Training the Logistic Regression algorithm")
            clf = LogisticRegression(random_state=0,max_iter=300)
            clf.fit(self.X_train, self.Y_train)
            return clf

        except Exception as e:
            logger.error("Error training the gaussian naive classifier")
            raise(e)

    def _train_neural_net(self,params:object):
        """[Model to train the artificial neural network algorithm]
        
        Args:
            params (object): [Parameters to use for the corresponding trained model]

        Returns:
            [clf]: [Trained model sklearn object]
        """
        try:
            logger.info("Training the Artificial Neural Net algorithm")
            clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'],random_state=1, max_iter=300).fit(self.X_train, self.Y_train)
            return clf

        except Exception as e:
            logger.error("Error training the artificial neural network")
            raise(e)

    def _train_svm(self,params:object):
        """[Model to train support vector machine algorithm]

        Args:
            params (object): [Parameters to use for the corresponding trained model]

        Returns:
            [clf]: [Trained model sklearn object]
        """
        try:
            logger.info("Training the Support Vector Machine Algorithm algorithm")

            clf = make_pipeline(StandardScaler(), SVC(C=params['C'], gamma='auto'))
            clf.fit(self.X_train, self.Y_train)
            return clf

        except Exception as e:
            logger.error("Error training the gaussian naive classifier")
            raise(e)