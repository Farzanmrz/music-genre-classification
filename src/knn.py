from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class KNN:
    """
    Use KNN to train and predict labels on the processed dataset

    Attributes
    ----------
    n : int
        number of neighbors

    Methods
    _______
    __init__(n)
        Store the processed data and labels in numpy matrices

    __train_and_predict()
        Initialize sklearn.neighbors.KNeighborsClassifier, train using train data and train labels,
        and return accuracy on test data.

    """

    def __init__(self, n):
        self.n = n

        self.train_data = np.loadtxt(open('processed_dataset/train_data.csv'), delimiter=",")
        self.test_data = np.loadtxt(open('processed_dataset/test_data.csv'), delimiter=",")

        self.train_labels = np.loadtxt(open('processed_dataset/train_labels.csv'), delimiter=",")
        self.test_labels = np.loadtxt(open('processed_dataset/test_labels.csv'), delimiter=",")

        self.neigh = neigh = KNeighborsClassifier(n_neighbors=self.n)

    def train_and_predict(self):
        print("[PROCESS] Training KNN classifier and gathering prediction accuracy...")

        # Initialize and train KNN classifier

        self.neigh.fit(self.train_data, self.train_labels)

        # Predict labels of test data
        predicted_labels = self.neigh.predict(self.test_data)

        # Compute accuracy of KNN classifier
        accuracy = accuracy_score(self.test_labels, predicted_labels)

        error_rate = np.mean(predicted_labels != self.test_labels)

        f1 = f1_score(self.test_labels, predicted_labels, average='macro')

        print(f"[SCORE] KNN classifier correctly classified {round(accuracy * 100, 2)}% of testing instances\n")
        return accuracy, error_rate, f1

