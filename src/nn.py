import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


class NN:
    def __init__(self):
        self.train_data = np.loadtxt(open('processed_dataset/train_data.csv'), delimiter=",")
        self.test_data = np.loadtxt(open('processed_dataset/test_data.csv'), delimiter=",")

        self.train_labels = np.loadtxt(open('processed_dataset/train_labels.csv'), delimiter=",")
        self.test_labels = np.loadtxt(open('processed_dataset/test_labels.csv'), delimiter=",")

        self.nn = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', max_iter=200)

    def train_and_predict(self):

        print("[PROCESS] Training NN classifiers and gathering prediction accuracies...")

        self.nn.fit(self.train_data, self.train_labels)

        predict_test = self.nn.predict(self.test_data)

        accuracy = accuracy_score(self.test_labels, predict_test)

        f1 = f1_score(self.test_labels, predict_test, average='macro')

        print(f"[SCORE] NN classifier correctly classified {round(accuracy * 100, 2)}% of testing instances\n")
        return accuracy, f1


