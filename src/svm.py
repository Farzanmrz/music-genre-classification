from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class SVM:
    def __init__(self):
        self.train_data = np.loadtxt(open('processed_dataset/train_data.csv'), delimiter=",")
        self.test_data = np.loadtxt(open('processed_dataset/test_data.csv'), delimiter=",")

        self.train_labels = np.loadtxt(open('processed_dataset/train_labels.csv'), delimiter=",")
        self.test_labels = np.loadtxt(open('processed_dataset/test_labels.csv'), delimiter=",")

        self.ovo_svm = svm.SVC(decision_function_shape='ovo')
        self.ovr_svm = svm.SVC(decision_function_shape='ovr')
        self.linear_svm = svm.SVC(kernel='linear')
        self.rbf_svm = svm.SVC(kernel='rbf')

    def train_and_predict(self):
        print("[PROCESS] Training SVM classifiers and gathering prediction accuracies...")

        # Train one-versus-one SVM
        self.ovo_svm.fit(self.train_data, self.train_labels)

        # Train one-versus-rest SVM
        self.ovr_svm.fit(self.train_data, self.train_labels)

        # Train linear kernel SVM
        self.linear_svm.fit(self.train_data, self.train_labels)

        # Train rbf kernel SVM
        self.rbf_svm.fit(self.train_data, self.train_labels)

        # Predict test labels using all classifiers
        ovo_predictions = self.ovo_svm.predict(self.test_data)
        ovr_predictions = self.ovr_svm.predict(self.test_data)
        linear_predictions = self.linear_svm.predict(self.test_data)
        rbf_predictions = self.rbf_svm.predict(self.test_data)

        # Get accuracy of each classifier
        ovo_accuracy = accuracy_score(ovo_predictions, self.test_labels)
        ovr_accuracy = accuracy_score(ovr_predictions, self.test_labels)
        linear_accuracy = accuracy_score(linear_predictions, self.test_labels)
        rbf_predictions = accuracy_score(rbf_predictions, self.test_labels)

        # Return accuracies
        svm_accuracies = [ovo_accuracy, ovr_accuracy, linear_accuracy, rbf_predictions]

        f1_ovo = f1_score(self.test_labels, ovo_predictions, average='macro')
        f1_ovr = f1_score(self.test_labels, ovr_predictions, average='macro')
        f1_linear = f1_score(self.test_labels, linear_predictions, average='macro')

        f1 = np.average([f1_ovo, f1_ovr, f1_linear])

        print(f"[SCORE] SVM one-v-one correctly classified {round(svm_accuracies[0] * 100, 2)}% of testing instances")
        print(f"[SCORE] SVM one-v-rest correctly classified {round(svm_accuracies[1] * 100, 2)}% of testing instances")
        print(
            f"[SCORE] Linear Kernel SVM correctly classified {round(svm_accuracies[2] * 100, 2)}% of testing instances")
        print(f"[SCORE] RBF Kernel SVM correctly classified {round(svm_accuracies[3] * 100, 2)}% of testing instances")
        print(
            f"[SCORE] Mean accuracy of SVM classifiers is {round(sum(svm_accuracies) * 100 / len(svm_accuracies), 2)}%\n")
        return svm_accuracies, f1