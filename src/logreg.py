#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# In[ ]:


class LogReg:
    def __init__(self):
        self.train_data = np.loadtxt(open('processed_dataset/train_data.csv'), delimiter=",")
        self.test_data = np.loadtxt(open('processed_dataset/test_data.csv'), delimiter=",")

        self.train_labels = np.loadtxt(open('processed_dataset/train_labels.csv'), delimiter=",")
        self.test_labels = np.loadtxt(open('processed_dataset/test_labels.csv'), delimiter=",")

        self.mulnom_sag = LogisticRegression(solver='sag',max_iter=1000,multi_class='multinomial')
        self.liblin = LogisticRegression(solver='liblinear',max_iter=1000)
     
    def train_and_predict(self):
        print("[PROCESS] Training Logistic Regression classifiers and gathering prediction accuracies...")

        # Train the multinomial model, ovr approach. sag solver, L2 regularization for non-sparse matrix
        self.mulnom_sag.fit(self.train_data,self.train_labels)
        
        # Train the liblinear model, ovr approach. Coordinate Descent for large linear classification

        self.liblin.fit(self.train_data,self.train_labels)

        # Do the predictions with test labels
        mulnom_pred = self.mulnom_sag.predict(self.test_data)
        liblin_pred = self.liblin.predict(self.test_data)
        
        # Get accuracies of all models
        mulnom_accuracy = accuracy_score(mulnom_pred, self.test_labels)
        liblin_accuracy = accuracy_score(liblin_pred, self.test_labels)

        f1_mulnom = f1_score(self.test_labels, mulnom_pred, average='macro')
        f1_liblin = f1_score(self.test_labels, liblin_pred, average='macro')

        f1 = np.average([f1_mulnom, f1_liblin])

        print(
            f"[SCORE] Multinomial Logistic Regression correctly classified {round(mulnom_accuracy * 100, 2)}% of testing instances")
        print(
            f"[SCORE] Linear Library Logistic Regression correctly classified {round(liblin_accuracy * 100, 2)}% of testing instances\n")
        return [mulnom_accuracy, liblin_accuracy], f1