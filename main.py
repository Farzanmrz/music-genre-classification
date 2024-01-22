# Run this file to use the different ML algorithms on the dataset.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score

from src.knn import KNN
from src.svm import SVM
from src.logreg import LogReg
from src.nn import NN
from src.cnn import CNN


def process_dataset(path, training_fraction):
    """
    Process the dataset by separating the labels from the features using pandas dataframes.
    Processed dataset is stored in data/ directory.
    path: path to the dataset (.csv)
    """
    print("[PROCESS] Processing dataset...\n")

    # Shuffle dataset after storing in dataframe
    dataset_df = pd.read_csv(path)
    shuffled_dataset_df = dataset_df.sample(frac=1)

    # Extract labels from dataset
    labels = shuffled_dataset_df['label'].tolist()

    # Convert labels to integers (0-9)
    labels = np.where(labels == 'blues', 0, labels)
    labels = np.where(labels == 'classical', 1, labels)
    labels = np.where(labels == 'country', 2, labels)
    labels = np.where(labels == 'disco', 3, labels)
    labels = np.where(labels == 'hiphop', 4, labels)
    labels = np.where(labels == 'jazz', 5, labels)
    labels = np.where(labels == 'metal', 6, labels)
    labels = np.where(labels == 'pop', 7, labels)
    labels = np.where(labels == 'reggae', 8, labels)
    labels = np.where(labels == 'rock', 9, labels)
    labels = np.where(labels == 'blues', 0, labels)

    # transpose to store in 1xn table
    labels_df = pd.DataFrame(labels).T

    # Remove 'label' column from dataset
    shuffled_dataset_df.drop(['filename', 'label'], axis=1, inplace=True)

    # Mean normalization of features. This increases KNN accuracy by almost 50%
    normalized_shuffled_dataset_df = (shuffled_dataset_df - shuffled_dataset_df.mean()) / shuffled_dataset_df.std()

    # Create directory for storing processed dataset
    if not os.path.isdir('processed_dataset/'):
        os.mkdir('processed_dataset/')

    # Find index where we will split the dataframe into training and testing dataframes
    number_of_examples = normalized_shuffled_dataset_df.shape[0]
    train_max_index = int(number_of_examples * training_fraction)

    # Split shuffled data set
    train_data_df = normalized_shuffled_dataset_df.iloc[:train_max_index, :]
    test_data_df = normalized_shuffled_dataset_df.iloc[train_max_index:, :]

    # Split labels
    train_labels_df = labels_df.iloc[:, :train_max_index]
    test_labels_df = labels_df.iloc[:, train_max_index:]

    # Convert the train and test dataframes to CSV
    train_data_df.to_csv(f'processed_dataset/train_data.csv', index=False, header=False)
    train_labels_df.to_csv(f'processed_dataset/train_labels.csv', index=False, header=False)
    test_data_df.to_csv(f'processed_dataset/test_data.csv', index=False, header=False)
    test_labels_df.to_csv(f'processed_dataset/test_labels.csv', index=False, header=False)


def plot_feature_importance():
    print("[PROCESS] Computing feature importance...\n")

    data = np.loadtxt(open('processed_dataset/train_data.csv'), delimiter=",")
    labels = np.loadtxt(open('processed_dataset/train_labels.csv'), delimiter=",")

    feature_names = [f"feature {i}" for i in range(data.shape[1])]
    forest = RandomForestClassifier()
    forest.fit(data, labels)

    importance = forest.feature_importances_

    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.xlabel("Feature")
    pyplot.ylabel("Mean decrease in impurity")
    pyplot.title("Mean decrease in impurity using Random Forest Classifier")
    pyplot.show()


def cross_validate(models):
    print("[PROCESS] Computing 10-fold cross-validation on each model...\n")

    # Load data and labels
    train_data = np.loadtxt(open('processed_dataset/train_data.csv'), delimiter=",")
    test_data = np.loadtxt(open('processed_dataset/test_data.csv'), delimiter=",")

    train_labels = np.loadtxt(open('processed_dataset/train_labels.csv'), delimiter=",")
    test_labels = np.loadtxt(open('processed_dataset/test_labels.csv'), delimiter=",")

    # Combine training and testing sets
    data = np.concatenate((train_data, test_data), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    scores = []
    mean_scores = []

    # Compute cross-validation scores for each model
    for model in models:
        cross_val_scores = cross_val_score(model, data, labels, cv=10)
        scores.append(cross_val_scores)
        mean_scores.append(cross_val_scores.mean())

    model_names = ["KNN", "LR SAG", "LR LIBLIN", "SVM OVO", "SVM OVR", "SVM LINEAR", "SVM RBF"]

    # Plot results
    pyplot.bar(model_names, mean_scores)
    pyplot.xlabel("Learning Model")
    pyplot.ylabel("Mean accuracy")
    pyplot.title("Mean accuracies from 10-fold cross-validation")
    pyplot.show()


def plot_svm_accuracies(accuracies):
    svm_names = ["ovo", "ovr", "linear", "rbf"]
    pyplot.bar(svm_names, accuracies)
    pyplot.xlabel("SVM Learning Model")
    pyplot.ylabel("Accuracy")
    pyplot.title("Accuracies of SVM learning models")
    pyplot.show()


def plot_lr_accuracies(accuracies):
    lr_names = ["SAG", "LIBLIN"]
    pyplot.bar(lr_names, accuracies)
    pyplot.xlabel("Logistic Regression Learning Model")
    pyplot.ylabel("Accuracy")
    pyplot.title("Accuracies of LR learning models")
    pyplot.show()


def plot_knn_accuracies():
    n = 1

    errors = []
    for i in range(12):
        knn = KNN(n)
        accuracy, error, f1 = knn.train_and_predict()

        errors.append(error)
        n += 2

    n_values = ["1", "3", "5", "7", "9", "11", "13", "15", "17", "19", "21", "23"]
    pyplot.bar(n_values, errors)
    pyplot.xlabel("KNN K value")
    pyplot.ylabel("Error Rate")
    pyplot.title("KNN K value vs Error Rate")
    pyplot.show()


def plot_f1_scores(scores):
    names = ["KNN", "LR", "SVM", "NN"]
    pyplot.bar(names, scores)
    pyplot.xlabel("Learning Algorithm")
    pyplot.ylabel("F1 score")
    pyplot.title("F1 score of each learning algorithm")
    pyplot.show()


if __name__ == '__main__':
    print("[PROCESS] Running Music Genre Classifiers Project...")

    # Process data set and split into training (70%) and testing (30%)
    process_dataset('features_30_sec.csv', 0.7)

    # Classify using KNN
    knn_classifier = KNN(3)
    knn_accuracy, error_rate, knn_f1 = knn_classifier.train_and_predict()

    # Classify using Logistic Regression
    logreg_classifiers = LogReg()
    logreg_accuracies, logreg_f1 = logreg_classifiers.train_and_predict()

    plot_lr_accuracies(logreg_accuracies)

    # Classify using SVM
    svm_classifiers = SVM()
    svm_accuracies, svm_f1 = svm_classifiers.train_and_predict()

    plot_svm_accuracies(svm_accuracies)

    # Classify using Neural Networks
    nn_classifiers = NN()
    nn_accuracy, nn_f1 = nn_classifiers.train_and_predict()

    # Classify using Convolutional Neural Networks
    cnn_classifiers = CNN()
    cnn_accuracy = cnn_classifiers.train_and_predict()

    plot_feature_importance()

    # Cross validate learning models
    models = [knn_classifier.neigh, logreg_classifiers.mulnom_sag, logreg_classifiers.liblin, svm_classifiers.ovo_svm,
              svm_classifiers.ovr_svm, svm_classifiers.linear_svm, svm_classifiers.rbf_svm]
    cross_validate(models)

    # plot for accuracy of each algorithm
    print('Plotting accuracy for Machine Learning Algorithms...\n')

    data = {'CNN': round(cnn_accuracy * 100, 2), 'SVM': round(svm_accuracies[0] * 100, 2),
            'LR': round(logreg_accuracies[0] * 100, 2),
            'KNN': round(knn_accuracy * 100, 2), 'NN': round(nn_accuracy * 100, 2)}
    courses = list(data.keys())
    values = list(data.values())

    figure = plt.figure(figsize=(8, 6))

    # creating the bar plot
    plt.bar(courses, values, color='skyblue', width=0.3)

    plt.xlabel("Machine Learning Algorithm")
    plt.ylabel("Accuracy")
    plt.title("MLA Accuracy Plot")
    plt.show()

    # Plot KNN error vs K value
    print('Plotting Error Rate vs K-value for Knn, to find the best value for k...\n')
    plot_knn_accuracies()

    print('Plotting F1 Scores for Machine Learning Algorithms...')
    plot_f1_scores([knn_f1, logreg_f1, svm_f1, nn_f1])
