# MusicGenreClassifiers

_This repository contains the software developed by Mayank Hirani, Farzan Mirza, and Enzo Saba for their Machine Learning Final Project._

This project uses different discriminative machine learning classifers to classify the genre of 30-second audio tracks. <br />

## Dataset
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) on Kaggle. <br />
It contains 100 tracks for the following 10 music genres: _blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock._

## How to run and use this project
This project used anaconda as a package manager. To run the project, you will first need to install the libraries listed <br />
at the bottom of this page. <br />

Run ```main.py``` and it will train/test every model, as well as generate figures for the performance of each model, <br />
the impurity of the features in the dataset, 10-fold cross-validation, and the F1 score of each model.

Note: The dataset is already contained in the repository/zip file and is called ```features_30_sec.csv``` 

## Classifiers used
The following classifiers were implemented to classify our dataset: <br />
1. KNN
2. SVM
3. NN
4. LR
5. CNN

## Results
1. KNN: ~67% accuracy
2. SVM: 71% accurary
3. NN: 74% accuracy
4. LR: 71% accuracy
5. CNN: 75% accuracy

### List of libraries
1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. tensorflow
6. keras
