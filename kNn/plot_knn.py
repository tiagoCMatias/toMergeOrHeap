
import sys
sys.path.append('../')

from geral.helpFunctions import load_dataset, generatePredictionFile, first_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

  
def knnClassifer(trainDataSet, features, targetDataSet):
    X = trainDataSet[features]
    y = trainDataSet['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    neigh = KNeighborsClassifier(n_neighbors=4, weights='distance', algorithm='auto')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    pred = (accuracy_score(y_test, y_pred) * 100)
    print("Prediction: ", pred)
    neigh.fit(trainDataSet[features], trainDataSet['target'])
    prediction = neigh.predict(targetDataSet)
    generatePredictionFile(targetDataSet, prediction, file_name="knnPrediction.csv")


def main():
    
    targetFile = "../Dados/main-features2.csv"
    trainFile = '../boa.csv'

    trainFeatures = first_dataset(trainFile)

    trainDataSet = load_dataset(trainFile, trainFeatures).drop('id', 1)

    trainFeatures.pop(0)
    trainFeatures.pop()

    targetFeatures = first_dataset(targetFile)
    targetDataSet = load_dataset(targetFile, targetFeatures)
    targetFeatures.pop(0)

    print(trainFeatures)
    print(targetFeatures)

    knnClassifer(trainDataSet, trainFeatures, targetDataSet)

if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")
