import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(data_file, cols):
    return pd.read_csv(data_file, encoding="latin_1", sep=",", names=cols, engine='python', memory_map=True)


def create_classifier(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = svm.SVC(kernel='poly') # kernel types ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)
    return clf


def main():
    data_file = '../Dados/sample-feature.csv'

    #features = ['len', 'n_inversions', 'repeats', 'mean', 'median']
    #cols = ['id', 'len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv', 'stdev', 'pstdev', 'median', 'mean',
    #                'pvariance', 'target']
    cols = ['id', 'array_id', 'len', 'heap_time', 'merge_time', 'target']

    dataSet = load_dataset(data_file, cols)
    dataSet = dataSet.drop('id', 1)
    dataSet = dataSet.drop('array_id', 1)
    features = ['heap_time', 'merge_time']
    dt = create_classifier(dataSet, features)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")