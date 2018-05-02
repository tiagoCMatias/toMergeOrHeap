import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use("ggplot")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import sys

def load_dataset(data_file, cols):
    return pd.read_csv(data_file, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python', memory_map=True)


def create_classifier(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_best_pred = 0
    place = 0
    max_iter = 1000                 #activation = ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    clf = MLPClassifier(hidden_layer_sizes=(len(features)+1,), learning_rate='constant', learning_rate_init=0.05 , activation='identity' , verbose=True, random_state=1, max_iter=max_iter, warm_start=True)
    for i in range(max_iter):
        clf.fit(X, y)
        y_pred = clf.predict(X_test)
        pred = accuracy_score(y_test, y_pred) * 100
        if pred > y_best_pred:
            y_best_pred = pred
            place = i
        print("Accuracy is ", pred)
        sys.stdout.flush()
    print("Best Prediction: ", format(y_best_pred, '.2f'), place)
    #clf = MLPClassifier(hidden_layer_sizes=(len(features),len(features),len(features)))
    #clf.fit(X_train, y_train)

    return clf


def main():
    data_file = '../Dados/sample-feature.csv'
    file_path = "../Dados/train-feature5.csv"

    cols2 = ['id','len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                    'stdev', 'pstdev', 'median', 'mean', 'pvariance',
                    'max_mean', 'min_mean',
                    'max_length', 'min_length',
                    'target']
    features = ['len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                'stdev', 'pstdev', 'median', 'mean', 'pvariance',
                'max_mean', 'min_mean',
                'max_length', 'min_length']

    #cols = ['id', 'array_id', 'len', 'heap_time', 'merge_time', 'target']

    dataSet = load_dataset(file_path, cols2)
    dataSet = dataSet.drop('id', 1)

    #dataSet = dataSet.drop('array_id', 1)
    #features = ['heap_time', 'merge_time']
    dt = create_classifier(dataSet, features)
    #test(dataSet, features )

if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")