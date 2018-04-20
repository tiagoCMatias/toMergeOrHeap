import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

def loadDataSet(file_path, cols):
    return pd.read_csv(file_path, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python', memory_map=True)


def knnClassifer(dataset, features):
    feature_number = (features.__len__())-1
    X = dataset[features]
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_best_pred = 0
    k_best = 0
    for K in range(50):
        K_value = K + 1
        neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        pred = (accuracy_score(y_test, y_pred) * 100)
        if pred > y_best_pred:
            y_best_pred = pred
            cm = confusion_matrix(y_test, y_pred)
            k_best = K_value
        print ("Accuracy is ", pred, "% for K-Value :", K_value)
        sys.stdout.flush()
    print("Best Prediction: ", y_best_pred,"% for K-Value:", k_best)
    #plt.figure(figsize=(10, 7))
    #sn.heatmap(neigh, annot=True)
    #classifier = KNeighborsClassifier(n_neighbors=5)
    #classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    #cm = confusion_matrix(y_test, y_pred)



def main():

    cols2 = ['id','len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                    'stdev', 'pstdev', 'median', 'mean', 'pvariance',
                    'max_mean', 'min_mean',
                    'max_length', 'min_length',
                    'target']
    features = ['len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                'stdev', 'pstdev', 'median', 'mean', 'pvariance',
                'max_mean', 'min_mean',
                'max_length', 'min_length']

    cols = ['id', 'array_id', 'len', 'heap_time', 'merge_time', 'target']
    file_path = "../Dados/train-feature5.csv"
    dataset = loadDataSet(file_path, cols2)
    dataset = dataset.drop('id', 1)
    #dataset = dataset.drop('len', 1)
    #dataset = dataset.drop('inv1', 1)
    #dataset = dataset.drop('inv2', 1)
    #dataset = dataset.drop('inv3', 1)
    #dataset = dataset.drop('inv4', 1)
    #dataset = dataset.drop('inv5', 1)
    #dataset = dataset.drop('pstdev', 1)
    knnClassifer(dataset, features)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")
