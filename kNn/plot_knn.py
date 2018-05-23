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
from sklearn.pipeline import make_pipeline
import joblib

from sklearn import model_selection

def loadDataSet(file_path, cols):
    return pd.read_csv(file_path, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python', memory_map=True)

def first_dataset(data_file):
    firstline = pd.read_csv(data_file)
    dataset = list(firstline)  
    return dataset

def inRange(begin, final):
    result = []    
    for x in range (begin, final):
        result.append(x)
    return result
  
def knnClassifer(dataset, features):

    X = dataset[features]
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier()
    parameters = {
        'n_neighbors': inRange(1,50),
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'weights': ['uniform','distance'],
    }
    rsearch = model_selection.GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1,cv=10)
    rsearch.fit(dataset[features], dataset['target'])

    print('The best parameters are : ')
    print(rsearch.best_params_)

def main():
    
    data_file = 'boa.csv'

    name_of_features = first_dataset(data_file)

    dataset = loadDataSet(data_file, name_of_features)
    dataset = dataset.drop('id', 1)

    name_of_features.pop(0)
    name_of_features.pop()
    
    knnClassifer(dataset, name_of_features)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")
