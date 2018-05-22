import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.model_selection import validation_curve


def load_dataset(data_file, cols):
    return pd.read_csv(data_file, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python', memory_map=True)

def first_dataset(data_file):
    firstline = pd.read_csv(data_file)
    dataset = list(firstline)
    return dataset

def svcClassifer(dataset, features):

    X = dataset[features]
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = svm.SVC()
    parameters = {
        'coef0': [0.0],
        'degree': [1, 2, 3, 4, 6, 8, 10],
        #"gamma": ['rbf','auto','poly','sigmoid'],
        'kernel': ['linear','poly','rbf','sigmoid','precomputed','callable'],
        'C': [1.0],
    }
    rsearch = model_selection.GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1,cv=10)
    rsearch.fit(dataset[features], dataset['target'])

    #summarize the results of the random parameter search
    #print(rsearch)
    print(rsearch.best_score_)
    print(rsearch.best_params_)

    rsearch.fit(X_train, y_train)
    y_pred = rsearch.predict(X_test)
    pred = accuracy_score(y_test, y_pred) * 100

    print("Best Prediction: ", pred)

def create_target(dataSet, features, targetDataSet, target_features):
    y = dataSet['target']
    X = dataSet[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    dt = svm.SVC(C=1.0, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    dt.fit(dataSet[features], dataSet['target'])
    y_pred = dt.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)
    prediction = dt.predict(targetDataSet[target_features])

    cols = ['Predicted']
    cenas = targetDataSet['id_target']
    features = pd.DataFrame(prediction, columns=cols)
    file_name = "output2.csv"
    features.set_index(cenas, inplace = True)
    features.to_csv(file_name, sep=',', encoding='utf-8')


def create_classifier(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = svm.SVC(C=1.0, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False) # kernel types ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’


    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)
    return clf



def main():
    target_file = 'main-features2.csv'
    data_file = 'boa.csv'
    target_features = first_dataset(target_file)
    name_of_features = first_dataset(data_file)

    dataSet = load_dataset(data_file, name_of_features)
    dataSet = dataSet.drop('id', 1)

    targetDataSet = load_dataset(target_file, target_features)


    name_of_features.pop(0)
    name_of_features.pop(0)

    name_of_features.pop()


    target_features.pop(0)

    svcClassifer(dataSet,name_of_features)

    #dt = create_classifier(dataSet, name_of_features)
    #create_target(dataSet, name_of_features, targetDataSet, target_features)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")