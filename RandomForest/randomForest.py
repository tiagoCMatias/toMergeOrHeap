import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt

def load_dataset(data_file, cols):
    return pd.read_csv(data_file, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python', memory_map=True)

def first_dataset(data_file):
    firstline = pd.read_csv(data_file)
    dataset = list(firstline)
    return dataset


def create_classifier(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # RFC with fixed hyperparameters max_depth, max_features and min_samples_leaf
    rfc = RandomForestClassifier()

    # Use a grid over parameters of interest
    param_grid = {
        "n_estimators": [9, 18, 27, 36, 45, 54, 63],
        "max_depth": [1, 5, 10, 15, 20, 25, 30],
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
    }

    CV_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, cv=10)
    CV_rfc.fit(X_train, y_train)
    print (CV_rfc.best_params_)

    n_estim = CV_rfc.best_params_['n_estimators']
    max_depth = CV_rfc.best_params_['max_depth']
    min_samples_leaf = CV_rfc.best_params_['min_samples_leaf']
    max_features = CV_rfc.best_params_['max_features']

    print(n_estim, max_depth, min_samples_leaf)

    clf = RandomForestClassifier(n_estimators=n_estim, max_depth=max_depth, max_features=max_features, min_samples_leaf = min_samples_leaf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print( "Accurycy: ", accuracy_score(y_test, y_pred) * 100)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def forestClassifer(dataset, features):

    X = dataset[features]
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier()
    parameters = {
        'n_estimators': [9, 18, 27, 36, 45, 54, 63],
        'max_features': ["auto", "sqrt", "log2"],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
        'criterion': ['gini'],
        'max_depth': [1, 5, 10, 15, 20, 25, 30],
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

def gridSearch():

    data_file = 'boa.csv'

    name_of_features = first_dataset(data_file)

    dataSet = load_dataset(data_file, name_of_features)
    dataSet = dataSet.drop('id', 1)

    name_of_features.pop(0)
    name_of_features.pop()

    print(name_of_features)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100)
    # Fit the random search model
    rf_random.fit(dataSet[name_of_features], dataSet['target'])

    report(rf_random.cv_results_)

def main(): 
    data_file = 'boa.csv'

    features = first_dataset(data_file)

    dataSet = load_dataset(data_file, features)
    dataSet = dataSet.drop('id', 1)

    features.pop(0)
    features.pop()

    forestClassifer(dataSet,features)

if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")