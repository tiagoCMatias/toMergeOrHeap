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
sys.path.append('../')

from geral.helpFunctions import load_dataset, generatePredictionFile, first_dataset
from scipy.stats import uniform as sp_rand

from sklearn import model_selection

def prepare_data(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_best_pred = 0
    place = 0
    max_iter = 100                 #activation = ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    clf = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,
                       beta_2=0.999, early_stopping=False, epsilon=1e-08,
                       hidden_layer_sizes=(100,), learning_rate='constant',
                       learning_rate_init=0.001, max_iter=200, momentum=0.9,
                       nesterovs_momentum=True, power_t=0.5, random_state=None,
                       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                       verbose=False, warm_start=False)

    for i in range(max_iter):
        clf.fit(X_train, y_train)
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

def create_target(dataSet, features, targetDataSet, target_features):
    y = dataSet['target']
    X = dataSet[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)

    X_test = targetDataSet[target_features]
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    dt = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

    dt.fit(X_train, y_train)
    prediction = dt.predict(X_test)

    cols = ['Predicted']
    cenas = targetDataSet['id_target']
    print(cenas.all())
    features = pd.DataFrame(prediction, columns=cols)
    file_name = "nn_test.csv"
    features.set_index(cenas, inplace = True)
    features.to_csv(file_name, sep=',', encoding='utf-8')


def generateOutput():
    data_file = '../boa.csv'

    target_file = '../Dados/main-features2.csv'

    name_of_features = first_dataset(data_file)
    target_features = first_dataset(target_file)

    dataSet = load_dataset(data_file, name_of_features)
    targetDataSet = load_dataset(target_file, target_features)
    dataSet = dataSet.drop('id', 1)

    name_of_features.pop(0)
    name_of_features.pop()

    #target_features.pop(0)

    create_target(dataSet, name_of_features, targetDataSet, target_features)


def train():

    file_path = "../Dados/master-features.csv"
    data_file = '../boa.csv'

    name_of_features = first_dataset(data_file)

    dataSet = load_dataset(data_file, name_of_features)
    dataSet = dataSet.drop('id', 1)

    name_of_features.pop(0)
    name_of_features.pop()

    print(name_of_features)


def gridSearch(targetFile, targetFeatures):

    data_file = '../boa.csv'

    name_of_features = first_dataset(data_file)

    dataSet = load_dataset(data_file, name_of_features)
    dataSet = dataSet.drop('id', 1)

    name_of_features.pop(0)
    name_of_features.pop()

    print(name_of_features)

    print(len(name_of_features), len(targetFeatures))

    # prepare a uniform distribution to sample for the alpha parameter
    #param_grid = {'alpha': sp_rand()}

    # create and fit a ridge regression model, testing random alpha values
    model = MLPClassifier()
    parameters = {
        #'learning_rate': ["constant", "invscaling"],
        'hidden_layer_sizes': [(100, 10), (100, 20), (100, 30), (100, 1), (100, 5)],
        'learning_rate': ['constant', 'invscaling'],
        'learning_rate_init': [0.05 , 0.01, 0.1],
        #'alpha': [10.0 ** -np.arange(1, 7)],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    }
    rsearch = model_selection.GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1,cv=10)
    rsearch.fit(dataSet[name_of_features], dataSet['target'])

    print(rsearch)
    # summarize the results of the random parameter search
    print(rsearch.best_score_)
    print(rsearch.best_estimator_.alpha)
    if targetFile:
        prediction = rsearch.predict(targetFile[targetFeatures])
        cols = ['Predicted']
        cenas = targetFile['id_target']
        features = pd.DataFrame(prediction, columns=cols)
        file_name = "output2.csv"
        features.set_index(cenas, inplace=True)
        features.to_csv(file_name, sep=',', encoding='utf-8')


def createNeuralNetworkClassifier(trainDataSet, features, targetDataSet):
    X = trainDataSet[features]
    y = trainDataSet['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    neural = MLPClassifier(activation='relu',
                           hidden_layer_sizes=(100, 20),
                           learning_rate='constant',
                           learning_rate_init=0.01)
    neural.fit(X_train, y_train)
    y_pred = neural.predict(X_test)
    pred = (accuracy_score(y_test, y_pred) * 100)
    print("Prediction: ", pred)
    neural.fit(trainDataSet[features], trainDataSet['target'])
    prediction = neural.predict(targetDataSet)
    generatePredictionFile(targetDataSet, prediction, file_name="neuralPrediction.csv")


def main():
    targetFile = "../Dados/main-features2.csv"
    trainFile = '../boa.csv'

    trainFeatures = first_dataset(trainFile)

    trainDataSet = load_dataset(trainFile, trainFeatures).drop('id', 1)

    trainFeatures.pop(0)
    trainFeatures.pop()

    targetFeatures = first_dataset(targetFile)
    targetDataSet = load_dataset(targetFile, targetFeatures)

    print(trainFeatures)
    print(targetFeatures)

    createNeuralNetworkClassifier(trainDataSet, trainFeatures, targetDataSet)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")