import sys
sys.path.append('../')
from geral.helpFunctions import load_dataset, generatePredictionFile, first_dataset, inRange
import pandas as pd
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection


def createSVMClassifier(trainDataSet, features, targetDataSet):
    X = trainDataSet[features]
    y = trainDataSet['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    support = svm.SVC(C=0.5, kernel='rbf')
    support.fit(X_train, y_train)
    y_pred = support.predict(X_test)
    print(y_pred)
    pred = (accuracy_score(y_test, y_pred) * 100)
    print("Prediction: ", pred)
    support.fit(trainDataSet[features], trainDataSet['target'])
    prediction = support.predict(targetDataSet)
    generatePredictionFile(targetDataSet, prediction, file_name="svmPrediction.csv")


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
        'gamma': ['auto'],
        'kernel': ['linear','poly','rbf','sigmoid'],
        'C': [0.2, 0.1, 0.05, 0.07, 0.01],
    }
    rsearch = model_selection.GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1,cv=10, verbose=True)
    rsearch.fit(X_train, y_train)
    print('The best parameters are : ')
    print(rsearch.best_params_)
    y_pred = rsearch.predict(X_test)
    pred = (accuracy_score(y_test, y_pred) * 100)
    print("Prediction: ", pred)


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

    #svcClassifer(trainDataSet, trainFeatures)
    createSVMClassifier(trainDataSet, trainFeatures, targetDataSet)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")