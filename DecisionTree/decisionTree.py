from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../')

from geral.helpFunctions import load_dataset, generatePredictionFile, first_dataset


def create_classifier(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #scaler = StandardScaler()
    #scaler.fit(X_train)

    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)

    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=1000, random_state=99)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)
    print(dt.feature_importances_)

    return dt

def create_target(dataSet, features, targetDataSet, target_features):
    y = dataSet['target']
    X = dataSet[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)

    X_test = targetDataSet[target_features]
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=1000, random_state=99)

    dt.fit(X_train, y_train)
    prediction = dt.predict(X_test)

    cols = ['Predicted']
    cenas = targetDataSet['id_target']
    print(cenas.all())
    features = pd.DataFrame(prediction, columns=cols)
    file_name = "decision.csv"
    features.set_index(cenas, inplace=True)
    features.to_csv(file_name, sep=',', encoding='utf-8')


def createTreeClassifier(trainDataSet, features, targetDataSet):
    X = trainDataSet[features]
    y = trainDataSet['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    decisionTree = DecisionTreeClassifier(criterion='gini',
                                          min_samples_split=1000,
                                          random_state=69,
                                          max_features='auto')
    decisionTree.fit(X_train, y_train)
    y_pred = decisionTree.predict(X_test)
    pred = (accuracy_score(y_test, y_pred) * 100)
    print("Prediction: ", pred)
    decisionTree.fit(trainDataSet[features], trainDataSet['target'])
    prediction = decisionTree.predict(targetDataSet)
    generatePredictionFile(targetDataSet, prediction, file_name="treePrediction.csv")
    return decisionTree


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree,
                        out_file=f,
                        feature_names=feature_names,
                        class_names=['1', '2'],
                        filled=True, rounded=True,
                        special_characters=True)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]

    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

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

    decisionTree = createTreeClassifier(trainDataSet, trainFeatures, targetDataSet)
    visualize_tree(decisionTree, trainFeatures)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")