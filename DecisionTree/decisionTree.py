from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

from sklearn import model_selection
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

def load_dataset(data_file, cols):
    return pd.read_csv(data_file, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python', memory_map=True)

def first_dataset(data_file):
    firstline = pd.read_csv(data_file)
    dataset = list(firstline)
    return dataset
def treeClassifier(dataset, features):

    X = dataset[features]
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = DecisionTreeClassifier()
    parameters = {
        'criterion': ['gini'],
        'min_samples_split': [100, 500, 1000, 2000],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'random_state': [1, 10, 50, 99],
        'class_weight': ['balanced', None],
    }
    rsearch = model_selection.GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1,cv=10)
    rsearch.fit(dataset[features], dataset['target'])

    #summarize the results of the random parameter search
    #print(rsearch)
    #print(rsearch.best_score_)
    print(rsearch.best_params_)

    rsearch.fit(X_train, y_train)
    y_pred = rsearch.predict(X_test)
    pred = accuracy_score(y_test, y_pred) * 100

    #print("Best Prediction: ", pred)

def create_classifier(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=1000, random_state=99)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)
    print(dt.feature_importances_)

    return dt

def create_target(dataSet, features, targetDataSet, target_features):
    y = dataSet['target']
    X = dataSet[features]

    X_test, y_test = train_test_split(X, y, test_size=0.2)

    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=1000, random_state=99)
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
    data_file = 'boa.csv'

    target_file = 'main-features2.csv'

    name_of_features = first_dataset(data_file)
    target_features = first_dataset(target_file)

    dataSet = load_dataset(data_file, name_of_features)
    targetDataSet = load_dataset(target_file, target_features)

    dataSet = dataSet.drop('id', 1)

    name_of_features.pop(0)
    name_of_features.pop(0)

    name_of_features.pop()

    target_features.pop(0)
    #print (name_of_features)
    #print (target_features)


    #create_target(dataSet, name_of_features, targetDataSet, target_features)
    
    treeClassifier(dataSet, name_of_features)

if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")