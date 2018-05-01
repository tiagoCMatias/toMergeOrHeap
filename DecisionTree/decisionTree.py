from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


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

    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=500, random_state=99)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)
    return dt


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
    data_file = '../Dados/train-feature5.csv'

    cols2 = ['id','len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                    'stdev', 'pstdev', 'median', 'mean', 'pvariance',
                    'max_mean', 'min_mean',
                    'max_length', 'min_length',
                    'target']
    features = ['len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                'stdev', 'pstdev', 'median', 'mean', 'pvariance',
                'max_mean', 'min_mean',
                'max_length', 'min_length']

    dataSet = load_dataset(data_file, cols2)
    dataSet = dataSet.drop('id', 1)
    #dataSet = dataSet.drop('array_id', 1)
    dt = create_classifier(dataSet, features)
    visualize_tree(dt, features)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")