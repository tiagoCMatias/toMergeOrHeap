from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import random
import subprocess
import pandas as pd
import numpy as np

def load_dataset(data_file, cols):
    return pd.read_csv(data_file, encoding="latin_1", sep=",", names=cols, engine='python', memory_map=True)


def create_classifier(dataSet, features):
    y = dataSet['target']
    X = dataSet[features]
    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X, y)
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
    features = ['len']
    cols = ['id', 'len', 'target']
    data_file = '../Dados/feature-file.csv'
    dataSet = load_dataset(data_file, cols)
    dataSet = dataSet.drop('id', 1)
    dt = create_classifier(dataSet, features)
    visualize_tree(dt, features)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")