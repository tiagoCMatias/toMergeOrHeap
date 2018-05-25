import sys
sys.path.append('../')
from geral.helpFunctions import load_dataset, generatePredictionFile, first_dataset, inRange
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection



def forestClassifer(dataset, features):

    X = dataset[features]
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier()
    parameters = {
        'max_features': ["auto", "sqrt", "log2"],
        "max_depth": [3],
        'criterion': ['gini'],
        'min_samples_split': inRange(2,20),
        'min_samples_leaf': inRange(2,20)
    }
    rsearch = model_selection.GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1,cv=10, verbose=True)
    rsearch.fit(X_train, y_train)

    print('The best parameters are : ')
    print(rsearch.best_params_)
    y_pred = rsearch.predict(X_test)
    pred = (accuracy_score(y_test, y_pred) * 100)
    print("Prediction: ", pred)


def createRandomForest(trainDataSet, features, targetDataSet):
    X = trainDataSet[features]
    y = trainDataSet['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    forest = RandomForestClassifier(criterion='gini',
                                    max_features=None,
                                    min_samples_split=1000,
                                    min_samples_leaf=1000,
                                    n_estimators=10,
                                    min_weight_fraction_leaf=0.18,
                                    min_impurity_decrease=0.0165,
                                    random_state=99)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    pred = (accuracy_score(y_test, y_pred) * 100)
    print("Prediction: ", pred)
    forest.fit(trainDataSet[features], trainDataSet['target'])
    prediction = forest.predict(targetDataSet)
    generatePredictionFile(targetDataSet, prediction, file_name="randomPrediction.csv")



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

    #forestClassifer(trainDataSet, trainFeatures)
    createRandomForest(trainDataSet, trainFeatures, targetDataSet)



if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")