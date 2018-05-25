import pandas as pd


def load_dataset(data_file, cols):
    return pd.read_csv(data_file, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python', memory_map=True)

def first_dataset(data_file):
    firstline = pd.read_csv(data_file)
    dataset = list(firstline)
    return dataset

def generatePredictionFile(targetDataSet, prediction, file_name="prediction.csv"):
    cols = ['Predicted']
    indexCol = targetDataSet['id_target']
    indexCol.columns = ['Id']
    features = pd.DataFrame(prediction, columns=cols)
    features.set_index(indexCol, inplace=True)
    features.to_csv(file_name, sep=',', encoding='utf-8')



def inRange(begin, final):
    result = []
    for x in range (begin, final):
        result.append(x)
    return result