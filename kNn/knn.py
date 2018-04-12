import csv
import random
import math
import operator
import pandas as pd


def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, "rt", encoding='utf8') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        nr = str(dataset[1])
        for x in range(1,len(dataset)):
            for y in range(nr.count(',')+1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
        

def euclideanDistance(instance1, instance2, length):
    """
    Euclidean distance is defined as 
    the square root of the sum of the squared differences between the two arrays of numbers
    """
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    """
    collect the k most similar instances for a given unseen instance
    returns k most similar neighbors from the training set for a given test instance
    """
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    """
    sums the total correct predictions and 
    returns the accuracy as a percentage of correct classifications.
    """
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    data_file = './train-feature.csv'
    loadDataset(data_file, split, trainingSet, testSet)
    print ('Train set: ', repr(len(trainingSet)))
    print ('Test set: ', repr(len(testSet)))
    # generate predictions
    predictions=[]
    k =  2
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=', repr(result) , ', actual=' , repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' , repr(accuracy) , '%')


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")