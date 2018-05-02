import pandas as pd
import numpy as np
import os
import argparse
import sys
import timeit
import collections
from splitFile import split
from statistics import *
from io import StringIO
from sortAlgorithms import ( mergeSort, heapSort )
import matplotlib.pyplot as plt

# Script VERSION
__VERSION = "1.0"


def doWithMerge(array_to_sort):  
    """
        Test the array file with Merge Sort
        Args: File
    """
    array_length = len(array_to_sort)
    start_time = timeit.default_timer()
    mergeSort.mergeSort(array_to_sort)
    return timeit.default_timer() - start_time
    #print("Merge time - ", format(elapsed, '.6f'))

    # Save result to file
    #df = pd.DataFrame(array_to_sort)
    #df.to_csv("./outputs/result_merge.csv")


def doWithHeap(array_to_sort):
    """
        Test the array file with Heap Sort
        Args: File
    """
    start_time = timeit.default_timer()
    heapSort.heapSort(array_to_sort)
    return timeit.default_timer() - start_time
    #print("Heap time - ", format(elapsed, '.6f'))

    # Save result to file
    #df = pd.DataFrame(array_to_sort)
    #df.to_csv("./outputs/result_heap.csv")


def getPredicted(index, prediction):
    """
        Save to array the predicted data
        Args: array
    """
    preditec_file = "../Dados/train-target.csv"
    data = pd.read_csv(preditec_file, sep=",",engine = 'python', names=['ID', 'Predicted'])
    array_predited = data['Predicted'].iloc[index]
    prediction = array_predited
    if(int(prediction) == 1):
        print("Prediction: ",prediction,"Heap - ID:", index)
    else:
        print("Prediction: ",prediction,"Merge - ID:", index)


def trySort(chunksize=10):
    big_file = "Data/train-arrays.csv"
    predictedFile = "Data/train-target.csv"
    data_pred = pd.read_csv(predictedFile, sep=",", names=['id', 'predicted'])
    count_total = 0
    count_error = 0
    featureList = []
    merge_array = list()
    cols = ['id', 'len', 'array']
    cols_feature = ['count', 'id', 'len', 'maior', 'menor', 'mean', 'median', 'target']
    for data_train in pd.read_csv(big_file, encoding ="latin_1", sep=",", names=cols , engine='python', iterator=True, chunksize=chunksize, memory_map=True):
        for index, row in data_train.iterrows():
            if row['len'] == 100:
                data_array = row['array'].replace("]", "").replace("[", "")
                data_array = [int(s) for s in data_array.split('  ')]
                length = row['len']
                merge_time = doWithMerge(data_array)
                heap_time = doWithHeap(data_array)
                predicted = int(data_pred['predicted'][data_pred['id'] == row['id']].values)
                actual_result = 0
                if heap_time < merge_time:
                    actual_result = 1
                else:
                    actual_result = 2
                if(actual_result != predicted):
                    count_error += 1
                if predicted == 2:
                    merge_array.append(data_array)
                    print("In")
                if count_error > 100:
                    # Make a figure
                    fig = plt.figure()
                    # Make room for legend at bottom
                    fig.subplots_adjust(bottom=0.2)
                    # Your second subplot, for lists 4&5
                    ax1 = fig.add_subplot(111)
                    # Plot lines 4&5
                    ax1.plot(merge_array,label='list 4', color="red")
                    ax1.plot(data_array,label='list 5', color="green")

                    # Display the figure
                    plt.show()
                    return;    
                count_total += 1
                
        if(row['len'] > 100):
            
            print("Erros: ", count_error, "Total: ", count_total)
            return






def extractFeature(chunksize=10, stop=500):
    big_file = "Dados/train-arrays.csv"
    predictedFile = "Dados/knn-train.csv"
    data_pred = pd.read_csv(predictedFile, sep=",", names=['id', 'predicted'])
    count = 0
    featureList = []
    cols = ['id', 'len', 'array']
    cols_feature = ['count', 'id', 'len', 'maior', 'menor', 'mean', 'median', 'target']
    for data_train in pd.read_csv(big_file, encoding ="latin_1", sep=",", names=cols , engine='python', iterator=True, chunksize=chunksize, memory_map=True):
        count += 1
        for index, row in data_train.iterrows():
            start_time = timeit.default_timer()
            data_array = row['array'].replace("]", "").replace("[", "")
            data_array = [int(s) for s in data_array.split('  ')]
            length = row['len']
            elapsed = timeit.default_timer() - start_time
            featureList.append(
                {
                    'count': count,
                    'id': row['id'],
                    'len': length,
                    'maior': max(data_array),
                    'menor': min(data_array),
                    'mean': int(np.mean(data_array)),
                    'median': int(np.median(data_array)),
                    #'deviation': stdev(data_array),
                    #'extractTime': format(elapsed, '.6f'),
                    'predicted': int(data_pred['predicted'][data_pred['id'] == row['id']].values)
                 }
            )
        if count%10 == 0:
            print("Itera: ", count, "array_size:", row[1])
            sys.stdout.flush()
        if count%stop == 0:
            break

    features = pd.DataFrame(featureList, columns=cols_feature)
    file_name = "Dados/train-feature2.csv"
    features.to_csv(file_name, sep=',')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version",
                        action="store_true",
                        help="Show current version of this amazing script.")
    parser.add_argument("--merge",
                        action="store_true",
                        help="Run script with Merge Sort")
    parser.add_argument("--heap",
                        action="store_true",
                        help="Run script with Heap Sort")        
    parser.add_argument("--splitFile",
                        action="store_true",
                        help="Run script to split a file")
    parser.add_argument("-f", "--feature",
                        help="Create a new file with features")
    args = parser.parse_args()

    if args.version:
        print("Amazing Script Version: " + __VERSION)
        trySort()
        exit()
    if args.feature:
        print("Extract features")
        stop_condition = args.feature
        extractFeature(stop=int(stop_condition))
    if args.merge:
        # doWithMerge(test_array)
        print("Do merge")
    if args.heap:
        # doWithHeap(test_array)
        print("Do heap")
    if args.splitFile:       
        split(open("../Dados/train-arrays.csv", "r"))
        exit()
    else:
        print("usage: mergeOrHeap.py [-v] [-f] [--merge] [--heap]")


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")