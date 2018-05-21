import pandas as pd
import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
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
from collections import Counter
import csv

csv.field_size_limit(100000000)

# Script VERSION
__VERSION = "1.0"


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def doWithMerge(array_to_sort):  
    """
        Test the array file with Merge Sort
        Args: File
    """
    array_length = len(array_to_sort)
    start_time = timeit.default_timer()
    mergeSort.mergeSort(array_to_sort)
    elapsed = timeit.default_timer() - start_time
    print("Merge time - ", format(elapsed, '.6f'))
    return elapsed
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
    elapsed = timeit.default_timer() - start_time
    print("Heap time - ", format(elapsed, '.6f'))

    return elapsed
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


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


def drawGraph(array):
    # Make a figure
    fig = plt.figure()
    # Make room for legend at bottom
    fig.subplots_adjust(bottom=0.2)
    # Your second subplot, for lists 4&5
    ax1 = fig.add_subplot(111)
    # Plot lines 4&5
    # ax1.plot(merge_array, label='list 4', color="red")
    ax1.plot(array, label='Array', color="red")
    # Display the figure
    plt.show()


def drawMultiArrayGraph(merge_array, heap_array):
    #calcular picos
    max_merge, min_merge = peakdet(merge_array,.3)
    max_heap, min_heap = peakdet(heap_array, .3)
    # Make a figure
    fig = plt.figure()
    # Make room for legend at bottom
    fig.subplots_adjust(bottom=0.2)
    # Create subplot for multiple lines
    ax1 = fig.add_subplot(111)
    # Plot lines
    ax1.plot(merge_array, label='Merge', color="red")
    ax1.plot(heap_array, label='Heap', color="green")
    # Display the figure
    plt.title("Compare Merge and Heap arrays")
    plt.legend(loc='upper left', numpoints=1)
    print("merge:", len(max_merge), " - heap:", len(max_heap))
    print("mergeMaxAveg:", (mean(max_merge.flatten())), " - heapMaxAveg:", mean(max_heap.flatten()))
    plt.show()


def getPredictedData(file):
    return pd.read_csv(file, sep=",", names=['id', 'target'])


def getFeatureFromArray(data_array, listOfFeatures, target):
        parts = 5
        divided_array = split_list(data_array, wanted_parts=parts)
        n_inversions = 0
        inv = []
        for array in divided_array:
            i = 1
            for i in range(len(array)):
                value_before = array[i - 1]
                if value_before > array[i]:
                    n_inversions += 1
            inv.append(n_inversions)
            n_inversions = 0
        mean_value = mean(data_array)
        maxtab, mintab = peakdet(data_array,.3)
        max_mean = (mean(maxtab.flatten()))
        min_mean = (mean(mintab.flatten()))
        listOfFeatures.append({
            'len': len(data_array),
            'inv1': inv[0],
            'inv2': inv[1],
            'inv3': inv[2],
            'inv4': inv[3],
            'inv5': inv[4],
            'total_inv': sum(inv),
            'median': format(median(data_array), '.0f'),
            'mean': format(mean_value, '.2f'),
            'max_mean': max_mean,
            'min_mean': min_mean,
            'max_length': len(maxtab),
            'min_length': len(mintab),
            'target': target
        })


def extractSampleFeature(chunksize=10):
    big_file = "Dados/train-arrays.csv"
    data_pred = getPredictedData("Dados/knn-train.csv")
    count_total = 0
    merge_array = list()
    heap_array = list()
    featureList = []
    cols = ['id', 'len', 'array']
    start_time = timeit.default_timer()
    for data_train in pd.read_csv(big_file, encoding ="latin_1", sep=",", names=cols , engine='python', iterator=True, chunksize=chunksize):
        count_total += 1
        for index, row in data_train.iterrows():
            if row['len'] == 100:
                data_array = row['array'].replace("]", "").replace("[", "")
                data_array = [int(s) for s in data_array.split('  ')]
                predicted = int(data_pred['target'][data_pred['id'] == row['id']].values)
                getFeatureFromArray(data_array, featureList, predicted)
                count_total += 1
            #    if merge_array and heap_array:
            #       drawMultiArrayGraph(merge_array, heap_array)
            if row['len'] > 100:
                cols_feature = ['len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                                'median' , 'mean',
                                'max_mean', 'min_mean',
                                'max_length', 'min_length',
                                'target']

                features = pd.DataFrame(featureList, columns=cols_feature)
                file_name = "Dados/train-feature5.csv"
                features.to_csv(file_name, sep=',', encoding='utf-8', index_label='id')
                return

        if count_total%10 == 0:
            print("Itera: ", count_total, "array_size:", row[1])
            sys.stdout.flush()



def extractFeature(chunksize=10):
    big_file = "Dados/train-arrays.csv"
    data_pred = getPredictedData("Dados/knn-train.csv")
    count_total = 0
    featureList = []
    cols = ['id', 'len', 'array']
    cols_feature = ['id_target', 'len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                    'median', 'mean',
                    'max_mean', 'min_mean',
                    'max_length', 'min_length',
                    'target']
    start_time = timeit.default_timer()
    for data_train in pd.read_csv(big_file, encoding="latin_1", sep=",", names=cols, engine='python', iterator=True, chunksize=chunksize):
        count_total += 1
        for index, row in data_train.iterrows():
                data_array = row['array'].replace("]", "").replace("[", "")
                data_array = [int(s) for s in data_array.split('  ')]
                predicted = int(data_pred['target'][data_pred['id'] == row['id']].values)
                getFeatureFromArray(data_array, featureList,  target=predicted)
        if count_total % 10 == 0:
            print("Itera: ", count_total, "array_size:", row[1], "id:", row['id'])
            sys.stdout.flush()
    elapsed = timeit.default_timer() - start_time
    print("time - ", format(elapsed, '.2f'))
    features = pd.DataFrame(featureList, columns=cols_feature)
    file_name = "Dados/main-features.csv"
    features.to_csv(file_name, sep=',', encoding='utf-8', index_label='id')



def setFeatures(data_array, featureList, id):
    parts = 5
    divided_array = split_list(data_array, wanted_parts=parts)
    n_inversions = 0
    inv = []
    for array in divided_array:
        i = 1
        for i in range(len(array)):
            value_before = array[i - 1]
            if value_before > array[i]:
                n_inversions += 1
        inv.append(n_inversions)
        n_inversions = 0
    mean_value = mean(data_array)
    maxtab, mintab = peakdet(data_array, .3)
    max_mean = (mean(maxtab.flatten()))
    min_mean = (mean(mintab.flatten()))
    featureList.append({
        'id_target': id,
        'len': len(data_array),
        'inv1': inv[0],
        'inv2': inv[1],
        'inv3': inv[2],
        'inv4': inv[3],
        'inv5': inv[4],
        'total_inv': sum(inv),
        'median': format(median(data_array), '.0f'),
        'mean': format(mean_value, '.2f'),
        'max_mean': max_mean,
        'min_mean': min_mean,
        'max_length': len(maxtab),
        'min_length': len(mintab)
    })

    #id,array_id,len,n_inversions,repeats,mean,median,target



def getPercentage():
    dataSet_File = "Dados/train-arrays.csv"
    target_file = "Dados/train-target.csv"
    cols=['id', 'predicted']

    for x in  pd.read_csv(dataSet_File, index_col=False, names=['x'], iterator=True, chunksize=10, header=None).iloc[:, 1]:
        print(x.head())



def prepareData():
    target_file = "../Dados/target.csv"
    count_total = 0
    featureList = []
    cols = ['id', 'len', 'array']
    start_time = timeit.default_timer()
    for data_train in pd.read_csv(target_file, encoding="latin_1", sep=",", names=cols, engine='python', iterator=True, chunksize=10):
        count_total += 1
        for index, row in data_train.iterrows():
                data_array = row['array'].replace("]", "").replace("[", "")
                data_array = [int(s) for s in data_array.split('  ')]
                setFeatures(data_array, featureList, id=row['id'])
                count_total += 1
        if count_total % 10 == 0:
            print("Itera: ", count_total, "array_size:", row[1], "id:", row['id'])
            sys.stdout.flush()

#    cols_feature = [ 'id', 'len', 'n_inversions', 'median', 'mean', 'repeats']
    cols_feature = ['id_target', 'len', 'inv1', 'inv2', 'inv3', 'inv4', 'inv5', 'total_inv',
                    'median', 'mean',
                    'max_mean', 'min_mean',
                    'max_length', 'min_length',
                    ]

    features = pd.DataFrame(featureList, columns=cols_feature)
    file_name = "Dados/main-features2.csv"
    features.to_csv(file_name, sep=',', encoding='utf-8', index_label='id')




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
    parser.add_argument("-e", "--extractfeatures",
                        action="store_true",
                        help="Extract Features from file")

    args = parser.parse_args()


    if args.version:
        print("Amazing Script Version: " + __VERSION)
        exit()
    if args.feature:
        print("Extract features")
        stop_condition = int(args.feature)
        if(stop_condition == 1):
            print("extractSampleFeature")
            extractSampleFeature()
        else:
            print("extractFeature")
            extractFeature()
    if args.merge:
        # doWithMerge(test_array)
        print("Do merge")
    if args.heap:
        # doWithHeap(test_array)
        print("Do heap")
    if args.extractfeatures:
        print("preparing Data")
        prepareData()
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


#          H  |  M
# 10       1     1412
# 100      565   834
# 1000     1332  89
# 100000   76    1335
# 1000000  4     1382
