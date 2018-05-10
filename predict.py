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

from DecisionTree.decisionTree import create_classifier


csv.field_size_limit(100000000)

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


def first_dataset(data_file):
    firstline = pd.read_csv(data_file)
    dataset = list(firstline)
    return dataset

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def getFeatureFromArray(data_array, listOfFeatures, target):
    parts = 5
    divided_array = split_list(data_array, wanted_parts=parts)
    n_inversions = 0
    inv = []
    maxtab =  []
    mintab = []
    count = 0
    for array in divided_array:
        i = 1
        maxOp, minOp = peakdet(array, .3)
        maxtab.append(len(maxOp))
        mintab.append(len(minOp))

        for i in range(len(array)):
            value_before = array[i - 1]
            if value_before > array[i]:
                n_inversions += 1
        inv.append(n_inversions)
        n_inversions = 0
        count += 1
    mean_value = mean(data_array)

    merge_c = 0
    heap_c = 0

    if len(data_array) == 100:
        merge_c = 0.6
        heap_c = 0.4
    if len(data_array) == 1000:
        merge_c = 0.06
        heap_c = 0.94
    if len(data_array) == 10000:
        merge_c = 0.45
        heap_c = 0.55

    start_time = timeit.default_timer()
    count_merge = mergeSort(data_array)
    elapsed_merge = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    count_heap = heapSort(data_array)
    elapsed_heap = timeit.default_timer() - start_time

    #print("Count:", count[0])

    listOfFeatures.append({
        'len': len(data_array),
        'inv1': inv[0],
        'inv2': inv[1],
        'inv3': inv[2],
        'inv4': inv[3],
        'inv5': inv[4],
        'max_inv': sum(inv),

        'merge_count': count_merge[0],
        'heap_count': count_heap,

        'merge_time': format(elapsed_merge, '.4f'),
        'heap_time': format(elapsed_heap, '.4f'),
        #'merge_c': merge_c,

        #'heap_c': heap_c,

        'mean': mean_value,

        'max_1': maxtab[0],
        'max_2': maxtab[1],
        'max_3': maxtab[2],
        'max_4': maxtab[3],
        'max_5': maxtab[4],

        'min_1': mintab[0],
        'min_2': mintab[1],
        'min_3': mintab[2],
        'min_4': mintab[3],
        'min_5': mintab[4],

        'max_mean': mean(maxtab),

        'min_mean': mean(mintab),

        'target': target
    })
    #print(listOfFeatures)
    #drawMultiArrayGraph(maxtab, mintab)


def drawMultiArrayGraph(array1, array2, label1="Max", label2="Min"):
    #calcular picos
    max_merge, min_merge = peakdet(array1,.3)
    max_heap, min_heap = peakdet(array2, .3)
    # Make a figure
    fig = plt.figure()
    # Make room for legend at bottom
    fig.subplots_adjust(bottom=0.2)
    # Create subplot for multiple lines
    ax1 = fig.add_subplot(111)
    # Plot lines
    ax1.plot(array1, label=label1, color="red")
    ax1.plot(array2, label=label2, color="green")
    # Display the figure
    plt.title("Compare Merge and Heap arrays")
    plt.legend(loc='upper left', numpoints=1)
    #print("merge:", len(max_merge), " - heap:", len(max_heap))
    #print("mergeMaxAveg:", (mean(max_merge.flatten())), " - heapMaxAveg:", mean(max_heap.flatten()))
    plt.show()


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

def getPredictedData(file):
    return pd.read_csv(file, sep=",", names=['id', 'target'])


def extractFeature(chunksize=10):
    big_file = "Dados/train-arrays.csv"
    data_pred = getPredictedData("Dados/knn-train.csv")
    count_total = 0
    featureList = []
    cols = ['id', 'len', 'array']
    for data_train in pd.read_csv(big_file, encoding="latin_1", sep=",", names=cols, engine='python', iterator=True, chunksize=chunksize):
        count_total += 1
        for index, row in data_train.iterrows():
                if row['len'] == 100 or row['len'] == 1000 or row['len'] == 10000:
                    data_array = row['array'].replace("]", "").replace("[", "")
                    data_array = [int(s) for s in data_array.split('  ')]
                    predicted = int(data_pred['target'][data_pred['id'] == row['id']].values)
                    getFeatureFromArray(data_array, featureList,  target=predicted)
                    #print(row['len'])
                    #print(data_array)
                    #drawGraph(data_array)
                    #return;
                if row['len'] > 10000:
                    cols = list(pd.DataFrame(featureList).columns.values)
                    features = pd.DataFrame(featureList, columns=cols)
                    file_name = "TrashTesting/features.csv"
                    features.to_csv(file_name, sep=',', encoding='utf-8', index_label='id')
                    #create_classifier(features, cols)
                    return
        if count_total%10 == 0:
            print("Itera: ", count_total, "array_size:", row[1])
            sys.stdout.flush()



def getPercentagem(data_file):
    merge_count = 0
    heap_count = 0

    merge_10 = 0
    merge_100 = 0
    merge_1000 = 0
    merge_10000 = 0
    merge_100000 = 0
    merge_1000000 = 0

    heap_10 = 0
    heap_100 = 0
    heap_1000 = 0
    heap_10000 = 0
    heap_100000 = 0
    heap_1000000 = 0

    cols = first_dataset(data_file)

    for data_train in pd.read_csv(data_file, encoding="latin_1", sep=",", names=cols, engine='python', iterator=True,
                                  chunksize=10):
        for index, row in data_train.iterrows():
            if row['target'] == 2:
                merge_count += 1

                if row['len'] == 10:
                    merge_10 += 1
                if row['len'] == 100:
                    merge_100 += 1
                if row['len'] == 1000:
                    merge_1000 += 1
                if row['len'] == 10000:
                    merge_10000 += 1
                if row['len'] == 100000:
                    merge_100000 += 1
                if row['len'] == 1000000:
                    merge_1000000 += 1
            if row['target'] == 1:
                heap_count += 1
                if row['len'] == 10:
                    heap_10 += 1
                if row['len'] == 100:
                    heap_100 += 1
                if row['len'] == 1000:
                    heap_1000 += 1
                if row['len'] == 10000:
                    heap_10000 += 1
                if row['len'] == 100000:
                    heap_100000 += 1
                if row['len'] == 1000000:
                    heap_1000000 += 1

    print("10:      ", merge_10, heap_10, " - ", merge_10 / (merge_10 + heap_10) * 100)
    print("100:     ", merge_100, heap_100, " - ", merge_100 / (merge_100 + heap_100) * 100)
    print("1000:    ", merge_1000, heap_1000, " - ", merge_1000 / (merge_1000 + heap_1000) * 100)
    print("10000:   ", merge_10000, heap_10000, " - ", merge_10000 / (merge_10000 + heap_10000) * 100)
    print("100000:  ", merge_100000, heap_100000, " - ", merge_100000 / (merge_100000 + heap_100000) * 100)
    print("1000000: ", merge_1000000, heap_1000000, " - ", merge_1000000 / (merge_1000000 + heap_1000000) * 100)

    print("Count: M", merge_count, "H:", heap_count)



def prepareData():
    file = "TrashTesting/test_features.csv"

    cols = first_dataset(file)

    base_features = pd.read_csv(file, encoding="latin_1", sep=",", skiprows=1, names=cols, engine='python')
    data_pred = getPredictedData("Dados/knn-train.csv")

    #print(data_pred)

   # base_features.drop(0)

    for index, row in base_features.iterrows():
        predicted = int(data_pred['target'][data_pred['id'] == row['id']].values)
        base_features.loc[base_features.index[index], 'target'] = int(predicted)
        #print(predicted)

    base_features.to_csv("boa.csv", sep=',', encoding='utf-8', index_label='id')

    print("Done")

def main():
    print("teste")

    data_file = "Dados/master-features.csv"
    #getPercentagem(data_file)
    #extractFeature()
    prepareData()

def mergeSort(alist):
    count = 0
    leftcount = 0
    rightcount = 0
    blist = []
    if len(alist) > 1:
       mid = len(alist) // 2
       lefthalf = alist[:mid]
       righthalf = alist[mid:]
       leftcount, lefthalf = mergeSort(lefthalf)
       rightcount, righthalf = mergeSort(righthalf)

       i = 0
       j = 0

       while i < len(lefthalf) and j < len(righthalf):
         if lefthalf[i] < righthalf[j]:
             blist.append(lefthalf[i])
             i += 1
         else:
             blist.append(righthalf[j])
             j += 1
             count += len(lefthalf[i:])

       while i < len(lefthalf):
          blist.append(lefthalf[i])
          i += 1

       while j < len(righthalf):
          blist.append(righthalf[j])
          j += 1
    else:
        blist = alist[:]

    return count + leftcount + rightcount, blist

def heapify(arr, n, i):
    count = 0
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        count += 1
        arr[i],arr[largest] = arr[largest],arr[i]
        count += heapify(arr, n, largest)
    return count

def heapSort(arr):
    n = len(arr)
    count = 0
    for i in range(n, -1, -1):
        heapify(arr, n, i)
        count += heapify(arr, i, 0)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        count += heapify(arr, i, 0)
    return count



if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")
