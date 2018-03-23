import pandas as pd
import os
import argparse
import csv 
import timeit
from splitFile import split
from io import StringIO
from mergeSort import mergeSort 
from heapSort import heapSort   

# Script VERSION
__VERSION = "1.0"

def doWithMerge(array_to_sort):  
    """
        Test the array file with Merge Sort
        Args: File
    """
    start_time = timeit.default_timer()
    mergeSort(array_to_sort)
    elapsed = timeit.default_timer() - start_time
    print("Merge time - ", elapsed)

    # Save result to file
    df = pd.DataFrame(array_to_sort)
    df.to_csv("./outputs/result_merge.csv")

def doWithHeap(array_to_sort):
    """
        Test the array file with Heap Sort
        Args: File
    """
    start_time = timeit.default_timer()
    heapSort(array_to_sort)
    elapsed = timeit.default_timer() - start_time
    print("Heap time - ", elapsed)

    # Save result to file
    df = pd.DataFrame(array_to_sort)
    df.to_csv("./outputs/result_heap.csv")


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
    args = parser.parse_args()

    array_file = 'GeneratedData/output_51.csv'

    data = pd.read_csv(array_file, sep=",",engine = 'python', names=['ID', 'length','array'])

    for index, row in data.iterrows():
        new_array = row['array'].replace("]", "").replace("[", "")

    test_array = [int(s) for s in new_array.split('  ')]

    maior = test_array[0]
    menor = test_array[0]
    for item in test_array:
        if(item > maior):
            maior = item
        if(item < menor):
            menor = item
        
    print("Maior: ", maior)
    print("Menor: ", menor)

    print("Tamanho: ", row['length'])

    if args.version:
        print("Amazing Script Version: " + __VERSION)
        exit()
    if args.merge:       
        doWithMerge(test_array)
    if args.heap:       
        doWithHeap(test_array)
    if args.splitFile:       
        split(open("../Dados/train-arrays.csv", "r"))
        exit()
    else:
        print("usage: mergeOrHeap.py [-v] [--merge] [--heap]")



if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print("Interrupt received! Exiting cleanly...")