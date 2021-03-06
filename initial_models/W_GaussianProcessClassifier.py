# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:33:44 2020

@author: Vegard
"""

import pandas as pd
from nltk import ngrams
import string
import numpy as np
import timeit
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Importing data start timer
print('Importing data')
start = timeit.default_timer()

# Set datafile
datafile =  'data/data.csv'
# Load datafile
data = pd.read_csv(datafile)

# Removing unwanted variables
del datafile

# Importing data stop timer
stop = timeit.default_timer()
print('Time: ', stop - start)

# Dividing into training, development, and test.
data_train = data.where(data['split']=='train').dropna(how='all')
data_dev = data.where(data['split']=='dev').dropna(how='all')
data_test = data.where(data['split']=='test').dropna(how='all')

# Creating a function to generate ngram content
def getNgrams(inputfield, **kwargs):
    # Set of ngrams
    ngramList = set()
    # Creating a dictionary that holds which reviews has which ngrams
    NgramDict = {}
    # Creating a dictionary that tracks how many times an ngram occurs
    # in a review
    NgramCountDict = {}
    # List of all nGrams
    TotalGrams = []
    # Select length of ngrams
    N = 3
    # Makingsure inputfield is numpy
    inputfield = inputfield.to_numpy()
    for n in range(inputfield.shape[0]):
        # Checking if it is NaN or not
        if not pd.isnull(inputfield[n]):
            # Pulling in the input in question
            tempInput = inputfield[n]
            # Getting ngrams
            TempGrams = ngrams(tempInput.split(), N)
            # We go over the grams
            for grams in TempGrams:
                if grams in NgramDict:
                    NgramDict[grams].add(n)
                    NgramCountDict[grams].append(n)
                else:
                    NgramDict[grams] = set([n])
                    NgramCountDict[grams] = [n]
                ngramList.add(tuple(grams))
                TotalGrams.append(grams)
    # transforming from set into list
    ngramList = list(ngramList)
    if 'output' in kwargs:
        output = kwargs['output']
    else:
        output = 0
    
    if output == 0:
        return ngramList, NgramDict, NgramCountDict, TotalGrams
    elif output == 1:
        return NgramDict, TotalGrams
    else:
        return NgramCountDict, TotalGrams

# Creating a binary matrix from whatever column is chosen.
def getBinaryMatrix(inputfield, N):
    # Inputfield being the source of the text
    # N being the number of ngrams to return in binary format
    # We call the getNgrams function internally
    NgramDict, TotalGrams = getNgrams(inputfield, output = 1)
    # Length of ngramList
    L = len(inputfield)
    # Counting most popular items
    NgramIDX = Counter(TotalGrams).most_common(N*100)
    # Creating our BinaryMatrix
    binary_matrix = np.zeros((N, L))
    for n in range(N):
        idx = list(NgramDict[NgramIDX[n][0]])
        binary_matrix[n, idx] = 1
    return binary_matrix

# Creating a ngram count matrix from whatever column is chosen.
def getCountMatrix(inputfield, N):
    # Inputfield being the source of the text
    # N being the number of ngrams to return in binary format
    # We call the getNgrams function internally
    NgramCountDict, TotalGrams = getNgrams(inputfield, output=2)
    # Length of ngramList
    L = len(inputfield)
    # Counting most popular items
    NgramIDX = Counter(TotalGrams).most_common(N*100)
    # Creating our BinaryMatrix
    count_matrix = np.zeros((N, L))
    for n in range(N):
        idx = list(NgramCountDict[NgramIDX[n][0]])
        idx_count = np.bincount(idx)
        if len(idx_count) != len(inputfield):
            difference = len(inputfield)-len(idx_count)
            idx_count = np.hstack((idx_count, np.zeros(difference)))
        count_matrix[n, :] = idx_count
    return count_matrix

# Creating initial binary matrix for PCA
X = np.zeros((2, len(data_train)))
Xdev = np.zeros((2, len(data_dev)))
Xtest = np.zeros((2, len(data_test)))

# Making a list of category columns
columns = ['category', 'source']

# Populating initial binary matrix start timer
print('Populating initial binary matrix')
start = timeit.default_timer()
# Creating a list for unique values in categories and source.
CatSourceUnique = []
# Populating the initial binary matrix
for t in range (2):
    col = columns[t]
    temp = list(data_train[col].unique())
    idx_train = data_train[col].to_numpy()
    idx_dev = data_dev[col].to_numpy()
    idx_test = data_test[col].to_numpy()
    for n in range(len(temp)):
        idx = np.where(idx_train == temp[n])
        X[t, idx] = n
        idx = np.where(idx_dev == temp[n])
        Xdev[t, idx] = n
        idx = np.where(idx_test == temp[n])
        Xtest[t, idx] = n
    CatSourceUnique.append(temp)

# Populating initial binary matrix stop timer
stop = timeit.default_timer()
print('Time: ', stop - start)

# Creating ngram binary matrix start timer
print('Creating ngram binary matrix')
start = timeit.default_timer()
# Making a list of relevant columns for Ngrams
columns = ['pros', 'cons', 'content']

for col in columns:
    if col != 'content':
        N = 1000
    else:
        N = 10000
    #temp = getBinaryMatrix(data_train[col], N)
    temp = getCountMatrix(data_train[col], N)
    X = np.vstack((X, temp))
    #temp = getBinaryMatrix(data_dev[col], N)
    temp = getCountMatrix(data_dev[col], N)
    Xdev = np.vstack((Xdev, temp))
    #temp = getBinaryMatrix(data_test[col], N)
    temp = getCountMatrix(data_test[col], N)
    Xtest = np.vstack((Xtest, temp))

del idx, idx_train, idx_dev, idx_test, n, t, columns, temp, col
# Creating ngram binary matrix stop timer
stop = timeit.default_timer()
print('Time: ', stop - start)

# Creating a function that normalizes each matrix
def NormalizeMatrix(X):
    # Getting the number of rows and columns
    [M, N] = X.shape

    # Preprocessing the matrix by row.
    for m in range(M):
        # Centering each entry around zero
        X[m,:] -= np.mean(X[m,:])
        # Specifying the standard deviation of each row
        Xstd = np.std(X[m,:])
        # If standard deviation is not zero, then perform divide. This is to
        # avoid divide by zero errors.
        if Xstd != 0:
            X[m,:] /= Xstd
    return X

# Normalize matrices start time
print('Normalize matrices')
start = timeit.default_timer()
# Normalize matrices
X = NormalizeMatrix(X)
Xdev = NormalizeMatrix(Xdev)
Xtest = NormalizeMatrix(Xtest)
# Normalize matrices stop timer
stop = timeit.default_timer()
print('Time: ', stop - start)

### CREATING CLASSIFIERS ###
# Creating list of true classes
XCtrain = data_train['rating'].to_numpy()
XCdev = data_dev['rating'].to_numpy()
XCtest = data_test['rating'].to_numpy()

for n in np.arange(1,4):
    if n == 1:
        a = 1
        b = 2
    elif n == 2:
        a = 3
        b = 4
    else:
        a = 5
        b = 6
    XCtrain = np.where(XCtrain == a, n, XCtrain)
    XCtrain = np.where(XCtrain == b, n, XCtrain)
    XCdev = np.where(XCdev == a, n, XCdev)
    XCdev = np.where(XCdev == b, n, XCdev)
    XCtest = np.where(XCtest == a, n, XCtest)
    XCtest = np.where(XCtest == b, n, XCtest)

del a, b, n

## Testing dev both as one single large block and divided into 
print('\nGaussian Process Classifier')

start = timeit.default_timer()
kernel = 1.0 * RBF(1.0)
cfunc = GaussianProcessClassifier(kernel=kernel, random_state=0,
            multi_class="one_vs_rest", n_jobs=4)
cfunc.fit(np.transpose(X), XCtrain)
y_pred = cfunc.predict(np.transpose(Xdev))
result = np.sum(y_pred == XCdev)
print('Accuracy single block: ' + str(round(result/Xdev.shape[1], 3)))
stop = timeit.default_timer()
print('Time: ', stop - start)

temptrain = data_train['category'].to_numpy()
tempdev = data_dev['category'].to_numpy()
for n in range(len(CatSourceUnique[0])):
    start = timeit.default_timer()
    idx_train = np.where(temptrain == CatSourceUnique[0][n])
    idx_dev = np.where(tempdev == CatSourceUnique[0][n])
    xtemp = np.transpose(X[:, idx_train])[:,0,:]
    ytemp = XCtrain[idx_train]
    devtemp = np.transpose(Xdev[:,idx_dev])[:,0,:]
    cfunc = GaussianProcessClassifier(kernel=kernel, random_state=0,
            multi_class="one_vs_rest", n_jobs=4)
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)

## Parzen
