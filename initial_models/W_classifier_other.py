# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:46:42 2020

@author: Vegard
"""

print('\nAda Boost Classifier')
start = timeit.default_timer()
from sklearn.ensemble import AdaBoostClassifier

cfunc = AdaBoostClassifier(n_estimators=200, algorithm="SAMME.R", random_state=0)
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
    cfunc = AdaBoostClassifier(n_estimators=200, algorithm="SAMME.R", random_state=0)
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)

## Naive Bayes Gaussian
print('\nNaive Bayes Gaussian')
from sklearn.naive_bayes import GaussianNB
start = timeit.default_timer()
y_pred = GaussianNB().fit(np.transpose(X), XCtrain).predict(np.transpose(Xdev))
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
    cfunc = GaussianNB()
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)

## K nearest neighbors
print('\n K nearest neighbors - majority vote implementation')
from sklearn.neighbors import KNeighborsClassifier
# Number of neighbors to take into account
KNN = int(round(np.sqrt(len(np.transpose(X)))))
# Start timer
start = timeit.default_timer()
cfunc = KNeighborsClassifier(n_neighbors=KNN, n_jobs=4)
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
    KNN = int(round(np.sqrt(len(ytemp))))
    cfunc = KNeighborsClassifier(n_neighbors=KNN, n_jobs=4)
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
## Decision Tree
print('\nDecision Tree')
start = timeit.default_timer()
from sklearn.tree import DecisionTreeClassifier

cfunc = DecisionTreeClassifier()
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
    cfunc = DecisionTreeClassifier()
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
## Random Forrest
print('\nRandom Forest Classifier')
start = timeit.default_timer()
from sklearn.ensemble import RandomForestClassifier

cfunc = RandomForestClassifier()
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
    cfunc = RandomForestClassifier()
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)

## MLP Classifier
print('\nMLPClassifier')
start = timeit.default_timer()
from sklearn.neural_network import MLPClassifier

cfunc = MLPClassifier(max_iter=2000, learning_rate="adaptive")
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
    cfunc = MLPClassifier(max_iter=4000, learning_rate="adaptive")
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)


##### UNSUITABLE #####

## Gaussian PRocess Classifier
print('\nGaussian Process Classifier')

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

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

## Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
print('\n Quadratic Discriminant Analysis')
start = timeit.default_timer()
cfunc = QuadraticDiscriminantAnalysis()
cfunc.fit(np.transpose(X), XCtrain)
y_pred = cfunc.predict(np.transpose(Xdev))

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
    cfunc = QuadraticDiscriminantAnalysis()
    cfunc.fit(xtemp, ytemp)
    y_pred = cfunc.predict(devtemp)
    result = np.sum(y_pred == XCdev[idx_dev])
    print('Accuracy ' + CatSourceUnique[0][n] + ' category:'
          + str(round(result/len(idx_dev[0]),3)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)