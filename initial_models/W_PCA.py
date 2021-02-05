# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:55:46 2020

@author: Vegard
"""

## NOTE! You must provide X yourself, as a normalized numeric matrix where
## columns are the categories. Once that is done just run the section below
## and the ideal PCA will be spat out.

# Importing necessary components.
import numpy as np

# Part-3: SVD computation
def getSVD(cov_matrix):
    U, S, V = np.linalg.svd(cov_matrix,  full_matrices=False)
    # Since only U is called for only U is returned.
    return U

# Compute PCA matrix (K dimensional)
def getKComponents(U, X, K):
    Ureduced = U[:,0:K]
    PCA = np.transpose(Ureduced)@X
    return PCA

# Compute Reconstruction Error
def getVarianceRatio(PCA, K, X):
    # Since many of these equations use Z for the PCA matrix we do the same
    # to keep confusion to a minimum.
    Z = PCA
    # We need covariant matrix
    C = np.cov(X)
    # We need the U matrix.
    U = getSVD(C)
    # Reduce the U matrix
    Ureduced = U[:,0:K]
    #Find Xapprox
    Xapprox = Ureduced@Z
    # Define M (may not actually be necessary, since both numerator and
    # denominator has 1/M in it)
    M = np.shape(X)[0]
    # Calculating the Variance Ratio.
    Xtemp = X - Xapprox
    numerator = (1/M)*sum(np.linalg.norm(Xtemp, axis=1)**2)
    denominator = (1/M)*sum(np.linalg.norm(X, axis=1)**2)
    VarianceRatio = numerator/denominator
    return VarianceRatio

### Find best K value
# Find covariance matrix
C = np.cov(X)
# Define U
U = getSVD(C)
# We define Variance Ratio = 1 to ensure that we do not accidentally
# stop the bisection method before it starts.
VarianceRatio = 1

### IMPORTANT!###
# We write in desired variance ratio
DesiredVariance = 0.01

# Define K1, K2, and K3 which will be used by the bisection algorithm.
K1 = 1
K2 =round(np.shape(X)[0])
K3 = round(np.shape(X)[0]/2)
# While test checks that KBisect is not 1.
KBisect = 0
# Number of iterations
Niter = 0

while KBisect != 1:
    # Perform VarianceRatio test with current K value (that is K3)
    Z = getKComponents(U, X, K3)
    VarianceRatio = getVarianceRatio(Z, K3, X)
    # Carry out bisection algorithm. We continually reduce 
    if VarianceRatio < DesiredVariance:
        K2 = K3
        K3 = K1 + round((K3-K1)/2)
    else:
        K1 = K3
        K3 += round((K2-K3)/2)
    if K3 == K1 or K3 == K2:
        KBisect = 1
        # If K3 == K1, then that suggests K3 is one less than it should be
        # to get the desired precision. If on the other hand K3 = K2
        # we are presumably at the right spot.
        if K3 == K1:
            K = K3 + 1
            # This also means that the K3 we calculated our VarianceRatio
            # for was too low, so we must calculate it again.
            Z = getKComponents(U, X, K)
            VarianceRatio = getVarianceRatio(Z, K, X)
        else:
            K = K3
    Niter += 1

print('The best K is ' + str(K) + '\nThe variance ratio is then ' +
      str(VarianceRatio) + '\nIt took ' + str(Niter) + ' iterations to find K')

PCA = getKComponents(U, X, K)