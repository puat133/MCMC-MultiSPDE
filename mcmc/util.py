# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:17:09 2018

@author: puat133
"""
# import math
import h5py
import scipy.io as sio
import numpy as np
import scipy.linalg as sla
import numba as nb
import time
import math
FASTMATH=True
PARALLEL = False
CACHE=False
# from numba import complex64, complex128, float32, float64, int32, jit, njit, prange
SQRT2 = np.sqrt(2)
njitSerial = nb.njit(fastmath=FASTMATH,cache=CACHE)
jitSerial = nb.jit(fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
jitParallel = nb.jit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)


@njitParallel
def inner(u,v):
    sumUV = 0
    for i in nb.prange(len(u)):
        sumUV += u[i]*v[i]
    return sumUV

@njitSerial
def eigenFunction1D(i,t):
    """
    Return an eigen function of Laplacian operator in one dimension
    i           - index int
    t           - time  float 
    """
    return np.exp(2*np.pi*1j*i*t)

@njitParallel
def matMulti(A,D):
    """
    Matrix multiplication A@D where A,D is a diagonal matrices, and D is a diagonal matrix
    """
    C = np.zeros(A.shape,dtype=np.complex128)
    for i in nb.prange(A.shape[0]):
        for j in nb.prange(A.shape[1]):
            C[i,j] = A[i,j]*D[j,j]

    return C

# @njitParallel
# def logDet(L):
#     """
#     # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
#     # L^dagger L is Hermitian
#     """
#     return (np.linalg.slogdet(L)[1])
#     # return 0.5*(np.linalg.slogdet(L.T.conj()@L)[1])
#     # return 0.5*np.sum(np.log(np.linalg.eigvalsh(L.T.conj()@L)))
#     # return  np.sum(np.log(np.absolute(np.linalg.eigvals(L))))

@njitParallel
def kappaFun(ut):
    """
    kappa function as a function of u in time domain
    """
    # res = np.zeros(ut.shape[0],dtype=np.float64)
    # for i in nb.prange(ut.shape[0]):
    #     res[i] = math.exp(-ut[i])
    # return res
    return np.exp(-ut)

# @njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
@njitParallel
def kappa_pow_min_nu(ut):
    # res = np.zeros(ut.shape[0],dtype=np.float64)
    # for i in nb.prange(ut.shape[0]):
    #     res[i] = math.exp(1.5*ut[i])
    # return res
    return kappaFun(ut)**(-1.5)
    # return np.exp(1.5*ut)

@njitParallel
def kappa_pow_half(ut):
    # res = np.zeros(ut.shape[0],dtype=np.float64)
    # for i in nb.prange(ut.shape[0]):
    #     res[i] = math.exp(-0.5*ut[i])
    # return res
    # return np.exp(-0.5*ut)
    return np.sqrt(kappaFun(ut))

@njitParallel
def norm2(u):
    """
    Compute euclidean squared norm 2 of a complex vector
    """
    norm2=0
    for i in nb.prange(len(u)):
        norm2 += u[i].imag**2 + u[i].real**2
    return norm2

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()