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
CACHE=True
# from numba import complex64, complex128, float32, float64, int32, jit, njit, prange
SQRT2 = np.sqrt(2)
njitSerial = nb.njit(fastmath=FASTMATH,cache=CACHE)
jitSerial = nb.jit(fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
jitParallel = nb.jit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)

@njitSerial
def construct_w_Half(n):
    wHalf = np.random.randn(n)+1j*np.random.randn(n)
    # wHalf[0] = wHalf[0].real*np.sqrt(2)
    wHalf[0] = 2*wHalf[0].real
    # return wHalf/np.sqrt(2)
    return wHalf/SQRT2

@njitParallel
def inner(u,v):
    sumUV = 0
    for i in nb.prange(len(u)):
        sumUV += u[i]*v[i]
    return sumUV

@nb.vectorize([nb.complex128(nb.int64,nb.float64)],cache=CACHE,nopython=True)
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

@nb.vectorize([nb.float64(nb.float64)],cache=CACHE,nopython=True)
def kappaFun(ut):
    """
    kappa function as a function of u in time domain
    """
    # res = np.zeros(ut.shape[0],dtype=np.float64)
    # for i in nb.prange(ut.shape[0]):
    #     res[i] = math.exp(-ut[i])
    # return res
    return np.exp(-ut)

# @njitParallel
@nb.vectorize([nb.float64(nb.float64)],cache=CACHE,nopython=True)
def kappa_pow_min_nu(ut):
    # res = np.zeros(ut.shape[0],dtype=np.float64)
    # for i in nb.prange(ut.shape[0]):
    #     res[i] = math.exp(1.5*ut[i])
    # return res
    # return kappaFun(ut)**(-1.5)
    return np.exp(1.5*ut)

# @njitParallel
@nb.vectorize([nb.float64(nb.float64)],cache=CACHE,nopython=True)
def kappa_pow_half(ut):
    # res = np.zeros(ut.shape[0],dtype=np.float64)
    # for i in nb.prange(ut.shape[0]):
    #     res[i] = math.exp(-0.5*ut[i])
    # return res
    return np.exp(-0.5*ut)
    # return np.sqrt(kappaFun(ut))

@njitParallel
def norm2(u):
    """
    Compute euclidean squared norm 2 of a complex vector
    """
    norm2=0
    for i in nb.prange(len(u)):
        norm2 += u[i].imag*u[i].imag + u[i].real*u[i].real
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

@njitSerial
def sigmasLancos(n):
    """
    sigma Lancos coefficients for calculating inverse Fourier Transforms
    """
    k = np.arange(1,n+1)
    return np.sin(np.pi*(k/(n+1)))/(np.pi*(k/(n+1)))

@njitSerial
def updateWelford(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1 
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# # retrieve the mean, variance and sample variance from an aggregate
@njitSerial
def finalizeWelford(existingAggregate):
    (count, mean, M2) = existingAggregate
    # (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    (mean, variance) = (mean, M2/count) 
    # if count < 2:
        # return float('nan')
    # else:
    return (mean, variance)

@njitSerial
def extend(uSymmetric,num): 
    n = (uSymmetric.shape[0]+1)//2
    if num> n:
        z = np.zeros(2*num-1,dtype=np.complex128)
        z[(num-1)-(n-1):(num-1)+n] = uSymmetric
        return z    
    else: 
        return uSymmetric

@njitSerial
def symmetrize(w_half):
    w = np.concatenate((w_half[:0:-1].conj(),w_half)) #symmetrize
    return w

def kaczmarz(A,b,max_iteration):
    m = A.shape[0]
    n = A.shape[1]
    if m != b.shape[0]:
        raise Exception("Matrix and vector size missmatch")
    
    #set initial condition:
    x = np.zeros(n,dtype=np.complex128)
    #computing probability of each row
    # prob_row = np.zeros(m,dtype=np.float64)
    A_row_squared_norm = np.zeros(m,dtype=np.float64)
    # A_normalized = A
    for i in nb.prange(m):
        A_row_squared_norm[i] = norm2(A[i,:])
        #in_place_normalization
        A[i,:] = A[i,:]/np.sqrt(A_row_squared_norm[i])
        b[i] = b[i]/np.sqrt(A_row_squared_norm[i])
            
    # prob_row = A_row_squared_norm/np.sum(A_row_squared_norm)
    # cum_prob_row = np.zeros(m+1,dtype=np.float64)
    # cum_prob_row[0] = prob_row[0]
    # for i in nb.prange(1,m):
        # cum_prob_row[i] = cum_prob_row[i-1]+prob_row[i-1]
        
    # error = norm2(A@x - b)
    # while(error>tolerance):
    for k in nb.prange(max_iteration):
        i = k%m
        # i = get_random_index(cum_prob_row,np.random.rand(),m)
        x = x + (b[i] - inner(A[i,:],x.conj()) )*A[i,:]
        # error = norm2(A@x - b)
        # print('error = {0}, i = {1}'.format(error,i))
    error = norm2(A@x - b)
    return x,error

# @njitParallel
def random_kaczmarz(A,b,max_iteration):
    m = A.shape[0]
    n = A.shape[1]
    if m != b.shape[0]:
        raise Exception("Matrix and vector size missmatch")
    
    #set initial condition:
    x = np.zeros(n,dtype=np.complex128)
    #computing probability of each row
    prob_row = np.zeros(m,dtype=np.float64)
    A_row_squared_norm = np.zeros(m,dtype=np.float64)
    for i in nb.prange(m):
        A_row_squared_norm[i] = norm2(A[i,:])
            
    prob_row = A_row_squared_norm/np.sum(A_row_squared_norm)
    cum_prob_row = np.zeros(m+1,dtype=np.float64)
    cum_prob_row[0] = prob_row[0]
    for i in nb.prange(1,m):
        cum_prob_row[i] = cum_prob_row[i-1]+prob_row[i-1]
        
    # error = norm2(A@x - b)
    # while(error>tolerance):
    for k in nb.prange(max_iteration):
        i = get_random_index(cum_prob_row,np.random.rand(),m)
        x = x + (b[i] - inner(A[i,:],x.conj()) )*A[i,:]/A_row_squared_norm[i]
        # error = norm2(A@x - b)
        # print('error = {0}, i = {1}'.format(error,i))
    error = norm2(A@x - b)
    return x,error
    
@njitSerial       
def get_random_index(cum_prob_row,randNumber,m):
    i = 0
    while cum_prob_row[i]<randNumber and i<m-1:
        i +=1
    return i