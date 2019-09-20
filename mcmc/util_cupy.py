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
import cupy as cp
import time
import math
import mcmc.image_cupy as im
import h5py
from cupy.prof import TimeRangeDecorator as cupy_profile
from numba import cuda
SQRT2 = cp.float32(1.41421356)
PI = cp.float32(cp.pi)


def construct_w_Half(n):
    wHalf = cp.random.randn(n,dtype=cp.float32)+1j*cp.random.randn(n,dtype=cp.float32)
    # wHalf[0] = wHalf[0].real*cp.sqrt(2)
    wHalf[0] = 2*wHalf[0].real
    # return wHalf/cp.sqrt(2)
    return wHalf/SQRT2


def inner(u,v):
    return cp.inner(u,v)


# @nb.vectorize([nb.complex128(nb.int64,nb.float64)],cache=CACHE,nopython=True)

def eigenFunction1D(i,t):
    """
    Return an eigen function of Laplacian operator in one dimension
    i           - index int
    t           - time  float 
    """
    return cp.exp(2*PI*1j*i*t)


def matMulti(A,D):
    """
    Matrix multiplication A@D where A,D is a diagonal matrices, and D is a diagonal matrix
    """
    return A@D

# 
# def logDet(L):
#     """
#     # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
#     # L^dagger L is Hermitian
#     """
#     return (cp.linalg.slogdet(L)[1])
#     # return 0.5*(cp.linalg.slogdet(L.T.conj()@L)[1])
#     # return 0.5*cp.sum(cp.log(cp.linalg.eigvalsh(L.T.conj()@L)))
#     # return  cp.sum(cp.log(cp.absolute(cp.linalg.eigvals(L))))

# @nb.vectorize([nb.float64(nb.float64)],cache=CACHE,nopython=True)
@cupy_profile()
def kappaFun(ut):
    """
    kappa function as a function of u in time domain
    """
    # res = cp.zeros(ut.shape[0],dtype=cp.float64)
    # for i in nb.prange(ut.shape[0]):
    #     res[i] = math.exp(-ut[i])
    # return res
    # xp = cp.get_array_module(ut)
    return cp.exp(-ut)



# @nb.vectorize([nb.float64(nb.float64)],cache=CACHE,nopython=True)

# def kappa_pow_min_nu(ut):
#     # res = cp.zeros(ut.shape[0],dtype=cp.float64)
#     # for i in nb.prange(ut.shape[0]):
#     #     res[i] = math.exp(1.5*ut[i])
#     # return res
#     # return kappaFun(ut)**(-1.5)
#     xp = cp.get_array_module(ut)
#     return xp.exp(1.5*ut)


# # @nb.vectorize([nb.float64(nb.float64)],cache=CACHE,nopython=True)

# def kappa_pow_half(ut):
#     # res = cp.zeros(ut.shape[0],dtype=cp.float64)
#     # for i in nb.prange(ut.shape[0]):
#     #     res[i] = math.exp(-0.5*ut[i])
#     # return res
#     xp = cp.get_array_module(ut)
#     return xp.exp(-0.5*ut)
#     # return cp.sqrt(kappaFun(ut))


def norm2(u):
    """
    Compute euclidean squared norm 2 of a complex vector
    """
    # xp = cp.get_array_module(u)
    return cp.linalg.norm(u)**2

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


def sigmasLancos(n):
    """
    sigma Lancos coefficients for calculating inverse Fourier Transforms
    """
    k = cp.arange(1,n+1)
    return cp.sin(PI*(k/(n+1)))/(PI*(k/(n+1)))


def updateWelford(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1 
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# # retrieve the mean, variance and sample variance from an aggregate

def finalizeWelford(existingAggregate):
    (count, mean, M2) = existingAggregate
    # (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    (mean, variance) = (mean, M2/count) 
    # if count < 2:
        # return float('nan')
    # else:
    return (mean, variance)


def extend(uSymmetric,num):
    # xp = cp.get_array_module(uSymmetric) 
    n = (uSymmetric.shape[0]+1)//2
    if num> n:
        z = cp.zeros(2*num-1,dtype=cp.complex64)
        z[(num-1)-(n-1):(num-1)+n] = uSymmetric
        return z    
    else: 
        return uSymmetric


def symmetrize(w_half):
    # xp = cp.get_array_module(w_half)
    w = cp.concatenate((w_half[:0:-1].conj(),w_half)) #symmetrize
    return w


# def construct_w_Half_2D(n):
#     wHalf = construct_w_Half(2*n*n-2*n+1)
#     return fromUHalfToUHalf2D(wHalf,n)/SQRT2

  

def construct_w_Half_2D_ravelled(n):
    wHalf = construct_w_Half(2*n*n-2*n+1)
    return wHalf/SQRT2


# def fromUHalfToUHalf2D(uHalf,n):
#     xp = cp.get_array_module(uHalf)
#     uHalf = xp.concatenate((uHalf[n-1:0:-1].conj(),uHalf))
#     return uHalf.reshape((n,2*n-1)).T


# def fromUHalf2DToUHalf(uHalf2D,n):
#     uHalf = uHalf2D.T.ravel()
#     return uHalf[n-1:]


def construct_w_2D_ravelled(n):
    uHalf = construct_w_Half_2D_ravelled(n)
    return cp.concatenate((uHalf[:0:-1].conj(),uHalf))


def symmetrize_2D(uHalf2D):
    # xp = cp.get_array_module(uHalf2D)
    uHalfW=uHalf2D[:,1:]
    uHalf2Dc = uHalfW[::-1,:][:,::-1].conj()
    return cp.hstack((uHalf2Dc,uHalf2D))


# def from_u_2D_ravel_to_u_2D(u,n):
#     return u.reshape(2*n-1,2*n-1)


def from_u_2D_ravel_to_uHalf_2D(u,n):
    return u.reshape(2*n-1,2*n-1,order=im.ORDER)[:,n-1:]




def extend2D(uIn,num): 
    if uIn.shape[1] != uIn.shape[0]: #uHalfCase
        n = uIn.shape[1]
        if num> n:
            z = cp.zeros((2*num-1,num),dtype=cp.complex64)
            z[(num-1)-(n-1):(num-1)+n,:n] = uIn
            return z    
        else: 
            return uIn
    else:
        n = (uIn.shape[0]+1)//2
        if num> n:
            z = cp.zeros((2*num-1,2*num-1),dtype=cp.complex64)
            z[(num-1)-(n-1):(num-1)+n,(num-1)-(n-1):(num-1)+n] = uIn
            return z    
        else: 
            return uIn


def kappa_pow_min_nu(u):
    """
    for d=2, and alpha =2 nu = 1
    """
    return 1/kappaFun(u)


def kappa_pow_d_per_2(u):
    """
    for d=2, and d/2 = 1
    """
    return kappaFun(u)
    

def rfft2(z,n):
    # xp = cp.get_array_module(z)
    m = z.shape[0]
    zrfft = cp.fft.fftshift(cp.fft.rfft2(z,norm="ortho"),axes=0)
    return zrfft[m//2 -(n-1):m//2 +n,:n]
    

def irfft2(uHalf2D,num):
    """
    Fourier transform of one dimensional signal
    ut   = 1D signal 
    num  = Ut length - 1
    dt   = timestep
    (now using cp.fft.fft) in the implementation
    """
    # xp = cp.get_array_module(uHalf2D)
    uHalfExtended = extend2D(uHalf2D,num)

   
    uh = cp.fft.ifftshift(uHalfExtended,axes=0)
    uh = cp.fft.irfft2(uh,s=(2*num-1,2*num-1),norm="ortho")
    # return cp.fft.irfft2(uh,s=(num,num))
    return uh


def constructU(uHalf2D,index):
    n = uHalf2D.shape[1]
    
    res = extend2D(symmetrize_2D(uHalf2D),2*n-1)[index]
    return res

    

def constructMatexplicit(uHalf2D,fun,num,index):
    temp = fun(irfft2(uHalf2D,num))
    temp2 = rfft2(temp,uHalf2D.shape[1])
    return constructU(temp2,index)


def constructLexplicit(uHalf2D,D,num,sqrtBeta,index):
    Ku_pow_min_nu = constructMatexplicit(uHalf2D,kappa_pow_min_nu,num,index)
    Ku_pow_d_per_2 = constructMatexplicit(uHalf2D,kappa_pow_d_per_2,num,index)
    L = (matMulti(Ku_pow_min_nu,D) - Ku_pow_d_per_2)/sqrtBeta
    return L


def createUindex(n):
    innerlength = (2*n-1)
    length = innerlength**2
    shape = (length,length)
    iX = cp.zeros(shape,dtype=cp.int32)*(innerlength-1)
    iY = cp.zeros(shape,dtype=cp.int32)*(innerlength-1)
    for i in range(innerlength):
        for j in range(innerlength):
            # if cp.abs(j-i)<n:
            iX[i*innerlength:(i+1)*innerlength,j*innerlength:(j+1)*innerlength] = (j-i)+(innerlength-1)
            for k in range(innerlength):
                for l in range(innerlength):
                    iShift = i*innerlength
                    jShift = j*innerlength
                    iY[k+iShift,l+jShift] = (k-l)+(innerlength-1)
    
    return (iY,iX)


def eigenFunction2D(tx,ty,kx,ky):
    """
    Return an eigen function of Laplacian operator in one dimension
    i           - index int
    t           - time  float 
    Numpy FFT implementation using a constant of 2*PI only for fft2
    beware of this!!
    """
    return cp.exp(1j*(2*PI)*(kx*tx+ky*ty)) #<-- why the eigen function has to be in this form?


def constructH(tx,ty,ix,iy):
    """
    (iX,iY) are meshgrid, but ravelled
    (tx,ty) also ravelled meshgrid
    """
    # H = cp.empty((ix.shape[0],tx.shape[0]),dtype=cp.complex64)
    # for i in nb.prange(ix.shape[0]):
    #     H[i,:] = eigenFunction2D(tx,ty,ix[i],iy[i])
    H = cp.empty((tx.shape[0],ix.shape[0]),dtype=cp.complex64)
    for i in range(tx.shape[0]):
        H[i,:] = eigenFunction2D(tx[-i],ty[-i],ix,iy)
    return H


@cuda.jit
def matMultiParallel(A,B,C):
    """
    CUDA Kernel to do matrix multiplication where
    A, D are square matrices of same shape
    and D is a diagonal matrix
    """
    i, j = cuda.grid(2)
    if i<A.shape[0] and j<A.shape[1]:
        C[i,j] = A[i,j]*B[j,j]


def sigmasLancosTwo(n):
    """
    sigma Lancos coefficients for calculating inverse Fourier Transforms
    """
    temp = cp.zeros(2*n-1)
    for i in  cp.arange(2*n-1):
        k=i-(n-1)
        if k==0:
            temp[i] = 1
            continue
        else:
            temp[i] = cp.sin(PI*(k/n))/(PI*(k/n))

    return cp.outer(temp,temp)

"""
f is either h5py.File or h5py.group
this will save simulation object recursively
"""            
def _save_object(f,obj,end_here=False):
    for key,value in obj.__dict__.items():
        if isinstance(value,int) or isinstance(value,float) or isinstance(value,str) or isinstance(value,bool):
            f.create_dataset(key,data=value)
            continue
        elif isinstance(value,cp.core.core.ndarray):
            if key == 'H' or  key == 'I' or key =='H_t_H' or key == 'Dmatrix' or key == 'Imatrix' or key == 'ix' or key =='iy':#do not save again H and H_t_H
                continue
            else:
                if value.ndim >0:
                    f.create_dataset(key,data=cp.asnumpy(value),compression='gzip')
                else:
                    f.create_dataset(key,data=cp.asnumpy(value))
        elif isinstance(value,np.ndarray):
            if value.ndim >0:
                f.create_dataset(key,data=value,compression='gzip')
            else:
                f.create_dataset(key,data=value)
            continue
        else:
            if not end_here:
                
                typeStr = str(type(value))
                if isinstance(value,list):
                    for i in range(len(value)):
                        grp = f.create_group(key + ' {0}'.format(i))
                        _save_object(grp,value[i],end_here=True)
                elif ('pCN' in typeStr) or ('TwoDMeasurement' in typeStr) or ('FourierAnalysis_2D' in typeStr):
                    grp = f.create_group(key)
                    _save_object(grp,value,end_here=True)