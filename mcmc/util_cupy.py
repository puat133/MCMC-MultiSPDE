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
import gc
import mcmc.image_cupy as im
import h5py
from skimage.transform import radon
from cupy.prof import TimeRangeDecorator as cupy_profile
from numba import cuda
SQRT2 = cp.float32(1.41421356)
PI = cp.float32(cp.pi)
TPBn = 8#4*32
TPB = (TPBn,TPBn)
import mcmc.util as u_nb

from numba import cuda
from math import sin,cos,sqrt,pi
from cmath import exp
from mcmc.extra_linalg import solve_triangular
from cupy.linalg import qr
import cupyx as cpx
#@cupy_profile()
def construct_w_Half(n):
    wHalf = cp.random.randn(n,dtype=cp.float32)+1j*cp.random.randn(n,dtype=cp.float32)
    # wHalf[0] = wHalf[0].real*cp.sqrt(2)
    wHalf[0] = 2*wHalf[0].real
    # return wHalf/cp.sqrt(2)
    return wHalf/SQRT2

#@cupy_profile()
def inner(u,v):
    return cp.inner(u,v)


# @nb.vectorize([nb.complex128(nb.int64,nb.float64)],cache=CACHE,nopython=True)
#@cupy_profile()
def eigenFunction1D(i,t):
    """
    Return an eigen function of Laplacian operator in one dimension
    i           - index int
    t           - time  float 
    """
    return cp.exp(2*PI*1j*i*t)

#@cupy_profile()
def matMulti(A,D):
    """
    Matrix multiplication A@D where A is herimitian matrix, and D is a diagonal matrix
    """
    return A@D

def matMulti_sparse(A,D):
    """
    Matrix multiplication A@D where A is herimitian matrix, and D is a sparse diagonal matrix
    """
    C = cp.zeros_like(A,dtype=A.dtype)
    diag_D = D.diagonal()
    bpg=((A.shape[0]+TPBn-1)//TPBn,(A.shape[1]+TPBn-1)//TPBn)
    _matMulti_sparse[bpg,TPB](A,diag_D,C)
    # for i in range(A.shape[0]):
    #     for j in range(A.shape[1]):
    #         C[i,j] = A[i,j]*diag_D[j]
    return C

@cuda.jit()
def _matMulti_sparse(A,diag_D,C):
    i,j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        C[i,j] = A[i,j]*diag_D[j]

def slogdet(L):
    """
    # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
    # L^dagger L is Hermitian
    # cupy slogdet seems cannot handle complex matrix
    """
    cp.linalg.slogdet(L)
    
    temp = L.conj().T@L
    temp = 0.5*(temp+temp.conj().T)
    res =  0.5*cp.sum(cp.log(cp.linalg.eigvalsh(temp)))
    del temp
    cp._default_memory_pool.free_all_blocks()
    return res,0


# @nb.vectorize([nb.float64(nb.float64)],cache=CACHE,nopython=True)
#@cupy_profile()
def kappaFun(ut):
    """
    kappa function as a function of u in time domain
    """
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

#@cupy_profile()
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

#@cupy_profile()
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
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    # (mean, variance) = (mean, M2/count) 
    # if count < 2:
        # return float('nan')
    # else:
    return (mean, variance,sampleVariance)

#@cupy_profile()
def extend(uSymmetric,num):
    # xp = cp.get_array_module(uSymmetric) 
    n = (uSymmetric.shape[0]+1)//2
    if num> n:
        z = cp.zeros(2*num-1,dtype=cp.complex64)
        z[(num-1)-(n-1):(num-1)+n] = uSymmetric
        return z    
    else: 
        return uSymmetric

#@cupy_profile()
def symmetrize(w_half):
    # xp = cp.get_array_module(w_half)
    w = cp.concatenate((w_half[:0:-1].conj(),w_half)) #symmetrize
    return w


# def construct_w_Half_2D(n):
#     wHalf = construct_w_Half(2*n*n-2*n+1)
#     return fromUHalfToUHalf2D(wHalf,n)/SQRT2

  
#@cupy_profile()
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

#@cupy_profile()
def construct_w_2D_ravelled(n):
    uHalf = construct_w_Half_2D_ravelled(n)
    return cp.concatenate((uHalf[:0:-1].conj(),uHalf))

#@cupy_profile()
def symmetrize_2D(uHalf2D):
    # xp = cp.get_array_module(uHalf2D)
    uHalfW=uHalf2D[:,1:]
    uHalf2Dc = uHalfW[::-1,:][:,::-1].conj()
    return cp.hstack((uHalf2Dc,uHalf2D))


# def from_u_2D_ravel_to_u_2D(u,n):
#     return u.reshape(2*n-1,2*n-1)

#@cupy_profile()
def from_u_2D_ravel_to_uHalf_2D(u,n):
    return u.reshape(2*n-1,2*n-1,order=im.ORDER)[:,n-1:]



#@cupy_profile()
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

#@cupy_profile()
def kappa_pow_min_nu(u):
    """
    for d=2, and alpha =2 nu = 1
    """
    return cp.exp(u)#1/kappaFun(u)

#@cupy_profile()
def kappa_pow_d_per_2(u):
    """
    for d=2, and d/2 = 1
    """
    return cp.exp(-u)#kappaFun(u)
    
#@cupy_profile()
def rfft2(z,n):
    # xp = cp.get_array_module(z)
    m = z.shape[0]
    zrfft = cp.fft.fftshift(cp.fft.rfft2(z,norm="ortho"),axes=0)
    return zrfft[m//2 -(n-1):m//2 +n,:n]
    
#@cupy_profile()
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
    return uh

#@cupy_profile()
def constructU(uHalf2D,index):
    n = uHalf2D.shape[1]
    
    res = extend2D(symmetrize_2D(uHalf2D),2*n-1)[index]
    return res

def constructU_cuda(uHalf2D):
    n = uHalf2D.shape[1]
    innerlength = 2*n-1
    length = innerlength**2
    U = cp.zeros((length,length),dtype=cp.complex64)
    bpg = ((U.shape[0]+TPBn-1)//TPBn,(U.shape[1]+TPBn-1)//TPBn)
    _construct_U[bpg,TPB](symmetrize_2D(uHalf2D),n,innerlength,U)
    return U

def constructU_from_uSym2D_cuda(uSym2D):
    innerlength = uSym2D.shape[0]
    n = (innerlength+1)//2
    length = innerlength**2
    U = cp.zeros((length,length),dtype=cp.complex64)
    bpg = ((U.shape[0]+TPBn-1)//TPBn,(U.shape[1]+TPBn-1)//TPBn)
    _construct_U[bpg,TPB](uSym2D,n,innerlength,U)
    return U

#@cupy_profile()
def constructMatexplicit(uHalf2D,fun,num,index):
    temp = fun(irfft2(uHalf2D,num))
    temp2 = rfft2(temp,uHalf2D.shape[1])
    return constructU(temp2,index)

def constructMatexplicit_cuda(uHalf2D,fun,num):
    temp = fun(irfft2(uHalf2D,num))
    temp2 = rfft2(temp,uHalf2D.shape[1])
    return constructU_cuda(temp2)

#@cupy_profile()
def constructLexplicit(uHalf2D,D,num,sqrtBeta,index):
    Ku_pow_min_nu = constructMatexplicit(uHalf2D,kappa_pow_min_nu,num,index)
    Ku_pow_d_per_2 = constructMatexplicit(uHalf2D,kappa_pow_d_per_2,num,index)
    L = (matMulti(Ku_pow_min_nu,D) - Ku_pow_d_per_2)/sqrtBeta
    return L

#@cupy_profile()
# def createUindex(n):
#     innerlength = (2*n-1)
#     length = innerlength**2
#     shape = (length,length)
#     iX = cp.zeros(shape,dtype=cp.int32)#*(innerlength-1)
#     iY = cp.zeros(shape,dtype=cp.int32)#*(innerlength-1)
#     for i in range(innerlength):
#         for j in range(innerlength):
#             # if cp.abs(j-i)<n:
#             # iX[i*innerlength:(i+1)*innerlength,j*innerlength:(j+1)*innerlength] = (j-i)+(innerlength-1)
#             iX[i*innerlength:(i+1)*innerlength,j*innerlength:(j+1)*innerlength] = (i-j)+(innerlength-1)
#             for k in range(innerlength):
#                 for l in range(innerlength):
#                     iShift = i*innerlength
#                     jShift = j*innerlength
#                     iY[k+iShift,l+jShift] = (l-k)+(innerlength-1)
#                     # iY[k+iShift,l+jShift] = (k-l)+(innerlength-1)
    
#     return (iY,iX)
def createUindex(n):
    innerlength = (2*n-1)
    length = innerlength**2
    shape = (length,length)
    iX = cp.zeros(shape,dtype=cp.int8)#*(innerlength-1)
    iY = cp.zeros(shape,dtype=cp.int8)#*(innerlength-1)
    for i in range(innerlength):
        for j in range(innerlength):
            iShift = i*innerlength
            jShift = j*innerlength
            # iY[i*innerlength:(i+1)*innerlength,j*innerlength:(j+1)*innerlength] = (i-j)+(innerlength-1)#innerlength-1 adalah shiftnya jadi innerlength-1 itu nol
            iY[iShift:iShift+innerlength,jShifth:jShift+innerlength] = (i-j)+(innerlength-1)#innerlength-1 adalah shiftnya jadi innerlength-1 itu nol
            for k in range(innerlength):
                for l in range(innerlength):
                    # iShift = i*innerlength
                    # jShift = j*innerlength
                    iX[k+iShift,l+jShift] = (k-l)+(innerlength-1)
    
    return (iY,iX)#because iY is row index, iX is column index
    # return (iX,iY)#because iY is row index, iX is column index
#@cupy_profile()
def eigenFunction2D(tx,ty,kx,ky):
    """
    Return an eigen function of Laplacian operator in one dimension
    i           - index int
    t           - time  float 
    Numpy FFT implementation using a constant of 2*PI only for fft2
    beware of this!!
    """
    return cp.exp(1j*(2*PI)*(kx*tx+ky*ty)) #<-- why the eigen function has to be in this form?

#@cupy_profile()
def constructH(tx,ty,ix,iy):
    """
    (iX,iY) are meshgrid, but ravelled
    (tx,ty) also ravelled meshgrid
    """
    H = cp.empty((tx.shape[0],ix.shape[0]),dtype=cp.complex64)
    for i in range(tx.shape[0]):
        # H[i,:] = eigenFunction2D(tx[-i],ty[-i],ix,iy)
        H[i,:] = eigenFunction2D(tx[i],ty[i],ix,iy)
    return H

@cuda.jit()
def _construct_H(tx,ty,ix,iy,H):
    """
    (iX,iY) are meshgrid, but ravelled
    (tx,ty) also ravelled meshgrid
    """
    
    
    # H[i,j] = eigenFunction2D(tx[-i],ty[-i],ix,iy)
    i,j = cuda.grid(2)
    H[i,j] = exp(1j*2*pi*(ix[j]*tx[i]+iy[j]*ty[i]))


@cuda.jit()
def _construct_U(uSym2D,N,innerlength,U):
    '''CUDA kernel for constructing U matrix from uSym2D
    '''
    m,n = cuda.grid(2)
    if m < U.shape[0] and n < U.shape[1]:
        i = m//innerlength
        j = n//innerlength
        k = m%innerlength
        l = n%innerlength
        delta_IJ = (i-j)
        delta_KL = (k-l)

        if -(N-1)<=delta_IJ < N:
            if -(N-1)<=delta_KL < N:
                U[m,n] = uSym2D[delta_KL+(N-1),delta_IJ+(N-1)] #<-- this is correct already!!
                # U[m,n] = uSym2D[delta_IJ+(N-1),delta_KL+(N-1)] #<-- this is correct already!!
    
          
@cuda.jit()
def _calculate_H_Tomography(r,theta,ix,iy,H):
    """
    (iX,iY) are meshgrid for Fourier Index
    (tx,ty) also ravelled meshgrid for original location grid (0 to 1)
    CUDA kernel function, with cuda jit
    """
    m,n = cuda.grid(2)
    if m < H.shape[0] and n < H.shape[1]:
        
        
        # theta_m = theta[m]
        theta_m = theta[-(m+1)]+0.5*pi
        sTheta = sin(theta_m)
        cTheta = cos(theta_m)
        r_m = r[m]
            
        kx = ix[n]
        ky = iy[n]
        k_tilde_u = kx*cTheta+ky*sTheta
        k_tilde_v = -kx*sTheta+ky*cTheta
        l = sqrt(0.25-r_m*r_m)
        if k_tilde_v*k_tilde_v > 0.0:
            H[m,n] =  exp(1j*pi*((kx+ky)-2*k_tilde_v*r_m))*(sin(2*pi*k_tilde_u*l))/(pi*k_tilde_u)
        else:
            H[m,n] =  exp(1j*pi*((kx+ky)-2*k_tilde_v*r_m))*(2*l)

# # ASELI untuk 180 derajat sesuai buku NETTER
@cuda.jit()
def _calculate_H_Tomography_DEANS(r,theta,ix,iy,H):
    """
    (iX,iY) are meshgrid for Fourier Index
    (tx,ty) also ravelled meshgrid for original location grid (0 to 1)
    CUDA kernel function, with cuda jit
    """
    m,n = cuda.grid(2)
    if m < H.shape[0] and n < H.shape[1]:
        
        
        theta_m = theta[m]
        # theta_m = theta[-(m+1)]
        sTheta = sin(theta_m)
        cTheta = cos(theta_m)
        r_m = r[m]
            
        kx = ix[n]
        ky = iy[n]
        k_tilde_u = kx*cTheta+ky*sTheta
        k_tilde_v = -kx*sTheta+ky*cTheta
        l = sqrt(0.25-r_m*r_m)
        if k_tilde_v*k_tilde_v > 0.0:
            H[m,n] =  exp(1j*pi*((kx+ky)+2*k_tilde_u*r_m))*(sin(2*pi*k_tilde_v*l))/(pi*k_tilde_v)
        else:
            H[m,n] =  exp(1j*pi*((kx+ky)+2*k_tilde_u*r_m))*(2*l)

def calculate_H_Tomography_skimage(theta,ix,iy,H,image_size):
    temp = np.linspace(0.,1.,num=image_size,endpoint=True)
    # extended_base_number = (image_size+1)//2
    tx,ty = np.meshgrid(temp,temp)
    mask = ((tx-0.5)**2+(ty-0.5)**2 > 0.25)
    for k,ix_now,iy_now in zip(range(ix.shape[0]),ix,iy):
        phi_image = np.exp(1j*2*cp.pi*(ix_now*tx+iy_now*ty))
        phi_image[mask] = 0
        rad = radon(phi_image.real,theta=theta,circle=True)+1j*radon(phi_image.imag,theta=theta,circle=True)
        H[:,k] = rad.ravel()
    H /= image_size

def norm2(x):
    return cp.sum(x*x.conj()).real

# @nb.jit(target='cuda')
# def calculate_H_Tomography(r,theta,ix,iy,H):
#     for m in range(H.shape[0]):
#         for n in range(H.shape[1]):            
#             sTheta = sin(theta[m])
#             cTheta = cos(theta[m])
#             r_m = r[m] #+ (0.5*(cTheta+sTheta))
#             kx = ix[n]
#             ky = iy[n]
#             k_tilde_u = kx*cTheta+ky*sTheta
#             k_tilde_v = -kx*sTheta+ky*cTheta
#             l = sqrt(0.25-r_m*r_m)
#             if k_tilde_v != 0:
#                 H[m,n] = exp(1j*PI*(kx+ky))*exp(1j*2*PI*k_tilde_u*r_m)*(sin(2*PI*k_tilde_v*l))/(PI*k_tilde_v)
#                 # H[m,n] = exp(1j*2*pi*k_tilde_u*r[m])*(sin(2*pi*k_tilde_v*l))/(pi*k_tilde_v)
#             else:
#                 H[m,n] = exp(1j*PI*(kx+ky))*exp(1j*2*PI*k_tilde_u*r_m)*(2*l) #<-- Suprisingly this does not compile,For teslaV100!!! 
    
#@cupy_profile()
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
    excluded_matrix = ['H','Ht','I','In','H_t_H','Imatrix','Dmatrix','ix','iy','y','ybar']
    for key,value in obj.__dict__.items():
        if isinstance(value,int) or isinstance(value,float) or isinstance(value,str) or isinstance(value,bool):
            f.create_dataset(key,data=value)
            continue
        elif isinstance(value,cp.core.core.ndarray):
            if key in excluded_matrix:
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
                elif ('pCN' in typeStr) or ('TwoDMeasurement' in typeStr) or ('FourierAnalysis_2D' in typeStr) or ('Sinogram' in typeStr):
                    grp = f.create_group(key)
                    _save_object(grp,value,end_here=True)


"""
There is a bug in cupy lstsq, so that it gives solution to the complex problem is 
the complex conjugate of the actual solution (obtained by numpy for comparison)
"""
def lstsq(a, b, rcond=1e-15,in_cpu=False):
    # x, resids, rank, s = cp.linalg.lstsq(a,b,rcond)
    # return x.conj(),resids,rank,s

    

    if not in_cpu:
        #do traditional way using QR factorization
        # #1. compute QR factorization of a
        q,r = qr(a) #

        # # #2. multiply q.conj().T x b
        d = q.conj().T@b

        # # #solve rx=d <-- this should be solved using solve_triangular
        x = solve_triangular(r,d)# 
        return x,0,0,0
    else:
        x,m,n,o = np.linalg.lstsq(cp.asnumpy(a),cp.asnumpy(b),rcond=rcond)
        return cp.asarray(x),m,n,o


    
"""
delete variable from memory
"""
def remove_var(x):
    del x
    cp._default_memory_pool.free_all_blocks()

def shift_2D(uSym,k):
    result = cp.zeros_like(uSym)
    #only do shifting if k is one dimensional
    if k.ndim == 1:
        if k[0]> 0:
            if k[1]> 0:
                result[k[0]:,k[1]:] = uSym[:-k[0],:-k[1]]
            elif k[1]< 0:
                result[k[0]:,:k[1]] = uSym[:-k[0],-k[1]:]
            else:
                result[k[0]:,:] = uSym[:-k[0],:]

        elif k[0]< 0:
            if k[1]> 0:
                result[:k[0],k[1]:] = uSym[-k[0]:,:-k[1]]
            elif k[1]< 0:
                result[:k[0],:k[1]] = uSym[-k[0]:,-k[1]:]
            else:
                result[:k[0],:] = uSym[-k[0]:,:]
        else:
            if k[1]> 0:
                result[:,k[1]:] = uSym[:,:-k[1]]
            elif k[1]< 0:
                result[:,:k[1]] = uSym[:,-k[1]:]
            else:
                result = uSym
    return result


def expm(A,delta=1e-10):
    j = max(0,cp.int(1+cp.log2(cp.linalg.norm(A,cp.inf))))
    A = A/(2**j)
    q = u_nb.expm_eps_less_than(delta)
    n = A.shape[0]
    I = cp.eye(n)
    D = I
    N = I
    X = I
    c = 1
    sign = 1
    for k in range(1,q+1):
        c = c*(q-k+1)/((2*q - k+ 1)*k)
        X = A@X
        N = N + c*X
        sign = -1*sign
        D = D + sign*c*X
    
    F = cp.linalg.solve(D,N)
    for _ in range(j):
        F = F@F
    
    return F