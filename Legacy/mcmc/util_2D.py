# import math
import h5py
import scipy.io as sio
import numpy as np
import scipy.linalg as sla
import mcmc.util as util
import numba as nb
import time
import math
FASTMATH=True
PARALLEL = True
CACHE=True
# from numba import complex64, complex128, float32, float64, int32, jit, njit, prange
SQRT2 = np.sqrt(2)
njitSerial = nb.njit(fastmath=FASTMATH,cache=CACHE)
jitSerial = nb.jit(fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
jitParallel = nb.jit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)

@njitParallel
def construct_w_Half_2D(n):
    wHalf = util.construct_w_Half(2*n*n-2*n+1)
    return fromUHalfToUHalf2D(wHalf,n)/SQRT2

  
@njitParallel
def construct_w_Half_2D_ravelled(n):
    wHalf = util.construct_w_Half(2*n*n-2*n+1)
    return wHalf/SQRT2

@njitParallel
def fromUHalfToUHalf2D(uHalf,n):
    uHalf = np.concatenate((uHalf[n-1:0:-1].conj(),uHalf))
    return uHalf.reshape((n,2*n-1)).T

@njitParallel
def fromUHalf2DToUHalf(uHalf2D,n):
    uHalf = uHalf2D.T.ravel()
    return uHalf[n-1:]

@njitParallel
def construct_w_2D_ravelled(n):
    uHalf = construct_w_Half_2D_ravelled(n)
    return np.concatenate((uHalf[:0:-1].conj(),uHalf))

@njitParallel
def symmetrize_2D(uHalf2D):
    uHalfW=uHalf2D[:,1:]
    uHalf2Dc = uHalfW[::-1,:][:,::-1].conj()
    return np.hstack((uHalf2Dc,uHalf2D))

@njitParallel
def from_u_2D_ravel_to_u_2D(u,n):
    return u.reshape(2*n-1,2*n-1)

@njitParallel
def from_u_2D_ravel_to_uHalf_2D(u,n):
    return u.reshape(2*n-1,2*n-1)[:,n-1:]



@njitParallel
def extend2D(uIn,num): 
    if uIn.shape[1] != uIn.shape[0]: #uHalfCase
        n = uIn.shape[1]
        if num> n:
            z = np.zeros((2*num-1,num),dtype=np.complex128)
            z[(num-1)-(n-1):(num-1)+n,:n] = uIn
            return z    
        else: 
            return uIn
    else:
        n = (uIn.shape[0]+1)//2
        if num> n:
            z = np.zeros((2*num-1,2*num-1),dtype=np.complex128)
            z[(num-1)-(n-1):(num-1)+n,(num-1)-(n-1):(num-1)+n] = uIn
            return z    
        else: 
            return uIn

@njitParallel
def kappa_pow_min_nu(u):
    """
    for d=2, and alpha =2 nu = 1
    """
    return 1/util.kappaFun(u)

@njitParallel
def kappa_pow_d_per_2(u):
    """
    for d=2, and d/2 = 1
    """
    return util.kappaFun(u)
    
@njitParallel
def rfft2(z,n):
    m = z.shape[0]
    with nb.objmode(zrfft='complex128[:,:]'):
        zrfft = np.fft.fftshift(np.fft.rfft2(z,norm="ortho"),axes=0)
    return zrfft[m//2 -(n-1):m//2 +n,:n]
    
@njitParallel
def irfft2(uHalf2D,num):
    """
    Fourier transform of one dimensional signal
    ut   = 1D signal 
    num  = Ut length - 1
    dt   = timestep
    (now using cp.fft.fft) in the implementation
    """
    uHalfExtended = extend2D(uHalf2D,num)

    with nb.objmode(uh='float64[:,:]'):
        # uh = np.fft.ifftshift(uHalfExtended,axes=0)
        uh = np.fft.ifftshift(uHalfExtended,axes=0)
        uh = np.fft.irfft2(uh,s=(2*num-1,2*num-1),norm="ortho")
    # return np.fft.irfft2(uh,s=(num,num))
    return uh

@njitParallel
def constructU(uHalf2D,index):
    n = uHalf2D.shape[1]
    with nb.objmode(res='complex128[:,:]'):
        res = extend2D(symmetrize_2D(uHalf2D),2*n-1)[index]
    return res

    
@njitParallel
def constructMatexplicit(uHalf2D,fun,num,index):
    temp = fun(irfft2(uHalf2D,num))
    temp2 = rfft2(temp,uHalf2D.shape[1])
    return constructU(temp2,index)

@njitParallel
def constructLexplicit(uHalf2D,D,num,sqrtBeta,index):
    Ku_pow_min_nu = constructMatexplicit(uHalf2D,kappa_pow_min_nu,num,index)
    Ku_pow_d_per_2 = constructMatexplicit(uHalf2D,kappa_pow_d_per_2,num,index)
    L = (util.matMulti(Ku_pow_min_nu,D) - Ku_pow_d_per_2)/sqrtBeta
    return L

@njitParallel
def createUindex(n):
    innerlength = (2*n-1)
    length = innerlength**2
    shape = (length,length)
    iX = np.zeros(shape,dtype=np.int64)*(innerlength-1)
    iY = np.zeros(shape,dtype=np.int64)*(innerlength-1)
    for i in range(innerlength):
        for j in range(innerlength):
            # if np.abs(j-i)<n:
            iX[i*innerlength:(i+1)*innerlength,j*innerlength:(j+1)*innerlength] = (j-i)+(innerlength-1)
            for k in range(innerlength):
                for l in range(innerlength):
                    iShift = i*innerlength
                    jShift = j*innerlength
                    iY[k+iShift,l+jShift] = (k-l)+(innerlength-1)
    
    return (iY,iX)

@njitParallel
def eigenFunction2D(tx,ty,kx,ky):
    """
    Return an eigen function of Laplacian operator in one dimension
    i           - index int
    t           - time  float 
    Numpy FFT implementation using a constant of 2*PI only for fft2
    beware of this!!
    """
    return np.exp(1j*(2*np.pi)*(kx*tx+ky*ty)) #<-- why the eigen function has to be in this form?

@njitParallel
def constructH(tx,ty,ix,iy):
    """
    (iX,iY) are meshgrid, but ravelled
    (tx,ty) also ravelled meshgrid
    """
    # H = np.empty((ix.shape[0],tx.shape[0]),dtype=np.complex128)
    # for i in nb.prange(ix.shape[0]):
    #     H[i,:] = eigenFunction2D(tx,ty,ix[i],iy[i])
    H = np.empty((tx.shape[0],ix.shape[0]),dtype=np.complex128)
    for i in nb.prange(tx.shape[0]):
        H[i,:] = eigenFunction2D(tx[i],ty[i],ix,iy)
    return H

@njitParallel
def createSampleMeasurement(tx,ty,stdev):
    """
    Create one dimensional Sample measurement
    (tx,ty) a ravelled meshgrid
    """
    xShift = 0.5
    yShift = 0.5
    var    = 1/16
    vt = np.zeros(tx.shape[0])
    for i in range(tx.shape[0]):
        # if (0.2+0.1*stdev*np.random.randn()<tx[i]<0.8+0.1*stdev*np.random.randn()) and (-0.8+0.1*stdev*np.random.randn()<ty[i]<0.8+0.1*stdev*np.random.randn()):
        if (0.6<tx[i]<0.8) and (0.2<ty[i]<0.8):
            vt[i]=2
            continue
        else:
            logExp = -((tx[i]-xShift)**2+(ty[i]-yShift)**2)/(var)
            vt[i] = 2*np.exp(logExp)-1
        
            
    e = stdev*np.random.randn(vt.shape[0])
    y = vt+e
    return (y,vt)