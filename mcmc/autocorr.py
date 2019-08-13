import numba as nb
import numpy as np
FASTMATH=True
PARALLEL = True
CACHE=True
njitSerial = nb.njit(fastmath=FASTMATH,cache=CACHE)
jitSerial = nb.jit(fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
jitParallel = nb.jit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)

#def autocorr2(u,lags):
#    '''manualy compute, non partial'''
#
#    mean=np.mean(u)
#    var=np.var(u)
#    xp=u-mean
#    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
#
#    return np.array(corr)

@jitParallel
def autocorr(uHalfHistory):
    length = uHalfHistory.shape[0]
    nFourier = uHalfHistory.shape[1]
    uHalfMean = np.mean(uHalfHistory,0)
    uHalfNormalized = uHalfHistory - uHalfMean
    uHalfVar = np.var(uHalfHistory,0)
#    lags = np.arange(length)
    xcorr = np.zeros((nFourier,length))
    for i in nb.prange(nFourier):
        for j in nb.prange(length):
            if j==0:
                xcorr[i,j]  = 1
            else:
                xcorr[i,j] = np.sum(uHalfNormalized[j:,i]*uHalfNormalized[:-j,i].conj()).real/(length*uHalfVar[i])
        
    return xcorr