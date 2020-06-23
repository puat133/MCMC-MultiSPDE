import numpy as np
cimport numpy as np #cimport is used to import special compile-time information about numpy module
cimport cython
from cython.parallel import prange

cmplTYPE = np.complex128
ctypedef np.complex128_t cmplTYPE_t
fltTYPE = np.float64
ctypedef np.float64_t fltTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def autocorrelation(np.ndarray u_half_chain):
    cdef:
        int length = u_half_chain.shape[0]
        int basis_number = u_half_chain.shape[1]
        np.ndarray xcorr = np.zeros([basis_number,length],dtype=fltTYPE)
        int i,j,k,count
        float temp
        double complex a,b
    

    
    cdef np.ndarray u_half_mean = np.zeros([basis_number],dtype=cmplTYPE)
    cdef np.ndarray u_half_normalized = np.zeros([basis_number,length],dtype=cmplTYPE)
    cdef np.ndarray u_half_var = np.zeros([basis_number],dtype=fltTYPE)
    u_half_mean = np.mean(u_half_chain,axis=0)
    u_half_var = np.var(u_half_chain,axis=0)
    u_half_normalized = u_half_chain-u_half_mean

    for i in range(basis_number):
        for j in range(length):
            if j==0:
                xcorr[i,j] = 1.
            else:
                for k in range(j,length):
                    a = u_half_normalized[k,i]
                    b = u_half_normalized[k-j,i]
                    temp = (a.real+b.real)+(a.imag+b.imag)
                xcorr[i,j] = temp/(length*u_half_var[i])
    return xcorr

    
