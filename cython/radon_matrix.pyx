#adapted from https://github.com/suurj/tomo/matrix.pyx

import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sp
from cython.parallel import prange
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import math
from libc.math cimport sqrt,fabs,exp, cos, tan,sin,M_SQRT2,M_PI,abs
from libcpp.list cimport list as cpplist
from libcpp.deque cimport deque
from libcpp.vector cimport vector
import time
from cython.parallel import threadid as thid
from libc.stdlib cimport malloc 

# Construct a discrete Radon transform matrix.
# The method is from Peter Thoft's PhD thesis The Radon Transform - Theory and Implementation: https://orbit.dtu.dk/files/5529668/Binder1.pdf
# The result is a matrix of ceil(sqrt(2)xN)xT x NxN. Four points are used in first order pixel oriented interpolation within each pixel's neighbourhood.
ctypedef vector[int]* diptr
ctypedef vector[double]* dfptr
cdef double MP2 = M_PI/2.0
cdef double MP4 = M_PI/4.0
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True) 
def radonmatrix(size,theta,Nthreads=4):
  
    cdef int Nth = Nthreads
    cdef int i
    
    mrows = new vector[diptr](Nth)
    mcols = new vector[diptr](Nth)
    mdata = new vector[dfptr](Nth)
    row = mrows[0]
    col = mcols[0]
    data = mdata[0]
    
    for i in range(Nth):
        row[i] = new vector[int]()
        col[i] = new vector[int]()
        data[i] = new vector[double]()
    
    cdef int T = theta.shape[0]
    cdef int N = size
    cdef double dx = 1
    cdef double dy = 1 
    cdef int M = N
    cdef int R = math.ceil(M_SQRT2*N)
    cdef double xmin = -(N-1.0)/2.0
    cdef double dp = 2.0*M_SQRT2*abs(xmin)/(R-1.0)
    cdef double pmin = -(R-1.0)/2.0*dp
    cdef double tmin = theta[0]
    cdef double ymin = xmin
    cdef double pmax = (R-1.0)/2.0*dp
    cdef double tmax = theta[theta.shape[0]-1]
    cdef double tt,ray,dt
    cdef int t,m,n,r,th
    if (T == 1):
        dt = 0
    else:
        dt = (theta[1]-theta[0])
        
    # One might comment out the first ray row or alternatively 
    # comment out the followinf five and comment the first one. Averaging four values
    # might lead to more realistic sinogram with large dimensions and angles and it would make the operator denser.
    # However, the averaging might make the sinogram perhaps worse with sparse angles (at least the angle averaging should be reconsidered one should use rhoo
    # averaging only).
    # See Peter Thoft's PhD thesis above
    # (First order pixel oriented interpolation).
    start = time.time() 
    with nogil:                     
        for r in prange (0,R,num_threads=Nth):
            for t in range (0,T):               
                tt = -(tmin + t*dt)
                for n in range (0,N):
                    for m in range( 0,M):
                        ray = dx/2.0 * gs(2.0*(pmin+r*dp -(xmin+m*dx)*cos(tt)-(ymin+n*dy)*sin(tt) )/dx,tt)
                        #ray = dx/2.0 * gs(2.0*(pmin+r*dp+dp/4.0 -(xmin+m*dx)*cos(tt+dt/4.0)-(ymin+n*dy)*sin(tt+dt/4.0) )/dx,tt+dt/4.0)
                        #ray = ray + dx/2.0 * gs(2.0*(pmin+r*dp+dp/4.0 -(xmin+m*dx)*cos(tt-dt/4.0)-(ymin+n*dy)*sin(tt-dt/4.0) )/dx,tt-dt/4.0)
                        #ray = ray + dx/2.0 * gs(2.0*(pmin+r*dp-dp/4.0 -(xmin+m*dx)*cos(tt+dt/4.0)-(ymin+n*dy)*sin(tt+dt/4.0) )/dx,tt+dt/4.0)
                        #ray = ray + dx/2.0 * gs(2.0*(pmin+r*dp-dp/4.0 -(xmin+m*dx)*cos(tt-dt/4.0)-(ymin+n*dy)*sin(tt-dt/4.0) )/dx,tt-dt/4.0)
                        #ray = ray/4.0
                        if(ray > 0.0):
                            th = thid()
                            row[th].push_back(r*T+t)
                            col[th].push_back(n*M+m)
                            data[th].push_back(ray) 

            
    print("Radon matrix was constructed in " + str(time.time()-start) + " seconds")     
     
    cdef int Nel = 0
    for i in range(0,Nth):
        Nel = Nel + row[i].size()

    coo_row  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_rowv  = coo_row
    coo_col  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_colv  = coo_col
    coo_data = np.zeros((Nel,))
    cdef double [:] coo_datav  = coo_data

    i = 0
    with nogil:
        for j in range(0,Nth):
            for k in range(row[j].size()):
                coo_rowv[i] = row[j].back()
                row[j].pop_back()

                coo_colv[i] = col[j].back()
                col[j].pop_back()

                coo_datav[i] = data[j].back()
                data[j].pop_back()
                i = i + 1


    
    radonM = coo_matrix((coo_data, (coo_row, coo_col)), shape=(R*T,N*N))
    radonM = csc_matrix(radonM)
    
    return radonM 

# Function which is called, when Radon operator matrix is constucted.
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True) 
cdef inline double gs(double p,double t) nogil:
    cdef double x1m
    cdef double x1
    cdef double y1
    if (p<0):
        p = -p
   
    #t = (t%(M_PI/2.0))
    t = (t % (MP2) + (MP2)) % MP2
    if(t >= MP4):
        t = MP2-t
 
    
    if( p > M_SQRT2):
        return 0.0
        #return a
    else:
        x1m = p/cos(t) + tan(t)
        x1 = p/cos(t) - tan(t)
        y1 = p/sin(t) - 1.0/tan(t)
        
   
        if (x1 < 1.0 and x1m  < 1.0):
            return sqrt(4.0+(x1-x1m)*(x1-x1m))
            #return a
            
        elif (x1 < 1.0 and x1m  > 1.0):
            return sqrt((1.0-x1)*(1.0-x1) + (1.0-y1)*(1.0-y1))
            #return a
            
        elif (x1 >=1.0):
            return 0.0
            #return a
            
        else:
            return -9.0