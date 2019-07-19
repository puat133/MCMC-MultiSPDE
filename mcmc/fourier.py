import numpy as np
import numba as nb
import mcmc.util as util

spec = [
    ('fourier_basis_number', nb.int64),               
    ('fourier_extended_basis_number', nb.int64),               
    ('t_end', nb.float64),               
    ('t_start', nb.float64),               
    ('dt', nb.float64),
    ('t', nb.float64[:]),
    ('Dmatrix', nb.float64[:,:]),
    ('Imatrix', nb.float64[:,:]),
    ('cosFun', nb.float64[:,:]),          
    ('sinFun', nb.float64[:,:]),          
    ('eigenFun', nb.complex128[:,:]),
    ('prepared',nb.boolean)
]

@nb.jitclass(spec)
class FourierAnalysis:
    def __init__(self, fourier_basis_number,fourier_extended_basis_number,t_start = 0,t_end=1):
        self.fourier_basis_number = fourier_basis_number
        self.fourier_extended_basis_number = fourier_extended_basis_number
        self.t_end = t_end
        self.t_start = t_start
        
        self.Dmatrix = -(2*np.pi)**2*np.diag(np.arange(-(self.fourier_basis_number-1),self.fourier_basis_number)**2)
        self.Imatrix = np.eye(2*self.fourier_basis_number-1)
        self.prepare()

    
    def prepare(self):
        self.t = np.linspace(self.t_start,self.t_end,self.fourier_extended_basis_number)
        self.dt = self.t[1] - self.t[0]
        self.eigenFun = np.empty((self.fourier_basis_number,self.fourier_extended_basis_number),dtype=np.complex128)
        self.cosFun = np.empty((self.fourier_basis_number,self.fourier_extended_basis_number),dtype=np.float64)
        self.sinFun = np.empty((self.fourier_basis_number,self.fourier_extended_basis_number),dtype=np.float64)
        for i in range(self.fourier_basis_number):
            self.eigenFun[i,:] = util.eigenFunction1D(-i,self.t) 
            self.cosFun[i,:] = np.cos(2*np.pi*self.t*i)
            self.sinFun[i,:] = np.sin(2*np.pi*self.t*i)
        self.prepared = True

    
    def inverseFourierLimited(self,uHalf):
        y = np.zeros(self.sinFun.shape[1],dtype=np.float64)
        for i in range(1,len(uHalf)):
            for j in range(self.sinFun.shape[1]):
                y[j] += 2*(uHalf[i].real*self.cosFun[i,j] - uHalf[i].imag*self.sinFun[i,j])
            # y += 2*(u[i].real*cosFun[i,:] - u[i].imag*sinFun[i,:])
        
        for j in range(self.sinFun.shape[1]):
            y[j] += uHalf[0].real
        # y +=   u[0].real
        return y

    def fourierTransformHalf(self,ut):
        uHalf = np.zeros(self.eigenFun.shape[0],np.complex128)
        for i in range(self.eigenFun.shape[0]):
            # uHalf[i] = inner(ut,eigenFun[i,:])*dt
            uHalf[i] = util.inner(ut,self.eigenFun[i,:])
        return uHalf*self.dt

    def constructU(self,uHalf):
        """
        Construct Toeplitz Matrix
        """
        #using native scipy toeplitz function is not working in nopython mode
        # uFull = np.concatenate((uHalf,np.zeros(len(uHalf)-1)), axis=None)
        # U = sla.toeplitz(uFull).conj()

        
        # LU = len(uHalf)
        #np.zeros only supported with two argument, but somehow if I include dtype in np.zeros it
        #is not working, so this is the solution
        Ushape = (2*self.fourier_basis_number-1,2*self.fourier_basis_number-1)
        U = np.zeros(Ushape,dtype=np.complex128)
        # U = U.astype(complex)
        for i in nb.prange(2*self.fourier_basis_number-1):
            for j in nb.prange(2*self.fourier_basis_number-1):
                index = i-j #(j-i)
                if 0<= index <self.fourier_basis_number :
                    U[i,j] = uHalf[index]
                    continue
                if 0< -index < self.fourier_basis_number:
                    U[i,j] = uHalf[-index].conjugate()
                    # continue
        return U
    
    def constructMatexplicit(self,uHalf,fun):
        temp = fun(self.inverseFourierLimited(uHalf))
        temp2 = self.fourierTransformHalf(temp)
        return self.constructU(temp2)