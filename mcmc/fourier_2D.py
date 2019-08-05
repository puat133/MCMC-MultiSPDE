import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.util_2D as u2

spec = [
    ('fourier_basis_number', nb.int64),               
    ('fourier_extended_basis_number', nb.int64),               
    ('t_end', nb.float64),               
    ('t_start', nb.float64),               
    ('dt', nb.float64),
    ('t', nb.float64[:]),
    ('Dmatrix', nb.float64[:,:]),
    ('Imatrix', nb.float64[:,:]),
    ('Index',nb.int64[:,:])
    # ('cosFun', nb.float64[:,:]),          
    # ('sinFun', nb.float64[:,:]),          
    # ('eigenFun', nb.complex128[:,:]),
    # ('prepared',nb.boolean)
]

@nb.jitclass(spec)
class FourierAnalysis:
    def __init__(self, fourier_basis_number,fourier_extended_basis_number,t_start = 0,t_end=1):
        self.fourier_basis_number = fourier_basis_number
        self.fourier_extended_basis_number = fourier_extended_basis_number
        self.t_end = t_end
        self.t_start = t_start
        d_diag = np.zeros((2*self.fourier_basis_number-1)**2)
        for i in range(2*self.fourier_basis_number-1):
            for j in range(2*self.fourier_basis_number-1):
                d_diag[i*10+j] = (i**2+j**2)
        self.Dmatrix = -(2*np.pi)**2*np.diag(d_diag)
        self.Imatrix = np.eye((2*self.fourier_basis_number-1)**2)
        self.Index = u2.createUindex(self.fourier_basis_number)
    
    def inverseFourierLimited(self,uHalf):
        return u2.irfft2(uHalf,self.fourier_extended_basis_number)

    def fourierTransformHalf(self,z):
        return u2.rfft2(z,self.fourier_basis_number)

    def constructU(self,uHalf):
        """
        Construct Toeplitz Matrix
        """
        return u2.constructU(uHalf,self.Index)
    
    def constructMatexplicit(self,uHalf,fun):
        temp = fun(self.inverseFourierLimited(uHalf))
        temp2 = self.fourierTransformHalf(temp)
        return self.constructU(temp2)