import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.util_2D as u2
# import mcmc.fourier as fourier

# fourier_type = nb.deferred_type()
# fourier_type.define(fourier.FourierAnalysis.class_type.instance_type)
spec = [
    # ('fourier',fourier_type),
    ('basis_number',nb.int64),
    ('sqrt2',nb.float64)
]

@nb.jitclass(spec)
class RandomGenerator:
    def __init__(self,basis_number):
        # self.fourier = fourier
        self.basis_number = basis_number
        self.sqrt2 = np.sqrt(2)

    def construct_w_half(self):
        wHalf = np.random.randn(self.basis_number)+1j*np.random.randn(self.basis_number)
        # wHalf[0] = wHalf[0].real*np.sqrt(2)
        wHalf[0] = self.sqrt2*wHalf[0].real
        # return wHalf/np.sqrt(2)
        return wHalf/self.sqrt2

    def construct_w(self):
        w_half = self.construct_w_half()
        w = self.symmetrize(w_half) #symmetrize
        return w
    
    def symmetrize(self,w_half):
        w = np.concatenate((w_half[:0:-1].conj(),w_half)) #symmetrize
        # w = np.zeros(2*w_half.shape[0]-1,dtype=np.complex128)
        return w

spec2D = [
    # ('fourier',fourier_type),
    ('basis_number',nb.int64),
    ('sqrt2',nb.float64)
]

@nb.jitclass(spec2D)
class RandomGenerator_2D:
    def __init__(self,basis_number):
        # self.fourier = fourier
        self.basis_number = basis_number
        self.sqrt2 = np.sqrt(2)

    def construct_w_Half_2D(self):
        return u2.construct_w_Half_2D(self.basis_number)

    
    def construct_w_Half_2D_ravelled(self):
        return u2.construct_w_Half_2D_ravelled(self.basis_number)

    def fromUHalfToUHalf2D(self,uHalf):
        return u2.fromUHalfToUHalf2D(uHalf,self.basis_number)

    def fromUHalf2DToUHalf(self,uHalf2D):
        return u2.fromUHalf2DToUHalf(uHalf2D,self.basis_number)

    def construct_w_2D_ravelled(self):
        
        return u2.construct_w_Half_2D_ravelled(self.basis_number)

    def symmetrize_2D(self,uHalf2D):
        return u2.symmetrize_2D(uHalf2D)
