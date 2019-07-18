import numpy as np
import numba as nb
import mcmc.util as util
# import mcmc.fourier as fourier

# fourier_type = nb.deferred_type()
# fourier_type.define(fourier.FourierAnalysis.class_type.instance_type)
spec = [
    # ('fourier',fourier_type),
    ('length',nb.int64),
    ('sqrt2',nb.float64)
]

@nb.jitclass(spec)
class RandomGenerator:
    def __init__(self,length):
        # self.fourier = fourier
        self.length = length
        self.sqrt2 = np.sqrt(2)

    def construct_w_half(self):
        wHalf = np.random.randn(self.length)+1j*np.random.randn(self.length)
        # wHalf[0] = wHalf[0].real*np.sqrt(2)
        wHalf[0] = 2*wHalf[0].real
        # return wHalf/np.sqrt(2)
        return wHalf/self.sqrt2

    def construct_w(self):
        w_half = self.construct_w_half()
        w = self.symmetrize(w_half) #symmetrize
        return w
    
    def symmetrize(self,w_half):
        w = np.concatenate((w_half[:0:-1].conj(),w_half)) #symmetrize
        return w

