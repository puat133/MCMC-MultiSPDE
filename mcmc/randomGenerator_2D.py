import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.util_2D as u2
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

    def construct_w_Half_2D(self):
        return u2.construct_w_Half_2D(self.length)

    
    def construct_w_Half_2D_ravelled(self):
        return u2.construct_w_Half_2D_ravelled(self.length)

    def fromUHalfToUHalf2D(self,uHalf):
        return u2.fromUHalfToUHalf2D(uHalf,self.length)

    def fromUHalf2DToUHalf(self,uHalf2D):
        return u2.fromUHalf2DToUHalf(uHalf2D,self.length)

    def construct_w_2D_ravelled(self):
        
        return u2.construct_w_Half_2D_ravelled(self.length)

    def symmetrize_2D(self,uHalf):
        return u2.symmetrize_2D(uHalf)

    # @njitParallel
    # def construct_w_2D(n):
    #     return symmetrize_2D(construct_w_Half_2D(n))

    

