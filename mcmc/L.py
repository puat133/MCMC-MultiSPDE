import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier

fourier_type = nb.deferred_type()
fourier_type.define(fourier.FourierAnalysis.class_type.instance_type)  
spec = [
    ('fourier',fourier_type),
    ('sqrtBeta_v',nb.float64),
    ('current_L',nb.complex128[:,:]),
    ('latest_computed_L',nb.complex128[:,:]),
]

@nb.jitclass(spec)
class Lmatrix():
    def __init__(self,fourier,sqrtBeta_v):
        self.fourier = fourier
        self.sqrtBeta_v = sqrtBeta_v

        #initialize self.lastComputedL as zero
        self.current_L = np.zeros((2*self.fourier.fourier_basis_number-1,2*self.fourier.fourier_basis_number-1),dtype=np.complex128)
        self.latest_computed_L = self.current_L

    def construct_from(self,uHalf):
        assert uHalf.shape[0] == self.fourier.fourier_basis_number
        Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf,util.kappa_pow_min_nu)
        Ku_pow_half = self.fourier.constructMatexplicit(uHalf,util.kappa_pow_min_nu)
        L = ( util.matMulti(Ku_pow_min_nu,self.fourier.Dmatrix) - Ku_pow_half)/self.sqrtBeta_v
        
        #set LatestComputedL as L, but dont change currentL
        self.latest_computed_L = L
        return L

    def logDet(self):
        """
        # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
        # L^dagger L is Hermitian
        """
        return (np.linalg.slogdet(self.current_L)[1])

    
    def set_current_L_to_latest(self):
        self.current_L = self.latest_computed_L
    
    def is_current_L_equals_to_the_latest(self):
        return np.all(self.current_L == self.latest_computed_L)



