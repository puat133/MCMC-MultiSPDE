import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.util_2D as u2
import mcmc.fourier as fourier

fourier_type = nb.deferred_type()
fourier_type.define(fourier.FourierAnalysis.class_type.instance_type)  
spec = [
    ('fourier',fourier_type),
    ('sqrt_beta',nb.float64),
    ('current_L',nb.complex128[:,::1]),
    ('latest_computed_L',nb.complex128[:,::1]),
]

@nb.jitclass(spec)
class Lmatrix():
    def __init__(self,f,sqrt_beta):
        self.fourier = f
        self.sqrt_beta = sqrt_beta

        #initialize self.lastComputedL as zero
        self.current_L = np.zeros((2*self.fourier.basis_number-1,2*self.fourier.basis_number-1),dtype=np.complex128)
        self.latest_computed_L = self.current_L
        

    def construct_from(self,uHalf):
        assert uHalf.shape[0] == self.fourier.basis_number
        Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf,util.kappa_pow_min_nu)
        Ku_pow_half = self.fourier.constructMatexplicit(uHalf,util.kappa_pow_half)
        L = ( util.matMulti(Ku_pow_min_nu,self.fourier.Dmatrix) - Ku_pow_half)/self.sqrt_beta
        
        #set LatestComputedL as L, but dont change currentL
        self.latest_computed_L = L
        return L

    def construct_from_with_sqrt_beta(self,uHalf,sqrt_beta):
        assert uHalf.shape[0] == self.fourier.basis_number
        Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf,util.kappa_pow_min_nu)
        Ku_pow_half = self.fourier.constructMatexplicit(uHalf,util.kappa_pow_half)
        L = ( util.matMulti(Ku_pow_min_nu,self.fourier.Dmatrix) - Ku_pow_half)/sqrt_beta
        
        #set LatestComputedL as L, but dont change currentL
        self.latest_computed_L = L
        return L

    def logDet(self,new):
        """
        # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
        # L^dagger L is Hermitian
        """
        if new:
            return (np.linalg.slogdet(self.latest_computed_L)[1])
        else:
            return (np.linalg.slogdet(self.current_L)[1])

    
    def set_current_L_to_latest(self):
        self.current_L = self.latest_computed_L.copy()
        
    
    def is_current_L_equals_to_the_latest(self):
        return np.all(self.current_L == self.latest_computed_L)


# fourier_type = nb.deferred_type()
# fourier_type.define(fourier.FourierAnalysis_2D.class_type.instance_type)  
# spec = [
#     ('fourier',fourier_type),
#     ('sqrt_beta',nb.float64),
#     ('current_L',nb.complex128[:,::1]),
#     ('latest_computed_L',nb.complex128[:,::1]),
# ]

# @nb.jitclass(spec)
# class Lmatrix_2D:
#     def __init__(self,f,sqrt_beta):
#         self.fourier = f
#         self.sqrt_beta = sqrt_beta

#         #initialize self.lastComputedL as zero
#         self.current_L = np.zeros((self.fourier.basis_number_2D_sym,self.fourier.basis_number_2D_sym),dtype=np.complex128)
#         self.latest_computed_L = self.current_L
        

#     def construct_from(self,uHalf):
#         assert uHalf.shape[0] == self.fourier.basis_number
#         Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf,u2.kappa_pow_min_nu)
#         Ku_pow_half = self.fourier.constructMatexplicit(uHalf,u2.kappa_pow_half)
#         L = ( util.matMulti(Ku_pow_min_nu,self.fourier.Dmatrix) - Ku_pow_half)/self.sqrt_beta
#         #set LatestComputedL as L, but dont change currentL
#         self.latest_computed_L = L
#         return L

#     def logDet(self,new):
#         """
#         # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
#         # L^dagger L is Hermitian
#         """
#         if new:
#             return (np.linalg.slogdet(self.latest_computed_L)[1])
#         else:
#             return (np.linalg.slogdet(self.current_L)[1])

    
#     def set_current_L_to_latest(self):
#         self.current_L = self.latest_computed_L
        
    
#     def is_current_L_equals_to_the_latest(self):
#         return np.all(self.current_L == self.latest_computed_L)
