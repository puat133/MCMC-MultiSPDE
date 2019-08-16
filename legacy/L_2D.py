import numpy as np
import numba as nb
import mcmc.L as L
import mcmc.util_2D as u2
import mcmc.util as util
import mcmc.fourier_2D as f2

# fourier_type = nb.deferred_type()
# fourier_type.define(f2.FourierAnalysis_2D.class_type.instance_type)  
# spec = [
#     ('fourier',fourier_type),
#     ('sqrt_beta',nb.float64),
#     ('current_L',nb.complex128[:,:]),
#     ('latest_computed_L',nb.complex128[:,:]),
# ]

# @nb.jitclass(spec)
# # class Lmatrix_2D(L.Lmatrix):
# class Lmatrix_2D:
#     def __init__(self,f,sqrt_beta):
#         self.fourier = f
#         self.sqrt_beta = sqrt_beta

#         #initialize self.lastComputedL as zero
#         self.current_L = np.zeros(((2*self.fourier.fourier_basis_number-1)**2,(2*self.fourier.fourier_basis_number-1)**2),dtype=np.complex128)
#         self.latest_computed_L = self.current_L
        

#     def construct_from(self,uHalf):
#         assert uHalf.shape[0] == self.fourier.fourier_basis_number
#         Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf,u2.kappa_pow_min_nu)
#         Ku_pow_half = self.fourier.constructMatexplicit(uHalf,u2.kappa_pow_half)
#         L = ( util.matMulti(Ku_pow_min_nu,self.fourier.Dmatrix) - Ku_pow_half)/self.sqrt_beta
#         #set LatestComputedL as L, but dont change currentL
#         self.latest_computed_L = L
#         return L

#     def logDet(self):
#         """
#         # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
#         # L^dagger L is Hermitian
#         """
#         return (np.linalg.slogdet(self.current_L)[1])

    
#     def set_current_L_to_latest(self):
#         self.current_L = self.latest_computed_L.copy()
        
    
#     def is_current_L_equals_to_the_latest(self):
#         return np.all(self.current_L == self.latest_computed_L)