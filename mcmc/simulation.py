import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier
import mcmc.L as L
import mcmc.randomGenerator as randomGenerator
import mcmc.pCN as pCN
import scipy as scp

fourier_type = nb.deferred_type()
fourier_type.define(fourier.FourierAnalysis.class_type.instance_type) 
L_matrix_type = nb.deferred_type()
L_matrix_type.define(L.Lmatrix.class_type.instance_type)  
Rand_gen_type = nb.deferred_type()
Rand_gen_type.define(randomGenerator.RandomGenerator.class_type.instance_type)  
pCN_type = nb.deferred_type()
pCN_type.define(pCN.pCN.class_type.instance_type)  

spec = [
    ('fourier',fourier_type),
    ('random_gen',Rand_gen_type),
    ('pcn',pCN_type),
    ('sqrtBeta_v',nb.float64),
    ('current_L',nb.complex128[:,:]),
    ('latest_computed_L',nb.complex128[:,:]),
]

class Simulation():
    def __init__(self,n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,uHalfInit=None,
                    kappa = 1e17,sigma_u = 5e6,sigma_v = 10,printInterval = 100,
                    seed=1,burnPercentage = 5.0,useLaTeX=True,randVectInitiated=False,
                    showFigures=True):

    self.n_samples = n_samples
    self.fourier = fourier.FourierAnalysis(n,num)
    self.random_gen = randomGenerator.RandomGenerator(self.fourier)
    self.pcn = pCN.pCN(,LMat,rand_genn,uStdev,beta=1)

