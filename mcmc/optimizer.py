import numpy as np
# import scipy.linalg as sla
# import scipy.optimize as sciop
import numba as nb
import mcmc.util as util
from scipy.optimize import minimize

FASTMATH=True
PARALLEL = False
CACHE=False
SQRT2 = np.sqrt(2)
# from numba import complex64, complex128, float32, float64, int32, jit, njit, prange
njitSerial = nb.njit(fastmath=FASTMATH,cache=CACHE)
jitSerial = nb.jit(fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
jitParallel = nb.jit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)

@njitParallel
def xToUHalf(x):
    n = (x.shape[0]+1)//2
    uHalf = np.zeros(n,dtype=np.complex128)
    uHalf = x[:n]+1j*np.concatenate((np.array([0.0]),x[n:]))
    return uHalf

@njitParallel
def uHalfToX(uHalf):
    return np.concatenate((uHalf.real,uHalf[1:].imag))

# @njitParallel
def negLogPosterior(x,Layers):
    """
    Layers are numba typed List
    """
    negLogPost = 0.0
    uHalf_all = xToUHalf(x) # this is just like having new sample at the bottom layer
    n = Layers[0].pcn.fourier.fourier_basis_number
    uHalf_0 = uHalf_all[0:n]
    Layers[0].new_sample = uHalf_0
    Layers[0].new_sample_symmetrized = Layers[0].pcn.random_gen.symmetrize(Layers[0].new_sample)
    Layers[0].new_sample_scaled_norm = util.norm2(Layers[0].new_sample/Layers[0].stdev)
    Layers[0].update_current_sample()
    negLogPost += Layers[0].current_sample_scaled_norm
    for i in range(1,len(Layers)):
        Layers[i].LMat.construct_from(Layers[i-1].new_sample)
        Layers[i].new_log_L_det = np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1]
        Layers[i].LMat.set_current_L_to_latest()
        # Layers[i].sample()
        if i== len(Layers)-1:
            wNew = util.symmetrize(uHalf_all[n*(i-1):n*i]) 
            eNew = np.random.randn(Layers[i].pcn.measurement.num_sample)
            wBar = np.concatenate((eNew,wNew))
            
            LBar = np.vstack((Layers[i].pcn.H,Layers[i].LMat.current_L))
            Layers[i].new_sample_symmetrized, res, rnk, s = np.linalg.lstsq(LBar,Layers[i].pcn.yBar-wBar )#,rcond=None)
            Layers[i].new_sample = Layers[i].new_sample_symmetrized[Layers[i].pcn.fourier.fourier_basis_number-1:]
        else:
            uHalf_i = uHalf_all[n*(i-1):n*i]
            Layers[i].new_sample = uHalf_i
        Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_symmetrized)
        Layers[i].update_current_sample()
        negLogPost += 0.5*Layers[i].current_sample_scaled_norm
        negLogPost -= Layers[i].current_log_L_det    
    
    # self.current_neg_log_posterior = negLogPost
    return negLogPost

# def negLogPosterior(x,Layers):
#     """
#     Layers are numba typed List
#     """
#     negLogPost = 0.0
#     uHalf_0 = xToUHalf(x) # this is just like having new sample at the bottom layer
#     Layers[0].new_sample = uHalf_0
#     Layers[0].new_sample_symmetrized = Layers[0].pcn.random_gen.symmetrize(Layers[0].new_sample)
#     Layers[0].new_sample_scaled_norm = util.norm2(Layers[0].new_sample/Layers[0].stdev)
#     Layers[0].update_current_sample()
#     negLogPost += Layers[0].current_sample_scaled_norm
#     for i in range(1,len(Layers)):
#         Layers[i].LMat.construct_from(Layers[i-1].new_sample)
#         Layers[i].new_log_L_det = np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1]
#         Layers[i].LMat.set_current_L_to_latest()
#         Layers[i].sample()
#         Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_symmetrized)
#         Layers[i].update_current_sample()
#         negLogPost += 0.5*Layers[i].current_sample_scaled_norm
#         negLogPost -= Layers[i].current_log_L_det    
    
#     # self.current_neg_log_posterior = negLogPost
#     return negLogPost

class Optimizer():
    """
    Layers are numba typed List
    """
    def __init__(self,Layers,method = 'Powell',uHalf_Start = None,max_iter = 10000,current_neg_log_posterior = 0):
        self.Layers = Layers
        self.method = method
        self.max_iter = max_iter
        self.current_neg_log_posterior = current_neg_log_posterior
        self.uHalf_Start = uHalf_Start
        self.n_f_eval = 0

    # def negLogPosterior(self,x,Layers):
    #     """
    #     Layers are numba typed List
    #     """
    #     negLogPost = 0.0
    #     uHalf_0 = xToUHalf(x) # this is just like having new sample at the bottom layer
    #     Layers[0].new_sample = uHalf_0
    #     Layers[0].new_sample_symmetrized = Layers[0].pcn.random_gen.symmetrize(Layers[0].new_sample)
    #     Layers[0].new_sample_scaled_norm = util.norm2(Layers[0].new_sample/Layers[0].stdev)
    #     Layers[0].update_current_sample()
    #     negLogPost -= Layers[0].current_sample_scaled_norm
    #     for i in range(1,len(Layers)-1):
    #         Layers[i].LMat.construct_from(Layers[i-1].new_sample)
    #         Layers[i].new_log_L_det = np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1]
    #         Layers[i].LMat.set_current_L_to_latest()
    #         Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_symmetrized)
    #         negLogPost -= 0.5*Layers[i].current_sample_scaled_norm
    #         negLogPost -= Layers[i].current_log_L_det    
        
    #     self.current_neg_log_posterior = negLogPost
    #     return negLogPost

    def optimize(self):
        """
        this function is a wrapper of scipy minimisation solver
        """
        if np.all(self.uHalf_Start == None):
            self.Layers[0].sample()
            self.uHalf_Start = self.Layers[0].new_sample
            for i in range(1,len(self.Layers)):
                self.Layers[i].sample()
                self.uHalf_Start = np.concatenate((self.uHalf_Start,self.Layers[i].new_sample))
                
        
        #it seems that minimize cannot handle complex
        xStart = uHalfToX(self.uHalf_Start)
        
        # global NFEVAL
        # global MAXITER
        # NFEVAL = 1
        # MAXITER = maxiter
        res = minimize(lambda x: negLogPosterior(x,self.Layers)\
                    ,xStart
                    ,method = self.method
    #                   ,full_output=1
    #                  
    #                   retall=1, 
    #                   ,iprint=1
                    #    ,callback=self.callbackF
    #                   ,bounds = bound
                    ,options={'maxiter':self.max_iter,'disp':True}
                    )
        return res

    ##Print callback function
    def callbackF(self):
        
        util.printProgressBar(self.n_f_eval%self.max_iter, self.max_iter, 'Current Log Posterior {0:.5} - Progress:'.format(self.current_neg_log_posterior), suffix = 'Complete', length = 50)
        self.n_f_eval += 1