import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier
import mcmc.L as L
import mcmc.randomGenerator as randomGenerator

L_matrix_type = nb.deferred_type()
Rand_gen_type = nb.deferred_type()
L_matrix_type.define(L.Lmatrix.class_type.instance_type)  
Rand_gen_type.define(randomGenerator.RandomGenerator.class_type.instance_type)  
spec = [
    ('LMat',L_matrix_type),
    ('rand_genn',Rand_gen_type),
    ('uStdev',nb.float64[:]),
    ('beta',nb.float64),
    ('betaZ',nb.float64),
    ('current_sample',nb.complex128[:]),
    ('norm_current_sample',nb.float64),
    ('current_log_det_L',nb.float64),

]

@nb.jitclass(spec)
class pCN():
    def __init__(self,LMat,rand_genn,uStdev,init_sample,beta=1):
        self.LMat = LMat
        self.rand_genn = rand_genn
        self.uStdev = uStdev
        self.beta = beta
        self.betaZ = 1-np.sqrt(beta**2)
        
        #TODO: modify this
        self.current_sample = init_sample
        self.LMat.construct_from(init_sample)
        self.LMat.set_current_L_to_latest()
        self.norm_current_sample = util.norm2(self.current_sample/self.uStdev)
        self.current_log_det_L = self.LMat.logDet()


    def sample(self):
        newSample = self.betaZ*self.current_sample + self.beta*self.uStdev*self.rand_genn.construct_w_half()
        return newSample

    def computelogRatio(self,norm_L_v_2,norm_newL_v_2,norm_new_sample,log_det_newL):
        #compute log Ratio
        logRatio = 0.5*(norm_L_v_2 - norm_newL_v_2)
        logRatio += 0.5*(self.norm_current_sample-norm_new_sample)
        logRatio += (log_det_newL-self.current_log_det_L)
        return logRatio

    def get_current_L(self):
        return self.LMat.current_L
        
    def oneStep(self,v):
        norm_L_v_2 = util.norm2(self.LMat.current_L@v)

        newSample = self.sample()
        newL = self.LMat.construct_from(newSample)

        log_det_newL = (np.linalg.slogdet(newL)[1])
        norm_newL_v_2 = util.norm2(newL@v)
        norm_new_sample = util.norm2(newSample/self.uStdev)

        logRatio = self.computelogRatio(norm_L_v_2,norm_newL_v_2,norm_new_sample,log_det_newL)

        if logRatio>np.log(np.random.randn()):
            self.current_sample = newSample
            self.norm_current_sample = norm_new_sample
            self.current_log_det_L = log_det_newL
            self.LMat.set_current_L_to_latest()
            accepted = 1
        else:
            accepted=0
        
        return accepted



        
        


        

    
