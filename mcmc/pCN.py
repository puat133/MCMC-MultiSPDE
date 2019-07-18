import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier
# import mcmc.layer as layer
import mcmc.randomGenerator as randomGenerator
import mcmc.measurement as meas



# Rand_gen_type = nb.deferred_type()
# Rand_gen_type.define(randomGenerator.RandomGenerator.class_type.instance_type)
# meas_type = nb.deferred_type()
# meas_type.define(meas.Measurement.class_type.instance_type)
# fourier_type = nb.deferred_type()
# fourier_type.define(fourier.FourierAnalysis.class_type.instance_type)    
# spec = [
#     ('n_layers',nb.int64),
#     ('beta',nb.float64),
#     ('betaZ',nb.float64),
#     ('H',nb.complex128[:,:]),
#     # ('Layers',Layer_type[:]),
#     ('random_gen',Rand_gen_type),
#     ('measurement',meas_type),
#     ('fourier',fourier_type),
#     ('yBar',nb.float64[:]),
# ]

# @nb.jitclass(spec)
class pCN():
    def __init__(self,n_layers,rg,measurement,f,beta=1):
        self.n_layers = n_layers
        self.beta = beta
        self.betaZ = np.sqrt(1-beta**2)
        self.random_gen = rg
        self.measurement = measurement
        self.fourier = f
        self.H = self.measurement.get_measurement_matrix(self.fourier.fourier_basis_number)/self.measurement.stdev
        self.yBar = np.concatenate((self.measurement.yt/self.measurement.stdev,np.zeros(2*self.fourier.fourier_basis_number-1)))
        


    

    # # def computelogRatio(self,Layers):
      
        
    #     return logRatio

    

    def set_beta(self,newBeta):
        self.beta = newBeta
        self.betaZ = np.sqrt(1-newBeta**2)
        
    def oneStep(self,Layers):
        logRatio = 0.0
        for i in range(self.n_layers):
            
            Layers[i].sample()
            # new_sample = Layers[i].new_sample
            if i> 0:
                Layers[i].LMat.construct_from(Layers[i-1].new_sample)
                Layers[i].new_log_L_det = (np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1])
                Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_symmetrized)
                Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].LMat.latest_computed_L@Layers[i].new_sample_symmetrized)

                logRatio += 0.5*(Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
                logRatio += (Layers[i].new_log_L_det-Layers[i].current_log_L_det)
            else:
                Layers[i].new_sample_scaled_norm = 0.5*util.norm2(Layers[i].new_sample/Layers[i].stdev)
                #TODO: Check whether 0.5 factor should be added below
                logRatio += (Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
        
        if logRatio>np.log(np.random.rand()):
            for i in range(self.n_layers):
                Layers[i].update_current_sample()
                if not Layers[i].is_static:
                    Layers[i].LMat.set_current_L_to_latest()
                
            accepted = 1
        else:
            accepted=0
        
        for i in range(self.n_layers):
            Layers[i].record_sample()

        return accepted



        
        


        

    
