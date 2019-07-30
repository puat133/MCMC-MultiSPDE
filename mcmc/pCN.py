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
#     ('random_gen',Rand_gen_type),
#     ('measurement',meas_type),
#     ('fourier',fourier_type),
#     ('H',nb.complex128[:,:]),
#     ('yBar',nb.float64[:]),
#     ('gibbs_step',nb.int64),
#     ('aggresiveness',nb.float64),
#     ('target_acceptance_rate',nb.float64),
#     ('beta_feedback_gain',nb.float64),
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
        self.gibbs_step = 0
        self.aggresiveness = 0.2
        self.target_acceptance_rate = 0.234
        self.beta_feedback_gain = 2.1
        


    

    # # def computelogRatio(self,Layers):
      
        
    #     return logRatio
    def adapt_beta(self,current_acceptance_rate):
        #based on Algorithm 2 of: Computational Methods for Bayesian Inference in Complex Systems: Thesis
        self.set_beta(self.beta*np.exp(self.beta_feedback_gain*(current_acceptance_rate-self.target_acceptance_rate)))

    def more_aggresive(self):
        self.set_beta(np.min(np.array([(1+self.aggresiveness)*self.beta,1],dtype=np.float64)))
    
    def less_aggresive(self):
        self.set_beta(np.min(np.array([(1-self.aggresiveness)*self.beta,1e-5],dtype=np.float64)))

    def set_beta(self,newBeta):
        self.beta = newBeta
        self.betaZ = np.sqrt(1-newBeta**2)
        
    def oneStep(self,Layers):
        logRatio = 0.0
        for i in range(self.n_layers):
        # i = int(self.gibbs_step//len(Layers))
            
            Layers[i].sample()
            # new_sample = Layers[i].new_sample
            if i> 0:
                Layers[i].LMat.construct_from(Layers[i-1].new_sample)
                Layers[i].new_log_L_det = np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1]
                # if i < self.n_layers - 1 :
                #     Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].current_sample_symmetrized)
                # else:
                Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_symmetrized)
                Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].LMat.latest_computed_L@Layers[i].new_sample_symmetrized)

                logRatio += 0.5*(Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
                logRatio += (Layers[i].new_log_L_det-Layers[i].current_log_L_det)
            else:
                #TODO: Check whether 0.5 factor should be added below
                Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].new_sample/Layers[i].stdev)
                logRatio += (Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
            
        if logRatio>np.log(np.random.rand()):
            for i in range(self.n_layers):
                Layers[i].update_current_sample()
                if not Layers[i].is_stationary:
                    Layers[i].LMat.set_current_L_to_latest()
                
            accepted = 1
        else:
            accepted=0
        # self.gibbs_step +=1
        
        for i in range(self.n_layers):
            Layers[i].record_sample()

        return accepted

    def one_step_one_element(self,Layers,element_index):
        logRatio = 0.0
        for i in range(self.n_layers):
        # i = int(self.gibbs_step//len(Layers))
            if i == 0:
                Layers[i].sample_one_element(element_index)
            else:
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
                #TODO: Check whether 0.5 factor should be added below
                # Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].new_sample/Layers[i].stdev)
                # logRatio += (Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
                logRatio += np.abs((Layers[i].new_sample[element_index]-Layers[i].current_sample[element_index])/Layers[i].stdev[element_index].real)**2
            
        if logRatio>np.log(np.random.rand()):
            for i in range(self.n_layers):
                Layers[i].update_current_sample()
                if not Layers[i].is_stationary:
                    Layers[i].LMat.set_current_L_to_latest()
                
            accepted = 1
        else:
            accepted=0
        # self.gibbs_step +=1
        
        for i in range(self.n_layers):
            Layers[i].record_sample()

        return accepted



        
        


        

    
