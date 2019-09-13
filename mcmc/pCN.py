import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier
# import mcmc.layer as layer
import mcmc.randomGenerator as randomGenerator
import mcmc.measurement as meas



Rand_gen_type = nb.deferred_type()
Rand_gen_type.define(randomGenerator.RandomGenerator.class_type.instance_type)
meas_type = nb.deferred_type()
meas_type.define(meas.Measurement.class_type.instance_type)
fourier_type = nb.deferred_type()
fourier_type.define(fourier.FourierAnalysis.class_type.instance_type)    
spec = [
    ('n_layers',nb.int64),
    ('record_skip',nb.int64),
    ('record_count',nb.int64),
    ('max_record_history',nb.int64),
    ('beta',nb.float64),
    ('betaZ',nb.float64),
    ('random_gen',Rand_gen_type),
    ('measurement',meas_type),
    ('fourier',fourier_type),
    ('H',nb.complex128[:,::1]),
    ('I',nb.float64[:,::1]),
    ('H_t_H',nb.float64[:,::1]),
    ('H_dagger',nb.complex128[:,::1]),
    ('yBar',nb.float64[::1]),
    ('y',nb.float64[::1]),
    ('gibbs_step',nb.int64),
    ('aggresiveness',nb.float64),
    ('target_acceptance_rate',nb.float64),
    ('beta_feedback_gain',nb.float64),
    ('variant',nb.typeof("string")),
    ('non_centered',nb.boolean),
    ('pcn_step_sqrtBetas',nb.float64),
    ('Layers_sqrtBetas',nb.float64[:]),
    ('stdev_sqrtBetas',nb.float64[:]),
    ('sqrtBetas_history',nb.float64[:,::1]),
    
    

]

@nb.jitclass(spec)
class pCN():
    def __init__(self,n_layers,rg,measurement,f,beta=1,variant="sari"):
        self.n_layers = n_layers
        self.beta = beta
        self.betaZ = np.sqrt(1-beta**2)
        self.random_gen = rg
        self.measurement = measurement
        self.fourier = f
        self.variant=variant
        self.H = self.measurement.get_measurement_matrix(self.fourier.basis_number)/self.measurement.stdev
        self.I = np.eye(self.measurement.num_sample)
        self.y = self.measurement.yt/self.measurement.stdev
        self.yBar = np.concatenate((self.measurement.yt/self.measurement.stdev,np.zeros(2*self.fourier.basis_number-1)))
        
        self.gibbs_step = 0
        self.aggresiveness = 0.2
        self.target_acceptance_rate = 0.234
        self.beta_feedback_gain = 2.1
        self.record_skip = 1
        self.record_count = 0
        self.max_record_history = 10000
        # temp = self.H.conj().T
        # self.H_dagger = temp
        temp2 = self.H.conj().T@self.H
        self.H_t_H = 0.5*(temp2+temp2.conj().T).real

        self.Layers_sqrtBetas = np.zeros(self.n_layers,dtype=np.float64)
        self.pcn_step_sqrtBetas = 1e-1
        self.stdev_sqrtBetas = np.ones(self.n_layers,dtype=np.float64)
        self.sqrtBetas_history = np.empty((10000, self.n_layers), dtype=np.float64)
        
        
        
        


    

    # # def computelogRatio(self,Layers):
      
        
    #     return logRatio
    def adapt_beta(self,current_acceptance_rate):
        #based on Algorithm 2 of: Computational Methods for Bayesian Inference in Complex Systems: Thesis
        self.set_beta(self.beta*np.exp(self.beta_feedback_gain*(current_acceptance_rate-self.target_acceptance_rate)))

    def more_aggresive(self):
        self.set_beta(np.min(np.array([(1+self.aggresiveness)*self.beta,1],dtype=np.float64)))
    
    def less_aggresive(self):
        self.set_beta(np.min(np.array([(1-self.aggresiveness)*self.beta,1e-10],dtype=np.float64)))

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
                #     Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].current_sample_sym)
                # else:
                Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_sym)
                Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].LMat.latest_computed_L@Layers[i].new_sample_sym)

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
        # only record when needed
        if (self.record_count%self.record_skip) == 0:
            # print('recorded')
            for i in range(self.n_layers):
                Layers[i].record_sample()
        self.record_count += 1

        return accepted

    def one_step_non_centered_sari(self,Layers):
        accepted = 0
        Layers[self.n_layers-1].sample_non_centered()
        wNew = Layers[self.n_layers-1].new_noise_sample
        eNew = np.random.randn(self.measurement.num_sample)
        wBar = np.concatenate((eNew,wNew))
        LBar = np.vstack((self.H,Layers[self.n_layers-1].LMat.current_L))
        v, res, rnk, s = np.linalg.lstsq(LBar,self.yBar-wBar )#,rcond=None)
        Layers[self.n_layers-1].new_sample_sym = v
        Layers[self.n_layers-1].new_sample = v[self.fourier.basis_number-1:]
        logRatio = 0.0
        for i in range(self.n_layers):
            if i<self.n_layers-1:
                Layers[i].sample_non_centered()

            if i>0:
                Layers[i].LMat.construct_from(Layers[i-1].new_sample)
                if i<self.n_layers-1:
                    Layers[i].new_sample_sym = np.linalg.solve(Layers[i].LMat.latest_computed_L,Layers[i].new_noise_sample)
            else:
                Layers[i].new_sample_sym = Layers[i].stdev_sym*Layers[i].new_noise_sample
            Layers[i].new_sample = Layers[i].new_sample_sym[self.fourier.basis_number-1:]

            if i == self.n_layers-1:
                logRatio += Layers[i].LMat.logDet(True) - Layers[i].LMat.logDet(False)
                logRatio += 0.5*(util.norm2(Layers[i].LMat.current_L@v)-util.norm2(Layers[i].LMat.latest_computed_L@v))
            if i<self.n_layers-1:
                logRatio += 0.5*(util.norm2(Layers[i].current_noise_sample) - util.norm2(Layers[i].new_noise_sample))
                


            
        if logRatio>np.log(np.random.rand()):
            accepted = 1
            for i in range(self.n_layers):
                Layers[i].update_current_sample()
                if i<self.n_layers-1 and not Layers[i+1].is_stationary:
                    Layers[i+1].LMat.set_current_L_to_latest()

            # only record when needed
        if (self.record_count%self.record_skip) == 0:
            # print('recorded')
            for i in range(self.n_layers):
                Layers[i].record_sample()
        self.record_count += 1
        return accepted
        
    def one_step_non_centered_dunlop(self,Layers):
        accepted = 0
        logRatio = 0.0
        for i in range(self.n_layers-1):
            Layers[i].sample_non_centered()
            if i>0:
                Layers[i].LMat.construct_from(Layers[i-1].new_sample)
                Layers[i].new_sample_sym = np.linalg.solve(Layers[i].LMat.latest_computed_L,Layers[i].new_noise_sample)
            else:
                Layers[i].new_sample_sym = Layers[i].stdev_sym*Layers[i].new_noise_sample
            Layers[i].new_sample = Layers[i].new_sample_sym[self.fourier.basis_number-1:]
        meas_var = self.measurement.stdev**2
        # y = self.measurement.yt
        L = Layers[-1].LMat.current_L
        temp = L.conj().T@L
        r = 0.5*(temp+temp.conj().T) + self.H_t_H
        c = np.linalg.cholesky(r)
        Ht = np.linalg.solve(c,self.H.conj().T)
        
        # R_inv = self.I/meas_var - (Ht.conj().T@Ht).real/meas_var
        # R_inv = (self.I - (Ht.conj().T@Ht).real)/meas_var
        # logRatio -= 0.5*(self.y@R_inv@self.y - np.linalg.slogdet(R_inv)[1])
        R_inv = self.I - (Ht.conj().T@Ht).real
        logRatio = 0.5*(self.y@R_inv@self.y - np.linalg.slogdet(R_inv/meas_var)[1])

        

        L = Layers[-1].LMat.construct_from(Layers[-2].new_sample)
        temp = L.conj().T@L
        r = 0.5*(temp+temp.conj().T) + self.H_t_H
        c = np.linalg.cholesky(r)
        Ht = np.linalg.solve(c,self.H.conj().T)
        
        # R_inv = (self.I - (Ht.conj().T@Ht).real)/meas_var
        # logRatio -= 0.5*(self.y@R_inv@self.y - np.linalg.slogdet(R_inv)[1])
        R_inv = self.I - (Ht.conj().T@Ht).real
        logRatio -= 0.5*(self.y@R_inv@self.y - np.linalg.slogdet(R_inv/meas_var)[1])
        
                        


            
        if logRatio>np.log(np.random.rand()):
            accepted = 1
            #sample the last layer
            Layers[-1].sample_non_centered()
            wNew = Layers[-1].new_noise_sample
            eNew = np.random.randn(self.measurement.num_sample)
            wBar = np.concatenate((eNew,wNew))
            LBar = np.vstack((self.H,Layers[-1].LMat.latest_computed_L))
            v, res, rnk, s = np.linalg.lstsq(LBar,self.yBar-wBar )#,rcond=None)
            Layers[-1].new_sample_sym = v
            Layers[-1].new_sample = v[self.fourier.basis_number-1:]
            for i in range(self.n_layers):
                Layers[i].update_current_sample()
                if not Layers[i].is_stationary:
                    Layers[i].LMat.set_current_L_to_latest()

            # only record when needed
        #adapt sqrtBetas
        # self.one_step_for_sqrtBetas(Layers)
        if (self.record_count%self.record_skip) == 0:
            # print('recorded')
            self.sqrtBetas_history[self.record_count,:] = self.Layers_sqrtBetas
            for i in range(self.n_layers):
                Layers[i].record_sample()
            
            

        self.record_count += 1

        

        return accepted

    def one_step_for_sqrtBetas(self,Layers):
        # pcn_step_sqrtBetas = 1e-1
        # stdev_sqrtBetas = 1
        sqrt_beta_noises = self.stdev_sqrtBetas*np.random.randn(self.n_layers)
        # sqrtBetas = np.zeros(self.n_layers,dtype=np.float64)
        propSqrtBetas = np.zeros(self.n_layers,dtype=np.float64)

        for i in range(self.n_layers):
            
            temp = np.sqrt(1-self.pcn_step_sqrtBetas**2)*Layers[i].sqrt_beta + self.pcn_step_sqrtBetas*sqrt_beta_noises[i]
            propSqrtBetas[i] = max(temp,1e-4)
            if i==0:
                stdev_sym_temp = (propSqrtBetas[i]/Layers[i].sqrt_beta)*Layers[i].stdev_sym
                Layers[i].new_sample_sym = stdev_sym_temp*Layers[i].current_noise_sample
            else:
                Layers[i].LMat.construct_from_with_sqrt_beta(Layers[i-1].new_sample,propSqrtBetas[i])
                if i < self.n_layers-1:
                    Layers[i].new_sample_sym = np.linalg.solve(Layers[i].LMat.latest_computed_L,Layers[i].current_noise_sample)
                else:        
                    wNew = Layers[-1].current_noise_sample
                    eNew = np.random.randn(self.measurement.num_sample)
                    wBar = np.concatenate((eNew,wNew))
                    LBar = np.vstack((self.H,Layers[-1].LMat.latest_computed_L))
                    v, res, rnk, s = np.linalg.lstsq(LBar,self.yBar-wBar )
                    Layers[-1].new_sample_sym = v
                    Layers[i].new_sample = Layers[i].new_sample_sym[self.fourier.basis_number-1:]

        logRatio = 0.5*(util.norm2(self.y/self.measurement.stdev - self.H@Layers[-1].current_sample_sym))
        logRatio -= 0.5*(util.norm2(self.y/self.measurement.stdev - self.H@Layers[-1].new_sample_sym))

        if logRatio>np.log(np.random.rand()):
            # print('Proposal sqrt_beta accepted!')
            self.Layers_sqrtBetas = propSqrtBetas
            for i in range(self.n_layers):
                Layers[i].sqrt_beta = propSqrtBetas[i]
                Layers[i].LMat.set_current_L_to_latest()
                if Layers[i].is_stationary:
                    Layers[i].stdev_sym = stdev_sym_temp
                    Layers[i].stdev = Layers[i].stdev_sym[self.fourier.basis_number-1:]

        

    # def one_step_non_centered_new(self,Layers):
    #     accepted = 0
        
        
    #     for i in range(self.n_layers):
    #         Layers[i].sample_non_centered()
    #         if i>0:
    #             Layers[i].LMat.construct_from(Layers[i-1].new_sample)
    #             # if i<self.n_layers-1:
    #             Layers[i].new_sample_sym = np.linalg.solve(Layers[i].LMat.latest_computed_L,Layers[i].new_noise_sample)
    #         else:
    #             Layers[i].new_sample_sym = Layers[i].stdev_sym*Layers[i].new_noise_sample
    #         Layers[i].new_sample = Layers[i].new_sample_sym[self.fourier.basis_number-1:]

    #     logRatio = 0.5*(util.norm2(self.measurement.yt/self.measurement.stdev - self.H@Layers[self.n_layers-1].current_sample_sym) - util.norm2(self.measurement.yt/self.measurement.stdev - self.H@Layers[self.n_layers-1].new_sample_sym))
    #     # a = np.min(np.array([1,np.exp(logRatio)]))
    #     if logRatio>np.log(np.random.rand()):
    #     # if a>np.random.rand():
    #         accepted = 1
    #         for i in range(self.n_layers):
    #             Layers[i].update_current_sample()
    #             if not Layers[i].is_stationary:
    #                 Layers[i].LMat.set_current_L_to_latest()

    #         # only record when needed
    #     if (self.record_count%self.record_skip) == 0:
    #         # print('recorded')
    #         for i in range(self.n_layers):
    #             Layers[i].record_sample()
    #     self.record_count += 1
    #     return accepted


    # def one_step_one_element(self,Layers,element_index):
    #     logRatio = 0.0
    #     for i in range(self.n_layers):
    #     # i = int(self.gibbs_step//len(Layers))
    #         if i == 0:
    #             Layers[i].sample_one_element(element_index)
    #         else:
    #             Layers[i].sample()
    #         # new_sample = Layers[i].new_sample
    #         if i> 0:
    #             Layers[i].LMat.construct_from(Layers[i-1].new_sample)
    #             Layers[i].new_log_L_det = (np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1])
    #             Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_sym)
    #             Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].LMat.latest_computed_L@Layers[i].new_sample_sym)

    #             logRatio += 0.5*(Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
    #             logRatio += (Layers[i].new_log_L_det-Layers[i].current_log_L_det)
    #         else:
    #             #TODO: Check whether 0.5 factor should be added below
    #             # Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].new_sample/Layers[i].stdev)
    #             # logRatio += (Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
    #             logRatio += np.abs((Layers[i].new_sample[element_index]-Layers[i].current_sample[element_index])/Layers[i].stdev[element_index].real)**2
            
    #     if logRatio>np.log(np.random.rand()):
    #         for i in range(self.n_layers):
    #             Layers[i].update_current_sample()
    #             if not Layers[i].is_stationary:
    #                 Layers[i].LMat.set_current_L_to_latest()
                
    #         accepted = 1
    #     else:
    #         accepted=0
    #     # self.gibbs_step +=1
        
    #     for i in range(self.n_layers):
    #         Layers[i].record_sample()

    #     return accepted
    
    # def oneStep_pair(self,Layers):
        
    #     accepted = 0
    #     for i in range(self.n_layers-1,0,-1):#do it from the back
    #     # i = int(self.gibbs_step//len(Layers))
    #         logRatio = 0.0
    #         if i == self.n_layers-1:    
    #             Layers[i].sample()
    #         Layers[i-1].sample()
    #         #Layer i
    #         Layers[i].LMat.construct_from(Layers[i-1].new_sample)
    #         Layers[i].new_log_L_det = np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1]
            
    #         Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_sym)
    #         # assert Layers[i].current_sample_scaled_norm != np.nan
    #         Layers[i].new_sample_scaled_norm = util.norm2(Layers[i].LMat.latest_computed_L@Layers[i].new_sample_sym)
    #         # assert Layers[i].new_sample_scaled_norm != np.nan

    #         logRatio += 0.5*(Layers[i].current_sample_scaled_norm-Layers[i].new_sample_scaled_norm)
    #         logRatio += (Layers[i].new_log_L_det-Layers[i].current_log_L_det)
            
    #         #Layer i-1
    #         if i-1>0:
    #             Layers[i-1].new_sample_scaled_norm = util.norm2(Layers[i-1].LMat.current_L@Layers[i-1].new_sample_sym)
    #         else:
    #             Layers[i-1].new_sample_scaled_norm = util.norm2(Layers[i-1].new_sample/Layers[i-1].stdev)
                    
    #         logRatio += 0.5*(Layers[i-1].current_sample_scaled_norm-Layers[i-1].new_sample_scaled_norm)
    #         if logRatio>np.log(np.random.rand()):
    #             for i in range(self.n_layers):
    #                 Layers[i].update_current_sample()
    #                 if not Layers[i].is_stationary:
    #                     Layers[i].LMat.set_current_L_to_latest()
                    
    #             accepted += 1
        
            
    #         # for i in range(self.n_layers):
    #         #     Layers[i].record_sample()
    #         #  only record when needed
    #         if (self.record_count%self.record_skip) == 0:
    #             # print('recorded')
    #             for i in range(self.n_layers):
    #                 Layers[i].record_sample()
    #         self.record_count += 1

    #     return accepted