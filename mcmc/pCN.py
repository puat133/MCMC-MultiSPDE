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
    ('yBar',nb.float64[::1]),
    ('gibbs_step',nb.int64),
    ('aggresiveness',nb.float64),
    ('target_acceptance_rate',nb.float64),
    ('beta_feedback_gain',nb.float64),
    ('non_centered',nb.boolean)
]

@nb.jitclass(spec)
class pCN():
    def __init__(self,n_layers,rg,measurement,f,beta=1,non_centered=False):
        self.n_layers = n_layers
        self.beta = beta
        self.betaZ = np.sqrt(1-beta**2)
        self.random_gen = rg
        self.measurement = measurement
        self.fourier = f
        self.non_centered=non_centered
        self.H = self.measurement.get_measurement_matrix(self.fourier.basis_number)/self.measurement.stdev
        self.I = np.eye(self.measurement.num_sample)
        self.yBar = np.concatenate((self.measurement.yt/self.measurement.stdev,np.zeros(2*self.fourier.basis_number-1)))
        self.gibbs_step = 0
        self.aggresiveness = 0.2
        self.target_acceptance_rate = 0.234
        self.beta_feedback_gain = 2.1
        self.record_skip = 1
        self.record_count = 0
        self.max_record_history = 10000
        temp = self.H.conj().T@self.H
        self.H_t_H = 0.5*(temp+temp.conj().T).real
        


    

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

    def one_step_non_centered(self,Layers):
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
        
    def one_step_non_centered_new(self,Layers):
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
        y = self.measurement.yt
        L = Layers[-1].LMat.current_L
        r = L.conj().T@L + self.H_t_H
        c = np.linalg.cholesky(r)
        Ht = np.linalg.solve(c.conj().T,self.H.T)
        R_inv = self.I/meas_var - (Ht.conj().T@Ht).real/meas_var
        logRatio = 0.5*(y@R_inv@y - np.linalg.slogdet(R_inv)[1])
        L = Layers[-1].LMat.construct_from(Layers[-2].new_sample)
        r = L.conj().T@L + self.H_t_H
        c = np.linalg.cholesky(r)
        Ht = np.linalg.solve(c.conj().T,self.H.T)
        R_inv = self.I/meas_var - (Ht.conj().T@Ht).real/meas_var
        logRatio -= 0.5*(y@R_inv@y - np.linalg.slogdet(R_inv)[1])
        # Ht = np.linalg.solve(L.T.conj(),self.H.T)
        # R = (Ht.T.conj()@Ht + self.I).real*meas_var
        # logRatio = 0.5*(y@(np.linalg.solve(R,y))+np.linalg.slogdet(R)[1])
        # L = Layers[-1].LMat.construct_from(Layers[-2].new_sample)
        # Ht = np.linalg.solve(L.T.conj(),self.H.T)
        # R = (Ht.T.conj()@Ht + self.I).real*meas_var
        # # C = np.linalg.cholesky(R)
        # logRatio -= 0.5*(y@(np.linalg.solve(R,y))+np.linalg.slogdet(R)[1])
                        


            
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
        if (self.record_count%self.record_skip) == 0:
            # print('recorded')
            for i in range(self.n_layers):
                Layers[i].record_sample()
        self.record_count += 1
        return accepted        
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





        
        


        

    
