import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier
import mcmc.L as L
import mcmc.randomGenerator as randomGenerator
import mcmc.pCN as pCN
import mcmc.measurement as meas
# import scipy as scp
import time

fourier_type = nb.deferred_type()
fourier_type.define(fourier.FourierAnalysis.class_type.instance_type) 
L_matrix_type = nb.deferred_type()
L_matrix_type.define(L.Lmatrix.class_type.instance_type)  
Rand_gen_type = nb.deferred_type()
Rand_gen_type.define(randomGenerator.RandomGenerator.class_type.instance_type)  
pCN_type = nb.deferred_type()
pCN_type.define(pCN.pCN.class_type.instance_type)  
meas_type = nb.deferred_type()
meas_type.define(meas.Measurement.class_type.instance_type) 

spec = [
    ('fft',fourier_type),
    ('random_gen',Rand_gen_type),
    ('pcn',pCN_type),
    # ('LMat',L_matrix_type),
    ('measurement',meas_type),
    ('n_samples',nb.int64),
    ('meas_samples_num',nb.int64),
    ('evaluation_interval',nb.int64),
    ('random_seed',nb.int64),
    ('accepted_count',nb.int64),
    ('burn_percentage',nb.float64),
    ('total_time',nb.float64),
    ('yBar',nb.float64[:]),
    ('H',nb.complex128[:,:]),
    ('u_history',nb.complex128[:,:]),
    ('v_history',nb.complex128[:,:]),
]

@nb.jitclass(spec)
class Simulation():
    def __init__(self,n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,
                    kappa = 1e17,sigma_u = 5e6,sigma_v = 10,evaluation_interval = 100,
                    seed=1,burn_percentage = 5.0):
        self.n_samples = n_samples
        self.meas_samples_num = num
        self.evaluation_interval = evaluation_interval
        self.burn_percentage = burn_percentage
        #set random seed
        self.random_seed = seed
        np.random.seed(self.random_seed)
        
        #setup parameters for 1 Dimensional simulation
        d = 1
        nu = 2 - d/2
        alpha = nu + d/2
        t_start = 0.0
        t_end = 1.0
        beta_u = (sigma_u**2)*(2**d * np.pi**(d/2))* 1.1283791670955126#<-- this numerical value is scp.special.gamma(alpha))/scp.special.gamma(nu)
        beta_v = beta_u*(sigma_v/sigma_u)**2
        sqrtBeta_v = np.sqrt(beta_v)
        sqrtBeta_u = np.sqrt(beta_u)
        
        self.fourier = fourier.FourierAnalysis(n,num,t_start,t_end)
        self.random_gen = randomGenerator.RandomGenerator(self.fourier.fourier_basis_number)
        LMat = L.Lmatrix(self.fourier,sqrtBeta_v)

        Lu = (1/sqrtBeta_u)*(self.fourier.Dmatrix*kappa**(-nu) - kappa**(2-nu)*self.fourier.Imatrix)
        init_sample = np.linalg.solve(Lu,self.random_gen.construct_w())[self.fourier.fourier_basis_number-1:]
        uStdev = -1/np.diag(Lu)
        uStdev = uStdev[self.fourier.fourier_basis_number-1:]
        self.pcn = pCN.pCN(LMat,self.random_gen,uStdev,init_sample,beta=beta)
        # self.pcn.current_sample = init_sample
        # self.pcn.LMat.construct_from(newSample)

        meas_std = 0.1
        self.measurement = meas.Measurement(num,meas_std,t_start,t_end)
        self.H = self.measurement.get_measurement_matrix(self.fourier.fourier_basis_number)/meas_std #<-- Normalized
        self.yBar = np.concatenate((self.measurement.yt/meas_std,np.zeros(2*self.fourier.fourier_basis_number-1)))#<-- Normalized

        

    
    
    def run(self):
        self.u_history = np.empty((self.n_samples, self.fourier.fourier_basis_number), dtype=np.complex128)
        self.v_history = np.empty((self.n_samples, self.fourier.fourier_basis_number), dtype=np.complex128)
        self.accepted_count = 0
        average_time_intv =0.0
        with nb.objmode(start_time='float64',start_time_intv='float64'):
            # if printProgress:
            util.printProgressBar(0, self.n_samples, prefix = 'Preparation . . . . ', suffix = 'Complete', length = 50)
            start_time = time.time()
            start_time_intv = start_time
        print('preparation . . . ')
        print('Checking random numbers normalRand & logUniform:')

        # totalTime = 0.0
        accepted_count_partial = 0
        
        for i in range(self.n_samples):#nb.prange(nSim):

            wNew = self.random_gen.construct_w()
            eNew = np.random.randn(self.meas_samples_num)
            wBar = np.concatenate((eNew,wNew))
            
            LBar = np.vstack((self.H,self.pcn.get_current_L()))

            #update v
            v, res, rnk, s = np.linalg.lstsq(LBar,self.yBar-wBar )#,rcond=None)
            
            


            # uHalf,L,logDetL,acceptedPropRes = pCN(uHalf,n,uStdev,beta,betaZ,normalRandChunk,sqrtBeta_v,v,D,dt,cosFun,sinFun,eigenFun,L,logDetL,logUniformChunk)
            accepted_count_partial += self.pcn.oneStep(v)
            
            #save history wheter or not it is accepted
            self.v_history[i,:] = v.copy()[self.fourier.fourier_basis_number-1:]
            self.u_history[i,:] = self.pcn.current_sample.copy()
            
            if (i+1)%(self.evaluation_interval) == 0:
                self.accepted_count += accepted_count_partial
                acceptancePercentage = self.accepted_count/(i+1)
                
                if acceptancePercentage> 0.5:
                    self.pcn.beta = np.min(np.array([1.1*self.pcn.beta,1],dtype=np.float64))
                    # beta = np.min(np.array([1.1*beta,1],dtype=np.float64))
                elif acceptancePercentage<0.3:
                    self.pcn.beta = np.max(np.array([0.9*self.pcn.beta,1e-5],dtype=np.float64))
                    # beta = np.max(np.array([0.9*beta,1e-5],dtype=np.float64))
                
                accepted_count_partial = 0
                self.pcn.betaZ = np.sqrt(1-self.pcn.beta**2)
                mTime = (i+1)/(self.evaluation_interval)

                with nb.objmode(average_time_intv='float64',start_time_intv='float64'):
                    # if printProgress:
                    end_time_intv = time.time()
                    time_intv = end_time_intv-start_time_intv
                    average_time_intv +=  (time_intv-average_time_intv)/mTime
                    start_time_intv = end_time_intv
                    remainingTime = average_time_intv*((self.n_samples - i)/self.evaluation_interval)
                    remainingTimeStr = time.strftime("%j-1 day(s),%H:%M:%S", time.gmtime(remainingTime))
                    util.printProgressBar(i, self.n_samples, prefix = 'Time Remaining {0}- Acceptance Rate {1:.2%} - Progress:'.format(remainingTimeStr,acceptancePercentage), suffix = 'Complete', length = 50)

        with nb.objmode(totalTime='float64'):
            # if printProgress:
            elapsedTimeStr = time.strftime("%j day(s),%H:%M:%S", time.gmtime(time.time()-start_time))
            # print('Complete')
            util.printProgressBar(self.n_samples, self.n_samples, 'Iteration Completed in {0}- Acceptance Rate {1:.2%} - Progress:'.format(elapsedTimeStr,acceptancePercentage), suffix = 'Complete', length = 50)
            self.total_time = time.time()-start_time

        
        # acceptancePercentage = acceptedProposalTotal/nSim