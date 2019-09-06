import numpy as np
import numba as nb
import mcmc.util as util


spec = [
    ('yt',nb.float64[::1]),
    ('vt',nb.float64[::1]),
    ('t',nb.float64[::1]),
    # ('H',nb.complex128[:,::1]),
    ('num_sample',nb.int64),
    ('stdev',nb.float64),
    ('t_end', nb.float64),               
    ('t_start', nb.float64),
]
@nb.jitclass(spec)
class Measurement():
    def __init__(self,num_sample,stdev,t_start=0.0,t_end=1.0):
        self.num_sample = num_sample
        self.stdev = stdev
        self.t_start = t_start
        self.t_end = t_end
        self.t = np.linspace(self.t_start,self.t_end,self.num_sample)
        self.sampleMeasurement()
        

    def sampleMeasurement(self):
        """
        Create one dimensional Sample measurement
        See example 7.1 in Hyperprior for Matern fields with application in 
        Bayesian Inversion
        t0 - start time
        tf - end time
        numsample - number of sample
        stdev - standard deviation of gaussian measurement nosie
        """
        
        
        self.vt = np.zeros(self.t.shape[0])
        for i in range(self.t.shape[0]):
            # if 0<self.t[i]< 0.5*self.t_end:
            #     self.vt[i] = np.exp(4 - 1/(2*self.t[i]-4*self.t[i]**2))
            #     continue
            # if 0.7*self.t_end<=self.t[i]<=0.8*self.t_end:
            #     self.vt[i] = 1
            #     continue
            # if 0.8*self.t_end< self.t[i] <= 0.9*self.t_end:
            #     self.vt[i] = -1
            if 0.2*self.t_end<self.t[i]< 0.8*self.t_end:
                self.vt[i] = 1
                continue
            # if 0.7*self.t_end<=self.t[i]<=0.8*self.t_end:
            #     self.vt[i] = 1
            #     continue
            # if 0.8*self.t_end< self.t[i] <= 0.9*self.t_end:
            #     self.vt[i] = -1

        e = self.stdev*np.random.randn(self.vt.shape[0])
        self.yt = self.vt+e

    def get_measurement_matrix(self,basis_number):
        phi = util.eigenFunction1D
        H = np.zeros((self.t.shape[0],2*basis_number-1),dtype=np.complex128)
        for i in range(-(basis_number-1),basis_number):
            # for j in range(t.shape[0]):       
            H[:,i+basis_number-1] = phi(i,self.t)
                # np.exp(2*np.pi*1j*j*t[i])
        return H

