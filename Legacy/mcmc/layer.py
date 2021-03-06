import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier
import mcmc.L as L
import mcmc.pCN as pCN

L_matrix_type = nb.deferred_type()
L_matrix_type.define(L.Lmatrix.class_type.instance_type)  
pCN_type = nb.deferred_type()
pCN_type.define(pCN.pCN.class_type.instance_type)
fourier_type = nb.deferred_type()
fourier_type.define(fourier.FourierAnalysis.class_type.instance_type)     
spec = [
    ('is_stationary',nb.boolean),
    ('sqrt_beta',nb.float64),
    ('order_number',nb.int64),
    ('n_samples',nb.int64),
    ('pcn',pCN_type),
    ('i_record',nb.int64),
    ('stdev',nb.complex128[::1]),
    ('stdev_sym',nb.complex128[::1]),
    ('samples_history',nb.complex128[:,::1]),
    ('current_sample',nb.complex128[::1]),
    ('current_sample_sym',nb.complex128[::1]),
    ('current_sample_scaled_norm',nb.float64),
    ('current_log_L_det',nb.float64),
    ('new_sample',nb.complex128[::1]),
    ('new_sample_sym',nb.complex128[::1]),
    ('new_sample_scaled_norm',nb.float64),
    ('new_log_L_det',nb.float64),
    ('LMat',L_matrix_type),
    ('new_noise_sample',nb.complex128[::1]),
    ('current_noise_sample',nb.complex128[::1]),
]

@nb.jitclass(spec)
class Layer():
    def __init__(self,is_stationary,sqrt_beta,order_number,n_samples,pcn,init_sample):
        self.is_stationary = is_stationary
        self.sqrt_beta = sqrt_beta
        self.order_number = order_number
        # self.current_above_sample = above_sample
        self.n_samples = n_samples

        #dummy declaration
        # a_pcn = pCN.pCN(pcn.n_layers,pcn.random_gen,pcn.measurement,pcn.fourier,pcn.beta)#numba cannot understand without this
        # self.pcn = a_pcn
        self.pcn = pcn

        
        # self.current_sample = np.zeros(f.basis_number,dtype=np.complex128)
        zero_compl_dummy =  np.zeros(self.pcn.fourier.basis_number,dtype=np.complex128)
        ones_compl_dummy =  np.ones(self.pcn.fourier.basis_number,dtype=np.complex128)

        self.stdev = ones_compl_dummy
        self.stdev_sym = util.symmetrize(self.stdev)
        self.samples_history = np.empty((self.n_samples, self.pcn.fourier.basis_number), dtype=np.complex128)
        
        #dummy declaration
        fpcn = self.pcn.fourier
        f = fourier.FourierAnalysis(fpcn.basis_number,fpcn.extended_basis_number,fpcn.t_start,fpcn.t_end)#numba cannot understand without this
        LMat = L.Lmatrix(f,self.sqrt_beta)
        # LMat.fourier = self.pcn.fourier
        self.LMat = LMat
        self.current_noise_sample = self.pcn.random_gen.construct_w()#noise sample always symmetric
        self.new_noise_sample = self.current_noise_sample.copy()
        
        
        if self.is_stationary:
            
            self.current_sample = init_sample
            self.new_sample = init_sample
            self.new_sample_sym = self.pcn.random_gen.symmetrize(self.new_sample)
            self.new_sample_scaled_norm = 0
            self.new_log_L_det = 0
            #numba need this initialization. otherwise it will not compile
            self.current_sample = init_sample.copy()
            self.current_sample_sym = self.new_sample_sym.copy()
            self.current_sample_scaled_norm = 0
            self.current_log_L_det = 0
            
            

        else:
            zero_init = np.zeros(self.pcn.fourier.basis_number,dtype=np.complex128)
            self.LMat.construct_from(init_sample)
            self.LMat.set_current_L_to_latest()
            self.new_sample_sym = np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())
            self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number-1:]
            self.new_sample_scaled_norm = util.norm2(self.LMat.current_L@self.new_sample_sym)#ToDO: Modify this
            self.new_log_L_det = self.LMat.logDet(True)#ToDO: Modify this
            # #numba need this initialization. otherwise it will not compile
            self.current_sample = init_sample.copy()
            self.current_sample_sym = self.new_sample_sym.copy()
            self.current_sample_scaled_norm = self.new_sample_scaled_norm
            self.current_log_L_det = self.new_log_L_det   
            
        # self.update_current_sample()
        self.i_record = 0


    def sample(self):
        #if it is the last layer
        if self.order_number == self.pcn.n_layers -1:
            wNew = self.pcn.random_gen.construct_w()
            eNew = np.random.randn(self.pcn.measurement.num_sample)
            wBar = np.concatenate((eNew,wNew))
            
            LBar = np.vstack((self.pcn.H,self.LMat.current_L))

            #update v
            self.new_sample_sym, res, rnk, s = np.linalg.lstsq(LBar,self.pcn.yBar-wBar )#,rcond=None)
            self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number-1:]
            # return new_sample
        elif self.order_number == 0:
            self.new_sample = self.pcn.betaZ*self.current_sample + self.pcn.beta*self.stdev*self.pcn.random_gen.construct_w_half()
            # self.new_sample_sym = self.pcn.random_gen.symmetrize(self.new_sample) 
        else:
            self.new_sample_sym = self.pcn.betaZ*self.current_sample_sym + self.pcn.beta*np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())
            # self.new_sample_sym = np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())
            self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number-1:]

    def sample_non_centered(self):
        self.new_noise_sample = self.pcn.betaZ*self.current_noise_sample+self.pcn.beta*self.pcn.random_gen.construct_w()
        # if self.order_number == self.pcn.n_layers -1:
        #     wNew = self.new_noise_sample
        #     eNew = np.random.randn(self.pcn.measurement.num_sample)
        #     wBar = np.concatenate((eNew,wNew))
        #     LBar = np.vstack((self.pcn.H,self.LMat.current_L))

        #     #update v
        #     self.new_sample_sym, res, rnk, s = np.linalg.lstsq(LBar,self.pcn.yBar-wBar )#,rcond=None)

        # elif self.order_number == 0:
        #     self.new_sample_sym = self.stdev_sym*self.new_noise_sample
        # else:
        #     self.new_sample_sym = np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())

        # self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number-1:]

       
    def sample_one_element(self,element_index):
        "Only valid if self.is_stationary == True"
        if self.is_stationary and 0 <= element_index< self.pcn.fourier.basis_number:
                if element_index == 0:
                    w = np.random.randn()
                elif element_index < self.pcn.fourier.basis_number:
                    w = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
                self.new_sample = self.current_sample.copy()
                self.new_sample[element_index] = self.pcn.betaZ*self.current_sample[element_index] + self.pcn.beta*self.stdev[element_index]*w
                # self.new_sample_sym = self.pcn.random_gen.symmetrize(self.new_sample)
                


    def record_sample(self):
        self.samples_history[self.i_record,:] = self.current_sample.copy()
        self.i_record += 1
    
    def update_current_sample(self):
        self.current_sample = self.new_sample.copy()
        self.current_sample_sym = self.new_sample_sym.copy()
        self.current_sample_scaled_norm = self.new_sample_scaled_norm
        self.current_log_L_det = self.new_log_L_det
        self.current_noise_sample = self.new_noise_sample.copy()