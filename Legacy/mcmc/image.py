from skimage.io import imread
from skimage.transform import resize
import warnings
import numpy as np
import scipy.linalg as sla
import os
import matplotlib.pyplot as plt
import pathlib
import numba as nb
import mcmc.util as util
import mcmc.util_2D as u2

FASTMATH=True
PARALLEL = True
CACHE=True
ORDER = 'C'
njitSerial = nb.njit(fastmath=FASTMATH,cache=CACHE)
jitSerial = nb.jit(fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
jitParallel = nb.jit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)


specFourier = [
    ('basis_number', nb.int64),               
    ('basis_number_2D_ravel', nb.int64),
    ('extended_basis_number', nb.int64),
    ('basis_number_2D', nb.int64),
    ('basis_number_2D_sym', nb.int64),               
    ('extended_basis_number_2D', nb.int64),               
    ('extended_basis_number_2D_sym', nb.int64),               
    ('t_end', nb.float64),               
    ('t_start', nb.float64),               
    ('dt', nb.float64),
    ('t', nb.float64[::1]),
    ('Dmatrix', nb.float64[:,::1]),
    ('Imatrix', nb.float64[:,::1]),
    ('ix', nb.int64[:,::1]),
    ('iy', nb.int64[:,::1]),
    ('Index',nb.typeof(u2.createUindex(2)))
]
@nb.jitclass(specFourier)
class FourierAnalysis_2D:
    def __init__(self,basis_number,extended_basis_number,t_start = 0,t_end=1):
        self.basis_number = basis_number
        self.extended_basis_number = extended_basis_number
        self.basis_number_2D = (2*basis_number-1)*basis_number
        self.basis_number_2D_ravel = (2*basis_number*basis_number-2*basis_number+1)
        self.basis_number_2D_sym = (2*basis_number-1)*(2*basis_number-1)
        self.extended_basis_number_2D = (2*extended_basis_number-1)*extended_basis_number
        self.extended_basis_number_2D_sym = (2*extended_basis_number-1)*(2*extended_basis_number-1)
        self.t_end = t_end
        self.t_start = t_start
        self.ix = np.zeros((2*self.basis_number-1,2*self.basis_number-1),dtype=np.int64)
        self.iy = np.zeros((2*self.basis_number-1,2*self.basis_number-1),dtype=np.int64)
        temp = np.arange(-(self.basis_number-1),self.basis_number)
        for i in range(2*self.basis_number-1):
            self.ix[i,:] = temp
            self.iy[:,i] = temp

        # d_diag = np.zeros((2*self.basis_number-1)**2)
        # for i in range(2*self.basis_number-1):
        #     for j in range(2*self.basis_number-1):
        #         d_diag[i*10+j] = (i**2+j**2)
        self.Dmatrix = -(2*np.pi)**2*np.diag(self.ix.ravel()**2+self.iy.ravel()**2)
        self.Imatrix = np.eye((2*self.basis_number-1)**2)
        Index = u2.createUindex(self.basis_number)
        self.Index = Index
    
    def inverseFourierLimited(self,uHalf2D):
        #if order = 'C' then it needs transpose
        return u2.irfft2(uHalf2D,self.extended_basis_number)
        # return u2.irfft2(uHalf2D,self.extended_basis_number)

    def fourierTransformHalf(self,z):
        return u2.rfft2(z,self.basis_number)

    def constructU(self,uHalf2D):
        """
        Construct Toeplitz Matrix
        """
        return u2.constructU(uHalf2D,self.Index)
    
    def constructMatexplicit(self,uHalf2D,fun):
        temp = fun(self.inverseFourierLimited(uHalf2D).T)
        temp2 = self.fourierTransformHalf(temp)
        return self.constructU(temp2)

fourier_type = nb.deferred_type()
fourier_type.define(FourierAnalysis_2D.class_type.instance_type)  
specL = [
    ('fourier',fourier_type),
    ('sqrt_beta',nb.float64),
    ('current_L',nb.complex128[:,::1]),
    ('latest_computed_L',nb.complex128[:,::1]),
]
@nb.jitclass(specL)
class Lmatrix_2D:
    def __init__(self,f,sqrt_beta):
        self.fourier = f
        self.sqrt_beta = sqrt_beta

        #initialize self.lastComputedL as zero
        self.current_L = np.zeros((self.fourier.basis_number_2D_sym,self.fourier.basis_number_2D_sym),dtype=np.complex128)
        self.latest_computed_L = self.current_L
        

    def construct_from_2D(self,uHalf2D):
        assert uHalf2D.shape[1] == self.fourier.basis_number
        Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf2D,u2.kappa_pow_min_nu)
        Ku_pow_d_per_2 = self.fourier.constructMatexplicit(uHalf2D,u2.kappa_pow_d_per_2)
        L = ( util.matMulti(Ku_pow_min_nu,self.fourier.Dmatrix) - Ku_pow_d_per_2)/self.sqrt_beta
        #set LatestComputedL as L, but dont change currentL
        self.latest_computed_L = L
        return L

    def construct_from(self,uHalf):
        uHalf2D = u2.from_u_2D_ravel_to_uHalf_2D(util.symmetrize(uHalf),self.fourier.basis_number)
        return self.construct_from_2D(uHalf2D)
         
    def logDet(self,new):
        """
        # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
        # L^dagger L is Hermitian
        """
        if new:
            return (np.linalg.slogdet(self.latest_computed_L)[1])
        else:
            return (np.linalg.slogdet(self.current_L)[1])

    
    def set_current_L_to_latest(self):
        self.current_L = self.latest_computed_L
        
    
    def is_current_L_equals_to_the_latest(self):
        return np.all(self.current_L == self.latest_computed_L)

specRG = [
    # ('fourier',fourier_type),
    ('basis_number',nb.int64),
    ('basis_number_2D_ravel',nb.int64),#This is half+1 of Fourier Frequency 2D
    ('sqrt2',nb.float64)
]
@nb.jitclass(specRG)
class RandomGenerator_2D:
    def __init__(self,basis_number):
        # self.fourier = fourier
        self.basis_number = basis_number
        self.basis_number_2D_ravel = (2*basis_number*basis_number-2*basis_number+1)
        self.sqrt2 = np.sqrt(2)

    def construct_w_Half_2D(self):
        return u2.construct_w_Half_2D(self.basis_number)
    
    def construct_w_Half(self):
        return util.construct_w_Half(self.basis_number_2D_ravel)
    
    def construct_w(self):
        w_half = self.construct_w_Half()
        w = self.symmetrize(w_half)
        return w
        
    def symmetrize(self,w_half):
        w = np.concatenate((w_half[:0:-1].conj(),w_half)) #symmetrize
        # w = np.zeros(2*w_half.shape[0]-1,dtype=np.complex128)
        return w
    
    # def construct_w_Half_2D_ravelled(self):
    #     return u2.construct_w_Half_2D_ravelled(self.basis_number)

    def fromUHalfToUHalf2D(self,uHalf):
        return u2.fromUHalfToUHalf2D(uHalf,self.basis_number)

    def fromUHalf2DToUHalf(self,uHalf2D):
        return u2.fromUHalf2DToUHalf(uHalf2D,self.basis_number)

    # def construct_w_2D_ravelled(self):
    #     return u2.construct_w_2D_ravelled(self.basis_number)

    def symmetrize_2D(self,uHalf2D):
        return u2.symmetrize_2D(uHalf2D)

class TwoDMeasurement:
    def __init__(self,file_name,target_size=128,stdev=0.1,relative_location=""):
        self.globalprefix = pathlib.Path.cwd() / relative_location
        if not self.globalprefix.exists():
            self.globalprefix.mkdir()
        self.file_name= self.globalprefix / file_name
        if target_size>512:
            raise Exception("Target image dimension are too large (" + str(target_size) + " x " + str(target_size) +")")
        self.dim = target_size
        self.stdev = stdev
        img = imread(self.file_name.absolute(),as_gray=True)
        if (img.shape[0]!=img.shape[1]):
            raise Exception("Image is not square")
        self.target_image = resize(img, (self.dim, self.dim), anti_aliasing=False, preserve_range=True,
                            order=1, mode='symmetric')
        self.corrupted_image = self.target_image + self.stdev*np.random.randn(self.target_image.shape[0],self.target_image.shape[1])
        self.v = self.target_image.ravel(ORDER)
        self.y = self.corrupted_image.ravel(ORDER)
        self.num_sample = self.y.size
        temp = np.linspace(0.,1.,num=self.dim,endpoint=True)
        ty,tx = np.meshgrid(temp,temp)
        self.ty = ty.ravel(ORDER)
        self.tx = tx.ravel(ORDER)

    def get_measurement_matrix(self,ix,iy):
        H = u2.constructH(self.tx,self.ty,ix.ravel(ORDER),iy.ravel(ORDER))
        return H
        
        

class pCN():
    def __init__(self,n_layers,rg,measurement,f,beta=1,variant="dunlop"):
        self.n_layers = n_layers
        self.beta = beta
        self.betaZ = np.sqrt(1-beta**2)
        self.random_gen = rg
        self.measurement = measurement
        self.fourier = f
        self.variant=variant
        
        matrix_folder = pathlib.Path.cwd() /'matrices'
        if not matrix_folder.exists():
            matrix_folder.mkdir()

        measurement_matrix_file_name = 'plain_2D_measurement_matrix_{0}x{1}-{2}.npz'.format(str(self.fourier.basis_number),str(self.measurement.dim),ORDER)
        self.measurement_matrix_file = matrix_folder/measurement_matrix_file_name
        if not (self.measurement_matrix_file).exists():
            self.H = measurement.get_measurement_matrix(self.fourier.ix.ravel(ORDER),self.fourier.iy.ravel(ORDER))
            temp2 = self.H.conj().T@self.H
            self.H_t_H = 0.5*(temp2+temp2.conj().T).real
            np.savez_compressed(self.measurement_matrix_file,H=self.H.astype(np.complex64),H_t_H=self.H_t_H.astype(np.float64))
        else:
            self.H = np.load(self.measurement_matrix_file)['H']
            self.H_t_H = np.load(self.measurement_matrix_file)['H_t_H']

        self.I = np.eye(self.measurement.num_sample)
        self.y = self.measurement.y/self.measurement.stdev
        self.yBar = np.concatenate((self.y,np.zeros(2*self.fourier.basis_number_2D_ravel-1)))
        
    
        self.aggresiveness = 0.2
        self.target_acceptance_rate = 0.234
        self.beta_feedback_gain = 2.1
        self.record_skip = 1
        self.record_count = 0
        self.max_record_history = 10000      

        self.Layers_sqrtBetas = np.zeros(self.n_layers,dtype=np.float64)
        self.pcn_step_sqrtBetas = 1e-1
        self.stdev_sqrtBetas = np.ones(self.n_layers,dtype=np.float64)
        self.sqrtBetas_history = np.empty((10000, self.n_layers), dtype=np.float64)

    def adapt_beta(self,current_acceptance_rate):
        self.set_beta(self.beta*np.exp(self.beta_feedback_gain*(current_acceptance_rate-self.target_acceptance_rate)))

    def more_aggresive(self):
        self.set_beta(np.min(np.array([(1+self.aggresiveness)*self.beta,1],dtype=np.float64)))
    
    def less_aggresive(self):
        self.set_beta(np.min(np.array([(1-self.aggresiveness)*self.beta,1e-10],dtype=np.float64)))

    def set_beta(self,newBeta):
        self.beta = newBeta
        self.betaZ = np.sqrt(1-newBeta**2)
        
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
            Layers[i].new_sample = Layers[i].new_sample_sym[self.fourier.basis_number_2D_ravel-1:]
        meas_var = self.measurement.stdev**2
        # y = self.measurement.yt
        L = Layers[-1].LMat.current_L
        temp = L.conj().T@L
        r = 0.5*(temp+temp.conj().T) + self.H_t_H
        c = np.linalg.cholesky(r)
        Ht = np.linalg.solve(c,self.H.conj().T)
        
        R_inv = self.I - (Ht.conj().T@Ht).real
        logRatio = 0.5*(self.y@R_inv@self.y - np.linalg.slogdet(R_inv/meas_var)[1])

        L = Layers[-1].LMat.construct_from(Layers[-2].new_sample)
        temp = L.conj().T@L
        r = 0.5*(temp+temp.conj().T) + self.H_t_H
        c = np.linalg.cholesky(r)
        Ht = np.linalg.solve(c,self.H.conj().T)
        
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
            Layers[-1].new_sample = v[self.fourier.basis_number_2D_ravel-1:]
            for i in range(self.n_layers):
                Layers[i].update_current_sample()
                if not Layers[i].is_stationary:
                    Layers[i].LMat.set_current_L_to_latest()

        # self.one_step_for_sqrtBetas(Layers)
        if (self.record_count%self.record_skip) == 0:
            self.sqrtBetas_history[self.record_count,:] = self.Layers_sqrtBetas
            for i in range(self.n_layers):
                Layers[i].record_sample()

        self.record_count += 1

        return accepted

    def one_step_for_sqrtBetas(self,Layers):
        sqrt_beta_noises = self.stdev_sqrtBetas*np.random.randn(self.n_layers)
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
                    Layers[i].new_sample = Layers[i].new_sample_sym[self.fourier.basis_number_2D_ravel-1:]

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
                    Layers[i].stdev = Layers[i].stdev_sym[self.fourier.basis_number_2D_ravel-1:]

"""
Sample in this Layer object is always one complex dimensional vector
It is the job of Fourier object to convert it to 2D object
"""
class Layer():
    def __init__(self,is_stationary,sqrt_beta,order_number,n_samples,pcn,init_sample):
        self.is_stationary = is_stationary
        self.sqrt_beta = sqrt_beta
        self.order_number = order_number
        self.n_samples = n_samples
        self.pcn = pcn

        zero_compl_dummy =  np.zeros(self.pcn.fourier.basis_number_2D_ravel,dtype=np.complex128)
        ones_compl_dummy =  np.ones(self.pcn.fourier.basis_number_2D_ravel,dtype=np.complex128)

        self.stdev = ones_compl_dummy
        self.stdev_sym = util.symmetrize(self.stdev)
        self.samples_history = np.empty((self.n_samples, self.pcn.fourier.basis_number_2D_ravel), dtype=np.complex128)
    
        self.LMat = Lmatrix_2D(self.pcn.fourier,self.sqrt_beta)
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
            self.current_sample_scaled_norm = 0
            self.current_log_L_det = 0

        else:
            self.LMat.construct_from(init_sample)
            self.LMat.set_current_L_to_latest()
            self.new_sample_sym = np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())
            self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number_2D_ravel-1:]
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
            self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number_2D_ravel-1:]
            # return new_sample
        elif self.order_number == 0:
            self.new_sample = self.pcn.betaZ*self.current_sample + self.pcn.beta*self.stdev*self.pcn.random_gen.construct_w_half()
        else:
            self.new_sample_sym = self.pcn.betaZ*self.current_sample_sym + self.pcn.beta*np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())
            self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number_2D_ravel-1:]
            

    def sample_non_centered(self):
        self.new_noise_sample = self.pcn.betaZ*self.current_noise_sample+self.pcn.beta*self.pcn.random_gen.construct_w()

    def record_sample(self):
        self.samples_history[self.i_record,:] = self.current_sample.copy()
        self.i_record += 1
    
    def update_current_sample(self):
        self.current_sample = self.new_sample.copy()
        self.current_sample_sym = self.new_sample_sym.copy()
        self.current_sample_scaled_norm = self.new_sample_scaled_norm
        self.current_log_L_det = self.new_log_L_det
        self.current_noise_sample = self.new_noise_sample.copy()      
