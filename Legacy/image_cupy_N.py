from skimage.io import imread
from skimage.transform import resize
import scipy.linalg as sla
import warnings
import numpy as np
import scipy.linalg as sla
import os
import matplotlib.pyplot as plt
import pathlib
import cupy as cp
import mcmc.util_cupy as util
import mcmc.util_2D as u2
import scipy.special as ssp
import time
import gc
import h5py
import numba as nb
import cupyx as cpx
import cupyx.scipy.fftpack as cpxFFT
from cupy.prof import TimeRangeDecorator as cupy_profile
from mcmc.extra_linalg import solve_triangular,qr
ORDER = 'F'
IN_CPU_LSTSQ = False
from skimage.transform import radon
import numba.cuda as cuda
TPBn = 32#4*32
TPB = (TPBn,TPBn)

import abc #<-- abstract base class

class FourierAnalysis_2D:
    def __init__(self,basis_number,extended_basis_number,t_start = 0,t_end=1,mempool=None):
        self.basis_number = basis_number
        self.extended_basis_number = extended_basis_number
        self.basis_number_2D = (2*basis_number-1)*basis_number
        self.basis_number_2D_ravel = (2*basis_number*basis_number-2*basis_number+1)
        self.basis_number_2D_sym = (2*basis_number-1)*(2*basis_number-1)
        self.extended_basis_number_2D = (2*extended_basis_number-1)*extended_basis_number
        self.extended_basis_number_2D_sym = (2*extended_basis_number-1)*(2*extended_basis_number-1)
        self.t_end = t_end
        self.t_start = t_start
        self.verbose = True
        if mempool is None:
            mempool = cp.get_default_memory_pool()
        # pinned_mempool = cp.get_default_pinned_memory_pool()
        
        
        # self.ix = cp.zeros((2*self.basis_number-1,2*self.basis_number-1),dtype=cp.int32)
        # self.iy = cp.zeros((2*self.basis_number-1,2*self.basis_number-1),dtype=cp.int32)
        temp = cp.arange(-(self.basis_number-1),self.basis_number,dtype=cp.int32)
        # for i in range(2*self.basis_number-1):
        #     self.ix[i,:] = temp
        #     self.iy[:,i] = temp

        self.ix,self.iy = cp.meshgrid(temp,temp)
        if self.verbose:
            print("Used bytes so far, after creating ix and iy {}".format(mempool.used_bytes()))
        
        
        self.Ddiag = -(2*util.PI)**2*(self.ix.ravel(ORDER)**2+self.iy.ravel(ORDER)**2)
        self.OnePerDdiag = 1/self.Ddiag
        self.Dmatrix = cpx.scipy.sparse.diags(self.Ddiag,dtype=cp.float32)
        if self.verbose:
            print("Used bytes so far, after creating Dmatrix {}".format(mempool.used_bytes()))

        # self.Imatrix = cp.eye((2*self.basis_number-1)**2,dtype=cp.int8)
        self.Imatrix = cpx.scipy.sparse.identity((2*self.basis_number-1)**2,dtype=cp.float32)
        self.Imatrix_dense = cp.eye((2*self.basis_number-1)**2,dtype=cp.float32)
        if self.verbose:
            print("Used bytes so far, after creating Imatrix {}".format(mempool.used_bytes()))

        #K matrix --> 1D fourier index to 2D fourier index
        ones_temp = cp.ones_like(temp)
        self.Kmatrix = cp.vstack((cp.kron(temp,ones_temp),cp.kron(ones_temp,temp)))
        if self.verbose:
            print("Used bytes so far, after creating Kmatrix {}".format(mempool.used_bytes()))
        
        #implement fft plan here
        self._plan_fft2 = None
        # self._fft2_axes = (-2, -1)
        self._plan_ifft2 = None

        x = np.concatenate((np.arange(self.basis_number)+1,np.zeros(self.basis_number-1)))
        toep = sla.toeplitz(x)
        self._Umask = cp.asarray(np.kron(toep,toep),dtype=cp.int16)
        
    def rfft2(self,z):
        m = z.shape[0]
        if self._plan_fft2 is None:
            self._plan_fft2 = cpxFFT.get_fft_plan(z + 1j*cp.zeros_like(z), shape=z.shape)
        temp = cpxFFT.fft2(z.astype(cp.complex64),shape=z.shape,plan=self._plan_fft2,overwrite_x=True)
        temp = cp.fft.fftshift(temp,axes=0)
        return temp[m//2 -(self.basis_number-1):m//2 +self.basis_number,:self.basis_number ]/(2*self.extended_basis_number-1)
    
    def irfft2(self,uHalf2D):
        """
        Fourier transform of one dimensional signal
        ut   = 1D signal 
        num  = Ut length - 1
        dt   = timestep
        (now using cp.fft.fft) in the implementation
        """
        u2D = util.extend2D(util.symmetrize_2D(uHalf2D),self.extended_basis_number)
        if self._plan_ifft2 is None:
            self._plan_ifft2 = cpxFFT.get_fft_plan(u2D, shape=u2D.shape)   
        u2D = cp.fft.ifftshift(u2D)
        temp = cpxFFT.ifft2(u2D,shape=u2D.shape,plan=self._plan_ifft2,overwrite_x=True)
        return temp.real*(2*self.extended_basis_number-1)

    def constructU(self,uSym2D):
        """
        Construct Toeplitz Matrix
        """
        # return util.constructU(uHalf2D,self.Index)
        return util.constructU_from_uSym2D_cuda(uSym2D)
    

    def constructMatexplicit(self,uHalf2D,fun,m=0):
        # temp = fun(self.irfft2(uHalf2D)).T#ini kenapa harus di transpose ya?
        temp = fun(self.irfft2(uHalf2D))#Harusnya ga ditranspose!!
        temp = self.rfft2(temp)

        #symmetrize here
        temp = util.symmetrize_2D(temp)
        if m != 0:
            temp = util.shift_2D(temp,self.Kmatrix[:,m])

        return self.constructU(temp)
    
    # This implementation is problematic
    def constructMat(self,uHalf2D,power):
        U = self.constructU(util.symmetrize_2D(uHalf2D))
        exp_power_U = util.expm(power*U) #sla.expm(power*cp.asnumpy(U))
        # res = cp.asarray(exp_power_U,dtype=cp.complex64)*self._Umask
        res = exp_power_U*self._Umask
        return res


class Lmatrix_2D:
    def __init__(self,f,sqrt_beta):
        self.fourier = f
        self.sqrt_beta = sqrt_beta

        #initialize self.lastComputedL as zero
        # self.current = cp.zeros((self.fourier.basis_number_2D_sym,self.fourier.basis_number_2D_sym),dtype=cp.complex64)
        self.current = None
        self.latest_computed = self.current
        
    #@cupy_profile()
    def construct_from_2D(self,uHalf2D,hybrid=False):
        assert uHalf2D.shape[1] == self.fourier.basis_number
        if not hybrid:
            Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_min_nu)
            Ku_pow_d_per_2 = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_d_per_2)
            # Ku_pow_min_nu = self.fourier.constructMat(uHalf2D,1)
            # Ku_pow_d_per_2 = self.fourier.constructMat(uHalf2D,-1)
            L = ( Ku_pow_min_nu*self.fourier.Ddiag - Ku_pow_d_per_2)/self.sqrt_beta
        else:
            #mode ngirit
            temp = -self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_d_per_2)
            temp /= self.sqrt_beta
            temp_cp = cp.asnumpy(temp)
            del temp
            cp._default_memory_pool.free_all_blocks()
            temp = util.matMulti_sparse(self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_min_nu),self.fourier.Dmatrix)/self.sqrt_beta
            temp_cp += cp.asnumpy(temp) 
            
            del temp
            cp._default_memory_pool.free_all_blocks()

            L =  cp.asarray(temp_cp)
        self.latest_computed = L
        return L
    
    """
    for d = 2
    nu = 1
    2 - nu = 1
    kappa^(-nu) = kappa^(-1)
    kappa^(2-nu) = kappa^(d/2) = kappa
    """
    def construct_derivative_from_2D(self,uHalf2D,m,hybrid=False):
        assert uHalf2D.shape[1] == self.fourier.basis_number
        if not hybrid:
            Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_min_nu,m)
            Ku_pow_d_per_2 = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_d_per_2,m)
            dL = ( Ku_pow_min_nu*self.fourier.Ddiag + Ku_pow_d_per_2)/self.sqrt_beta #<-- here using plus sign
        else:
            #mode ngirit
            temp = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_d_per_2,m) #<-- here using plus sign
            temp /= self.sqrt_beta
            temp_cp = cp.asnumpy(temp)
            del temp
            cp._default_memory_pool.free_all_blocks()
            temp = util.matMulti_sparse(self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_min_nu,m),self.fourier.Dmatrix)/self.sqrt_beta
            temp_cp += cp.asnumpy(temp) 
            
            del temp
            cp._default_memory_pool.free_all_blocks()

            dL =  cp.asarray(temp_cp)
        
        return dL

    #@cupy_profile()
    def construct_from(self,u_sym,hybrid=False):
        uHalf2D = util.from_u_2D_ravel_to_uHalf_2D(u_sym,self.fourier.basis_number)
        return self.construct_from_2D(uHalf2D,hybrid)
    
    def construct_derivative_from(self,u_sym,m,hybrid=False):
        uHalf2D = util.from_u_2D_ravel_to_uHalf_2D(u_sym,self.fourier.basis_number)
        return self.construct_derivative_from_2D(uHalf2D,m,hybrid)
    
    #@cupy_profile()    
    def logDet(self,new):
        """
        # The determinant of a Hermitian matrix is real;the determinant is the product of the matrix's eigenvalues
        # L^dagger L is Hermitian
        """
        if new:
            return (cp.linalg.slogdet(self.latest_computed)[1])
        else:
            return (cp.linalg.slogdet(self.current)[1])

    #@cupy_profile()
    def set_current_to_latest(self):
        self.current = self.latest_computed
        
    #@cupy_profile()
    def is_current_equals_to_latest(self):
        return cp.all(self.current == self.latest_computed)

'''
N is L_inverse,
and propagating N perhaps cheaper than propagating L
Nmatrix class inherit Lmatrix class, except from the construct_from_2D
'''
class Nmatrix_2D(Lmatrix_2D):
    def construct_from_2D(self,uHalf2D,hybrid=False):
        assert uHalf2D.shape[1] == self.fourier.basis_number
        if not hybrid:
            Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_min_nu)
            Ku_pow_d_per_2 = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_d_per_2)
            L = (Ku_pow_min_nu*self.fourier.Ddiag - Ku_pow_d_per_2)/self.sqrt_beta            
            N = (cp.linalg.solve(L,self.fourier.Imatrix_dense))
        else:
            N = None #TODO: to be implemented
        self.latest_computed = N
        return N

    def construct_derivative_from_2D(self,uHalf2D,m,hybrid=False):
        assert uHalf2D.shape[1] == self.fourier.basis_number
        if not hybrid:
            Ku_pow_min_nu = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_min_nu,m)
            Ku_pow_d_per_2 = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_d_per_2,m)            
            N = ( Ku_pow_min_nu - self.fourier.self.OnePerDdiag*Ku_pow_d_per_2)*self.sqrt_beta
        else:
            #mode ngirit
            temp = self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_min_nu,m)*self.sqrt_beta
            temp_cp = cp.asnumpy(temp)
            del temp
            cp._default_memory_pool.free_all_blocks()
            temp = self.fourier.self.OnePerDdiag*(self.fourier.constructMatexplicit(uHalf2D,util.kappa_pow_d_per_2))*self.sqrt_beta
            temp_cp -= cp.asnumpy(temp)             
            del temp
            cp._default_memory_pool.free_all_blocks()

            N =  cp.asarray(temp_cp)
        self.latest_computed = N
        return N

class RandomGenerator_2D:
    def __init__(self,basis_number):
        # self.fourier = fourier
        self.basis_number = basis_number
        self.basis_number_2D_ravel = (2*basis_number*basis_number-2*basis_number+1)
        self.sqrt2 = util.SQRT2
    #@cupy_profile()
    # def construct_w_Half_2D(self):
    #     return util.construct_w_Half_2D(self.basis_number)

    #@cupy_profile()
    def construct_w_Half(self):
        return util.construct_w_Half(self.basis_number_2D_ravel)
    #@cupy_profile()
    def construct_w(self):
        w_half = self.construct_w_Half()
        w = self.symmetrize(w_half)
        return w
    #@cupy_profile()    
    def symmetrize(self,w_half):
        w = cp.concatenate((w_half[:0:-1].conj(),w_half)) #symmetrize
        # w = cp.zeros(2*w_half.shape[0]-1,dtype=cp.complex64)
        return w

    def symmetrize_2D(self,uHalf2D):
        return util.symmetrize_2D(uHalf2D)

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
        self.target_image = cp.asarray(self.target_image,dtype=cp.float32)
        self.corrupted_image = self.target_image + self.stdev*cp.random.randn(self.target_image.shape[0],self.target_image.shape[1],dtype=cp.float32)
        self.v = self.target_image.ravel(ORDER)/self.stdev #Normalized
        self.y = self.corrupted_image.ravel(ORDER)/self.stdev #Normalized
        self.num_sample = self.y.size
        temp = cp.linspace(0.,1.,num=self.dim,endpoint=True)
        tx,ty = cp.meshgrid(temp,temp)
        self.ty = ty.ravel(ORDER)
        self.tx = tx.ravel(ORDER)
    #@cupy_profile()
    def get_measurement_matrix(self,ix,iy):
        H = util.constructH(self.tx,self.ty,ix.ravel(ORDER),iy.ravel(ORDER))
        # H = cp.empty((self.tx.size,ix.size),dtype=np.complex64)
        # bpg=((H.shape[0]+TPBn-1)//TPBn,(H.shape[1]+TPBn-1)//TPBn)
        # print("Block Per Grid is equal to {0}, and H shape is {1}".format(bpg,H.shape))
        # util._construct_H[bpg,TPB](self.tx.ravel(ORDER),self.ty.ravel(ORDER),ix,iy,H)
        return H#.conj()#<-- yang di loadnya conjugatenya!!
        
class Sinogram(TwoDMeasurement):
    def __init__(self,file_name,target_size=128,n_theta=50,stdev=0.1,relative_location=""):
        super().__init__(file_name,target_size,stdev,relative_location)
        self.use_skimage = False
        self.n_theta = n_theta
        theta = cp.linspace(0., 180., self.n_theta, endpoint=False)
        self.set_theta(theta)
        
        #Kombinasi r-nya sudah benar, jangan diotak-atik lagi
        self.n_r = self.dim
        edge = 0.5#
        self.r = cp.linspace(-edge,edge,self.n_r, endpoint=False)
        self.r = self.r[::-1]
        
    
    def get_measurement_matrix(self,ix,iy):
        shifted_theta = self.theta+self.theta[1]
        theta_grid,r_grid = cp.meshgrid(shifted_theta*util.PI/180,self.r) #Because theta will be on the x axis, and r will be on the y axis
        
        # theta_grid = theta_grid[:,::-1]
        H = cp.zeros((r_grid.size,ix.size),dtype=cp.complex64)
        
        if not self.use_skimage:
            #launch CUDA kernel
            if cuda.is_available():
                bpg=((H.shape[0]+TPBn-1)//TPBn,(H.shape[1]+TPBn-1)//TPBn)
                print("Cuda is available, now running CUDA kernel with Thread perblock = {}, Block Per Grids = {}, and H shape {}".format(TPB,bpg,H.shape))
                util._calculate_H_Tomography[bpg,TPB](r_grid.ravel(ORDER),theta_grid.ravel(ORDER),ix,iy,H)
                ratio = self.n_r/(2*(self.n_r//2))
                H *= ratio

                #Some Temporary Fixing
                nPart=4
                n_theta = self.n_theta
                n_r = self.n_r
                for i in range(n_theta//nPart):
                    H[i*n_r:(i+1)*n_r,:] = cp.roll(H[i*n_r:(i+1)*n_r,:],1,axis=0)
            
            # util.calculate_H_Tomography(r_grid.ravel(ORDER),theta_grid.ravel(ORDER),ix,iy,H)
            #due to some problem the resulted H is flipped upside down
            #hence 
            # H = cp.flipud(H)
            # norm_ratio = (self.n_r/2)/(self.n_r//2)
        else:
            H_n = cp.asnumpy(H)
            util.calculate_H_Tomography_skimage(cp.asnumpy(self.theta),cp.asnumpy(ix),cp.asnumpy(iy),H_n,self.target_image.shape[0])
            H = cp.asarray(H_n)
        return H.astype(cp.complex64)
    
    def set_theta(self,new_theta):
        self.theta = new_theta
        self.pure_sinogram = cp.asarray(radon(cp.asnumpy(self.target_image),cp.asnumpy(self.theta),circle=True))
        self.sinogram = self.pure_sinogram + self.stdev*cp.random.randn(self.pure_sinogram.shape[0],self.pure_sinogram.shape[1])
        self.y = self.sinogram.ravel(ORDER)/self.stdev
        self.num_sample = self.y.size
        self.v = self.pure_sinogram.ravel(ORDER)/self.stdev


class vanilla_pCN():
    def __init__(self,n_layers,rg,measurement,f,beta=1,variant="dunlop",verbose=True,hybrid_mode=False,mempool=None):
        self.n_layers = n_layers
        self.beta = cp.float32(beta)
        self.betaZ = cp.sqrt(1-beta**2).astype(cp.float32)
        self.random_gen = rg
        self.measurement = measurement
        self.fourier = f
        self.variant=variant
        self.meas_var = self.measurement.stdev**2
        self.verbose = verbose
        self.hybrid_mode = hybrid_mode
        self.epsilon = 0
        self.cholesky_stabilizer = 0
        if mempool is None:
            mempool = cp.get_default_memory_pool()

        self.create_matrix_file_name()
        if not (self.measurement_matrix_file).exists():
            self.create_H_matrix()
            
            
        else:
            self.load_H_matrix()

        #do normalizing
        self.H /= self.measurement.stdev
        if self.verbose:
            print("Used bytes so far, after creating H {}".format(mempool.used_bytes()))
        # self.H_t_H = self.H.conj().T@self.H
        self.H_t_H /= self.meas_var
        self.I = cp.eye(self.measurement.num_sample,dtype=cp.float32)
        self.In = cp.eye(self.fourier.basis_number_2D_sym,dtype=cp.float32)
        # self.I = cpx.scipy.sparse.identity(self.measurement.num_sample)
        # self.In = cpx.scipy.sparse.identity(self.fourier.basis_number_2D_sym)
        
        self.y = self.measurement.y
        self.yBar = cp.concatenate((self.y,cp.zeros(2*self.fourier.basis_number_2D_ravel-1)))


        self.beta_feedback_gain = 2.1
        self.target_acceptance_rate = 0.234
        self.record_skip = 1
        self.record_count = 0
        self.max_record_history = 1000000


    def reset_chol_stabilizer(self):
        self.cholesky_stabilizer = self.epsilon*cp.linalg.norm(self.H_t_H)

    def set_chol_epsilon(self,cholEpsilon):
        self.epsilon = cholEpsilon
        self.reset_chol_stabilizer()

    def create_matrix_file_name(self):
        self.matrix_folder = pathlib.Path.cwd() /'matrices'
        if not self.matrix_folder.exists():
            self.matrix_folder.mkdir()

        if isinstance(self.measurement,Sinogram):
            measurement_matrix_file_name = 'radon_matrix_Four_Basis{}_Image_size {}_[{},{}]N_theta{}_{}.npz'.format(str(self.fourier.basis_number),str(self.measurement.dim),str(self.measurement.theta[0]),str(self.measurement.theta[-1]),str(self.measurement.n_theta),ORDER)
            if self.measurement.use_skimage:
                measurement_matrix_file_name += '_skimage'

            print('using Tomography Measurement')
        elif isinstance(self.measurement,TwoDMeasurement):
            measurement_matrix_file_name = 'plain_2D_measurement_matrix_{0}x{1}-{2}.npz'.format(str(self.fourier.basis_number),str(self.measurement.dim),ORDER)
            print('using TwoDMeasurement')
        self.measurement_matrix_file = self.matrix_folder/measurement_matrix_file_name
        
    def create_H_matrix(self):
        # load H and H_t_H in host memory instead
        if self.hybrid_mode:
            self.H = cp.asnumpy(self.measurement.get_measurement_matrix(self.fourier.ix.ravel(ORDER),self.fourier.iy.ravel(ORDER)))
            temp2 = self.H.conj().T@self.H
            self.H_t_H = 0.5*(temp2+temp2.conj().T)#.real
            np.savez_compressed(self.measurement_matrix_file,H=self.H.astype(np.complex64),H_t_H=self.H_t_H.astype(np.complex64)) #<-- save non normalized version
        else:
            self.H = self.measurement.get_measurement_matrix(self.fourier.ix.ravel(ORDER),self.fourier.iy.ravel(ORDER))
            temp2 = self.H.conj().T@self.H
            self.H_t_H = 0.5*(temp2+temp2.conj().T)#.real
            cp.savez_compressed(self.measurement_matrix_file,H=self.H.astype(cp.complex64),H_t_H=self.H_t_H.astype(cp.complex64)) #<-- save non normalized version
        
        

    def load_H_matrix(self):
        if self.hybrid_mode:
            #load H and H_t_H in host memory instead
            with np.load(self.measurement_matrix_file) as data:
                self.H = data['H']
                self.H_t_H = data['H_t_H']
        else:
            with cp.load(self.measurement_matrix_file) as data:
                self.H = data['H']
                if 'H_t_H' in data.npz_file.files:
                    self.H_t_H = data['H_t_H']
                else:
                    self.H_t_H = self.H.conj().T@self.H
        
        

    #@cupy_profile()
    def adapt_beta(self,current_acceptance_rate):
        self.set_beta(self.beta*cp.exp(self.beta_feedback_gain*(current_acceptance_rate-self.target_acceptance_rate)))

    #@cupy_profile()
    def set_beta(self,newBeta):
        if 1e-7<newBeta<1:
            self.beta = newBeta.astype(cp.float32)
            self.betaZ = cp.sqrt(1-newBeta**2).astype(cp.float32)
    
    def one_step(self,Layers):      
        self.record_count += 1
        return 1
    

'''
dunlop pCN implementation with Layers
'''
class dunlop_pCN(vanilla_pCN):
    def __init__(self,n_layers,rg,measurement,f,beta=1,verbose=True,mempool=None):
        super().__init__(n_layers,rg,measurement,f,beta=beta,variant="dunlop",verbose=verbose,hybrid_mode=False,mempool=mempool)
        self.__using_NMatrix = False
        self._log_ratio = self._log_ratio_L
        self.Ht = self.H.conj().T


    def use_NMatrix(self,useit=False):
        self.__using_NMatrix = useit
        if self.__using_NMatrix:
            self._log_ratio = self._log_ratio_N


    def isusing_NMatrix(self):
        return self.__using_NMatrix

    #Log ratio if using L matrix
    def _log_ratio_L(self,L):
        # temp = L.conj().T@L
        # epsilon is added to make sure that cholesky factorization working
        # r = 0.5*(temp+temp.conj().T) + self.H_t_H + self.cholesky_stabilizer*self.In
        # c = cp.linalg.cholesky(r)
        # Ht = cp.linalg.solve(c.astype(cp.complex64),self.H.conj().T)
        # Q_inv = self.I - (Ht.conj().T@Ht)
        # logRatio = 0.5*(self.y@Q_inv@self.y - cp.linalg.slogdet(Q_inv/self.meas_var)[1])     

        LinvHt = cp.linalg.solve(L.conj().T,self.Ht)
        Q = (LinvHt.conj().T@LinvHt+self.I).real
        logRatio = 0.5*(self.y@cp.linalg.solve(Q,self.y) + cp.linalg.slogdet(Q*self.meas_var)[1])
        return logRatio

    #log ratio if using N Matrix
    def _log_ratio_N(self,N):
        HN = self.H@N
        Q = (HN@HN.conj().T+self.I)
        yQy = self.y@(cp.linalg.solve(Q,self.y))
        logRatio = 0.5*(yQy.real + cp.linalg.slogdet(Q*self.meas_var)[1])
        return logRatio

    def one_step(self,Layers):      
        accepted = 0
        logRatio = 0.0
        for i in range(self.n_layers-1):
            Layers[i].sample_non_centered()
            if i>0:
                Layers[i].Mat.construct_from(Layers[i-1].new_sample_sym)

            Layers[i].new_sample_sym = Layers[i].centralize_sample(True)
            Layers[i].new_sample = Layers[i].new_sample_sym[self.fourier.basis_number_2D_ravel-1:]
        
        logRatio = self._log_ratio(Layers[-1].Mat.current)

        logRatio -= self._log_ratio(Layers[-1].Mat.construct_from(Layers[-2].new_sample_sym))
                    
        if logRatio>cp.log(cp.random.rand()):
            accepted = 1
            #sample the last layer
            Layers[-1].sample_non_centered()
            wNew = Layers[-1].new_noise_sample
            eNew = cp.random.randn(self.measurement.num_sample)
            wBar = cp.concatenate((eNew,wNew))

        
            HBar = Layers[-1].obtain_HBar(True)
            v, res, rnk, s = util.lstsq(HBar,self.yBar-wBar ,in_cpu=IN_CPU_LSTSQ)
            Layers[-1].new_sample_sym = v
            Layers[-1].new_sample = v[self.fourier.basis_number_2D_ravel-1:]

            for i in range(self.n_layers):
                Layers[i].update_current_sample()
                if not Layers[i].is_stationary:
                    Layers[i].Mat.set_current_to_latest()

        # self.one_step_for_sqrtBetas(Layers)
        if (self.record_count%self.record_skip) == 0:
            # self.sqrtBetas_history[self.record_count,:] = self.Layers_sqrtBetas
            for i in range(self.n_layers):
                Layers[i].record_sample()

        self.record_count += 1
        return accepted





"""
Sample in this Layer object is always one complex dimensional vector
It is the job of Fourier object to convert it to 2D object
"""
class Layer():
    def __init__(self,is_stationary,sqrt_beta,order_number,n_samples,pcn,init_sample_sym):
        self.preinit(is_stationary,sqrt_beta,order_number,n_samples,pcn,init_sample_sym)
        self.Mat = Lmatrix_2D(self.pcn.fourier,self.sqrt_beta)
        self.initialize_sample(init_sample_sym)

        # self.update_current_sample()
        

    def preinit(self,is_stationary,sqrt_beta,order_number,n_samples,pcn,init_sample_sym):
        self.is_stationary = is_stationary
        self.sqrt_beta = sqrt_beta
        self.order_number = order_number
        self.n_samples = n_samples
        self.pcn = pcn

        # zero_compl_dummy =  cp.zeros(self.pcn.fourier.basis_number_2D_ravel,dtype=cp.complex64)
        ones_compl_dummy =  cp.ones(self.pcn.fourier.basis_number_2D_ravel,dtype=cp.complex64)

        self.stdev = ones_compl_dummy
        self.stdev_sym = util.symmetrize(self.stdev)
        self.samples_history = np.empty((self.n_samples, self.pcn.fourier.basis_number_2D_ravel), dtype=np.complex64)
        self.current_noise_sample = self.pcn.random_gen.construct_w()#noise sample always symmetric
        self.new_noise_sample = self.current_noise_sample.copy()
        self.i_record = 0

        

    def centralize_sample(self,latest=False):
        if self.order_number == 0:
            return self.stdev_sym*self.new_noise_sample
        else:
            if not latest:
                return cp.linalg.solve(self.Mat.current,self.new_noise_sample)
            else:
                return cp.linalg.solve(self.Mat.latest_computed,self.new_noise_sample)

    def obtain_HBar(self,latest=False):
        if not latest:
            return cp.vstack((self.pcn.H,self.Mat.current))
        else:
            return cp.vstack((self.pcn.H,self.Mat.latest_computed))


    def initialize_sample(self,init_sample_sym):
        if self.is_stationary:
            self.new_sample_sym = init_sample_sym 
            self.new_sample = init_sample_sym[self.pcn.fourier.basis_number_2D_ravel-1:]
            self.current_sample = self.new_sample.copy()
            self.current_sample_sym = self.new_sample_sym.copy()
            
            self.new_sample_scaled_norm = 0
            self.new_log_Mat_det = 0
            #numba need this initialization. otherwise it will not compile
            self.current_sample_scaled_norm = 0
            self.current_log_Mat_det = 0

        else:
            init_sample = init_sample_sym[self.pcn.fourier.basis_number_2D_ravel-1:]
            self.Mat.construct_from(init_sample_sym,self.pcn.hybrid_mode)
            self.Mat.set_current_to_latest()
            self.sample_non_centered()
            self.new_sample_sym = self.centralize_sample()#cp.linalg.solve(self.Mat.current,self.pcn.random_gen.construct_w())
            self.new_sample = self.new_sample_sym[self.pcn.fourier.basis_number_2D_ravel-1:]
            self.new_sample_scaled_norm = util.norm2(self.Mat.current@self.new_sample_sym)#ToDO: Modify this
            self.new_log_Mat_det = self.Mat.logDet(True)#ToDO: Modify this
            
            self.current_sample = init_sample.copy()
            self.current_sample_sym = self.new_sample_sym.copy()
            self.current_sample_scaled_norm = self.new_sample_scaled_norm
            self.current_log_Mat_det = self.new_log_Mat_det
            
    #@cupy_profile()
    def sample_non_centered(self):
        self.new_noise_sample = self.pcn.betaZ*self.current_noise_sample+self.pcn.beta*self.pcn.random_gen.construct_w()
    #@cupy_profile()
    def record_sample(self):
        if self.i_record < self.samples_history.shape[0]:
            self.samples_history[self.i_record,:] = cp.asnumpy(self.current_sample,order=ORDER)
            self.i_record += 1
    #@cupy_profile()
    def update_current_sample(self):
        self.current_sample = self.new_sample.copy()
        self.current_sample_sym = self.new_sample_sym.copy()
        self.current_sample_scaled_norm = self.new_sample_scaled_norm
        self.current_log_Mat_det = self.new_log_Mat_det
        self.current_noise_sample = self.new_noise_sample.copy()      


'''
Layer inheritance, where Layer Matrix is now Nmatrix_2D
'''
class NLayer(Layer):
    def __init__(self,is_stationary,sqrt_beta,order_number,n_samples,pcn,init_sample_sym):
        self.preinit(is_stationary,sqrt_beta,order_number,n_samples,pcn,init_sample_sym)
        self.Mat = Nmatrix_2D(self.pcn.fourier,self.sqrt_beta)
        self.initialize_sample(init_sample_sym)

    #this is what differ with normal layer
    def centralize_sample(self,non_centered_sample_sym,latest=False):
        if self.order_number == 0:
            return self.stdev_sym*non_centered_sample_sym
        else:
            if not latest:
                return self.Mat.current@non_centered_sample_sym
            else:
                return self.Mat.latest_computed@non_centered_sample_sym
        

    def obtain_HBar(self,latest=False):
        if not latest:
            return cp.vstack((self.pcn.H,cp.linalg.solve(self.Mat.current,self.pcn.fourier.Imatrix_dense)))
        else:
            return cp.vstack((self.pcn.H,cp.linalg.solve(self.Mat.latest_computed,self.pcn.fourier.Imatrix_dense)))

        
        

    
            


class Simulation():
    def __init__(self,n_layers,n_samples,n,n_extended,beta,kappa,sigma_0,sigma_v,sigma_scaling,meas_std,evaluation_interval,printProgress,
                    seed,burn_percentage,enable_beta_feedback,pcn_variant,phantom_name,meas_type='tomo',
                    n_theta=50,verbose=False,hybrid_GPU_CPU=False,use_NMatrix=False,use_max_H=False):
        self.n_samples = n_samples
        self.evaluation_interval = evaluation_interval
        self.burn_percentage = burn_percentage
        #set random seed
        self.random_seed = seed
        self.printProgress = printProgress
        self.n_layers=n_layers
        self.kappa = kappa
        self.sigma_0 = sigma_0
        self.sigma_v = sigma_v
        self.sigma_scaling = sigma_scaling
        self.enable_beta_feedback = enable_beta_feedback
        self.verbose = verbose
        self.hybrid = hybrid_GPU_CPU
        self.mempool = None
        self.acceptancePercentage = 0
        cp.random.seed(self.random_seed)

        #CUPY memory management
        self.mempool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(self.mempool.malloc)
        
        #setup parameters for 2 Dimensional simulation
        
        self.d = 2
        self.nu = 2 - self.d/2
        self.alpha = self.nu + self.d/2
        self.t_start = -0.5
        self.t_end = 0.5
        self.beta_u = (sigma_0**2)*(2**self.d * util.PI**(self.d/2) * ssp.gamma(self.alpha))/ssp.gamma(self.nu)
        self.beta_v = self.beta_u*(sigma_v/sigma_0)**2
        self.sqrtBeta_v = cp.sqrt(self.beta_v).astype('float32')
        self.sqrtBeta_0 = cp.sqrt(self.beta_u).astype('float32')
        self.use_NMatrix = use_NMatrix
        self.use_max_H = use_max_H

        
        # pinned_mempool = cp.get_default_pinned_memory_pool()
        
        if self.verbose:
            print("Used bytes so far, before creating FourierAnalysis_2D {}".format(self.mempool.used_bytes())) 
        
        f =  FourierAnalysis_2D(n,n_extended,self.t_start,self.t_end,mempool=self.mempool)
        self.fourier = f
        
        if self.verbose:
            print("Used bytes so far, after creating FourierAnalysis_2D {}".format(self.mempool.used_bytes()))
        rg = RandomGenerator_2D(f.basis_number)
        self.random_gen = rg
        
        if self.verbose:
            print("Used bytes so far, after creating RandomGenerator_2D {}".format(self.mempool.used_bytes()))
        

        
        Lu_diag = ((f.Dmatrix.diagonal()*self.kappa**(-self.nu) - self.kappa**(2-self.nu)*f.Imatrix.diagonal())/self.sqrtBeta_0).astype('float32')
        
        # uStdev_sym = -1/cp.diag(Lu)
        uStdev_sym = -1/Lu_diag
        uStdev = uStdev_sym[f.basis_number_2D_ravel-1:]
        uStdev[0] /= 2 #scaled

        if self.use_max_H:
            target_size = 511
        else:
            target_size = 2*f.extended_basis_number-1

        if meas_type == 'tomo':
            self.measurement = Sinogram(phantom_name,target_size=target_size,n_theta=n_theta,stdev=meas_std,relative_location='phantom_images')
            self.measurement.use_skimage = False
        else:
            self.measurement = TwoDMeasurement(phantom_name,target_size=target_size,stdev=meas_std,relative_location='phantom_images')
        
        if self.verbose:
            print("Used bytes so far, after creating measurement {}".format(self.mempool.used_bytes()))

        self.pcn_variant = pcn_variant
        self.pcn = dunlop_pCN(n_layers,rg,self.measurement,f,beta,verbose=self.verbose,mempool=self.mempool) #TODO:This is BAD
        self.pcn.use_NMatrix(self.use_NMatrix)
        if self.verbose:
            print("Used bytes so far, after creating pCN {}".format(self.mempool.used_bytes()))
        
        
        self.pcn.record_skip = np.max(cp.array([1,self.n_samples//self.pcn.max_record_history]))
        history_length = np.min(np.array([self.n_samples,self.pcn.max_record_history])) 
        

        Layers = []
        for i in range(self.n_layers):
            if i==0:
                init_sample_sym = uStdev_sym*self.pcn.random_gen.construct_w()
                if self.pcn.isusing_NMatrix():
                    lay = NLayer(True, self.sqrtBeta_0,i, n_samples, self.pcn,init_sample_sym)
                else:
                    lay = Layer(True, self.sqrtBeta_0,i, n_samples, self.pcn,init_sample_sym)
                lay.Mat.current = None
                lay.Mat.latest_computed = None
                lay.stdev_sym = uStdev_sym
                lay.stdev = uStdev
            elif i == n_layers-1:
                if self.pcn.isusing_NMatrix():
                    lay = NLayer(False, self.sqrtBeta_v,i, self.n_samples, self.pcn,Layers[i-1].current_sample_sym)
                else:
                    lay = Layer(False, self.sqrtBeta_v,i, self.n_samples, self.pcn,Layers[i-1].current_sample_sym)
                
                wNew =  self.pcn.random_gen.construct_w()
                eNew = cp.random.randn(self.pcn.measurement.num_sample,dtype=cp.float32)
                wBar = cp.concatenate((eNew,wNew))
                
                
                HBar = lay.obtain_HBar()
                lay.current_sample_sym, res, rnk, s = util.lstsq(HBar, self.pcn.yBar-wBar)#,rcond=None)
                lay.current_sample = lay.current_sample_sym[f.basis_number_2D_ravel-1:]
                    
            else:
                lay = Layer(False, self.sqrtBeta_v*self.sigma_scaling,i, self.n_samples, self.pcn,Layers[i-1].current_sample_sym)

            lay.update_current_sample()
            
            lay.samples_history = np.empty((history_length, self.pcn.fourier.basis_number_2D_ravel), dtype=np.complex64)
            Layers.append(lay)
            if self.verbose:
                print("Used bytes so far, after creating Layer {} {}".format(i,self.mempool.used_bytes()))

        self.Layers = Layers
    
    #@cupy_profile()
    def run(self):
        self.accepted_count = 0
        self.accepted_count_SqrtBeta = 0
        average_time_intv =0.0
        print(self.pcn_variant)
        if self.printProgress:
            util.printProgressBar(0, self.n_samples, prefix = 'Preparation . . . . ', suffix = 'Complete', length = 50)
        start_time = time.time()
        start_time_intv = start_time
        
        accepted_count_partial = 0
        linalg_error_occured = False
        

        for i in range(self.n_samples):#nb.prange(nSim):
            try:
                accepted_count_partial += self.pcn.one_step(self.Layers)
            except np.linalg.LinAlgError as err:
                linalg_error_occured = True
                print("Linear Algebra Error :",err)
                break
                # continue
            else:
                if (i+1)%(self.evaluation_interval) == 0:
                    self.accepted_count += accepted_count_partial
                    self.acceptancePercentage = self.accepted_count/(i+1)
                        
                    if self.enable_beta_feedback:
                        self.pcn.adapt_beta(self.acceptancePercentage)
                    
                    accepted_count_partial = 0
                    mTime = (i+1)/(self.evaluation_interval)
                    
                    end_time_intv = time.time()
                    time_intv = end_time_intv-start_time_intv
                    average_time_intv +=  (time_intv-average_time_intv)/mTime
                    start_time_intv = end_time_intv
                    remainingTime = average_time_intv*((self.n_samples - i)/self.evaluation_interval)
                    remainingTimeStr = time.strftime("%j-1 day(s),%H:%M:%S", time.gmtime(remainingTime))
                    if self.printProgress:
                        util.printProgressBar(i+1, self.n_samples, prefix = 'Time Remaining {0}- Acceptance Rate {1:.2%} - Progress:'.format(remainingTimeStr,self.acceptancePercentage), suffix = 'Complete', length = 50)
            

        if linalg_error_occured:
            if self.printProgress:
                #truncating the each layer history
                end_index = max(i%self.pcn.record_skip,1)
                for l in range(self.n_layers):
                    self.Layers[l].samples_history = self.Layers[l].samples_history[:end_index,:]
                print('Linear algebra errors occured during some simulation step(s). The simulation result may not be valid')
        else:
            elapsedTimeStr = time.strftime("%j day(s),%H:%M:%S", time.gmtime(time.time()-start_time))
        self.total_time = time.time()-start_time
        if self.printProgress:
            util.printProgressBar(self.n_samples, self.n_samples, 'Iteration Completed in {0}- Acceptance Rate {1:.2%} - Progress:'.format(elapsedTimeStr,self.acceptancePercentage), suffix = 'Complete', length = 50)
    

    def save(self,file_name,include_history=False):
        with h5py.File(file_name,'w') as f:
            util._save_object(f,self)


