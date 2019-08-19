from skimage.io import imread
from skimage.transform import radon, resize
import warnings
import numpy as np
import pywt
import scipy.interpolate as interpolate
from scipy.optimize import minimize
from scipy.signal import correlate
import time
import math
import sys
import scipy.sparse as sp
import scipy.linalg as sla
import os
import matplotlib.pyplot as plt
import argparse
import pathlib

import numba as nb
from skimage.transform._warps_cy import _warp_fast
import mcmc.util as util
import mcmc.util_2D as u2
import mcmc.L as L
import mcmc.fourier as fourier
import mcmc.randomGenerator as randomGenerator
FASTMATH=True
PARALLEL = True
CACHE=True
njitSerial = nb.njit(fastmath=FASTMATH,cache=CACHE)
jitSerial = nb.jit(fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)
jitParallel = nb.jit(fastmath=FASTMATH,cache=CACHE,parallel=PARALLEL)

class Tomograph:
    def __init__(self,file_name,fourier,target_size=128,n_theta=50,meas_std=0.1,relative_location=""):
        self.globalprefix = pathlib.Path.cwd() / relative_location
        if not self.globalprefix.exists():
            self.globalprefix.mkdir()
        self.file_name= self.globalprefix / file_name
        if target_size>512:
            raise Exception("Target image dimension are too large (" + str(target_size) + " x " + str(target_size) +")")
        self.dim = target_size
        self.meas_std = meas_std
        img = imread(self.file_name.absolute(),as_gray=True)
        if (img.shape[0]!=img.shape[1]):
            raise Exception("Image is not square")
        self.target_image = resize(img, (self.dim, self.dim), anti_aliasing=False, preserve_range=True,
                            order=1, mode='symmetric')
        self.flattened_target_image = self.target_image.ravel()
        self.image_num_points = self.flattened_target_image.shape[0]
        self.n_theta = n_theta
        self.theta = np.linspace(0., 180., self.n_theta, endpoint=False)
        self.n_r = math.ceil(np.sqrt(2)*self.dim)
        self.r = np.linspace(-self.n_r/2,self.n_r/2,self.n_r)
        temp = np.linspace(0,1,num=self.dim,endpoint=True)
        ty,tx = np.meshgrid(temp,temp)
        self.ty = ty
        self.tx = tx
        #TODO: modify this
        self.fourier = fourier
        self.basis_number = self.fourier.basis_number
        iy,ix = np.meshgrid(np.arange(-(self.basis_number-1),self.basis_number),np.arange(-(self.basis_number-1),self.basis_number))
        self.ix = ix
        self.iy = iy

        
        matrix_folder = pathlib.Path.cwd() /'matrices'
        if not matrix_folder.exists():
            matrix_folder.mkdir()
        
        radon_matrix_file_name = 'radon_matrix_{0}x{1}.npz'.format(str(self.dim), str(self.theta.shape[0]))
        self.radon_matrix_file = matrix_folder / radon_matrix_file_name
        if not self.radon_matrix_file.exists():
            from .matrices import radonmatrix
            self.radonoperator = radonmatrix(self.dim,self.theta*np.pi/180).T
            # self.radonoperator = radonmatrix(self.dim,self.theta)
            sp.save_npz(str(self.radon_matrix_file),self.radonoperator)
        else:
            self.radonoperator = sp.load_npz(self.radon_matrix_file)
            # self.radonoperator = sp.load_npz(matrix_file)

        self.radonoperator = sp.csc_matrix(self.radonoperator)
        self.radonoperator = self.radonoperator / self.dim

        self.pure_vector = self.flattened_target_image@self.radonoperator
        # self.pure_vector = self.radonoperator@self.flattened_target_image
        self.vector = self.pure_vector + np.max(self.pure_vector)*self.meas_std*np.random.randn(self.n_r*self.theta.shape[0])
        self.sinogram = np.reshape(self.vector,(self.n_r,self.n_theta))
        self.pure_sinogram = np.reshape(self.pure_vector,(self.n_r,self.n_theta))
        self.sinogram_flattened = self.sinogram.ravel()
        self.pure_sinogram_flattened = self.pure_sinogram.ravel()


        measurement_matrix_file_name = 'measurement_matrix_{0}_{1}x{2}.npz'.format(str(self.basis_number),str(self.dim), str(self.theta.shape[0]))
        self.measurement_matrix_file = matrix_folder/measurement_matrix_file_name
        if not (self.measurement_matrix_file).exists():
            self.H = self.constructH()
            # self.H = H.astype(np.complex64)
            np.savez_compressed(self.measurement_matrix_file,H=self.H.astype(np.complex64))
            # self.H = sp.csc_matrix(H)
            # sp.save_npz(measurement_matrix_file,self.H)
        else:
            self.H = np.load(self.measurement_matrix_file)['H']
            # self.H = sp.load_npz(measurement_matrix_file)




    def constructH(self):
        return _constructH(self.radonoperator,self.n_r,self.n_theta,self.tx.ravel(),self.ty.ravel(),self.ix.ravel(),self.iy.ravel())/self.meas_std
        # return _constructH(self.theta,self.r,self.tx.ravel(),self.ty.ravel(),self.ix.ravel(),self.iy.ravel())/self.meas_std

@jitParallel
def _constructH(radonoperator,n_r,n_theta,tx,ty,ix,iy):
    """
    (iX,iY) are meshgrid for Fourier Index
    (tx,ty) also ravelled meshgrid for original location grid (0 to 1)
    """
    H = np.empty((ix.shape[0],tx.shape),dtype=np.complex64)
    for i in nb.prange(ix.shape[0]):
        #TODO: this is point measurement, change this to a proper H
        # eigenSlice = u2.eigenFunction2D(tx,ty,ix[i],iy[i]).ravel()
        # H[i,:] = eigenslice@radonoperator
        H[i,:] = u2.eigenFunction2D(tx,ty,ix[i],iy[i]).ravel()
    return H.T


class pCN():
    def __init__(self,n_layers,rg,tomograph,f,beta=1):
        self.n_layers = n_layers
        self.beta = beta
        self.betaZ = np.sqrt(1-beta**2)
        self.random_gen = rg
        self.tomograph = tomograph
        self.fourier = f
        self.H = self.tomograph.H
        self.yBar = np.concatenate((self.tomograph.sinogram_flattened/self.tomograph.meas_std,np.zeros(self.fourier.basis_number_2D_sym)))
        self.aggresiveness = 0.2
        self.target_acceptance_rate = 0.234
        self.beta_feedback_gain = 2.1
        self.record_skip = 1
        self.record_count = 0
        self.max_record_history = 1000000
        
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
        # only record when needed
        if (self.record_count%self.record_skip) == 0:
            # print('recorded')
            for i in range(self.n_layers):
                Layers[i].record_sample()
        self.record_count += 1

        return accepted



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
        zero_compl_dummy =  np.zeros(self.pcn.fourier.basis_number_2D,dtype=np.complex128)
        ones_compl_dummy =  np.ones(self.pcn.fourier.basis_number_2D,dtype=np.complex128)

        self.stdev = ones_compl_dummy
        self.samples_history = np.empty((self.n_samples, self.pcn.fourier.basis_number_2D), dtype=np.complex128)
        
        
        self.LMat = L.Lmatrix_2D(self.pcn.fourier,self.sqrt_beta)
        
        
        if self.is_stationary:
            self.new_sample = init_sample
            self.new_sample_symmetrized = self.pcn.random_gen.symmetrize_2D(self.new_sample)
            self.new_sample_scaled_norm = 0
            self.new_log_L_det = 0
            #numba need this initialization. otherwise it will not compile
            self.current_sample = init_sample.copy()
            self.current_sample_symmetrized = self.new_sample_symmetrized.copy()
            self.current_sample_scaled_norm = 0
            self.current_log_L_det = 0
            

        else:
            zero_init = np.zeros(self.pcn.fourier.basis_number,dtype=np.complex128)
            self.LMat.construct_from(init_sample)
            self.LMat.set_current_L_to_latest()
            self.new_sample_symmetrized = np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w_2D_ravelled())
            self.new_sample = self.new_sample_symmetrized[self.pcn.fourier.basis_number-1:]
            self.new_sample_scaled_norm = util.norm2(self.LMat.current_L@self.new_sample_symmetrized)#ToDO: Modify this
            self.new_log_L_det = self.LMat.logDet()#ToDO: Modify this
            # #numba need this initialization. otherwise it will not compile
            self.current_sample = init_sample.copy()
            self.current_sample_symmetrized = self.new_sample_symmetrized.copy()
            self.current_sample_scaled_norm = self.new_sample_scaled_norm
            self.current_log_L_det = self.new_log_L_det   
            
        # self.update_current_sample()
        self.i_record = 0


    def sample(self):
        #if it is the last layer
        if self.order_number == self.pcn.n_layers -1:
            wNew = self.pcn.random_gen.construct_w_Half_2D_ravelled()
            eNew = np.random.randn(self.pcn.tomograph.image_num_points)
            wBar = np.concatenate((eNew,wNew))
            
            LBar = np.vstack((self.pcn.H,self.LMat.current_L))

            #update v
            self.new_sample_symmetrized, res, rnk, s = np.linalg.lstsq(LBar,self.pcn.yBar-wBar )#,rcond=None)
            self.new_sample = self.new_sample_symmetrized[self.pcn.fourier.basis_number-1:]
            # return new_sample
        elif self.order_number == 0:
            self.new_sample = self.pcn.betaZ*self.current_sample + self.pcn.beta*self.stdev*self.pcn.random_gen.construct_w_half()
            # self.new_sample_symmetrized = self.pcn.random_gen.symmetrize(self.new_sample) 
        else:
            self.new_sample_symmetrized = self.pcn.betaZ*self.current_sample_symmetrized + self.pcn.beta*np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())
            # self.new_sample_symmetrized = np.linalg.solve(self.LMat.current_L,self.pcn.random_gen.construct_w())
            self.new_sample = self.new_sample_symmetrized[self.pcn.fourier.basis_number-1:]
    def record_sample(self):
        self.samples_history[self.i_record,:] = self.current_sample.copy()
        self.i_record += 1
    
    def update_current_sample(self):
        self.current_sample = self.new_sample.copy()
        self.current_sample_symmetrized = self.new_sample_symmetrized.copy()
        self.current_sample_scaled_norm = self.new_sample_scaled_norm
        self.current_log_L_det = self.new_log_L_det    





