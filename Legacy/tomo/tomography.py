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
        # self.n_r = math.ceil(np.sqrt(2)*self.dim)
        self.n_r = self.dim
        # self.r = np.linspace(-self.n_r/2,self.n_r/2,self.n_r)
        self.r = np.linspace(-0.5,0.5,self.n_r)
        temp = np.linspace(-0.5,0.5,num=self.dim,endpoint=True)
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
        
        # radon_matrix_file_name = 'radon_matrix_{0}x{1}.npz'.format(str(self.dim), str(self.theta.shape[0]))
        # self.radon_matrix_file = matrix_folder / radon_matrix_file_name
        # if not self.radon_matrix_file.exists():
        #     # print(os.getcwd())
        #     from .matrices import radonmatrix
        #     self.radonoperator = radonmatrix(self.dim,self.theta*np.pi/180).T
        #     # self.radonoperator = radonmatrix(self.dim,self.theta)
        #     sp.save_npz(str(self.radon_matrix_file),self.radonoperator)
        # else:
        #     self.radonoperator = sp.load_npz(self.radon_matrix_file)
        #     # self.radonoperator = sp.load_npz(matrix_file)

        # self.radonoperator = sp.csc_matrix(self.radonoperator)
        # self.radonoperator = self.radonoperator / self.dim
        self.pure_sinogram = radon(self.target_image,self.theta,circle=True)
        self.sinogram = self.pure_sinogram + self.meas_std*np.random.randn(self.pure_sinogram.shape[0],self.pure_sinogram.shape[1])
        # self.pure_vector =  self.pure_sinogram.ravel()#self.flattened_target_image@self.radonoperator
        # self.pure_vector = self.radonoperator@self.flattened_target_image
        # self.vector = 
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
        # return _constructH(self.radonoperator,self.n_r,self.n_theta,self.tx.ravel(),self.ty.ravel(),self.ix.ravel(),self.iy.ravel())/self.meas_std
        theta_grid,r_grid = np.meshgrid(self.theta*np.pi/180,self.r)
        return _constructH(r_grid.ravel(),theta_grid.ravel(),self.ix.ravel(),self.iy.ravel())/self.meas_std

@njitParallel
def _constructH(r,theta,kx,ky):
    """
    (iX,iY) are meshgrid for Fourier Index
    (tx,ty) also ravelled meshgrid for original location grid (0 to 1)
    """
    H = np.empty((kx.shape[0],r.shape[0]),dtype=np.complex64)
    for m in nb.prange(kx.shape[0]):
        for n in nb.prange(r.shape[0]):
            #TODO: this is point measurement, change this to a proper H
            sTheta = np.sin(theta[n])
            cTheta = np.cos(theta[n])
            k_tilde_u = kx[m]*cTheta+ky[m]*sTheta
            k_tilde_v = -kx[m]*sTheta+ky[m]*cTheta
            l = np.sqrt(0.25-r[n]**2)
            if k_tilde_v != 0:
                H[m,n] = np.exp(1j*2*np.pi*k_tilde_u*r[n])*(np.sin(2*np.pi*k_tilde_v*l))/(np.pi*k_tilde_v)
            else:
                H[m,n] = np.exp(1j*2*np.pi*k_tilde_u*r[n])*(2*l)
    return H.T



        





