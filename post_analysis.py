# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:31:38 2019

@author: Muhammad
"""

import h5py
import mcmc.util_cupy as util
import matplotlib.pyplot as plt
import cupy as cp, numpy as np
import mcmc.image_cupy as imc
import pathlib

SimulationResult_dir = pathlib.Path("//data.triton.aalto.fi/work/emzirm1/SimulationResult/")
SimulationResult_dir = SimulationResult_dir /'result-20-Sep-2019_10_50'
file = h5py.File(str(SimulationResult_dir/'result.hdf5'),mode='r')
samples_history = file['Layers 1/samples_history'][()]
u_samples_history = file['Layers 0/samples_history'][()]
mean_field = util.symmetrize(cp.asarray(np.mean(samples_history,axis=0)))
u_mean_field = util.symmetrize(cp.asarray(np.mean(u_samples_history,axis=0)))
stdev_field = util.symmetrize(cp.asarray(np.std(samples_history,axis=0)))
n = file['fourier/basis_number'][()]
n_ext = file['fourier/extended_basis_number'][()]
t_start = file['t_start'][()]
t_end = file['t_end'][()]
target_image = file['measurement/target_image'][()]
corrupted_image = file['measurement/corrupted_image'][()]
fourier = imc.FourierAnalysis_2D(n,n_ext,t_start,t_end)
sL2 = util.sigmasLancosTwo(32)

vF = mean_field.reshape(2*n-1,2*n-1,order=imc.ORDER).T
uF = u_mean_field.reshape(2*n-1,2*n-1,order=imc.ORDER).T
vForiginal = sL2*util.symmetrize_2D(fourier.fourierTransformHalf(cp.array(target_image)))

vFn = cp.asnumpy(vF)
uFn = cp.asnumpy(uF)
reconstructed_image = fourier.inverseFourierLimited(vF[:,n-1:])
scalling_factor = 1/(cp.max(reconstructed_image)-cp.min(reconstructed_image))
u_image = fourier.inverseFourierLimited(uF[:,n-1:])
reconstructed_image_original = fourier.inverseFourierLimited(vForiginal[:,n-1:])

ri_n = cp.asnumpy(reconstructed_image)
ri_or_n = cp.asnumpy(reconstructed_image_original)
ri_n_scalled = ri_n/(np.max(ri_n)-np.min(ri_n))
u_n = cp.asnumpy(u_image/(np.max(ri_n)-np.min(ri_n)))
ell_n =np.exp(u_n)
fig, ax = plt.subplots(ncols=3,nrows=3,figsize=(15,15))
im = ax[0,0].imshow(ri_n_scalled,cmap=plt.cm.Greys_r);ax[0,0].set_title('Reconstructed Image From vF: RI')
im = ax[0,1].imshow(target_image,cmap=plt.cm.Greys_r);ax[0,1].set_title('Target Image : TI')
im = ax[0,2].imshow(ri_or_n,cmap=plt.cm.Greys_r);ax[0,2].set_title('Reconstructed Image From vFOriginal :RIO')
im = ax[1,0].imshow(np.abs(target_image-ri_n_scalled),cmap=plt.cm.Greys_r);ax[1,0].set_title('Absolute error:RI-TI')
im = ax[1,1].imshow(np.abs(ri_or_n-target_image),cmap=plt.cm.Greys_r);ax[1,1].set_title('Absolute error:RIO-TI')
im = ax[1,2].imshow(np.abs(ri_n_scalled-ri_or_n),cmap=plt.cm.Greys_r);ax[1,2].set_title('Absolute error:RI-RIO')
im = ax[2,0].imshow(u_n,cmap=plt.cm.Greys_r);ax[2,0].set_title('Field u:u')
im = ax[2,1].imshow(ell_n,cmap=plt.cm.Greys_r);ax[2,1].set_title('Length Scale of v:ell')
im = ax[2,2].imshow(corrupted_image,cmap=plt.cm.Greys_r);ax[2,2].set_title('Measurement (corrupted_image) :CI')
fig.savefig(str(SimulationResult_dir/'Result.pdf'), bbox_inches='tight')
