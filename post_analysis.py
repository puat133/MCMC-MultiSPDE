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
# import mcmc.measurement as mcmcMeas
from skimage.transform import iradon
import pathlib
import seaborn as sns
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import argparse
image_extension = '.png'

def _process_data(samples_history,u_samples_history,n,n_ext,t_start,t_end,target_image,corrupted_image,burn_percentage,isSinogram,sinogram,theta,fbp,SimulationResult_dir,result_file,cmap = plt.cm.seismic_r):
    burn_start_index = np.int(0.01*burn_percentage*u_samples_history.shape[0])
    
    #initial conditions
    samples_init = samples_history[0,:]

    #change
    u_samples_history = u_samples_history[burn_start_index:,:]
    samples_history = samples_history[burn_start_index:,:]
    N = u_samples_history.shape[0]
    
    #initial condition
    vF_init = util.symmetrize(cp.asarray(samples_init)).reshape(2*n-1,2*n-1,order=imc.ORDER)
    # vF_init = vF_init.conj()
    

    vF_mean = util.symmetrize(cp.asarray(np.mean(samples_history,axis=0)))
    vF_stdev = util.symmetrize(cp.asarray(np.std(samples_history,axis=0)))
    vF_abs_stdev = util.symmetrize(cp.asarray(np.std(np.abs(samples_history),axis=0)))
   
    
    fourier = imc.FourierAnalysis_2D(n,n_ext,t_start,t_end)
    sL2 = util.sigmasLancosTwo(cp.int(n))
    
    # if isSinogram:
    #     vF_init = util.symmetrize_2D(fourier.rfft2(cp.asarray(fbp,dtype=cp.float32)))
    
    
#    if not isSinogram:
    vForiginal = util.symmetrize_2D(fourier.rfft2(cp.array(target_image)))
    reconstructed_image_original = fourier.irfft2(vForiginal[:,n-1:])
    reconstructed_image_init = fourier.irfft2(vF_init[:,n-1:])
    
    samples_history_cp = cp.asarray(samples_history)
    v_image_count=0
    v_image_M = cp.zeros_like(reconstructed_image_original)
    v_image_M2 = cp.zeros_like(reconstructed_image_original)
    v_image_aggregate = (v_image_count,v_image_M,v_image_M2)
    for i in range(N):
        vF = util.symmetrize(samples_history_cp[i,:]).reshape(2*n-1,2*n-1,order=imc.ORDER)
        v_temp = fourier.irfft2(vF[:,n-1:])
        v_image_aggregate = util.updateWelford(v_image_aggregate,v_temp)
        
    
    
    v_image_mean,v_image_var,v_image_s_var = util.finalizeWelford(v_image_aggregate)
    
    #TODO: This is sign of wrong processing, Remove this
    # if isSinogram:
    #     reconstructed_image_init = cp.fliplr(reconstructed_image_init)
    #     v_image_mean = cp.fliplr(v_image_mean)
    #     v_image_s_var = cp.fliplr(v_image_s_var)
    
    mask = cp.zeros_like(reconstructed_image_original)
    r = (mask.shape[0]+1)//2
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            x = 2*(i - r)/mask.shape[0]
            y = 2*(j - r)/mask.shape[1]
            if (x**2+y**2 < 1):
                mask[i,j]=1.
    
    u_samples_history_cp = cp.asarray(u_samples_history)
    u_image = cp.zeros_like(v_image_mean)
    # ell_image = cp.zeros_like(v_image_mean)
    
    u_image_count=0
    u_image_M = cp.zeros_like(u_image)
    u_image_M2 = cp.zeros_like(u_image)
    u_image_aggregate = (u_image_count,u_image_M,u_image_M2)
    ell_image_count=0
    ell_image_M = cp.zeros_like(u_image)
    ell_image_M2 = cp.zeros_like(u_image)
    ell_image_aggregate = (ell_image_count,ell_image_M,ell_image_M2)
    for i in range(N):
        uF = util.symmetrize(u_samples_history_cp[i,:]).reshape(2*n-1,2*n-1,order=imc.ORDER)
        u_temp = fourier.irfft2(uF[:,n-1:])
        u_image_aggregate = util.updateWelford(u_image_aggregate,u_temp)
        ell_temp = cp.exp(u_temp)
        ell_image_aggregate = util.updateWelford(ell_image_aggregate, ell_temp)
    u_image_mean,u_image_var,u_image_s_var = util.finalizeWelford(u_image_aggregate)
    ell_image_mean,ell_image_var,ell_image_s_var = util.finalizeWelford(ell_image_aggregate)

    
    # if isSinogram:
        # u_image_mean = cp.flipud(u_image_mean) #cp.rot90(cp.fft.fftshift(u_image),1) 
        # u_image_var = cp.flipud(u_image_var) #cp.rot90(cp.fft.fftshift(u_image),1) 
        # ell_image_mean = cp.flipud(ell_image_mean)# cp.rot90(cp.fft.fftshift(ell_image),1) 
        # ell_image_var = cp.flipud(ell_image_var)# cp.rot90(cp.fft.fftshift(ell_image),1) 
        
    ri_fourier = cp.asnumpy(reconstructed_image_original)
    
    if isSinogram:
        ri_compare = fbp
    else:
        ri_compare = ri_fourier
   
    is_masked=True
    if is_masked:
        reconstructed_image_var = mask*v_image_s_var
        reconstructed_image_mean = mask*v_image_mean
        reconstructed_image_init = mask*reconstructed_image_init
        u_image_mean = mask*u_image_mean #cp.rot90(cp.fft.fftshift(u_image),1) 
        u_image_s_var = mask*u_image_s_var #cp.rot90(cp.fft.fftshift(u_image),1) 
        ell_image_mean = mask*ell_image_mean# cp.rot90(cp.fft.fftshift(ell_image),1) 
        ell_image_s_var = mask*ell_image_s_var# cp.rot90(cp.fft.fftshift(ell_image),1) 
    else:
        reconstructed_image_mean = v_image_mean        
    
    
    ri_init = cp.asnumpy(reconstructed_image_init)
    
    # ri_fourier = fourier.irfft2((sL2.astype(cp.float32)*vForiginal)[:,n-1:])
    vForiginal_n = cp.asnumpy(vForiginal)
    vF_init_n = cp.asnumpy(vF_init)
    ri_fourier_n = cp.asnumpy(ri_fourier)
    vF_mean_n = cp.asnumpy(vF_mean.reshape(2*n-1,2*n-1,order=imc.ORDER))
    vF_stdev_n = cp.asnumpy(vF_stdev.reshape(2*n-1,2*n-1,order=imc.ORDER))
    vF_abs_stdev_n = cp.asnumpy(vF_abs_stdev.reshape(2*n-1,2*n-1,order=imc.ORDER))
    ri_mean_n = cp.asnumpy(reconstructed_image_mean)
    ri_var_n = cp.asnumpy(reconstructed_image_var)
    ri_std_n = np.sqrt(ri_var_n)

#    ri_n_scalled = ri_n*cp.asnumpy(scalling_factor)
    u_mean_n = cp.asnumpy(u_image_mean)
    u_var_n = cp.asnumpy(u_image_s_var)
    ell_mean_n = cp.asnumpy(ell_image_mean)
    ell_var_n = cp.asnumpy(ell_image_s_var)
    
    
    #Plotting one by one
    #initial condition
    fig = plt.figure()
    plt.subplot(1,2,1)
    im = plt.imshow(np.absolute(vF_init_n),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Fourier - real part')
    plt.subplot(1,2,2)
    im = plt.imshow(np.angle(vF_init_n),cmap=cmap,vmin=-np.pi,vmax=np.pi)
    fig.colorbar(im)
    plt.title('Fourier - imaginary part')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'vF_init')+image_extension, bbox_inches='tight')
    plt.close()

    #vF Original 
    fig = plt.figure()
    plt.subplot(1,2,1)
    im = plt.imshow(np.absolute(vForiginal_n),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Fourier - absolute')
    plt.subplot(1,2,2)
    im = plt.imshow(np.angle(vForiginal_n),cmap=cmap,vmin=-np.pi,vmax=np.pi)
    fig.colorbar(im)
    plt.title('Fourier - angle')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'vForiginal')+image_extension, bbox_inches='tight')
    plt.close()

    #vF Original 
    fig = plt.figure()
    plt.subplot(1,2,1)
    im = plt.imshow(np.absolute(vF_mean_n),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Fourier - absolute')
    plt.subplot(1,2,2)
    im = plt.imshow(np.angle(vF_mean_n),cmap=cmap,vmin=-np.pi,vmax=np.pi)
    fig.colorbar(im)
    plt.title('Fourier - phase')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'vF_mean')+image_extension, bbox_inches='tight')
    plt.close()

    #Absolute error of vF - vForiginal
    fig = plt.figure()
    im = plt.imshow(np.abs(vF_mean_n-vForiginal_n),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Fourier abs Error')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'abs_err_vF_mean')+image_extension, bbox_inches='tight')
    plt.close()

    #Absolute error of vF_init - vForiginal
    fig = plt.figure()
    im = plt.imshow(np.abs(vF_init_n-vForiginal_n),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Fourier abs Error')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'abs_err_vF_init')+image_extension, bbox_inches='tight')
    plt.close()

    #Absolute error of vF_init - vForiginal
    fig = plt.figure()
    im = plt.imshow(np.abs(vF_init_n-vF_mean_n),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Fourier abs Error')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'abs_err_vF_init_vF_mean')+image_extension, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    im = plt.imshow(ri_mean_n,cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Reconstructed Image mean')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'ri_mean_n')+image_extension, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    im = plt.imshow(ri_fourier,cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Reconstructed Image through Fourier')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'ri_or_n')+image_extension, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    im = plt.imshow(ri_init,cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Reconstructed Image through Fourier')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'ri_init')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow(ri_var_n,cmap=cmap)
    fig.colorbar(im)
    plt.title('Reconstructed Image variance')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'ri_var_n')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow(target_image,cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Target Image')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'target_image')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow(ri_compare,cmap=cmap,vmin=-1,vmax=1)
    if isSinogram:        
        plt.title('Filtered Back Projection -FBP')
    else:
        plt.title('Reconstructed Image From vFOriginal')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'ri_compare')+image_extension, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    im = plt.imshow((target_image-ri_mean_n),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Error SPDE')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'err_RI_TI')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow((target_image-ri_compare),cmap=cmap)#,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Error SPDE')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'err_RIO_TI')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow((ri_compare-target_image),cmap=cmap,vmin=-1,vmax=1)
    fig.colorbar(im)
    plt.title('Error FPB')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'err_RI_CMP')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow(u_mean_n,cmap=cmap)
    fig.colorbar(im)
    plt.title('Mean $u$')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'u_mean_n')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow(u_var_n,cmap=cmap)
    plt.title('Var $u$')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'u_var_n')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow(ell_mean_n,cmap=cmap)
    fig.colorbar(im)
    plt.title('Mean $\ell$')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'ell_mean_n')+image_extension, bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    im = plt.imshow(ell_var_n,cmap=cmap)
    fig.colorbar(im)
    plt.title('Var $\ell$')
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'ell_var_n')+image_extension, bbox_inches='tight')
    plt.close()
    
    
    fig = plt.figure()
    if isSinogram:
        im = plt.imshow(sinogram,cmap=cmap)
        plt.title('Sinogram')
    else:
        im = plt.imshow(corrupted_image,cmap=cmap)
        plt.title('corrupted_image --- CI')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(str(SimulationResult_dir/'measurement')+image_extension, bbox_inches='tight')
    plt.close()

    #plot several slices
    N_slices = 16
    t_index = np.arange(target_image.shape[1])
    for i in range(N_slices):
        fig = plt.figure()
        slice_index = target_image.shape[0]*i//N_slices
        plt.plot(t_index,target_image[slice_index,:],'-k',linewidth=0.5,markersize=1)
        plt.plot(t_index,ri_fourier_n[slice_index,:],'-r',linewidth=0.5,markersize=1)
        plt.plot(t_index,ri_mean_n[slice_index,:],'-b',linewidth=0.5,markersize=1)
        
        plt.fill_between(t_index,ri_mean_n[slice_index,:]-2*ri_std_n[slice_index,:],
                        ri_mean_n[slice_index,:]+2*ri_std_n[slice_index,:], 
                        color='b', alpha=0.1)
        plt.plot(t_index,ri_compare[slice_index,:],':k',linewidth=0.5,markersize=1)
        plt.savefig(str(SimulationResult_dir/'1D_Slice_{}'.format(slice_index-(target_image.shape[0]//2)))+image_extension, bbox_inches='tight')
        plt.close()

    
    f_index = np.arange(n)
    for i in range(N_slices):
        fig = plt.figure()
        slice_index = vForiginal_n.shape[0]*i//N_slices
        plt.plot(f_index,np.abs(vForiginal_n[slice_index,n-1:]),'-r',linewidth=0.5,markersize=1)
        plt.plot(f_index,np.abs(vF_init_n[slice_index,n-1:]),':k',linewidth=0.5,markersize=1)
        plt.plot(f_index,np.abs(vF_mean_n[slice_index,n-1:]),'-b',linewidth=0.5,markersize=1)
        
        plt.fill_between(f_index,np.abs(vF_mean_n[slice_index,n-1:])-2*vF_abs_stdev_n[slice_index,n-1:],
                        np.abs(vF_mean_n[slice_index,n-1:])+2*vF_abs_stdev_n[slice_index,n-1:], 
                        color='b', alpha=0.1)
        plt.savefig(str(SimulationResult_dir/'1D_F_Slice_{}'.format(slice_index-n))+image_extension, bbox_inches='tight')
        plt.close()
#    fig.colorbar(im, ax=ax[:,:], shrink=0.8)
#    fig.savefig(str(SimulationResult_dir/'Result')+image_extension, bbox_inches='tight')
#    for ax_i in ax.flatten():
#        extent = ax_i.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#    #    print(ax_i.title.get_text())
#        fig.savefig(str(SimulationResult_dir/ax_i.title.get_text())+''+image_extension, bbox_inches=extent.expanded(1.2, 1.2))
#    
#    fig = plt.figure()
#    plt.hist(u_samples_history[:,0],bins=50,density=1)
    error = (target_image-ri_mean_n)
    error_CMP = (target_image-ri_compare)
    
    L2_error = np.linalg.norm(error)
    MSE = np.sum(error*error)/error.size
    PSNR = 10*np.log10(np.max(ri_mean_n)**2/MSE)
    SNR = np.mean(ri_mean_n)/np.sqrt(MSE*(error.size/(error.size-1)))
    
    L2_error_CMP = np.linalg.norm(error_CMP)
    MSE_CMP = np.sum(error_CMP*error_CMP)/error_CMP.size
    PSNR_CMP = 10*np.log10(np.max(ri_compare)**2/MSE_CMP)
    SNR_CMP = np.mean(ri_compare)/np.sqrt(MSE_CMP*(error_CMP.size/(error_CMP.size-1)))
    metric = {'L2_error':L2_error,
               'MSE':MSE,
               'PSNR':PSNR,
               'SNR':SNR,
                'L2_error_CMP':L2_error_CMP,
                'MSE_CMP':MSE_CMP,
                'PSNR_CMP':PSNR_CMP,
                'SNR_CMP':SNR_CMP}
    with h5py.File(result_file,mode='a') as file:
        for key,value in metric.items():
            if key in file.keys():
                del file[key]
            # else:
            file.create_dataset(key,data=value)
        
    print('Shallow-SPDE : L2-error {}, MSE {}, SNR {}, PSNR {},'.format(L2_error,MSE,SNR,PSNR))
    print('FBP : L2-error {}, MSE {}, SNR {}, PSNR {}'.format(L2_error_CMP,MSE_CMP,SNR_CMP,PSNR_CMP))

def post_analysis(input_dir,relative_path_str="/scratch/work/emzirm1/SimulationResult",folder_plot=True,filename='result.hdf5',cmap = plt.cm.seismic_r):
    
    sns.set_style("ticks")
    sns.set_context('paper')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    relative_path = pathlib.Path(relative_path_str)
    SimulationResult_dir = relative_path /input_dir
    result_file = str(SimulationResult_dir/filename)
    if not folder_plot:
        with h5py.File(result_file,mode='r') as file:
            n_layers = file['n_layers'][()]
            try:
                samples_history = file['Layers {}/samples_history'.format(n_layers-1)][()]
            except KeyError:
                pass
            finally:
                samples_history = file['Layers {}/samples_history'.format(n_layers-1)][()]
            
            u_samples_history = file['Layers {}/samples_history'.format(n_layers-2)][()]
            n = file['fourier/basis_number'][()]
            n_ext = file['fourier/extended_basis_number'][()]
            t_start = file['t_start'][()]
            t_end = file['t_end'][()]
            target_image = file['measurement/target_image'][()]
            corrupted_image = file['measurement/corrupted_image'][()]
            burn_percentage = file['burn_percentage'][()]

            #    meas_std = file['measurement/stdev'][()]
            isSinogram = 'sinogram' in file['measurement'].keys()
            
            if isSinogram:
                sinogram = file['measurement/sinogram'][()]
                theta = file['measurement/theta'][()]
                fbp = iradon(sinogram,theta,circle=True)
    else:

        for result_file in SimulationResult_dir.iterdir():
            loaded_first_file=False
        
            if result_file.name.endswith('hdf5'):
                with h5py.File(result_file,mode='r') as file:
                    if not loaded_first_file:
                        n_layers = file['n_layers'][()]                    
                        samples_history = file['Layers {}/samples_history'.format(n_layers-1)][()]
                        u_samples_history = file['Layers {}/samples_history'.format(n_layers-2)][()]
                        n = file['fourier/basis_number'][()]
                        n_ext = file['fourier/extended_basis_number'][()]
                        t_start = file['t_start'][()]
                        t_end = file['t_end'][()]
                        target_image = file['measurement/target_image'][()]
                        corrupted_image = file['measurement/corrupted_image'][()]
                        burn_percentage = file['burn_percentage'][()]
                        isSinogram = 'sinogram' in file['measurement'].keys()
        
                        if isSinogram:
                            sinogram = file['measurement/sinogram'][()]
                            theta = file['measurement/theta'][()]
                            fbp = iradon(sinogram,theta,circle=True)
        
                        loaded_first_file = True
                    else:
                        samples_history = file['Layers {}/samples_history'.format(n_layers-1)][()]
                        u_samples_history = file['Layers {}/samples_history'.format(n_layers-2)][()]
                    
                    
    _process_data(samples_history,u_samples_history,n,n_ext,t_start,t_end,target_image,corrupted_image,burn_percentage,
                isSinogram,sinogram,theta,fbp,SimulationResult_dir,result_file,cmap = plt.cm.seismic_r)

    


def drawSlices(ri_mean_n,N_slices):
    """ beta2 in ps / km
        C is chirp
        z is an array of z positions """
    t = np.arange(ri_mean_n.shape[1])
    X,Y = np.meshgrid(t,np.arange(0,ri_mean_n.shape[1],ri_mean_n.shape[1]//N_slices))

    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')

     
    verts = []
    for i in range(N_slices):
        slice_index = ri_mean_n.shape[0]*i//N_slices
        verts.append(list(zip(t,ri_mean_n[slice_index, :])))

    poly = PolyCollection(verts, facecolors=(1,1,1,1), edgecolors=(0,0,0,1),linewidths=0.5)
    poly.set_alpha(0.3)
    ax.add_collection3d(poly, zs=Y[:, 0], zdir='y')
    ax.set_xlim3d(np.min(X), np.max(X))
    ax.set_ylim3d(np.min(Y), np.max(Y))
    ax.set_zlim3d(np.min(ri_mean_n), np.max(ri_mean_n))
    plt.savefig('Slices'+image_extension, bbox_inches='tight')

def make_summaries(parent_path_str="/scratch/work/emzirm1/SimulationResult",dim=2):
    parent_path = pathlib.Path(parent_path_str)
    Series = []
    column_names = []
    column_names_sino=[]
    column_names_sino_with_error = []
    index_column = ['file_name']
    for f in parent_path.iterdir():
        for fname in f.iterdir():
            if str(fname).endswith('.hdf5'):
                try:
                    with h5py.File(fname,mode='r+') as file:
                        
                            if 'd' in file.keys():
                                d =file['d'][()] 
                                if d ==dim:
                                    #Check wheter Column_names still empty, if it is add column names
                                    if not column_names:
                                        print('creating column name')
                                        column_names = list(file.keys())
                                        
                                        #remove the groups
                                        column_names.remove('Layers 0')
                                        column_names.remove('Layers 1')
                                        column_names.remove('fourier')
                                        column_names.remove('measurement')
                                        column_names.remove('pcn')
                                        
                                        #add some necessaries
                                        column_names.append('fourier/basis_number')
                                        column_names.append('fourier/t_start')
                                        column_names.append('fourier/t_end')

                                        
                                        column_names.append('measurement/stdev')
                                        column_names.append('pcn/n_layers')
                                        column_names.append('pcn/beta')
                                        column_names.append('pcn/beta_feedback_gain')
                                        column_names.append('pcn/target_acceptance_rate')
                                        column_names.append('pcn/max_record_history')
                                        if 'file_name' not in column_names:
                                            column_names.append('file_name')
                                    if not column_names_sino:
                                        column_names_sino = column_names.copy()
                                        if 'measurement/n_r' in file.keys():
                                            column_names_sino.append('measurement/n_r')
                                            column_names_sino.append('measurement/n_theta')
                                    
                                    if not column_names_sino_with_error and 'L2_error' in file.keys():
                                        column_names_sino_with_error = column_names_sino.copy()
                                        column_names_sino_with_error.append('L2_error')
                                        column_names_sino_with_error.append('L2_error_CMP')
                                        column_names_sino_with_error.append('MSE')
                                        column_names_sino_with_error.append('MSE_CMP')
                                        column_names_sino_with_error.append('PSNR')
                                        column_names_sino_with_error.append('PSNR_CMP')
                                        
                                        
                                        
                                        
                                    if 'file_name' in file.keys():
                                        del file['file_name']
                                    
                                    file.create_dataset('file_name',data=str(fname.absolute()))
                                    print('Appending Series')
                                    if 'sinogram' not in file['measurement'].keys():
                                        pass
                                        # content = [file[key][()] for key in column_names]
                                        # Series.append(pd.Series(content,index=column_names_sino_with_error))
                                    else:
                                        if 'L2_error' in file.keys():
                                            content = [file[key][()] for key in column_names_sino_with_error]
                                            Series.append(pd.Series(content,index=column_names_sino_with_error))
                                        # else:
                                            # content = [file[key][()] for key in column_names_sino]
                                            # Series.append(pd.Series(content,index=column_names_sino_with_error))
                                    
                                    
                                    
                                else:
                                    #print('Dimension not match')
                                    continue
                except Exception as e:
                    print('Something bad happen when opening hdf5 file {}: {}'.format(str(fname),e.args))
                    continue
            else:
                continue
    df = pd.DataFrame(Series,columns=column_names_sino_with_error)
    df.to_excel('Summary MCMC-Simulations {} Dimension.xlsx'.format(d))
   
    

#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder',default="",type=str,help='Folder to analyze, Default=empty')
    parser.add_argument('--dim',default=2,type=int,help='Dimension, Default=2')
    
    args = parser.parse_args()
    if args.dim == 2:
        post_analysis(args.result_folder)
    elif args.dim == 1:
        post_analysis1D(args.result_folder)
    
