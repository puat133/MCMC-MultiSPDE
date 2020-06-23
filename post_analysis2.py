# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:31:38 2019

@author: Muhammad
"""
from skimage.io import imread
from skimage.transform import radon,iradon,resize
import h5py
import mcmc.util_cupy as util
import matplotlib.pyplot as plt
import cupy as cp, numpy as np
import mcmc.image_cupy as imc
import mcmc.measurement as mcmcMeas
import pathlib
import seaborn as sns
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import argparse


def remove_pdf_files(path):
    for f in path.iterdir():
        if f.name.endswith('.pdf'):
            os.remove(str(f))


def savefig(f):
    # if f.exists():
    #     os.remove(str(f))
    plt.savefig(str(f), bbox_inches='tight')

def imshow(zvalue,cmap,vmin,vmax,title,file_name,SimulationResult_dir,fontsize='xx-large'):
    fig = plt.figure()
    if (vmin is None) or (vmax is None):
        im = plt.imshow(zvalue,cmap=cmap)
    else:
        im = plt.imshow(zvalue,cmap=cmap,vmin=vmin,vmax=vmax)
    fig.colorbar(im)
    plt.title(title,fontsize=fontsize)
    plt.tight_layout()
    plt.axis('off')
    file_name += '.pdf'
    f = SimulationResult_dir/file_name
    savefig(f)
    plt.close()


def _process_data(samples_history,u_samples_history,n,n_ext,t_start,t_end,target_image,corrupted_image,burn_percentage,isSinogram,sinogram,theta,fbp,SimulationResult_dir,result_file,cmap = plt.cm.seismic_r,desired_n_ext=256):

    #remove pdf files:
    remove_pdf_files(SimulationResult_dir)

    burn_start_index = np.int(0.01*burn_percentage*u_samples_history.shape[0])
    fourier = imc.FourierAnalysis_2D(n,desired_n_ext,t_start,t_end)
    sL2 = util.sigmasLancosTwo(cp.int(n))
    # n_ext = 2*n
    scalling_factor = (2*fourier.extended_basis_number-1)/(2*n_ext-1)

    #initial conditions
    samples_init = samples_history[0,:]

    #change
    u_samples_history = u_samples_history[burn_start_index:,:]
    samples_history = samples_history[burn_start_index:,:]
    N = u_samples_history.shape[0]
    
    #initial condition
    vF_init = util.symmetrize(cp.asarray(samples_init)).reshape(2*n-1,2*n-1,order=imc.ORDER)*scalling_factor
    # vF_init = vF_init.conj()
    

    vF_mean = util.symmetrize(cp.asarray(np.mean(samples_history,axis=0)))*scalling_factor
    vF_stdev = util.symmetrize(cp.asarray(np.std(samples_history,axis=0)))
    vF_abs_stdev = util.symmetrize(cp.asarray(np.std(np.abs(samples_history),axis=0)))
   
    
    
    
    print('fourier n_ext = {}'.format(n_ext))
    # if isSinogram:
    #     vF_init = util.symmetrize_2D(fourier.rfft2(cp.asarray(fbp,dtype=cp.float32)))
    
#    if not isSinogram:
    vForiginal = util.symmetrize_2D(fourier.rfft2(cp.array(target_image,dtype=cp.float32)))#target image does not need to be scalled
    reconstructed_image_original = fourier.irfft2(vForiginal[:,n-1:])
    reconstructed_image_init = fourier.irfft2(vF_init[:,n-1:])
    
    
    samples_history_cp = cp.asarray(samples_history)*scalling_factor
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
    
    u_samples_history_cp = cp.asarray(u_samples_history)*scalling_factor
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
        u_image_mean = u_image_mean #cp.rot90(cp.fft.fftshift(u_image),1) 
        u_image_s_var = u_image_s_var #cp.rot90(cp.fft.fftshift(u_image),1) 
        ell_image_mean = ell_image_mean# cp.rot90(cp.fft.fftshift(ell_image),1) 
        ell_image_s_var = ell_image_s_var# cp.rot90(cp.fft.fftshift(ell_image),1) 
    else:
        reconstructed_image_mean = v_image_mean
        reconstructed_image_var = v_image_s_var
        reconstructed_image_mean = v_image_mean
        reconstructed_image_init = reconstructed_image_init
        u_image_mean = u_image_mean #cp.rot90(cp.fft.fftshift(u_image),1) 
        u_image_s_var = u_image_s_var #cp.rot90(cp.fft.fftshift(u_image),1) 
        ell_image_mean = ell_image_mean# cp.rot90(cp.fft.fftshift(ell_image),1) 
        ell_image_s_var = ell_image_s_var# cp.rot90(cp.fft.fftshift(ell_image),1)         
    
    
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
    # plt.savefig(str(SimulationResult_dir/'vF_init.pdf'), bbox_inches='tight')
    savefig(SimulationResult_dir/'vF_init.pdf')
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
    # plt.savefig(SimulationResult_dir/'vForiginal.pdf'), bbox_inches='tight')
    savefig(SimulationResult_dir/'vForiginal.pdf')
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
    # plt.savefig(SimulationResult_dir/'vF_mean.pdf'), bbox_inches='tight')
    savefig(SimulationResult_dir/'vF_mean.pdf')
    plt.close()

    #Absolute error of vF - vForiginal
    imshow(np.abs(vF_mean_n-vForiginal_n),cmap,-1,1,'Fourier abs Error','abs_err_vF_mean',SimulationResult_dir)
    

    #Absolute error of vF_init - vForiginal
    imshow(np.abs(vF_init_n-vForiginal_n),cmap,-1,1,'Fourier abs Error','abs_err_vF_init',SimulationResult_dir)

    #Absolute error of vF_init - vForiginal
    imshow(np.abs(vF_init_n-vF_mean_n),cmap,-1,1,'Fourier abs Error','abs_err_vF_init_vF_mean',SimulationResult_dir)
    

    #Ri_mean
    imshow(ri_mean_n,cmap,-1,1,'Posterior mean','ri_mean_n',SimulationResult_dir)
    
    #Ri_fourier
    imshow(ri_fourier,cmap,-1,1,'Reconstructed image through Fourier','ri_or_n',SimulationResult_dir)
    
    #Ri_fourier
    imshow(ri_init,cmap,-1,1,'Reconstructed image through Fourier','ri_init',SimulationResult_dir)

    #Reconstructed Image variance
    imshow(ri_var_n,cmap,None,None,'Posterior variance','ri_var_n',SimulationResult_dir)

    #Target Image
    imshow(target_image,cmap,-1,1,'Target Image','target_image',SimulationResult_dir)
    
    #Filtered Back Projection
    imshow(ri_compare,cmap,-1,1,'Filtered Back Projection','ri_compare',SimulationResult_dir)
    
    #Errors
    imshow((target_image-ri_mean_n),cmap,-1,1,'Error FPB','err_RI_TI',SimulationResult_dir)

    #Errors
    imshow((target_image-ri_compare),cmap,-1,1,'Error FPB-SPDE','err_RIO_TI',SimulationResult_dir)

    #Errors
    imshow((ri_mean_n-ri_compare),cmap,-1,1,'Error SPDE','err_RI_CMP',SimulationResult_dir)

    #Mean $u$
    imshow(u_mean_n,cmap,None,None,'Mean $u$','u_mean_n',SimulationResult_dir)

    #'Var $u$'
    imshow(u_var_n,cmap,None,None,'Var $u$','u_var_n',SimulationResult_dir)

    #'Mean $\ell$'
    imshow(ell_mean_n,cmap,None,None,r'Mean $\ell$','ell_mean_n',SimulationResult_dir)

    #'Var $\ell$'
    imshow(ell_var_n,cmap,None,None,r'Var $\ell$','ell_var_n',SimulationResult_dir)  
    
    
    fig = plt.figure()
    if isSinogram:
        im = plt.imshow(sinogram,cmap=cmap)
        plt.title('Sinogram')
    else:
        im = plt.imshow(corrupted_image,cmap=cmap)
        plt.title('corrupted_image --- CI')
    fig.colorbar(im)
    plt.tight_layout()
    # plt.savefig(SimulationResult_dir/'measurement.pdf'), bbox_inches='tight')
    savefig(SimulationResult_dir/'measurement.pdf')
    plt.close()

    #plot several slices
    N_slices = 16
    t_index = np.arange(target_image.shape[1])
    for i in range(N_slices):
        fig = plt.figure()
        slice_index = target_image.shape[0]*i//N_slices
        plt.plot(t_index,target_image[slice_index,:],'-k',linewidth=0.25,markersize=1)
        plt.plot(t_index,ri_fourier_n[slice_index,:],'-r',linewidth=0.25,markersize=1)
        plt.plot(t_index,ri_mean_n[slice_index,:],'-b',linewidth=0.25,markersize=1)
        
        plt.fill_between(t_index,ri_mean_n[slice_index,:]-2*ri_std_n[slice_index,:],
                        ri_mean_n[slice_index,:]+2*ri_std_n[slice_index,:], 
                        color='b', alpha=0.1)
        plt.plot(t_index,ri_compare[slice_index,:],':k',linewidth=0.25,markersize=1)
        # plt.savefig(SimulationResult_dir/'1D_Slice_{}.pdf'.format(slice_index-(target_image.shape[0]//2))), bbox_inches='tight')
        savefig(SimulationResult_dir/'1D_Slice_{}.pdf'.format(slice_index-(target_image.shape[0]//2)))
        plt.close()

    
    f_index = np.arange(n)
    for i in range(N_slices):
        fig = plt.figure()
        slice_index = vForiginal_n.shape[0]*i//N_slices
        plt.plot(f_index,np.abs(vForiginal_n[slice_index,n-1:]),'-r',linewidth=0.25,markersize=1)
        plt.plot(f_index,np.abs(vF_init_n[slice_index,n-1:]),':k',linewidth=0.25,markersize=1)
        plt.plot(f_index,np.abs(vF_mean_n[slice_index,n-1:]),'-b',linewidth=0.25,markersize=1)
        
        plt.fill_between(f_index,np.abs(vF_mean_n[slice_index,n-1:])-2*vF_abs_stdev_n[slice_index,n-1:],
                        np.abs(vF_mean_n[slice_index,n-1:])+2*vF_abs_stdev_n[slice_index,n-1:], 
                        color='b', alpha=0.1)
        # plt.savefig(SimulationResult_dir/'1D_F_Slice_{}.pdf'.format(slice_index-n)), bbox_inches='tight')
        savefig(SimulationResult_dir/'1D_F_Slice_{}.pdf'.format(slice_index-n))
        plt.close()

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
    # with h5py.File(result_file,mode='a') as file:
    #     for key,value in metric.items():
    #         if key in file.keys():
    #             del file[key]
    #         # else:
    #         file.create_dataset(key,data=value)
        
    print('Shallow-SPDE : L2-error {}, MSE {}, SNR {}, PSNR {},'.format(L2_error,MSE,SNR,PSNR))
    print('FBP : L2-error {}, MSE {}, SNR {}, PSNR {}'.format(L2_error_CMP,MSE_CMP,SNR_CMP,PSNR_CMP))

def post_analysis(input_dir,relative_path_str="/scratch/work/emzirm1/SimulationResult",folder_plot=True,filename='result.hdf5',cmap = plt.cm.seismic_r,desired_n_ext = 512,thinning = 1):
    
    sns.set_style("ticks")
    sns.set_context('paper')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    relative_path = pathlib.Path(relative_path_str)
    SimulationResult_dir = relative_path /input_dir
    result_file = str(SimulationResult_dir/filename)
    
    
    #Phantom folder
    phantom_file = pathlib.Path('/scratch/work/emzirm1/GitHub/mcmc-spde/phantom_images/shepp.png')
    img = imread(phantom_file,as_gray=True)
    # n_ext = 256 #file['fourier/extended_basis_number'][()]
    
    dim = 2*desired_n_ext-1
    target_image = resize(img, (dim, dim), anti_aliasing=False, preserve_range=True,order=1, mode='symmetric')
    
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
                        # target_image = file['measurement/target_image'][()]
                        corrupted_image = file['measurement/corrupted_image'][()]
                        burn_percentage = file['burn_percentage'][()]
                        isSinogram = 'sinogram' in file['measurement'].keys()
        
                        if isSinogram:
                            # sinogram = file['measurement/sinogram'][()]
                            theta = file['measurement/theta'][()]
                            stdev = file['measurement/stdev'][()]
                            sinogram = radon(target_image,theta,circle=True)
                            sinogram += stdev*np.random.randn(sinogram.shape[0],sinogram.shape[1])
                            fbp = iradon(sinogram,theta,circle=True)
        
                        loaded_first_file = True

                        samples_history = samples_history[::thinning,:]
                        u_samples_history = u_samples_history[::thinning,:]
                    else:
                        samples_history_new = file['Layers {}/samples_history'.format(n_layers-1)][()]
                        u_samples_history_new = file['Layers {}/samples_history'.format(n_layers-2)][()]
                        samples_history_new = samples_history_new[::thinning,:]
                        u_samples_history_new = u_samples_history_new[::thinning,:]
                        samples_history = np.vstack((samples_history,samples_history_new))
                        u_samples_history = np.vstack((u_samples_history,u_samples_history_new))

    
    
                  

    _process_data(samples_history,u_samples_history,n,n_ext,t_start,t_end,target_image,corrupted_image,burn_percentage,
                isSinogram,sinogram,theta,fbp,SimulationResult_dir,result_file,cmap = plt.cm.seismic_r,desired_n_ext=desired_n_ext) 

#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder',default="",type=str,help='Folder to analyze, Default=empty')
    parser.add_argument('--dim',default=2,type=int,help='Dimension, Default=2')
    parser.add_argument('--desired-n-ext',default=256,type=int,help='target dimension//2, Default=256')
    parser.add_argument('--thinning',default=1,type=int,help='thinning, Default=1')
    
    args = parser.parse_args()
    if args.dim == 2:
        post_analysis(args.result_folder,desired_n_ext=args.desired_n_ext,thinning=args.thinning)

    
