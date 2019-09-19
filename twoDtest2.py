#%%
import matplotlib.pyplot as plt
import mcmc.image_cupy as im
import mcmc.plotting as p
import numpy as np
import scipy.linalg as sla
import scipy.special as ssp
import mcmc.util_cupy as util
import cupy as cp
import importlib
import datetime
import pathlib,os
import argparse
import parser_help as ph
importlib.reload(im)
importlib.reload(util)
#%%
# n_layers =2
# n_samples=100
# n=16
# n_extended=4*n
# step = 1
# kappa = 1e17
# sigma_u = 5e6
# sigma_v = 1e2
# simga_scalling=0.1
# stdev = 0.1
# evaluation_interval=5
# printProgress = True
# seed=1
# burn_percentage=0
# enable_beta_feedback=True
# pcn_variant='dunlop'

# sim = im.Simulation(n_layers,n_samples,n,n_extended,step,kappa,sigma_u,sigma_v,simga_scalling,stdev,evaluation_interval,printProgress,
#                     seed,burn_percentage,enable_beta_feedback,pcn_variant)

# folderName = 'result-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M')
# if 'WRKDIR' in os.environ:
#     simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult'/folderName
# elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult').exists():
#     simResultPath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult')/folderName
# else:
#     simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
# if not simResultPath.exists():
#     simResultPath.mkdir()
# #%%
# sim.run()
# #%%
# sim.save(str(simResultPath/'result.hdf5'))
# #%%
# mean_field = util.symmetrize(cp.asarray(np.mean(sim.Layers[-1].samples_history[:n_samples,:],axis=0)))
# u_mean_field = util.symmetrize(cp.asarray(np.mean(sim.Layers[0].samples_history[:n_samples,:],axis=0)))
# vF = mean_field.reshape(2*sim.fourier.basis_number-1,2*sim.fourier.basis_number-1,order=im.ORDER).T
# uF = u_mean_field.reshape(2*sim.fourier.basis_number-1,2*sim.fourier.basis_number-1,order=im.ORDER).T
# vForiginal = util.symmetrize_2D(sim.fourier.fourierTransformHalf(sim.measurement.target_image))
# vFwithNoise = util.symmetrize_2D(sim.fourier.fourierTransformHalf(sim.measurement.corrupted_image))
# vFn = cp.asnumpy(vF)
# uFn = cp.asnumpy(uF)
# vForiginaln  = cp.asnumpy(vForiginal)
# fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(20,20))
# ax[0,0].imshow(vFn.real,cmap=plt.cm.Greys_r)
# ax[0,1].imshow(vForiginaln.real,cmap=plt.cm.Greys_r)
# ax[1,0].imshow(vFn.imag,cmap=plt.cm.Greys_r)
# ax[1,1].imshow(vForiginaln.imag,cmap=plt.cm.Greys_r)

# fig.savefig(str(simResultPath/'FourierDomain.pdf'), bbox_inches='tight')
# #%%
# reconstructed_image = sim.fourier.inverseFourierLimited(vF[:,sim.fourier.basis_number-1:])
# reconstructed_image_length_scale = sim.fourier.inverseFourierLimited(uF[:,sim.fourier.basis_number-1:])
# reconstructed_image_original = sim.fourier.inverseFourierLimited(vForiginal[:,sim.fourier.basis_number-1:])
# reconstructed_image_withNoise = sim.fourier.inverseFourierLimited(vFwithNoise[:,sim.fourier.basis_number-1:])
# scale = (cp.max(reconstructed_image)-cp.min(reconstructed_image))/(cp.max(reconstructed_image_original)-cp.min(reconstructed_image_original))
# ri_n = cp.asnumpy(scale*reconstructed_image)
# ri_or_n = cp.asnumpy(reconstructed_image_original)
# ri_wn_n = cp.asnumpy(reconstructed_image_withNoise)
# ri_ls_n = cp.asnumpy(cp.exp(reconstructed_image_length_scale))
# fig, ax = plt.subplots(nrows=4,figsize=(20,20))
# ax[0].imshow(ri_n,cmap=plt.cm.Greys_r)
# ax[1].imshow(ri_or_n,cmap=plt.cm.Greys_r)
# ax[2].imshow(ri_wn_n,cmap=plt.cm.Greys_r)
# ax[3].imshow(ri_ls_n,cmap=plt.cm.Greys_r)
# fig.savefig(str(simResultPath/'Reconstructed.pdf'), bbox_inches='tight')

#%%

#%%
if __name__=='__main__':
    # n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,uHalfInit=None,
    #                 kappa = 1e17,sigma_u = 5e6,sigma_v = 10,printInterval = 100,
    #                 seed=1,burnPercentage = 5,useLaTeX=True,randVectInitiated=True,
    #                 showFigures=True
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-layers',default=2,type=int,help='number SPDE layers, Default=2')
    parser.add_argument('--n',default=2**4,type=int,help='number of Fourier basis, Default=16')
    parser.add_argument('--seed',default=1,type=int,help='random generator seed, Default=1')
    parser.add_argument('--n-extended',default=2**6,type=int,help='number of point per axis, Default=64')
    parser.add_argument('--n-samples',default=100,type=int,help='number of MCMC samples per computer core, Default=100')
    parser.add_argument('--evaluation-interval',default=5,type=int,help='interval to print and reevaluate beta, Default=5')
    parser.add_argument('--beta',default=1,type=float,help='preconditioned Crank Nicholson beta parameter, Default=1')
    parser.add_argument('--kappa',default=1e17,type=float,help='kappa constant for u_t, Default=1e17')
    parser.add_argument('--sigma-0',default=1e7,type=float,help='Sigma_u constant, Default=1e7')
    parser.add_argument('--sigma-v',default=1e3,type=float,help='Sigma_v constant, Default=1e3')
    parser.add_argument('--sigma-scaling',default=1e-3,type=float,help='Sigma_scaling constant, Default=1e-3')
    parser.add_argument('--burn-percentage',default=25.0,type=float,help='Burn Percentage, Default=25.0')
    parser.add_argument('--variant',default="dunlop",type=str,help='preconditioned Crank Nicholson multilayered algorithm variant, Default=dunlop')
    ph.add_boolean_argument(parser,'enable-beta-feedback',default=True,messages='Whether beta-feedback will be enabled, Default=True')
    ph.add_boolean_argument(parser,'print-progress',default=True,messages='Whether progress is printed, Default=True')

    args = parser.parse_args()
    sim = im.Simulation(n_layers=args.n_layers,n_samples = args.n_samples,n = args.n,n_extended = args.n_extended,beta = args.beta,
                    kappa = args.kappa,sigma_0 = args.sigma_0,sigma_v = args.sigma_v,sigma_scaling=args.sigma_scaling,meas_std=0.1,evaluation_interval = args.evaluation_interval,printProgress=args.print_progress,
                    seed=args.seed,burn_percentage = args.burn_percentage,enable_beta_feedback=args.enable_beta_feedback,pcn_variant=args.variant)
    
    folderName = 'result-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M')
    if 'WRKDIR' in os.environ:
        simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult'/folderName
    elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult').exists():
        simResultPath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult')/folderName
    else:
        simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
    if not simResultPath.exists():
        simResultPath.mkdir()
    sim.run()
    sim.save(str(simResultPath/'result.hdf5'))
    #%%
    mean_field = util.symmetrize(cp.asarray(np.mean(sim.Layers[-1].samples_history,axis=0)))
    u_mean_field = util.symmetrize(cp.asarray(np.mean(sim.Layers[0].samples_history,axis=0)))
    vF = mean_field.reshape(2*sim.fourier.basis_number-1,2*sim.fourier.basis_number-1,order=im.ORDER).T
    uF = u_mean_field.reshape(2*sim.fourier.basis_number-1,2*sim.fourier.basis_number-1,order=im.ORDER).T
    vForiginal = util.symmetrize_2D(sim.fourier.fourierTransformHalf(sim.measurement.target_image))
    vFwithNoise = util.symmetrize_2D(sim.fourier.fourierTransformHalf(sim.measurement.corrupted_image))
    vFn = cp.asnumpy(vF)
    uFn = cp.asnumpy(uF)
    vForiginaln  = cp.asnumpy(vForiginal)
    fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(40,40))
    ax[0,0].imshow(vFn.real,cmap=plt.cm.Greys_r)
    ax[0,1].imshow(vForiginaln.real,cmap=plt.cm.Greys_r)
    ax[0,2].imshow((vForiginaln-vFn).real,cmap=plt.cm.Greys_r)
    ax[1,0].imshow(vFn.imag,cmap=plt.cm.Greys_r)
    ax[1,1].imshow(vForiginaln.imag,cmap=plt.cm.Greys_r)
    ax[1,2].imshow((vForiginaln-vFn).imag,cmap=plt.cm.Greys_r)

    fig.savefig(str(simResultPath/'FourierDomain.pdf'), bbox_inches='tight')
    #%%
    reconstructed_image = sim.fourier.inverseFourierLimited(vF[:,sim.fourier.basis_number-1:])
    reconstructed_image_uF = sim.fourier.inverseFourierLimited(uF[:,sim.fourier.basis_number-1:])
    reconstructed_image_original = sim.fourier.inverseFourierLimited(vForiginal[:,sim.fourier.basis_number-1:])
    reconstructed_image_withNoise = sim.fourier.inverseFourierLimited(vFwithNoise[:,sim.fourier.basis_number-1:])
    scale = (cp.max(reconstructed_image)-cp.min(reconstructed_image))/(cp.max(reconstructed_image_original)-cp.min(reconstructed_image_original))
    ri_n = cp.asnumpy(scale*reconstructed_image)
    ri_or_n = cp.asnumpy(reconstructed_image_original)
    ri_wn_n = cp.asnumpy(reconstructed_image_withNoise)
    ri_ls_n = cp.asnumpy(cp.exp(reconstructed_image_uF))
    fig, ax = plt.subplots(nrows=5,figsize=(40,40))
    ax[0].imshow(ri_n,cmap=plt.cm.Greys_r)
    ax[1].imshow(ri_or_n,cmap=plt.cm.Greys_r)
    ax[2].imshow(ri_wn_n,cmap=plt.cm.Greys_r)
    ax[3].imshow((ri_n-ri_or_n),cmap=plt.cm.Greys_r)
    ax[4].imshow(ri_ls_n,cmap=plt.cm.Greys_r)
    fig.savefig(str(simResultPath/'Reconstructed.pdf'), bbox_inches='tight')