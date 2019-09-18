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
importlib.reload(im)
importlib.reload(util)
#%%
n_layers =2
n_samples=1000
n=16
n_extended=4*n
step = 1
kappa = 1e17
sigma_u = 5e6
sigma_v = 1e2
simga_scalling=0.1
stdev = 0.1
evaluation_interval=5
printProgress = True
seed=1
burn_percentage=0
enable_beta_feedback=True
pcn_variant='dunlop'

sim = im.Simulation(n_layers,n_samples,n,n_extended,step,kappa,sigma_u,sigma_v,simga_scalling,stdev,evaluation_interval,printProgress,
                    seed,burn_percentage,enable_beta_feedback,pcn_variant)

folderName = 'result-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M')
if 'WRKDIR' in os.environ:
    simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult'/folderName
elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult').exists():
    simResultPath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult')/folderName
else:
    simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
if not simResultPath.exists():
    simResultPath.mkdir()
#%%
sim.run()
#%%
sim.save(str(simResultPath/'result.hdf5'))
#%%
mean_field = util.symmetrize(cp.asarray(np.mean(sim.Layers[-1].samples_history[:n_samples,:],axis=0)))
u_mean_field = util.symmetrize(cp.asarray(np.mean(sim.Layers[0].samples_history[:n_samples,:],axis=0)))
vF = mean_field.reshape(2*sim.fourier.basis_number-1,2*sim.fourier.basis_number-1,order=im.ORDER).T
uF = u_mean_field.reshape(2*sim.fourier.basis_number-1,2*sim.fourier.basis_number-1,order=im.ORDER).T
vForiginal = util.symmetrize_2D(sim.fourier.fourierTransformHalf(sim.measurement.target_image))
vFwithNoise = util.symmetrize_2D(sim.fourier.fourierTransformHalf(sim.measurement.corrupted_image))
vFn = cp.asnumpy(vF)
uFn = cp.asnumpy(uF)
vForiginaln  = cp.asnumpy(vForiginal)
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(20,20))
ax[0,0].imshow(vFn.real,cmap=plt.cm.Greys_r)
ax[0,1].imshow(vForiginaln.real,cmap=plt.cm.Greys_r)
ax[1,0].imshow(vFn.imag,cmap=plt.cm.Greys_r)
ax[1,1].imshow(vForiginaln.imag,cmap=plt.cm.Greys_r)

fig.savefig(str(simResultPath/'FourierDomain.pdf'), bbox_inches='tight')
#%%
reconstructed_image = sim.fourier.inverseFourierLimited(vF[:,sim.fourier.basis_number-1:])
reconstructed_image_length_scale = sim.fourier.inverseFourierLimited(uF[:,sim.fourier.basis_number-1:])
reconstructed_image_original = sim.fourier.inverseFourierLimited(vForiginal[:,sim.fourier.basis_number-1:])
reconstructed_image_withNoise = sim.fourier.inverseFourierLimited(vFwithNoise[:,sim.fourier.basis_number-1:])
scale = (cp.max(reconstructed_image)-cp.min(reconstructed_image))/(cp.max(reconstructed_image_original)-cp.min(reconstructed_image_original))
ri_n = cp.asnumpy(scale*reconstructed_image)
ri_or_n = cp.asnumpy(reconstructed_image_original)
ri_wn_n = cp.asnumpy(reconstructed_image_withNoise)
ri_ls_n = cp.asnumpy(cp.exp(reconstructed_image_length_scale))
fig, ax = plt.subplots(nrows=4,figsize=(20,20))
ax[0].imshow(ri_n,cmap=plt.cm.Greys_r)
ax[1].imshow(ri_or_n,cmap=plt.cm.Greys_r)
ax[2].imshow(ri_wn_n,cmap=plt.cm.Greys_r)
ax[3].imshow(ri_ls_n,cmap=plt.cm.Greys_r)
fig.savefig(str(simResultPath/'Reconstructed.pdf'), bbox_inches='tight')

#%%
