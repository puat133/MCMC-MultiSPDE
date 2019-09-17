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
f = im.FourierAnalysis_2D(32,64,-0.5,0.5)
rg = im.RandomGenerator_2D(f.basis_number)
#add regulising
#add regulising
sigma_u = 5e6
sigma_v = 1e2
kappa = 1e17
d = 2
nu = 2 - d/2
alpha = nu + d/2
beta_u = (sigma_u**2)*(2**d * util.PI**(d/2) * ssp.gamma(alpha))/ssp.gamma(nu)
beta_v = beta_u*(sigma_v/sigma_u)**2
sqrtBeta_v = cp.sqrt(beta_v).astype('float32')
sqrtBeta_0 = cp.sqrt(beta_u).astype('float32')
LuReal = ((f.Dmatrix*kappa**(-nu) - kappa**(2-nu)*f.Imatrix)/sqrtBeta_0).astype('float32')
Lu = LuReal + 1j*cp.zeros(LuReal.shape,dtype=cp.float32)
        
uStdev_sym = -1/cp.diag(Lu)
uStdev = uStdev_sym[f.basis_number_2D_ravel-1:]
uStdev[0] /= 2 #scaled
n_layers = 2
#%%
measurement = im.TwoDMeasurement('shepp.png',target_size=f.extended_basis_number,stdev=1e-1,relative_location='phantom_images')

#%%
pcn = im.pCN(n_layers,rg,measurement,f,1,'dunlop')
#%%
n_samples = 100
pcn.record_skip = 1#np.max([1, n_samples// pcn.max_record_history])
history_length = n_samples#np.min([ n_samples, pcn.max_record_history]) 
#%%
# v,_,_,_ = cp.linalg.lstsq(pcn.H,pcn.y)
# vF = v.reshape(2*f.basis_number-1,2*f.basis_number-1,order=im.ORDER).T
# vForiginal = util.symmetrize_2D(f.fourierTransformHalf(measurement.target_image))
# vFn = cp.asnumpy(vF)
# vForiginaln  = cp.asnumpy(vForiginal)
# fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(20,20))
# ax[0,0].imshow(vFn.real,cmap=plt.cm.Greys_r)
# ax[0,1].imshow(vForiginaln.real,cmap=plt.cm.Greys_r)
# ax[1,0].imshow(vFn.imag,cmap=plt.cm.Greys_r)
# ax[1,1].imshow(vForiginaln.imag,cmap=plt.cm.Greys_r)

# reconstructed_image = f.inverseFourierLimited(vF[:,f.basis_number-1:])
# reconstructed_image_original = f.inverseFourierLimited(vForiginal[:,f.basis_number-1:])
# ri_n = cp.asnumpy(reconstructed_image)
# ri_or_n = cp.asnumpy(reconstructed_image_original)
# fig, ax = plt.subplots(ncols=2,figsize=(20,20))
# ax[0].imshow(ri_n,cmap=plt.cm.Greys_r)
# ax[1].imshow(ri_or_n,cmap=plt.cm.Greys_r)
#%%
temp = cp.linalg.solve(Lu,rg.construct_w())
Layers  = []
for i in range( n_layers):
    if i==0:
        init_sample_sym = uStdev_sym*pcn.random_gen.construct_w()
        lay = im.Layer(True, sqrtBeta_0,i, n_samples, pcn,init_sample_sym)
        lay.LMat.current_L = Lu
        lay.LMat.latest_computed_L = Lu
        lay.stdev_sym = uStdev_sym
        lay.stdev = uStdev
    else:
        if i == n_layers-1:
            lay = im.Layer(False, sqrtBeta_v,i, n_samples, pcn,Layers[i-1].current_sample_sym)
            wNew =  pcn.random_gen.construct_w()
            eNew = cp.random.randn(pcn.measurement.num_sample,dtype=cp.float32)
            wBar = cp.concatenate((eNew,wNew))
            LBar = cp.vstack(( pcn.H,lay.LMat.current_L))
            lay.current_sample_sym, res, rnk, s = cp.linalg.lstsq(LBar, pcn.yBar-wBar,rcond=-1)#,rcond=None)
            lay.current_sample = lay.current_sample_sym[f.basis_number_2D_ravel-1:]
        else:
            # lay = layer.Layer(False, sqrtBeta_v*np.sqrt(sigma_scaling),i, n_samples, pcn,Layers[i-1].current_sample)
            lay = im.Layer(False, sqrtBeta_v*0.1,i, n_samples, pcn,Layers[i-1].current_sample)

    lay.update_current_sample()
    pcn.Layers_sqrtBetas[i] = lay.sqrt_beta
    Layers.append(lay)
#%%
accepted_count = 0
for i in range( n_layers):
    Layers[i].i_record=0
for i in range(n_samples):
    accepted_count += pcn.one_step_non_centered_dunlop(Layers)
    print('Completed step {0}'.format(i))
    if i+1 % 5 == 0:
        acceptancePercentage = accepted_count/(i+1)
        pcn.adapt_beta(acceptancePercentage)

#%%

folderName = 'result-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M')
if 'WRKDIR' in os.environ:
    simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult'/folderName
elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult').exists():
    simResultPath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult')/folderName
else:
    simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
simResultPath.mkdir()


#%%
mean_field = util.symmetrize(cp.asarray(np.mean(Layers[-1].samples_history[:n_samples,:],axis=0)))
u_mean_field = util.symmetrize(cp.asarray(np.mean(Layers[0].samples_history[:n_samples,:],axis=0)))
vF = mean_field.reshape(2*f.basis_number-1,2*f.basis_number-1,order=im.ORDER).T
uF = u_mean_field.reshape(2*f.basis_number-1,2*f.basis_number-1,order=im.ORDER).T
vForiginal = util.symmetrize_2D(f.fourierTransformHalf(measurement.target_image))
vFwithNoise = util.symmetrize_2D(f.fourierTransformHalf(measurement.corrupted_image))
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
reconstructed_image = f.inverseFourierLimited(vF[:,f.basis_number-1:])
reconstructed_image_length_scale = f.inverseFourierLimited(uF[:,f.basis_number-1:])
reconstructed_image_original = f.inverseFourierLimited(vForiginal[:,f.basis_number-1:])
reconstructed_image_withNoise = f.inverseFourierLimited(vFwithNoise[:,f.basis_number-1:])
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

# plt.show()



#%%
