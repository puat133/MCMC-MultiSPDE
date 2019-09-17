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
importlib.reload(im)
importlib.reload(util)
#%%
f = im.FourierAnalysis_2D(32,128,-0.5,0.5)
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
sqrtBeta_v = cp.sqrt(beta_v)
sqrtBeta_0 = cp.sqrt(beta_u)
LuReal = ((f.Dmatrix*kappa**(-nu) - kappa**(2-nu)*f.Imatrix)/sqrtBeta_0).astype('float32')
Lu = LuReal + 1j*cp.zeros(LuReal.shape,dtype=cp.float32)
        
uStdev = -1/cp.diag(Lu)
# uStdev = uStdev[self.fourier.basis_number-1:]
# uStdev[0] /= 2 #scaled
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
v,_,_,_ = util.lstsq(pcn.H,pcn.y)
vF = v.reshape(2*f.basis_number-1,2*f.basis_number-1,order=im.ORDER).T
vForiginal = util.symmetrize_2D(f.fourierTransformHalf(measurement.target_image))
vFn = cp.asnumpy(vF)
vForiginaln  = cp.asnumpy(vForiginal)
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(20,20))
ax[0,0].imshow(vFn.real,cmap=plt.cm.Greys_r)
ax[0,1].imshow(vForiginaln.real,cmap=plt.cm.Greys_r)
ax[1,0].imshow(vFn.imag,cmap=plt.cm.Greys_r)
ax[1,1].imshow(vForiginaln.imag,cmap=plt.cm.Greys_r)

reconstructed_image = f.inverseFourierLimited(vF[:,f.basis_number-1:])
reconstructed_image_original = f.inverseFourierLimited(vForiginal[:,f.basis_number-1:])
ri_n = cp.asnumpy(reconstructed_image)
ri_or_n = cp.asnumpy(reconstructed_image_original)
fig, ax = plt.subplots(ncols=2,figsize=(20,20))
ax[0].imshow(ri_n,cmap=plt.cm.Greys_r)
ax[1].imshow(ri_or_n,cmap=plt.cm.Greys_r)
#%%


#%%
