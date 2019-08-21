#%%
import matplotlib.pyplot as plt
from skimage.transform import radon
import tomo.tomography as tg
import mcmc.fourier as fourier
import mcmc.plotting as p
import mcmc.randomGenerator as rg
import mcmc.L as L
import numpy as np
import scipy.linalg as sla
import scipy.special as ssp
#%%
basis_number = 2**5
f = fourier.FourierAnalysis_2D(basis_number,64,0.0,1.0)
t = tg.Tomograph('shepp.png',f,target_size=2*f.extended_basis_number,n_theta=50,relative_location='phantom_images')
#%%
#add regulising
sigma_u = 5e6
sigma_v = 10
kappa = 1e17
d = 2
nu = 2 - d/2
alpha = nu + d/2
beta_u = (sigma_u**2)*(2**d * np.pi**(d/2) * ssp.gamma(alpha))/ssp.gamma(nu)
beta_v = beta_u*(sigma_v/sigma_u)**2
sqrtBeta_v = np.sqrt(beta_v)
sqrtBeta_u = np.sqrt(beta_u)
#%%
Lu = (1/sqrtBeta_u)*(f.Dmatrix*kappa**(-nu) - kappa**(2-nu)*f.Imatrix)
rgen = rg.RandomGenerator_2D(basis_number)
LMat = L.Lmatrix_2D(f,sqrtBeta_v)
wNew = rgen.construct_w_2D_ravelled()
eNew = np.random.randn(t.sinogram_flattened.shape[0])
wBar = np.concatenate((wNew,eNew))
LBar = np.vstack((t.H,LMat.current_L))
#%%
yBar = np.concatenate((t.pure_sinogram_flattened/t.meas_std,np.zeros(f.basis_number_2D_sym,dtype=np.complex128)))
#%%
v,res,rank,s = sla.lstsq(LBar,yBar-wBar,lapack_driver='gelsy')
vForiginalHalf = f.fourierTransformHalf(t.target_image)
vF = v.reshape(2*f.basis_number-1,2*f.basis_number-1)
#%%

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(vF.real,cmap=plt.cm.Greys_r)
ax[1].imshow(vF.imag,cmap=plt.cm.Greys_r)
ax[2].imshow(p.colorize(vF))
#%%
reconstructed_image = f.inverseFourierLimited(vF[:,basis_number-1:])
reconstructed_image2 = f.inverseFourierLimited(vForiginalHalf)
scale = np.max(reconstructed_image2-np.min(reconstructed_image2))/np.max(reconstructed_image-np.min(reconstructed_image))
reconstructed_image = reconstructed_image*scale
#%%
fig, ax = plt.subplots(ncols=3)
res = ax[0].imshow(reconstructed_image,cmap=plt.cm.Greys_r);fig.colorbar(res,ax=ax[0])
res = ax[1].imshow(reconstructed_image2,cmap=plt.cm.Greys_r);fig.colorbar(res,ax=ax[1])
res = ax[2].imshow(np.abs(reconstructed_image-reconstructed_image2),cmap=plt.cm.Greys_r);fig.colorbar(res,ax=ax[2])
#%%
# plt.show()
sinogram_scp = radon(t.target_image,t.theta,circle=False)
scale = (np.max(t.pure_sinogram)-np.min(t.pure_sinogram))/(np.max(sinogram_scp)-np.min(sinogram_scp))
scaled_sinogram_scp = scale*sinogram_scp
makesense = np.allclose(scaled_sinogram_scp,t.pure_sinogram)
print({True:'it is close',False:'it is not close'}[makesense])
fig, ax = plt.subplots(ncols=3)
res = ax[0].imshow(t.pure_sinogram)
fig.colorbar(res,ax=ax[0])
res = ax[1].imshow(scaled_sinogram_scp)
fig.colorbar(res,ax=ax[1])
res = ax[2].imshow(np.abs(scaled_sinogram_scp-t.pure_sinogram))
fig.colorbar(res,ax=ax[2])
plt.show()
