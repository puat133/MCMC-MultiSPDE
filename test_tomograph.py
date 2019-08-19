import matplotlib.pyplot as plt
import tomo.tomography as tg
import mcmc.fourier as fourier
import numpy as np

basis_number = 2**6
f = fourier.FourierAnalysis_2D(basis_number,4*basis_number,0.0,1.0)
t = tg.Tomograph('shepp.png',f,target_size=256,relative_location='phantom_images')
v,_,_,_ = np.linalg.lstsq(t.H,t.flattened_target_image)
vForiginal = f.fourierTransformHalf(t.target_image)
vF = v.reshape(2*f.basis_number-1,2*f.basis_number-1);fig, ax = plt.subplots(ncols=2);ax[0].imshow(vF.real);ax[1].imshow(vF.imag);
image = f.inverseFourierLimited(vF[:,f.basis_number-1:]);
image_if_fft = f.inverseFourierLimited(vForiginal)
fig, ax = plt.subplots(ncols=3);
ax[0].imshow(image);
ax[1].imshow(image_if_fft);
ax[2].imshow(image-image_if_fft);
plt.show()
# sinogram_scp = radon(tomograph.target_image,tomograph.theta,circle=False)
# scale = (np.max(tomograph.pure_sinogram)-np.min(tomograph.pure_sinogram))/(np.max(sinogram_scp)-np.min(sinogram_scp))
# scaled_sinogram_scp = scale*sinogram_scp
# makesense = np.allclose(scaled_sinogram_scp,tomograph.pure_sinogram)
# print({True:'it is close',False:'it is not close'}[makesense])
# fig, ax = plt.subplots(ncols=3)
# res = ax[0].imshow(tomograph.pure_sinogram)
# fig.colorbar(res,ax=ax[0])
# res = ax[1].imshow(scaled_sinogram_scp)
# fig.colorbar(res,ax=ax[1])
# res = ax[2].imshow(scaled_sinogram_scp-tomograph.pure_sinogram)
# fig.colorbar(res,ax=ax[2])
# plt.show()
