import tomo.tomography as tg
import numpy as np
tomograph = tg.Tomograph('shepp.png',target_size=128,relative_location='phantom_images')
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
