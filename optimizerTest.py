import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import seaborn as sns
from numba.typed import List
# from numba.typed import List
import mcmc.util as util
import mcmc.fourier as fourier
import mcmc.L as L
import mcmc.layer as layer
import mcmc.randomGenerator as randomGenerator
import mcmc.pCN as pCN
import mcmc.measurement as meas
import mcmc.optimizer as optm
import mcmc.simulationResults as simRes
# import scipy as scp
import time

n_layers=3
n_samples = 100000
n = 2**6
beta = 1
num = 8*n
kappa = 1e17
sigma_0 = 5e6
sigma_v = 1e2
sigma_scaling= 1e-8
evaluation_interval = 50
printProgress=True
seed=1
burn_percentage = 50.0
d = 1
nu = 2 - d/2
alpha = nu + d/2
t_start = 0.0
t_end = 1.0
beta_0 = (sigma_0**2)*(2**d * np.pi**(d/2))* 1.1283791670955126#<-- this numerical value is scp.special.gamma(alpha))/scp.special.gamma(nu)
beta_v = beta_0*(sigma_v/sigma_0)**2
sqrtBeta_v = np.sqrt(beta_v)
sqrtBeta_0 = np.sqrt(beta_0)

f =  fourier.FourierAnalysis(n,num,t_start,t_end)
# self.fourier = f

random_gen = randomGenerator.RandomGenerator(f.fourier_basis_number)
# self.random_gen = rg


LuReal = (1/sqrtBeta_0)*(f.Dmatrix*kappa**(-nu) - kappa**(2-nu)*f.Imatrix)
Lu = LuReal + 1j*np.zeros(LuReal.shape)

uStdev = -1/np.diag(Lu)
uStdev = uStdev[f.fourier_basis_number-1:]
uStdev[0] /= 2 #scaled

meas_std = 0.1
measurement = meas.Measurement(num,meas_std,t_start,t_end)
# pcn = pCN.pCN(n_layers,rg,measurement,f,beta)
pcn = pCN.pCN(n_layers,random_gen,measurement,f,beta)


#initialize Layers
# n_layers = 2
Layers = List()
# factor = 1e-8
for i in range(n_layers):
    if i==0:
        init_sample = np.linalg.solve(Lu,random_gen.construct_w())[f.fourier_basis_number-1:]
        lay = layer.Layer(True,sqrtBeta_0,i,n_samples,pcn,init_sample)
        lay.stdev = uStdev
        lay.current_sample_scaled_norm = util.norm2(lay.current_sample/lay.stdev)#ToDO: Modify this
        lay.new_sample_scaled_norm = lay.current_sample_scaled_norm
    else:
        
        if i == n_layers-1:
            
            lay = layer.Layer(False,sqrtBeta_v,i,n_samples,pcn,Layers[i-1].current_sample)
            wNew = pcn.random_gen.construct_w()
            eNew = np.random.randn(pcn.measurement.num_sample)
            wBar = np.concatenate((eNew,wNew))
            
            LBar = np.vstack((pcn.H,lay.LMat.current_L))

            #update v
            lay.current_sample_symmetrized, res, rnk, s = np.linalg.lstsq(LBar,pcn.yBar-wBar,rcond=-1)#,rcond=None)
            lay.current_sample = lay.current_sample_symmetrized[pcn.fourier.fourier_basis_number-1:]
        else:
            lay = layer.Layer(False,sqrtBeta_v*np.sqrt(sigma_scaling),i,n_samples,pcn,Layers[i-1].current_sample)
    lay.update_current_sample()
    #TODO: toggle this if pcn.one_step_one_element is not used
    # lay.samples_history = np.empty((lay.n_samples*f.fourier_basis_number, f.fourier_basis_number), dtype=np.complex128)

    Layers.append(lay)


#allowable methods: ‘Nelder-Mead’,‘Powell’,‘COBYLA’,‘trust-constr’, '‘L-BFGS-B'
method = 'L-BFGS-B'
optimizer = optm.Optimizer(Layers,method=method)
opt_Result = optimizer.optimize()
u0_Half_optimized = optm.xToUHalf(opt_Result.x)
Layers[0].new_sample = u0_Half_optimized
Layers[0].new_sample_symmetrized = Layers[0].pcn.random_gen.symmetrize(Layers[0].new_sample)
Layers[0].new_sample_scaled_norm = util.norm2(Layers[0].new_sample/Layers[0].stdev)
Layers[0].update_current_sample()
for i in range(1,len(Layers)):
    Layers[i].LMat.construct_from(Layers[i-1].new_sample)
    Layers[i].new_log_L_det = np.linalg.slogdet(Layers[i].LMat.latest_computed_L)[1]
    Layers[i].LMat.set_current_L_to_latest()
    Layers[i].sample()
    Layers[i].current_sample_scaled_norm = util.norm2(Layers[i].LMat.current_L@Layers[i].new_sample_symmetrized)
    Layers[i].update_current_sample()
    # negLogPost += 0.5*Layers[i].current_sample_scaled_norm
    # negLogPost -= Layers[i].current_log_L_det
plt.figure()
plt.plot(measurement.t,f.inverseFourierLimited(Layers[-1].current_sample))
plt.figure()
plt.plot(measurement.t,np.exp(-f.inverseFourierLimited(u0_Half_optimized)))
plt.show()