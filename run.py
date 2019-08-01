#%%
import mcmc.simulation as s
import mcmc.plotting as p
import matplotlib.pyplot as plt
import numpy as np
import argparse
#%%
# n = 2**6
# kappa_default =1e17
# sigma_0_default = 5e6#5e6
# sigma_v_default = 1e1#1e2
# kappa_factor = 1
# kappa = kappa_default/kappa_factor
# sigma_0 = sigma_0_default*np.sqrt(kappa_factor)
# sigma_v = sigma_v_default*np.sqrt(kappa_factor)


    





#%%
if __name__=='__main__':
    # n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,uHalfInit=None,
    #                 kappa = 1e17,sigma_u = 5e6,sigma_v = 10,printInterval = 100,
    #                 seed=1,burnPercentage = 5,useLaTeX=True,randVectInitiated=True,
    #                 showFigures=True
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-layers',default=2,type=int,help='number SPDE layers, Default=2')
    parser.add_argument('--n',default=2**6,type=int,help='number of Fourier basis, Default=64')
    parser.add_argument('--num',default=2**8,type=int,help='number measurement points, Default=256')
    parser.add_argument('--n-samples',default=1000,type=int,help='number of MCMC samples per computer core, Default=10000')
    parser.add_argument('--evaluation-interval',default=100,type=int,help='interval to print and reevaluate beta, Default=100')
    parser.add_argument('--beta',default=0.2,type=float,help='preconditioned Crank Nicholson beta parameter, Default=0.2')
    parser.add_argument('--kappa',default=1e17,type=float,help='kappa constant for u_t, Default=1e17')
    parser.add_argument('--sigma-0',default=5e6,type=float,help='Sigma_u constant, Default=5e6')
    parser.add_argument('--sigma-v',default=1e2,type=float,help='Sigma_v constant, Default=10.0')
    parser.add_argument('--sigma-scaling',default=1e-4,type=float,help='Sigma_scaling constant, Default=1e-4')
    parser.add_argument('--burn-percentage',default=5.0,type=float,help='Burn Percentage, Default=5.0')
    # parser.add_argument('--use-latex',default=True,type=bool,help='Whether to use Latex for plotting or not, Default=True')
    # parser.add_argument('--show-figures',default=True,type=bool,help='Whether to show simulation results figures or not, Default=True')
    # parser.add_argument('--rand-vect-init',default=False,type=bool,help='Whether to preallocate random vector before running MCMC, Default=False')
    # parser.add_argument('--init-file',default=None,type=str,help='Hdf5 file containing data set called cummMeanU, with size match parameter n, Default=None')
    # parser.add_argument('--pcn',default=False,type=bool,help='Whether using preconditioned Crank Nicholson algorithm or HMC, Default=False')

    args = parser.parse_args()
    # (self,n_layers,n_samples,n,beta,num,kappa,sigma_0,sigma_v,sigma_scaling,evaluation_interval,printProgress,
    #                 seed,burn_percentage)
    sim = s.Simulation(n_layers=args.n_layers,n_samples = args.n_samples,n = args.n,beta = args.beta,num = args.num,
                    kappa = args.kappa,sigma_0 = args.sigma_0,sigma_v = args.sigma_v,sigma_scaling=args.sigma_scaling,evaluation_interval = args.evaluation_interval,printProgress=True,
                    seed=1,burn_percentage = args.burn_percentage)
    sim.pcn.beta_feedback_gain = 2.1
    sim.run()
    sim.analyze()
    p.plotResult(sim)