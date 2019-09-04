#%%
import mcmc.simulation as s
import mcmc.plotting as p
import matplotlib.pyplot as plt
import numpy as np
import argparse
import parser_help as ph
#%%
# n = 2**6
# kappa_default =1e17
# sigma_0_default = 5e6#5e6
# sigma_v_default = 1e1#1e2
# kappa_factor = 1
# kappa = kappa_default/kappa_factor
# sigma_0 = sigma_0_default*np.sqrt(kappa_factor)
# sigma_v = sigma_v_default*np.sqrt(kappa_factor)
#https://stackoverflow.com/a/36194213/11764120

    





#%%
if __name__=='__main__':
    # n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,uHalfInit=None,
    #                 kappa = 1e17,sigma_u = 5e6,sigma_v = 10,printInterval = 100,
    #                 seed=1,burnPercentage = 5,useLaTeX=True,randVectInitiated=True,
    #                 showFigures=True
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-layers',default=2,type=int,help='number SPDE layers, Default=2')
    parser.add_argument('--n',default=2**6,type=int,help='number of Fourier basis, Default=64')
    parser.add_argument('--seed',default=1,type=int,help='random generator seed, Default=1')
    parser.add_argument('--num',default=2**8,type=int,help='number measurement points, Default=256')
    parser.add_argument('--n-samples',default=1000,type=int,help='number of MCMC samples per computer core, Default=10000')
    parser.add_argument('--evaluation-interval',default=100,type=int,help='interval to print and reevaluate beta, Default=100')
    parser.add_argument('--beta',default=1,type=float,help='preconditioned Crank Nicholson beta parameter, Default=1')
    parser.add_argument('--kappa',default=1e17,type=float,help='kappa constant for u_t, Default=1e17')
    parser.add_argument('--sigma-0',default=5e6,type=float,help='Sigma_u constant, Default=5e6')
    parser.add_argument('--sigma-v',default=1e2,type=float,help='Sigma_v constant, Default=10.0')
    parser.add_argument('--sigma-scaling',default=1e-4,type=float,help='Sigma_scaling constant, Default=1e-4')
    parser.add_argument('--burn-percentage',default=25.0,type=float,help='Burn Percentage, Default=25.0')
    ph.add_boolean_argument(parser,'include-history',default=False,messages='Whether to include Layer simulation history in hdf5, Default=False')
    ph.add_boolean_argument(parser,'enable-beta-feedback',default=False,messages='Whether beta-feedback will be enabled, Default=True')
    ph.add_boolean_argument(parser,'print-progress',default=True,messages='Whether progress is printed, Default=True')
    ph.add_boolean_argument(parser,'use-latex',default=True,messages='Whether latex is used during results plotting, Default=True')

    args = parser.parse_args()
    sim = s.Simulation(n_layers=args.n_layers,n_samples = args.n_samples,n = args.n,beta = args.beta,num = args.num,
                    kappa = args.kappa,sigma_0 = args.sigma_0,sigma_v = args.sigma_v,sigma_scaling=args.sigma_scaling,evaluation_interval = args.evaluation_interval,printProgress=args.print_progress,
                    seed=args.seed,burn_percentage = args.burn_percentage,enable_beta_feedback=args.enable_beta_feedback)
    sim.pcn.beta_feedback_gain = 2.1
    sim.run()
    sim.analyze()
    p.plotResult(sim,include_history=args.include_history,useLaTeX=args.use_latex)

