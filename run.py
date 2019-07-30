#%%
import mcmc.simulation as s
import mcmc.plotting as p
import matplotlib.pyplot as plt
#%%
n = 2**6
sim = s.Simulation(n_layers=3,n_samples = 100000,n = n,beta = 1,num = 8*n,
                    kappa = 1e17,sigma_0 = 5e6,sigma_v = 1e2,sigma_scaling= 1e-8,evaluation_interval = 50,printProgress=True,
                    seed=1,burn_percentage = 50.0)
    
sim.pcn.beta_feedback_gain = 2.1
sim.run()
sim.analyze()
p.plotResult(sim)




#%%
