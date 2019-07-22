#%%
import mcmc.simulation as s
import mcmc.plotting as p
import matplotlib.pyplot as plt
#%%
n = 2**4
sim = s.Simulation(n_layers=2,n_samples = 100000,n = n,beta = 1,num = 2**8,
                kappa = 1e17,sigma_u = 5e6,sigma_v = 10,evaluation_interval = 100,
                seed=1,burn_percentage = 50.0)

# sim.pcn.aggresiveness = 0.2
sim.pcn.beta_feedback_gain = 2.1
sim.run()
sim.analyze()
p.plotResult(sim)
# sim.analyze()
# p.plotResult(sim)




#%%
