import mcmc.simulation as s
import mcmc.plotting as p

sim = s.Simulation(n_samples = 10000,n = 2**6,beta = 2e-1,num = 2**8,
                kappa = 1e17,sigma_u = 5e6,sigma_v = 10,evaluation_interval = 100,
                seed=1,burn_percentage = 5.0)
sim.run()
sim.analyze()
p.plotResult(sim)