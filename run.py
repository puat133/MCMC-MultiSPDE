import mcmc.simulation as s

sim = s.Simulation(n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,
                kappa = 1e17,sigma_u = 5e6,sigma_v = 10,evaluation_interval = 100,
                seed=1,burn_percentage = 5.0);s
sim.run()
sim.analyze()