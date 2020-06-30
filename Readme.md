# Deep Gaussian inversion 
This repository contains an implmentation of a deep Gaussian inversion using Galerkin method.

This repository is tested to run against these packages
1. `Numba` version should be later than `0.47`,
2. `CUPY` version `7.0.0bc3`, with `CUDA` version `10.1`,
3. Standard installation of `numpy` and `scipy`,
4. `h5py`.
5. Python version `3.7`.
6. Replace your `conda/envs/CUPYrc/lib/python3.7/site-packages/cupy/linalg/norms.py /home/emzirm1/.conda/envs/Numba/lib/python3.7/site-packages/cupy/linalg/norms.py` with the one in the `patch` folder.


Explanation of files and folders:

1. To run the MCMC, run `twoDsim.py`.
2. This repository is a companion of an arXiv paper, [Non-Stationary Multi-layered Gaussian Priors for Bayesian Inversion](http://arxiv.org/abs/2006.15634)

As this repository is a working folder, things might break without notification. Should you face any problem, please contact <muhammad.emzir@aalto.fi>.
