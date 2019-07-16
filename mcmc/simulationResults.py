# import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import pathlib
import seaborn as sns
# import mcmc.util as util
# import mcmc.fourier as fourier
# import mcmc.L as L
# import mcmc.randomGenerator as randomGenerator
# import mcmc.pCN as pCN
# import mcmc.measurement as meas
# import scipy as scp
# import time


specResult = [
    # ('t',nb.float64[:]),
    # ('fourier_basis_number',nb.int64),
    # ('fourier_extended_basis_number',nb.int64),
    ('vtHalf',nb.complex128[:]),
    ('vtF',nb.float64[:]),
    ('vHalfMean',nb.complex128[:]),
    # ('LMat',L_matrix_type),
    ('vHalfStdReal',nb.float64[:]),
    ('vHalfStdImag',nb.float64[:]),
    ('lMean',nb.float64[:]),
    ('lStd',nb.float64[:]),
    ('vtMean',nb.float64[:]),
    ('vtStd',nb.float64[:]),
    # ('cummMeanU',nb.complex128[:,:]),
    # ('indexCumm',nb.int64[:]),
]
@nb.jitclass(specResult)
class SimulationResult():
    def __init__(self,
    vtHalf,vtF,vHalfMean,vHalfStdReal,vHalfStdImag,
                lMean,lStd,vtMean,vtStd):
                # self.t = t
                # self.fourier_basis_number = fourier_basis_number
                # self.fourier_extended_basis_number = fourier_extended_basis_number
                self.vtHalf = vtHalf
                self.vtF = vtF
                self.vHalfMean = vHalfMean
                self.vHalfStdReal = vHalfStdReal
                self.vHalfStdImag = vHalfStdImag
                self.lMean = lMean
                self.lStd = lStd
                self.vtMean = vtMean
                self.vtStd = vtStd
                # self.cummMeanU = cummMeanU
                # self.indexCumm = indexCumm

