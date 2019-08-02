import numpy as np
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
    ('uHalfMean',nb.complex128[:,:]),
    # ('LMat',L_matrix_type),
    ('uHalfStdReal',nb.float64[:,:]),
    ('uHalfStdImag',nb.float64[:,:]),
    ('elltMean',nb.float64[:,:]),
    ('elltStd',nb.float64[:,:]),
    ('utMean',nb.float64[:,:]),
    ('utStd',nb.float64[:,:]),
    # ('cummMeanU',nb.complex128[:,:]),
    # ('indexCumm',nb.int64[:]),
]
@nb.jitclass(specResult)
class SimulationResult():
    def __init__(self):
                # self.t = t
                # self.fourier_basis_number = fourier_basis_number
                # self.fourier_extended_basis_number = fourier_extended_basis_number
                self.vtHalf = np.empty((2),dtype=np.complex128)
                self.vtF = np.empty((2),dtype=np.float64)
                #for two D matrix, numba does not really now how to assign properly at this stage so:
                dummyComplex = np.empty((1,2),dtype=np.complex128)
                dummyReal = np.empty((1,2),dtype=np.float64)
                self.uHalfMean = dummyComplex
                self.uHalfStdReal = dummyReal
                self.uHalfStdImag = dummyReal
                self.elltMean = dummyReal
                self.elltStd = dummyReal
                self.utMean = dummyReal
                self.utStd = dummyReal
                # self.assign_two_dim_matrix(uHalfMean,uHalfStdReal,uHalfStdImag,
                # elltMean,elltStd,utMean,utStd)
                # self.cummMeanU = cummMeanU
                # self.indexCumm = indexCumm
    def assign_values(self,vtHalf,vtF,uHalfMean,uHalfStdReal,uHalfStdImag,
                elltMean,elltStd,utMean,utStd):
                self.vtHalf = vtHalf
                self.vtF = vtF
                self.uHalfMean = uHalfMean
                self.uHalfStdReal = uHalfStdReal
                self.uHalfStdImag = uHalfStdImag
                self.elltMean = elltMean
                self.elltStd = elltStd
                self.utMean = utMean
                self.utStd = utStd


# specResult = [
#     # ('t',nb.float64[:]),
#     # ('fourier_basis_number',nb.int64),
#     # ('fourier_extended_basis_number',nb.int64),
#     ('vtHalf',nb.complex128[:]),
#     ('vtF',nb.float64[:]),
#     ('vHalfMean',nb.complex128[:]),
#     # ('LMat',L_matrix_type),
#     ('vHalfStdReal',nb.float64[:]),
#     ('vHalfStdImag',nb.float64[:]),
#     ('lMean',nb.float64[:]),
#     ('lStd',nb.float64[:]),
#     ('vtMean',nb.float64[:]),
#     ('vtStd',nb.float64[:]),
#     # ('cummMeanU',nb.complex128[:,:]),
#     # ('indexCumm',nb.int64[:]),
# ]
# @nb.jitclass(specResult)
# class SimulationResult():
#     def __init__(self,
#     vtHalf,vtF,vHalfMean,vHalfStdReal,vHalfStdImag,
#                 lMean,lStd,vtMean,vtStd):
#                 # self.t = t
#                 # self.fourier_basis_number = fourier_basis_number
#                 # self.fourier_extended_basis_number = fourier_extended_basis_number
#                 self.vtHalf = vtHalf
#                 self.vtF = vtF
#                 self.vHalfMean = vHalfMean
#                 self.vHalfStdReal = vHalfStdReal
#                 self.vHalfStdImag = vHalfStdImag
#                 self.lMean = lMean
#                 self.lStd = lStd
#                 self.vtMean = vtMean
#                 self.vtStd = vtStd
#                 # self.cummMeanU = cummMeanU
#                 # self.indexCumm = indexCumm