# import numba_.prepare as prep
# import numba_.utilNumba as util
# import numba_.mcmc as mcmc
import mcmc.simulation as s
import mcmc.simulationResults as sr
import mcmc.plotting as p
import numpy as np
import pathlib
import datetime
from mpi4py import MPI
import argparse
import sys
import pathlib
import h5py

def runSimulations(n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,uHalfInit=None,
                    kappa = 1e17,sigma_u = 2e6,sigma_v = 10,evaluation_interval = 100,
                    seed=1,burnPercentage = 5):

    #initialize parameter                   
    
    
    #prepare MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    sim = s.Simulation(n_samples = n_samples,n = n,beta = beta,num = num,
                    kappa = kappa,sigma_u = sigma_u,sigma_v = sigma_v,evaluation_interval = evaluation_interval,printProgress=False,
                    seed=rank+1,burn_percentage = burnPercentage)

        
    if rank==0:
        yt = sim.pcn.measurement.yt
        yBar = sim.pcn.yBar
    else:
        n = sim.fourier.fourier_basis_number
        m = sim.meas_samples_num
        yt = np.empty(m,dtype=np.float64)
        yBar = np.empty(m+2*n-1,dtype=np.float64)

    
    #Broadcast y and yBar
    comm.Bcast([yt,MPI.DOUBLE],root=0)#comm.bcast(y,root=0)
    comm.Bcast([yBar,MPI.DOUBLE],root=0)

    if rank != 0:
        sim.pcn.yBar = yBar
        sim.pcn.measurement.yt = yt

    print('Rank {0:d} prepared for MCMC running'.format(rank))
    sys.stdout.flush()

    sim.run()
    print('Rank {0:d} completed pCN simulation run with n_samples = {1:d} for {2:f} seconds'.format(rank,n_samples,sim.total_time))
    sys.stdout.flush()

    

    print('Rank {0:d} start analysing simulation results'.format(rank))
    sys.stdout.flush()

    sim.analyze()
    print('Rank {0:d} analysis completed'.format(rank))
    sys.stdout.flush()


    # # #Doing some reduce here
    print('Rank {0:d} sending analysis result to Rank 0'.format(rank))
    sys.stdout.flush()

    # uHistoryTotal = None
    # vHistoryTotal = None

    # vtEsTotal = None
    # utTotal = None
    # lUTotal = None

    vtEsSum = None
    lUSum = None

    vHalfSum = None
    cummMeanUSum = None
    
    vHalfVarRealSum = None
    vHalfVarImagSum = None
    vtVarSum = None
    lVarSum = None
    vHalfVarRealSumCorr = None
    vHalfVarImagSumCorr = None
    vtVarSumCorr = None
    lVarSumCorr = None

    vHalfMeanAll = None
    vtMeanAll = None
    lMeanAll = None

    startIndex = np.int(sim.burn_percentage*sim.n_samples//100)
    cummU = np.cumsum(sim.Layers[0].samples_history[startIndex:,:],axis=0)

    # #set buffer
    if rank==0:
        vtEsSum     = np.empty(sim.sim_result.vtMean.shape,dtype=np.float64)
        lUSum     = np.empty(sim.sim_result.lMean.shape,dtype=np.float64)
            
        vHalfVarRealSum = np.empty(sim.sim_result.vtHalf.shape,dtype=np.float64)
        vHalfVarImagSum = np.empty(sim.sim_result.vtHalf.shape,dtype=np.float64)
        vtVarSum = np.empty(sim.sim_result.vtF.shape,dtype=np.float64)
        lVarSum = np.empty(sim.sim_result.vtF.shape,dtype=np.float64)
        
        vHalfSum     = np.empty(sim.sim_result.vHalfMean.shape,dtype=np.complex128)
        cummMeanUSum     = np.empty(cummU.shape,dtype=np.complex128)

        vHalfVarRealSumCorr = np.empty(sim.sim_result.vtHalf.shape,dtype=np.float64)
        vHalfVarImagSumCorr = np.empty(sim.sim_result.vtHalf.shape,dtype=np.float64)
        vtVarSumCorr = np.empty(sim.sim_result.vtF.shape,dtype=np.float64)
        lVarSumCorr = np.empty(sim.sim_result.vtF.shape,dtype=np.float64)
    
    
        

    #Doing REDUCE SUM HERE --> TO BE AVERAGED IN RANK 0
    comm.Reduce([sim.sim_result.vtMean,MPI.DOUBLE],vtEsSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.lMean,MPI.DOUBLE],lUSum,MPI.SUM,root=0)
    

    comm.Reduce([sim.sim_result.vHalfStdReal**2,MPI.DOUBLE],vHalfVarRealSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.vHalfStdImag**2,MPI.DOUBLE],vHalfVarImagSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.vtStd**2,MPI.DOUBLE],vtVarSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.lStd**2,MPI.DOUBLE],lVarSum,MPI.SUM,root=0)

    comm.Reduce([sim.sim_result.vHalfMean,MPI.DOUBLE_COMPLEX],vHalfSum,MPI.SUM,root=0)
    
    
    indexCumm = np.arange(1,len(cummU)+1)
    cummMeanU = cummU.T/indexCumm
    cummMeanU = cummMeanU.T
    comm.Reduce([cummMeanU,MPI.DOUBLE_COMPLEX],cummMeanUSum,MPI.SUM,root=0)

    # acceptancePercentage = comm.reduce(sim.accepted_count/sim.n_samples, op=MPI.MIN, root=0)
    # beta = comm.reduce(beta, op=MPI.MIN, root=0)

    print('Rank {0:d} sending analysis completed'.format(rank))
    sys.stdout.flush()

    if rank==0:
        #averaging
        print('Rank {0:d} computed vtMeanAll, vHalfMeanAll, lMeanAll'.format(rank))
        sys.stdout.flush()
        vtMeanAll = vtEsSum/size
        vHalfMeanAll = vHalfSum/size
        lMeanAll = lUSum/size
        
        cummMeanU = cummMeanUSum/size
        indexCumm = np.arange(1,len(cummMeanU)+1)

    else:
        print('Rank {0:d} allocate empty arrays for vtMeanAll, vHalfMeanAll, lMeanAll'.format(rank))
        sys.stdout.flush()
        vHalfMeanAll = np.empty(sim.sim_result.vHalfMean.shape,dtype=np.complex128)
        vtMeanAll = np.empty(sim.sim_result.vtF.shape,dtype=np.float64)   
        lMeanAll = np.empty(sim.sim_result.vtF.shape,dtype=np.float64)
    
    print('Rank {0:d} broadcast vtMeanAll, vHalfMeanAll, lMeanAll from rank 0'.format(rank))
    sys.stdout.flush()
    # #Broadcast mean of all
    comm.Bcast([vHalfMeanAll,MPI.DOUBLE_COMPLEX],root=0)
    comm.Bcast([vtMeanAll,MPI.DOUBLE],root=0)
    comm.Bcast([lMeanAll,MPI.DOUBLE],root=0)

    print('Rank {0:d} computing vHalfVar, vtVar,lVar'.format(rank))
    sys.stdout.flush()
    #Variance corrections
    vHalfVarRealCorrection = (sim.sim_result.vHalfMean.real - vHalfMeanAll.real)**2
    vHalfVarImagCorrection = (sim.sim_result.vHalfMean.imag - vHalfMeanAll.imag)**2
    vtVarCorrection = (sim.sim_result.vtMean - vtMeanAll)**2
    lVarCorrection = (sim.sim_result.lMean - lMeanAll)**2

    comm.Reduce([vHalfVarRealCorrection,MPI.DOUBLE],vHalfVarRealSumCorr,MPI.SUM,root=0)
    comm.Reduce([vHalfVarImagCorrection,MPI.DOUBLE],vHalfVarImagSumCorr,MPI.SUM,root=0)
    comm.Reduce([vtVarCorrection,MPI.DOUBLE],vtVarSumCorr,MPI.SUM,root=0)
    comm.Reduce([lVarCorrection,MPI.DOUBLE],lVarSumCorr,MPI.SUM,root=0)
    
    if rank==0:
        try:
            print('Final Analysis Completed at Rank 0')
            sys.stdout.flush()
            vHalfStdReal = np.sqrt((vHalfVarRealSum+vHalfVarRealSumCorr)/size)
            vHalfStdImag = np.sqrt((vHalfVarImagSum+vHalfVarImagSumCorr)/size)
            vtStd = np.sqrt((vtVarSum+vtVarSumCorr)/size)
            lStd = np.sqrt((lVarSum+lVarSumCorr)/size)
            sim_result0 = sr.SimulationResult(sim.sim_result.vtHalf,sim.sim_result.vtF,vHalfMeanAll,vHalfStdReal,vHalfStdImag,lMeanAll,lStd,vtMeanAll,vtStd)
            print('Plotting Result at Rank 0')
            sys.stdout.flush()
            sim.sim_result = sim_result0
            p.plotResult(sim,indexCumm=indexCumm,cummMeanU=cummMeanU,showFigures=False)
            print('Plotting Completed Rank 0: Enjoy your day')
            sys.stdout.flush()
        except Exception:
            print('Ups...')
            sys.stdout.flush()
        finally:
            comm.Disconnect()
        

    
        

if __name__=='__main__':
    # n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,uHalfInit=None,
    #                 kappa = 1e17,sigma_u = 5e6,sigma_v = 10,printInterval = 100,
    #                 seed=1,burnPercentage = 5,useLaTeX=True,randVectInitiated=True,
    #                 showFigures=True
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',default=2**6,type=int,help='number of Fourier basis, Default=64')
    parser.add_argument('--num',default=2**8,type=int,help='number measurement points, Default=256')
    parser.add_argument('--n-samples',default=1000,type=int,help='number of MCMC samples per computer core, Default=10000')
    parser.add_argument('--evaluation-interval',default=100,type=int,help='interval to print and reevaluate beta, Default=100')
    parser.add_argument('--beta',default=0.2,type=float,help='preconditioned Crank Nicholson beta parameter, Default=0.2')
    parser.add_argument('--kappa',default=1e17,type=float,help='kappa constant for u_t, Default=1e17')
    parser.add_argument('--sigma-u',default=5e6,type=float,help='Sigma_u constant, Default=5e6')
    parser.add_argument('--sigma-v',default=1e2,type=float,help='Sigma_v constant, Default=10.0')
    parser.add_argument('--burn-percentage',default=5.0,type=float,help='Burn Percentage, Default=5.0')
    # parser.add_argument('--use-latex',default=True,type=bool,help='Whether to use Latex for plotting or not, Default=True')
    # parser.add_argument('--show-figures',default=True,type=bool,help='Whether to show simulation results figures or not, Default=True')
    # parser.add_argument('--rand-vect-init',default=False,type=bool,help='Whether to preallocate random vector before running MCMC, Default=False')
    # parser.add_argument('--init-file',default=None,type=str,help='Hdf5 file containing data set called cummMeanU, with size match parameter n, Default=None')
    # parser.add_argument('--pcn',default=False,type=bool,help='Whether using preconditioned Crank Nicholson algorithm or HMC, Default=False')

    args = parser.parse_args()

    


    runSimulations( n_samples = args.n_samples,n = args.n,beta = args.beta,num = args.num,
                    kappa = args.kappa,sigma_u = args.sigma_u,sigma_v = args.sigma_v,evaluation_interval = args.evaluation_interval,
                    burnPercentage = args.burn_percentage)