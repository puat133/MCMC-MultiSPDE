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

def runSimulations(n_layers,n_samples,n,beta,num,kappa,sigma_0,sigma_v,sigma_scaling,evaluation_interval,printProgress,
seed,burn_percentage,pcn_pair_layers,enable_beta_feedback,use_latex):

    #initialize parameter                   
    
    
    #prepare MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    sim = s.Simulation(n_layers=n_layers,n_samples =n_samples,n =n,beta =beta,num =num,kappa =kappa,sigma_0 =sigma_0,sigma_v =sigma_v,sigma_scaling=sigma_scaling,          
                        evaluation_interval =evaluation_interval,printProgress=False,seed=seed-1+rank,burn_percentage = burn_percentage,pcn_pair_layers=pcn_pair_layers,enable_beta_feedback=enable_beta_feedback)

        
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

    utSum = None
    elltSum = None

    uHalfSum = None
    cummMeanUSum = None
    
    uHalfVarRealSum = None
    uHalfVarImagSum = None
    utVarSum = None
    elltVarSum = None
    uHalfVarRealSumCorr = None
    uHalfVarImagSumCorr = None
    utVarSumCorr = None
    elltVarSumCorr = None

    uHalfMeanAll = None
    utMeanAll = None
    elltMeanAll = None

    startIndex = np.int(sim.burn_percentage*sim.n_samples//100)
    cummU = np.cumsum(sim.Layers[0].samples_history[startIndex:,:],axis=0)

    # #set buffer
    if rank==0:
        utSum     = np.empty(sim.sim_result.utMean.shape,dtype=np.float64)
        elltSum     = np.empty(sim.sim_result.elltMean.shape,dtype=np.float64)
            
        uHalfVarRealSum = np.empty(sim.sim_result.uHalfStdReal.shape,dtype=np.float64)
        uHalfVarImagSum = np.empty(sim.sim_result.uHalfStdImag.shape,dtype=np.float64)
        utVarSum = np.empty(sim.sim_result.utStd.shape,dtype=np.float64)
        elltVarSum = np.empty(sim.sim_result.elltStd.shape,dtype=np.float64)
        
        uHalfSum     = np.empty(sim.sim_result.uHalfMean.shape,dtype=np.complex128)
        cummMeanUSum     = np.empty(cummU.shape,dtype=np.complex128)

        uHalfVarRealSumCorr = np.empty(sim.sim_result.uHalfStdReal.shape,dtype=np.float64)
        uHalfVarImagSumCorr = np.empty(sim.sim_result.uHalfStdImag.shape,dtype=np.float64)
        utVarSumCorr = np.empty(sim.sim_result.utStd.shape,dtype=np.float64)
        elltVarSumCorr = np.empty(sim.sim_result.elltStd.shape,dtype=np.float64)
    
    
        

    #Doing REDUCE SUM HERE --> TO BE AVERAGED IN RANK 0
    comm.Reduce([sim.sim_result.utMean,MPI.DOUBLE],utSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.elltMean,MPI.DOUBLE],elltSum,MPI.SUM,root=0)
    

    comm.Reduce([sim.sim_result.uHalfStdReal**2,MPI.DOUBLE],uHalfVarRealSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.uHalfStdImag**2,MPI.DOUBLE],uHalfVarImagSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.utStd**2,MPI.DOUBLE],utVarSum,MPI.SUM,root=0)
    comm.Reduce([sim.sim_result.elltStd**2,MPI.DOUBLE],elltVarSum,MPI.SUM,root=0)

    comm.Reduce([sim.sim_result.uHalfMean,MPI.DOUBLE_COMPLEX],uHalfSum,MPI.SUM,root=0)
    
    
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
        print('Rank {0:d} computed utMeanAll, uHalfMeanAll, elltMeanAll'.format(rank))
        sys.stdout.flush()
        utMeanAll = utSum/size
        uHalfMeanAll = uHalfSum/size
        elltMeanAll = elltSum/size
        
        cummMeanU = cummMeanUSum/size
        indexCumm = np.arange(1,len(cummMeanU)+1)

    else:
        print('Rank {0:d} allocate empty arrays for utMeanAll, uHalfMeanAll, elltMeanAll'.format(rank))
        sys.stdout.flush()
        uHalfMeanAll = np.empty(sim.sim_result.uHalfMean.shape,dtype=np.complex128)
        utMeanAll = np.empty(sim.sim_result.utMean.shape,dtype=np.float64)   
        elltMeanAll = np.empty(sim.sim_result.elltMean.shape,dtype=np.float64)
    
    print('Rank {0:d} broadcast utMeanAll, uHalfMeanAll, elltMeanAll from rank 0'.format(rank))
    sys.stdout.flush()
    # #Broadcast mean of all
    comm.Bcast([uHalfMeanAll,MPI.DOUBLE_COMPLEX],root=0)
    comm.Bcast([utMeanAll,MPI.DOUBLE],root=0)
    comm.Bcast([elltMeanAll,MPI.DOUBLE],root=0)

    print('Rank {0:d} computing uHalfVar, utVar,elltVar'.format(rank))
    sys.stdout.flush()
    #Variance corrections
    uHalfVarRealCorrection = (sim.sim_result.uHalfMean.real - uHalfMeanAll.real)**2
    uHalfVarImagCorrection = (sim.sim_result.uHalfMean.imag - uHalfMeanAll.imag)**2
    utVarCorrection = (sim.sim_result.utMean - utMeanAll)**2
    elltVarCorrection = (sim.sim_result.elltMean - elltMeanAll)**2

    comm.Reduce([uHalfVarRealCorrection,MPI.DOUBLE],uHalfVarRealSumCorr,MPI.SUM,root=0)
    comm.Reduce([uHalfVarImagCorrection,MPI.DOUBLE],uHalfVarImagSumCorr,MPI.SUM,root=0)
    comm.Reduce([utVarCorrection,MPI.DOUBLE],utVarSumCorr,MPI.SUM,root=0)
    comm.Reduce([elltVarCorrection,MPI.DOUBLE],elltVarSumCorr,MPI.SUM,root=0)

    try:
        if rank==0:

            print('Final Analysis Completed at Rank 0')
            sys.stdout.flush()
            uHalfStdReal = np.sqrt((uHalfVarRealSum+uHalfVarRealSumCorr)/size)
            uHalfStdImag = np.sqrt((uHalfVarImagSum+uHalfVarImagSumCorr)/size)
            utStd = np.sqrt((utVarSum+utVarSumCorr)/size)
            elltStd = np.sqrt((elltVarSum+elltVarSumCorr)/size)
            sim_result0 = sr.SimulationResult()
            sim_result0.assign_values(sim.sim_result.vtHalf,sim.sim_result.vtF,uHalfMeanAll,uHalfStdReal,uHalfStdImag,elltMeanAll,elltStd,utMeanAll,utStd)
            
            print('Plotting Result at Rank 0')
            sys.stdout.flush()
            sim.sim_result = sim_result0
            # p.plotResult(sim,indexCumm=indexCumm,cummMeanU=cummMeanU,showFigures=False)
            p.plotResult(sim,include_history=False,useLaTeX=use_latex)
            print('Plotting Completed Rank 0: Enjoy your day')
            sys.stdout.flush()
    except Exception:
        print('Ups problem in rank {0} '.format(rank))
        sys.stdout.flush()
        raise Exception
    finally:
        print('Disconnecting rank {0} '.format(rank))
        sys.stdout.flush()
        MPI.Finalize()
        # comm.Disconnect()
        
        

        

    
        

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
    parser.add_argument('--sigma-0',default=5e6,type=float,help='Sigma_0 constant, Default=5e6')
    parser.add_argument('--sigma-v',default=1e2,type=float,help='Sigma_v constant, Default=10.0')
    parser.add_argument('--sigma-scaling',default=1e-4,type=float,help='Sigma_scaling constant, Default=1e-4')
    parser.add_argument('--burn-percentage',default=25.0,type=float,help='Burn Percentage, Default=25.0')
    parser.add_argument('--include-history',default=False,type=bool,help='Whether to include Layer simulation history in hdf5, Default=False')
    parser.add_argument('--pcn-pair-layers',default=False,type=bool,help='Whether pCN will be calculated each two consecutive layers, Default=False')
    parser.add_argument('--enable-beta-feedback',default=True,type=bool,help='Whether beta-feedback will be enabled, Default=False')
    parser.add_argument('--use-latex',default=False,type=bool,help='Whether to use Latex for plotting or not, Default=False')

    args = parser.parse_args()
    runSimulations(n_layers=args.n_layers,n_samples = args.n_samples,n = args.n,beta = args.beta,num = args.num,
                    kappa = args.kappa,sigma_0 = args.sigma_0,sigma_v = args.sigma_v,sigma_scaling=args.sigma_scaling,evaluation_interval = args.evaluation_interval,printProgress=False,
                    seed=args.seed,burn_percentage = args.burn_percentage,pcn_pair_layers=args.pcn_pair_layers,enable_beta_feedback=args.enable_beta_feedback,use_latex=args.use_latex)