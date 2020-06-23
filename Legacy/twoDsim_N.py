#%%
#! python
import h5py
import matplotlib.pyplot as plt
import mcmc.image_cupy_N as im
import mcmc.plotting as p
import numpy as np
import scipy.linalg as sla
import scipy.special as ssp
import mcmc.util_cupy as util
import cupy as cp
import importlib
import datetime
import pathlib,os
import argparse
import parser_help as ph
import post_analysis2 as pa
import subprocess
from skimage.transform import iradon


#%%
def initialize_using_FBP(sim):
    #Set initial condition of fourier index as the fourier index of FBP
    FBP_image = iradon(cp.asnumpy(sim.measurement.sinogram),theta=cp.asnumpy(sim.measurement.theta),circle=True)
    if sim.use_max_H:
        f = im.FourierAnalysis_2D(sim.fourier.basis_number,256,0.,1.)
        fbp_fHalf = f.rfft2(cp.asarray(FBP_image,dtype=cp.float32))
    else:
        fbp_fHalf = sim.fourier.rfft2(cp.asarray(FBP_image,dtype=cp.float32))
    fbp_fSym2D = util.symmetrize_2D(fbp_fHalf)
    fbp_fSym = fbp_fSym2D.ravel(im.ORDER)

    sim.Layers[-1].current_sample_sym = fbp_fSym
    sim.Layers[-1].current_sample =  sim.Layers[-1].current_sample_sym[sim.fourier.basis_number_2D_ravel-1:]
    sim.Layers[-1].record_sample()

def initialize_from_folder(target_folder,sim,sequence_no):
    #TODO:HARD CODED relative path BADDD    
    relative_path = pathlib.Path("/scratch/work/emzirm1/SimulationResult")
    init_folder = relative_path /target_folder
    file_name = 'result_{}.hdf5'.format(sequence_no)
    init_file_path = init_folder/file_name
    if not init_file_path.exists():
        initialize_using_FBP(sim)
    else:
        
        with h5py.File(init_file_path,mode='r') as file:
            n_layers = file['n_layers'][()]
            sim.pcn.beta = file['pcn/beta'][()]
            for i in range(n_layers):
                samples_history = file['Layers {}/samples_history'.format(i)][()]
                init_Sym = util.symmetrize(cp.asarray(samples_history[-1]))
                del samples_history
                sim.Layers[i].current_sample_sym = init_Sym
                sim.Layers[i].current_sample =  sim.Layers[i].current_sample_sym[sim.fourier.basis_number_2D_ravel-1:]
                sim.Layers[i].record_sample()              
                        
        
        




#%%
if __name__=='__main__':
    # n_samples = 1000,n = 2**6,beta = 2e-1,num = 2**8,uHalfInit=None,
    #                 kappa = 1e17,sigma_u = 5e6,sigma_v = 10,printInterval = 100,
    #                 seed=1,burnPercentage = 5,useLaTeX=True,randVectInitiated=True,
    #                 showFigures=True
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter',default=0,type=int,help='number of iteration, Default=0')
    parser.add_argument('--seq-no',default=0,type=int,help='sequence number, Default=0')
    parser.add_argument('--n-layers',default=2,type=int,help='number SPDE layers, Default=2')
    parser.add_argument('--n-theta',default=9,type=int,help='number theta, Default=50')
    parser.add_argument('--n',default=32,type=int,help='number of Fourier basis, Default=32')
    parser.add_argument('--seed',default=1,type=int,help='random generator seed, Default=1')
    parser.add_argument('--n-extended',default=64,type=int,help='number of point per axis, Default=64')
    parser.add_argument('--n-samples',default=1000,type=int,help='number of MCMC samples per computer core, Default=100')
    parser.add_argument('--evaluation-interval',default=10,type=int,help='interval to print and reevaluate beta, Default=5')
    parser.add_argument('--beta',default=1,type=float,help='preconditioned Crank Nicholson beta parameter, Default=1')
    parser.add_argument('--kappa',default=5e9,type=float,help='kappa constant for u_t, Default=1e9')
    parser.add_argument('--chol-epsilon',default=1e-6,type=float,help='epsilon to ensure cholesky factorization always result in PD, Default=1e-6')
    parser.add_argument('--sigma-0',default=1e9,type=float,help='Sigma_u constant, Default=1e7')
    parser.add_argument('--sigma-v',default=3.2e4,type=float,help='Sigma_v constant, Default=1e4')
    parser.add_argument('--meas-std',default=0.2,type=float,help='Measurement stdev, Default=0.2')
    parser.add_argument('--sigma-scaling',default=4e-1,type=float,help='Sigma_scaling constant, Default=1e-3')
    parser.add_argument('--burn-percentage',default=25.0,type=float,help='Burn Percentage, Default=25.0')
    parser.add_argument('--variant',default="dunlop",type=str,help='preconditioned Crank Nicholson multilayered algorithm variant, Default=dunlop')
    parser.add_argument('--phantom-name',default="shepp.png",type=str,help='Phantom name, Default=shepp.png')
    parser.add_argument('--meas-type',default="tomo",type=str,help='Two D Measurement, Default=tomo')
    parser.add_argument('--init-folder',default="",type=str,help='Initial condition for the states, Default=empty')
    ph.add_boolean_argument(parser,'enable-beta-feedback',default=True,messages='Whether beta-feedback will be enabled, Default=True')
    ph.add_boolean_argument(parser,'print-progress',default=True,messages='Whether progress is printed, Default=True')
    ph.add_boolean_argument(parser,'verbose',default=True,messages='Verbose mode, Default=True')
    ph.add_boolean_argument(parser,'hybrid',default=False,messages='Use both GPU and CPU memory, Default=False')
    ph.add_boolean_argument(parser,'NMatrix',default=False,messages='Use Nmatrix instead of L, Default=False')
    ph.add_boolean_argument(parser,'MaxH',default=False,messages='Use maximum possible target image size (511), Default=False')

    args = parser.parse_args()
    if args.n_theta == 18:
        args.seed = 2
        
    sim = im.Simulation(n_layers=args.n_layers,n_samples = args.n_samples,n = args.n,n_extended = args.n_extended,beta = args.beta,
                    kappa = args.kappa,sigma_0 = args.sigma_0,sigma_v = args.sigma_v,sigma_scaling=args.sigma_scaling,meas_std=args.meas_std,evaluation_interval = args.evaluation_interval,printProgress=args.print_progress,
                    seed=args.seed,burn_percentage = args.burn_percentage,enable_beta_feedback=args.enable_beta_feedback,pcn_variant=args.variant,phantom_name=args.phantom_name
                    ,meas_type=args.meas_type,n_theta=args.n_theta,verbose=args.verbose,hybrid_GPU_CPU=args.hybrid,use_NMatrix=args.NMatrix,use_max_H=args.MaxH)
    
    # set pcn epsilon for cholesky
    sim.pcn.set_chol_epsilon(args.chol_epsilon)
    # sim.pcn.target_acceptance_rate = 0.5#change acceptance rate to 50%

    #only create result folder for the first sequence
    if args.seq_no == 0:    
        folderName = 'result-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M_%S')
    else:
        folderName = args.init_folder





    if args.seq_no == 0:
        initialize_using_FBP(sim)    
    else:
        initialize_from_folder(args.init_folder,sim,args.seq_no-1)
        
    print("Used bytes so far, before even running the simulation {}".format(sim.mempool.used_bytes()))
    sim.run()

    if 'WRKDIR' in os.environ:
            simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult'/folderName
    elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult').exists():
        simResultPath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult')/folderName
    else:
        simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
    if not simResultPath.exists():
        simResultPath.mkdir()
    
    file_name = 'result_{}.hdf5'.format(args.seq_no)
    sim.save(str(simResultPath/file_name))
    print('simulation result stored in {}'.format(simResultPath/file_name))

    if args.seq_no < args.iter:
        if sim.use_max_H:
            if args.n_layers == 3:
                #Run next batch
                subprocess.run(['sbatch','runner_nL3_maxH.sh',str(args.iter),str(args.seq_no+1),str(args.n),
                                str(args.n_extended),str(args.n_theta),str(args.n_samples),str(args.sigma_v),
                                str(args.sigma_scaling),folderName])
            elif args.n_layers ==2:
                    #Run next batch
                subprocess.run(['sbatch','runner_nL2_maxH.sh',str(args.iter),str(args.seq_no+1),str(args.n),
                                str(args.n_extended),str(args.n_theta),str(args.n_samples),str(args.sigma_v),
                                str(args.sigma_scaling),folderName])
        else:
            if args.n_layers == 3:
                #Run next batch
                subprocess.run(['sbatch','runner_nL3.sh',str(args.iter),str(args.seq_no+1),str(args.n),
                                str(args.n_extended),str(args.n_theta),str(args.n_samples),str(args.sigma_v),
                                str(args.sigma_scaling),folderName])
            elif args.n_layers ==2:
                    #Run next batch
                subprocess.run(['sbatch','runner_nL2.sh',str(args.iter),str(args.seq_no+1),str(args.n),
                                str(args.n_extended),str(args.n_theta),str(args.n_samples),str(args.sigma_v),
                                str(args.sigma_scaling),folderName])
    else:
        #do post_analysis on the last chain
        pa.post_analysis(folderName,simResultPath.parent,filename=file_name,)


    # 
    #do analysis offline
    #pa.post_analysis(folderName,simResultPath.parent)
    