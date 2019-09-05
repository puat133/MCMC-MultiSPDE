import matplotlib.pyplot as plt
# import numba as nb
import mcmc.util as util
import numpy as np
import pathlib
import seaborn as sns
import pathlib
import datetime
import os
from colorsys import hls_to_rgb
def plotResult(sim,indexCumm=None,cummMeanU=None,simResultPath=None,useLaTeX=True,showFigures=False,save_hdf5=True,include_history=False):
    #unpack sim.sim_result
    vtHalf = sim.sim_result.vtHalf
    vtF = sim.sim_result.vtF
    # ut = sim.sim_result['ut']
    # lU = sim.sim_result['lU']
    
        
    if np.any(indexCumm==None) or np.any(cummMeanU==None):
        startIndex = np.int(sim.burn_percentage*sim.Layers[0].samples_history.shape[0]//100)
        cummU = np.cumsum(sim.Layers[0].samples_history[startIndex:,:],axis=0)
        indexCumm = np.arange(1,len(cummU)+1)
        cummMeanU = cummU.T/indexCumm
        cummMeanU = cummMeanU.T
    # uHistory = sim.sim_result['uHistory']
    # vHistory = sim.sim_result['vHistory']
    if simResultPath == None:
        folderName = 'result-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M')
        if 'WRKDIR' in os.environ:
            simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult'/folderName
        else:
            simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
        simResultPath.mkdir()
        


        # n = sim.fourier.basis_number
        # numNew = sim.fourier.extended_basis_number
    if save_hdf5:
        #save Simulation Result
        sim.save(str(simResultPath/'result.hdf5'),include_history)
    t = sim.pcn.measurement.t
    # tNew = sim.sim_result.t
    # sigmas = util.sigmasLancos(n)
    vt =  sim.pcn.measurement.vt
    y =  sim.pcn.measurement.yt

    sns.set_style("ticks")
    sns.set_context('paper')

    #clear figure
    plt.clf()
    if useLaTeX:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    plt.figure()
    plt.plot(cummMeanU.real,linewidth=0.5)
    plt.xlabel('$n$ simulation')
    plt.ylabel('Real components of cummulative average of $u$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'uCummReal.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(cummMeanU.imag,linewidth=0.5)
    plt.xlabel('$n$ simulation')
    plt.ylabel('Imaginary components of cummulative average of $u$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'uCummImag.pdf'), bbox_inches='tight')
    plt.close()

    for j in range(sim.n_layers):
        uHalfMean = sim.sim_result.uHalfMean[j,:]
        uHalfStdReal = sim.sim_result.uHalfStdReal[j,:]
        uHalfStdImag = sim.sim_result.uHalfStdImag[j,:]
        utMean = sim.sim_result.utMean[j,:]
        utStd = sim.sim_result.utStd[j,:]
        elltMean = sim.sim_result.elltMean[j,:]
        elltStd = sim.sim_result.elltStd[j,:]
        # cummU = sim.sim_result['cummU']

        # plt.ion()
        plt.figure()
        plt.plot(t,utMean,'-b',linewidth=0.5)
        plt.fill_between(t,utMean-2*utStd,utMean+2*utStd, color='b', alpha=0.1)
        if j == sim.n_layers-1:
            plt.plot(t,y,'.k',linewidth=0.5,markersize=1)
            plt.plot(t,vt,':k',t,vtF,'-r',linewidth=0.5,markersize=1)
            plt.ylabel('$v(t)$')
        else:
            plt.ylabel('$u(t)$')
        plt.xlabel('$t$')
        plt.tight_layout()
        if j == sim.n_layers-1:
            plt.savefig(str(simResultPath/'upsilont.pdf'), bbox_inches='tight')
        else:
            plt.savefig(str(simResultPath/'ut-{0}.pdf'.format(j)), bbox_inches='tight')
        plt.close()


        # kIFMean = kappaU.mean(axis=1)
        # kIFStd = kappaU.std(axis=1)
        if j<sim.n_layers-1:
            plt.figure()
            plt.semilogy(t,elltMean,'-b',linewidth=0.5)
            plt.fill_between(t,elltMean-2*elltStd,elltMean+2*elltStd, color='b', alpha=.1)
            plt.ylabel(r'$\ell(t)$')
            plt.xlabel('$t$')
            plt.tight_layout()
            if j == sim.n_layers-2:
                plt.savefig(str(simResultPath/'ell-upsilon.pdf'), bbox_inches='tight')
            else:
                plt.savefig(str(simResultPath/'ell-{0}.pdf'.format(j+1)), bbox_inches='tight')
            plt.close()
        

        
        iHalf = np.arange(len(vtHalf))
        plt.figure()
        if j == sim.n_layers-1:
            plt.plot(iHalf,vtHalf.real, '-r', iHalf, uHalfMean.real,'-b', linewidth=0.5)
        else:
            plt.plot(iHalf, uHalfMean.real,'-b', linewidth=0.5)
        plt.fill_between(iHalf,uHalfMean.real-2*uHalfStdReal,uHalfMean.real+2*uHalfStdReal, color='b', alpha=.1)
        plt.xlabel(r'Frequency $2 \pi n$')
        plt.ylabel('Real components of $v_n$')
        plt.tight_layout()
        if j == sim.n_layers-1:
            plt.savefig(str(simResultPath/'vComponentReal.pdf'), bbox_inches='tight')
        else:
            plt.savefig(str(simResultPath/'uComponentReal-{0}.pdf'.format(j)), bbox_inches='tight')
        plt.close()
        
        plt.figure()
        if j == sim.n_layers-1:
            plt.plot(iHalf,vtHalf.imag, '-r', iHalf, uHalfMean.imag,'-b', linewidth=0.5)
        else:
            plt.plot(iHalf, uHalfMean.imag,'-b', linewidth=0.5)
        plt.fill_between(iHalf,uHalfMean.imag-2*uHalfStdImag,uHalfMean.imag+2*uHalfStdImag, color='b', alpha=.1)
        plt.xlabel(r'Frequency $2 \pi n$')
        plt.ylabel('Imaginary components of $v_n$')
        plt.tight_layout()
        if j == sim.n_layers-1:
            plt.savefig(str(simResultPath/'vComponentImag.pdf'), bbox_inches='tight')
        else:
            plt.savefig(str(simResultPath/'uComponentImag-{0}.pdf'.format(j)), bbox_inches='tight')
        plt.close()

        plt.figure()
        uHalfStdAbs = np.sqrt(uHalfStdImag**2 + uHalfStdReal**2)
        if j == sim.n_layers-1:
            plt.plot(iHalf,np.abs(vtHalf),'-r',iHalf,np.abs(uHalfMean),'-b',linewidth=0.5)
        else:
            plt.plot(iHalf,np.abs(uHalfMean),'-b',linewidth=0.5)
        plt.fill_between(iHalf,np.abs(uHalfMean)-2*uHalfStdAbs,np.abs(uHalfMean)+2*uHalfStdAbs, color='b', alpha=0.1)
        plt.ylabel(r'Absolute value of $v_n$')
        plt.xlabel(r'Frequency $2 \pi n$')
        plt.tight_layout()
        if j == sim.n_layers-1: 
            plt.savefig(str(simResultPath/'vComponentAbs.pdf'), bbox_inches='tight')
        else:
            plt.savefig(str(simResultPath/'uComponentAbs-{0}.pdf'.format(j)), bbox_inches='tight')
        plt.close()
        
    print("Plotting complete")
    if showFigures:
        plt.show()



"""
https://stackoverflow.com/a/20958684/11764120
"""
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c
