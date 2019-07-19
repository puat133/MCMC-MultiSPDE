import matplotlib.pyplot as plt
# import numba as nb
import mcmc.util as util
import numpy as np
import pathlib
import seaborn as sns
import pathlib
import datetime
def plotResult(sim,indexCumm=None,cummMeanU=None,simResultPath=None,useLaTeX=True,showFigures=False):
    #unpack sim.sim_result
    vtHalf = sim.sim_result.vtHalf
    vtF = sim.sim_result.vtF
    # ut = sim.sim_result['ut']
    # lU = sim.sim_result['lU']
    vHalfMean = sim.sim_result.vHalfMean
    vHalfStdReal = sim.sim_result.vHalfStdReal
    vHalfStdImag = sim.sim_result.vHalfStdImag
    vtMean = sim.sim_result.vtMean
    vtStd = sim.sim_result.vtStd
    lMean = sim.sim_result.lMean
    lStd = sim.sim_result.lStd
    # cummU = sim.sim_result['cummU']
    
    if np.any(indexCumm==None) or np.any(cummMeanU==None):
        startIndex = np.int(sim.burn_percentage*sim.n_samples//100)
        cummU = np.cumsum(sim.Layers[-2].samples_history[startIndex:,:],axis=0)
        indexCumm = np.arange(1,len(cummU)+1)
        cummMeanU = cummU.T/indexCumm
        cummMeanU = cummMeanU.T
    # uHistory = sim.sim_result['uHistory']
    # vHistory = sim.sim_result['vHistory']
    if simResultPath == None:
        folderName = 'result-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M')
        simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
        simResultPath.mkdir()
    


    # n = sim.fourier.fourier_basis_number
    # numNew = sim.fourier.fourier_extended_basis_number

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

    # plt.ion()
    # plt.figure()
    plt.plot(t,vtMean,'-b',linewidth=0.5)
    plt.fill_between(t,vtMean-2*vtStd,vtMean+2*vtStd, color='b', alpha=0.1)
    plt.plot(t,y,'.k',linewidth=0.5,markersize=1)
    plt.plot(t,vt,':k',t,vtF,'-r',linewidth=0.5,markersize=1)
    plt.ylabel('$v(t)$')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'upsilon.pdf'), bbox_inches='tight')
    # plt.show()


    # kIFMean = kappaU.mean(axis=1)
    # kIFStd = kappaU.std(axis=1)

    plt.figure()
    plt.semilogy(t,lMean,'-b',linewidth=0.5)
    plt.fill_between(t,lMean-2*lStd,lMean+2*lStd, color='b', alpha=.1)
    plt.ylabel(r'$\ell(t)$')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'ell.pdf'), bbox_inches='tight')

    plt.figure()
    plt.plot(cummMeanU.real,linewidth=0.5)
    plt.xlabel('$n$ simulation')
    plt.ylabel('Real components of cummulative average of $u$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'uCummReal.pdf'), bbox_inches='tight')

    plt.figure()
    plt.plot(cummMeanU.imag,linewidth=0.5)
    plt.xlabel('$n$ simulation')
    plt.ylabel('Imaginary components of cummulative average of $u$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'uCummImag.pdf'), bbox_inches='tight')

    
    iHalf = np.arange(len(vtHalf))
    plt.figure()
    plt.plot(iHalf,vtHalf.real, '-r', iHalf, vHalfMean.real,'-b', linewidth=0.5)
    plt.fill_between(iHalf,vHalfMean.real-2*vHalfStdReal,vHalfMean.real+2*vHalfStdReal, color='b', alpha=.1)
    plt.xlabel(r'Frequency $2 \pi n$')
    plt.ylabel('Real components of $v_n$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'vComponentReal.pdf'), bbox_inches='tight')

    plt.figure()
    plt.plot(iHalf,vtHalf.imag, '-r', iHalf, vHalfMean.imag,'-b', linewidth=0.5)
    plt.fill_between(iHalf,vHalfMean.imag-2*vHalfStdImag,vHalfMean.imag+2*vHalfStdImag, color='b', alpha=.1)
    plt.xlabel(r'Frequency $2 \pi n$')
    plt.ylabel('Imaginary components of $v_n$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'vComponentImag.pdf'), bbox_inches='tight')

    plt.figure()
    vHalfStdAbs = np.sqrt(vHalfStdImag**2 + vHalfStdReal**2)
    plt.plot(iHalf,np.abs(vtHalf),'-r',iHalf,np.abs(vHalfMean),'-b',linewidth=0.5)
    plt.fill_between(iHalf,np.abs(vHalfMean)-2*vHalfStdAbs,np.abs(vHalfMean)+2*vHalfStdAbs, color='b', alpha=0.1)
    plt.ylabel(r'Absolute value of $v_n$')
    plt.xlabel(r'Frequency $2 \pi n$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'vComponentAbs.pdf'), bbox_inches='tight')

    print("Plotting complete")
    if showFigures:
        plt.show()

