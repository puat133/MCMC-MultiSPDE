import matplotlib.pyplot as plt
# import numba as nb
import mcmc.util as util
import pathlib
import seaborn as sns
def plotResult(simResults,simResultPath,useLaTeX=True,showFigures=True):
    #unpack simResults
    vtHalf = simResults.vtHalf
    vtF = simResults.vtF
    # ut = simResults['ut']
    # lU = simResults['lU']
    vHalfMean = simResults.vHalfMean
    vHalfStdReal = simResults.vHalfStdReal
    vHalfStdImag = simResults.vHalfStdImag
    vtMean = simResults.vtMean
    vtStd = simResults.vtStd
    lMean = simResults.lMean
    lStd = simResults.lStd
    # cummU = simResults['cummU']
    indexCumm = simResults.indexCumm
    cummMeanU = simResults.cummMeanU
    # uHistory = simResults['uHistory']
    # vHistory = simResults['vHistory']
    
    


    n = simResults.fourier_basis_number
    numNew = simResults.fourier_extended_basis_number

    t = simResults.t
    tNew = simResults.t
    sigmas = floatVectParams['sigmas']
    vt =  floatVectParams['vt']
    y =  floatVectParams['y']

    sns.set_style("ticks")
    sns.set_context('paper')

    #clear figure
    plt.clf()
    if useLaTeX:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    # plt.ion()
    # plt.figure()
    plt.plot(tNew,vtMean,'-b',linewidth=0.5)
    plt.fill_between(tNew,vtMean-2*vtStd,vtMean+2*vtStd, color='b', alpha=0.1)
    plt.plot(t,y,'.k',linewidth=0.5,markersize=1)
    plt.plot(t,vt,':k',tNew,vtF,'-r',linewidth=0.5,markersize=1)
    plt.ylabel('$v(t)$')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig(str(simResultPath/'upsilon.pdf'), bbox_inches='tight')
    # plt.show()


    # kIFMean = kappaU.mean(axis=1)
    # kIFStd = kappaU.std(axis=1)

    plt.figure()
    plt.semilogy(tNew,lMean,'-b',linewidth=0.5)
    plt.fill_between(tNew,lMean-2*lStd,lMean+2*lStd, color='b', alpha=.1)
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
    if showFigures:
        plt.show()

def saveResult(simResults,simResultPath):
    # folderName = 'result - '+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M')
    # simulationResultPath = pathlib.Path.home() / 'Documents' / folderName
    # simulationResultPath.mkdir() 
    filePath = simResultPath / 'result.hdf5'
    util.saveToH5(filePath,simResults)
    # util.saveToMatlab(str(filePath),locals(),do_compression=False)