import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.fourier as fourier
import mcmc.L as L
import mcmc.layer as layer
import mcmc.randomGenerator as randomGenerator
import mcmc.pCN as pCN
import mcmc.measurement as meas
import mcmc.simulationResults as simRes
# import scipy as scp
import time


# fourier_type = nb.deferred_type()
# fourier_type.define(fourier.FourierAnalysis.class_type.instance_type) 
# Layer_type = nb.deferred_type()
# Layer_type.define(layer.Layer.class_type.instance_type)
# L_matrix_type = nb.deferred_type()
# L_matrix_type.define(L.Lmatrix.class_type.instance_type)  
# Rand_gen_type = nb.deferred_type()
# Rand_gen_type.define(randomGenerator.RandomGenerator.class_type.instance_type)  
# pCN_type = nb.deferred_type()
# pCN_type.define(pCN.pCN.class_type.instance_type)  
# meas_type = nb.deferred_type()
# meas_type.define(meas.Measurement.class_type.instance_type) 
# sim_result_type = nb.deferred_type()
# sim_result_type.define(simRes.SimulationResult.class_type.instance_type)
# spec = [
#     ('fourier',fourier_type),
#     ('random_gen',Rand_gen_type),
#     ('pcn',pCN_type),
#     ('sim_result',sim_result_type),
#     ('Layers',Layer_type[:]),
#     # ('LMat',L_matrix_type),
#     # ('measurement',meas_type),
#     ('n_samples',nb.int64),
#     ('meas_samples_num',nb.int64),
#     ('evaluation_interval',nb.int64),
#     ('random_seed',nb.int64),
#     ('accepted_count',nb.int64),
#     ('burn_percentage',nb.float64),
#     ('total_time',nb.float64),
#     ('yBar',nb.float64[:]),
#     ('H',nb.complex128[:,:]),
#     ('u_history',nb.complex128[:,:]),
#     ('v_history',nb.complex128[:,:]),
#     ('printProgress',nb.boolean),
    
# ]

# @nb.jitclass(spec)
class Simulation():
    def __init__(self,n_layers=2,n_samples = 1000,n = 2**6,beta = 2e-3,num = 2**8,
                    kappa = 1e17,sigma_u = 5e6,sigma_v = 10,evaluation_interval = 100,printProgress=True,
                    seed=1,burn_percentage = 5.0):
        self.n_samples = n_samples
        self.meas_samples_num = num
        self.evaluation_interval = evaluation_interval
        self.burn_percentage = burn_percentage
        #set random seed
        self.random_seed = seed
        self.printProgress = printProgress
        self.n_layers=n_layers
        np.random.seed(self.random_seed)
        
        
        #setup parameters for 1 Dimensional simulation
        d = 1
        nu = 2 - d/2
        alpha = nu + d/2
        t_start = 0.0
        t_end = 1.0
        beta_u = (sigma_u**2)*(2**d * np.pi**(d/2))* 1.1283791670955126#<-- this numerical value is scp.special.gamma(alpha))/scp.special.gamma(nu)
        beta_v = beta_u*(sigma_v/sigma_u)**2
        sqrtBeta_v = np.sqrt(beta_v)
        sqrtBeta_u = np.sqrt(beta_u)
        
        f =  fourier.FourierAnalysis(n,num,t_start,t_end)
        self.fourier = f
        
        rg = randomGenerator.RandomGenerator(f.fourier_basis_number)
        self.random_gen = rg
        

        LuReal = (1/sqrtBeta_u)*(self.fourier.Dmatrix*kappa**(-nu) - kappa**(2-nu)*self.fourier.Imatrix)
        Lu = LuReal + 1j*np.zeros(LuReal.shape)
        
        uStdev = -1/np.diag(Lu)
        uStdev = uStdev[self.fourier.fourier_basis_number-1:]
        uStdev[0] /= 2 #scaled

        meas_std = 0.1
        measurement = meas.Measurement(num,meas_std,t_start,t_end)
        pcn = pCN.pCN(n_layers,rg,measurement,f,beta)
        self.pcn = pcn

        
        #initialize Layers
        # n_layers = 2
        Layers = []
        factor = 1e-4
        for i in range(self.n_layers):
            if i==0:
                init_sample = np.linalg.solve(Lu,self.random_gen.construct_w())[self.fourier.fourier_basis_number-1:]
                lay = layer.Layer(True,sqrtBeta_u,i,self.n_samples,self.pcn,init_sample)
                lay.stdev = uStdev
                lay.current_sample_scaled_norm = util.norm2(lay.current_sample/lay.stdev)#ToDO: Modify this
                lay.new_sample_scaled_norm = lay.current_sample_scaled_norm
            else:
                
                if i == n_layers-1:
                    
                    lay = layer.Layer(False,sqrtBeta_v,i,self.n_samples,self.pcn,Layers[i-1].current_sample)
                    wNew = self.pcn.random_gen.construct_w()
                    eNew = np.random.randn(self.pcn.measurement.num_sample)
                    wBar = np.concatenate((eNew,wNew))
                    
                    LBar = np.vstack((self.pcn.H,lay.LMat.current_L))

                    #update v
                    lay.current_sample_symmetrized, res, rnk, s = np.linalg.lstsq(LBar,self.pcn.yBar-wBar,rcond=-1)#,rcond=None)
                    lay.current_sample = lay.current_sample_symmetrized[self.pcn.fourier.fourier_basis_number-1:]
                else:
                    lay = layer.Layer(False,sqrtBeta_v*np.sqrt(factor),i,self.n_samples,self.pcn,Layers[i-1].current_sample)
            lay.update_current_sample()
            Layers.append(lay)
                



        self.Layers = Layers
        # self.pcn.current_sample = init_sample
        # self.pcn.LMat.construct_from(newSample)

        
        # self.H = self.measurement.get_measurement_matrix(self.fourier.fourier_basis_number)/meas_std #<-- Normalized
        # self.yBar = np.concatenate((self.measurement.yt/meas_std,np.zeros(2*self.fourier.fourier_basis_number-1)))#<-- Normalized 
        #initializing is important
        dummy_complexs = np.array([0.0+1j*1.0,1.0+1j*0.0])
        dummy_floats = np.array([1.0,1.0])

        sim_result = simRes.SimulationResult(dummy_complexs,dummy_floats,dummy_complexs,dummy_floats,dummy_floats,dummy_floats,dummy_floats,dummy_floats,dummy_floats)     
        self.sim_result = sim_result
    
    def run(self):
        
        self.accepted_count = 0
        average_time_intv =0.0
        with nb.objmode(start_time='float64',start_time_intv='float64'):
            if self.printProgress:
                util.printProgressBar(0, self.n_samples, prefix = 'Preparation . . . . ', suffix = 'Complete', length = 50)
            start_time = time.time()
            start_time_intv = start_time
        # print('preparation . . . ')
        # print('Checking random numbers normalRand & logUniform:')

        # totalTime = 0.0
        accepted_count_partial = 0
        
        for i in range(self.n_samples):#nb.prange(nSim):
            accepted_count_partial += self.pcn.oneStep(self.Layers)
            if (i+1)%(self.evaluation_interval) == 0:
                self.accepted_count += accepted_count_partial
                acceptancePercentage = self.accepted_count/(i+1)
                
                # if acceptancePercentage> 0.5:
                #     self.pcn.more_aggresive()
                # elif acceptancePercentage<0.3:
                #     self.pcn.less_aggresive()
                self.pcn.adapt_beta(acceptancePercentage)
                
                accepted_count_partial = 0
                mTime = (i+1)/(self.evaluation_interval)

                with nb.objmode(average_time_intv='float64',start_time_intv='float64'):
                
                    end_time_intv = time.time()
                    time_intv = end_time_intv-start_time_intv
                    average_time_intv +=  (time_intv-average_time_intv)/mTime
                    start_time_intv = end_time_intv
                    remainingTime = average_time_intv*((self.n_samples - i)/self.evaluation_interval)
                    remainingTimeStr = time.strftime("%j-1 day(s),%H:%M:%S", time.gmtime(remainingTime))
                    if self.printProgress:
                        util.printProgressBar(i+1, self.n_samples, prefix = 'Time Remaining {0}- Acceptance Rate {1:.2%} - Progress:'.format(remainingTimeStr,acceptancePercentage), suffix = 'Complete', length = 50)

        with nb.objmode():
            
            elapsedTimeStr = time.strftime("%j day(s),%H:%M:%S", time.gmtime(time.time()-start_time))
            self.total_time = time.time()-start_time
            # print('Complete')
            if self.printProgress:
                util.printProgressBar(self.n_samples, self.n_samples, 'Iteration Completed in {0}- Acceptance Rate {1:.2%} - Progress:'.format(elapsedTimeStr,acceptancePercentage), suffix = 'Complete', length = 50)
    

    def analyze(self):
        startIndex = np.int(self.burn_percentage*self.n_samples//100)
        
        # vtEs = np.empty((self.measurement.t.shape[0],len(vHistoryBurned)))
        # ut = np.empty((self.measurement.t.shape[0],len(uHistoryBurned)))
        # lU = np.empty((self.measurement.t.shape[0],len(uHistoryBurned)))

        vtM = np.zeros(self.pcn.measurement.t.shape,dtype=np.float64)
        vtM2 = np.zeros(self.pcn.measurement.t.shape,dtype=np.float64)
        vtCount = 0
        vtAggregateNow = (vtCount,vtM,vtM2) 

        luM = np.zeros(self.pcn.measurement.t.shape,dtype=np.float64)
        luM2 = np.zeros(self.pcn.measurement.t.shape,dtype=np.float64)
        luCount = 0
        luAggregateNow = (luCount,luM,luM2)

        vHalfRealM = np.zeros(self.fourier.fourier_basis_number,dtype=np.float64)
        vHalfRealM2 = np.zeros(self.fourier.fourier_basis_number,dtype=np.float64)
        vHalfRealCount = 0
        vHalfRealAggregateNow = (vHalfRealCount,vHalfRealM,vHalfRealM2)

        vHalfImagM = np.zeros(self.fourier.fourier_basis_number,dtype=np.float64)
        vHalfImagM2 = np.zeros(self.fourier.fourier_basis_number,dtype=np.float64)
        vHalfImagCount = 0
        vHalfImagAggregateNow = (vHalfImagCount,vHalfImagM,vHalfImagM2)

        sigmas = util.sigmasLancos(self.fourier.fourier_basis_number)

        vtHalf = self.fourier.fourierTransformHalf(self.pcn.measurement.vt)
        vtF = self.fourier.inverseFourierLimited(vtHalf*sigmas)
        for i in range(startIndex,self.n_samples):
            utNow = self.fourier.inverseFourierLimited(self.Layers[-2].samples_history[i,:]*sigmas)
            # lUNow = 1/np.flip(util.kappaFun(utNow))#<-  ini aneh ni kenapa harus di flip!!!
            lUNow = 1/util.kappaFun(utNow)#<-  ini aneh ni kenapa harus di flip!!!
            vtEsNow = self.fourier.inverseFourierLimited(self.Layers[-1].samples_history[i,:]*sigmas)
            luAggregateNow = util.updateWelford(luAggregateNow,lUNow)
            vtAggregateNow = util.updateWelford(vtAggregateNow,vtEsNow)
            vHalfRealAggregateNow = util.updateWelford(vHalfRealAggregateNow,self.Layers[-1].samples_history[i,:].real)
            vHalfImagAggregateNow = util.updateWelford(vHalfImagAggregateNow,self.Layers[-1].samples_history[i,:].imag)

        # for i in range(len(vHistoryBurned)):
        # utMean, variance, sampleVariance = util.finalizeWelford(utAggregateNow)
        vtMean = vtAggregateNow[1]
        vtVar = vtAggregateNow[2]/vtAggregateNow[0]

        lMean = luAggregateNow[1]
        lVar = luAggregateNow[2]/luAggregateNow[0]

        vHalfMean = vHalfRealAggregateNow[1]+1j*vHalfImagAggregateNow[1]
        vHalfVarReal = vHalfRealAggregateNow[2]/vHalfRealAggregateNow[0]
        vHalfVarImag = vHalfImagAggregateNow[2]/vHalfImagAggregateNow[0]



        # vtMean = vtEs.mean(axis=1)
        # vtVar = vtEs.var(axis=1)
        
        # lMean = lU.mean(axis=1)
        # lVar = lU.var(axis=1)

        # cummU = np.cumsum(self.u_history[startIndex:,:])
        # indexCumm = np.arange(1,len(cummU)+1)
        # cummMeanU = cummU.T/indexCumm
        # cummMeanU = cummMeanU.T

        sim_result = simRes.SimulationResult(vtHalf,vtF,vHalfMean,np.sqrt(vHalfVarReal),np.sqrt(vHalfVarImag),lMean,np.sqrt(lVar),vtMean,np.sqrt(vtVar))
        self.sim_result = sim_result