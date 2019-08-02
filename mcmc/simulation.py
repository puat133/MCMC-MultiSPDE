import numpy as np
import numba as nb
from numba.typed.typedlist import List
# from numba.typed import List
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
import h5py


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
#     ('n_layers',nb.int64),
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
    def __init__(self,n_layers,n_samples,n,beta,num,kappa,sigma_0,sigma_v,sigma_scaling,evaluation_interval,printProgress,
                    seed,burn_percentage):
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
        beta_0 = (sigma_0**2)*(2**d * np.pi**(d/2))* 1.1283791670955126#<-- this numerical value is scp.special.gamma(alpha))/scp.special.gamma(nu)
        beta_v = beta_0*(sigma_v/sigma_0)**2
        sqrtBeta_v = np.sqrt(beta_v)
        sqrtBeta_0 = np.sqrt(beta_0)
        
        f =  fourier.FourierAnalysis(n,num,t_start,t_end)
        self.fourier = f
        
        rg = randomGenerator.RandomGenerator(f.fourier_basis_number)
        self.random_gen = rg
        

        LuReal = (1/sqrtBeta_0)*(self.fourier.Dmatrix*kappa**(-nu) - kappa**(2-nu)*self.fourier.Imatrix)
        Lu = LuReal + 1j*np.zeros(LuReal.shape)
        
        uStdev = -1/np.diag(Lu)
        uStdev = uStdev[self.fourier.fourier_basis_number-1:]
        uStdev[0] /= 2 #scaled

        meas_std = 0.1
        measurement = meas.Measurement(num,meas_std,t_start,t_end)
        # pcn = pCN.pCN(n_layers,rg,measurement,f,beta)
        self.pcn = pCN.pCN(n_layers,rg,measurement,f,beta)

        
        #initialize Layers
        Layers = List()
        # Layers = []
        # factor = 1e-8
        for i in range(self.n_layers):
            if i==0:
                init_sample = np.linalg.solve(Lu,self.random_gen.construct_w())[self.fourier.fourier_basis_number-1:]
                lay = layer.Layer(True,sqrtBeta_0,i,self.n_samples,self.pcn,init_sample)
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
                    lay = layer.Layer(False,sqrtBeta_v*np.sqrt(sigma_scaling),i,self.n_samples,self.pcn,Layers[i-1].current_sample)
            lay.update_current_sample()
            # TODO: toggle this if pcn.one_step_one_element is not used
            lay.samples_history = np.empty((lay.n_samples*(self.n_layers-1), self.fourier.fourier_basis_number), dtype=np.complex128)

            Layers.append(lay)
                



        self.Layers = Layers
        sim_result = simRes.SimulationResult()     
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
            # for j in range(self.fourier.fourier_basis_number):
                # accepted_count_partial += self.pcn.one_step_one_element(self.Layers,j)
            if (i+1)%(self.evaluation_interval) == 0:
                self.accepted_count += accepted_count_partial
                acceptancePercentage = self.accepted_count/((i+1)*(self.n_layers-1))

                #TODO: toggle this if pcn.one_step_one_element is not used
                # acceptancePercentage = self.accepted_count/((i+1)*self.fourier.fourier_basis_number)
                
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

        utM = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        utM2 = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        utCount = np.zeros(self.n_layers)
        utAggregateNow=[] 
        for i in range(self.n_layers):
            utAggregate_Layer_i_Now = (utCount[i],utM[i,:],utM2[i,:])
            utAggregateNow.append(utAggregate_Layer_i_Now) 

        elltM = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        elltM2 = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        elltCount = np.zeros(self.n_layers)
        elltAggregateNow=[] 
        for i in range(self.n_layers):
            elltAggregate_Layer_i_Now = (elltCount[i],elltM[i,:],elltM2[i,:])
            elltAggregateNow.append(elltAggregate_Layer_i_Now)

        

        uHalfRealM = np.zeros((self.n_layers,self.fourier.fourier_basis_number),dtype=np.float64)
        uHalfRealM2 = np.zeros((self.n_layers,self.fourier.fourier_basis_number),dtype=np.float64)
        uHalfRealCount = np.zeros(self.n_layers)
        uHalfRealAggregateNow=[] 
        for i in range(self.n_layers):
            uHalfRealAggregate_Layer_i_Now = (uHalfRealCount[i],uHalfRealM[i,:],uHalfRealM2[i,:])
            uHalfRealAggregateNow.append(uHalfRealAggregate_Layer_i_Now) 

        uHalfImagM = np.zeros((self.n_layers,self.fourier.fourier_basis_number),dtype=np.float64)
        uHalfImagM2 = np.zeros((self.n_layers,self.fourier.fourier_basis_number),dtype=np.float64)
        uHalfImagCount = np.zeros(self.n_layers)
        uHalfImagAggregateNow=[] 
        for i in range(self.n_layers):
            uHalfImagAggregate_Layer_i_Now = (uHalfImagCount[i],uHalfImagM[i,:],uHalfImagM2[i,:])
            uHalfImagAggregateNow.append(uHalfImagAggregate_Layer_i_Now) 

        sigmas = util.sigmasLancos(self.fourier.fourier_basis_number)

        vtHalf = self.fourier.fourierTransformHalf(self.pcn.measurement.vt)
        vtF = self.fourier.inverseFourierLimited(vtHalf*sigmas)
        for i in range(startIndex,self.n_samples):
            for j in range(self.n_layers): 
                utNow = self.fourier.inverseFourierLimited(self.Layers[j].samples_history[i,:]*sigmas)
                # lUNow = 1/np.flip(util.kappaFun(utNow))#<-  ini aneh ni kenapa harus di flip!!!
                elltNow = util.kappaFun(-utNow)#<-  ini aneh ni kenapa harus di flip!!!
                
                utAggregateNow[j] = util.updateWelford(utAggregateNow[j],utNow)
                elltAggregateNow[j] = util.updateWelford(elltAggregateNow[j],elltNow)
                uHalfRealAggregateNow[j] = util.updateWelford(uHalfRealAggregateNow[j],self.Layers[j].samples_history[i,:].real)
                uHalfImagAggregateNow[j] = util.updateWelford(uHalfImagAggregateNow[j],self.Layers[j].samples_history[i,:].imag)

        # for i in range(len(vHistoryBurned)):
        # utMean, variance, sampleVariance = util.finalizeWelford(utAggregateNow)
        utMean = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        utVar = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        elltMean = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        elltVar = np.zeros((self.n_layers,self.pcn.measurement.t.shape[0]),dtype=np.float64)
        uHalfMean = np.zeros((self.n_layers,self.fourier.fourier_basis_number),dtype=np.complex128)
        uHalfRealVar = np.zeros((self.n_layers,self.fourier.fourier_basis_number),dtype=np.float64)
        uHalfImagVar = np.zeros((self.n_layers,self.fourier.fourier_basis_number),dtype=np.float64)
        for j in range(self.n_layers):
            utMean[j,:] = utAggregateNow[j][1]
            utVar[j,:] = utAggregateNow[j][2]/utAggregateNow[j][0]

            elltMean[j,:] = elltAggregateNow[j][1]
            elltVar[j,:] = elltAggregateNow[j][2]/elltAggregateNow[j][0]

            uHalfMean[j,:] = uHalfRealAggregateNow[j][1]+1j*uHalfImagAggregateNow[j][1]
            uHalfRealVar[j,:] = uHalfRealAggregateNow[j][2]/uHalfRealAggregateNow[j][0]
            uHalfImagVar[j,:] = uHalfImagAggregateNow[j][2]/uHalfImagAggregateNow[j][0]



        # sim_result = simRes.SimulationResult()
        # sim_result.assign_values(vtHalf,vtF,uHalfMean,np.sqrt(uHalfRealVar),np.sqrt(uHalfImagVar),elltMean,np.sqrt(elltVar),utMean,np.sqrt(utVar))
        self.sim_result.assign_values(vtHalf,vtF,uHalfMean,np.sqrt(uHalfRealVar),np.sqrt(uHalfImagVar),elltMean,np.sqrt(elltVar),utMean,np.sqrt(utVar))


    def save(self,file_name):
        with h5py.File(file_name,'w') as f:
            for key,value in self.__dict__.items():
                NumbaType = 'numba.' in str(type(value))
                if not NumbaType:
                    f.create_dataset(key,data=value)
                else:
                    if key == 'sim_result':
                        f.create_dataset('vtHalf',data=value.vtHalf)
                        f.create_dataset('vtF',data=value.vtF)
                        f.create_dataset('uHalfMean',data=value.uHalfMean)
                        f.create_dataset('uHalfStdReal',data=value.uHalfStdReal)
                        f.create_dataset('uHalfStdImag',data=value.uHalfStdImag)
                        f.create_dataset('elltMean',data=value.elltMean)
                        f.create_dataset('elltStd',data=value.elltStd)
                        f.create_dataset('utMean',data=value.utMean)
                        f.create_dataset('utStd',data=value.utStd)
                    if key == 'pcn':
                        f.create_dataset('beta',data=value.beta)
                    
            
                    

                


