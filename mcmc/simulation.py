import numpy as np
import numba as nb
import importlib
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
                    seed,burn_percentage,pcn_pair_layers,enable_beta_feedback):
        self.n_samples = n_samples
        self.meas_samples_num = num
        self.evaluation_interval = evaluation_interval
        self.burn_percentage = burn_percentage
        #set random seed
        self.random_seed = seed
        self.printProgress = printProgress
        self.n_layers=n_layers
        self.kappa = kappa
        self.sigma_0 = sigma_0
        self.sigma_v = sigma_v
        self.sigma_scaling = sigma_scaling
        self.enable_beta_feedback = enable_beta_feedback
        np.random.seed(self.random_seed)
        
        
        #setup parameters for 1 Dimensional simulation
        self.d = 1
        self.nu = 2 - self.d/2
        self.alpha = self.nu + self.d/2
        self.t_start = 0.0
        self.t_end = 1.0
        self.beta_0 = (sigma_0**2)*(2**self.d * np.pi**(self.d/2))* 1.1283791670955126#<-- this numerical value is scp.special.gamma(alpha))/scp.special.gamma(nu)
        self.beta_v = self.beta_0*(sigma_v/sigma_0)**2
        self.sqrtBeta_v = np.sqrt(self.beta_v)
        self.sqrtBeta_0 = np.sqrt(self.beta_0)
        
        f =  fourier.FourierAnalysis(n,num,self.t_start,self.t_end)
        self.fourier = f
        
        rg = randomGenerator.RandomGenerator(f.basis_number)
        self.random_gen = rg
        

        LuReal = (1/self.sqrtBeta_0)*(self.fourier.Dmatrix*self.kappa**(-self.nu) - self.kappa**(2-self.nu)*self.fourier.Imatrix)
        Lu = LuReal + 1j*np.zeros(LuReal.shape)
        
        uStdev = -1/np.diag(Lu)
        uStdev = uStdev[self.fourier.basis_number-1:]
        uStdev[0] /= 2 #scaled

        meas_std = 0.1
        measurement = meas.Measurement(num,meas_std,self.t_start,self.t_end)
        # pcn = pCN.pCN(n_layers,rg,measurement,f,beta)
        self.pcn = pCN.pCN(n_layers,rg,measurement,f,beta)
        self.pcn_pair_layers = pcn_pair_layers

        
        #initialize Layers
        typed_list_status = importlib.util.find_spec('numba.typed.typedlist')
        if typed_list_status is None:
            Layers = []
        else:
            from numba.typed.typedlist import List
            Layers = List()
        # Layers = []
        # factor = 1e-8
        for i in range(self.n_layers):
            if i==0:
                init_sample = np.linalg.solve(Lu,self.random_gen.construct_w())[self.fourier.basis_number-1:]
                lay = layer.Layer(True,self.sqrtBeta_0,i,self.n_samples,self.pcn,init_sample)
                lay.stdev = uStdev
                lay.current_sample_scaled_norm = util.norm2(lay.current_sample/lay.stdev)#ToDO: Modify this
                lay.new_sample_scaled_norm = lay.current_sample_scaled_norm
            else:
                
                if i == n_layers-1:
                    
                    lay = layer.Layer(False,self.sqrtBeta_v,i,self.n_samples,self.pcn,Layers[i-1].current_sample)
                    wNew = self.pcn.random_gen.construct_w()
                    eNew = np.random.randn(self.pcn.measurement.num_sample)
                    wBar = np.concatenate((eNew,wNew))
                    
                    LBar = np.vstack((self.pcn.H,lay.LMat.current_L))

                    #update v
                    lay.current_sample_symmetrized, res, rnk, s = np.linalg.lstsq(LBar,self.pcn.yBar-wBar,rcond=-1)#,rcond=None)
                    lay.current_sample = lay.current_sample_symmetrized[self.pcn.fourier.basis_number-1:]
                else:
                    lay = layer.Layer(False,self.sqrtBeta_v*np.sqrt(sigma_scaling),i,self.n_samples,self.pcn,Layers[i-1].current_sample)
            lay.update_current_sample()

            if self.pcn_pair_layers:
                self.pcn.record_skip = np.max([1,(lay.n_samples*self.n_layers-1)//self.pcn.max_record_history])
                history_length = np.min([lay.n_samples*(self.n_layers-1),self.pcn.max_record_history]) 
            else:
                self.pcn.record_skip = np.max([1,lay.n_samples//self.pcn.max_record_history])
                history_length = np.min([lay.n_samples,self.pcn.max_record_history]) 
            lay.samples_history = np.empty((history_length, self.fourier.basis_number), dtype=np.complex128)
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
            # for j in range(self.fourier.basis_number):
                # accepted_count_partial += self.pcn.one_step_one_element(self.Layers,j)
            if (i+1)%(self.evaluation_interval) == 0:
                self.accepted_count += accepted_count_partial

                if self.pcn_pair_layers:
                    self.acceptancePercentage = self.accepted_count/((i+1)*(self.n_layers-1))                    
                else:
                    self.acceptancePercentage = self.accepted_count/(i+1)
                    
                if self.enable_beta_feedback:
                    self.pcn.adapt_beta(self.acceptancePercentage)
                else:
                    if self.acceptancePercentage> 0.5:
                        self.pcn.more_aggresive()
                    elif self.acceptancePercentage<0.3:
                        self.pcn.less_aggresive()
                #TODO: toggle this if pcn.one_step_one_element is not used
                # acceptancePercentage = self.accepted_count/((i+1)*self.fourier.basis_number)
                
                
                
                
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
                        util.printProgressBar(i+1, self.n_samples, prefix = 'Time Remaining {0}- Acceptance Rate {1:.2%} - Progress:'.format(remainingTimeStr,self.acceptancePercentage), suffix = 'Complete', length = 50)

        with nb.objmode():
            
            elapsedTimeStr = time.strftime("%j day(s),%H:%M:%S", time.gmtime(time.time()-start_time))
            self.total_time = time.time()-start_time
            # print('Complete')
            if self.printProgress:
                util.printProgressBar(self.n_samples, self.n_samples, 'Iteration Completed in {0}- Acceptance Rate {1:.2%} - Progress:'.format(elapsedTimeStr,self.acceptancePercentage), suffix = 'Complete', length = 50)
    

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

        

        uHalfRealM = np.zeros((self.n_layers,self.fourier.basis_number),dtype=np.float64)
        uHalfRealM2 = np.zeros((self.n_layers,self.fourier.basis_number),dtype=np.float64)
        uHalfRealCount = np.zeros(self.n_layers)
        uHalfRealAggregateNow=[] 
        for i in range(self.n_layers):
            uHalfRealAggregate_Layer_i_Now = (uHalfRealCount[i],uHalfRealM[i,:],uHalfRealM2[i,:])
            uHalfRealAggregateNow.append(uHalfRealAggregate_Layer_i_Now) 

        uHalfImagM = np.zeros((self.n_layers,self.fourier.basis_number),dtype=np.float64)
        uHalfImagM2 = np.zeros((self.n_layers,self.fourier.basis_number),dtype=np.float64)
        uHalfImagCount = np.zeros(self.n_layers)
        uHalfImagAggregateNow=[] 
        for i in range(self.n_layers):
            uHalfImagAggregate_Layer_i_Now = (uHalfImagCount[i],uHalfImagM[i,:],uHalfImagM2[i,:])
            uHalfImagAggregateNow.append(uHalfImagAggregate_Layer_i_Now) 

        sigmas = util.sigmasLancos(self.fourier.basis_number)

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
        uHalfMean = np.zeros((self.n_layers,self.fourier.basis_number),dtype=np.complex128)
        uHalfRealVar = np.zeros((self.n_layers,self.fourier.basis_number),dtype=np.float64)
        uHalfImagVar = np.zeros((self.n_layers,self.fourier.basis_number),dtype=np.float64)
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


    def save(self,file_name,include_history=False):
        with h5py.File(file_name,'w') as f:
            for key,value in self.__dict__.items():
                NumbaType = 'numba.' in str(type(value))
                ListType = isinstance(value,list)
                if not NumbaType and not ListType:
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
                        continue
                    if key == 'pcn':
                        f.create_dataset('beta',data=value.beta)
                        continue
                    if include_history and key == 'Layers':
                        for i in range(self.n_layers):
                            f.create_dataset('Layer - {0} samples'.format(i),data=value[i].samples_history)
                    
            
                    

                


