import numpy as np
import numba as nb
import mcmc.util as util
import mcmc.util_2D as u2

spec = [
    ('basis_number', nb.int64),               
    ('extended_basis_number', nb.int64),               
    ('t_end', nb.float64),               
    ('t_start', nb.float64),               
    ('dt', nb.float64),
    ('t', nb.float64[::1]),
    ('index', nb.int64[:,::1]),
    ('Dmatrix', nb.float64[:,::1]),
    ('Imatrix', nb.float64[:,::1]),
    ('cosFun', nb.float64[:,::1]),          
    ('sinFun', nb.float64[:,::1]),          
    ('eigenFun', nb.complex128[:,::1]),
    ('prepared',nb.boolean)
]

@nb.jitclass(spec)
class FourierAnalysis:
    def __init__(self, basis_number,extended_basis_number,t_start = 0,t_end=1):
        self.basis_number = basis_number
        self.extended_basis_number = extended_basis_number
        self.t_end = t_end
        self.t_start = t_start
        
        self.Dmatrix = -(2*np.pi)**2*np.diag(np.arange(-(self.basis_number-1),self.basis_number)**2)
        self.Imatrix = np.eye(2*self.basis_number-1)
        self.prepare()

    
    def prepare(self):
        self.t = np.linspace(self.t_start,self.t_end,self.extended_basis_number)
        self.dt = self.t[1] - self.t[0]
        self.eigenFun = np.empty((self.basis_number,self.extended_basis_number),dtype=np.complex128)
        self.cosFun = np.empty((self.basis_number,self.extended_basis_number),dtype=np.float64)
        self.sinFun = np.empty((self.basis_number,self.extended_basis_number),dtype=np.float64)
        for i in range(self.basis_number):
            self.eigenFun[i,:] = util.eigenFunction1D(-i,self.t) 
            self.cosFun[i,:] = np.cos(2*np.pi*self.t*i)
            self.sinFun[i,:] = np.sin(2*np.pi*self.t*i)
        self.index = self.createUindex()
        self.prepared = True

    
    def inverseFourierLimited(self,uHalf):
        y = np.zeros(self.sinFun.shape[1],dtype=np.float64)
        for i in range(1,len(uHalf)):
            for j in range(self.sinFun.shape[1]):
                y[j] += 2*(uHalf[i].real*self.cosFun[i,j] - uHalf[i].imag*self.sinFun[i,j])
            # y += 2*(u[i].real*cosFun[i,:] - u[i].imag*sinFun[i,:])
        
        for j in range(self.sinFun.shape[1]):
            y[j] += uHalf[0].real
        # y +=   u[0].real
        return y

    def fourierTransformHalf(self,ut):
        uHalf = np.zeros(self.eigenFun.shape[0],np.complex128)
        for i in range(self.eigenFun.shape[0]):
            # uHalf[i] = inner(ut,eigenFun[i,:])*dt
            uHalf[i] = util.inner(ut,self.eigenFun[i,:])
        return uHalf*self.dt

    def constructU(self,uHalf):
        """
        Construct Toeplitz Matrix
        """
        #using native scipy toeplitz function is not working in nopython mode
        # uFull = np.concatenate((uHalf,np.zeros(len(uHalf)-1)), axis=None)
        # U = sla.toeplitz(uFull).conj()

        
        # LU = len(uHalf)
        #np.zeros only supported with two argument, but somehow if I include dtype in np.zeros it
        #is not working, so this is the solution
        Ushape = (2*self.basis_number-1,2*self.basis_number-1)
        U = np.zeros(Ushape,dtype=np.complex128)
        # U = U.astype(complex)
        for i in nb.prange(2*self.basis_number-1):
            for j in nb.prange(2*self.basis_number-1):
                index = i-j #(j-i)#
                if 0<= index <self.basis_number :
                    U[i,j] = uHalf[index]
                    continue
                if 0< -index < self.basis_number:
                    U[i,j] = uHalf[-index].conjugate()
                    # continue
        return U
    
    def constructMatexplicit(self,uHalf,fun):
        temp = fun(self.inverseFourierLimited(uHalf))
        temp2 = self.fourierTransformHalf(temp)
        return self.constructU(temp2)

    def createUindex(self):
        shape = (2*self.basis_number-1,2*self.basis_number-1)
        index = np.zeros(shape,dtype=np.int64)
        for i in nb.prange(2*self.basis_number-1):
            for j in nb.prange(2*self.basis_number-1):
                index[i,j] = (i-j)+(2*self.basis_number-1)
        return index

    def constructU_with_Index(self,uHalf):
        uprepared = util.extend(util.symmetrize(uHalf),2*self.basis_number)
        # with nb.objmode(U='complex128[:,:]'):
            # res = u.extend2D(symmetrize_2D(uHalf),2*n-1)[index]
        U = uprepared[self.index]
        return U



# spec2D = [
#     ('basis_number', nb.int64),               
#     ('extended_basis_number', nb.int64),
#     ('basis_number_2D', nb.int64),
#     ('basis_number_2D_sym', nb.int64),               
#     ('extended_basis_number_2D', nb.int64),               
#     ('extended_basis_number_2D_sym', nb.int64),               
#     ('t_end', nb.float64),               
#     ('t_start', nb.float64),               
#     ('dt', nb.float64),
#     ('t', nb.float64[::1]),
#     ('Dmatrix', nb.float64[:,::1]),
#     ('Imatrix', nb.float64[:,::1]),
#     ('ix', nb.int64[:,::1]),
#     ('iy', nb.int64[:,::1]),
#     ('Index',nb.typeof(u2.createUindex(2)))
# ]

# ORDER = 'C'
# @nb.jitclass(spec2D)
# class FourierAnalysis_2D:
#     def __init__(self,basis_number,extended_basis_number,t_start = 0,t_end=1):
#         self.basis_number = basis_number
#         self.extended_basis_number = extended_basis_number
#         self.basis_number_2D = (2*basis_number-1)*basis_number
#         self.basis_number_2D_sym = (2*basis_number-1)*(2*basis_number-1)
#         self.extended_basis_number_2D = (2*extended_basis_number-1)*extended_basis_number
#         self.extended_basis_number_2D_sym = (2*extended_basis_number-1)*(2*extended_basis_number-1)
#         self.t_end = t_end
#         self.t_start = t_start
#         self.ix = np.zeros(2*self.basis_number-1,2*self.basis_number-1,dtype=np.int64)
#         self.iy = np.zeros(2*self.basis_number-1,2*self.basis_number-1,dtype=np.int64)
#         temp = np.arange(-(self.basis_number-1),self.basis_number)
#         for i in range(2*self.basis_number-1)
#             self.ix[i,:] = temp
#             self.iy[:,i] = temp

#         # d_diag = np.zeros((2*self.basis_number-1)**2)
#         # for i in range(2*self.basis_number-1):
#         #     for j in range(2*self.basis_number-1):
#         #         d_diag[i*10+j] = (i**2+j**2)
#         # self.Dmatrix = -(2*np.pi)**2*np.diag(d_diag)
#         self.Imatrix = np.eye((2*self.basis_number-1)**2)
#         Index = u2.createUindex(self.basis_number)
#         self.Index = Index
    
#     def inverseFourierLimited(self,uHalf):
#         return u2.irfft2(uHalf,self.extended_basis_number)

#     def fourierTransformHalf(self,z):
#         return u2.rfft2(z,self.basis_number)

#     def constructU(self,uHalf):
#         """
#         Construct Toeplitz Matrix
#         """
#         return u2.constructU(uHalf,self.Index)
    
#     def constructMatexplicit(self,uHalf,fun):
#         temp = fun(self.inverseFourierLimited(uHalf))
#         temp2 = self.fourierTransformHalf(temp)
#         return self.constructU(temp2)