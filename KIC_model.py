import numpy as np
import time
import os
import pickle
import scipy.linalg as la
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import json
from numpy import sin, cos, pi
from scipy.signal import lsim
import matplotlib.animation as anim

def warn(*args, **kwargs):            # Suppress warnings from neural network
    pass
import warnings
warnings.warn = warn

class KIC_model(object):
    
    def __init__(self,
                 K=30, corr = -1e-3,NN_max = 100,NN_width = 500,n_batch=20,n_iter=500,n_eig_grad=10,n_B_grad=10,alpha=.01,\
                fac_ini=1e3,denom=20,gamma=.99,eps=1e-10,sum_fac=1e7,use_DMDc = True):
        
        self.trained = False
        self.use_DMDc = use_DMDc
        self.K = K                              # Number of (complex) dimensions for K-space; A has size 2*K
        self.lay = (NN_width,)                  # hidden_layer_sizes for NN (might want to include extra scale factor)
        self.corr = corr                        # corrected eigenvalue
        self.reals = -.1*self.sqz(np.ones([1,K]))    # real parts for matrix A
        self.imags = self.sqz(np.zeros([1,K]))            # imag parts for matrix A
        self.A = self.make_A(self.reals,self.imags)
        self.NN_max = NN_max                    # max iter for NN grad descent, default 1000, been using 100
        self.n_batch = n_batch                  # Number per batch for mini-batch grad desc
        self.n_iter = n_iter                    # Number of times for overall descent (per run of for loop)
        self.n_eig_grad = n_eig_grad            # Number of steps for "inner" grad descent for eigs 
        self.n_B_grad = n_B_grad                # Number of steps for "inner" grad descent for B
        self.alpha = alpha                      # adj factor for descent
        self.fac_ini = fac_ini                  # initial descent factor; default has been 1e3
        self.denom = denom                      # denominator for annealing descent
        self.gamma = gamma                      # decay for moving average of gradient
        self.eps = eps                          # denom for moving average of gradient
        self.sum_fac = sum_fac
        self.reg = MLPRegressor(hidden_layer_sizes = self.lay, warm_start = True,max_iter = NN_max)
        self.k_0 = np.array([0,1]*K)
        self.C = np.eye(2*K)

    
    @staticmethod
    def relu(x):
        return np.maximum(x,0)

    @staticmethod
    def relu_cpx(x):
        rp = np.real(x)
        im = np.imag(x)
        return self.relu(rp) + 1j*self.relu(im)

    @staticmethod
    def col(a): #np 1D array to column-matrix 
        return np.reshape(a,(-1,1))

    @staticmethod
    def sqz(a): #make a into a 1D array
        return np.squeeze(np.asarray(a))

    @staticmethod                              #change make_ methods and below into instance methods to use self.dt
    def make_Ad(A,dt):
        return np.asmatrix(la.expm(dt*A))

    @classmethod
    def make_Bd(cls,A,B,dt):
        I = np.eye(A.shape[0])
        Ad = cls.make_Ad(A,dt)
        return np.asmatrix(np.dot(la.lstsq(A,Ad-I)[0],B))
    
    @classmethod
    def make_A(cls,reals,imags): # update matrix corresponding to eigenvalues
        A = np.kron(np.diag(cls.sqz(reals)),np.eye(2)) + np.kron(np.diag(cls.sqz(imags)),np.matrix('[0,-1;1,0]'))
        return np.asmatrix(A)
    
    @classmethod
    def err_mat(cls,A,B,reg,u,k,x):
        if not np.iscomplexobj(A):
            return np.asarray(reg.predict((A*k + B*u).T).T-x)
        else:
            A0,A1 = reg.coefs_                  # Extract NN matrix parameters 
            A0 = np.asmatrix(A0).T
            A1 = np.asmatrix(A1).T
            b0 = cls.col(reg.intercepts_[0])
            b1 = cls.col(reg.intercepts_[1])
            pred = A*k + B*u
            return np.asarray(A1*cls.relu(A0*pred + b0) + b1 - x)    

    
    def norm_der(self,u,k,x,H,dt,h=1e-8):
        '''Use Higham's complex step approximation
           Citation:
           Al-Mohy, A.H., Higham, N.J. 
           The complex step approximation to the Fréchet derivative of a matrix function. 
           Numer Algor 53, 133 (2010). https://doi.org/10.1007/s11075-009-9323-y
           '''
        A_d = self.make_Ad(self.A + 1j*h*H,dt)
        B_d = self.make_Bd(self.A + 1j*h*H,self.B,dt)
        s = np.sum(self.err_mat(A_d,B_d,self.reg,u,k,x)**2)
        return np.imag(s)/h

    def grad_eigs(self,u,k,x,dt):
        # eigenvalues
        K = self.K
        d_real = []
        d_imag = []
        for i in range(K):
            H = np.zeros((2*K,2*K))
            inds = [2*i,2*i+1]
            H[inds,inds] = 1/2**.5
            d_real.append(self.norm_der(u,k,x,H,dt))
            H = np.zeros((2*K,2*K))
            H[i,i+1] = -1/2**.5
            H[i+1,i] = 1/2**.5
            d_imag.append(self.norm_der(u,k,x,H,dt))
        return np.asarray(d_real),np.asarray(d_imag)

    def grad_real(self,u,k,x,dt):
        # real eigenvalues
        K = self.K
        d_real = []
        for i in range(K):
            H = np.zeros((2*K,2*K))
            inds = [2*i,2*i+1]
            H[inds,inds] = 1/2**.5
            d_real.append(self.norm_der(u,k,x,H,dt))
        return np.asarray(d_real)

    def grad_imag(self,u,k,x,dt):
        # imaginary eigenvalues
        K = self.K
        d_imag = []
        for i in range(K):
            H = np.zeros((2*K,2*K))
            H[i,i+1] = -1/2**.5
            H[i+1,i] = 1/2**.5
            d_imag.append(self.norm_der(u,k,x,H,dt))
        return np.asarray(d_imag)



    def grad_B(self,u,k,x,dt):
        '''Computes gradient for updating matrix B
            A,B: 
            u: input data (stoch sample)
            k: Corresp k-state data
            x: Corresp state data (next time step)
            A0,b0,A1: NN matrices + inner bias
            err: f - x_t term'''
        err = self.err_mat(self.A_d,self.B_d,self.reg,u,k,x)
        I = np.eye(2*self.K)
        A0,A1 = self.reg.coefs_                  # Extract NN matrix parameters 
        A0 = np.asmatrix(A0).T
        A1 = np.asmatrix(A1).T
        b0 = self.reg.intercepts_[0]
        S = np.zeros(self.B.shape).T
        M = np.asmatrix(la.lstsq(self.A,self.A_d-I)[0])
        for t in range(u.shape[1]):
            p_array = np.argwhere(A0*self.col(k[:,t]) + self.col(b0) >= 0)[:,0]
            S += 2*self.col(u[:,t])*(self.col(err[:,t]).T*A1[:,p_array]*A0[p_array,:])*M
        if la.norm(S) > 1:
            S = S/la.norm(S) #normalize gradient
        return S.T
    
    def norm_out(self,x):
        return self.m_norm*x + self.b
    
    def unnorm_out(self,x):
        return self.m_unnorm*(x-b)
    
    def normalize(self,data):
        self.x_0 = data[:,0]
        mins_out = np.min(data, axis=1)
        maxs_out = np.max(data, axis=1)
        self.idx_mvmt = self.sqz(np.argwhere(mins_out < maxs_out))    # indices where measurements are not constant
        data_red = data[self.idx_mvmt,:]

        mins_out = mins_out[self.idx_mvmt]
        maxs_out = maxs_out[self.idx_mvmt]
        slope_out = 1/(maxs_out - mins_out)
        int_out = -mins_out/(maxs_out - mins_out)
        self.m_norm = self.col(slope_out)
        self.m_unnorm = self.col(1/slope_out)
        self.b = self.col(int_out)
        
        return self.norm_out(data_red)
    
    def DMDc_fit(self,in_trn,out_trn,dt):
        n = self.n
        k = self.K//n
        N_trn = out_trn.shape[1]
        M = np.zeros((0,N_trn-k))
        N = np.zeros((0,N_trn-k))
        for i in range(k):
            M = np.vstack((M,out_trn[:,i:N_trn+i-k]))
            N = np.vstack((N,out_trn[:,1+i:N_trn+i-k+1]))
        M = np.vstack((M,in_trn[:,:N_trn-k]))
        Amat = np.dot(N,la.pinv(M))[:k*n,:k*n]
        dmd_eigs = np.log(la.eig(Amat)[0])/dt
        ini_eigs = self.sqz(dmd_eigs[np.argwhere(np.imag(dmd_eigs)>=0)])
        self.reals[:len(ini_eigs)] = np.real(ini_eigs)
        self.imags[:len(ini_eigs)] = np.imag(ini_eigs)
        self.A = self.make_A(self.reals,self.imags)
        
    def k_gen(self,u,t):
        return lsim((self.A,self.B,self.C,self.D), u.T, t, X0 = self.k_0, interp = False)[2].T
    
    def predict(self,u,t):
        return self.reg.predict(self.k_gen(u,t).T).T
    
    def descent_reals(self,grad_fac,grad_err,u,k,x,dt):
        max_eig = 0
        fix = False
        for j in range(self.n_eig_grad):
            d_real = self.grad_real(u,k,x,dt) # eigs gradient
#                 self.grad_reals_avg = self.gamma*self.grad_reals_avg + self.sum_fac*(1-self.gamma)*d_real**2
#                 re_try = self.reals - (grad_fac/self.n_batch)*d_real/np.sqrt(self.eps + self.grad_reals_avg)
            re_try = self.reals - (grad_fac/self.n_batch)*d_real
            A_try = self.make_A(re_try,self.imags)
            err_try = la.norm(self.err_mat(self.make_Ad(A_try,dt),self.make_Bd(A_try,self.B,dt),self.reg,u,k,x))
            if err_try <= grad_err:
                self.reals = re_try
                grad_err = err_try
                max_eig = max(max_eig,np.max(self.reals))
                for j in range(self.K):
                    if self.reals[j] > 0:
                        fix = True
                        self.reals[j] = self.corr
            else:
                grad_fac *= self.alpha
        return grad_err,fix,max_eig
    
    
    def descent_imags(self,grad_fac,grad_err,u,k,x,dt):
        for j in range(self.n_eig_grad):
            d_imag = self.grad_imag(u,k,x,dt) # eigs gradient
#                 self.grad_imags_avg = self.gamma*self.grad_imags_avg + self.sum_fac*(1-self.gamma)*d_imag**2
#                 im_try = self.imags - (grad_fac/self.n_batch)*d_imag/np.sqrt(self.eps + self.grad_imags_avg)
            im_try = self.imags - (grad_fac/self.n_batch)*d_imag
            A_try = self.make_A(self.reals,im_try)
            err_try = la.norm(self.err_mat(self.make_Ad(A_try,dt),self.make_Bd(A_try,self.B,dt),self.reg,u,k,x))
            if err_try <= grad_err:
                self.imags = im_try
                grad_err = err_try
            else:
                grad_fac *= self.alpha
        return grad_err
    
    def descent_B(self,grad_fac,grad_err,u,k,x,dt):
        for j in range(self.n_B_grad):
            d_B = self.grad_B(u,k,x,dt)
            B_try = self.B - (grad_fac/self.n_batch)*d_B
#                 self.grad_B_avg = self.gamma*self.grad_B_avg + self.sum_fac*(1-self.gamma)*d_B**2
#                 B_try = self.B - (grad_fac/self.n_batch)*d_B/np.sqrt(self.eps + self.grad_B_avg)
            err_try = la.norm(self.err_mat(self.A_d,self.make_Bd(self.A,B_try,dt),self.reg,u,k,x))
            if err_try <= grad_err:
                self.B = B_try  
                grad_err = err_try
            else:
                grad_fac *= self.alpha
        return grad_err
    
    
    
    
    def fit(self,in_trn,out_train,dt,target=None,label=None): #target data to check true error 
        '''
        Target is a tuple with the following:
        - input data
        - output data
        - values of t
        '''
        K = self.K
        N_trn = in_trn.shape[1]
        t = np.arange(N_trn)*dt
        if in_trn.shape[1]!=out_train.shape[1]:
            raise Exception("Input and output must have same number of columns")        
        if not self.trained:
            self.trained = True
            self.m = in_trn.shape[0]
            m = self.m
            self.dt = dt
            self.n_true = out_train.shape[0]
            self.B = np.random.rand(2*K,m)          # matrix B
            self.D = np.zeros([2*K,m])
            if label == None:
                self.label = ['x'+str(i) for i in range(self.n_true)]
            else:
                if len(label)!=self.n_true:
                    raise Exception("Incorrect number of labels")
                self.label = label
            out_trn = self.normalize(out_train)
            self.n = out_trn.shape[0]
#             selg.grad_B_avg = np.zeros(self.B.shape)
#             self.grad_reals_avg = np.zeros(self.reals.shape)
#             self.grad_imags_avg = np.zeros(self.reals.shape)
            self.real_maxes = []
            self.real_mins = []
            self.imag_maxes = [] 
            self.imag_mins = []
            self.errs = [[],[],[],[]]
            self.errs_bef = [1e5]
            self.dists = [[],[],[]]
            self.err_true = []
            if self.use_DMDc and self.n<=self.K:
                self.DMDc_fit(in_trn,out_trn,dt)
        else:
            if in_trn.shape[0]!=self.m:
                raise Exception("Input dimension does not match model")
            if out_trn.shape[0]!=self.n_true:
                raise Exception("Output dimension does not match model")
            if dt!=self.dt:
                raise Exception("Time step does not match model")
            out_trn = self.norm_out(out_train[self.idx_mvmt,:])
        reals_old = self.reals
        imags_old = self.imags
        B_old = self.B
        fac_cur = self.fac_ini
        
        if target != None:
            in_comp = target[0]
            out_comp = self.norm_out(target[1][self.idx_mvmt,:])
            t_comp = target[2]
        
        print('Begin Training: %d iterations'%self.n_iter)
        start_time = time.time()
        for it in range(self.n_iter):
            if it%10 == 0:
                print('-------------------- \n iteration: %d'%(len(self.errs[0])))
                print(' ' + time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start_time)))
            idx = np.random.choice(N_trn-1,self.n_batch,replace = False)                   # totally random batch
            t_eval = t[:np.max(idx)+1]
            u_eval = in_trn[:,:np.max(idx)+1]
            k_trn =  self.k_gen(u_eval,t_eval)  # generate k-space trajectory
            k_batch = k_trn[:,idx]              # batch of k-states
            x_fit = out_trn[:,idx]              # batch of states; NN-training
            x_batch = out_trn[:,idx+1]          # batch of states; grad desc
            u_batch = in_trn[:,idx]             # batch of inputs
            if it > 0:
                self.errs_bef.append(la.norm(self.err_mat(self.A_d,self.B_d,self.reg,u_batch,k_batch,x_batch)))
            self.reg.fit(k_batch.T,x_fit.T)          # Train the NN
            self.A_d = self.make_Ad(self.A,dt)
            self.B_d = self.make_Bd(self.A,self.B,dt)

            grad_err = la.norm(self.err_mat(self.A_d,self.B_d,self.reg,u_batch,k_batch,x_batch))
            self.errs[0].append(grad_err)
            grad_fac = fac_cur
            grad_err = self.descent_B(fac_cur,grad_err,u_batch,k_batch,x_batch,dt) #gradient descent over B
            self.errs[1].append(grad_err) 
            
            grad_err,fix,max_eig = self.descent_reals(grad_fac,grad_err,u_batch,k_batch,x_batch,dt) #over reals
            self.errs[2].append(grad_err)
            
            grad_err = self.descent_imags(grad_fac,grad_err,u_batch,k_batch,x_batch,dt) #over imags
            self.errs[3].append(grad_err)
            
            self.A = self.make_A(self.reals,self.imags)
            if fix:
                print("fixed eig at iteration %d"%len(self.errs[0]))
                print("max real part: %f"%max_eig)
            self.real_maxes.append(np.max(self.reals))
            self.real_mins.append(np.min(self.reals))
            self.imag_maxes.append(np.max(self.imags))
            self.imag_mins.append(np.min(self.imags))    

            self.dists[0].append(la.norm(self.B - B_old))
            self.dists[1].append(la.norm(self.reals - reals_old))
            self.dists[2].append(la.norm(self.imags - imags_old))

            B_old = self.B
            reals_old = self.reals
            imags_old = self.imags
            
            fac_cur *= (len(self.errs[0])+self.denom)/(len(self.errs[0])+self.denom+1)  #"annealing" descent
            
            if target != None:
                if it%10==0:
                    pred_mat = self.predict(in_comp,t_comp)
                    self.err_true.append(la.norm(pred_mat - out_comp))
        print('====================\n    Final step: fit NN to full data')
        self.fit_NN(in_trn,out_trn,t,target = (in_comp,out_comp,t_comp))
        print("~~~~~~~~~~~~~~~~~~~~\n  TRAINING COMPLETE\n~~~~~~~~~~~~~~~~~~~~")
    
    
    def fit_NN(self,in_trn,out_trn,t,target = None):
        if target != None:
            pred_mat = self.predict(target[0],target[2])
            err_before = la.norm(pred_mat - target[1])
        k_trn = self.k_gen(in_trn,t)
        self.reg.max_iter = 1000
        self.reg.fit(k_trn.T,out_trn.T)
        self.reg.max_iter = self.NN_max
        if target != None:
            pred_mat = self.predict(target[0],target[2])
            err_after = la.norm(pred_mat - target[1])
            print('before: %f'%err_before)
            print('after: %f'%err_after)
              
    def plot_changes(self,start = 0):
        titles = ('B','reals','imags')
        for i in range(len(self.dists)):
            fig, ax = plt.subplots()
            ax.plot(np.convolve(self.dists[i][start:], np.ones(100), 'valid') / 100)
            ax.set_title('Incremental change in '+ titles[i] + ' (Moving average)')
    
    def plot_error(self,start = 0):
        plt.plot(self.err_true[start:])
        plt.title('Overall error in fit')
    
    def plot_comparison(self,target,n_start,n_end,normed=True, save=None):
        t_all = target[2]
        t_plot = t_all[n_start:n_end]
        pred_mat = self.predict(target[0],t_all)
        out_mat = self.norm_out(target[1][self.idx_mvmt,:])
        for ind in range(len(self.idx_mvmt)):
            fig, ax = plt.subplots()
            ax.plot(t_plot,out_mat[ind,n_start:n_end])
            ax.plot(t_plot,pred_mat[ind,n_start:n_end])
            ax.set_title('Normalized Trajectory: ' + self.label[self.idx_mvmt[ind]])
            ax.legend(('Data','Prediction'))
            if save != None:
                plt.savefig(save+'Cont-Comparison-'+self.label[self.idx_mvmt[ind]])
    
    def plot_eigs(self):
        plt.plot(self.col(self.reals).T,self.col(self.imags).T,marker = 'x')
        plt.grid(b=True)
        
    def set_label(self):
        pass
        
        
        