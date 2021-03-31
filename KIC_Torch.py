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


# NOTE: WE ASSUME ALL DATA IS IN float32 rather than float64 precision
# Requires conversion if using pandas


class Lin(nn.Module):
    def __init__(self, K, m, dt, n_round=10,lr = 1e3, wd=0, n_batch=10, denom=10, damp=0, n_rep=10, alpha=.1, exp=1):
        super(Lin,self).__init__()
        real_init = -.1
        self.lr = lr
        self.alpha = alpha
        self.n_batch = n_batch
        self.n_round = n_round
        self.n_rep = n_rep
        self.K = K
        self.m = m
        self.dt = dt
        self.reals = nn.Parameter(torch.ones(K))   # real parts for matrix A
        self.reals.mul(real_init)
        self.imags = nn.Parameter(torch.zeros(K))  # imag parts for matrix A
        self.B = nn.Parameter(torch.randn((2*K,m)))          # matrix B
        self.C = np.eye(2*K)
        self.D = np.zeros([2*K,m])
        self.k_0 = np.array([0,1]*K)
#         self.optimizer = torch.optim.Adagrad(self.parameters(), lr=lr, lr_decay=.01)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum = 0.1, weight_decay = wd, dampening = damp)
#         self.optimizer = torch.optim.Adadelta(self.parameters(), lr = lr, rho = 0.5)
#         self.sched = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, (lambda n:((n+denom)/(n+denom+1))**exp))
#         self.alpha = alpha
     
        
    def forward(self,u,k):
        self.make_mats()
        return k@self.A_d.T + u@self.B_d.T 
        
    def make_mats(self):
        I = torch.eye(2*self.K)
        self.A = torch.kron(torch.diag(self.reals),torch.eye(2)) + \
            torch.kron(torch.diag(self.imags),torch.tensor([[0,-1.],[1,0]]))
        self.A_d = torch.matrix_exp(self.dt*self.A)
        self.B_d = torch.linalg.inv(self.A)@(self.A_d-I)@self.B
        
    @torch.no_grad()
    def k_gen(self,u,t):
        k = lsim((self.A.numpy(),self.B.numpy(),self.C,self.D), u.T, t, X0 = self.k_0, interp = False)[2]
        return torch.tensor(k) #transposed from usual

                

class Net(nn.Module):
    def __init__(self,K,n,layer_sizes, n_round=10, lr=1e-1*10, n_batch=100, wd=0, denom=10, n_rep=5, alpha=.5):
        super(Net,self).__init__()
        self.alpha = alpha
        self.n_round = n_round
        self.n_rep = n_rep
        self.n_batch = n_batch
        fcs = []
        fcs.append(nn.Linear(2*K,layer_sizes[0]))
        for k in range(len(layer_sizes)-1):
            fcs.append(nn.Linear(layer_sizes[k],layer_sizes[k+1]))
        fcs.append(nn.Linear(layer_sizes[-1],n))
        self.fcs = nn.ModuleList(fcs)
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = wd)
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr=lr, weight_decay = wd, rho = 0.5)
#         self.sched = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, (lambda n:(n+denom)/(n+denom+1)))
#         self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        
  
        
    def forward(self,x):
        for k in range(len(self.fcs)-1):
            x = self.fcs[k](x)
            x = F.relu(x)
        return self.fcs[-1](x)        

        
class KIC_model(object):

    # (15,10,8)
    def __init__(self, K=30, corr=-1e-3, n_batch=20, n_round=100, n_iter=1000, use_DMDc=True, layer_sizes = (2000,)):
        super(KIC_model, self).__init__()
        self.trained = False
        self.use_DMDc = use_DMDc
        self.K = K                              # Number of (complex) dimensions for K-space; A has size 2*K
        self.corr = corr                        # corrected eigenvalue
        self.n_batch = n_batch                  # Number per batch for mini-batch grad desc
        self.n_round = n_round
        self.n_iter = n_iter                    # Number of times for overall descent (per run of for loop)
        self.loss_fn = nn.MSELoss()
        self.layer_sizes = layer_sizes
        self.n_sets = len(layer_sizes)+1
        
    
    @staticmethod
    def col(a): #np 1D array to column-matrix 
        return np.reshape(a,(-1,1))

    @staticmethod
    def sqz(a): #make a into a 1D array
        return np.squeeze(np.asarray(a))
        
    
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

    
    def initialize(self,in_trn,out_train,dt,label):
        self.trained = True
        m = in_trn.shape[0]
        self.m = m        
        self.dt = dt
        self.n_true = out_train.shape[0]
        self.linear = Lin(self.K,m,dt)
        if label == None:
            self.label = ['x'+str(i) for i in range(self.n_true)]
        else:
            if len(label)!=self.n_true:
                raise Exception("Incorrect number of labels")
            self.label = label
        out_trn = self.normalize(out_train)
        self.n = out_trn.shape[0]
        self.net = Net(self.K,self.n,self.layer_sizes)
        
        self.real_maxes = []
        self.real_mins = []
        self.imag_maxes = [] 
        self.imag_mins = []
        self.errs = []
#         self.errs_bef = [1e5]
        self.dists = [[],[],[]]
        self.dists_weight = [[] for i in range(self.n_sets)]
        self.dists_bias = [[] for i in range(self.n_sets)]
        self.errs_true = []
        if self.use_DMDc and self.n<=self.K:
            self.DMDc_fit(in_trn,out_trn,dt)
        return out_trn
        
    
    def DMDc_fit(self,in_trn,out_trn,dt):
        n = self.n
        k = self.linear.K//n
        N_trn = out_trn.shape[1]
        M = np.zeros((0,N_trn-k))
        N = np.zeros((0,N_trn-k))
        for i in range(k):
            M = np.vstack((M,out_trn[:,i:N_trn+i-k]))
            N = np.vstack((N,out_trn[:,1+i:N_trn+i-k+1]))
        M = np.vstack((M,in_trn[:,:N_trn-k]))
        Amat = np.dot(N,la.pinv(M))[:k*n,:k*n]
        dmd_eigs = np.log(la.eig(Amat)[0])/dt
        ini_eigs = dmd_eigs[np.imag(dmd_eigs)>=0]
        with torch.no_grad():
            self.linear.reals[:len(ini_eigs)] = torch.tensor(np.real(ini_eigs))
            self.linear.imags[:len(ini_eigs)] = torch.tensor(np.imag(ini_eigs))
            scale = 2*torch.max(torch.abs(self.linear.imags))
            self.linear.imags[len(ini_eigs):] = scale*(2*torch.rand(self.linear.K-len(ini_eigs))-1)
    
    
    def train(self,in_trn,out_trn,t): 
        k_trn = self.linear.k_gen(in_trn,t)       # generate k-space trajectory
        u_trn = torch.tensor(in_trn.T)
        x_trn = torch.tensor(out_trn.T)
        data = TensorDataset(u_trn[:-1,:],k_trn[:-1,:],x_trn[1:,:])
        loader = DataLoader(data, batch_size=self.net.n_batch)
        load_iter = iter(loader)
        n_round = min(self.net.n_round,t.size//(self.net.n_batch*2))
        rd_errs = []
        for i in range(n_round):
            # Compute prediction error
            u,k,x = next(load_iter)
            
            for j in range(self.net.n_rep):
                pred = self.net(self.linear(u,k))
                loss = self.loss_fn(pred, x)

                # Backpropagation
                self.net.optimizer.zero_grad()
                self.linear.optimizer.zero_grad()
                loss.backward()
                self.net.optimizer.step()
                self.linear.optimizer.step()
                
#             for j in range(self.linear.n_rep):

                with torch.no_grad():
                    self.linear.reals[self.linear.reals>0] = self.corr
            
            rd_errs.append(float(loss))
            
#         self.net.sched.step()
#         self.linear.sched.step()
        self.errs.append(np.average(rd_errs)) #or, can compute true overall error


         
    
    def fit(self,in_trn,out_train,dt,target=None,label=None,NN_final_step=True): #target data to check true error 
        '''
        Target is a tuple with the following:
        - input data
        - output data
        - values of t
        '''
        N_trn = in_trn.shape[1]
        t = np.arange(N_trn)*dt
        if in_trn.shape[1]!=out_train.shape[1]:
            raise Exception("Input and output must have same number of columns")        
        if not self.trained:
            out_trn = self.initialize(in_trn,out_train,dt,label)
        else:
            if in_trn.shape[0]!=self.m:
                raise Exception("Input dimension does not match model")
            if out_train.shape[0]!=self.n_true:
                raise Exception("Output dimension does not match model")
            if dt!=self.dt:
                raise Exception("Time step does not match model")
            out_trn = self.norm_out(out_train[self.idx_mvmt,:])
        reals_old = self.linear.reals.detach().clone()
        imags_old = self.linear.imags.detach().clone()
        B_old = self.linear.B.detach().clone()
        
        weights_old = [self.net.fcs[i].weight.detach().clone() for i in range(self.n_sets)]
        biases_old = [self.net.fcs[i].bias.detach().clone() for i in range(self.n_sets)]

        self.linear.make_mats()
        
        if target != None:
            in_comp = target[0]
            out_comp = self.norm_out(target[1][self.idx_mvmt,:])
            t_comp = target[2]
        
        self.true_check = 10
        print('Begin Training: %d iterations'%self.n_iter)
        start_time = time.time()
        for it in range(self.n_iter):
            if it%10 == 0:
                print('-------------------- \n iteration: %d'%(len(self.errs)))
                print(' ' + time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start_time)))
            self.train(in_trn,out_trn,t)
            with torch.no_grad():
                self.real_maxes.append(float(torch.max(self.linear.reals)))
                self.real_mins.append(float(torch.min(self.linear.reals)))
                self.imag_maxes.append(float(torch.max(self.linear.imags)))
                self.imag_mins.append(float(torch.min(self.linear.imags)))

                self.dists[0].append(float(torch.norm(self.linear.B - B_old)))
                self.dists[1].append(float(torch.norm(self.linear.reals - reals_old)))
                self.dists[2].append(float(torch.norm(self.linear.imags - imags_old)))
                
                for i in range(self.n_sets):
                    self.dists_weight[i].append(float(torch.norm(self.net.fcs[i].weight - weights_old[i])))
                    self.dists_bias[i].append(float(torch.norm(self.net.fcs[i].bias - biases_old[i])))
                
                reals_old = self.linear.reals.clone()
                imags_old = self.linear.imags.clone()
                B_old = self.linear.B.clone()
                
                weights_old = [self.net.fcs[i].weight.clone() for i in range(self.n_sets)]
                biases_old = [self.net.fcs[i].bias.clone() for i in range(self.n_sets)]
            
#             model.optimizer.param_groups[0]['lr'] *= ((len(self.errs[0])+self.denom)/(len(self.errs[0])+self.denom+1))
            
            if target != None:
                if it%self.true_check==0:
                    pred = self.predict(in_comp,t_comp)
                    self.errs_true.append(la.norm(pred-out_comp))                        
                    
#             if it%250 == 0 and it > 0:
            if it%500 == 0 and it > 0:
#                 self.net.optimizer.param_groups[0]['lr'] *= 10
                print('NN fit:')
                if target != None:
                    pred_mat = self.predict(in_comp,t_comp)
                    err_before = la.norm(pred_mat - out_comp)
                for i in range(5):
                    self.fit_NN(in_trn,out_trn,t,target = (in_comp,out_comp,t_comp))
                if target != None:
                    pred_mat = self.predict(in_comp,t_comp)
                    err_after = la.norm(pred_mat - out_comp)
                print('Before... %f'%err_before)
                print('After...%f'%err_after)
                self.linear.optimizer.param_groups[0]['lr'] *= self.linear.alpha
#                 self.net.optimizer.param_groups[0]['lr'] *= self.net.alpha
#                 self.net.optimizer.param_groups[0]['lr'] /= 10
        
        if NN_final_step:
            print('====================\n    Final step: fit NN to full data')
            if target != None:
                pred_mat = self.predict(in_comp,t_comp)
                err_before = la.norm(pred_mat - out_comp)
            for i in range(5):
                self.fit_NN(in_trn,out_trn,t,target = (in_comp,out_comp,t_comp))
            if target != None:
                pred_mat = self.predict(in_comp,t_comp)
                err_after = la.norm(pred_mat - out_comp)
            print('Before... %f'%err_before)
            print('After...%f'%err_after)
        print("~~~~~~~~~~~~~~~~~~~~\n  TRAINING COMPLETE\n~~~~~~~~~~~~~~~~~~~~")
    
#     def plot_maxes(self, ran=10):
#         plt.plot(np.convolve(self.real_maxes, np.ones(ran),'valid')/ran)
#         plt.title('Maximum Eigenvalue Real Part (Moving Average)')
    
    @torch.no_grad()
    def predict(self,u,t):
        self.linear.make_mats()
        return self.net(self.linear.k_gen(u,t)).numpy().T       
    
    def fit_NN(self,in_trn,out_trn,t,target = None):
#         lr = 1e-3
        n_batch = min(self.net.n_batch*10,t.size)
        n_round = self.net.n_round
        if target != None:
            pred_mat = self.predict(target[0],target[2])
            err_before = la.norm(pred_mat - target[1])
        k_trn = self.linear.k_gen(in_trn,t)       # generate k-space trajectory
        x_trn = torch.tensor(out_trn.T)
        data = TensorDataset(k_trn,x_trn)
        loader = DataLoader(data, batch_size=n_batch)
#         net_optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        for k,x in loader:
            # Compute prediction error
            for j in range(self.linear.n_rep):
                pred = self.net(k)
                loss = self.loss_fn(pred, x)

                # Backpropagation
                self.net.optimizer.zero_grad()
                loss.backward()
                self.net.optimizer.step()
    
    
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
                plt.savefig(save+self.label[self.idx_mvmt[ind]])
        print('Numerical error in fit (overall):')
        print(la.norm(pred_mat - out_mat))    
    
    def plot_errs(self, ran = 10, start = 0):
        plt.plot(np.convolve(self.errs[start:], np.ones(ran),'valid')/ran)
        plt.title('Average error over round (Moving Average)')
        
    def plot_errs_true(self, ran = 1, start = 0):
        plt.plot(np.convolve(self.errs_true[start:], np.ones(ran),'valid')/ran)
        plt.title('True error, every %d rounds'%self.true_check)
        
    def plot_changes(self,start = 0, ran=10, cum = False):
        titles = ('B','reals','imags')
        for i in range(len(self.dists)):
            fig, ax = plt.subplots()
            vals = self.dists[i][start:]
            if cum:
                vals = np.cumsum(vals)
            ax.plot(np.convolve(vals, np.ones(ran), 'valid') / ran)
            ax.set_title('Incremental change in '+ titles[i] + ' (Moving average)')
            
    def plot_changes_weight(self,start = 0, ran=10, cum = False):
        titles = ['Layer %d'%(i+1) for i in range(self.n_sets)]
        for i in range(self.n_sets):
            fig, ax = plt.subplots()
            vals = self.dists_weight[i][start:]
            if cum:
                vals = np.cumsum(vals)
            ax.plot(np.convolve(vals, np.ones(ran), 'valid') / ran)
            ax.set_title('Incremental change in '+ titles[i] + ' (Moving average)')
            
    def plot_changes_bias(self,start = 0, ran=10, cum = False):
        titles = ['Layer %d'%(i+1) for i in range(self.n_sets)]
        for i in range(self.n_sets):
            fig, ax = plt.subplots()
            vals = self.dists_bias[i][start:]
            if cum:
                vals = np.cumsum(vals)
            ax.plot(np.convolve(vals, np.ones(ran), 'valid') / ran)
            ax.set_title('Incremental change in '+ titles[i] + ' (Moving average)')
            
    def plot_maxes(self,start = 0, ran=10):
        plt.plot(np.convolve(self.real_maxes[start:], np.ones(ran),'valid')/ran)
        plt.title('Maximum Eigenvalue Real Part (Moving Average)')

    @torch.no_grad()
    def plot_eigs(self, markersize=10):
        plt.plot(self.col(self.linear.reals).T,self.col(self.linear.imags).T,marker = 'x',markersize=markersize)
#         plt.scatter(self.linear.reals,self.linear.imags,marker='x')
        plt.grid(b=True)
        plt.title('Eigenvalues')
        
    def set_label(self,label):
        if not self.trained:
            raise Exception("Label can only be set after training on data")
        if len(label) != self.n_true:
            raise Exception("Number of labels must match output dimension")
        self.label = label