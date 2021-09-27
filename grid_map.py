#!/usr/bin/env python 
# -*- coding: utf-8 -*-
#print __doc__

# Description: function for mapping of the irregulary sampled noise radial profiles
#              the regular grid. mapping is done by inversion of the grid projection 
#              matrix. Inversion is regularized by tikhonov regularization with 
#              an assumption of the minimal diffusion in a cylindrical geometry 
#                ('power' density is expected as constant guess).  Moreover by 'zero_edge' 
#                can by profile forced to by zero at the edge,
#              Robusthes against outliers is improved by two pass calculation and 
    #               error correction by  Winsorizing estimator. 
# Author: Tomas Odstrcil
# mail:tomas.odstrcil@gmail.com
    
import numpy as np
import matplotlib 
import gc
import scipy as sc
#from matplotlib.pyplot import * 
import scipy.sparse as sp
import matplotlib.animation as manimation
import os,sys
import time
from  scipy.stats.mstats import mquantiles
from scipy.ndimage.morphology import binary_erosion,binary_dilation, binary_opening
from collections import OrderedDict
np.seterr(all='raise')
#import warnings
#warnings.simplefilter('error', UserWarning)

debug = False

#try:
from sksparse.cholmod import  cholesky, analyze,cholesky_AAt,CholmodError
chol_inst = True
#except:
    #chol_inst = False
    #print('!CHOLMOD is not installed, slow LU decomposition will be used!')
        ##alternative when cholmod is not availible, spsolve is about 5x slower 
        ##and with higher memory consumption
    #print('try to load OMFIT as module load omfit/conda_0.24.')



def update_fill_between(fill,x,y_low,y_up,min,max ):
    
    paths, = fill.get_paths()
    nx = len(x)

    
    y_low = maximum(y_low, min)
    y_low[y_low==max] = min
    y_up = minimum(y_up,max)
    

    paths.vertices[1:nx+1,1] = y_up
    paths.vertices[nx+1,1] = y_up[-1]
    paths.vertices[nx+2:-1,1] = y_low[::-1]
    paths.vertices[0,1] = y_up[0]
    paths.vertices[-1,1] = y_up[0]
    

    


class map2grid():
    r_min = 0.
    r_max = 1.1  
    def __init__(self,R,T,Y,Yerr,P,W,nr_new,dt ):
        if debug:
            np.savez('map2grid',R=R,T=T,Y=np.array(Y), Yerr=np.array(Yerr),P=P,W=W, nr_new=nr_new,nt_new=dt)
  
        self.dtype='double' #single is much slower, why?

        self.Y = Y.ravel()
        Yerr = Yerr.ravel()
        self.Yerr = np.ma.array(Yerr)
        self.Yerr.mask = (Yerr<=0)|~np.isfinite(Yerr)|self.Yerr.mask 

        self.valid   =  np.isfinite(self.Yerr.data)#&(self.R < self.r_max)
        valid_p =  self.valid[P.flat] & (W.flat != 0)
        self.n_points = np.sum(self.valid)  #BUG 
        
        #remove invalid points from P
        ind_ = np.zeros(len(self.valid), dtype='uint32')
        ind_[np.where(self.valid)[0]] = np.arange(self.n_points, dtype='uint32')

        self.P = ind_[P.flat][valid_p]  #point index - for nonlocal measuremenrs
        self.W = W.flat[valid_p]  #weight   - for nonlocal measuremenrs
        self.R = R.flat[valid_p]  #radial postion 
        self.T = T.flat[valid_p]  #temporal position 
      

        medY = np.median(abs(self.Y[self.valid]))
        self.norm = medY/2

        self.t_min = self.T.min()
        self.t_max = self.T.max()
   
       
    
        self.nr_new = nr_new
        self.nt_new0 = int(np.ceil(round((self.t_max-self.t_min)/dt,2)))+1
        self.nt_new = self.nt_new0 #actual number of profiles, when the regiosn with missing data are removed
       
        self.g = np.zeros((self.nt_new0,nr_new))*np.nan
        self.g0 = np.zeros((self.nt_new0,nr_new)) 

        self.g_u = np.zeros((nr_new,self.nt_new0))*np.nan
        self.g_d = np.zeros((nr_new,self.nt_new0))*np.nan
        
        self.K = np.zeros((nr_new,self.nt_new0))
        self.Kerr_u = np.zeros((nr_new,self.nt_new0))*np.nan
        self.Kerr_d = np.zeros((nr_new,self.nt_new0))*np.nan

        self.retro_f = np.zeros_like(self.Y)*np.nan
        self.missing_data = np.ones(self.nt_new0,dtype=bool)

        self.chi2 = np.nan
        self.lam = np.nan

        
        self.corrected=False
        self.prepared=False
        self.fitted=False
 

    def PrepareCalculation(self, zero_edge=False, core_discontinuties = [],
                           edge_discontinuties = [],transformation = None,even_fun=True,
                           robust_fit=False,pedestal_rho=None, elm_phase=None):
        if debug: print('\nPrepareCalculation')
   
        #BUG removing large number of points, will change missing_data index!!
        
        #savez('discontinuties' ,core_discontinuties,edge_discontinuties,elm_phase)
    
        self.robust_fit = robust_fit
        
        if len(self.P) != sum(self.valid) and transformation[2](100) != 1:
            print('Only linear transformation can be used with line integrated measurements')
            transformation = None
        
        if transformation is None:
            transformation = (lambda x:x,)*2+(lambda x:1,)
 
        self.trans, self.invtrans, self.deriv_trans = transformation
        
 
        #create a geometry matrix 
        dt0 = 1
        if self.nt_new0 > 1:
            dt0 = (self.t_max-self.t_min)/(self.nt_new0-1)


        dr =  (self.r_max-self.r_min)/(self.nr_new-1)
        it =  (self.T-self.t_min)/dt0
        
        #new grid for output
        r_new  = np.linspace(self.r_min,self.r_max,self.nr_new, dtype=self.dtype )
        t_new0 = t_new = np.linspace(self.t_min,self.t_max,self.nt_new0, dtype=self.dtype)
 
 
        #define contribution matrix 
        it[it<0] = 0
        it[it>self.nt_new0-1] = self.nt_new0-1
        points = self.P
        ir = (self.R-self.r_min)/dr
        ir[ir<0] = 0
        ir[ir>self.nr_new-1] = self.nr_new-1
        it = it.astype(self.dtype)
        ir = ir.astype(self.dtype)
        weight  = np.tile(self.W, (4,1))
        index_t = np.empty((4,len(points)),dtype=np.uint32)
        index_r = np.empty((4,len(points)),dtype=np.uint32)
        index_p = np.tile(points, (4,1))

        floor_it = np.int32(it) 
        index_t[ ::2] = floor_it
        index_t[1::2] = np.minimum(floor_it+1, self.nt_new0-1)

        floor_ir = np.int32(ir)
        index_r[:2] = floor_ir
        index_r[2:] = np.minimum(floor_ir+1, self.nr_new-1)
        
        frac_it = np.round(it-floor_it,4).astype(self.dtype) #increase sparsity for regular data!
        frac_ir = ir-floor_ir
 
        weight[ ::2] *= 1.-frac_it
        weight[1::2] *= frac_it
        weight[  :2] *= 1.-frac_ir
        weight[  2:] *= frac_ir

        if elm_phase is None:
            #if elm syncing is not used
            #time regions which are not covered by any measurements
            self.missing_data[index_t[0]] = False
            self.missing_data[index_t[1]] = False

            #weakly constrained timepoints
            weak_data,_ = np.histogram(index_t,self.nt_new0,weights=weight,range=(0,self.nt_new0))
            
            self.missing_data[weak_data<np.mean(weak_data)*.02] = True #almost missing data
                    
            weak_data = (weak_data<np.mean(weak_data)/5.)[~self.missing_data]
            weak_data = weak_data[1:]|weak_data[:-1]

            #correction of dt for regions with a missing or weakly constrained data 
            dt = np.ones(self.nt_new0)
            dt = np.ediff1d(np.cumsum(dt)[~self.missing_data])
            dt = (dt/(1+weak_data))*dt0 
            self.nt_new = np.sum(~self.missing_data)
            
            #skipping a fit in regions without the data
            used_times = np.cumsum(~self.missing_data)-1
            index_t    = used_times[index_t]
            t_new = t_new0[~self.missing_data]
        else:
            self.nt_new = self.nt_new0
            dt = dt0*np.ones(self.nt_new-1 )
            used_times = np.arange(len(t_new0))
            
        self.r_new,self.t_new = np.meshgrid(r_new,t_new)

        weight  =  weight.ravel()
        nonzero = weight != 0  #add only nonzero elements to matrix !
        index_p = index_p.ravel()[nonzero]
        index_rt= (index_r.ravel()*self.nt_new+index_t.ravel())[nonzero]
        npix = self.nr_new*self.nt_new

        # Now, we'll exploit a sparse csc_matrix to build the 2D histogram...
        self.M = sp.csc_matrix((weight[nonzero],(index_p,index_rt)),
                                shape=(self.n_points,npix),dtype=self.dtype)
   
        #imshow(self.M.sum(0).reshape(self.nr_new,self.nt_new), interpolation='nearest', aspect='auto');colorbar();show()
        #imshow(self.M[25000].todense().reshape(self.nr_new,self.nt_new), interpolation='nearest', aspect='auto');colorbar();show()
  

        #prepare smoothing matrix 
        
        #calculate (1+c)*d/dr(1/r*dF/dr) + (1-c)*d^2F/dt^2
        rvec = np.linspace(self.r_min,self.r_max,self.nr_new)
        rvec_b = (rvec[1:]+rvec[:-1])/2
        
        #radial weightng function, it will keep zero gradient in core and allow pedestal 
        self.ifun = np.ones(self.nr_new-1)
        if even_fun:
            self.ifun /= (rvec_b*np.arctan(np.pi*rvec_b)-np.log((np.pi*rvec_b)**2+1)/(2*np.pi))/rvec_b
 
        
        #allow large gradints at pedestal
        if pedestal_rho is not None:
            def gauss(x, x0, s):
                y = np.zeros_like(x)
                ind = np.abs(x-x0)/s < 4
                y[ind] = np.exp(-(x[ind]-x0)**2/(2*s**2))
                return y
            self.ifun/= 1+gauss(rvec_b,pedestal_rho,0.02)*10 +gauss(rvec_b,pedestal_rho+.05,.05)*5
        rweight = np.r_[self.ifun.max(),self.ifun]
        tweight =  np.exp(-rvec)

        #==================time domain===============
        #prepare 3 matrices, for core, midradius and edge
        DTDT = []
        
        #discontinuties
        self.time_breaks = OrderedDict()
        self.time_breaks['core']   = (0.,.3), core_discontinuties
        self.time_breaks['middle'] = (.3,.7), []
        self.time_breaks['edge']   = (.7,2.), edge_discontinuties

        
        for region, (rho_range, time_breaks) in self.time_breaks.items():
            
            break_ind = []
            if not time_breaks is None and len(time_breaks) != 0:
                break_ind = np.unique(t_new0.searchsorted(time_breaks))
                break_ind = break_ind[(break_ind < len(t_new0)-2)&(break_ind > 3)] #to close to boundary 
                if any(break_ind):
                    break_ind = break_ind[np.ediff1d(break_ind,to_begin=10)>1] #to close discontinuties
      
            #remove discontinuties when there are no measurements
            if len(break_ind) > 1 and elm_phase is None:
                break_ind = break_ind[~binary_opening(self.missing_data)[break_ind]]
                #print used_times.shape, break_ind.max()
                break_ind = np.unique(used_times[break_ind])
   
            if self.nt_new > 1:
                DT = np.zeros((3,self.nt_new),dtype=self.dtype)
                #minimize second derivative
                DT[0,0:-2] =  .5/(dt[:-1]+dt[1:])/dt[:-1]
                DT[1,1:-1] = -.5/(dt[:-1]+dt[1:])*(1/dt[:-1]+1/dt[1:])
                DT[2,2:  ] =  .5/(dt[:-1]+dt[1:])/dt[ 1:]
                #force zero 1. derivative at the edge
                DT[[2,0],[1,-2]] =  1/dt0**2/2
                DT[   1 ,[0,-1]] = -1/dt0**2/2

                #discontinuties
                if len(break_ind) > 0:
                    DT[0, break_ind-2] =  1/dt0
                    DT[1, break_ind-1] = -1/dt0
                    DT[2, break_ind-0] = 0
                    DT[0, break_ind-1] = 0
                    DT[1, break_ind-0] = -1/dt0
                    DT[2, break_ind+1] =  1/dt0
                DT *= dt0**2
                DT = sp.spdiags(DT,(-1,0,1),self.nt_new,self.nt_new)
         
                if elm_phase is not None and region == 'edge':
                    phase = np.interp(t_new,elm_phase[0],elm_phase[1], left=0,right=0)
                    elm_start = elm_phase[0][elm_phase[1]==-1]
                    elm_start_ind = np.arange(len(elm_phase[0]))[elm_phase[1]==-1]
                    DT_elm_sync = []
                    for it,(t,p) in enumerate(zip(t_new,phase)):
                    
                        #find point in the next elm with nearest phase value
                        if t > elm_start[-1]: continue
                            
                        ip = elm_start_ind[elm_start.searchsorted(t)]  
                        if ip < 6 or ip >= len(elm_phase[0]) -2:
                            continue
                    
                        ind_next = (t_new>=elm_phase[0][ip  ])&(t_new<=elm_phase[0][ip+2])
                        ind_prev = (t_new>=elm_phase[0][ip-6])&(t_new<=elm_phase[0][ip-4])
             
                        if any(ind_next):
                            next_it2 = min(len(t_new)-1 ,t_new.searchsorted(elm_phase[0][ip])+phase[ind_next].searchsorted(p))
                            if phase[next_it2]< p: next_it2 -= 1
                            next_it1 = next_it2-1 if phase[next_it2-1] < phase[next_it2] else next_it2
                            w = max(0,(phase[next_it2]-p)/(phase[next_it2]-phase[next_it1])) if next_it2!=next_it1  else .5
                            
                            if next_it2 < len(t_new):
                                DT_elm_sync.append((-w, it,next_it1))  
                                DT_elm_sync.append((-(1-w), it,next_it2))  
                                

                            
                        if any(ind_prev):
                            prev_it2 = max(0 ,t_new.searchsorted(elm_phase[0][ip-6])+phase[ind_prev].searchsorted(p))
                            if phase[prev_it2]< p: prev_it2 -= 1
                            prev_it1 = prev_it2-1 if phase[prev_it2-1] < phase[prev_it2] else prev_it2
                            w = max(0,(phase[prev_it2]-p)/(phase[prev_it2]-phase[prev_it1])) if prev_it2!=prev_it1  else .5
                            if prev_it1 >= 0:
                                DT_elm_sync.append((-w, it,prev_it1))  
                                DT_elm_sync.append((-(1-w), it,prev_it2))  
                    if len(DT_elm_sync) == 0:
                        print('No ELMS for synchronisation')
 
                    else:
                        #add elm syncronisation to time derivative matrix 
                        A,I,J = np.array(DT_elm_sync).T
                        B = sp.coo_matrix((A,(I,J)),DT.shape)
                        B = B- sp.spdiags(B.sum(1).T,0,*DT.shape )
                        DT = sp.vstack((B/2., DT/4.),  format='csr', dtype=self.dtype)
    
                    
                
            else:
                DT = np.matrix(0)
      
            r_range = (r_new>=rho_range[0])&(r_new<rho_range[1])
            W = sp.diags(tweight[r_range],0, dtype=self.dtype)
            DTDT.append(sp.kron(W**2,DT.T*DT))
        self.DTDT = sp.block_diag(DTDT, format='csc')

        #import IPython
        #IPython.embed()
    
        #==============radial domain===============
        DR = np.zeros((3,self.nr_new),dtype=self.dtype)

        #zero 2. derivative at the edge
        DR[0, :-2] =  self.ifun[:-1]
        DR[1,1:-1] = -self.ifun[:-1]-self.ifun[1:]
        DR[2,2:  ] =  self.ifun[ 1:]
        if self.deriv_trans(0) == 1:#linear regularisation
            #zero 1. derivative at the edge
            DR[0, -2] =  self.ifun[-1]
            DR[1,-1 ] = -self.ifun[-1]
            #DR[1,-2] += DR[0,-3]
            #DR[0,-3] = 0
        if zero_edge and self.trans(0) == 0:  
            #zero 0. derivative at the edge
            #DR[1,-2] += DR[0,-3]
            #DR[0,-3] = 0
            DR[0, -2] =  self.ifun[-1]
            DR[1,-1 ] = -self.ifun[-1]
            #press edge to zero 
            DR[1,-1] = 10
 
    
        DR = sp.spdiags(DR,(-1,0,1),self.nr_new,self.nr_new)
        self.DRDR = sp.kron(DR.T*DR,sp.eye(self.nt_new, dtype=self.dtype), format='csc')
        self.prepared = True
            
     
    
    def PreCalculate(self ):
    
  
        
        if not self.prepared:
            self.PrepareCalculation()
        
        if debug:print('Precalculate')
        #first pass - decrease the weight of the outliers

            
        Y    = np.copy(self.Y[self.valid])/self.norm
        Yerr = np.copy(self.Yerr.data[self.valid])/self.norm
        Yerr *= self.deriv_trans(Y)
        
        Yerr[self.Yerr.mask[self.valid]] = np.infty

        #transform and weight by uncertainty
        try:
            self.f = self.trans(Y)/Yerr 
        except:
            print('s fit error np.sum(Yerr == 0), np.sum(Yerr < 0), np.sum(~np.isfinite(self.trans(Y)))', np.sum(Yerr == 0), np.sum(Yerr < 0), np.sum(~np.isfinite(self.trans(Y))))
            raise

        self.V = sp.spdiags(1/Yerr,0, self.n_points,self.n_points,format='csr')*self.M
        self.V.eliminate_zeros()  
        self.VV = self.V.T*self.V #slow 

        
        vvtrace = self.VV.diagonal().sum()
        lam = np.exp(16)/self.DRDR.diagonal().sum()*vvtrace
        dt_diag = self.DTDT.diagonal().sum()
        if dt_diag == 0: dt_diag = 1
        eta = np.exp(11)/dt_diag*vvtrace
    
        if self.nt_new == 1: eta = 0
        AA = self.VV+lam*self.DRDR+eta*self.DTDT
   
        if chol_inst:
            self.Factor = analyze(AA, ordering_method='colamd')
 
        self.corrected=True

        if self.robust_fit:
        
            if chol_inst:
                self.Factor.cholesky_inplace(AA)
            else:
                self.Factor = sp.linalg.factorized(AA)
                print('umfpack')
 
            g=np.squeeze(self.Factor(self.V.T*self.f))

            #make a more robust fit in one iteration
            dist = self.V*g-self.f  
            
            #Winsorizing M estimator, penalize outliers
            Yerr_corr = ((dist/3.)**2+1)**(1)  

            self.f /= Yerr_corr
            self.V = sp.spdiags(1/Yerr_corr,0, self.n_points,self.n_points,format='csr')*self.V
            self.VV = self.V.T*self.V


            
        
    def Calculate(self,lam,eta):

        if not self.corrected:
            self.PreCalculate()
        if debug:print('Calculate')

        np.random.seed(0)
        #noise vector for estimation of uncertainty
        n_noise_vec = 50
        noise = np.random.randn(self.n_points, n_noise_vec).astype(self.dtype)

        noise += np.random.randn(n_noise_vec)#correlated noise, most pesimistic assumption

        #BUG hardcodded!!
        R0 = 1.7 #m  
        a0 = 0.6 #m               
  

        vvtrace = self.VV.diagonal().sum()
        lam = np.exp( 8*(lam-.5)+14)/self.DRDR.diagonal().sum()*vvtrace*lam/(1.001-lam)
        dtdiag = self.DTDT.diagonal().sum()
        if dtdiag== 0: dtdiag= 1
        eta = np.exp(20*(eta-.5)+ 5)/dtdiag*vvtrace*eta/(1.001-eta)
        if self.nt_new == 1: eta = 0

        AA = self.VV+lam*self.DRDR+eta*self.DTDT

        #print 'TRACE %.3e  %.3e'%( self.DRDR.diagonal().sum()/vvtrace,self.DTDT.diagonal().sum()/vvtrace)
        
        try:
            self.Factor.cholesky_inplace(AA)
        except:
            self.Factor = sp.linalg.factorized(AA)

  
        Y = self.Y[self.valid]/self.norm

        Yerr = self.deriv_trans(Y)*self.Yerr.data[self.valid]/self.norm
        Yerr[self.Yerr.mask[self.valid]] = np.infty
        Y = self.trans(Y)

        g = np.squeeze(self.Factor(self.V.T*self.f))
        self.chi2 = np.linalg.norm((self.M*g-Y)/Yerr)**2/np.sum(np.isfinite(Yerr))
        self.lam = lam
        
        #estimate uncertainty of the fit
        noise_scale = np.maximum(abs(self.V*g-self.f), 1)
          
        #import IPython
        #IPython.embed()
        g_noise = self.Factor((self.V.T)*(noise*noise_scale[:,None]))#SLOW 
        g_noise += (g - g_noise.mean(1))[:,None]
        g_noise = g_noise.reshape(self.nr_new,self.nt_new,n_noise_vec).T
        g_noise = self.invtrans(g_noise)*self.norm
    
        #print '\nchi2: %.2f reg:%.2f'%(self.chi2,self.lam)
        
        
    
        self.retro_f[self.valid] = self.invtrans((self.M*g))*self.norm
 
        self.g = self.invtrans(g.reshape(self.nr_new,self.nt_new).T)*self.norm

 
        #if too many points was removed, it can happen  that some timepoinst was not measured at all
        valid_times = np.reshape(self.V.sum(0) > 0,(self.nr_new,self.nt_new)).T
        valid_times = np.any(np.asarray(valid_times),1)
        valid_times |= True

        self.g = self.g[valid_times] 
        g_noise= g_noise[:,valid_times] 
        self.g_samples = np.copy(g_noise)
 
        #find lower and upper uncertainty level, much faster then using mquantiles..
        rvec = np.linspace(self.r_min,self.r_max,self.nr_new)
        rvec_ = (rvec[1:]+rvec[:-1])/2

        K       = -np.diff(self.g )/(self.g[:,1:]+self.g[:,:-1]      )/(rvec_*np.diff(rvec*a0))*R0*2
        K_noise = -np.diff(g_noise)/(g_noise[...,1:]+g_noise[...,:-1])/(rvec_*np.diff(rvec*a0))*R0*2
        
        K_noise.sort(0)

        Kerr_u = np.nanmean(K_noise[n_noise_vec//2:],axis=0)
        Kerr_d = np.nanmean(K_noise[:n_noise_vec//2],axis=0)
        
        g_noise.sort(0)

        self.g_u = np.mean(g_noise[n_noise_vec//2:],axis=0)
        self.g_d = np.mean(g_noise[:n_noise_vec//2],axis=0)

        self.K = np.c_[K,K[:,-1]]
        self.Kerr_u = np.c_[Kerr_u,Kerr_u[:,-1]] 
        self.Kerr_d = np.c_[Kerr_d,Kerr_d[:,-1]]
        self.fitted=True


        self.g_t = self.t_new[valid_times] 
        self.g_r = self.r_new[valid_times] 
        
        return self.g,self.g_u,self.g_d, self.g_t, self.g_r
        
        
    def finish(self):
        try:
            del self.Factor, self.DTDT,self.DRDR,self.VV, self.V
        except:
            pass
        gc.collect()
      
        

























def main():
    
    import pickle as pkl
    from matplotlib.pylab import plt

    with open('data_for_tomas.pkl','rb') as f:
        data = pkl.load(f)
    out_ne, out_Te = data
    rhop_ne, ne, ne_err, rhop_ne_err = out_ne
    rhop_Te, Te, Te_err, rhop_Te_err = out_ne
    
    nt = 1
    nr = 201
    nd = len(ne)
    
    MG = map2grid(rhop_Te,np.zeros(nd),Te,Te_err,np.arange(nd),np.ones(nd),nr,nt)
    
    MG.r_max = 1.04
    
    transformation =  lambda x: np.sqrt(np.maximum(0, x)), np.square,lambda x:.5/np.sqrt(np.maximum(1e-5, x))
    transformation =   lambda x: np.log(np.maximum(x,0)/.1+1),  lambda x:np.maximum(np.exp(x)-1,1.e-6)*.1,   lambda x:1/(.1+np.maximum(0, x))  #not a real logarithm..

    MG.PrepareCalculation(transformation=transformation, zero_edge=False)
    MG.PreCalculate( )
    
    
    MG.Calculate(0.1, 0.1)
    
    
    plt.errorbar(rhop_Te, Te, Te_err, fmt='.')
    plt.plot( MG.r_new[0], MG.g_samples[:,0].T,lw=.2,c='r')
    plt.plot( MG.r_new[0], MG.g[0] ,lw=2,c='r')
    plt.xlim(0,1.1)
    plt.ylim(0,None)
    plt.show()
    
    
if __name__ == "__main__":
    main()
 

