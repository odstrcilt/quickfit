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
    
#algorithm shares similarity with Kalman Filters???
#computation complexity O(nr*nt+min(nr,nt)), weakly dependent on npoints
#
# Author: Tomas Odstrcil
# mail:tomas.odstrcil@gmail.com
    
import numpy as np
import matplotlib 
import gc
import scipy as sc
from matplotlib.pyplot import * 
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
from  IPython import embed
debug = False
import matplotlib.pylab as plt
import traceback

#try:
from sksparse.cholmod import  cholesky, analyze,cholesky_AAt,CholmodError
chol_inst = True
#except:
    #chol_inst = False
    #print('!CHOLMOD is not installed, slow LU decomposition will be used!')
        ##alternative when cholmod is not availible, spsolve is about 5x slower 
        ##and with higher memory consumption
    #print('try to load OMFIT as module load omfit/conda_0.24.')

 


class map2grid():
    r_min = 0.
    r_max = 1.1  
    dtype = 'double' #single is much slower? 
    
    def __init__(self,R,T,Y,Yerr,P=None,W=None,nr_new=101,dt=1 ):
        if debug:
            TT = time.time()
            #np.savez('map2grid',R=R,T=T,Y=np.array(Y), Yerr=np.array(Yerr),P=P,W=W, nr_new=nr_new,nt_new=dt)
  
        
        if W is None:
            W = np.ones_like(Y)
        if P is None:
            P = np.arange(Y.size)
                        

        self.Y = Y.ravel()
        Yerr = Yerr.ravel()
        self.Yerr = np.ma.array(Yerr)
        self.Yerr.mask = (Yerr<=0) |~np.isfinite(Yerr) | self.Yerr.mask 

        self.valid   =  np.isfinite(self.Yerr.data)
        
 
        valid_p =  self.valid[P.flat] & (W.flat != 0)&(R.ravel() < self.r_max)
        self.n_points = np.sum(self.valid)  #BUG 
        
        #remove invalid points from P
        ind_ = np.zeros(len(self.valid), dtype='uint32')
        ind_[self.valid] = np.arange(self.n_points, dtype='uint32')

        self.P = ind_[P.flat][valid_p]  #point index - for nonlocal measuremenrs
        self.W = W.flat[valid_p]  #weight   - for nonlocal measuremenrs
        self.R = R.flat[valid_p]  #radial postion 
        self.T = T.flat[valid_p]  #temporal position 

        medY = np.median(abs(self.Y[self.valid]))
        self.norm = medY/2


        self.t_min = self.T.min()
        self.t_max = self.T.max()
        
        #shift of time grid to align with most of the temepoints to reduce spartiy of V matrix, 
        it = (self.T-self.t_min)/dt+1e-4#add small constant due to rounding errors
        it-= np.int32(it)
        t_shift = np.median(it) 
        self.t_min += t_shift*dt #little shift 
        if t_shift > 0: #self.t_min must be always less then the min(T)
            self.t_min -= dt 
        self.t_new0 = np.arange(self.t_min,self.t_max+dt*1.5, dt)
        
        #self.t_max = self.t_min+np.ceil((self.t_max-self.t_min)/dt+1)*dt
        self.t_max = self.t_new0[-1]
        self.nt_new0 = len(self.t_new0)
        
        self.dt = dt
        self.nr_new = nr_new
        #self.nt_new0 = int(round((self.t_max-self.t_min)/dt,2))
        #self.nt_new = self.nt_new0 #actual number of profiles, when the regiosn with missing data are removed
    
        #t_new0 = t_new = np.linspace(self.t_min,self.t_max,self.nt_new0)


        #embed()
       
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
        #IPython.embed()
        #return
 
        #r_new = np.linspace(self.r_min,self.r_max,self.nr_new)
        
        #sigma_t = 0.02
        #sigma_r = 0.005
        #dist_d = (self.R-self.R[:,None])**2/sigma_r+(self.T-self.T[:,None])**2/sigma_t
        #ind = dist_d < 3
        #print('density', ind.sum()/ind.size)
        #Kdd = sp.csr_matrix((np.exp(-dist_d[ind]), np.where(ind)), shape=(self.n_points,self.n_points))
        #sigma_d = sp.spdiags(self.Yerr.data[self.valid],0, self.n_points,self.n_points,format='csr')
        #factor =  cholesky(Kdd+sigma_d)  #slo by updatovat diagonalu!!
        #KY = factor.solve_A(self.Y[self.valid])
        
        ##this needs to be evaluated for every time or radius
        #T0 = 5 #s

        #Ksd = np.exp(np.maximum(-(self.R-r_new[:,None])**2/sigma_r+-(self.T-T0)**2/sigma_t,-10))
        #mu = np.dot(Ksd,KY)
        #plot(r_new, mu);
        #ind_d = abs(self.T-T0) < 0.01
        #plt.plot(self.R[ind_d], self.Y[self.valid][ind_d],'o' )
        #plt.show()

        

        
 
        if debug:
            print('init',time.time()-TT)
            
    def PrepareCalculation(self, zero_edge=False, core_discontinuties = [],
                           edge_discontinuties = [],transformation = None,even_fun=True,
                           robust_fit=False,pedestal_rho=None, elm_phase=None):
        if debug:
            TT = time.time()
            print('\nPrepareCalculation')
            #np.savez('discontinuties' ,core_discontinuties,edge_discontinuties,elm_phase)

        #BUG removing large number of points, will change missing_data index!!
        #if debug:

    
        self.robust_fit = robust_fit
        #embed()
        if len(self.P) > self.n_points and transformation[2](100) != 1:
            print('Only linear transformation can be used with line integrated measurements')
            transformation = None
        
        if transformation is None:
            transformation = (lambda x:x,)*2+(lambda x:1,)
 
        self.trans, self.invtrans, self.deriv_trans = transformation
        
 
        #=============== define contribution matrix  ==================
        # it is a sparse matrix representation of bilinear interpolation

        dr = (self.r_max-self.r_min)/(self.nr_new-1)
        it = (self.T-self.t_min)/self.dt
        ir = (self.R-self.r_min)/dr
        
 
        #new grid for output
        r_new  = np.linspace(self.r_min,self.r_max,self.nr_new)
        #t_new0 = t_new = np.linspace(self.t_min,self.t_max,self.nt_new0)
        
        floor_it = np.uint32(it) 
        floor_ir = np.uint32(ir)

        weight  = np.tile(self.W, (4,1))
        index_p = np.tile(self.P, (4,1))
        index_t = np.tile(floor_it, (4,1))
        index_r = np.tile(floor_ir, (4,1))
 
        index_t[1::2] += 1
        index_r[2:  ] += 1
        
        #fast rounding by 3 digits to increase sparsity for regularly spaced data and remove  rounding error
        frac_it = np.uint32((it-floor_it)*1e3+0.5)/1e3
        frac_ir = ir-floor_ir
        
        #bilinear weights
        weight[ ::2] *= 1.-frac_it
        weight[1::2] *= frac_it
        weight[  :2] *= 1.-frac_ir
        weight[2:  ] *= frac_ir
   
        #skip fit of temporal regions without any data
        if elm_phase is None:
            #if elm syncing is not used
            #idenify time bins which are not covered by any measurements
            try:
                self.missing_data[index_t[0]] = False
                self.missing_data[index_t[1]] = False
            except:
                print('error:  self.missing_data[index_t[1]] = False ')
            #weakly constrained timepoints
            weak_data,_ = np.histogram(index_t,self.nt_new0,weights=weight,range=(0,self.nt_new0))
             
            self.missing_data[weak_data<np.mean(weak_data)*.02] = True #almost unconstrained time bins
       
            #correction of dt for regions with a missing or weakly constrained data 
            dt = np.ones(self.nt_new0)
            dt = np.ediff1d(np.cumsum(dt)[~self.missing_data])
            
            #BUG what was this part for??
            #weak_data = (weak_data<np.mean(weak_data)*.2)[~self.missing_data]
            #weak_data = weak_data[1:]|weak_data[:-1]
            #dt = dt/(1+weak_data)
           
            self.nt_new = np.sum(~self.missing_data)
            #skipping a fit in regions without the data
            used_times = np.cumsum(~self.missing_data)-1
            index_t    = used_times[index_t]
            t_new = self.t_new0[~self.missing_data]
            self.elm_phase = False
        else:
            t_new = self.t_new0
            self.nt_new = self.nt_new0
            dt = np.ones(self.nt_new-1 )
            used_times = np.arange(len(self.t_new0))
            self.elm_phase = True
        
        dt *= self.dt
        self.r_new,self.t_new = np.meshgrid(r_new,t_new)

        weight  = weight.ravel()
        nonzero = weight != 0  #add only nonzero elements to matrix !
        index_p = index_p.ravel()[nonzero]
        index_rt= (index_r.ravel()*self.nt_new+index_t.ravel())[nonzero]
        if np.any(index_rt < 0):
            index_rt[index_rt < 0] = 0
        npix = self.nr_new*self.nt_new
        # Now, we'll exploit a sparse csc_matrix to build the 2D histogram...
        try:
            self.M = sp.csc_matrix((weight[nonzero],(index_p,index_rt)),
                                    shape=(self.n_points,npix))
        except:
            
            traceback.print_exc()
            embed()

    
        if debug:
            print('compression',self.M.data.size/(len(self.P)*4))
            print('prepare V', time.time()-TT)
            TT = time.time()

        #imshow(self.M.sum(0).reshape(self.nr_new,self.nt_new), interpolation='nearest', aspect='auto');colorbar();show()
        #imshow(self.M[25000].todense().reshape(self.nr_new,self.nt_new), interpolation='nearest', aspect='auto');colorbar();show()
  

        #prepare regularisation matrix 
        
        #calculate (1+c)*d/dr(1/r*dF/dr) + (1-c)*d^2F/dt^2
        rvec = np.linspace(self.r_min,self.r_max,self.nr_new)
        rvec_b = (rvec[1:]+rvec[:-1])/2
        
        #radial weightng function, it will keep zero gradient in core and allow pedestal 
        diffusion = np.ones(self.nr_new-1)
        if even_fun:
            #A zero slope constraint is imposed at the magnetic axis
            #diffusion /= (rvec_b*np.arctan(np.pi*rvec_b)-np.log((np.pi*rvec_b)**2+1)/(2*np.pi))/rvec_b
            #diffusion /= np.arctan( 3/2*rvec_b)
            
            #from  scipy.special import erf
            #erf = 2/sqrt(pi)*integral(exp(-t**2), t=0..z).
            #diffusion /= erf(rvec_b)
            
            #Gamma = 1/(r*pi*2*pi*R)*integral r*S
            #for S is a gaussian profile of the source exp(-x^2)
            diffusion /= (1-np.exp(-rvec_b**2))/rvec_b

            
            #plt.plot(erf(rvec_b),':')
            #plt.plot(np.arctan( 3/2*rvec_b))
            #plt.plot((rvec_b*np.arctan(np.pi*rvec_b)-np.log((np.pi*rvec_b)**2+1)/(2*np.pi))/rvec_b,'-.')
            #plt.plot((1-np.exp(-rvec_b**2))/rvec_b*1.5,'--')
            #plt.show()



        
        #allow large gradints at pedestal
        if pedestal_rho is not None:
            def gauss(x, x0, s):
                y = np.zeros_like(x)
                ind = np.abs(x-x0)/s < 4
                y[ind] = np.exp(-(x[ind]-x0)**2/(2*s**2))
                return y
            diffusion /= 1+gauss(rvec_b,pedestal_rho,0.02)*10 +gauss(rvec_b,pedestal_rho+.05,.05)*5
        
        tweight =  np.exp(-rvec)

        #==================time domain===============
        #prepare 3 matrices, for core, midradius and edge
        DTDT = []
        
        #discontinuties, regions must cover whole range and do not overlap
        self.time_breaks = OrderedDict()
        self.time_breaks['core']   = (0.,.3), core_discontinuties
        self.time_breaks['middle'] = (.3,.6), []
        self.time_breaks['edge']   = (.6,2.), edge_discontinuties
        
        if len(core_discontinuties) == 0:
            self.time_breaks['middle'] = (0,.6), []
            del self.time_breaks['core']
        if len(edge_discontinuties) == 0:
            del self.time_breaks['edge']
            self.time_breaks['middle'] = (self.time_breaks['middle'][0][0],2), []

        #iterate over all regions
        for region, (rho_range, time_breaks) in self.time_breaks.items():
            
            break_ind = []
            if not time_breaks is None and len(time_breaks) != 0:
                break_ind = np.unique(self.t_new0.searchsorted(time_breaks))
                break_ind = break_ind[(break_ind < len(self.t_new0)-2)&(break_ind > 3)] #to close to boundary 
                if any(break_ind):
                    break_ind = break_ind[np.ediff1d(break_ind,to_begin=10)>1] #to close discontinuties
      
            #TODO remove discontinuties when there are no measurements
            if len(break_ind) > 1 and elm_phase is None:
                break_ind = break_ind[~binary_opening(self.missing_data)[break_ind]]
                break_ind = np.unique(used_times[break_ind])
   
            if self.nt_new > 1:
                DT = np.zeros((3,self.nt_new))
                #minimize second derivative
                DT[0,0:-2] =  .5/(dt[:-1]+dt[1:])/dt[:-1]
                DT[1,1:-1] = -.5/(dt[:-1]+dt[1:])*(1/dt[:-1]+1/dt[1:])
                DT[2,2:  ] =  .5/(dt[:-1]+dt[1:])/dt[ 1:]
                #force zero 1. derivative at the edge of the discontinuity
                DT[[2,0],[1,-2]] =  1/self.dt**2/2
                DT[   1 ,[0,-1]] = -1/self.dt**2/2

                #introduce discontinuties to a time derivative matrix
                if len(break_ind) > 0:
                    DT[0, break_ind-2] =  1/self.dt
                    DT[1, break_ind-1] = -1/self.dt
                    DT[2, break_ind-0] = 0
                    DT[0, break_ind-1] = 0
                    DT[1, break_ind-0] = -1/self.dt
                    DT[2, break_ind+1] =  1/self.dt
                    
                DT *= self.dt
                DT = sp.spdiags(DT,(-1,0,1),self.nt_new,self.nt_new)
                
        
                
                #add correlation between timeslices at the same elm phase
                if elm_phase is not None and region == 'edge':
                    phase = np.interp(t_new,elm_phase[0],elm_phase[1], left=0,right=0)
                    nelm = len(elm_phase[0])
                    elm_start = elm_phase[0][elm_phase[1]==-1]
                    elm_start_ind = np.arange(nelm)[elm_phase[1]==-1]
                    DT_elm_sync = [] #indexes and weighs in the elm synchronisatiom matrix
                    #iterate over each timeslice of the timegrid
                    for it,(t,p) in enumerate(zip(t_new,phase)):
                        #iterate over one left on the left and one on the right
   
                        for side in ['L','R']:
                            #find point in the next elm with nearest phase value  
            
                            #check that the current elm is not the first/last elm
                            if len(elm_start) > 2 and (t < elm_start[-1] and side == 'R') or (t > elm_start[1] and side == 'L'): 
                                #index of the previous/next elm
                                ielm = elm_start.searchsorted(t)
                                
                                if side == 'L': ielm -= 2
                                assert ielm >= 0, 'ielm > 0'
                                
                                #select the elm
                                ip = elm_start_ind[ielm]                              
                                elm_beg = elm_phase[0][ip]
                                elm_end = np.inf if ip+3 > nelm else elm_phase[0][ip+2]
                                dt_elm = elm_end-elm_beg
                                #interval inside of range [elm_beg, elm_end]
                                ind_next = slice(*t_new.searchsorted((elm_beg, elm_end)))
                                
                                #at least 6 time slices within the elm, else skip it
                                if ind_next.stop-ind_next.start > 5:
                                    iphase = phase[ind_next].searchsorted(p) 
                                    if 0 < iphase < (ind_next.stop-ind_next.start): #inside of left and right edge
                                        
                                        next_it_r = ind_next.start+iphase
                                        next_it_l = next_it_r-1
                                        
                                        #weight of the left point
                                        w = (phase[next_it_r]-p)/(phase[next_it_r]-phase[next_it_l])
                                        assert 0<=w<=1, 'w > 0'

                                        
                                        DT_elm_sync.append((it,next_it_l, -w*dt_elm))  
                                        DT_elm_sync.append((it,next_it_r,-(1-w)*dt_elm)) 
                 

                                    elif iphase == 0 and ind_next.start != 0:#if it is not edge of the grid 
                                        DT_elm_sync.append((it,ind_next.start, -dt_elm))  
                                    
                                    elif iphase == (ind_next.stop-ind_next.start) and ind_next.stop != len(t_new):#if it is not edge of the grid 
                                        DT_elm_sync.append((it,ind_next.stop-1, -dt_elm))  
  
                    if len(DT_elm_sync) == 0:
                        print('No ELMS for synchronisation')
 
                    else:
                        #add elm synchronisation to time derivative matrix 
                        I,J,W = np.array(DT_elm_sync).T
                        B = sp.coo_matrix((W/self.dt,(I,J)),(self.nt_new,self.nt_new))  #normalise it by average lenght of elms??
                        B = B - sp.spdiags(B.sum(1).T,0,self.nt_new,self.nt_new )#diagonal value at it should by +2
                        DT = sp.vstack((B/4., DT/4.),  format='csr')

                
            else:
                DT = np.matrix(0)
            #apply DT just in a selected region
            r_range = (r_new>=rho_range[0])&(r_new<rho_range[1])
            W = sp.diags(tweight[r_range],0)
            DTDT.append(sp.kron(W**2,DT.T*DT))
        
        #merge all regions together
        if len(DTDT) > 1:
            self.DTDT = sp.block_diag(DTDT,format='csc')
        else:
            self.DTDT = DTDT[0]
            
    
        #==============radial domain===============  
        #build operator of 1. derivative
        DR = np.zeros((2,self.nr_new))
        DR[0,0:] =  1
        DR[1,0:] = -1
        DR = sp.spdiags(DR,(1,0),self.nr_new,self.nr_new)
        DD = sp.spdiags(diffusion,0,self.nr_new,self.nr_new)
        #grad D grad operator
        DR =  DR*DD*DR

        if zero_edge and self.trans(0) == 0: 
            #press edge to zero
            DR = DR.tolil()
            DR[-1,-1] = 10
        elif self.trans(0) != 0:
            #zero 2. derivative at the edge
            DR[-2,-2:] = 0
        

        I = sp.eye(self.nt_new)
        self.DRDR = sp.kron(DR.T*DR,I, format='csc')

        self.prepared = True

            
        if debug:
            print('prepare DT',time.time()-TT)
            
    
    def PreCalculate(self ):
    
  
        
        if not self.prepared:
            self.PrepareCalculation()
        
        if debug:
            print('Precalculate')
            TT = time.time()

        #first pass - decrease the weight of the outliers
        #embed()
            
        Y    = self.Y[self.valid]/self.norm
        Yerr = self.Yerr.data[self.valid]/self.norm
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
        #VV is called "precision matrix {\displaystyle \mathbf {\Lambda } =\mathbf {\Sigma } ^{-1}}" https://en.wikipedia.org/wiki/Gaussian_process_approximations
        
        self.VV = self.V.T*self.V #slow 

        #heurestic method for estimate of lambda 
        vvtrace = self.VV.diagonal().sum()
        lam = np.exp(16)/self.DRDR.diagonal().sum()*vvtrace
        dt_diag = self.DTDT.diagonal().sum()
        if dt_diag == 0: dt_diag = 1
        eta = np.exp(11)/dt_diag*vvtrace
        
        #/(self.dt/0.01)
    
        if self.nt_new == 1: eta = 0
        #DRDR -s 5-diagonal, DTDT is also 5 diagonal, VV 7 diagonal, AA is 9 diagonal
        AA = self.VV+lam*self.DRDR+eta*self.DTDT

        if chol_inst:
            #TODO use METIS only for line interated data and elm sync??
            if self.elm_phase:
                method='metis' #colamd and amd has troubles with large lower sparsity matrices
            else:
                method='colamd'
            self.Factor = analyze(AA, ordering_method=method) #colamd and amd has troubles with large lower sparsity matrices
            #print(method)
            #self.Factor.cholesky_inplace(AA)#BUG
 
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


        if debug:
            print('Precalculate',time.time()-TT)
    
              
    def Evidence(self,lam,eta):
 
        vvtrace = self.VV.diagonal().sum()
        lam = np.exp( 8*(lam-.5)+14)/self.DRDR.diagonal().sum()*vvtrace*lam/(1.001-lam)
        eta = np.exp(20*(eta-.5)+-10)*vvtrace*eta/(1.001-eta)
        
        #vvtrace = self.VV.diagonal().sum()
        #lam = np.exp( 8*(lam-.5)+14)/self.DRDR.diagonal().sum()*vvtrace*lam/(1.001-lam)
        #eta = np.exp(20*(eta-.5)+-10)*vvtrace*eta/(1.001-eta)
        
        
        if self.nt_new == 1: eta = 0
        #embed()

        AA = self.VV+lam*self.DRDR+eta*self.DTDT

        self.Factor.cholesky_inplace(AA,1e-8)
        #TODO calculate L-^1*V
        X = self.V*self.Factor(self.V.T.todense())
        invSprior = np.eye(self.n_points)-X

        #embed()
        
        #LtVt = self.Factor.solve_Lt(self.Factor.apply_P(self.V.T.todense()),use_LDLt_decomposition=False)
        #np.dot(LtVt.T, LtVt)
        
        #%timeit self.Factor.apply_Pt(self.Factor.solve_Lt(self.Factor.apply_P(self.V.T),use_LDLt_decomposition=False))
        
        
        #logdet = np.sum(np.log(np.diag(np.linalg.cholesky(Sprior))))*2
        s,logdet = np.linalg.slogdet(invSprior)
        #embed()
        logdet *= -1 #inversion

        fit = np.array(np.dot(self.f, np.dot(invSprior,self.f).T))

        logev = -0.5*(logdet+fit+self.n_points*np.log(2*np.pi))
        
        #embed()


        V = np.linalg.eigvalsh(X)
        n = len(invSprior)
        f = (1-np.sum(V)/n)**2

        tr = (np.trace(invSprior)/n)**2
        chi2n = np.sum(np.dot(invSprior,self.f))**2/n
        
        #return chi2n/tr
        

        
        
        return logev,-np.log(chi2n/f) 
    
    def GCV(self,lam,eta):
 
        #linear in n_p*n_t*n_r**2
         
 
        vvtrace = self.VV.diagonal().sum()
        lam = np.exp( 8*(lam-.5)+14)/self.DRDR.diagonal().sum()*vvtrace*lam/(1.001-lam)
        eta = np.exp(20*(eta-.5)+-10)*vvtrace*eta/(1.001-eta)
         
        
        if self.nt_new == 1: eta = 0

        AA = lam*self.DRDR+eta*self.DTDT

        self.Factor.cholesky_inplace(AA,1e-8)
        #embed()
        
        #DV = self.Factor(self.V.T.todense())
        
        LPV = self.Factor.solve_L(self.Factor.apply_P(self.V.T)) 
        zero_ins = np.array(LPV.sum(1)==0).ravel()|(self.Factor.D()<=0)
        
        LPV = LPV.toarray() #2.5ms
        LPV = LPV[~zero_ins].T  #.5ms
        sqrtD = np.sqrt(self.Factor.D()[~zero_ins])
        LPV/= sqrtD
  
        LL = np.dot(LPV, LPV.T)  #Gram matrix
  
        from  scipy.linalg import eigh 
        s,u = eigh(LL,overwrite_a=True, check_finite=False,lower=True)  
        s,u = np.sqrt(np.maximum(s,0))[::-1], u[:,::-1] 

        try:
            rank = np.where(np.cumprod((np.diff(np.log(s[s!=0]))>-5)|(s[s!=0][1:]>np.median(s))))[0][-1]+2
        except:
            print(np.diff(np.log(s[s!=0])))
            for r in s: print(s)
   
        S = s[:rank]
        U = u[:,:rank]
        
        w = 1./(1.+S**-2)
        prod = np.array(U.T.dot(self.f).T)
        
        gcv = (np.sum((((w-1)*prod))**2))/rank/(1-np.mean(w))**2
        
        from scipy.linalg import solve
        
        Sprior =   np.eye(self.n_points)+self.V*self.Factor(self.V.T.todense())
        s,logdet = np.linalg.slogdet(Sprior)#+np.sum(np.log(Yerr))*2
        try:
            fit = np.array(np.dot(self.f, solve(Sprior,self.f,assume_a='pos').T))
        except:
            fit = np.array(np.dot(self.f, solve(Sprior,self.f ).T))

        logev = -0.5*(logdet+fit+self.n_points*np.log(2*np.pi))
        
     
    

        return -np.log(gcv), logev
        
        #DV = self.Factor.apply_Pt(self.Factor.solve_Lt(self.V.T.todense(),use_LDLt_decomposition=False))
        #U,S,V = np.linalg.svd(DV,full_matrices=False)

        #w = 1./(1.+S**-2)
        
        #DV = self.Factor.apply_Pt(self.Factor.solve_Lt(self.V.T.todense(),use_LDLt_decomposition=False))
        #U,S,V = np.linalg.svd(DV,full_matrices=False)
        #V = solve_banded((1,1),WD,V, overwrite_ab=True,overwrite_b=True,check_finite=False) 


        #Sprior =   np.eye(self.n_points)+self.V*self.Factor(self.V.T.todense())
        ##logdet = np.sum(np.log(np.diag(np.linalg.cholesky(Sprior))))*2
        #s,logdet = np.linalg.slogdet(Sprior)#+np.sum(np.log(Yerr))*2
        #fit = np.array(np.dot(self.f, np.linalg.solve(Sprior,self.f).T))
        
        #logev = -0.5*(logdet+fit+self.n_points*np.log(2*np.pi))
        
        #return logev   
        ##g_noise = self.Factor.apply_Pt(self.Factor.solve_Lt(noise,use_LDLt_decomposition=False))

        
            
    def Evidence2(self,lam,eta):
 
        #linear in n_p*n_t*n_r**2
         
 
        vvtrace = self.VV.diagonal().sum()
        lam = np.exp( 8*(lam-.5)+14)/self.DRDR.diagonal().sum()*vvtrace*lam/(1.001-lam)
        eta = np.exp(20*(eta-.5)+-10)*vvtrace*eta/(1.001-eta)
         
        
        if self.nt_new == 1: eta = 0

        AA = lam*self.DRDR+eta*self.DTDT

        self.Factor.cholesky_inplace(AA,1e-8)
        
        Sprior =   np.eye(self.n_points)+self.V*self.Factor(self.V.T.todense())
        #logdet = np.sum(np.log(np.diag(np.linalg.cholesky(Sprior))))*2
        s,logdet = np.linalg.slogdet(Sprior)#+np.sum(np.log(Yerr))*2
        fit = np.array(np.dot(self.f, np.linalg.solve(Sprior,self.f).T))
        
        logev = -0.5*(logdet+fit+self.n_points*np.log(2*np.pi))
        
        return logev   
        #g_noise = self.Factor.apply_Pt(self.Factor.solve_Lt(noise,use_LDLt_decomposition=False))

        
        
 
        
    def Calculate(self,lam,eta, n_noise_vec = 500):

        if not self.corrected:
            self.PreCalculate()
        if debug:
            print('Calculate')
            TT = time.time()
            
        #ng = 10
        #lambda_vals = np.linspace(0,1,ng+2)[1:-1]
        #eta_vals = np.linspace(0,1,ng+2)[1:-1]
   
        
        #EV = np.zeros((len(lambda_vals),len(eta_vals)))
        #t = time.time()
        #GCV = np.zeros((len(lambda_vals),len(eta_vals)))

        #for i,l in enumerate(lambda_vals):
            #print('%d%%'%(100*i/len(lambda_vals)))
            #for j,ee in enumerate(eta_vals):
                #try:
                    #GCV[i,j], EV[i,j] = self.GCV(l,ee)
                #except:
                    #EV[i,j] = GCV[i,j]= -np.inf
                ##try:
                    ##EV[i,j],GCV[i,j] = self.Evidence(l,ee)
                ##except:
                    ##EV[i,j] =GCV[i,j]= -np.inf
                    
                     
                    
        #print((time.time()-t)/(len(lambda_vals)*len(eta_vals)))
        ##t = time.time()
        ##for i,l in enumerate(lambda_vals):
            ##print('%d%%'%(100*i/len(lambda_vals)))
            ##for j,ee in enumerate(eta_vals):
                ##try:
                    ##GCV[i,j] = self.GCV(l,ee)
                ##except:
                    ##GCV[i,j] = -np.inf
        ##print((time.time()-t)/(len(lambda_vals)*len(eta_vals)))
          
        ##embed()         
                    
        ##EV2 = np.zeros((len(lambda_vals),len(eta_vals)))
        ##print((time.time()-t)/(len(lambda_vals)*len(eta_vals)))
        ##t = time.time()
        ##for i,l in enumerate(lambda_vals):
            ##print('%d%%'%(100*i/len(lambda_vals)))
            ##for j,ee in enumerate(eta_vals):
                ##try:
                    ##EV2[i,j] = self.Evidence2(l,ee)
                ##except:
                    ##EV2[i,j] = -np.inf
        ##print((time.time()-t)/(len(lambda_vals)*len(eta_vals)))
          
        ##embed()
        ##import matplotlib.pylab as plt
        #i,j = np.unravel_index(np.argmax(EV),EV.shape)
        ###take_along_axis
        #lam,eta = lambda_vals[i],eta_vals[j]
        #print(lam,eta)
        #plt.figure()

        #plt.contourf(lambda_vals,eta_vals,EV.T,30 )
        #plt.plot(lam,eta,'go')
        #plt.colorbar()
        #plt.ylabel('radius')
        #plt.xlabel('time')
        ###plt.show()
        #plt.savefig('ev1.png')


        #i,j = np.unravel_index(np.argmax(GCV),GCV.shape)
        #lam,eta = lambda_vals[i],eta_vals[j]
        #print(lam,eta)
        #plt.clf()

        ##plt.figure()
        #plt.contourf(lambda_vals,eta_vals,GCV.T,30 )
        #plt.plot(lam,eta,'rx')
        #plt.colorbar()
        #plt.ylabel('radius')
        #plt.xlabel('time')
        #plt.savefig('gcv.png')
        #plt.clf()

        

        
        #lam,eta = lambda_vals[i],eta_vals[j]
        #print(lam,eta)

        

        #np.random.seed(0)
        #noise vector for estimation of uncertainty
        
        #perfecty uncorrelated noise, most optimistic assumption
        noise = np.random.randn(self.n_points, n_noise_vec)
        #correlated noise, most pesimistic assumption
        noise += np.random.randn(n_noise_vec) 

        #BUG hardcodded!!
        R0 = 1.7 #m  
        a0 = 0.6 #m               
  

        vvtrace = self.VV.diagonal().sum()
        lam = np.exp( 8*(lam-.5)+14)/self.DRDR.diagonal().sum()*vvtrace*lam/(1.001-lam)
        eta = np.exp(20*(eta-.5)+-10)*vvtrace*eta/(1.001-eta)
        
        
        if self.nt_new == 1: eta = 0
        
        
        

        AA = self.VV +lam*self.DRDR+eta*self.DTDT
        #t = time.time()
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
        noise*=noise_scale[:,None]
        g_noise = self.Factor( self.V.T * noise )#SLOW but paraellised 

        #correct approach how to generate samples from the posterior, but it is noisy, but in radial and temporal domain!!
        #n_noise_vec = 200
        #noise = np.random.randn(self.V.shape[1], n_noise_vec)
        #g_noise = self.Factor.apply_Pt(self.Factor.solve_Lt(noise,use_LDLt_decomposition=False))
        ###estimate of systematic errors, assuming that that they are perfectly corrected and of the size of the errorbars
        #g_sys_err = np.squeeze(self.Factor(self.V.T*np.ones_like(self.f)))
        #g_noise += g_sys_err[:,None]* np.random.randn(  n_noise_vec)
        #g_noise*= max(1,np.sqrt(self.chi2))
        
       
        g_noise += (g - g_noise.mean(1))[:,None]
        g_noise = g_noise.reshape(self.nr_new,self.nt_new,n_noise_vec).T
        g_noise = self.invtrans(g_noise)*self.norm
            
        
    
        self.retro_f[self.valid] = self.invtrans((self.M*g))*self.norm
 
        self.g = self.invtrans(g.reshape(self.nr_new,self.nt_new).T)*self.norm

 
        #if too many points was removed, it can happen  that some timepoinst was not measured at all,
        #remove invalid earlier?
      
        valid_times = slice(None,None)
        #valid_times = np.reshape(self.V.sum(0) > 0,(self.nr_new,self.nt_new)).T
        #valid_times = np.any(np.asarray(valid_times),1)

        self.g = self.g[valid_times] 
        g_noise= g_noise[:,valid_times] 
        self.g_samples = np.copy(g_noise)

        #find lower and upper uncertainty level, much faster then using mquantiles..
        rvec = np.linspace(self.r_min,self.r_max,self.nr_new)
        rvec_ = (rvec[1:]+rvec[:-1])/2
        K       = -np.diff(self.g )/(self.g[:,1:]+self.g[:,:-1]      )/(rvec_*np.diff(rvec*a0))*R0*2
        K_noise = -np.diff(g_noise)/(g_noise[...,1:]+g_noise[...,:-1])/(rvec_*np.diff(rvec*a0))*R0*2 #slow
        
        K_noise.sort(0)#slow

        Kerr_u = np.mean(K_noise[n_noise_vec//2:],axis=0) #can have nans!
        Kerr_d = np.mean(K_noise[:n_noise_vec//2],axis=0)
        
        g_noise.sort(0)#slow

        self.g_u = np.mean(g_noise[n_noise_vec//2:],axis=0)
        self.g_d = np.mean(g_noise[:n_noise_vec//2],axis=0)

        self.K = np.c_[K,K[:,-1]]
        self.Kerr_u = np.c_[Kerr_u,Kerr_u[:,-1]] 
        self.Kerr_d = np.c_[Kerr_d,Kerr_d[:,-1]]
        self.fitted=True

        self.g_t = self.t_new[valid_times] 
        self.g_r = self.r_new[valid_times] 
        if debug:
            print('calculate',time.time()-TT)
            
        #embed()
        
        #f,ax = plt.subplots(2,2,sharex=True)
        #ax[0,0].errorbar(self.R,self.Y[self.valid],self.Yerr[self.valid],ls='none')
        #ax[0,0].plot(self.g_r.T,self.g[0],c='k')
        #ax[0,0].set_xlim(0,1.1)
        #ax[0,0].set_ylim(0,5e19)
        
        #ax[1,0].errorbar(self.R,self.Y[self.valid],self.Yerr[self.valid],ls='none')
        #ax[1,0].plot(self.g_r.T,self.g_samples[:,0].T,lw=.1,c='y')
        #ax[1,0].set_xlim(0,1.1)
        #ax[1,0].set_ylim(0,5e19)
        
        #dr = np.gradient(self.g_r[0])
        ##ax[0,0].errorbar(self.R,self.Y[self.valid],self.Yerr[self.valid],ls='none')
        #ax[0,1].plot(self.g_r[0],np.gradient(np.log(self.g[0]))/dr,c='k')
        #ax[0,1].set_xlim(0,1.1)
        ##ax[0,].set_ylim(0,5e19)
        
        ##ax[1,0].errorbar(self.R,self.Y[self.valid],self.Yerr[self.valid],ls='none')
        #ax[1,1].plot(self.g_r[0],np.gradient(np.log(self.g_samples[:,0].T))[0]/dr[:,None],lw=.1,c='y')
        #ax[1,1].set_xlim(0,1.1)
        ##ax[1,1].set_ylim(0,5e19)
        #ax[0,0].set_ylabel('n_e')
        #ax[0,1].set_ylabel('dln(n_e)/dr/n_e')
        #ax[1,0].set_ylabel('n_e')
        #ax[1,1].set_ylabel('dln(n_e)/dr/n_e')
        #for a in ax.flatten(): a.axvline(1,ls='--',c='k',lw=.5)
             
        
        #plt.show()
        
        
        #TODO test autoregularizace pro uela data?
        
        
      
        
        return self.g,self.g_u,self.g_d, self.g_t, self.g_r
        
        
    def finish(self):
        try:
            del self.Factor, self.DTDT,self.DRDR,self.VV, self.V
        except:
            pass
        gc.collect()
      
        















def main():
 
    nr_fit = 201
    #nt = 5
    dt = 0.05
    eta = 0.5
    lam = 0.5
    edge = 0
    pedestal_rho = 1.01
    
    
    invalid = ~np.isfinite(yexp)| ~np.isfinite(yerr)
    yerr[invalid] = -np.inf
    yexp[invalid] = -1
    
    #data transformations
    #  fun, inversed fun, derivative
    transformations = OrderedDict()
    transformations['linear']   = lambda x:x,lambda x:x, lambda x:1
    transformations['log']   = lambda x: np.log(np.maximum(x,0)/.1+1),  lambda x:np.maximum(np.exp(x)-1,1.e-6)*.1,   lambda x:1/(.1+np.maximum(0, x))  #not a real logarithm..
    transformations['sqrt']  = lambda x: np.sqrt(np.maximum(0, x)), np.square,lambda x:.5/np.sqrt(np.maximum(1e-5, x))
    transformations['asinh'] = np.arcsinh, np.sinh, lambda x:1./np.hypot(x,1)
 
 
    MG = map2grid(xexp,tvec,yexp,yerr,nr_new=nr_fit, dt=dt)
    MG.PrepareCalculation( zero_edge=edge,
                          transformation = transformations['log'],
                          even_fun=True,robust_fit=False,
                          pedestal_rho=pedestal_rho)
                           
 
    MG.PreCalculate()
    
    
    MG.Calculate(lam,eta)
    
    
  
    plt.errorbar(xexp.flatten(), yexp.flatten(), yerr.flatten(), fmt='.')
    print(MG.g_samples.shape)
    #plt.plot( MG.r_new[0], MG.g_samples.reshape(-1, nr_fit).T,lw=.2,c='r')
    plt.plot( MG.r_new[0], MG.g.T ,lw=2,c='r')
    plt.xlim(0,1.1)
    plt.ylim(0,None)
    plt.show()




#def main2():

    #np.random.seed(0)

    ##MG = map2grid
    
    #n = 100
    #R = np.random.rand(n)
    #T = np.random.rand(n)
    #Y = (1-R**2)**1.5*(1-T**3/3)
    #Yerr = np.ones_like(Y)/3

    #Y += np.random.randn(n)*Yerr
    
    #log_trans    = lambda x: np.log(np.maximum(x,0)/.1+1),  lambda x:np.maximum(np.exp(x)-1,1.e-6)*.1,   lambda x:1/(.1+np.maximum(0, x))  #not a real logarithm..

    ##radial resolution strongly increases computation time
    #MG = map2grid(R,T,Y,Yerr/10,dt = 0.01,nr_new=100)
    #MG.PrepareCalculation(zero_edge=True, transformation =None)
    #MG.PreCalculate( )
    #MG.Calculate(0.4, 0.6)
    
    #subplot(121)
    #plot( MG.g_r[0],  MG.g[::10].T,'r')
    #plot( MG.g_r[::10].T ,  np.maximum(1-MG.g_r.T**2,0)[:,::10]**1.5*np.maximum(0,1-MG.g_t.T**3/3)[:,::10],'b--')
    #subplot(122)
    #plot( MG.g_t[:,0],  MG.g[:,::10] ,'r')
    #plot( MG.g_t[:,::10] ,  np.maximum(1-MG.g_r**2,0)[:,::10]**1.5*np.maximum(0,1-MG.g_t**3/3)[:,::10],'b--')
    #show()
    
    
    
    #embed()
    
    #exit()
    ##10, 0.01
    ##30, 0.089
    ##100, 0.26
    ##300, 1.36
    ##1000, 8 
    
    
    


    #discontinuties = np.load('discontinuties.npz' )
    #data = np.load('map2grid.npz' )
    #MG = map2grid(data['R'],data['T'],data['Y'],data['Yerr'],data['P'],data['W'],data['nr_new'],data['nt_new'] )
    #MG.PrepareCalculation(   zero_edge=False, core_discontinuties = [],
                           #edge_discontinuties = [],transformation = None,even_fun=True,
                           #robust_fit=False,pedestal_rho=None, elm_phase=None)
                           
    ##print(discontinuties['arr_0'])
    ##print(discontinuties['arr_1'])
    ##print(discontinuties['arr_2'])

    ##MG.PrepareCalculation(  zero_edge=False)
    
    ##MG.PrepareCalculation(  zero_edge=False, edge_discontinuties = discontinuties['arr_1'], elm_phase = discontinuties['arr_2'].T)
    #MG.PreCalculate( )
    
    
    #MG.Calculate(0.1, 0.1)
    
    
    
    #exit()
    
    #import pickle as pkl
    #from matplotlib.pylab import plt

    #with open('data_for_tomas.pkl','rb') as f:
        #data = pkl.load(f)
    #out_ne, out_Te = data
    #rhop_ne, ne, ne_err, rhop_ne_err = out_ne
    #rhop_Te, Te, Te_err, rhop_Te_err = out_ne
    
    #nt = 1
    #nr = 201
    #nd = len(ne)
    
    #MG = map2grid(rhop_Te,np.zeros(nd),Te,Te_err,np.arange(nd),np.ones(nd),nr,nt)
    
    #MG.r_max = 1.04
    
    #transformation =  lambda x: np.sqrt(np.maximum(0, x)), np.square,lambda x:.5/np.sqrt(np.maximum(1e-5, x))
    #transformation =   lambda x: np.log(np.maximum(x,0)/.1+1),  lambda x:np.maximum(np.exp(x)-1,1.e-6)*.1,   lambda x:1/(.1+np.maximum(0, x))  #not a real logarithm..

    #MG.PrepareCalculation(transformation=transformation, zero_edge=False)
    #MG.PreCalculate( )
    
    
    #MG.Calculate(0.1, 0.1)
    
  
    
    
if __name__ == "__main__":
    main()
 

