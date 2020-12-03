#!/usr/bin/env python 
# -*- coding: utf-8 -*-
#print(__doc__)

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
 
from numpy import *
import matplotlib 
import gc
import scipy as sc
##from matplotlib.pyplot import * 
import scipy.sparse as sp
import matplotlib.animation as manimation
import os,sys
import time
from  scipy.stats.mstats import mquantiles
from scipy.ndimage.morphology import binary_erosion,binary_dilation, binary_opening
from collections import OrderedDict

try:
    from sksparse.cholmod import  cholesky, analyze,cholesky_AAt,CholmodError
    chol_inst = True
except:
    chol_inst = False
    print('!CHOLMOD is not installed, slow LU decomposition will be used!')
        #alternative when cholmod is not availible, spsolve is about 5x slower 
        #and with higher memory consumption



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
    
    def __init__(self,R,T,Y,Yerr,nr_new,dt ):
        
        #savez('map2grid',R=R,T=T,Y=Y, Yerr=Yerr, nr_new=nr_new,nt_new=dt)
        
        self.dtype='double'
        self.R = R.ravel()
        self.T = T.ravel()



        self.Y = Y.ravel()
        self.Yerr = ma.array(Yerr.ravel())
        self.Yerr.mask = (Yerr<=0)|~isfinite(Yerr)|self.Yerr.mask 
        
        self.valid = isfinite(Yerr.data).ravel()

        medY = median(abs(Y[self.valid]))
        self.n_points = sum(self.valid)
        self.norm = medY/2
        
   
        self.t_min = amin(self.T[self.valid])
        self.t_max = amax(self.T[self.valid])
        self.r_min = 0.
        self.r_max = 1.1  
    
        self.nr_new = nr_new
        self.nt_new0 = int(ceil(round((self.t_max-self.t_min)/dt,2)))+1
        self.nt_new = self.nt_new0 #actual number of profiles, when the regiosn with missing data are removed
        


        #self.Y   /= self.norm    
        #self.Yerr.data/= self.norm
        
        #print(self.n_points, self.R.shape, self.T[self.valid].shape, self.Y.shape, self.R.shape, self.T.shape, self.Y.shape)


        self.g = zeros((self.nt_new0,nr_new))*nan
        self.g0 = zeros((self.nt_new0,nr_new)) 

        self.g_u = zeros((nr_new,self.nt_new0))*nan
        self.g_d = zeros((nr_new,self.nt_new0))*nan
        
        self.K = zeros((nr_new,self.nt_new0))
        self.Kerr_u = zeros((nr_new,self.nt_new0))*nan
        self.Kerr_d = zeros((nr_new,self.nt_new0))*nan

        self.retro_f = zeros_like(self.Y)*nan
        self.missing_data = ones(self.nt_new0,dtype=bool)

        self.chi2 = nan
        self.lam = nan

        
        self.corrected=False
        self.prepared=False
        self.fitted=False


    def PrepareCalculation(self, zero_edge=False, core_discontinuties = [],edge_discontinuties = [],transformation = None,
                           robust_fit=False,pedestal_rho=None):
        print('\nPrepareCalculation')
        
        #BUG removing large number of points, will change missing_data index!!
 
        self.robust_fit = robust_fit
        
        if transformation is None:
            transformation = (lambda x:x,)*2+(lambda x:1,)
 
        self.trans, self.invtrans, self.deriv_trans = transformation
        
        
        
        #create a geometry matrix 
        dt0 = (self.t_max-self.t_min)/(self.nt_new0-1)
        dr =  (self.r_max-self.r_min)/(self.nr_new-1)
        it =  (self.T[self.valid]-self.t_min)/dt0
        
        #new grid for output
        r_new  = linspace(self.r_min,self.r_max,self.nr_new )
        t_new0 = linspace(self.t_min,self.t_max,self.nt_new0)

        #define contribution matrix 
        it[it<0] = 0
        it[it>self.nt_new0-1] = self.nt_new0-1
        
        ir = (self.R[self.valid]-self.r_min)/dr
        ir[ir<0] = 0
        ir[ir>self.nr_new-1] = self.nr_new-1

        weight  = empty((self.n_points,4))
        index_t = empty((self.n_points,4),dtype=int)
        index_r = empty((self.n_points,4),dtype=int)
        index_p = tile(arange(self.n_points,dtype=int), (4,1)).T

        index_t[:,0] = floor(it)
        index_t[:,1] =  ceil(it)
        index_t[:,2] = index_t[:,0]
        index_t[:,3] = index_t[:,1]
                

        index_r[:,0] = floor(ir)
        index_r[:,1] = index_r[:,0]
        index_r[:,2] =  ceil(ir)
        index_r[:,3] = index_r[:,2]


        weight[:,0] = (floor(it)+1-it)*(floor(ir)+1-ir)
        weight[:,1] = (it-floor(it))*(floor(ir)+1-ir)
        weight[:,2] = (floor(it)+1-it)*(ir-floor(ir))
        weight[:,3] = (it-floor(it))*(ir-floor(ir))
        
        #time regions which are not covered by any measurements
        self.missing_data[index_t[:,0]] = False
        self.missing_data[index_t[:,1]] = False

        #weakly constrained timepoints
        weak_data,_ = histogram(index_t,self.nt_new0,weights=weight,range=(0,self.nt_new0))
        
        self.missing_data[weak_data<mean(weak_data)*.02] = True #almost missing data
                
        weak_data = (weak_data<mean(weak_data)/5.)[~self.missing_data]
        weak_data = weak_data[1:]|weak_data[:-1]

        #correction of dt for regions with a missing or weakly constrained data 
        dt = ones(self.nt_new0)
        dt = ediff1d(cumsum(dt)[~self.missing_data])
        dt = (dt/(1+weak_data))*dt0 
        self.nt_new = sum(~self.missing_data)

        #skipping a fit in regions without the data
        used_times = cumsum(~self.missing_data)-1
        index_t    = used_times[index_t]
        t_new = t_new0[~self.missing_data]
        self.r_new,self.t_new = meshgrid(r_new,t_new)

 
        weight  =  weight.ravel()
        index_p = index_p.ravel()
        index_r = index_r.ravel()
        self.index_t = index_t.ravel()
        index_rt= index_r*self.nt_new+self.index_t
        npix = self.nr_new*self.nt_new

        self.M = sp.csc_matrix((copy(weight),(copy(index_p),copy(index_rt))),
                                shape=(self.n_points,npix),dtype=self.dtype)

        #imshow(self.M.sum(0).reshape(self.nr_new,self.nt_new), interpolation='nearest', aspect='auto');colorbar();show()

        
        #prepare smoothing matrix 
        
        #calculate (1+c)*d/dr(1/r*dF/dr) + (1-c)*d^2F/dt^2
        rvec = linspace(self.r_min,self.r_max,self.nr_new)
        rvec_b = (rvec[1:]+rvec[:-1])/2
        
        #radial weightng function, it will keep zero gradient in core and allow pedestal 
        fun_r2 = (rvec_b*arctan(pi*rvec_b)-log((pi*rvec_b)**2+1)/(2*pi))/rvec_b  #alternative
        self.ifun =  1/fun_r2
        
        
        if pedestal_rho is not None:
            gauss = lambda x, x0, s: exp(-(x-x0)**2/(2*s**2))
            self.ifun/= 1+gauss(rvec_b,pedestal_rho,0.02)*10 +gauss(rvec_b,pedestal_rho+.05,.05)*5
        rweight = r_[self.ifun.max(),self.ifun]
        tweight =  exp(-rvec)

        #==================time domain===============
        #prepare 3 matrices, for core, midradius and edge
        DTDT = []
        
        #discontinuties
        self.time_breaks = OrderedDict()
        self.time_breaks['core']   = (0.,.3), core_discontinuties
        self.time_breaks['middle'] = (.3,.7), []
        self.time_breaks['edge']   = (.7,2.), edge_discontinuties

  
        
        for region, (rho_range, time_breaks) in self.time_breaks.iteritems():
            
            break_ind = []
            if not time_breaks is None and len(time_breaks) != 0:
                break_ind = unique(t_new0.searchsorted(time_breaks))
                break_ind = break_ind[(break_ind < len(t_new0)-2)&(break_ind > 1)] #to close to boundary  
                break_ind = break_ind[ediff1d(break_ind,to_begin=10)>1] #to close discontinuties
      

            
            #remove discontinuties when there are no measurements
            if len(break_ind) > 1:
                break_ind = break_ind[~binary_opening(self.missing_data)[break_ind]]
                break_ind = unique(used_times[break_ind])

            #plot(used_times[unique(t_new0.searchsorted(time_breaks))],'o')
            #plot( t_new.searchsorted(time_breaks),'s')
            #show()

            DT = zeros((3,self.nt_new),dtype=self.dtype)
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
            
            DT = sp.spdiags(DT*dt0**2,(-1,0,1),self.nt_new,self.nt_new)
            
            r_range = (r_new>=rho_range[0])&(r_new<rho_range[1])
            W = sp.diags(tweight[r_range],0)
            #imshow(array(sqrt(abs(DT.todense())))*array(sign(DT.todense())) , interpolation='nearest',   cmap='seismic',vmin = -1, vmax=1)
            #colorbar()
            #[axvline(b,ls='--') for b in break_ind]
            #show()
 
            DTDT.append(sp.kron(W**2,DT.T*DT))#, format='csc'))
            #[axvline(b,ls=':') for b in self.time_breaks['edge'][1]]
        self.DTDT = sp.block_diag(DTDT, format='csc')


    
        #==============radial domain===============
        DR = zeros((3,self.nr_new),dtype=self.dtype)

        
        DR[0, :-2] =  self.ifun[:-1]
        DR[1,1:-1] = -self.ifun[:-1]-self.ifun[1:]
        DR[2,2:  ] =  self.ifun[ 1:]
        if zero_edge and self.trans(0) == 0:  
            #zero gradient at th edge BUG dr? 
            DR[1,-2] += DR[0,-3]
            DR[0,-3] = 0
            #press edge to zero 
            DR[1,-1] = 10
        
            
        DR = sp.spdiags(DR,(-1,0,1),self.nr_new,self.nr_new)
        self.DRDR = sp.kron(DR.T*DR,sp.eye(self.nt_new)**2, format='csc')
        self.prepared = True
            
        
    
    #def CalcG0mtahn(self):
        
        
        #r_new = self.r_new[0]
        #t_new = self.t_new[:,0]
        
        #TT = time.time()
        #outside_region = .7

        #ind = self.valid&(self.R >  outside_region)&~self.Yerr.mask
        #Yerr = self.Yerr.data[ind]
        #Y    = self.Y[ind]
        
        #Yerr*= self.norm
        #Y   *= self.norm
        #T    = self.T[ind]
        #R    = self.R[ind]
        

        #self.g0 = zeros_like(self.r_new)
        #input_it =  []
        #input = []
        #for it,t in enumerate(t_new):
            #t1 =  (t_new[max(0,it-1)]+t)/2
            #t2 =  (t+t_new[min(it+1,len(t_new)-1)])/2

            #ind = (T > t1) & (T < t2) 
            #if sum(ind) < 10: continue
            #x = R[ind]
            #y = Y[ind] 
            #e = Yerr[ind] 
            #input.append((x, y, e, r_new))
            #input_it.append(it)
            

        #from multiprocessing import Process, Pool, cpu_count
        #p = Pool(cpu_count())
        #output = p.map(lmfit_mtanh2_par, input)

        #from scipy.interpolate import interp1d
        #chi2_tot = 0
        #for it, (x, fity,chi2) in zip(input_it, output):
            #self.g0[it] = fity
            #chi2_tot+= chi2
            
        #ind = in1d(arange(len(t_new)), input_it)
        #self.g0[~ind] = interp1d(t_new[ind],self.g0[ind],kind='nearest',fill_value='extrapolate',
                                 #copy=False,axis=0,assume_sorted=True)(t_new[~ind])
        #self.g0/= self.norm
        
    def PreCalculate(self ):
   
  
        
        if not self.prepared:
            print('not yet prepared!')
            self.PrepareCalculation()
        
        print('Precalculate')
        #first pass - decrease the weight of the outliers
        lam = 5  #BUG 

        Y = copy(self.Y[self.valid])/self.norm
        Yerr = copy(self.Yerr.data[self.valid])/self.norm
        Yerr *= self.deriv_trans(Y)
        
        Yerr[self.Yerr.mask[self.valid]] = infty

        #transform and weight by uncertainty
        self.f = self.trans(Y)/Yerr

        self.V = sp.spdiags(1/Yerr,0, self.n_points,self.n_points,format='csr')*self.M
        self.V.eliminate_zeros()  #possible issues with cholesky analyze?
        self.VV = self.V.T*self.V

        
        vvtrace = self.VV.diagonal().sum()
        lam = exp(16)/self.DRDR.diagonal().sum()*vvtrace
        eta = exp(11)/self.DTDT.diagonal().sum()*vvtrace
        AA = self.VV+lam*self.DRDR+eta*self.DTDT
        #print('TRACE %.3e  %.3e'%( self.DRDR.diagonal().sum()/vvtrace,self.DTDT.diagonal().sum()/vvtrace))
        
 

        if chol_inst:
            self.Factor = analyze(AA)
        
        self.corrected=True

        if self.robust_fit:
        
            try:
                if chol_inst:
                    self.Factor.cholesky_inplace(AA)
                else:
                    self.Factor = sp.linalg.factorized(AA)
                    print('umfpack')


            except Exception as e:
                print(e)
                lam = lam+2
                self.Factor = sp.linalg.factorized(self.VV+10**lam*self.DD)
                print('umfpack')

            g=squeeze(self.Factor(self.V.T*self.f))

            #make a more robust fit in one iteration
            dist = self.V*g-self.f  

            
            #Winsorizing M estimator, penalize outliers
            Yerr_corr = ((dist/3)**2+1)**(1)  

            self.f /= Yerr_corr
            self.V = sp.spdiags(1/Yerr_corr,0, self.n_points,self.n_points,format='csr')*self.V
            self.VV = self.V.T*self.V


            
        
    def Calculate(self,lam,eta):

        if not self.corrected:
            self.PreCalculate()
        print('Calculate')


        #noise vector for estimation of uncertainty
        n_noise_vec = 50
        noise = random.randn(self.n_points, n_noise_vec)

        noise += random.randn(n_noise_vec)#correlated noise, most pesimistic assumption

        R0 = 1.7 #m  #BUG hardcodded!!
        a0 = 0.6 #m               
  
  
        
        vvtrace = self.VV.diagonal().sum()
        lam = exp( 8*(lam-.5)+12)/self.DRDR.diagonal().sum()*vvtrace
        eta = exp(15*(eta-.5)+ 3)/self.DTDT.diagonal().sum()*vvtrace
        DD = lam*self.DRDR+eta*self.DTDT
        print('TRACE %.3e  %.3e'%( self.DRDR.diagonal().sum()/vvtrace,self.DTDT.diagonal().sum()/vvtrace))
        
        try:
            self.Factor.cholesky_inplace(self.VV+DD)
        except:
            self.Factor = sp.linalg.factorized(self.VV+DD)

  
        Y = self.Y[self.valid]

        Yerr = self.deriv_trans(Y)*self.Yerr.data[self.valid]
        Yerr[self.Yerr.mask[self.valid]] = infty
        Y = self.trans(Y)

        g = squeeze(self.Factor(self.V.T*self.f))
        self.chi2 = linalg.norm((self.M*g-Y)/Yerr)**2/sum(isfinite(Yerr))
        self.lam = lam
        
        #estimate uncertainty of the fit
        noise_scale = maximum(abs(self.V*g-self.f), 1)
        g_noise = self.Factor((self.V.T)*(noise*noise_scale[:,None]))#SLOW 
        g_noise += (g - g_noise.mean(1))[:,None]
        g_noise = reshape(g_noise,(self.nr_new,self.nt_new,n_noise_vec)).T
        g_noise = self.invtrans(g_noise)*self.norm
    
        print('\nchi2: %.2f reg:%.2f'%(self.chi2,self.lam))
        
        
    
        self.retro_f[self.valid] = self.invtrans((self.M*g))*self.norm
 
        self.g = self.invtrans(reshape(g,(self.nr_new,self.nt_new)).T)*self.norm
 
        #if too many points was removed, it can happen  that some timepoinst was not measured at all
        valid_times = reshape(self.V.sum(0) > 0,(self.nr_new,self.nt_new)).T
        valid_times = any(asarray(valid_times),1)
 

        self.g = self.g[valid_times] 
        g_noise= g_noise[:,valid_times] 
        self.g_samples = copy(g_noise)

           
        #find lower and upper uncertainty level, much faster then using mquantiles..
        rvec = linspace(self.r_min,self.r_max,self.nr_new)
        rvec_ = (rvec[1:]+rvec[:-1])/2

        K       = -diff(self.g )/(self.g[:,1:]+self.g[:,:-1]      )/(rvec_*diff(rvec*a0))*R0*2
        K_noise = -diff(g_noise)/(g_noise[...,1:]+g_noise[...,:-1])/(rvec_*diff(rvec*a0))*R0*2
        
        K_noise.sort(0)

        Kerr_u = nanmean(K_noise[n_noise_vec/2:],axis=0)
        Kerr_d = nanmean(K_noise[:n_noise_vec/2],axis=0)
        
        g_noise.sort(0)

        self.g_u = mean(g_noise[n_noise_vec/2:],axis=0)
        self.g_d = mean(g_noise[:n_noise_vec/2],axis=0)

        self.K = c_[K,K[:,-1]]
        self.Kerr_u = c_[Kerr_u,Kerr_u[:,-1]] 
        self.Kerr_d = c_[Kerr_d,Kerr_d[:,-1]]
        self.fitted=True


        self.g_t = self.t_new[valid_times] 
        self.g_r = self.r_new[valid_times] 


        
        return self.g,self.g_u,self.g_d, self.g_t, self.g_r
        
        
    def PlotFit(self,show_plot,make_movie):
        
        path = os.getenv('HOME')+'/tr_client/profiles/'
        if not os.path.exists(path): os.makedirs(path)
        

        
        fig = figure('retro 2',figsize=(18,10))
        fig.clf()
        T = copy(self.T )

        self.Y    = reshape(self.Y   ,shape(T))
        self.Yerr = reshape(self.Yerr,shape(T))
        self.Y[self.Y == 0] = nan
        ax = fig.add_subplot(111)
        T[where(diff(T)>3)[0]] = nan
        
  
        ax.plot(T,self.retro_f,'r-')   
        ax.plot(T.ravel(),self.Y.ravel(),'-')
        ax.axis('tight')
        
        maxlim = mquantiles(self.retro_f[self.valid],0.99)
        minlim = mquantiles(self.retro_f[self.valid],0.02)
        if minlim == maxlim:  minlim,maxlim = 0,1
        ax.set_ylim(minlim-(maxlim-minlim)*0.1,maxlim+(maxlim-minlim)*0.1)
        
      
        fig = figure('smooth fit2')
        fig.clf()

        ax = fig.add_subplot(111)
        ax.plot(self.t_new,self.K,'k-',lw=0.2) 
        ax.axis('tight')
        fig.show()

      
        if not (show_plot or make_movie):
            return self.g, self.t_new, self.r_new
        
        
        
        
        #plot results
        extent = [self.r_min,self.r_max ,self.t_min,self.t_max]
    
        fig1 = figure('invert diffusion')
        fig1.clf()
        ax_im1 = fig1.add_subplot(111)
        im = ax_im1.imshow(self.K[:,::-1],extent=extent, aspect='auto')
        #ax_im1.axis(extent)
        ax_im1.set_xlabel('$\\rho_{pol}$ [-]')
        ax_im1.set_ylabel('$t$ [s]')
        
        fig1.colorbar(im)
        CS = ax_im1.contour(self.K,[0,],c='w',origin='lower',lw=1,
                extent=extent)
        #fig1.savefig(path+self.name+'_diffusion.png')
        pause(0.01)
    
        fig2 = figure('smooth fit')
        fig2.clf()
        ax_im2 = fig2.add_subplot(111)
        im = ax_im2.imshow(self.g[::-1,:], extent=extent, aspect='auto')
        ax_im2.axis(extent)

        ax_im2.set_xlabel('$\\rho_{pol}$ [-]')
        ax_im2.set_ylabel('$t$ [s]')
        fig2.colorbar(im)

        CS = ax_im2.contour(self.g,[0,],c='w',origin='lower',lw=1,extent=extent)
        #fig2.savefig(path+self.name+'_fit.png')
    
        pause(0.01)
        
        close()
        close()
        close()
        
        if not (show_plot or make_movie):
            return self.g, self.t_new, self.r_new

        fig = figure('retrofit',figsize=(7,8))
        fig.clf()
        #def init():
        ax1 = fig.add_subplot(211)
        replot_plot, = ax1.plot([],[],'+',label='retrofit', animated=not make_movie,zorder=98)
        plotline, caplines, barlinecols = ax1.errorbar(0,0,0,fmt='.',capsize=0, 
                        label='measured data', animated=not make_movie)

        fit_plot, = ax1.plot([],[],'-',linewidth = 0.5, animated=not make_movie,zorder=99)
        maxlim = mquantiles(self.g,0.98)
        minlim = mquantiles(self.g,0.02)
        if minlim == maxlim:  minlim,maxlim = 0,1
        ax1.set_ylim(0,maxlim+(maxlim-minlim)*0.3)

        ax1.set_xlim(0,1.1)
        ax1.legend()
        ax1.xaxis.set_major_formatter( NullFormatter() )
        ax1.set_title('Fitted data: '+'chi2: %.2f reg:%.2f'%(self.chi2,self.lam))
        
        title_template = '%s, time: %.3fs'    # prints running simulation time
        time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
        
        ax2 = fig.add_subplot(212)
        ax2.set_title('Numerical inverted diffusion')
        k_plot,= ax2.plot([],[],'r-', animated=not make_movie)
        g_confidence_plot = ax1.fill_between(self.r_new[0],ones(self.nr_new)*1.5,
                        ones(self.nr_new), alpha=.2, facecolor='b', edgecolor='None',
                        label='68% confidence interval', animated=not make_movie)
        
        K_confidence_plot = ax2.fill_between(self.r_new[0],ones(self.nr_new)*1.5,
                        ones(self.nr_new), alpha=.2, facecolor='b', edgecolor='None',
                        label='68% confidence interval', animated=not make_movie)
        
        
        
        ax2.set_xlim(0,1.1)
        maxlim = mquantiles(self.K,0.98)
        minlim = min(0,mquantiles(self.K,0.02))

        if minlim == maxlim:  minlim,maxlim = 0,1
        ax2.set_ylim(minlim-(maxlim-minlim)*0.1,maxlim+(maxlim-minlim)*0.1)
        ax2.set_xlabel('$\\rho_{pol}$ [-]')
        ax2.axhline(y=0, linestyle='--')
            #return fit_plot, confidence_plot, replot_plot, time_text, k_plot

        #FFMpegWriter = manimation.writers['mencoder']
        #metadata = dict(title=self.name, artist='Matplotlib', comment='profile animation')
        #writer = FFMpegWriter(fps=20, metadata=metadata,codec='mpeg4',bitrate=2000)
        import matplotlib.animation as animation
    

        #n_frames = min(self.nt_new0,800)
        dt_step = (self.t_max-self.t_min)/self.nt_new0
        
        #run_time = time.time()
        #with writer.saving(fig, path+self.name+'_profile.avi', 100):
        #for t in linspace(self.t_min,self.t_max,self.nt_new0):  
        def animate(t):
            ##fps = 1/(time.time()-run_time)
            ##run_time = time.time()

            #if make_movie:
                #sys.stdout.write('\r'+path+self.name+'_profile.wmv <-'+" writting %2.3fs fps:%2.1f"%(t,fps))
            #else:
                #sys.stdout.write('\r'+self.name+"  plotting %2.3fs, fps: %2.1f"%(t,fps))

            sys.stdout.flush()

            ind = (abs(self.T- t) < dt_step/2) 
            
            time_text.set_text('time: %.3fs'%t)
            
            if any(ind):

                x = self.R[ind]
                y = self.Y[ind]
                yerr = self.Yerr[ind]
                ry = self.retro_f[ind]
                replot_plot.set_data(x,ry)

                
                # Replot the data first
                plotline.set_data(x,y)

                # Find the ending points of the errorbars
                error_positions = (x,y-yerr), (x,y+yerr)

                # Update the caplines
                #for j,pos in enumerate(error_positions):
                    #caplines[j].set_data(pos)

                # Update the error bars
                barlinecols[0].set_segments(zip(zip(x,y-yerr), zip(x,y+yerr))) 
                
            ind = argmin(abs(self.t_new[:,0]-t))
            fit_plot.set_data(self.r_new[ind,:].T,self.g[ind,:].T)

            #paths, = confidence_plot.get_paths()
            
            #inv K and errorbars
            k_plot.set_data(self.r_new[ind],self.K[ind])
        
            #u_err = self.K[ind]+self.Kerr_u[ind]
            #b_err = self.K[ind]-self.Kerr_d[ind]
        
            update_fill_between(K_confidence_plot,self.r_new[ind], self.Kerr_d[ind],
                                self.Kerr_u[ind], -infty, infty)
            
            update_fill_between(g_confidence_plot,self.r_new[ind], self.g_d[ind],
                                self.g_u[ind], -infty, infty)

            #paths.vertices[1:self.nr_new,1] = u_err

            #paths.vertices[self.nr_new+1:-1,1] = b_err[::-1]
            #paths.vertices[self.nr_new,1] = b_err[-1]
            #paths.vertices[0,1] = b_err[0]#0.2
            
            #if make_movie:
                #writer.grab_frame() 
            return fit_plot, g_confidence_plot,K_confidence_plot, replot_plot, time_text, k_plot,plotline,  barlinecols[0]


        ani = animation.FuncAnimation(fig, animate, linspace(self.t_min,self.t_max,self.nt_new0),
                             interval=25, blit=True)
        show()
        print('\n')
        close()

        return
        
    def finish(self):
        try:
            del self.Factor, self.DD,self.DT,self.DR,self.VV, self.V
            gc.collect()
        except:
            pass

        




































from lmfit import minimize, Parameters


 
# Define the mtanh2 fit function for use in fitting subroutines like lmfit_mtanh2
def mtanh2(pars, x_, data=None, eps_data=None,debug=False):
    #seterr(under='raise')

    """
    Modified hyperbolic tangent function for use with lmfit

    :param pars: Fit parameters

    :param x_: Independent variable

    :param data: Measured dependent variable to compare to model function

    :param eps_data: Uncertainty in data

    :return: Depends on inputs:
        model is returned if no data are provided,
        difference between model and data is returned if no uncertainties are provided,
        actual deviates are returned if data and eps_data are input.
    """

    # Pull out basic parameters
    ped = pars['ped'].value
    offset = pars['offset'].value
    symm = pars['symm'].value
    width = pars['width'].value

    # z is the argument in exp()
    
    width = exp(width)
    z = (x_ - symm) / width

    # Calculate primary tanh part
    model = 0.5 * (ped - offset)  * (1 - tanh(z)) + offset

    # Avoid overflow
    zz = copy(z)
    zz[zz > 20] = 20
    zz[zz < -20] = -20

    # Add in the polynomial modification part
    zz = z  / (1. + exp(zz*2))
    
    p = pars['p'].value

    poly_extra = p*zz*(2*symm + width*zz)  

 
        
    model += poly_extra 

    # Figure out what to return
    if data is None:
        return model
    else:
        dev = model - data
        if eps_data is None:
            return dev
        else:
            dev/= eps_data
            return dev
 


# Define analytic partial derivatives of the mtanh2 function for use in subroutines such as lmfit_mtanh2
def mtanh_pder(pars, x_, data=None, eps_data=None): 
    """
    Analytic calculation of partial derivatives for mtanh2. Used for error propagation in error = pder.*covar.*pder

    :param pars: LMFIT Parameters instance from the fit output (like fit_out.params)

    :param x_: Independent variable at which partial derivatives should be evaluated.
    
    :param data: Measured dependent variable to compare to model function

    :param eps_data: Uncertainty in data

    :return: An nx by np array containing partial derivatives
    """

    
    npar = sum([ pars[p].vary for p in pars])
    pder = zeros([len(x_), npar])  # ped, symm, with, offset, p1, p2

    ped   = pars['ped'].value
    offset= pars['offset'].value
    symm  = pars['symm'].value
    width = pars['width'].value

    width = exp(width)
    # z is the argument in tanh()
    z = (x_ - symm) / width
        
    thz = tanh(z)

    # For reference, the primary tanh part is:
    # model = (ped-offset)/2.0 * (1 - thz) + offset

        
    dz_ds = -1. / width
    dz_dw = -1. * (x_ - symm) / width
    nped = -0.5 * (ped - offset)
    seterr(under='ignore')
    dy_dz = nped / cosh(z) ** 2  # Trips math errors only sometimes.
    seterr(under='warn')

    pder[:, 0] = 0.5 * (1 - thz)  # dy/dped
    pder[:, 1] = dy_dz * dz_ds  # dy/dsymm
    pder[:, 2] = dy_dz * dz_dw  # dy/dwidth
    if pars['ped'].vary:
        pder[:, 3] = 0.5 * (1 + thz)   # dy/doffset


    # Add in the polynomial modification part
    # Avoid overflow
    
    # Avoid overflow
    zz = copy(z)
    zz[zz > 20] = 20
    zz[zz < -20] = -20

    # Add in the polynomial modification part
    zz = z  / (1. + exp(zz*2))

    p = pars['p'].value

    # For reference:
    #poly_extra = p*zz*(2*symm + width*zz)  
    # model += poly_extra  

    pder[:, -1] = zz*(2*symm + width*zz)    # dy/dp2
 
    z[z == 0] = 1  #avoid zero division
    dzz_dz = zz*(1./z - 1. - thz)  
    
    dpoly_dz = 2*p*(symm + width  * zz) * dzz_dz
    dpoly_ds =  p*zz*2
    dpoly_dw =  p*zz**2


    pder[:, 1] +=  dpoly_dz * dz_ds+dpoly_ds  # Additional contribution to dy/dsymm
    pder[:, 2] +=  dpoly_dz * dz_dw+dpoly_dw  # Additional contribution to dy/dwidth

    if eps_data is None:
        return pder
    

    return pder/eps_data[:,None]


def lmfit_mtanh2(x, y, e, xout,params=None,zero_edge=True):   
    """
    Fit to modified hyperbolic tangent function with lmfit

    :param x: Independent variable (probably psi_N)

    :param y: Dependent variable (probably Te, ne, etc.)

    :param e: Uncertainty in y

    :return: fitx, fity, fite:
        fitx: nice, evenly spaced x grid for displaying the fit
        fity: fit evaluated on fitx
        fite: propagated uncertainty in fity
    """


    # Set up parameter guesses and limits
    
    if params is None:
        params = Parameters()
        ymean = y.mean()

        # The order in which the parameters are added must match the partial derivative function.
        params.add('ped', value=ymean, min=0.0)
        params.add('symm', value=0.95, min=0.8, max=1.05)
        params.add('width', value=-3, min=-4, max=-2)  #exp(width)
        if zero_edge:
            params.add('offset', value=0,vary=False)
        else:
            params.add('offset', value=0.01*ymean, min=0.0, max=.5*ymean)

        params.add('p', value=-0.05,max=0  ) #almost monotonicity

        
    fit_out = minimize(mtanh2, params, args=(x, y, e), method='leastsq', Dfun= mtanh_pder ,xtol=1e-6, maxfev=1000 )  # <<<<<<<<<<< FIT
 
    fity = mtanh2(fit_out.params, xout) 
    #print(fit_out.params['p']/ymean)
    #import IPython
    #IPython.embed()
            
    #retro = mtanh2(fit_out.params, x)  
    #fjac = mtanh_pder(fit_out.params,xout ) 
    #fite = sqrt(sum(dot(fjac,fit_out.covar)*fjac,1))

    return xout, fity,fit_out.chisqr/fit_out.nfree
 #, fite, retro , fit_out.params

def lmfit_mtanh2_par(x, y, e, xout): 
    return  lmfit_mtanh2(x, y, e, xout)





def main():
    
    
    
    data =  load('map2grid.npz')
    #(self,R,T,Y,Yerr,nr_new,nt_new,time_breaks,eta=0,name=''):
    tvec = loadtxt('sawtooths_163303.txt')
#R,T,Y,Yerr,nr_new,nt_new,time_breaks
    #print(tvec)
    transform = lambda x: log(maximum(x,0)/.1+1),  lambda x:(exp(x)-1)*.1,   lambda x:1/(.1+maximum(0, x)) 
    transform = lambda x: x,  lambda x:x,   lambda x:1 


    sawteeth,elms = load('discontinuties.npy')
    TT = time.time()


    ##in
    T = data['T']
    R = data['R']  
    Y = data['Y']
    Yerr = data['Yerr']
    
    ind = isfinite(Yerr)&(R > .5)
    norm = mean(Y)
    #Yerr/= norm
    #Y   /= norm
    Yerr = Yerr[ind]
    Y    = Y[ind]
    T    = T[ind]
    R    = R[ind]
    #print(Y.shape)
    xout = linspace(0,1.2,1000)
 
    
    input = []
    for t in arange(T.min() ,T.max(),0.005):
        ind = (T > t) & (T < t+.005) 
        if sum(ind) < 10: continue
        x = R[ind]
        y = Y[ind] 
        e = Yerr[ind] 
        input.append((x, y, e,xout))
 
    from multiprocessing import Process, Pool, cpu_count
    p = Pool(cpu_count())
    output = p.map(lmfit_mtanh2_par, input)
    out = [o[1] for o in output]
    chi2 = mean([o[2] for o in output])


        #fitx, fity, fite,retro, params = lmfit_mtanh2(x, y, e,xout,params)
        #params = None
        
        #out.append(fity)
        
        #continue

        #title(T)
        #errorbar(x,y,e,fmt='b.')
        #errorbar(x,y-retro,e,fmt='r.')
        #axhline(0)
        #plot(xout, fity)
        #plot()
        #print params
        #show()
        
        
    print(time.time()-TT, chi2)
    #exit()

    #plot(xout,array(out).T);figure()
    #pcolor(out);show()
    #show()
    #plot(x,y,'x')
    #errorbar(fitx, fity, fite)
    #show()
    
    #exit()

            
    MG = map2grid(data['R'],data['T'],data['Y'],data['Yerr'],101,data['nt_new'])
    MG.PrepareCalculation( zero_edge=True,core_discontinuties =sawteeth,edge_discontinuties= elms, robust_fit=True)
    MG.PreCalculate( )
    MG.Calculate(0.5, 0.5)
    MG.PlotFit(True,True)
    #show()
    

    exit()
    n = 100
    m = 60
    SNR = 20
    
    
    xgrid=linspace(0,1.1,m)
    y0=(1.2-xgrid**1.1)**.9
    y0 = y0
    x2d_arr=tile(xgrid,(n,1))
    data_arr=y0+zeros((n,1))
    nt=size(x2d_arr,axis=0)

    t_arr=arange(nt)*1e-2

    opt_d={}
    opt_d['Dt_average']=('0','0')
    opt_d['shot']='26233'
    opt_d['diag']=('CEC','A')
    opt_d['sig']=('ne','te')
    opt_d['rho_lbl']='rho_pol'
    opt_d['tres']=1e-2

    opt_d['Map_exp']='AUGD'
    opt_d['Map_diag']='EQH'
    opt_d['Map_ed']=0
    opt_d['ed']=0
    opt_d['exp']='AUGD'
    

        
    data_d={'tgrid':(t_arr,),'xgrid':(x2d_arr,),'data':(data_arr+random.randn(n,m)/SNR,),\
    'data_err':(ones_like(data_arr)/SNR,),'pre':('X','E'),'tvec_crash':[0.1,0.2]}
    #opt_d['tvec_crash']=[0.1,0.2]
    
    myroot = tk.Tk(className=' Profiles')
    mlp = DataFit(myroot,opt_d,data_d)
    #mlp.quick()
    myroot.mainloop()
    #plt.pause(100)
  
if __name__ == "__main__":
    main()
 

