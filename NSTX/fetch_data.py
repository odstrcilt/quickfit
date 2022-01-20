from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import MDSplus
import numpy as np
from time import time
from scipy.interpolate import interp1d,RectBivariateSpline,NearestNDInterpolator,LinearNDInterpolator,interpn

from collections import OrderedDict
import tkinter.messagebox
from copy import deepcopy 
import xarray
import sys,os
np.seterr(all='raise')
from IPython import embed
import matplotlib.pylab as plt
import warnings

try: 
    #preferably use OMFITncDataset class from OMFIT, data will be stored as CDF files
    from omfit_classes.omfit_data import OMFITncDataset
    Dataset = OMFITncDataset
except:
    #ignore file argument
    def Dataset(file,*args, **kwargs):
        return xarray.Dataset(*args, **kwargs)
   

def print_line(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def printe(message):
    CSI="\x1B["
    reset=CSI+"m"
    red_start = CSI+"31;40m"
    red_end = CSI + "0m" 
    print(red_start,message,red_end)
    
def mds_load(MDSconn,TDI, tree, shot):
    if tree is not None:
        MDSconn.openTree(tree, shot)
    data = []
    for tdi in TDI:
        try:
            data.append(np.atleast_1d(MDSconn.get(tdi).data()))
        except:
            print('Loading failed: '+tdi)
            data.append(np.array([]))
    try:
        if tree is not None:
            MDSconn.closeTree(tree, shot)
    except:
        pass
        
    return data
 
def default_settings(MDSconn, shot):
    #Load revisions of Thompson scattering
    ts_revisions = []
    CHERS_revisions = []
    if MDSconn is not None:
        try: 
            #load all avalible TS revision
            MDSconn.openTree('ACTIVESPEC', shot)
            ts_revisions = MDSconn.get('getnci("MPTS.OUTPUT_DATA.*", "node")').data()
            if not isinstance(ts_revisions[0],str): 
                ts_revisions = [r.decode() for r in ts_revisions]
                
            ts_revisions = [r.strip() for r in ts_revisions]

            CHERS_revisions = MDSconn.get('getnci("CHERS.ANALYSIS.*", "node")').data()
            revision_len = MDSconn.get('getnci("CHERS.ANALYSIS.*:DATEANALYZED", "length")').data()
            #use only if any data are availible
            CHERS_revisions = CHERS_revisions[revision_len > 0]
            if not isinstance(CHERS_revisions[0],str): 
                CHERS_revisions = [r.decode() for r in CHERS_revisions]
            CHERS_revisions = [r.strip() for r in CHERS_revisions]

            MDSconn.closeTree('ACTIVESPEC', shot)
        except Exception as e:
            printe('Error '+str(e))
            embed()
            
 
    
    #build a large dictionary with all settings
    default_settings = OrderedDict()
            #'load_options':{'CER system':{'Analysis':('best', ('best','fit','auto','quick')),
                                      #'Corrections':{'Zeeman Splitting':True, 'Wall reflections':False}} }}
        
    #share position error between all diags
    horiz_error =  {'R shift [cm]':0.0}
    cf_correction = ('Poloidal asymmetry correction',('None',['None','LFS','FSA']))

    default_settings['Ti']={'systems':{'CER system':[]},\
        'load_options':{'CER system':{'Analysis':('CT1', CHERS_revisions),'Position error':horiz_error}},
        }

    default_settings['omega']= {'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':('CT1', CHERS_revisions),'Position error':horiz_error}}}
   
    default_settings['nC6'] = {'systems':{'CER system':[] },
        'load_options':{'CER system':OrderedDict((('Analysis',('CT1', CHERS_revisions)),('Position error',horiz_error), cf_correction))}}
    
    TS_options = OrderedDict((("TS revision",('BEST', ts_revisions)),('Position error',horiz_error) ))
    
    default_settings['Te']= {'systems':{'TS system':(['LFS',True],['HFS',True]) },
    'load_options':{'TS system':TS_options}}
    
    TS_options = OrderedDict((("TS revision",('BEST', ts_revisions)),('Position error',horiz_error),cf_correction))
    
    default_settings['ne']= {'systems':{'TS system':(['LFS',True],['HFS',True])},
    'load_options':{'TS system':TS_options}}
      

    default_settings['Mach']= {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':('CT1', CHERS_revisions)}}}
    
    default_settings['Te/Ti']= {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':('CT1', CHERS_revisions)}}}        
 
    default_settings['Zeff']= {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':OrderedDict((('Analysis',('CT1', CHERS_revisions)),cf_correction))}}        
 
 
 
    return default_settings

class data_loader:
    
    def __init__(self,MDSconn, shot, eqm, rho_coord, raw={}):
        
        self.MDSconn = MDSconn
        self.shot = shot
        self.eqm = eqm
        self.rho_coord = rho_coord
        self.RAW = raw
        
    def eq_mapping(self,diag, dr=0,dz=0):
        #update equilibrium mapping only if necessary

        if 'EQM' in diag and diag['EQM']['id'] == id(self.eqm) and self.eqm.diag == diag['EQM']['ed']\
                    and diag['EQM']['dz'] == np.mean(dz) and diag['EQM']['dr'] == np.mean(dr):
            #skip mapping
            return diag


        for sys in diag['systems']:
            #check if loaded
            if not sys in diag:
                continue
             
            #list of channels
            if isinstance(diag[sys], list):
                R,Z,T,I = [],[],[],[]
                nt = 0
                for ch in diag[sys]:
                    R.append(ch['R'].values)
                    Z.append(ch['Z'].values)
                    T.append(ch['time'].values)
                    I.append(slice(nt, nt+len(T[-1])))
                    nt += len(T[-1])
                #empty diag
                if nt == 0:
                    continue
                
                R,Z,T = np.hstack(R)[:,None], np.hstack(Z)[:,None],np.hstack(T)
                
            else:
                R = diag[sys]['R'].values
                Z = diag[sys]['Z'].values
                T = diag[sys]['time'].values
            
            #do mapping 
            rho = self.eqm.rz2rho(R+dr,Z+dz,T,self.rho_coord)
            try:
                if isinstance(diag[sys], list):
                    for ch,ind in zip(diag[sys],I):
                        ch['rho'].values  = rho[ind,0]
                elif 'rho' in diag[sys]:
                    diag[sys]['rho'].values  =  rho                     
                else:
                    diag[sys]['rho'] = xarray.DataArray(rho, dims=['time','channel'])
                    
            except Exception as e:
                print('Error eq_mapping')
                printe(e)
                embed()
                
        diag['EQM'] = {'id':id(self.eqm),'dr':np.mean(dr), 'dz':np.mean(dz),'ed':self.eqm.diag}

        return diag
            
            
        
    def __call__(self,  quantity=[], options=None,spline_fits=False, tbeg=0, tend=10 ):
   
        if spline_fits:
            return self.load_splines()
        
        
        if quantity == 'elms':
            return self.load_elms(options)
        
            
        if quantity == 'sawteeth':
            return self.load_sawteeth()
        
            
        
        T = time()

        
        options = options[quantity]
        


        
        systems = []
        if quantity in ['Ti', 'omega', 'nC6','Mach','Te/Ti','Zeff']:
            systems.append('CHERS')
         
                
        if  quantity in ['Te', 'ne']:
            for sys, stat in options['systems']['TS system']:
                if stat.get(): systems.append(sys)
        
        data = []
        if quantity in ['Te', 'ne', ]:
            data.append(self.load_ts(tbeg, tend, systems, options['load_options']['TS system']))

        if quantity in ['Ti', 'omega', 'nC6'] and len(systems) > 0:
            data.append(self.load_cer(tbeg,tend, systems,options['load_options']['CER system']))
            
 
        #derived quantities
        if quantity == "Mach":
            cer = self.load_cer(tbeg,tend,['CHERS'],options['load_options']['CER system'])             
            from scipy.constants import e,m_u
            Mach = deepcopy(cer)
            
            omg = cer['CHERS']['omega'].values
            omg_err = cer['CHERS']['omega_err'].values
            ti = np.copy(cer['CHERS']['Ti'].values)
            ti_err = cer['CHERS']['Ti_err'].values
            r = cer['CHERS']['R'].values
            Mach['CHERS'] = cer['CHERS'].drop(['omega','omega_err','Ti','Ti_err'])

            vtor = omg*r
            vtor_err = omg_err*r
            ti[ti<=0] = 1 #avoid zero division
            vtor[vtor==0] = 1 #avoid zero division
            mach = np.sqrt(2*m_u/e*vtor**2/(2*ti))
            mach_err = mach*np.hypot(vtor_err/vtor,ti_err/ti/2.)*np.sign(ti_err)
    
            #deuterium mach number 
            Mach['CHERS']['Mach'] = xarray.DataArray(mach, dims=['time','channel'], attrs={'units':'-','label':'M_D'})
            Mach['CHERS']['Mach_err'] = xarray.DataArray(mach_err, dims=['time','channel'])
            
 
            data.append(Mach)
        

        if quantity == "Te/Ti" :
            TS = self.load_ts(tbeg,tend,['LFS','HFS'] )
            CER = self.load_cer(tbeg,tend, systems ,options['load_options']['CER system'] )

            R_Te,tvec_Te,data_Te,err_Te = [],[],[],[]
            
            for sys in TS['systems']:
                if sys not in TS: continue 
                t = TS[sys]['time'].values
                te = TS[sys]['Te'].values
                e = TS[sys]['Te_err'].values
                r = TS[sys]['R'].values
                r,t = np.meshgrid(r,t)
 
                ind = np.isfinite(e)|(te>0)|(e>0)
                tvec_Te.append(t[ind])
                data_Te.append(te[ind])
                err_Te.append(e[ind])
                R_Te.append(r[ind]) 
 
            R_Te  = np.hstack(R_Te)
            tvec_Te = np.hstack(tvec_Te)
            data_Te = np.hstack(data_Te)
            err_Te  = np.hstack(err_Te)

            interp = LinearNDInterpolator(np.vstack((tvec_Te,R_Te)).T, np.copy(data_Te),fill_value=-100)
            Te_Ti = deepcopy(CER)
            for sys in CER['systems']:
                if sys not in CER: continue
                
                r = CER[sys]['R'].values
                t = CER[sys]['time'].values
                r,t = np.meshgrid(r,t)
                
                interp.values[:] = np.copy(data_Te)[:,None]                 
                Te = np.single(interp(t,r))
                
                interp.values[:] = np.copy(err_Te)[:,None] 
                Te_err = np.single(interp(t,r))
 
                Ti = CER[sys]['Ti'].values
                Ti_err = CER[sys]['Ti_err'].values
                
                TeTi_err = Te/(Ti+1)*np.hypot(Te_err/(Te+1),Ti_err/(Ti+1))
                TeTi_err[Ti_err < 0] = -np.infty
                Te_Ti[sys] = CER[sys].drop(['Ti','Ti_err','omega','omega_err','nC6','nC6_err'])
                Te_Ti[sys]['Te/Ti'] = xarray.DataArray(Te/(Ti+1),dims=['time','channel'], attrs={'units':'-','label':'T_e/T_i'})
                Te_Ti[sys]['Te/Ti_err'] = xarray.DataArray(TeTi_err,dims=['time','channel'])

         
            data.append(Te_Ti)
            
            
 
        if quantity == "Zeff" :
            TS = self.load_ts(tbeg,tend,['LFS','HFS'],options['load_options']['CER system'] )
            CER = self.load_cer(tbeg,tend, systems ,options['load_options']['CER system'] )

            R_ne,tvec_ne,data_ne,err_ne = [],[],[],[]
            for sys in TS['systems']:
                if sys not in TS: continue 
                t = TS[sys]['time'].values
                ne = TS[sys]['ne'].values
                e = TS[sys]['ne_err'].values
                r = TS[sys]['R'].values
                r,t = np.meshgrid(r,t)
                ind = np.isfinite(e)|(ne>0)|(e>0)
                tvec_ne.append(t[ind])
                data_ne.append(ne[ind])
                err_ne.append(e[ind])
                R_ne.append(r[ind]) 
 
            R_ne  = np.hstack(R_ne)
            tvec_ne = np.hstack(tvec_ne)
            data_ne = np.hstack(data_ne)
            err_ne  = np.hstack(err_ne)
            interp = LinearNDInterpolator(np.vstack((tvec_ne,R_ne)).T, np.copy(data_ne),fill_value=-100)

 
            Zeff = deepcopy(CER)
            for sys in CER['systems']:
                if sys not in CER: continue
                
                r = CER[sys]['R'].values
                t = CER[sys]['time'].values
                r,t = np.meshgrid(r,t)
          
                interp.values[:] = np.copy(data_ne)[:,None]                
                ne = np.single(interp(t,r))
                
                interp.values[:] = np.copy(err_ne)[:,None]   
                ne_err = np.single(interp(t,r))
                
                nC = CER[sys]['nC6'].values
                nC_err = CER[sys]['nC6_err'].values
                
                fC = nC/(ne+1)
                
                Zimp, Zmain = 6,1
                valid = (nC_err > 0)&np.isfinite(nC_err)
                zeff = Zimp*(Zimp - Zmain)*fC + Zmain
                zeff_err = np.ones_like(zeff)
                zeff_err[valid]  = (zeff[valid]-Zmain)*np.hypot(ne_err/(ne+1),nC_err/(nC+1))[valid]
                zeff_err[~valid] = -np.infty

                Zeff[sys] = CER[sys].drop(['Ti','Ti_err','omega','omega_err','nC6','nC6_err'])
                Zeff[sys]['Zeff'] = xarray.DataArray(zeff,dims=['time','channel'], attrs={'units':'-','label':'Z_\mathrm{eff}'})
                Zeff[sys]['Zeff_err'] = xarray.DataArray(zeff_err,dims=['time','channel'])
                
                
            try:
                tree = 'PASSIVESPEC' 
                self.MDSconn.openTree(tree, self.shot)
                VB_Zeff = self.MDSconn.get('_x=\\'+tree+'::TOP.VISIBLEBREM:Z_EFFECTIVE').data()
                VB_tvec = self.MDSconn.get('dim_of(_x,0)').data()
                self.MDSconn.closeTree(tree, self.shot)
                ind = (VB_Zeff < 6)&(VB_Zeff > 1)
                VB_Zeff = VB_Zeff[ind]
                VB_tvec = VB_tvec[ind]

                Zeff['VB'] = VB = Dataset('Zeff_VB', attrs={'system':'VB'})
                VB['Zeff'] = xarray.DataArray(VB_Zeff,dims=['time'], attrs={'units':'-','label':'Z_\mathrm{eff}'})
                VB['Zeff_err'] = xarray.DataArray(VB_Zeff*.1,dims=['time'], attrs={'units':'-','label':'Z_\mathrm{eff}'})
                #location is just a guess
                VB['rho']  = xarray.DataArray(np.zeros_like(VB_tvec)+.1,dims=['time'] )
                VB['diags']= xarray.DataArray(np.tile(('VB',), VB_tvec.size),dims=['time'])           
                VB['time'] = xarray.DataArray(VB_tvec,dims=['time'], attrs={'units':'s'})

                Zeff['systems'].append('VB')
                Zeff['diag_names']['VB']=['VB']            
            except Exception as e:
                print('VB Zeff error ',e)
                pass
            
            data.append(Zeff)
            
            
 
     
     
        #list of datasets 
        output = {'data':[],'diag_names':[]}
        times = []
        for d in data:
            if d is None or not 'systems' in d: continue
            for sys in d['systems']:
                if not sys in d: continue
                if isinstance(d[sys], list):
                    for dd in d[sys]:
                        if quantity in dd:
                            output['data'].append(dd)
                elif quantity in d[sys]:
                    output['data'].append(d[sys])
                output['diag_names'].extend(d['diag_names'][sys])

        #cut data in the selected range
        for i in range(len(output['data'])):
            times.append(output['data'][i]['time'].values)
            try:
                output['data'][i]= output['data'][i].sel(time=slice(tbeg,tend))
            except:
                #no data in the requested range 
                continue
        if len(output['diag_names']) == 0 or len(output['data'])==0:
            tkinter.messagebox.showerror('No data loaded',
                    'No data were loaded. Try to change the loading options ' +str(len(output['diag_names']))+' '+str(len(output['data'])))
            return 
 
        
        output['tres']=np.median(np.round(np.diff(np.hstack(times))*1e3,1))/1e3
        output['tres']=round(output['tres'],6)
        if output['tres'] == 0:    output['tres'] = 0.01
        output['rho_lbl'] = self.rho_coord
        
        return output





    def load_splines(self):
        
        
        
        
        if 'SPLINES' in self.RAW:            
            return self.eq_mapping(self.RAW['SPLINES'])

        
        TT = time()
        print_line('  * Fetching SPLINES ... ')
                    
        tree = 'ACTIVESPEC'
        chers_edition = 'CT1'
        chers_signals = ['ZEFFS', 'VTS','TIS', 'NCS', 'RS', 'TIME']
        TDI = [f'\\{tree}::TOP.CHERS.ANALYSIS.{chers_edition}:{sig}' for sig in chers_signals]
        mdts_signals = ['SPLINE_NE','SPLINE_TE', 'SPLINE_RADII','TS_TIMES']
        TDI += [f'\\{tree}::TOP.MPTS.OUTPUT_DATA.BEST:'+sig for sig in mdts_signals]

        Zeff,Vtor,Ti, nC, R,T, TS_ne,TS_Te,TS_R,TS_T = mds_load(self.MDSconn, TDI, tree, self.shot)
        print('\t done in %.1fs'%(time()-TT))

        #use SI units!!
        Vtor *= 1e3 #m/s
        Ti *= 1e3 #eV
        TS_Te *= 1e3 #eV
        TS_ne *= 1e6 #m^3
        nC *= 1e6 #m^3
        
        R /= 100 #m
        TS_R /= 100 #m
        
    
        invalid = Ti <= 0
        
        Vtor[invalid] = np.nan
        Ti[invalid] = np.nan
        nC[invalid] = np.nan
        Zeff[invalid] = np.nan
        Ti= np.maximum(Ti,10)

        ##hydrogen Mach number        
        from scipy.constants import e,m_u
        mach = np.sqrt(2*m_u/e*Vtor**2/(2*Ti))
        
        #truncate maximum mach number
        mach = np.minimum(mach,1)


        #Map Te on CHERS radial and temporal base
        Te_Ti = RectBivariateSpline(TS_R, TS_T, TS_Te,kx=1,ky=1)(R,T).T.astype('single')/Ti

        
        self.RAW['SPLINES'] = splines = {}
        
        splines['Te'] = splines['ne'] = ds = Dataset('spline_TS')
        ds['ne'] = xarray.DataArray(TS_ne.T, dims=['time','R'])
        ds['Te'] = xarray.DataArray(TS_Te.T, dims=['time','R'])
        ds['Z']  = xarray.DataArray(TS_R*0, dims=['R'])
        ds['R']  = xarray.DataArray(TS_R, dims=['R'])
        ds['time'] = xarray.DataArray(TS_T, dims=['time'])
 
        ds = Dataset('spline_CHERS') 
        splines['Zeff'] = splines['Mach'] = splines['Ti'] = splines['nC6']  = splines['Te/Ti'] = splines['omega'] = ds
        ds['Ti'] = xarray.DataArray(Ti, dims=['time','R'])
        ds['Zeff'] = xarray.DataArray(Zeff, dims=['time','R'])
        ds['nC6'] = xarray.DataArray(nC, dims=['time','R'])
        ds['omega'] = xarray.DataArray(Vtor/R, dims=['time','R'])
        ds['Mach'] = xarray.DataArray(mach, dims=['time','R'])
        ds['Te/Ti'] = xarray.DataArray(Te_Ti, dims=['time','R'])
        ds['Z']  = xarray.DataArray(R*0, dims=['R'])
        ds['R']  = xarray.DataArray(R, dims=['R'])
        ds['time'] = xarray.DataArray(T, dims=['time'])
        
        print('\t done in %.1fs'%(time()-TT))
        
        splines['systems'] = ['Te','Ti','Zeff','ne','nC6','omega','Te/Ti','Mach']
        
        
        self.eq_mapping(splines)
        
        
        splines['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}


        
        return splines
    
    
    
           

    def load_asymmetry(self,chers_edition='CT1',dR=0):
        
        
        self.RAW.setdefault('Asymmetry',{})
        asym = self.RAW['Asymmetry']

        A = 2 #Main ion mass
        Z = 1 #Main ion charge
        Zc = 6  #C charge
        Ac = 12 #C mass
    
        #calculate asymmetry correction factors on CHERS timebase as function of R?
        splines = self.load_splines()
            
        EQM = {'id':id(self.eqm),'ed':self.eqm.diag}
         
        if 'EQM' not in asym or EQM != asym['EQM']:
            asym['EQM'] = EQM
            
            rho_grid = np.linspace(0,1,102)[1:]
            theta_grid = np.linspace(0,np.pi*2,50,endpoint=False)
            #get flux surfaces 
            T = splines['Mach']['time'].values
            R,Z = self.eqm.rhoTheta2rz(rho_grid,theta_grid,T,coord_in=self.rho_coord, n_line=101)
            
            # calculate elemental dV for each R,Z
            dRdZ = np.array((np.gradient(Z,axis=[1,2]), 
                             np.gradient(R,axis=[1,2]))).T
            
            dV = 2*np.pi*R*np.linalg.det(dRdZ).T
           
            asym['FSA'] = FSA = Dataset('FSA')
            FSA['dV'] = xarray.DataArray(dV,dims=['time','theta','rho'], attrs={'units':'m^3'})
            FSA['R']  = xarray.DataArray(R,dims=['time','theta','rho'], attrs={'units':'m'})
            FSA['Z']  = xarray.DataArray(Z,dims=['time','theta','rho'], attrs={'units':'m'})
            FSA['Rlfs'] = xarray.DataArray(R[:,0],dims=['time','rho'], attrs={'units':'m'})
            FSA['rho'] = xarray.DataArray(rho_grid,dims=['rho'], attrs={'units':'-'})
            FSA['theta'] = xarray.DataArray(theta_grid,dims=['theta'], attrs={'units':'rad'})
            FSA['time'] = xarray.DataArray(T,dims=['time'], attrs={'units':'s'})
            
        
            
        #load from cache 
        FSA = asym['FSA']
        sepR = np.hstack([sr[sr > 0] for sr in self.eqm.separatrixR]) 
        Rgrid = np.linspace(sepR.min(),sepR.max(),100)

    
        mach = splines['Mach']['Mach'].values/np.sqrt(2) #hydrogen Mach number 
        Te_Ti = splines['Te/Ti']['Te/Ti'].values
        Zeff = splines['Zeff']['Zeff'].values
        spline_R = splines['Mach']['R'].values
        spline_T = splines['Mach']['time'].values
        
 
        
        Aeff = Zeff*A  #effective ion mass estimated from zeff, valid for D+C
        asym_factor_e = 1/(1+Zeff*Te_Ti)* Aeff * mach**2
        asym_factor_c = (1-Zc/Ac* Aeff*Te_Ti/(1+Zeff*Te_Ti))*Ac*mach**2

 
        valid = np.isfinite(asym_factor_c)

        rho_grid   = self.eqm.rz2rho(Rgrid,Rgrid*0,spline_T,self.rho_coord)
        spline_rho = self.eqm.rz2rho(spline_R,spline_R*0,spline_T,self.rho_coord)

        
        ne0_ne = np.zeros((len(spline_T), len(Rgrid)),dtype='single')
        nc0_nc = np.zeros((len(spline_T), len(Rgrid)),dtype='single')

        nefsa_ne = np.zeros((len(spline_T), len(Rgrid)),dtype='single')
        ncfsa_nc = np.zeros((len(spline_T), len(Rgrid)),dtype='single')

        for it in range(len(spline_T)):
            #find index closest to magnetic axis
            imin = np.argmin(rho_grid[it])
            
            #use only LFS CHERS data, HFS are often too poor
            valid[it] &= spline_R > Rgrid[imin]
                        
            #LFS R for each radial location 
            Rlfs_grid = np.interp(rho_grid[it], rho_grid[it,imin:], Rgrid[imin:]+dR)
                        

            #interpolate asymmetry factors on the LFS radial grid
            asym_factor_e_ = np.interp(Rlfs_grid, spline_R[valid[it]], asym_factor_e[it][valid[it]],right=0)
            asym_factor_c_ = np.interp(Rlfs_grid, spline_R[valid[it]], asym_factor_c[it][valid[it]],right=0)

            dR2 = (Rgrid/Rlfs_grid)**2-1
            
            #ratio between LFS and local density
            ne0_ne[it] = np.exp(-asym_factor_e_*dR2)
            nc0_nc[it] = np.exp(-asym_factor_c_*dR2)
            
            
                        
            i_eq = np.argmin(np.abs(spline_T[it]-FSA['time'].values))
            #interpolate asymmetry factors on the LFS radial grid
            R = FSA['R'].values[i_eq]
            Rlfs = FSA['Rlfs'].values[i_eq]
            rho = FSA['rho'].values

            asym_factor_e_ = np.interp(rho, spline_rho[it,valid[it]], asym_factor_e[it,valid[it]])
            #plt.plot( spline_rho[it,valid[it]], asym_factor_e[it,valid[it]] )
            asym_factor_c_ = np.interp(rho, spline_rho[it,valid[it]], asym_factor_c[it,valid[it]])
            
            dR2 = (R/Rlfs)**2-1

            neR_ne0 = np.exp(asym_factor_e_[None]*dR2)
            ncR_nc0 = np.exp(asym_factor_c_[None]*dR2)
            
            #calculate flux surface average
            dV = FSA['dV'].values[i_eq]
            
            
            #ratio between FSA and LFS density
            nefsa_ne0 = np.average(neR_ne0,0, dV)
            ncfsa_nc0 = np.average(ncR_nc0,0, dV) 
            
            #ratio between FSa and local density
            nefsa_ne[it] = np.interp(Rlfs_grid, Rlfs, nefsa_ne0)*ne0_ne[it]
            ncfsa_nc[it] = np.interp(Rlfs_grid, Rlfs, ncfsa_nc0)*nc0_nc[it]
  
        asym['correction'] = corr = Dataset('asym_correction')
        corr['ne0_ne'] = xarray.DataArray(ne0_ne,dims=['time','Rgrid'], attrs={'units':'-'})
        corr['nc0_nc'] = xarray.DataArray(nc0_nc,dims=['time','Rgrid'], attrs={'units':'-'})
        corr['nefsa_ne'] = xarray.DataArray(nefsa_ne,dims=['time','Rgrid'], attrs={'units':'-'})
        corr['ncfsa_nc'] = xarray.DataArray(ncfsa_nc,dims=['time','Rgrid'], attrs={'units':'-'})
        corr['time'] = xarray.DataArray(spline_T,dims=['time'], attrs={'units':'s'})
        corr['Rgrid'] = xarray.DataArray(Rgrid,dims=['Rgrid'], attrs={'units':'m'})
        
        return corr
         

    def load_cer(self,tbeg,tend, systems, options=None):
        #load Ti and omega at once
        TT = time()
        rshift = 0
        cf_correction = 'None'
        edition = 'CT1'

        tree = 'ACTIVESPEC'
        if options is not None:
            selected,editions = options['Analysis']
            edition = selected.get()
            if 'Position error' in options:
                rshift = options['Position error']['R shift [cm]'].get()/100. #[m] 
            if 'Poloidal asymmetry correction' in options:
                cf_correction = options['Poloidal asymmetry correction'][0].get()

 
        #TODO check if data exists at all!
        if edition is None:
            raise Exception('No CHERS analysis data')
        
        #Ti below 300eV is unreliable??
        
        self.RAW.setdefault('CHERS',{})
        cer = self.RAW['CHERS'].setdefault(edition,{})

        #load from catch if possible
        cer.setdefault('diag_names',{})
        cer['systems'] = systems
        
        #load only new systems
        load_systems = list(set(systems)-set(cer.keys()))
 
        #update equilibrium for already loaded systems
        cer = self.eq_mapping(cer, dr=rshift)
 
  
        if len(load_systems) == 0 and np.all([cer[sys].attrs['cf_correction'] == cf_correction for sys in cer['systems']]):
            return cer
        
   
        print_line( '  * Fetching CHERS '+edition.upper()+' data ...' )
        
        #AW, AWB active and background aplitude, unknown units 
        data = {
            'Ti': {'label':'Ti','unit':'eV','sig':['ZTI','DTI'],'scale':1e3},
            'omega': {'label':r'\omega_\varphi','unit':'rad/s','sig':['VT','DVT'],'scale':1e3},
            'nimp': {'label':r'n_c','unit':'m^{-3}','sig':['NC','DNC'],'scale':1e6},
            }
       
        #list of MDS+ signals for each channel
        signals = data['Ti']['sig']+data['omega']['sig']+data['nimp']['sig']+['RADIUS', 'TIME', 'VALID']

     

        TDI = []
        #prepare list of loaded signals
        for sig in signals:
            TDI.append( '\\'+tree+'::TOP.CHERS.ANALYSIS.'+edition+':'+sig)

        Ti,Tierr,Vtor,Vtorerr,Nc,Nc_err,R,tvec,valid = mds_load(self.MDSconn,TDI, tree, self.shot) 
        
        if len(tvec) == 0:
            raise Exception('No CHERS analysis data')

        
        Z = R*0
        R /= 100 #[m]

        valid = np.bool_(valid)
        
        #map to radial coordinate 
        rho = self.eqm.rz2rho(R+rshift,Z,tvec,self.rho_coord)
        
        
        #show these points but ignore them in the fit
        valid &=  np.isfinite(Nc > 0)&np.isfinite(Nc_err)
        valid[valid] &= (Ti[valid] > 0)&(Tierr[valid] > 0)
        valid[valid] &=  (Nc[valid] > 0)&(Nc_err[valid] > 0)

        #valid = np.bool_(valid)
        Tierr[~valid]  = -np.inf
        Vtorerr[~valid]  = -np.inf
        channel = np.arange(len(R))
        
        omega = Vtor/R
        omega_err = Vtorerr/R
        
        #guess of nC errorbars from time scatter
        Nc_err[:] = np.std(np.gradient(Nc)[0],0)[None]
        Nc_err[~valid]  = -np.inf
        
        
        if cf_correction != 'None':
            corr = self.load_asymmetry(chers_edition=edition,dR=rshift)
            if cf_correction == 'LFS':
                nratio = corr['nc0_nc'].values 
            elif cf_correction == 'FSA':
                nratio = corr['ncfsa_nc'].values 
            else:
                raise Exception('Asymetry correction '+cf_correction+' is not supported')

            corr = RectBivariateSpline(corr['time'].values,corr['Rgrid'].values, nratio)(tvec,R)

            Nc[valid] *= corr[valid]
            Nc_err[valid] *= corr[valid]
            
 

        for sys in systems:
            cer['diag_names'][sys]=['CHERS']            
            cer[sys] = Dataset('CHERS_'+sys,attrs={ 'system':sys, 'cf_correction': cf_correction})
            cer[sys]['Ti'] = xarray.DataArray(Ti*data['Ti']['scale'],dims=['time','channel'], attrs={'units':'eV','label':'T_i'})
            cer[sys]['Ti_err'] = xarray.DataArray(Tierr*data['Ti']['scale'],dims=['time','channel'] )
            cer[sys]['omega'] = xarray.DataArray(omega*data['omega']['scale'],dims=['time','channel'], attrs={'units':'rad/s','label':r'\omega_\phi'})
            cer[sys]['omega_err'] = xarray.DataArray(omega_err*data['omega']['scale'],dims=['time','channel'], attrs={'units':'rad/s'})
            cer[sys]['nC6'] = xarray.DataArray(Nc*data['nimp']['scale'],dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_{C^{6+}}'})
            cer[sys]['nC6_err'] = xarray.DataArray(Nc_err*data['nimp']['scale'],dims=['time','channel'], attrs={'units':'rad/s'})
            
            cer[sys]['diags']= xarray.DataArray( np.tile(('CHERS',), Ti.shape),dims=['time','channel'])            

            cer[sys]['R'] = xarray.DataArray(R, dims=['channel'], attrs={'units':'m'})
            cer[sys]['Z'] = xarray.DataArray(Z, dims=['channel'], attrs={'units':'m'})
            cer[sys]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
            cer[sys]['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
            cer[sys]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})
            

        
        cer['EQM'] = {'id':id(self.eqm),'dr':rshift, 'dz':0,'ed':self.eqm.diag}
        print('\t done in %.1fs'%(time()-TT))
        
        return cer
    
    
    
    
    
    
    def load_ts(self, tbeg,tend,systems, options=None):
    
        T = time()

        revision = 'BEST'
        rshift = 0
        cf_correction = 'LFS'
        if options is not None:
            if 'TS revision' in options:
                selected,revisions = options['TS revision']
                revision = selected.get()
            if 'Position error' in options:
                rshift = options['Position error']['R shift [cm]'].get()/100. #[m] 
            if 'Poloidal asymmetry correction' in options:
                cf_correction = options['Poloidal asymmetry correction'][0].get()
 

        #use cached data
        self.RAW.setdefault('TS',{})
        ts = self.RAW['TS'].setdefault(revision,{'systems':systems})

        ts['systems'] = list(systems)
        systems = list(set(systems)-set(ts.keys()))

  
        
        
        #update mapping of the catched data
        ts = self.eq_mapping(ts, dr =rshift)            
        ts.setdefault('diag_names',{})
 
        if len(systems) == 0 and np.all([ts[sys].attrs['cf_correction'] == cf_correction for sys in ts['systems']]):
            #assume that equilibrium could be changed
            return ts
        else:
            systems = ts['systems']

        print_line( '  * Fetching TS data ...')

            
        signals = 'FIT_NE', 'FIT_NE_ERR', 'FIT_TE', 'FIT_TE_ERR','TS_TIMES','FIT_RADII'
        
        
    
        tree = 'ACTIVESPEC'
        #prepare list of loaded signals
        tdi = '\\%s::TOP.MPTS.OUTPUT_DATA.%s:'%(tree,revision)
        TDI = [tdi+sig for sig in signals]

        ne,ne_err,Te,Te_err,tvec,R = mds_load(self.MDSconn, TDI, tree, self.shot)
        R /= 100 #m
        Z = R*0
        Te *= 1e3 #eV
        Te_err *= 1e3 #eV
        ne *= 1e6 #m^-3
        ne_err *= 1e6 #m^-3
        
 
        #these points will be ignored and not plotted (negative errobars )
        invalid = (Te_err<=0) | (Te <=0 ) | (ne_err<=0) | (ne <=0 )
        Te_err[invalid]  = -np.infty
        ne_err[invalid]  = -np.infty
        
        channel = np.arange(len(R))

        rho = self.eqm.rz2rho(R+rshift,Z,tvec,self.rho_coord)
        
                  
        if cf_correction !=  'None':
            corr = self.load_asymmetry(dR=rshift)
            if cf_correction == 'LFS':
                nratio = corr['ne0_ne'].values 
            if cf_correction == 'FSA':
                nratio = corr['nefsa_ne'].values 

            corr = RectBivariateSpline(corr['time'].values,corr['Rgrid'].values, nratio)(tvec,R).T

            ne[~invalid] *= corr[~invalid]
            ne_err[~invalid] *= corr[~invalid]
            
   
      
        imin = np.argmin(rho.mean(0))
        index = {'HFS':slice(0,imin), 'LFS':slice(imin,None)}
        for sys in systems:
            ind = index[sys]
            ts['diag_names'][sys]=['TS:'+sys]

            ts[sys] = Dataset( 'TS:'+sys ,attrs={'system':sys, 'cf_correction': cf_correction})
            ts[sys]['ne'] = xarray.DataArray(ne[ind].T,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
            ts[sys]['ne_err'] = xarray.DataArray(ne_err[ind].T,dims=['time','channel'], attrs={'units':'m^{-3}'})
            ts[sys]['Te'] = xarray.DataArray(Te[ind].T,dims=['time','channel'], attrs={'units':'eV','label':'T_e'})
            ts[sys]['Te_err'] = xarray.DataArray(Te_err[ind].T,dims=['time','channel'], attrs={'units':'eV'})
            ts[sys]['diags']= xarray.DataArray( np.tile(('TS:'+sys,), ne[ind].T.shape),dims=['time','channel'])            
            ts[sys]['R'] = xarray.DataArray(R[ind], dims=['channel'], attrs={'units':'m'})
            ts[sys]['Z'] = xarray.DataArray(Z[ind],dims=['channel'], attrs={'units':'m'})
            ts[sys]['rho'] = xarray.DataArray(rho[:,ind],dims=['time','channel'], attrs={'units':'-'})
            ts[sys]['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
            ts[sys]['channel'] = xarray.DataArray(channel[ind],dims=['channel'], attrs={'units':'-'})
            
 
        print('\t done in %.1fs'%(time()-T))
        ts['EQM'] = {'id':id(self.eqm),'dr':rshift, 'dz':0, 'ed':self.eqm.diag}

        return ts 
        
        
        
    
    def load_elms(self,option):
        node = option['elm_signal'].get()
        elm_time, elm_val, elm_beg, elm_end = [],[],[],[]
        self.RAW['ELMS'] = {}
        self.RAW['ELMS'][node] =  {'tvec': elm_time, 'data':elm_val, 
                     'elm_beg':elm_beg,'elm_end':elm_end,'signal':node}
        return self.RAW['ELMS'][node]
 
    
    def load_sawteeth(self):
        return {'tvec':[]}
    
    
    
    
    
    

 
 
 
 
#replace a GUI call
#np
def main():
    #mdsserver = 'localhost'
    #MDSconn = MDSplus.Connection(mdsserver)
    #MDSconn.openTree('ACTIVESPEC', 141716)
        
    #mdsserver = 'skylark.pppl.gov:8501'
    #import MDSplus
    #try:
        #MDSconn = MDSplus.Connection(mdsserver)
    #except:
    mdsserver = 'localhost'
    MDSconn = MDSplus.Connection(mdsserver)
    TT = time()
    shot = 115559
    #shot = 204179
    shot = 141040
    shot = 141040
    rho_coord = 'rho_tor'
   
 
    
    print(shot)
    print_line( '  * Fetching EFIT01 data ...')
    from map_equ import equ_map

    eqm = equ_map(MDSconn)
    eqm.Open(shot, 'EFIT01', exp='NSTXU')

    #load EFIT data from MDS+ 
    T = time()
    eqm._read_pfm()
    eqm.read_ssq()
    eqm._read_scalars()
    eqm._read_profiles()
    print('\t done in %.1f'%(time()-T))
 

    loader = data_loader(MDSconn, shot, eqm, rho_coord)

    import tkinter as tk
    myroot = tk.Tk(className=' Profiles')

    I = lambda x: tk.IntVar(value=x)
    S = lambda x: tk.StringVar(value=x)
    D = lambda x: tk.DoubleVar(value=x)
 
    
    ts_revisions = []
    CHERS_revisions = []
    
    default_settings = OrderedDict()

    default_settings['Ti']={'systems':{'CER system':([], )},\
        'load_options':{'CER system':{'Analysis':(S('CT2'), CHERS_revisions),
        'Corrections':{'Zeeman Splitting':I(1) }}}}

    default_settings['omega']= {'systems':{'CER system':([], )},
        'load_options':{'CER system':{'Analysis':(S('CT1'), CHERS_revisions)}}}
   
    default_settings['nC6'] = {'systems':{'CER system':([],) },
        'load_options':{'CER system':{'Analysis':(S('CT1'), CHERS_revisions)}}}
    
    default_settings['Te']= {'systems':{'TS system':(['LFS',I(1)],['HFS',I(1)]) },
    'load_options':{'TS system':{"TS revision":(S('BEST'), ts_revisions)}}}

    default_settings['ne']= {'systems':{'TS system':(['LFS',I(1)],['HFS',I(1)])},
    'load_options':{'TS system':{"TS revision":(S('BEST'), ts_revisions)}}}
      
    default_settings['Mach']= {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':(S('CT1'), CHERS_revisions)}}}
    
    default_settings['Te/Ti']= {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':(S('CT1'), CHERS_revisions)}}}
     
    loader.load_splines()
    try:
        data = loader( 'ne', default_settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])
    finally:
        MDSconn.disconnect()
    print('\t done in %.1f'%(time()-T))
     
    print('\n\t\t\t Total time %.1fs'%(time()-TT))

  
if __name__ == "__main__":
    main()
 




