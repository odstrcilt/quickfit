from __future__ import print_function

import MDSplus
import numpy as np
from time import time
from scipy.interpolate import interp1d,NearestNDInterpolator,LinearNDInterpolator

from collections import OrderedDict
import matplotlib
import tkinter.messagebox
from copy import deepcopy 
from multiprocessing.pool import ThreadPool, Pool
import xarray
import re,sys
np.seterr(all='raise')
from IPython import embed

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
 
    MDSconn.openTree(tree, shot)
    data = []
    for tdi in TDI:
        try:
            data.append(np.atleast_1d(MDSconn.get(tdi).data()))
        except:
            data.append(np.array([]))
    MDSconn.closeTree(tree, shot)

    return data


def default_settings(MDSconn, shot):
    '''Build a large dictionary with all settings.
    Note that input arguments are both actually dummies at the moment, but they are needed
    to ensure an identical call structure to the equivalent functions for DIII-D and AUG.
    '''
    kin_profs = 'Te', 'ne', 'Ti', 'vtor'
    default_settings = OrderedDict()


    default_settings['Te'] =  {\
        'systems':OrderedDict((( 'TS system',(['core',True], ['edge',True])),
                                ('ECE system',(['slow',False],['fast',False])))),
        'load_options':{'TS system':{},
                        'ECE system':OrderedDict((("shift",{'dR [cm] =': 0.0}),))}}



    default_settings['ne'] =  {\
            'systems':OrderedDict((( 'TS system',(['core',True], ['edge',True])),
                                    ( 'Reflectometer',(['all bands',False],)),
                                    ( 'TCI interf.',(['fit TCI data',False],['rescale TS',False])) ) ),
                'load_options':{'TS system':{},
                                'Reflectometer':{'Position error':{'R shift [cm]':0.0}}}}

    default_settings['Ti'] =  {\
            'systems':OrderedDict((( 'HIREX system',(['H-like',True], ['He-like',True])),
                                    ( 'TS system'  ,(['edge',True],)) ) ),
                'load_options':{'HIREX system':{'Edition':{'THT':0},
                                               'Correction':    {'Shift wrst. TS':True}}}}


    default_settings['vtor'] =  {\
            'systems':{ 'HIREX system':(['H-like',True], ['He-like',True])},
                'load_options':{'HIREX system':{'Edition':{'THT':0}}}}

    return default_settings
 
  

class data_loader:
    
    def __init__(self,MDSconn, shot, eqm, rho_coord, raw={}):
        
        self.MDSconn = MDSconn
        self.shot = shot
        self.eqm = eqm
        self.rho_coord = rho_coord
        self.RAW = raw
        
    def eq_mapping(self,diag, dr=0,dz=0):
        #update equilibrium mapping 

        if 'EQM' in diag and diag['EQM']['id'] == id(self.eqm) and self.eqm.diag == diag['EQM']['ed']\
                    and diag['EQM']['dz'] == dz and diag['EQM']['dr'] == dr:
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
                R = diag[sys]['R'].values.ravel()
                Z = diag[sys]['Z'].values.ravel()
                T = diag[sys]['time'].values
            
            #do mapping 
            rho = self.eqm.rz2rho(R+dr,Z+dz,T,self.rho_coord)
            
            
            
            if isinstance(diag[sys], list):
                for ch,ind in zip(diag[sys],I):
                    ch['rho'].values  = rho[ind,0]
            else:
                rho = rho.reshape(T.shape+diag[sys]['R'].shape)
                diag[sys]['rho'].values  =  rho 
        
        diag['EQM'] = {'id':id(self.eqm),'dr':dr, 'dz':dz,'ed':self.eqm.diag}
                


        return diag
            
            
        
    def __call__(self,  quantity=[], options=None,zipfit=False, tbeg=0, tend=10 ):
   
       
        
        if quantity == 'elms':
            return {'elm_beg':[], 'tvec': []} 
        
            
        if quantity == 'sawteeth':
            return {'tvec':[]}
        
            
        
        T = time()
        options = options[quantity]
         
        systems = []
      
        if  quantity in ['Te', 'ne']:
            for sys, stat in options['systems']['TS system']:
                if stat.get(): systems.append(sys)
   
        if  quantity in ['vtor', 'Ti']:
            for sys, stat in options['systems']['HIREX system']:
                if stat.get(): systems.append(sys)

            #for sys, stat in options['systems']['TS system']:
                #if stat.get(): systems.append(sys)
   
   
        data = []
    
        if quantity in ['Te', 'ne']  and len(systems) > 0:
            ts = self.load_ts(tbeg, tend, systems, options['load_options'])
            if quantity == 'ne' and options['systems']['TCI interf.'][1][1].get():
                ts = self.tci_correction(ts, tbeg, tend)
            data.append(ts)
            
        if quantity in ['vtor', 'Ti']  and len(systems) > 0:
            data.append(self.load_hirex(tbeg, tend, systems, options['load_options']))
            #constrain edge dt by TS diagnostic
            if quantity == 'Ti' and options['systems']['TS system'][0][1].get():
                #assume the same Ti and edge Te (collisionally-coupled with Ti)
                ts2 = deepcopy(self.load_ts(tbeg, tend, ['edge']))
                if 'edge' in ts2:
                    #use only timerangee where HIREX dta are availible
                    t1,t2 = np.infty, -np.infty
                    for sys in data[-1]['systems']:
                        if sys in data[-1]:
                            t1_,t2_ = data[-1][sys]['time'].values[[0,-1]]
                            t1, t2 = min(t1,t1_), max(t2, t2_)
                            
                    ts2['edge']['Ti'] = ts2['edge']['Te']
                    ts2['edge']['Ti_err'] = ts2['edge']['Te_err']
                    ts2['edge'] = ts2['edge'].sel(time=slice(t1,t2))
                    data.append(ts2)
     
        if quantity == 'ne' and options['systems']['TCI interf.'][0][1].get():
            data.append(self.load_tci(tbeg, tend))
        if quantity in ['ne'] and options['systems']['Reflectometer'][0][1].get():
            data.append(self.load_refl(tbeg,tend, options['load_options']['Reflectometer']))
        if quantity in ['Te'] and (options['systems']['ECE system'][0][1].get() or options['systems']['ECE system'][1][1].get()):
            data.append(self.load_ece(tbeg,tend,options))

   
        #list of datasets 
        output = {'data':[],'diag_names':[]}
        times = []
        for d in data:
            if d is None or not 'systems' in d: continue
            for sys in d['systems']:
                if not sys in d: continue
                if isinstance(d[sys], list):
                    output['data'].extend(d[sys])
                else:
                    output['data'].append(d[sys])
                output['diag_names'].extend(d['diag_names'][sys])

        #cut data in the selected range
        for i in range(len(output['data'])):
            times.append(output['data'][i]['time'].values)
            output['data'][i]= output['data'][i].sel(time=slice(tbeg,tend))
            #assume a minimum of 5% error on CMOD
            valid = np.isfinite(output['data'][i][quantity+'_err'].values)
            output['data'][i][quantity+'_err'].values[valid] = np.maximum(np.abs(output['data'][i][quantity].values[valid])*0.05,output['data'][i][quantity+'_err'].values[valid])

        if len(output['diag_names']) == 0:
            tkinter.messagebox.showerror('No data to load',
                    'At least one diagnostic must be selected')
            return 
        #udt, cnt = np.unique(np.round(np.diff(np.hstack(times))*1e3,1), return_counts=True)
        #sind = np.argsort(cnt)[::-1]
        #udt[sind[:4]]
        
        output['tres']=np.median(np.round(np.diff(np.hstack(times))*1e3,1))/1e3
        output['tres']=round(output['tres'],6)
        if output['tres'] == 0:    output['tres'] = 0.01
        output['rho_lbl'] = self.rho_coord
         
        return output


    def load_ts(self, tbeg,tend,systems, options=None):
        
        
        T = time()

    
        #use cached data
        self.RAW.setdefault('TS',{'systems':systems})

        ts = self.RAW['TS']
        ts['systems'] = list(systems)
        systems = list(set(systems)-set(ts.keys()))
    
    
        zshift =  0
        if options is not None:
            if 'TS position error' in options:
                zshift = options['TS position error']['Z shift [cm]'].get()/100. #[m] 
    
        
        #update mapping of the catched data
        ts = self.eq_mapping(ts, dz =zshift )            
        ts.setdefault('diag_names',{})

        if len(systems) == 0: #no new system to load
            return ts

        print_line( '  * Fetching TS data ...')

        
        tree = 'ELECTRONS'
        norm={}
        MDS_systems={'core':OrderedDict()}
        
        if self.shot<1070000000:
            mdspath = '\ELECTRONS::TOP.YAG.RESULTS.GLOBAL.PROFILE:'
            MDS_systems['core']['n_e']    = mdspath+'NE_RZ_T'
            MDS_systems['core']['n_e_err']= mdspath+'NE_ERR_ZT'
            MDS_systems['core']['T_e']    = mdspath+'TE_RZ_T'
            MDS_systems['core']['T_e_err']= mdspath+'TE_ERR_ZT'
        else:
            mdspath = '\ELECTRONS::TOP.YAG_NEW.RESULTS.PROFILES.'
            MDS_systems['core']['n_e']    = mdspath+'NE_RZ'
            MDS_systems['core']['n_e_err']= mdspath+'NE_ERR'
            MDS_systems['core']['T_e']    = mdspath+'TE_RZ'
            MDS_systems['core']['T_e_err']= mdspath+'TE_ERR'
        
        norm['core']= {'T_e':1e3,'n_e':1}
        MDS_systems['core']['tvec']  =  'dim_of('+MDS_systems['core']['n_e']+',0)'
        MDS_systems['core']['R']     = '\ELECTRONS::TOP.YAG.RESULTS.PARAM.R'
        MDS_systems['core']['Z']     = mdspath+'Z_SORTED'

        if self.shot>1000000000:
            mdspath = '\ELECTRONS::TOP.YAG_EDGETS.'
            MDS_systems['edge']= OrderedDict()
            MDS_systems['edge']['n_e']    =mdspath+'RESULTS:NE'
            MDS_systems['edge']['n_e_err']=mdspath+'RESULTS:NE:ERROR'
            MDS_systems['edge']['T_e']    =mdspath+'RESULTS:TE'
            MDS_systems['edge']['T_e_err']=mdspath+'RESULTS:TE:ERROR'
            MDS_systems['edge']['tvec']  = 'dim_of('+MDS_systems['edge']['n_e']+',0)'
            MDS_systems['edge']['R']     = '\ELECTRONS::TOP.YAG.RESULTS.PARAM.R'
            MDS_systems['edge']['Z']     = mdspath+'DATA:FIBER_Z'
            norm['edge'] = {'T_e':1,'n_e':1}

        
                    

        TDI = []        
        #prepare list of loaded signals
        for system in systems:
            if system in ts or system not in MDS_systems: 
                continue
            
            ts['diag_names'][system]=['TS:'+system]
            for sig,tdi in MDS_systems[system].items():
                TDI.append(tdi)

        out = mds_load(self.MDSconn, TDI, tree, self.shot)
        ne,ne_err,Te,Te_err,tvec,R,Z = np.asarray(out).reshape(-1,7).T
        

        for isys, sys in enumerate(systems):
            if len(tvec) <= isys or len(tvec[isys]) == 0: 
                ts['systems'].remove(sys)
                continue
            
            #embed()
            
            
            #these points will be ignored and not plotted (negative errobars )
            valid_TS = np.isfinite(Te_err[isys])&np.isfinite(ne_err[isys])
            valid_TS[valid_TS] &= (Te_err[isys][valid_TS]>0) & (Te[isys][valid_TS] > 0)
            valid_TS[valid_TS] &= (ne_err[isys][valid_TS]>0) & (ne[isys][valid_TS] > 0)  & (ne_err[isys][valid_TS] <1e20)
                        
            if sys ==  'core':
                #use average error between even and odd timeslices (i.e. for both lasers)
                ne_err_mean = np.zeros_like(ne_err[isys][:,::2])
                ne_err_mean[valid_TS[:, ::2]] += ne_err[isys][:, ::2][valid_TS[:, ::2]]
                ne_err_mean[valid_TS[:,1::2]] += ne_err[isys][:,1::2][valid_TS[:,1::2]]
                valid = valid_TS[:,1::2]|valid_TS[:,::2]
                ne_err_mean[valid]/= np.int_(valid_TS[:,::2][valid])+np.int_(valid_TS[:,1::2][valid])

                ne_err[isys][:, ::2] = ne_err_mean
                ne_err[isys][:,1::2] = ne_err_mean


            Te_err[isys][~valid_TS]  = -np.infty
            ne_err[isys][~valid_TS]  = -np.infty
            
            
            
            
            
            #core system do not measure well at low temperatures
            if sys ==  'core':
                too_low_Te = np.zeros_like(valid_TS)
                too_low_Te[valid_TS] = Te[isys][valid_TS]*norm[sys]['T_e'] < 100 
                Te_err[isys][too_low_Te] = np.infty
                ne_err[isys][too_low_Te] = np.infty
                
                 
                
            #store only time slices with some useful data
            valid = ~np.all(ne_err[isys]==0,0)
            
            
        
            channel = np.arange(Te_err[isys].shape[0])
            R0 = R[isys]*np.ones_like(channel)
            
            ts[sys] = xarray.Dataset(attrs={'system':sys})
            ts[sys]['ne'] = xarray.DataArray(ne[isys].T[valid]*norm[sys]['n_e'],dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
            ts[sys]['ne_err'] = xarray.DataArray(ne_err[isys].T[valid]*norm[sys]['n_e'],dims=['time','channel'], attrs={'units':'m^{-3}'})
            ts[sys]['Te'] = xarray.DataArray(Te[isys].T[valid]*norm[sys]['T_e'],dims=['time','channel'], attrs={'units':'eV','label':'T_e'})
            ts[sys]['Te_err'] = xarray.DataArray(Te_err[isys].T[valid]*norm[sys]['T_e'],dims=['time','channel'], attrs={'units':'eV'})
            ts[sys]['diags']= xarray.DataArray( np.tile(('TS:'+sys,), ne[isys].T[valid].shape),dims=['time','channel'])            
            ts[sys]['R'] = xarray.DataArray(R0, dims=['channel'], attrs={'units':'m'})
            ts[sys]['Z'] = xarray.DataArray(Z[isys],dims=['channel'], attrs={'units':'m'})
            ts[sys]['time'] = xarray.DataArray(tvec[isys][valid],dims=['time'], attrs={'units':'s'})
            ts[sys]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})

            rho = self.eqm.rz2rho(R0,Z[isys]+zshift,tvec[isys],self.rho_coord)
            ts[sys]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})

 
        print('\t done in %.1fs'%(time()-T))
        ts['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':zshift, 'ed':self.eqm.diag}
        
        return ts 
        
        
    def load_hirex(self, tbeg,tends,systems, options=None):
        
        tht = 0
        if options is not None:
            tht = int(options['HIREX system']['Edition']['THT'].get())
            
        T = time()

        # use cached data
        self.RAW.setdefault('HIREX',{})
        hirex = self.RAW['HIREX'].setdefault(tht,{'systems':systems})

        hirex['systems'] = list(systems)
        
        #new systems to be loaded
        systems = list(set(systems)-set(hirex.keys()))
    
  
        #update mapping of the catched data is not possible!!! data are already mapped
        #hirex = self.eq_mapping(hirex)            
        hirex.setdefault('diag_names',{})

        if len(systems) == 0: #no new system to load
            return hirex
            
        tree = 'spectroscopy'
        if tht == 0: tht = ''

        for sys in systems:
            if sys == 'H-like':
                node = 'HLIKE.PROFILES.LYA1'
            elif sys == 'He-like':
                node = 'HELIKE.PROFILES.Z'
            else:
                raise ValueError('Unrecognized node! '+sys)

            hirex['diag_names'][sys]=['HIREX:'+sys]

        
            # Load the nodes associated with inverted profile data
            node_path = r'\{}::TOP.HIREXSR.ANALYSIS.{}{}'.format(tree,node,tht)
    
            TDI  = [node_path+':'+p for p in ('rho','pro','proerr')]
            TDI += ['dim_of(%s,1)'%TDI[-1]]

            rho,pro,perr,tvec = mds_load(self.MDSconn, TDI,tree, self.shot)
            
            if len(rho) == 0:
                printe('Data for '+sys+' were not found')
                continue
            
            nt = (tvec > 0).sum()
            nr = rho.shape[1]

 
            tvec = tvec[:nt]
            rho = rho[:nt,:nr]
            pro = pro[:,:nt,:nr]
            perr = perr[:,:nt,:nr]
            
            #exclude corrupted points
            valid = np.isfinite(pro)&np.isfinite(perr)
            valid &= (rho <  .9)[None]#measurement too far outside are unreliable
            pro[~np.isfinite(pro)] = 0
            perr[~valid] = -np.infty
            
            #negative ion temperature
            perr[3][pro[3] < 0] = -np.infty
            #too high temperature
            perr[3][pro[3] > 10] = np.infty
            #too fast rotation 
            perr[1][np.abs(pro[1]) > 50] = np.infty

            # exclude obvious outliers            
            #outliers = (pro[3] < .5)|(perr[3] > .5)
            #perr[3,outliers] = np.inf
 
            hirex[sys] = xarray.Dataset(attrs={'system':sys})
            hirex[sys]['Ti'] = xarray.DataArray(pro[3]*1e3,dims=['time','bin'], attrs={'units':'eV','label':'T_i'})
            hirex[sys]['Ti_err'] = xarray.DataArray(perr[3]*1e3,dims=['time','bin'], attrs={'units':'eV'})
            hirex[sys]['vtor'] = xarray.DataArray(pro[1]*1e3,dims=['time','bin'], attrs={'units':'m/s\,','label':'v_\varphi'})
            hirex[sys]['vtor_err'] = xarray.DataArray(perr[1]*1e3,dims=['time','bin'], attrs={'units':'eV'})
            hirex[sys]['diags']= xarray.DataArray(np.tile(('HIREX:'+sys,),pro[0].shape),dims=['time','bin'])            
            hirex[sys]['rho'] = xarray.DataArray(rho,dims=['time','bin'], attrs={'units':'-'})
            hirex[sys]['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
            hirex[sys]['bin'] = xarray.DataArray(np.arange(nr),dims=['bin'])
 
        print('\t done in %.1fs'%(time()-T))

        return hirex 
        
        
        
    def load_refl(self, tbeg,tends, options=None):
        T = time()
         
        r_shift  = 0
        if options is not None:
            r_shift = options['Position error']['R shift [cm]'].get()/100 #[m]  
   
        if 'REFL' in self.RAW:
            #assume that equilibrium could be changed
            return self.eq_mapping(self.RAW['REFL'], dr = r_shift)   
         
        print_line( '  * Fetching reflectometer data ...')

        refl = self.RAW['REFL'] = {}
            
        refl['systems'] = ['FAST']
 
        mdspath =  r'\rf::top.reflect:result:'
        TDI = mdspath+'density',  mdspath+'radius', mdspath+'tavg', mdspath+'reliability'
        
        ne,R,tvec,msg = mds_load(self.MDSconn, TDI, 'RF', self.shot)
        
        if msg.item() == -1:
            print("Unable to fetch reflectometer reliability!")
        else:
            print("SOL reflectometer reliability=%d" % (msg.item()))

        
        if np.size(ne) == 0:
            printe( '\tNo Reflectometer data')
            return 
        
        refl['diag_names'] = {}
 
        R, ne = np.single(R.T), np.single(ne.T)
        z = np.zeros_like(R)

        channel = np.arange(ne.shape[1])
        refl['FAST'] = {}

        refl['diag_names']['FAST'] = ['REFL:FAST']
        refl['FAST'] = xarray.Dataset()
        refl['FAST']['ne'] = xarray.DataArray(ne,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})

        #just guess! of 10% errors!!
        refl['FAST']['ne_err'] = xarray.DataArray(ne*0.1+ne.mean()*0.01 ,dims=['time','channel'], attrs={'units':'m^{-3}'})
        refl['FAST']['diags']= xarray.DataArray(np.tile(('REFL:FAST',), R.shape),dims=['time','channel'])
        refl['FAST']['R'] = xarray.DataArray(R,dims=['time','channel'], attrs={'units':'m'})
        refl['FAST']['Z'] = xarray.DataArray(z,dims=['time','channel'], attrs={'units':'m'})
        refl['FAST']['rho'] = xarray.DataArray(self.eqm.rz2rho(R,z,tvec,self.rho_coord),dims=['time','channel'], attrs={'units':'-'})
        refl['FAST']['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
        refl['FAST']['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})

        
        refl['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}

        print('\t done in %.1fs'%(time()-T))

        return refl
                 
 
        
        
    def load_ece(self, tbeg,tend,options):
    
        T = time()
        
        fast =  bool(options['systems']['ECE system'][1][1].get())
        #print(options)
        dR_shift = float(options['load_options']['ECE system']["shift"]['dR [cm] ='].get())/100.

        rate = 'f' if fast else 's'
        
        if 'ECE' in self.RAW and rate in  self.RAW['ECE']:
            #assume that equilibrium could be changed
            return self.eq_mapping(self.RAW['ECE'][rate], dr = dR_shift)   
         
        #use cached data
        self.RAW.setdefault('ECE',{})
        self.RAW['ECE'].setdefault(rate,{})

 
        ece = self.RAW['ECE'][rate].setdefault('ECE',xarray.Dataset(attrs={'system':rate }))

        self.RAW['ECE'][rate]['diag_names'] = {'ECE':['ECE']}
        self.RAW['ECE'][rate]['systems'] = ['ECE']
        
        
        TDI = []
        for k in range(0, 32):
            TDI.append(r'frcece.data.ece%s%02d' % (rate, k + 1))
            TDI.append('frcece.data.rmid_%02d' % (k + 1))

        TDI.append(r'dim_of(frcece.data.ece%s%02d)' % (rate, 1))
        TDI.append(r'dim_of(frcece.data.rmid_%02d)' % (1))

        out = mds_load(self.MDSconn, TDI, 'electrons', self.shot)
        
        Te,R = np.reshape(out, (-1,2)).T
        
        #downsample by a factor of 10
        Te = [t[:len(t)//10*10].reshape(len(t)//10, 10).mean(1) for t in Te]
        
        Te, tvec = Te[:-1], Te[-1]
        R, Rtvec = R[:-1] ,  R[-1]
        Te = np.vstack(Te).T*1e3 #eV
        R = np.hstack(R)
        Z = np.zeros_like(R)
        channel = np.arange(Te.shape[1])
        #embed()

        rho = self.eqm.rz2rho(R+dR_shift,Z,Rtvec,self.rho_coord)
        R   = interp1d(Rtvec,R,axis=0)(np.clip(tvec, *Rtvec[[0,-1]]))
        rho = interp1d(Rtvec, rho,axis = 0)(np.clip(tvec, *Rtvec[[0,-1]]))
        Z   = np.zeros_like(R)        
        
        
        Te_err = np.abs(Te)*0.05+50.  #my naive guess of errorbars
        ece['Te'] = xarray.DataArray(Te, coords=[tvec, channel], dims=['time','channel'], attrs={'units':'eV','label':'T_e'} )
        ece['Te_err'] = xarray.DataArray(Te_err, dims=['time','channel'], attrs={'units':'eV'} )
        ece['diags'] = xarray.DataArray(np.tile(('ECE',),Te.shape), dims=['time','channel'], attrs={'units':'-'} )
        ece['R'] = xarray.DataArray(R, dims=['time','channel'], attrs={'units':'m'} )
        ece['Z'] = xarray.DataArray(Z, dims=['time','channel'], attrs={'units':'m'} )
        ece['rho'] = xarray.DataArray(rho, dims=['time','channel'], attrs={'units':'-'} )

        print('\t done in %.1fs'%(time()-T))
        self.RAW['ECE'][rate]['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}

        return self.RAW['ECE'][rate]








    
    def load_tci(self, tbeg,tend, calc_weights=True):
        
        T = time()

         
        TCI = self.RAW.setdefault('TCI',{})

        #update mapping of the catched data
        if 'systems' in TCI and (not calc_weights or 'weight' in TCI[TCI['systems'][0]]):
            TCI = self.eq_mapping(TCI) 
            #return catched data 
            return TCI
        
        
        TCI['systems'] = ['core']
        TCI.setdefault('diag_names',{})
        TCI['diag_names']['core'] = ['TCI']
        TCI['core'] = []

        print_line( '  * Fetching TCI interferometer data ...')
        
        #vertical LOS
        self.MDSconn.openTree('electrons', self.shot)                      
        Rlos = self.MDSconn.get(r'tci.results:rad').data()
        Zlos = Rlos*0


        TDI = []
        tree = 'ELECTRONS'
        for i in range(len(Rlos)):
            TDI.append(r'tci.results:nl_%02d' % (i + 1))
        TDI.append(r'dim_of(tci.results:nl_%02d)' % (1))
                
        out = mds_load(self.MDSconn, TDI, tree, self.shot)

        ne_, tvec_ = out[:-1], out[-1]
        ne_ = np.vstack(ne_).T
        
        Rlcfs,Zlcfs = self.eqm.rho2rz(0.995)
        n_path = 501        
        
        downsample = 5
        
        valid = np.ones_like(ne_, dtype='bool')
        valid[:-1] = np.diff(ne_,axis=0) < 3e19  #detect fringe jumps
        valid = np.bool_(np.cumprod(valid, axis=0)) & (ne_ > 1)
        
        
        t = np.linspace(0,1,n_path, dtype='single')        
        nt = len(tvec_)//downsample
        valid_ch = np.where(np.any(valid,0))[0]
        n_ch = len(valid_ch)

        ne = np.zeros((nt, n_ch), dtype='single')
        ne_err = np.zeros((nt, n_ch), dtype='single')
        R = np.zeros((n_ch,n_path), dtype='single')
        Z = np.zeros((n_ch,n_path), dtype='single')
        L = np.zeros((n_ch,n_path), dtype='single')
        L_cross = np.zeros((nt,n_ch), dtype='single')
        weight= np.zeros((nt, n_ch, n_path), dtype='single')
        tvec = tvec_[:nt*downsample].reshape(-1,downsample).mean(1)
        
        for ilos,ch in enumerate(valid_ch):
            
 
            #Calculate the error for the signal based on the drift before t=0
            ne[:,ilos] = ne_[:nt*downsample,ch].reshape(-1,downsample).mean(1)
            ne_err[:,ilos] = np.median(np.abs(ne[:,ilos][tvec<0]))+ne[:,ilos]*.05  #guess 5% error
            ne_err[~np.all(valid[:nt*downsample,ch].reshape(-1,downsample),axis=1),ilos] = np.inf
            R[ilos] = Rlos[ch]
            Z[ilos] = -0.5+t #between -.5 and .5m 
            L[ilos] = np.hypot(R[ilos]-R[ilos,0],Z[ilos]-Z[ilos,0])
      
            if calc_weights:
                dL = np.gradient(L[ilos])
                LOS_pt1 = np.array((R[ilos,0 ], Z[ilos, 0]))
                LOS_pt2 = np.array((R[ilos,-1], Z[ilos,-1]))

                #length of crossection over plasma
                L_cross_ = np.zeros(len(Rlcfs), dtype='single')
                for it,(r_,z_) in enumerate(zip(Rlcfs,Zlcfs)):
                    r,z = r_[0], z_[0]
                    A = ((r[1:]-r[:-1]),(z[1:]-z[:-1])),((-LOS_pt2[0]+LOS_pt1[0])*np.ones(len(z)-1),(-LOS_pt2[1]+LOS_pt1[1])*np.ones(len(z)-1))
                    A = np.array(A).T
                    b = np.array((LOS_pt1[0]-r[:-1],LOS_pt1[1]-z[:-1])).T
                    
                    try:
                        X = np.linalg.solve(A,b)
                    except:
                        continue
                    
                    cross_ind = (X[:,0]>0)&(X[:,0]<=1)
                    if len(X) > 0 and any(cross_ind):
                        crossings = -np.outer(A[0,:,1],X[cross_ind,1]).T+LOS_pt1
                        L_cross_[it] = np.linalg.norm(np.diff(crossings,axis=0))
                
                #L_cross is length over the plasma - just a normalisation for nice plotting 
                L_cross[:,ilos] = np.interp(tvec, self.eqm.t_eq ,L_cross_)
                L_cross[:,ilos] = np.maximum(L_cross[:,ilos], .1) #just to avoid zero division
                L_cross[:,ilos]*= .85 # correction just for better plotting of the data, it is not affecting fit
                #convert from m^-2 -> m^-3
                weight[:,ilos,:] = dL/L_cross[:,ilos][:,None]
                ne[:,ilos] /= L_cross[:,ilos]
                ne_err[:,ilos] /= L_cross[:,ilos]
               
        
        TCI['core'] = xarray.Dataset()
        TCI['core']['channel'] = xarray.DataArray( valid_ch ,dims=['channel'])
        TCI['core']['path'] = xarray.DataArray( t ,dims=['path'])
        TCI['core']['time'] = xarray.DataArray( tvec ,dims=['time'], attrs={'units':'s'})
        TCI['core']['ne'] = xarray.DataArray(ne, dims=['time', 'channel'], attrs={'units':'m^{-3}','label':'n_e'})
        TCI['core']['ne_err'] = xarray.DataArray(ne_err,dims=['time', 'channel'], attrs={'units':'m^{-3}'})
        TCI['core']['diags']= xarray.DataArray( np.tile(('TCI',), (nt, n_ch)),dims=['time', 'channel'])
        TCI['core']['R'] = xarray.DataArray(R,dims=['channel','path'],attrs={'units':'m'})
        TCI['core']['Z'] = xarray.DataArray(Z,dims=['channel','path'],attrs={'units':'m'})
        if not 'rho' in TCI['core']:
            TCI['core']['rho'] = xarray.DataArray(np.zeros((nt, n_ch, n_path), dtype='single'),dims=['time', 'channel','path'])
        TCI['core']['L'] = xarray.DataArray(L,dims=['channel','path'],attrs={'units':'m'})
   
        TCI['core']['L_cross']  =  xarray.DataArray(1)


     
        if 'EQM' in TCI: TCI.pop('EQM')
        
        TCI = self.eq_mapping(TCI) 
        rho_tg = TCI['core']['rho'].values.min(2)
        if calc_weights:
            weight[TCI['core']['rho'].values >= 1.05] = 0 #make is consistent with C02 correction
            TCI['core']['weight']  = xarray.DataArray(weight, dims=['time', 'channel','path'])
            TCI['core']['L_cross'] = xarray.DataArray(L_cross,dims=['time','channel'],attrs={'units':'m'})

        TCI['core']['rho_tg'] = xarray.DataArray(rho_tg,dims=['time', 'channel'])


        TCI['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}

        print('\t done in %.1fs'%(time()-T))

        return TCI
    
    
    
    def tci_correction(self,TS, tbeg,tend):
        
        T = time()
        
        #embed()
        if not 'core' in TS or not 'edge' in TS:
            print('TCI correction could not be done')
            return TS
  
        #BUG ignoring user selected wrong channels of TS 
        T = time()
        
        TCI = self.load_tci(tbeg,tend, calc_weights = False)['core']
        
        print_line( '  * Calculate TS correction using TCI ...')

        tbeg = max(tbeg, TCI['time'][0].values)
        tend = min(tend, TCI['time'][-1].values)
   

        tvec_compare =  TS['core']['time'].values
        t_ind = (tvec_compare > tbeg) & (tvec_compare < tend)
        ind = np.argsort(tvec_compare[t_ind])
        tvec_compare = tvec_compare[t_ind][ind]
 
       
        LOS_rho = interp1d(TCI['time'].values, TCI['rho'].values,copy=False, axis=0,
                           assume_sorted=True)(np.clip(tvec_compare,*TCI['time'].values[[0,-1]]))
        LOS_L = TCI['L'].values

        
        #1)  correction of the tang system with respect to core

        core_tvec = TS['core']['time'].values
        ind = slice(*core_tvec.searchsorted((tbeg, tend)))
        core_tvec = core_tvec[ind]
        #core_lasers  = TS['core']['laser'].values[ind]
        #core_lasers_list = np.unique(core_lasers)
        core_ne  = TS['core']['ne'].values[ind]
        core_err  = TS['core']['ne_err'].values[ind]
        core_rho = TS['core']['rho'].values[ind]
        
        
        edge_rho= interp1d(TS['edge']['time'],TS['edge']['rho'],axis=0,copy=False, assume_sorted=True)(core_tvec)
        edge_ne = interp1d(TS['edge']['time'],TS['edge']['ne'],axis=0,copy=False, assume_sorted=True)(core_tvec)
        edge_err= interp1d(TS['edge']['time'],TS['edge']['ne_err'] ,axis=0,copy=False, assume_sorted=True,kind='nearest')(core_tvec)

        #remove corrupted channels
        corrupted_core = np.sum(~np.isfinite(core_err),0)>core_err.shape[0]/5.
        corrupted_edge = np.sum(~np.isfinite(edge_err),0)>edge_err.shape[0]/5.
        
 
        rho = np.hstack((core_rho[:,~corrupted_core],edge_rho[:,~corrupted_edge]))
        rho_sort_ind  = np.argsort(rho,axis=1)
        core_ne = core_ne[:,~corrupted_core]

        
        time_sort_ind = np.tile(np.arange(rho.shape[0]), (rho.shape[1], 1)).T
        
        rho = rho[time_sort_ind, rho_sort_ind]
        def cost_fun(corr):
            edge_ne_ = np.double(edge_ne[:,~corrupted_edge])
            edge_ne_  *= np.exp(corr)
            ne  = np.hstack((core_ne, edge_ne_ ))/1e19
            ne  = ne[time_sort_ind, rho_sort_ind]
            val = np.nansum((np.diff(ne)/(ne[:,1:]+ne[:,:-1]+.1))**2)
            return val
            
        from scipy.optimize import minimize 
   
        p0 = [0]
        opt = minimize(cost_fun,  p0, tol = .0001 )
        corr = np.exp(opt.x)
  
        #correct edge system
        TS = deepcopy(TS)
        TS['edge']['ne'].values  *= corr
   

        #interpolate data on the time of the core system
        N,R = [],[]
        for diag in ['core','edge']:
            t,n,e,r = TS[diag]['time'].values, TS[diag]['ne'].values, TS[diag]['ne_err'].values, TS[diag]['rho'].values
            nchs = n.shape[1]
            n_interp = np.zeros((len(tvec_compare), n.shape[1]))
            for nch in range(nchs):
                ind = (n[:,nch] > 0) & (e[:,nch] > 0) & np.isfinite(e[:,nch])  
                if sum(ind) <= 2: continue  #corrupted channel 
                N.append(np.interp(tvec_compare,  t[ind], n[ind,nch]))
                R.append(np.interp(tvec_compare,  t[ind], r[ind,nch]))
 
        
        R = np.vstack(R).T
        N = np.vstack(N).T
        N[R > 1.05] = 0
        #sort them from the core to the edge
        rho_sort_ind = np.argsort(R,axis=1)
        time_sort_ind = np.tile(np.arange(R.shape[0]), (R.shape[1], 1)).T
        R = R[time_sort_ind, rho_sort_ind]
        N = N[time_sort_ind, rho_sort_ind]
 
        #NOTE random noise should averadge out, spline fit is not necessary  
        #interpolate density along LOS for each time 
        LOS_ne = [np.interp(lr,r,n,right=0) for lr, r, n in zip(LOS_rho, R,N)]
        #do line integration
        LOS_ne_int = np.trapz(LOS_ne,LOS_L,axis=-1)
        #core_lasers = np.unique(laser_index)
        
        tci_los_names  = TCI['channel'].values
        #laser_correction     = np.ones((len(core_lasers),len(tci_los_names)))*np.nan
        #laser_correction_err = np.ones((len(core_lasers),len(tci_los_names)))*np.nan
        los_dict =  {n:i for i,n in enumerate(tci_los_names)}
        valid = np.isfinite(TCI['ne_err'].values)
        time_tci = TCI['time'].values
        
        ne = TCI['ne'].values*TCI['L_cross'].values 
        laser_correction = np.ones(len(tci_los_names))*np.nan
        
        for ilos,los in enumerate( tci_los_names):
            if not np.any(valid[:,ilos]): continue
            t_ind = (tvec_compare > time_tci[valid[:,ilos]][0]) & (tvec_compare < time_tci[valid[:,ilos]][-1])
            if not np.any(t_ind): continue
            ratio = LOS_ne_int[t_ind ,ilos]/np.interp(tvec_compare[t_ind],time_tci[valid[:,ilos]],ne[valid[:,ilos], ilos])
            laser_correction[ilos] = np.median(ratio)

        #for il, l in enumerate(core_lasers):
            #ind = laser_index == l
  
            #for ilos,los in enumerate( tci_los_names):
                #if not np.any(valid[:,ilos]): continue
                #t_ind = (tvec_compare > time_tci[valid[:,ilos]][0]) & (tvec_compare < time_tci[valid[:,ilos]][-1])
                #if not np.any(ind&t_ind): continue
                #ratio = LOS_ne_int[ind&t_ind ,ilos]/np.interp(tvec_compare[ind&t_ind],time_tci[valid[:,ilos]],ne[valid[:,ilos], ilos])
                #laser_correction[il,ilos] = np.median(ratio)
                #laser_correction_err[il,ilos] = ratio.std()/np.sqrt(len(ratio))
                    
        mean_laser_correction = np.nanmean(laser_correction)
              


        #correction of laser intensity variation and absolute value
        print('TCI corrections:', (np.round(mean_laser_correction,3)), ' core vs. edge:', np.round(corr,3))
        for sys in ['core', 'edge']:
            TS[sys]['ne'].values/= mean_laser_correction
       
        print('\t done in %.1fs'%(time()-T))

        return TS 
    
    
    
    
    

            #if electrons is None:
        #electrons = MDSplus.Tree('electrons', shot)
    #if efit_tree is None:
        #p.efit_tree = eqtools.CModEFITTree(shot)
    #else:
        #p.efit_tree = efit_tree

    #Te_FRC = []
    #R_mid_FRC = []
    #t_FRC = []
    #channels = []
    #for k in xrange(0, 32):
        #N = electrons.getNode(r'frcece.data.ece%s%02d' % (rate, k + 1,))
        #Te = N.data()
        #Te_FRC.extend(Te)
        ## There appears to consistently be an extra point. Lacking a better
        ## explanation, I will knock off the last point:
        #t = N.dim_of().data()[:len(Te)]
        #t_FRC.extend(t)
        
        #N_R = electrons.getNode(r'frcece.data.rmid_%02d' % (k + 1,))
        #R_mid = N_R.data().flatten()
        #t_R_FRC = N_R.dim_of().data()
        #R_mid_FRC.extend(
            #scipy.interpolate.InterpolatedUnivariateSpline(t_R_FRC, R_mid)(t)
        #)
        
        
        
        
        
        
        
        
        
        
        
        
    """Returns a profile representing electron temperature from the FRCECE system.
    
    Parameters
    ----------
    shot : int
        The shot number to load.
    rate : {'s', 'f'}, optional
        Which timebase to use -- the fast or slow data. Default is 's' (slow).
    cutoff : float, optional
        The cutoff value for eliminating cut-off points. All points with values
        less than this will be discarded. Default is 0.15.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'Rmid'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
    """
    #p = BivariatePlasmaProfile(
        #X_dim=2,
        #X_units=['s', 'm'],
        #y_units='keV',
        #X_labels=['$t$', r'$R_{mid}$'],
        #y_label=r'$T_e$, FRCECE (%s)' % (rate,),
        #weightable=False
    #)

    #if electrons is None:
        #electrons = MDSplus.Tree('electrons', shot)
    #if efit_tree is None:
        #p.efit_tree = eqtools.CModEFITTree(shot)
    #else:
        #p.efit_tree = efit_tree

    #Te_FRC = []
    #R_mid_FRC = []
    #t_FRC = []
    #channels = []
    #for k in xrange(0, 32):
        #N = electrons.getNode(r'frcece.data.ece%s%02d' % (rate, k + 1,))
        #Te = N.data()
        #Te_FRC.extend(Te)
        ## There appears to consistently be an extra point. Lacking a better
        ## explanation, I will knock off the last point:
        #t = N.dim_of().data()[:len(Te)]
        #t_FRC.extend(t)
        
        #N_R = electrons.getNode(r'frcece.data.rmid_%02d' % (k + 1,))
        #R_mid = N_R.data().flatten()
        #t_R_FRC = N_R.dim_of().data()
        #R_mid_FRC.extend(
            #scipy.interpolate.InterpolatedUnivariateSpline(t_R_FRC, R_mid)(t)
        #)
        
        #channels.extend([k + 1] * len(Te))
    
    #Te = scipy.asarray(Te_FRC)
    #t = scipy.atleast_2d(scipy.asarray(t_FRC))
    #R_mid = scipy.atleast_2d(scipy.asarray(R_mid_FRC))
    
    #X = scipy.hstack((t.T, R_mid.T))
    
    #p.shot = shot
    #p.abscissa = 'Rmid'
    
    #p.add_data(X, Te, channels={1: scipy.asarray(channels)}, err_y=0.1 * scipy.absolute(Te))
    ## Remove flagged points:
    ## I think these are cut off channels, but I am not sure...
    #p.remove_points(p.y < cutoff)
    #if t_min is not None:
        #p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    #if t_max is not None:
        #p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    #p.convert_
 
            
        
        
##[docs]def neReflect(shot, absdcissa='Rmid', t_min=None, t_max=None, electrons=None,
              ##efit_tree=None, remove_edge=False, rf=None):
    #"""Returns a profile representing electron density from the LH/SOL reflectometer system.

    #Parameters
    #----------
    #shot : int
        #The shot number to load.
    #abscissa : str, optional
        #The abscissa to use for the data. The default is 'Rmid'.
    #t_min : float, optional
        #The smallest time to include. Default is None (no lower bound).
    #t_max : float, optional
        #The largest time to include. Default is None (no upper bound).
    #electrons : MDSplus.Tree, optional
        #An MDSplus.Tree object open to the electrons tree of the correct shot.
        #The shot of the given tree is not checked! Default is None (open tree).
    #efit_tree : eqtools.CModEFITTree, optional
        #An eqtools.CModEFITTree object open to the correct shot. The shot of the
        #given tree is not checked! Default is None (open tree).
    #remove_edge : bool, optional
        #If True, will remove points that are outside the LCFS. It will convert
        #the abscissa to psinorm if necessary. Default is False (keep edge).
    #rf : MDSplus.Tree, optional
        #An MDSplus.Tree object open to the RF tree of the correct shot.
        #The shot of the given tree is not checked! Default is None (open tree).
    #"""
    #p = BivariatePlasmaProfile(
        #X_dim=2,
        #X_units=['s', 'm'],
        #y_units='$10^{20}$ m$^{-3}$',
        #X_labels=['$t$', r'$R_{mid}$'],
        #y_label=r'$n_e$, reflect',
        #weightable=False
    #)
    #if rf is None:
        #rf = MDSplus.Tree('rf', shot)
    #if efit_tree is None:
        #p.efit_tree = eqtools.CModEFITTree(shot)
    #else:
        #p.efit_tree = efit_tree
    
    #t = rf.getNode(r'\rf::top.reflect:result:tavg').getData().data()
    #R = rf.getNode(r'\rf::top.reflect:result:radius').getData().data()
    #ne = rf.getNode(r'\rf::top.reflect:result:density').getData().data() / 1e20
    
    #try:
        #print("SOL reflectometer reliability=%d" % (rf.getNode(r'\rf::top.reflect:result:reliability').data(),))
    #except:
        #print("Unable to fetch reflectometer reliability!")
    
    #channels = range(0, ne.shape[1])

    #channel_grid, t_grid = scipy.meshgrid(channels, t)

    #ne = ne.ravel()
    #R = R.ravel()
    #channels = channel_grid.ravel()
    #t = t_grid.ravel()

    #X = scipy.vstack((t, R)).T

    #p.shot = shot
    #p.abscissa = 'Rmid'

    #p.add_data(X, ne, channels={1: channels}, err_y=0.1 * scipy.absolute(ne))
    
    ## Remove flagged points:
    #p.remove_points(p.y == 0)
    #if t_min is not None:
        #p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    #if t_max is not None:
        #p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)

    #p.convert_abscissa(abscissa)

    #if remove_edge:
        #p.remove_edge_points()

    #return p



        
    #def load_hirex(self,tbeg,tend, systems, options=None):
        
        
           ##load Ti and omega at once
        #T = time()

        #tree = 'IONS'
        ##if options is None:
            ##analysis_type = 'best'
            ##analysis_types= 'fit','auto','quick'
        ##else:
            ##selected,analysis_types = options['Analysis']
            ##analysis_type = selected.get()
            
        #analysis_type = self.get_cer_types(analysis_type)
         
        #self.RAW.setdefault('CER',{})
        #cer = self.RAW['CER'].setdefault(analysis_type,{})

        ##load from catch if possible
        #cer.setdefault('diag_names',{})
        #cer['systems'] = systems
        #load_systems = list(set(systems)-set(cer.keys()))
 
        ##update equilibrium for already loaded systems
        #cer = self.eq_mapping(cer)
        
        ##rho coordinate of the horizontal line, used later for separatrix aligment 
        #if 'horiz_cut' not in cer or 'EQM' not in cer or cer['EQM']['id'] != id(self.eqm) or cer['EQM']['ed'] != self.eqm.diag:
            #R = np.linspace(1.8,2.5)
            #rho_horiz = self.eqm.rz2rho(R, np.zeros_like(R), coord_out='rho_tor')
            #cer['horiz_cut'] = {'time':self.eqm.t_eq, 'rho': np.single(rho_horiz), 'R':R}
        
        #if len(load_systems) == 0:
            #return cer
   
        #print_line( '  * Fetching '+analysis_type.upper()+' data ...' )
        
        #cer_data = {
            #'Ti': {'label':'Ti','unit':'eV','sig':['TEMP','TEMP_ERR']},
            #'omega': {'label':r'\omega_\varphi','unit':'rad/s','sig':['ROT','ROT_ERR']},
            #'VB': {'label':r'VB','unit':'a.u.','sig':['VB','VB_ERR']},}
        ##NOTE visible bramstrahlung is not used
       
        ##list of MDS+ signals for each channel
        #signals = cer_data['Ti']['sig']+cer_data['omega']['sig'] + ['R','Z','STIME']

        #all_nodes = []
        
        #TDI = []
        #diags_ = []

        #try:
            #self.MDSconn.openTree(tree, self.shot)
            ##prepare list of loaded signals
            #for system in load_systems:
                #if system in cer:
                  ##already loaded
                    #continue
                #cer[system] = []
                #cer['diag_names'][system] = []
                #path = 'CER.%s.%s.CHANNEL*'%(analysis_type,system)
                #nodes = self.MDSconn.get('getnci("'+path+'","fullpath")')
                #for node in nodes:
                    ##if not isinstance(node, str):
                        ##print(node)
                    #try:
                        #node = node.decode()
                    #except:
                        #pass
                    #diags_.append(system)
 
                    #node = node.strip()
                    #TDI.append(node+':TIME')
                    #all_nodes.append(node)

        #except Exception as e:
            #raise
            #printe( 'MDS error: '+ str(e))
        #finally:
            #self.MDSconn.closeTree(tree, self.shot)
        
        #tvec = mds_load(self.MDSconn, TDI, tree, self.shot)

        
        #valid_node = [ch for ch,t in zip(all_nodes,tvec) if len(t)]
        #tvec = [t for t in tvec if len(t)]
    
        


#replace a GUI call

def main():
    
    mdsserver = 'alcdata.psfc.mit.edu'
    #import MDSplus
    #try:
        #MDSconn = MDSplus.Connection(mdsserver)
    #except:
    mdsserver = 'localhost'

    MDSconn = MDSplus.Connection(mdsserver)
    
    
    #embed()
    TT = time()

    from map_equ import equ_map
    import tkinter as tk
    myroot = tk.Tk(className=' Profiles')
 #python ./quickfit.py --mdsplus localhost  --device CMOD --shot 1160920013

    rho_coord = 'rho_tor'
    #shot =   175860
    shot = 1101209003
    print_line( '  * Fetching EFIT01 data ...')
    #embed()
    #MDSconn.openTree('EFIT01', shot)                      

    eqm = None

     #if tree.upper() == 'ANALYSIS':
            #root = '\\analysis::top.efit.results.'
        #else:
            #root = '\\'+tree+'::top.results.'
            
            
    eqm = equ_map(MDSconn)
    eqm.Open(shot, 'ANALYSIS', exp='CMOD')

    ##load EFIT data from MDS+ 
    T = time()
    eqm._read_profiles()
    eqm._read_pfm()
    eqm.read_ssq()
    eqm._read_scalars()
    print('\t done in %.1f'%(time()-T))

            
    #print 'test'
    #exit()
    #eqm = None
    loader = data_loader(MDSconn, shot, eqm, rho_coord)

    loader.load_hirex(1,2,['A','B'])


    settings = OrderedDict()
    I = lambda x: tk.IntVar(value=x)
    S = lambda x: tk.StringVar(value=x)
    D = lambda x: tk.DoubleVar(value=x)
 
    
    ts_revisions = []

    settings.setdefault('Ti', {\
        'systems':{'CER system':(['tangential',I(1)], ['vertical',I(0)])},\
        'load_options':{'CER system':{'Analysis':(S('auto'), ('best','fit','auto','quick'))} }})
        
    settings.setdefault('omega', {\
        'systems':{'CER system':(['tangential',I(1)], ['vertical',I(0)])},
        'load_options':{'CER system':{'Analysis':(S('best'), (S('best'),'fit','auto','quick'))}}})
    settings.setdefault('Mach', {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':(S('best'), (S('best'),'fit','auto','quick'))}}})
    settings.setdefault('Te/Ti', {\
        'systems':{'CER system':(['tangential',I(1)], ['vertical',I(0)] )},
        'load_options':{'CER system':{'Analysis':(S('best'), (S('best'),'fit','auto','quick'))}}})            
        
    settings.setdefault('nimp', {\
        'systems':{'CER system':(['tangential',I(1)], ['vertical',I(1)])},
        'load_options':{'CER system':OrderedDict((
                                ('Analysis', (S('best'), (S('best'),'fit','auto','quick'))),
                                ('Correction',{'Relative calibration':I(1)}  )))   }})
 
    settings.setdefault('Te', {\
    'systems':OrderedDict((( 'TS system',(['tangential',I(0)], ['core',I(0)],['divertor',I(0)])),
                            ('ECE system',(['slow',I(1)],['fast',I(0)])))),
    'load_options':{'TS system':{"TS revision":(S('BLESSED'),['BLESSED']+ts_revisions)},
                    'ECE system':OrderedDict((
                                ("shift",{'dR [cm] =': D(1.0)}),
                                ))   }})
        
    settings.setdefault('ne', {\
        'systems':OrderedDict((( 'TS system',(['edge',I(1)], ['core',I(1)] )),
                                ( 'Reflectometer',(['all bands',I(0)],)),
                                ( 'TCI interf.',(['fit TCI',I(1)],['rescale TS',I(1)])) ) ),
        'load_options':{'TS system':{"TS revision":(S('BLESSED'),['BLESSED']+ts_revisions)},
                        'Reflectometer':{'Position error':{'R shift [cm]':D(0.0)}}}})
        
    settings.setdefault('Zeff', {\
        'systems':OrderedDict(( ( 'VB array',  (['tangential',I(1)],                 )),
                                ( 'CER VB',    (['tangential',I(1)],['vertical',I(1)])),
                                ( 'CER system',(['tangential',I(1)],['vertical',I(1)])))), \
        'load_options':{'VB array':{'Corrections':{'radiative mantle':I(1),'rescale by TCI':I(1)}},\
                        'TS':{'Position error':{'Z shift [cm]':D(0.0)}},
                        'CER VB':{'Analysis':(S('auto'), (S('best'),'fit','auto','quick'))},
                        'CER system':OrderedDict((
                                ('Analysis', (S('best'), (S('best'),'fit','auto','quick'))),
                                ('Correction',    {'Relative calibration':I(0)}),
                                ('TS position error',{'Z shift [cm]':D(0.0)})))
                        }\
            })
        

        
    settings['elm_signal'] = S('fs01up')
    settings['elm_signal'] = S('fs03')

    #print settings['Zeff'] 
    #loader.load_ece(1,2, settings['Te'])
    #loader.load_refl(1,2,None)

    #exit()

    #exit()

    #TODO 160645 TCI correction is broken
    #160646,160657  crosscalibrace nimp nefunguje

    #load_zeff(self,tbeg,tend, options=None)
    data = loader( 'ne', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])
 
    #loader.load_elms(settings)
    
    #embed()


    #MDSconn.openTree('NB', shot)                      
    #nbi = MDSconn.get('\\NB::TOP.NB{0}:PINJ_{0}'.format('30L')).data()
    #t = MDSconn.get('\\NB::TOP:TIMEBASE').data()
    
    #plt.plot(t/1000,nbi/2500000*4);
    #plt.plot(data['data'][-3].time, data['data'][-3].Zeff.values[:,-7:-2].mean(1)) 
    #xlim(1,6)
    #np.savetxt('zeff%d'%shot,np.c_[data['data'][-3].time, data['data'][-3].Zeff.values[:,-7:-2].mean(1)])
    #show()
    
#xd
    
    #plot(tvec_,PHDMID_);show()


    embed()
    
    #for q in ['Te', 'ne', 'Ti','omega','nimp']:
        #data = loader( q, load_options,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])

    print('\n\t\t\t Total time %.1fs'%(time()-TT))

  
if __name__ == "__main__":
    main()
 




