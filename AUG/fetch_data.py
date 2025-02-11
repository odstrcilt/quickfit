from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
#np.seterr(all='raise')
from IPython import embed
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except Exception:
    from scipy.integrate import cumtrapz

#Note about output errorbars:
#positive finite - OK
#positive infinite - show points but do not use in the fit
#negative finite - disabled in GUI, but it can be enabled
#negative infinite - Do not shown, do not use


def print_line(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def printe(message):
    CSI="\x1B["
    reset=CSI+"m"
    red_start = CSI+"31;40m"
    red_end = CSI + "0m" 
    print(red_start,message,red_end)
    
def roman2int(string):
    val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    string = string.upper()
    total = 0
    while string:
        if len(string) == 1 or val[string[0]] >= val[string[1]]:
            total += val[string[0]]
            string = string[1:]
        else:
            total += val[string[1]] - val[string[0]]
            string = string[2:]
    return total

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

      
def inside_convex_hull(boundary,x):
    """x.shape = (...,2), boundary.shape = (:,2), boundary must have the clock wise direction
    it is !!not!! working for nonconvex boundaries"""
    def cross(o, a, b): 
        return (a[:,0] - o[:,0]) * (b[...,1,None] - o[:,1]) - (a[:,1] - o[:,1]) * (b[...,0,None] - o[:,0])

    #ind =  cross(boundary[:-1,:], boundary[1:,:],x)>0

    
    inlayers = np.all(cross(boundary[:-1,:], boundary[1:,:],x) > 0,axis=-1)

    return inlayers
        
 
 
kin_profs = 'Ti', 'omega', 'Te', 'ne','nimp', 'Zeff'   


def default_settings(MDSconn, shot):
         
    #build a large dictionary with all settings
    default_settings = OrderedDict()
    default_settings['Ti']= {\
        'systems':{'CER system':(['CEZ',True], ['COZ',False], ['CHZ',False], ['CMZ',False],
                                 ['CPZ',False], ['CAZ',False])},'load_options':{}}
        
    default_settings['omega']= {\
        'systems':{'CER system':(['CEZ',True], ['COZ',False], ['CHZ',False], ['CMZ',False] )},'load_options':{}}
    
    default_settings['nimp']= {\
        'systems':{'CER system':(['tangential',True], ['vertical',True])},'load_options':{}}
 
    default_settings['Zeff']= {\
    'systems':OrderedDict(( ( 'CER system',(['tangential',False],['vertical',False])),
                            ( 'CER VB',    (['tangential',True],['vertical',False])),
                            )),'load_options':{}}

    default_settings['Te']= {\
    'systems':OrderedDict((( 'TS system',(['core',False], ['edge',False])),
                           ('IDA',(['Te',True]  )),
                            ('ECE system',(['core',False]  )))),
    'load_options':{
                    'ECE system':OrderedDict((
                                ("Bt correction",{'Bt *=': 1.0}),
                                ('TS correction',{'rescale':False})))}}
        
    default_settings['ne']= {\
        'systems':OrderedDict((( 'TS system',(['core',False], ['edge',False])),
                                ( 'IDA',(['n_e',True],)),
                                ( 'DCN',(['fit DCN data',False],['rescale TS',True])) ) ),
        'load_options':{'TS system':{"TS revision":('BLESSED',['BLESSED']+ts_revisions)},
                        'Reflectometer':{'Position error':{'R shift [cm]':0.0}}}}
 
        
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
                R = diag[sys]['R'].values
                Z = diag[sys]['Z'].values
                T = diag[sys]['time'].values
            
            #do mapping 
            rho = self.eqm.rz2rho(R+dr,Z+dz,T,self.rho_coord)
            
            if isinstance(diag[sys], list):
                for ch,ind in zip(diag[sys],I):
                    ch['rho'].values  = rho[ind,0]
            else:
                diag[sys]['rho'].values  =  rho 
        
        diag['EQM'] = {'id':id(self.eqm),'dr':dr, 'dz':dz,'ed':self.eqm.diag}
 


        return diag
            
            
        
    def __call__(self,  quantity=[], options=None,zipfit=False, tbeg=0, tend=10 ):
   
 
        
        if quantity == 'elms':
            return self.load_elms(options)
        
            
        if quantity == 'sawteeth':
            return self.load_sawteeth()
        
            
        
        T = time()

        
        options = options[quantity]
         
        systems = []
        if quantity in ['Ti', 'omega', 'nimp','Mach','Te/Ti','Zeff']:
            for sys, stat in options['systems']['CER system']:
                if stat.get(): systems.append(sys)
                
        if  quantity in ['Te', 'ne']:
            for sys, stat in options['systems']['TS system']:
                if stat.get(): systems.append(sys)
        if  quantity in ['Zeff']:
            for sys, stat in options['systems']['CER VB']:
                if stat.get(): systems.append('CER VB '+sys[:4])
          
     
        data = []
        ts = None
        if quantity == 'ne' and options['systems']['DCN'][0][1].get():
            data.append(self.load_dcn(tbeg, tend))

        if quantity in ['Te', 'ne']  and len(systems) > 0:
            ts = self.load_ts(tbeg, tend, systems, options['load_options'].get('TS system',None))
            if quantity == 'ne' and options['systems']['DCN interf.'][1][1].get():
                ts = self.dcn_correction(ts, tbeg, tend)
            data.append(ts)
 
        if quantity in ['Ti', 'omega'] and len(systems) > 0:
            data.append(self.load_cer(tbeg,tend, systems,options['load_options'].get('CER system',None)))
            
        if quantity in ['nimp'] and len(systems) > 0:
            data.append(self.load_nimp(tbeg,tend, systems, options['load_options'].get('CER system',None)))
 
        if quantity in ['Te'] and (options['systems']['ECE system'][0][1].get() or options['systems']['ECE system'][1][1].get()):
            data.append(self.load_ece(tbeg,tend,options,ts))
        
        if quantity in ['Zeff']:
            data.append(self.load_zeff(tbeg,tend,systems, options['load_options']))
       
        
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


        if len(output['diag_names']) == 0:
            tkinter.messagebox.showerror('No data to load','At least one diagnostic must be selected')
            return 
        
        
        output['tres']=np.median(np.round(np.diff(np.hstack(times))*1e3,1))/1e3
        output['tres']=round(output['tres'],6)
        if output['tres'] == 0:    output['tres'] = 0.01
        output['rho_lbl'] = self.rho_coord
         
        return output



    def load_ts(self, tbeg,tend,systems, options=None):
    
        T = time()

        zshift = 0 
        if options is not None:
            if 'TS position error' in options:
                zshift = options['TS position error']['Z shift [cm]'].get()/100. #[m] 
  
        #use cached data
        self.RAW.setdefault('TS',{})
        ts = self.RAW['TS'].setdefault(revision,{'systems':systems})

        ts['systems'] = list(systems)
        systems = list(set(systems)-set(ts.keys()))
        
        
        #update mapping of the catched data
        ts = self.eq_mapping(ts, dz =zshift )            
        ts.setdefault('diag_names',{})

        if len(systems) == 0:
            #assume that equilibrium could be changed
            return ts

        print_line( '  * Fetching TS data ...')

        #extract setting from GUI 
        
        #VTN old diagnsotci is not supported!
 
         if self.dd.Open('VTA',self.shot):
            print('TS is not availible')
            return None
        
        for sys in systems
            if sys == 'core':
                tvec = self.dd.GetTimebase('TIM_CORE')
                Z = self.dd.GetSignal('Z_core')
                R = self.dd.GetSignal('R_core')
                ne = self.dd.GetSignal('Ne_c')
                Te = self.dd.GetSignal('Te_c')
                Te_err = (self.dd.GetSignal('Teupp_c')+ self.dd.GetSignal('Telow_c'))/2
                ne_err = (self.dd.GetSignal('Neupp_c')+ self.dd.GetSignal('Nelow_c'))/2
                laser = self.dd.GetSignal('Laser_c')  
            if sys == 'edge':
                tvec = self.dd.GetTimebase('TIM_EDGE')
                Z = self.dd.GetSignal('Z_edge')
                R = self.dd.GetSignal('R_edge')
                ne = self.dd.GetSignal('Ne_e')
                Te = self.dd.GetSignal('Te_e')
                Te_err = (self.dd.GetSignal('Teupp_e')+ self.dd.GetSignal('Telow_e'))/2
                ne_err = (self.dd.GetSignal('Neupp_e')+ self.dd.GetSignal('Nelow_e'))/2
                laser = self.dd.GetSignal('Laser_e') 
                
            
            Te_err[(Te_err<=0) | (Te <=0 )]  = -np.infty
            ne_err[(ne_err<=0) | (ne <=0 )]  = -np.infty
            channel = np.arange(Te_err.shape[1])

            
            rho = self.eqm.rz2rho(R,Z+zshift,tvec,self.rho_coord)
         
            ts[sys] = xarray.Dataset(attrs={'system':sys})
            ts[sys]['ne'] = xarray.DataArray(ne,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
            ts[sys]['ne_err'] = xarray.DataArray(ne_err,dims=['time','channel'], attrs={'units':'m^{-3}'})
            ts[sys]['Te'] = xarray.DataArray(Te,dims=['time','channel'], attrs={'units':'eV','label':'T_e'})
            ts[sys]['Te_err'] = xarray.DataArray(Te_err,dims=['time','channel'], attrs={'units':'eV'})
            ts[sys]['diags']= xarray.DataArray( np.tile(('TS:'+sys,), ne.shape),dims=['time','channel'])            
            ts[sys]['R'] = xarray.DataArray(R, dims=['channel'], attrs={'units':'m'})
            ts[sys]['Z'] = xarray.DataArray(Z,dims=['channel'], attrs={'units':'m'})
            ts[sys]['laser'] = xarray.DataArray(laser,dims=['time'], attrs={'units':'-'})
            ts[sys]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
            ts[sys]['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
            ts[sys]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})
            
        self.dd.Close()

        print('\t done in %.1fs'%(time()-T))
        ts['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':zshift, 'ed':self.eqm.diag}

        return ts 
        
        
    def load_IDZ(prof):
        #Zeff from IDA
        
        
        #use cached data
        ida = self.RAW.setdefault('IDZ',{})
        ida['systems'] = ['IDZ']
        if 'IDZ' in ida:
            return ida
 
        print_line( '  * Fetching IDZ data ...')


        if self.dd.Open('IDZ',self.shot):
            print('IDZ is not availible')
            return None

        tvec = self.dd.GetTimebase('time')
        rhop = self.dd.GetSignalGroup('rhop')
        Zeff = self.dd.GetSignalGroup('Zeff')
        Zeff_err = self.dd.GetSignalGroup('Zeff_unc')
        self.dd.Close()
        
        rho = self.eqm.rz2rho(rhop,tvec,coord_in=self.rho_coord,coord_out='rho_tor')

        ida['diag_names'] = {}
        ida['diag_names']['IDA'] = ['IDA']
        ida['IDA'] = xarray.Dataset( )
        ida['IDA']['Zeff'] = xarray.DataArray(Zeff,dims=['time','channel'], attrs={'units':'-','label':'Z_\mathrm{eff}'})
        ida['IDA']['Zeff_err'] = xarray.DataArray(Zeff_err ,dims=['time','channel'] )
 
        ida['IDA']['diags']= xarray.DataArray(np.tile(('IDZ',), ne.shape),dims=['time','channel'])
        ida['IDA']['rho']  = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
        ida['IDA']['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})

 
        print('\t done in %.1fs'%(time()-T))
        ida['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0, 'ed':self.eqm.diag}

    def load_IDA(prof):
        
        
        #use cached data
        ida = self.RAW.setdefault('IDA',{})
        ida['systems'] = ['IDA']
        if 'IDA' in ida:
            return ida
 
        print_line( '  * Fetching IDA data ...')


        if self.dd.Open('IDA',self.shot):
            print('IDA is not availible')
            return None

        tvec = self.dd.GetTimebase('time')
        
        if self.rho_coord == 'rhop':
            rho = self.dd.GetSignalGroup('rhop')
            rho = np.tile(rho, (len(tvec),1))
        elif self.rho_coord == 'rhot':
            rho = self.dd.GetSignalGroup('rhot')
        else:
            raise Exception('Coordinate %s is mot implemented for IDA'%self.rho_coord)
            
        ne = self.dd.GetSignalGroup('ne')
        Te = self.dd.GetSignalGroup('Te')
        
        ne_err = self.dd.GetSignalGroup('ne_unc')
        Te_err = self.dd.GetSignalGroup('Te_unc')
       
        self.dd.Close()
        
        
        ida['diag_names'] = {}
        ida['diag_names']['IDA'] = ['IDA']
        ida['IDA'] = xarray.Dataset( )
        ida['IDA']['ne'] = xarray.DataArray(ne,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
        ida['IDA']['ne_err'] = xarray.DataArray(ne_err ,dims=['time','channel'] )
        ida['IDA']['Te'] = xarray.DataArray(Te,dims=['time','channel'], attrs={'units':'eV','label':'T_e'})
        ida['IDA']['Te_err'] = xarray.DataArray(Te_err ,dims=['time','channel'] )
              
        ida['IDA']['diags']= xarray.DataArray(np.tile(('IDA',), ne.shape),dims=['time','channel'])
        ida['IDA']['rho']  = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
        ida['IDA']['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})

 
        print('\t done in %.1fs'%(time()-T))
        ida['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0, 'ed':self.eqm.diag}


    def load_LiBeam(prof):
        
        
        #use cached data
        lib = self.RAW.setdefault('LIN',{})
        lib['systems'] = ['LIN']
    
        if 'LIN' in lib:
            return self.eq_mapping(lib)            

        print_line( '  * Fetching Litium Beam data ...')


        if self.dd.Open('LIN',self.shot):
            print('LIN is not availible')
            return None

        tvec = self.dd.GetTimebase('time')
        
        R = self.dd.GetAreabase('R')
        z = self.dd.GetAreabase('Z')

        ne = self.dd.GetSignalGroup('ne')
        ne_err = self.dd.GetSignalGroup('ne_unc')
        
        rho = self.eqm.rz2rho(R,z,tvec,self.rho_coord)

        
        lib['diag_names'] = {}
        lib['diag_names']['IDA'] = ['IDA']
        lib['IDA'] = xarray.Dataset( )
        lib['IDA']['ne'] = xarray.DataArray(ne,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
        lib['IDA']['ne_err'] = xarray.DataArray(ne_err ,dims=['time','channel'] )
        lib['IDA']['diags']= xarray.DataArray(np.tile(('IDA',), ne.shape),dims=['time','channel'])
        lib['IDA']['rho']  = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
        lib['IDA']['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})

 
        print('\t done in %.1fs'%(time()-T))
        lib['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0, 'ed':self.eqm.diag}

        return 
    


    def load_cer(self,tbeg,tend, systems, options=None):
        #load Ti and omega at once
        T = time()

        tree = 'IONS'

            
        edition = 0
         
        self.RAW.setdefault('CER',{})
        cer = self.RAW['CER'].setdefault(edition,{})

        #load from catch if possible
        cer.setdefault('diag_names',{})
        cer['systems'] = systems
        load_systems = list(set(systems)-set(cer.keys()))
 
        #update equilibrium for already loaded systems
        cer = self.eq_mapping(cer)
        

        if len(load_systems) == 0:
            return cer
   
        print_line( '  * Fetching CXRS data ...' )
        
        
        
        for sys in load_systems:
            if self.dd.Open(sys,self.shot):
                tvec = self.dd.GetTimebase('time')
                names = self.dd.GetNames()
                
                if 'Rtime' in names:
                    R = self.dd.GetAreabase('Rtime')
                    Z = self.dd.GetAreabase('Ztime')
                else:
                    print('Watning: time variation in RmZ positions of measurements is not availible')
                    R = self.dd.GetAreabase('R')
                    Z = self.dd.GetAreabase('Z')
                    R = np.tile(R,(1,len(tvec)))
                    Z = np.tile(Z,(1,len(tvec)))
                    
                if 'Ti_c' in names:
                    Ti = self.dd.GetSignal('Ti_c')
                    Ti_err = self.dd.GetSignal('err_Ti_c')
                else:
                    Ti = self.dd.GetSignal('Ti')
                    Ti_err = self.dd.GetSignal('err_Ti')
                    
                if 'vr_c' in names:
                    omg = self.dd.GetSignal('vr_c')/R
                    omg_err = self.dd.GetSignal('err_vr_c')/R
                else:
                    omg = self.dd.GetSignal('vtor')/R
                    omg_err = self.dd.GetSignal('err_vtor')/R
                    
                      
        
                Ti_err[(Ti_err<=0) | (Ti <=0 )]  = -np.infty
                vtor_err[(vtor_err<=0) ]  = -np.infty
                channel = np.arange(Ti.shape[1])

                
                rho = self.eqm.rz2rho(R,Z+zshift,tvec,self.rho_coord)
            
                cer[sys] = xarray.Dataset(attrs={'system':sys})
                cer[sys]['Ti'] = xarray.DataArray(Ti,dims=['time','channel'], attrs={'units':'eV','label':'T_i'})
                cer[sys]['Ti_err'] = xarray.DataArray(Ti_err,dims=['time','channel'] )
                cer[sys]['omega'] = xarray.DataArray(omg,dims=['time','channel'], attrs={'units':'eV','label':'T_e'})
                cer[sys]['omega_err'] = xarray.DataArray(omg_err,dims=['time','channel'], attrs={'units':'eV'})
                cer[sys]['diags']= xarray.DataArray( np.tile(('CXRS:'+sys,), Ti.shape),dims=['time','channel'])            
                cer[sys]['R'] = xarray.DataArray(R, dims=['time','channel'], attrs={'units':'m'})
                cer[sys]['Z'] = xarray.DataArray(Z,dims=['time','channel'], attrs={'units':'m'})
                cer[sys]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
                cer[sys]['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
                cer[sys]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})
                
            self.dd.Close()

        cer['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}

        print('\t done in %.1fs'%(time()-T))
        


        
    
    def load_dcn(self, tbeg,tend, calc_weights=True):
        
        T = time()

         
        DCN = self.RAW.setdefault('DCN',{})

        #update mapping of the catched data
        if 'systems' in DCN and (not calc_weights or 'weight' in DCN[DCN['systems'][0]]):
            DCN = self.eq_mapping(DCN) 
            #return catched data 
            return DCN
        
        
        DCN['systems'] = ['DCN']
        DCN.setdefault('diag_names',{})
        DCN['diag_names']['DCN'] = ['DCN interferometer']
        DCN['DCN'] = []

        print_line( '  * Fetching DCN interferometer data ...')
        
 
        
        if self.dd.GetParameter('DCNgeo','H5LFSphi') is None:
            raise Exception('DCN geometry is not avalible')
        
        LOSs = ['H-1','H-2','H-0','H-4','H-5','V-3','V-2','V-1']
        signals = [s for s in self.dd.GetNames() if s[0] in ('H',)] 
        signals.sort()
        
        R_start = {s:self.dd.GetParameter('DCNgeo','H%sLFS_R'%s[2])/1e3 for s in signals}
        z_start = {s:self.dd.GetParameter('DCNgeo','H%sLFS_z'%s[2])/1e3 for s in signals}
        R_end   = {s:self.dd.GetParameter('DCNgeo','H%sHFS_R'%s[2])/1e3 for s in signals}
        z_end   = {s:self.dd.GetParameter('DCNgeo','H%sHFS_z'%s[2])/1e3 for s in signals}
        dd.Close()

        
        #values from diaggeom
        R_start['V-1'] = 1.785
        R_end['V-1'] =  1.785
        z_start['V-1'] =  1.200
        z_end['V-1'] = -1.200
        
        R_start['V-2'] = 1.200
        R_end['V-2'] =  1.200
        z_start['V-2'] =  1.200
        z_end['V-2'] = -1.200
        
        
        R_start['V-3'] = 1.135
        R_end['V-3'] =  1.135
        z_start['V-3'] =  1.200
        z_end['V-3'] = -1.200
        
        los_names = list(z_end.keys())

        if not self.dd.Open('DCK',self.shot):#fridge corrected
            if not self.dd.Open('DCN',self.shot):
                printe('DCN is not availible')
                    return None

        #Fetch data
        tvec = self.dd.GetTimebase('time')
        ne_ = [self.dd.GetSignal(key) for los in los_names]
            
        self.dd.Close()
        
            
        Rlcfs,Zlcfs = self.eqm.rho2rz(0.995)
        n_path = 501
        downsample = 5 
        n_ch = len(los_names)

        t = np.linspace(0,1,n_path, dtype='single')        
        tvec = np.mean(tvec[:len(tvec)//downsample*downsample].reshape(-1,downsample),-1)
        
        nt = len(tvec)

        ne     = np.zeros((nt, n_ch), dtype='single')
        ne_err = np.zeros((nt, n_ch), dtype='single')
          
        
        R = np.zeros((n_ch,n_path), dtype='single')
        Z = np.zeros((n_ch,n_path), dtype='single')
        L = np.zeros((n_ch,n_path), dtype='single')
        L_cross = np.zeros((nt,n_ch), dtype='single')
        weight= np.zeros((nt, n_ch, n_path), dtype='single')
        
        #downsample 10x
        for ilos, name in enumerate(los_names):
            ne[:,ilos] = np.mean(ne_[ilos][:len(ne_[ilos])//downsample*downsample].reshape( -1, downsample), -1)
        
            #Calculate the error for the signal based on the drift before t=0
            ne_err[:,ilos] = np.median(np.abs(ne[:,ilos][tvec<0]))+ne[:,ilos]*.05  #guess 5% error

            #Calculate the error for the signal based on the drift before t=0
            R[ilos] = (R_end[name]-R_start[name])*t+R_start[name] 
            Z[ilos] = (z_end[name]-z_start[name])*t+z_start[name] 
            L[ilos] = np.hypot(R[ilos]-R_start[name],Z[ilos]-z_start[name] )
      
            if calc_weights:
                dL = np.gradient(L[ilos])
                LOS_pt1 = np.array((R_start[name],z_start[name] ))
                #length of crossection over plasma
                L_cross_ = np.zeros(len(Rlcfs), dtype='single')
                for it,(r_,z_) in enumerate(zip(Rlcfs,Zlcfs)):
                    r,z = r_[0], z_[0]
                    A = ((r[1:]-r[:-1]),(z[1:]-z[:-1])),((-R_end[name]+R_start[name])*np.ones(len(z)-1),(-R_end[name]+z_start[name])*np.ones(len(z)-1))
                    A = np.array(A).T
                    b = np.array((v-r[:-1],z_start[name] -z[:-1])).T
                    
                    try:
                        X = np.linalg.solve(A,b)
                    except:
                        continue
                    
                    cross_ind = (X[:,0]>0)&(X[:,0]<=1)
                    if len(X) > 0 and any(cross_ind):
                        crossings = -np.outer(A[0,:,1],X[cross_ind,1]).T+LOS_pt1
                        L_cross_[it] = np.linalg.norm(np.diff(crossings,axis=0))
                
                #L_cross is length over the plasma - just a normalisation for nice plotting 
                L_cross[:,ilos] = np.interp(time, self.eqm.t_eq ,L_cross_)
                L_cross[:,ilos] = np.maximum(L_cross[:,ilos], .1) #just to avoid zero division
                L_cross[:,ilos]*= .9 # correction just for better plotting of the data 
                #convert from m^-2 -> m^-3
                weight[:,ilos,:] = dL/L_cross[:,ilos][:,None]
                ne[:,ilos] /= L_cross[:,ilos]
                ne_err[:,ilos] /= L_cross[:,ilos]
               
   
        #remove offset
        ne  -= ne[(time > -2)&(time < 0)].mean(0)   
           
        DCN['DCN'] = xarray.Dataset()
        DCN['DCN']['channel'] = xarray.DataArray( los_names ,dims=['channel'])
        DCN['DCN']['path'] = xarray.DataArray( t ,dims=['path'])
        DCN['DCN']['time'] = xarray.DataArray( time ,dims=['time'], attrs={'units':'s'})
        #DCN['DCN']['valid'] = xarray.DataArray( valid ,dims=['time','channel'], attrs={'units':'s'})
        DCN['DCN']['ne'] = xarray.DataArray(ne, dims=['time', 'channel'], attrs={'units':'m^{-3}','label':'n_e'})
        DCN['DCN']['ne_err'] = xarray.DataArray(ne_err,dims=['time', 'channel'], attrs={'units':'m^{-3}'})
        DCN['DCN']['diags']= xarray.DataArray( np.tile(('DCN interferometer',), (nt, n_ch)),dims=['time', 'channel'])
        DCN['DCN']['R'] = xarray.DataArray(R[None],dims=['','channel','path'],attrs={'units':'m'})
        DCN['DCN']['Z'] = xarray.DataArray(Z[None],dims=['','channel','path'],attrs={'units':'m'})
        if not 'rho' in DCN['DCN']:
            DCN['DCN']['rho'] = xarray.DataArray(np.zeros((nt, n_ch, n_path), dtype='single'),dims=['time', 'channel','path'])
        DCN['DCN']['L'] = xarray.DataArray(L,dims=['channel','path'],attrs={'units':'m'})
        DCN['DCN']['L_cross']  =  xarray.DataArray(1)


     
        if 'EQM' in DCN: DCN.pop('EQM')
        
        DCN = self.eq_mapping(DCN) 
        rho_tg = DCN['DCN']['rho'].values.min(2)
        if calc_weights:
            weight[DCN['DCN']['rho'].values >= 1.05] = 0 #make is consistent with DCN correction
            DCN['DCN']['weight']  = xarray.DataArray(weight, dims=['time', 'channel','path'])
            DCN['DCN']['L_cross'] = xarray.DataArray(L_cross,dims=['time','channel'],attrs={'units':'m'})

        DCN['DCN']['rho_tg'] = xarray.DataArray(rho_tg,dims=['time', 'channel'])


        DCN['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}

        print('\t done in %.1fs'%(time()-T))

        return DCN


    
    def dcn_correction(self,TS, tbeg,tend):
        
        T = time()
        if not 'core' in TS or not 'tangential' in TS:
            print('DCN correction could not be done')
            return TS
  
        #BUG ignoring user selected wrong channels of TS 
        T = time()
        
        DCN = self.load_dcn(tbeg,tend, calc_weights = False)['DCN']
        
        print_line( '  * Calculate TS correction using DCN ...')

        tbeg = max(tbeg, DCN['time'][0].values)
        tend = min(tend, DCN['time'][-1].values)
   

        tvec_compare =  TS['core']['time'].values
        laser_index  =  TS['core']['laser'].values
        t_ind = (tvec_compare > tbeg) & (tvec_compare < tend)
        ind = np.argsort(tvec_compare[t_ind])
        tvec_compare = tvec_compare[t_ind][ind]
        laser_index  = laser_index[t_ind][ind]
 
       
        LOS_rho = interp1d(DCN['time'].values, DCN['rho'].values,copy=False, axis=0,
                           assume_sorted=True)(np.clip(tvec_compare,*DCN['time'].values[[0,-1]]))
        LOS_L = DCN['L'].values

        
        #1)  correction of the tang system with respect to core

        #tang_tvec = TS['tangential']['time'].values
        #ind = slice(*tang_tvec.searchsorted((tbeg, tend)))
        #tang_tvec = tang_tvec[ind]
        #tang_lasers  = TS['tangential']['laser'].values[ind]
        #tang_lasers_list = np.unique(tang_lasers)
        #tang_ne  = TS['tangential']['ne'].values[ind]
        #tang_err  = TS['tangential']['ne_err'].values[ind]
        #tang_rho = TS['tangential']['rho'].values[ind]
        
        
        #core_rho= interp1d(TS['core']['time'],TS['core']['rho'],axis=0,copy=False, assume_sorted=True)(tang_tvec)
        #core_ne = interp1d(TS['core']['time'],TS['core']['ne'],axis=0,copy=False, assume_sorted=True)(tang_tvec)
        #core_err= interp1d(TS['core']['time'],TS['core']['ne_err'],axis=0,copy=False, assume_sorted=True,kind='nearest')(tang_tvec)

        #remove corrupted channels
        #corrupted_tg = np.sum(~np.isfinite(tang_err),0)>tang_err.shape[0]/5.
        #corrupted_core = np.sum(~np.isfinite(core_err),0)>core_err.shape[0]/5.
        
 
        #rho = np.hstack((tang_rho[:,~corrupted_tg],core_rho[:,~corrupted_core]))
        #rho_sort_ind  = np.argsort(rho,axis=1)
        #tang_lasers = tang_lasers[~corrupted_tg]
        #core_ne = core_ne[:,~corrupted_core]

        
        #time_sort_ind = np.tile(np.arange(rho.shape[0]), (rho.shape[1], 1)).T
        
        #rho = rho[time_sort_ind, rho_sort_ind]
        #def cost_fun(corr):
            #tang_ne_ = np.double(tang_ne[:,~corrupted_tg])
            #for l,c in zip(tang_lasers_list, corr):
                #tang_ne_[tang_lasers==l] *= np.exp(c)
            #ne  = np.hstack((tang_ne_,core_ne ))/1e19
            #ne  = ne[time_sort_ind, rho_sort_ind]
            #val = np.nansum((np.diff(ne)/(ne[:,1:]+ne[:,:-1]+.1))**2)
            #return val
            
        #from scipy.optimize import minimize 
   
        #p0 = np.zeros_like(tang_lasers_list)
        #opt = minimize(cost_fun,  p0, tol = .0001 )
        #corr = np.exp(opt.x)
 
        
        TS = deepcopy(TS)
        
        ##correct core system
        #for l,c in zip(tang_lasers_list , corr):
            #TS['tangential']['ne'].values[TS['tangential']['laser'].values == l] *= c
   

        #interpolate data on the time of the core system
        N,R = [],[]
        for diag in ['core','edge']:
            t,n,e,r = TS[diag]['time'].values, TS[diag]['ne'].values, TS[diag]['ne_err'].values, TS[diag]['rho'].values
            nchs = n.shape[1]
            n_interp = np.zeros((len(tvec_compare), n.shape[1]))
            for nch in range(nchs):
                ind = (n[:,nch] > 0) & (e[:,nch] > 0) & np.isfinite(e[:,nch]) &(n[:,nch] < 1.5e20)  
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
        core_lasers = np.unique(laser_index)
        
    
        #weight = interp1d(DCN['time'].values, DCN['weight'].values*DCN['L_cross'].values[:,:,None] ,copy=False, axis=0,
                #assume_sorted=True)(np.clip(tvec_compare,*DCN['time'].values[[0,-1]]))
              

        dcn_los_names  = DCN['channel'].values
        laser_correction     = np.ones((len(core_lasers),len(dcn_los_names)))*np.nan
        laser_correction_err = np.ones((len(core_lasers),len(dcn_los_names)))*np.nan
        los_dict =  {n:i for i,n in enumerate(dcn_los_names)}
        valid = DCN['valid'].values
        time_dcn = DCN['time'].values
        
        #import matplotlib.pylab as plt
        ne = DCN['ne'].values*DCN['L_cross'].values 
        #plt.plot(tvec_compare,np.sum( LOS_ne* weight,2 ) )
        #plt.plot(tvec_compare,LOS_ne_int,'--' )
        #plt.plot(time_co2, ne,':' );plt.show()


        for il, l in enumerate(core_lasers):
            ind = laser_index == l
  
            for ilos,los in enumerate( dcn_los_names):
                if not np.any(valid[:,ilos]): continue
                t_ind = (tvec_compare > time_dcn[valid[:,ilos]][0]) & (tvec_compare < time_dcn[valid[:,ilos]][-1])
                if not np.any(ind&t_ind): continue
                ratio = LOS_ne_int[ind&t_ind ,ilos]/np.interp(tvec_compare[ind&t_ind],time_dcn[valid[:,ilos]],ne[valid[:,ilos], ilos])
                laser_correction[il,ilos] = np.median(ratio)
                laser_correction_err[il,ilos] = ratio.std()/np.sqrt(len(ratio))
                    
        mean_laser_correction = np.nanmean(laser_correction,1)
              

        plot = False
        if plot:
            try:
                import matplotlib.pylab as plt

                #show comparims between TS and DCN
                interf_fig = plt.figure(figsize=(14,10))

                ax1 = interf_fig.add_subplot(211)
                for ilos, name in enumerate(dcn_los_names):
                    p, = ax1.plot(time_dcn,ne[:,ilos], label=name,lw=.5)
                    ax1.plot(time_dcn[valid[:,ilos]],ne[valid[:,ilos],ilos],c=p.get_color())
                ax1.axvline(tbeg)
                ax1.axvline(tend)

                ax1.set_prop_cycle(None)
                ax1.set_ylabel(r'$\langle n_e \rangle \mathrm{\ [m^{-2}]}$')
                ax1.set_xlabel('time [s]')
                ax1.plot(tvec_compare,LOS_ne_int,'--' )
                ax1.grid('on')
                leg = ax1.legend(loc='best')
                for l in leg.legendHandles:  l.set_linewidth(4)


                ax2 = interf_fig.add_subplot(212)
                colors = matplotlib.cm.brg(np.linspace(0,1,len(core_lasers)))                
                ax2.axhline(y=1,c='k')
                ax2.set_ylabel(r'$\langle n_e \rangle_\mathrm{TS}/\langle n_e \rangle_\mathrm{DCN} $')
                ax2.set_xlim(-0.5,len(dcn_los_names)-0.5)

                for il, l in enumerate(core_lasers):
                    ax2.plot([],[],'o',c=colors[il],label='laser %d'%l)
                    for ilos,name in enumerate(dcn_los_names):
                        ax2.errorbar(ilos+(il-5)/10.,laser_correction[il,ilos],
                                        laser_correction_err[il,ilos],fmt='o',c=colors[il])
                    ax2.plot(np.arange(len(dcn_los_names))+(il-5)/10., 
                                laser_correction[il]/mean_laser_correction[il],'x',c=colors[il])

                leg = ax2.legend(loc='best',numpoints=1)
                for l in leg.legendHandles:  l.set_linewidth(4)
            
                #co2_los_names
                
                ax2.set_xticks(np.arange(len(dcn_los_names)), dcn_los_names.tolist())#, rotation='vertical')
                locs = ax2.set_xticks(list(range(len(dcn_los_names))))
                labels = ax2.set_xticklabels( dcn_los_names )
                ax1.set_prop_cycle(None)    
                ax2.axvline(tbeg)
                ax2.axvline(tend)

                interf_fig.savefig('dcn_correction_%d.pdf'%self.shot)

                interf_fig.clf()
                plt.close()
                del interf_fig
            except:
                pass
        
        

         #correction of laser intensity variation and absolute value
        print('DCN corrections:', np.round(mean_laser_correction,3))
        for sys in ['core', 'edge']:
            laser = TS[sys]['laser'].values
            data = TS[sys]['ne'].values
            #total correction of all data
            data/= np.mean(mean_laser_correction)
            for l_ind, corr in zip(core_lasers,mean_laser_correction):
                #laser to laser variation from of the core system
                data[laser == l_ind] /= corr/np.mean(mean_laser_correction)
                
                    
        print('\t done in %.1fs'%(time()-T))

        return TS 
    
    
        
    
    def load_elms(self,option):
        print_line( '  * Fetching ELM data... ')
        T = time()
        #node = option['elm_signal'].get()
         
        self.RAW.setdefault('ELMS',{})
        
        if 'data' in self.RAW['ELMS']:
            return self.RAW['ELMS']['data']

     
        if not self.dd.Open('ELM',self.shot):#fridge corrected
    
            #Fetch data
            elm_start = self.dd.GetTimebase('t_begELM')
            elm_end = self.dd.GetTimebase('t_endELM')

            self.dd.Close()
            
            val = np.ones_like(elm_start)
            elm_val = np.c_[val, -val,val*0 ].flatten()
            t_elm_val = tvec[np.c_[ elm_start-1, elm_start, elm_end].flatten()]
  
        else:
            elm_time, elm_val, elm_beg, elm_end = [],[],[],[]

            
        self.RAW['ELMS']['data'] =  {'tvec': elm_time, 'data':elm_val, 
                     'elm_beg':elm_beg,'elm_end':elm_end  }
        
        print('\t done in %.1fs'%(time()-T))
        return  self.RAW['ELMS'][node]

    
    def load_sawteeth(self):
        #BUG how to detect them?
  
        return {'tvec':[]}
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
    def load_ece(self, tbeg,tend,options,TS=None):
    
        T = time()
        
        fast =  bool(options['systems']['ECE system'][1][1].get())
        bt_correction = float(options['load_options']['ECE system']["Bt correction"]['Bt *='].get())
        ts_correction = bool(options['load_options']['ECE system']["TS correction"]['rescale'].get())

        rate = 'fast' if fast else 'slow'
        suffix = 'F' if fast else ''

        #use cached data
        self.RAW.setdefault('ECE',{})
        self.RAW['ECE'].setdefault(rate,{})

        
        self.RAW['ECE'][rate].setdefault('ECE',xarray.Dataset(attrs={'system':rate } ))

        ece = self.RAW['ECE'][rate]['ECE']
        self.RAW['ECE'][rate]['diag_names'] = {'ECE':['ECE']}
        self.RAW['ECE'][rate]['systems'] = ['ECE']

        if not 'Te_raw' in ece:
            print_line( '  * Fetching ECE%s radiometer data ...'%suffix )
   
            tree = 'ECE'
    
            # set up strings to use in mds loading
            nodes = '\\'+tree+'::TOP.SETUP.'
            nodec = '\\'+tree+'::TOP.CALF:'
            noded = '\\'+tree+'::TOP.TECEF'
            subnodes_nch = nodec +  'NUMCHF'
            subnodes_z = nodes + 'ECEZH'
            subnodes_freq = nodes + 'FREQ'
            subnodes_valid = nodec + 'VALIDF'
            
            
            try:
                self.MDSconn.openTree(tree,self.shot)

                # get frequency settings
                freq = self.MDSconn.get(subnodes_freq).data()*1e9
                
                valid =  np.bool_(self.MDSconn.get(subnodes_valid).data())

                # get position settings
                z = self.MDSconn.get(subnodes_z).data()
                    
                if self.shot < 109400:
                    nchs = 32
                else:
                    nchs = self.MDSconn.get(subnodes_nch).data()
            
            except Exception as e:
                printe( 'MDS error: '+ str(e))
            finally:
                self.MDSconn.closeTree(tree, self.shot)
  
            TDI =  [r"\TECE%s%02d"%(suffix, d) for d in range(1,1+nchs)]
            TDI += [r'dim_of(\TECE%s01)'%suffix, ]
            
        
            out = mds_load(self.MDSconn, TDI, tree, self.shot)

            tvec,data_ = out[-1],out[:-1]
            tvec/=1e3
            
            data_ = np.vstack(data_).T
            data_ *= 1e3 #convert to eV units

            
            if fast:
                #fast downsample
                data_ = np.median(data_.T.reshape(nchs,  -1, 32), 2)#remove spikes
                data_ = np.mean(data_.reshape(nchs, -1, 4), -1).T
                tvec  = np.mean(tvec.reshape( -1, 32*4), -1)

            #remove offset
            data_-= data_[tvec<0].mean(0)[None,:]
                            
            #create dataset with raw data 
            channel = np.arange(nchs)

            ece['Te_raw'] = xarray.DataArray(data_, coords=[tvec, channel], dims=['time','channel'], attrs={'units':'eV'})
            ece['freq'] = xarray.DataArray(freq[:nchs], dims=['channel'], attrs={'units':'Hz'} )
            ece['valid'] = xarray.DataArray( np.r_[valid, np.zeros(nchs-len(valid), dtype=bool)], dims=['channel'], attrs={'units':'-'})
            ece['z0'] = xarray.DataArray(z, attrs={'units':'m'} )
            ece['channel'] = xarray.DataArray(channel, dims=['channel'], attrs={'units':'-'} )


                        
        #process only selected time range 
        ece = ece.sel(time=slice(tbeg,tend))

        nchs = len(ece['channel'])
        r_in = self.eqm.Rmesh
        z_in = ece['z0'].values
        t_ind = slice(self.eqm.t_eq.searchsorted(tbeg),self.eqm.t_eq.searchsorted(tend)+1)
        t_eq = self.eqm.t_eq[t_ind]
        B = self.eqm.rz2brzt(r_in,  z_in, t_eq)
        Btot = np.squeeze(np.linalg.norm(B,axis=0))
        #apply correction from GUI
        Btot *= bt_correction
        from scipy.constants import m_e, e, c,epsilon_0
        horiz_rho = self.eqm.rz2rho(r_in, z_in*np.ones_like(r_in),t_eq,coord_out=self.rho_coord)
        #Accounting for relativistic mass downshift
        zipfit = self.load_zipfit()
        

        
        try:
            Te_tvec = zipfit['Te']['tvec'].values
            Te_tvec[[0,-1]] = -10,100
    
            Te_ = interp1d(Te_tvec, zipfit['Te']['data'].values,axis=0,copy=False, assume_sorted=True)(t_eq)
            Te = np.zeros_like(horiz_rho)

            for it,t in enumerate(t_eq):
                Te[it] = np.interp(horiz_rho[it], zipfit['Te']['rho'].values, np.abs(Te_[it]))

            v=np.sqrt(2*Te*e/m_e)
            gamma = 1/np.sqrt(1-(v/c)**2)
        except:
            printe('relativistic mass downshift could not be done')
            gamma = np.ones((1,len(r_in)))

        wce = e*Btot/(m_e*gamma)

        nharm = 2
        R = np.zeros((len(t_eq),nchs))
        for it,t in enumerate(t_eq):
            R[it] = np.interp(-2*np.pi*ece['freq'],-wce[it]*nharm,r_in)
            
 

        r_lfs = self.eqm.rhoTheta2rz(1, 0, t_eq)[0].mean() #m 
        f_3harm_lfs = 3*wce[:,r_in.searchsorted(r_lfs)]/(2*np.pi) #freq above this value can be contaminated by 3. harmonics 
    
        #calculate ne_critical
        #everything in GHz 
        try:
    
            ne_ = zipfit['ne']['data'].values
            ne_err =  zipfit['ne']['data'].values*0.05 #zipfit errorbars are often wrong, while the fit is OK 
            ne_ += ne_err # upper boundary, to be sure that affected measurements will be removed
            ne_tvec = zipfit['ne']['tvec'].values
            ne = np.zeros((len(ne_tvec), len(r_in)))
            for it,t in enumerate(ne_tvec ):
                iteq = np.argmin(abs(t_eq-t))
                ne[it] = np.interp(horiz_rho[iteq], zipfit['ne']['rho'].values,  ne_[it])
            
            f_CE = interp1d(t_eq, wce/(2*np.pi*1e9),fill_value="extrapolate",
                            axis=0,copy=False, assume_sorted=True)(ne_tvec)  #f_ce at R
            f_PE = np.sqrt(np.maximum(ne,0)*e**2/m_e/epsilon_0)/(2*np.pi*1e9)#f_pe at R
            f_RHC=(0.5*f_CE)+np.sqrt((0.5*f_CE)**2 + f_PE**2)
            f_cut = np.maximum.accumulate(f_RHC[:,::-1], axis=1)[:,::-1]  #propagate maximum from LFS to HFS 
            f_cut_loc = np.zeros((len(ne_tvec), nchs))
            for it,t in enumerate(ne_tvec):
                iteq = np.argmin(abs(t_eq-t))
                f_cut_loc[it] = np.interp(R[iteq],  r_in, f_cut[it])

                
        except:
            printe( 'ZIPFIT ne data are missing, density cutoff was not estimated')
            f_cut_loc = np.zeros((2,nchs))
            ne_tvec = [0,10]

 
        #interpolate on the Te timescale
        tvec = ece['time'].values
        f_3harm_lfs = np.interp(tvec, t_eq, f_3harm_lfs)
        R         = interp1d(t_eq,R  ,axis=0,fill_value="extrapolate",
                             copy=False, assume_sorted=True)(tvec)
        f_cut_loc = interp1d(ne_tvec,f_cut_loc,axis=0,fill_value="extrapolate",
                             copy=False, assume_sorted=True)(tvec)
        z = ece['z0'].values*np.ones_like(R)
        rho = self.eqm.rz2rho(R,z,tvec,coord_out=self.rho_coord)

        data = ece['Te_raw'].values
        data_err = np.abs(data)*0.05+50.  #my naive guess of errorbars
        #it will be possible to manually return these removed points in GUI 
        mask = f_cut_loc*1e9> ece['freq'].values[None,:]
        mask |= f_3harm_lfs[:,None]  < ece['freq'].values[None,:]
        mask[:,~ece['valid'].values] |= True

        #masked errors will be negative, xarray do not support masked arrays 
        data_err[mask] *= -1
        

      
        if ts_correction:
            #rescale using TS data
            TS_systems = ['tangential','core']
            if TS is None:
                TS = self.load_ts(tbeg,tend,TS_systems)
            
            TS_rho,TS_Te,TS_err,TS_tvec = [],[],[],[]
            for s in TS_systems:
                
                TS_rho.append(TS[s]['rho'].values)
                TS_tvec.append(np.tile(TS[s]['time'].values,(len(TS[s]['channel']),1)).T)
                TS_Te.append(TS[s]['Te'].values)
                TS_err.append(TS[s]['Te_err'].values)            

            TS_rho = np.hstack([r.flatten() for r in TS_rho])
            TS_Te  = np.hstack([t.flatten() for t in TS_Te])
            TS_err = np.hstack([e.flatten() for e in TS_err])
            TS_tvec= np.hstack([t.flatten() for t in TS_tvec])
            
            valid = np.isfinite(TS_err)
            TS_rho, TS_Te,TS_tvec = TS_rho[valid], TS_Te[valid], TS_tvec[valid]
            #interpolate Te from TS on the time and radia positions of ECE
            NDinterp = NearestNDInterpolator(np.vstack([TS_tvec,TS_rho]).T,TS_Te ,rescale=True )

            #use only high Te part of the signals
            high_te_ind = data.max(1) > np.mean(data.max(1))/2
            high_te_ind = slice(*np.where(high_te_ind)[0][[0,-1]])
            
            TS_Te_ = NDinterp(np.vstack([np.tile(tvec[high_te_ind],(nchs,1)).T.flatten(),rho[high_te_ind].flatten()]).T)
            TS_Te_ = TS_Te_.reshape(data[high_te_ind].shape)
                          
            #apply correction
            data = deepcopy(data)
            for ch in range(nchs):
                valid = np.isfinite(data_err[high_te_ind,ch])
                if any(np.isfinite(valid)):
                    data[:,ch]*= np.median(TS_Te_[valid,ch])/np.median(data[high_te_ind,ch][valid])
            

        ece['Te'] = xarray.DataArray(data, dims=['time','channel'], attrs={'units':'eV','label':'T_e'} )
        ece['Te_err'] = xarray.DataArray(data_err, dims=['time','channel'], attrs={'units':'eV'} )
        ece['diags'] = xarray.DataArray(np.tile(('ECE',),data.shape), dims=['time','channel'], attrs={'units':'-'} )
        ece['R'] = xarray.DataArray(R.astype('single'), dims=['time','channel'], attrs={'units':'m'} )
        ece['Z'] = xarray.DataArray(z.astype('single'), dims=['time','channel'], attrs={'units':'m'} )
        ece['rho'] = xarray.DataArray(rho.astype('single'), dims=['time','channel'], attrs={'units':'-'} )

        print('\t done in %.1fs'%(time()-T))


        return {'ECE':ece,'diag_names':{'ECE':['ECE']}, 'systems':['ECE']
                                ,'EQM':{'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}}
    
    
    

    def load_nimp(self,tbeg,tend,systems, options):
        tree = 'IONS'

        selected,analysis_types = options['Analysis']
   
        rcalib = options['Correction']['Relative calibration'].get() 
        
        analysis_type = self.get_cer_types(selected.get(),impurity=True)
        
        Zimp = ''
        if 'Impurity Z' in options:
            Zimp = options['Impurity Z'][0].get()

        
        #load from catch if possible
        self.RAW.setdefault('nimp',{})
        nimp = self.RAW['nimp'].setdefault(analysis_type+Zimp ,{} )
        
        #which cer systems should be loaded
        load_systems = list(set(systems)-set(nimp.keys()))
         
        #rho coordinate of the horizontal line, used later for separatrix aligment 
        if 'horiz_cut' not in nimp or 'EQM' not in nimp or nimp['EQM']['id'] != id(self.eqm) or nimp['EQM']['ed'] != self.eqm.diag:
            R = np.linspace(1.8,2.5)
            rho_horiz = self.eqm.rz2rho(R, np.zeros_like(R), coord_out='rho_tor')
            nimp['horiz_cut'] = {'time':self.eqm.t_eq, 'rho': np.single(rho_horiz), 'R':R}
        

        if len(load_systems) == 0:
            #update equilibrium 
            nimp['systems'] = systems
            nimp = deepcopy(self.eq_mapping(nimp))
            #return correeced data if requested
            if rcalib:
                for ch in [ch for s in nimp['systems'] for ch in nimp[s]]:
                    if 'nimp_corr' in ch:
                        ch['nimp'] = ch['nimp_corr']
                        ch['nimp_err'] = ch['nimp_corr_err']
              
            return nimp

        print_line( '  * Fetching NIMP data ...')

        nimp = self.RAW['nimp'][analysis_type+Zimp]
        nimp.setdefault('diag_names',{})
        nimp['systems'] = systems
        #update equilibrium of catched channels
        nimp = self.eq_mapping(nimp)

        T = time()

        
        imp_path = '\%s::TOP.IMPDENS.%s.'%(tree,analysis_type) 
        nodes = 'IMPDENS', 'ERR_IMPDENS', 'R_IMPDENS', 'Z_IMPDENS', 'INDECIES', 'TIME','ZIMP'
        TDI = [imp_path+node for node in nodes]
        
        
        #array_order in the order as it is stored in INDECIES
        TDI += ['\IONS::TOP.CER.CALIBRATION:ARRAY_ORDER']

        #fast fetch
        out = mds_load(self.MDSconn, TDI, tree, self.shot)
        nz,nz_err, R,Z,ch_ind, tvec, zimp,array_order = out
        nz_err[(nz<=0)|(nz > 1e20)] = np.infty
        ch_ind = np.r_[ch_ind,len(tvec)]
        ch_nt = np.diff(ch_ind)
            
        try:
            array_order = [a.decode('utf-8') for a in array_order]
        except:
            pass
        array_order = [a[0]+a[4:].strip() for a in array_order]

        used_chan = np.array(array_order)[ch_nt[:len(array_order)]>0]
        
        TDI = []
        loaded_chan = []
        
        for sys in load_systems:
            nimp['diag_names'].setdefault(sys,[])
            nimp.setdefault(sys,[])

 
        for ch in used_chan:
            #load only requested systems
            system = 'tangential' if ch[0] == 'T' else 'vertical'
            if system not in load_systems:
                continue
            
            ich = array_order.index(ch)            
            ind = slice(ch_ind[ich],ch_ind[ich+1])
            
            #save data only for the selected impurity #BUG it shoudl fatch data only for thiss impurity 
            assert all(zimp[ind][0]==zimp[ind]), 'zimp[ind][0]!=zimp[ind]'
            if Zimp and zimp[ind][0]!= int(Zimp):
                continue
            

            loaded_chan.append(ch)

            path = '\\IONS::TOP.CER.%s.%s.CHANNEL%.2d'%(analysis_type,system,int(ch[1:]))
            TDI.append(path+':STIME')
            TDI.append(path+':TIME')
            TDI.append(path+':BEAMID')

            path = '\\IONS::TOP.CER.CALIBRATION.%s.CHANNEL%.2d'%(system,int(ch[1:]))
            nodes = 'LENS_PHI','LINEID','WAVELENGTH'
            TDI += [path+':'+n for n in  nodes]
                            

        #fast fetch
        out = mds_load(self.MDSconn, TDI, tree, self.shot)
        out = np.reshape(out,(-1,6))
        
        rho = self.eqm.rz2rho(R[:,None],Z[:,None],tvec/1e3,self.rho_coord)

 
        beamid_list = []
        #beamid is not avalible for shots < 167162
        for ch in loaded_chan:
            diag,ch_num = ch[0], int(ch[1:])
            if diag == 'V': #beamid is missing for vertical channels
                if ch_num in np.r_[1:7, 17:24]:
                    beamid_list += ['330L']
                else:
                    beamid_list += ['330R']
                    
            if diag == 'T':  #beamid is missing for HFS channels
                if ch_num in np.r_[1:8,17:23,33:37]: #LFS,HFS channels
                    beamid_list += ['30B']
                elif ch_num in np.r_[25:33, 37:41]: #LFS,HFS channels
                    beamid_list += ['210B']
                elif ch_num in np.r_[8:17, 23:25, 41:49]: #edge channels
                    beamid_list += ['330B']
                else:
                    print('unknow beam for channel '+ch)
        
        #load beam power timetraces only if necessary
        if self.shot <  167162 and ('330B' in beamid_list or '210B' in beamid_list or '30B' in beamid_list):
            beams = np.unique(beamid_list)
            load_beams = []
            for beam in beams:
                beam = beam[:2]+beam[-1]
                if 'B' in beam:
                    load_beams+= [beam[:-1]+'L', beam[:-1]+'R']
                else:
                    load_beams+= [beam]
            load_beams = list(np.unique(load_beams))
            TDI = ['\\NB::TOP.NB{0}:PINJ_{0}'.format(beam) for beam in load_beams] +['\\NB::TOP:TIMEBASE']
            PINJ = mds_load(self.MDSconn, TDI, 'NB', self.shot)
            beam_time = PINJ[-1]
            nbi_cum_pow = cumtrapz(np.double(PINJ[:-1]),beam_time,initial=0)
            
  
            
        #split to channels, create xarray.Datasets
        for ch,beam, (stime,t_cer,beam_cer, phi,lineid, lam) in zip(loaded_chan,beamid_list, out):

            #HFS 210 channel
            if len(lineid)==0:    lineid = np.array('C VI 8-7')
            if len(phi)==0:    phi = 0.
            if len(stime)==0:    stime = 5.
            
            ich = array_order.index(ch)
            
            ind = slice(ch_ind[ich],ch_ind[ich+1])

            stime = np.median(stime) #BUG can be changing in time!!
            tch = tvec[ind]+stime/2
            

            # List of chords with intensity calibration errors for FY15, FY16 shots after
            # CER upgraded with new fibers and cameras.
            disableChanVert = 'V3', 'V4', 'V5', 'V6', 'V23', 'V24'
            if 162163 <= self.shot <= 167627 and ch in disableChanVert:
                nz_err[ind] = np.infty
            if ch == 'T7' and self.shot >= 158695:
                nz[ind] *= 1.05
                
            if ch == 'T23' and   158695 <=  self.shot < 169546:
                nz[ind] *= 1.05
            if ch == 'T23' and  165322<=self.shot < 169546:
                nz[ind] *= 1.05
            
            #if ch == 'T23' and  self.shot == 157565:
                #nz[ind] *= 1.05
  
            lineid = lineid.item()
            if not isinstance(lineid,str):
                lineid = lineid.decode()
                
            beamid = np.zeros(len(tch), dtype='U4')

            diag = 'vertical' if ch[0] == 'V' else 'tangential'

            #beamid is missing for vertical channels
            if diag == 'vertical':
                if int(ch[1:]) in np.r_[1:7, 17:24]:
                    beamid[:] = '330L'
                else:
                    beamid[:] = '330R'
                
            elif diag == 'tangential':
                if (self.shot >=  167162): #beam power is not loaded
                    if int(ch[1:]) in np.r_[33:37,37:41]: #HFS channels
                        beamid[:] = beam
                    else:
                        #solve issues with extra timepoinst if tcer != tvec[ind]
                        t_cer = np.atleast_1d(np.int_(np.round(t_cer)))
                        beamid[:]= beam_cer[0] #guess for few missing timepoints
                        for i,t in enumerate(tvec[ind]):
                            j = t_cer.searchsorted(int(round(t)))
                            beamid[i] = beam_cer[min(j,len(beamid)-1)]
                            beamid[i] = beamid[i].lstrip('0')
                            
                else:#identify beams
                    sind = beam_time.searchsorted(np.hstack([tch-stime/2, tch+stime/2]))
                    
                    R_beam = load_beams.index(beam[:2]+'R')
                    L_beam = load_beams.index(beam[:2]+'L')

                    R_energy = nbi_cum_pow[R_beam,sind[len(tch):]]-nbi_cum_pow[R_beam,sind[:len(tch)]]
                    L_energy = nbi_cum_pow[L_beam,sind[len(tch):]]-nbi_cum_pow[L_beam,sind[:len(tch)]]
                    
                    beamid[L_energy > 1e5] = beam[:-1]+'L'
                    beamid[R_energy > 1e5] = beam[:-1]+'R'
                    beamid[np.maximum(L_energy,R_energy)/(L_energy+R_energy+1) < 0.9] = beam

            ##in some (older?) discharges is nt the beamid avalible
            #beamid = np.array(("%d"%phi,)*len(tch))

            
            tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', lineid)
            imp, charge, transition = tmp.group(1), tmp.group(2), tmp.group(3) 
            charge = roman2int(charge)            
            
            #split beam 30 system to the subsystems
            s = 'c' if int(phi) == 318 else ''  
            names = np.array([diag[0].upper()+'_'+ID.lstrip('0')+s+' '+imp+str(charge) for ID in beamid])
            unames,idx,inv_idx = np.unique(names,return_inverse=True,return_index=True)
            for name in unames:
                if not name in nimp['diag_names'][diag]:
                    nimp['diag_names'][diag].append(name)
            
            #split channels by beams
            for ID in np.unique(inv_idx):
                beam_ind = inv_idx == ID
                ds = xarray.Dataset(attrs={'channel':ch+'_'+beamid[idx[ID]], 'system': diag,'stime':stime})
                ds['nimp'] = xarray.DataArray(nz[ind][beam_ind], dims=['time'], 
                                        attrs={'units':'m^{-3}','label':'n_{%s}^{%d+}'%(imp,charge),'impurity':imp})
                ds['nimp_err']  = xarray.DataArray(nz_err[ind][beam_ind],dims=['time'], attrs={'units':'m^{-3}'})
                ds['R'] = xarray.DataArray(R[ind][beam_ind],dims=['time'], attrs={'units':'m'})
                ds['Z'] = xarray.DataArray(Z[ind][beam_ind],dims=['time'], attrs={'units':'m'})
                ds['rho'] = xarray.DataArray(rho[ind,0][beam_ind],dims=['time'], attrs={'units':'-'})
                 
                ds['diags']= xarray.DataArray(names[beam_ind],dims=['time'])
                ds['time'] = xarray.DataArray(np.single(tch[beam_ind]/1e3),dims=['time'], attrs={'units':'s'})
                nimp[diag].append(ds)
   

        nimp['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}
        ##update uncorrected data
        diag_names = sum([nimp['diag_names'][diag] for diag in nimp['systems']],[])
        
        
        impurities = np.hstack([[ch['nimp'].attrs['impurity'] for ch in nimp[s] if ch.system == s] for s in nimp['systems']])
        unique_impurities = np.unique(impurities)
       
        #reduce discrepancy between different CER systems
        if options['Correction']['Relative calibration'].get() and len(diag_names) > 1 and len(unique_impurities)==1:
    
            beams = [n[2:].split(' ')[0] for n in diag_names]
            load_beams = []
            for beam in beams:
                beam = beam[:2]+beam[-1]
                if 'B' in beam:
                    load_beams+= [beam[:-1]+'L', beam[:-1]+'R']
                else:
                    load_beams+= [beam]
            load_beams = list(np.unique(load_beams))

            voltages = {}
            times = {}
        
            if self.shot  > 169568:  #load time dependent voltage
                TDI = ['\\NB::TOP.NB%s:VBEAM'%beam for beam in load_beams] +['\\NB::TOP:TIMEBASE']
            else:   #laod scalar values
                TDI = ['\\NB::TOP.NB%s:NBVAC_SCALAR'%beam for beam in load_beams]
            V_data = mds_load(self.MDSconn, TDI, 'NB', self.shot)

            print(' ')
            for beam, V in zip(load_beams,V_data):
                turned_on = V>1e3
                if not any(turned_on):
                    continue

                voltages[beam] = V[turned_on].mean()/1e3
                second_beam = beam[:-1]+('R' if  beam[-1] == 'L' else 'L')
                if second_beam in load_beams:
                    turned_on &= V_data[load_beams.index(second_beam)] < 1e3
                if len(turned_on) > 1:
                    times[beam]  = np.sum(np.diff(V_data[-1])[turned_on[:-1]])/1e3
                else:
                    times[beam]  = np.inf 

                print('beam:%s  %dkV  single beam = %.2fs'%(beam,voltages[beam],times[beam]))

            #import IPython
            #IPython.embed()
                
            #if beam 30L is on ~80kV use it for cross-calibration
            if '30L' in beams and ( 77 < voltages['30L'] < 83) and times['30L'] > .5:
                print('Using beam 30L for cross calibration')
                calib_beam = '30L'

            elif '30R' in beams and (77 < voltages['30R'] < 83) and times['30R'] > .5:
                print('Using beam 30R for cross calibration')
                calib_beam = '30R'
                
            elif '30B' in beams and (( 77 < voltages['30R'] < 83) and ( 77 < voltages['30L'] < 83)):
                print('Using beam 30R+30L for cross calibration - less reliable')
                calib_beam = '30B'
            else:
                printe('No reliable beam for cross calibration, using 30L anyhow...')
                calib_beam = '30L'

            

            calib_channels = []
            other_channels = {}

            for ch in [ch for s in nimp['systems'] for ch in nimp[s]]:
                t = ch['time'].values  #normalize a typical time range with respect to typical rho range)
                beam_sys = ch['diags'].values[0]
                nz = ch['nimp'].values/1e18 #to avoid numerical overflow with singles
                nz_err = ch['nimp_err'].values/1e18 
                rho = ch['rho'].values

                ind =  (nz > 0)&(nz_err >0)&(nz_err< 1e2)&(rho < .9)
                if any(ind):
                    if 'T_'+calib_beam in beam_sys:
                        calib_channels.append((t[ind],rho[ind],nz[ind],nz_err[ind]))
                    
                    other_channels.setdefault(beam_sys,[])
                    other_channels[beam_sys].append((t[ind],rho[ind],nz[ind],nz_err[ind]))

            if len(calib_channels) == 0:
                printe('unsuccesful..')

                print('\t done in %.1fs'%(time()-T))
                return self.RAW['nimp'][analysis_type+Zimp] 
                        
            calib_channels = np.hstack(calib_channels).T
            other_channels = {n:np.hstack(d).T for n,d in other_channels.items()}
            
            interp = NearestNDInterpolator(calib_channels[:,:2], np.arange(len(calib_channels)))
            calib = {}
                  
            
            tmin = calib_channels[:,0].min()
            tmax = calib_channels[:,0].max()

            for sys,data in other_channels.items():
                #data - time, rho, ne, ne_err
                
                near_ind = interp(data[:,:2])
                dist = np.linalg.norm(calib_channels[near_ind,:2]-data[:,:2],axis=1)
                nearest_ind = np.argsort(dist)
                nearest_ind = nearest_ind[((dist < 0.05))[nearest_ind]]
                
                if len(nearest_ind) == 0:
                    printe(sys+' was not cross calibrated')
                    continue
                cdata = calib_channels[near_ind[nearest_ind],2]
                err = np.hypot(calib_channels[near_ind[nearest_ind],3],data[nearest_ind,3])
                from scipy.stats import trim_mean
                calib[sys] = trim_mean((cdata/err)**2,.1)/trim_mean(data[nearest_ind,2]*cdata/err**2,.1)
                print('correction '+sys+': %.2f'%calib[sys])
                
                #import matplotlib.pylab as plt
                #plt.title(sys)
                #plt.plot(cdata)
                #plt.plot(data[nearest_ind,2]*calib[sys])
                #plt.figure()
                #plt.title(sys)

                #plt.plot(calib_channels[:,0], calib_channels[:,1],'b.')
                #plt.plot(data[:,0],data[:,1],'r.')
                #plt.plot(np.c_[calib_channels[near_ind[nearest_ind],0], data[nearest_ind,0]].T,
                     #np.c_[calib_channels[near_ind[nearest_ind],1],data[nearest_ind,1] ].T,lw=.5)
                #plt.show()
                 

            
            #apply correction, store corrected data 
            for ch in [ch for s in nimp['systems'] for ch in nimp[s]]:
                sys = ch['diags'][0].values.item()
                if sys in calib and calib[sys] != 1.:
                    ch['nimp_corr'] = deepcopy(ch['nimp']) #copy including the attributes
                    ch['nimp_corr_err'] = deepcopy(ch['nimp_err']) 
                    ch['nimp_corr'] *= calib[sys]
                    ch['nimp_corr_err'] *= np.sqrt(calib[sys])  #sqrt to make sure that the errorbars will remain larger
              
            #self.RAW['nimp'][analysis_type+C].update(nimp)
        elif options['Correction']['Relative calibration'].get() and len(unique_impurities)>1:
            printe('Calibration is not implemented for two impurities in NIMP data')
            rcalib = False
                 
                   
                        
        #import IPython
        #IPython.embed()
        #nimp = self.RAW['nimp'][analysis_type]
        #return corrected data if requested
        if rcalib:
            nimp = deepcopy(nimp)
            for ch in [ch for s in nimp['systems'] for ch in nimp[s]]:
                if 'nimp_corr' in ch:
                    ch['nimp'] = ch['nimp_corr']
                    ch['nimp_err'] = ch['nimp_corr_err']
  

        #zkontrolavt jestli je vse konecne u 170700 
        print('\t done in %.1fs'%(time()-T))
        return nimp

 
    def load_zeff(self,tbeg,tend, systems, options=None):
        #load visible bremsstrahlung data
        TT = time()
   
        tbeg,tend = self.eqm.t_eq[[0,-1]]
        #use cached data
        zeff = self.RAW.setdefault('VB',{})
        
        cer_vb_diags = 'CER VB tang', 'CER VB vert'
        cer_diags = 'vertical', 'tangential'

        cer_analysis_type = ''
        if any(np.in1d(cer_vb_diags, systems)):
            analysis_type = 'best'
            analysis_types= 'fit','auto','quick'
            if options is not None:
                selected,analysis_types = options['CER VB']['Analysis']
                analysis_type = selected.get()
                
            cer_analysis_type = self.get_cer_types(analysis_type)
            for i,sys in enumerate(systems): 
                if sys in cer_vb_diags:
                    systems[i] += '_'+cer_analysis_type
                    

 
        self.RAW['VB']['systems'] = systems
        zeff['systems'] = systems
        systems = list(set(systems)-set(zeff.keys()))  #get only systems which are not loaded yet
        #update mapping of the catched data
        zeff = self.eq_mapping(zeff)            
        zeff.setdefault('diag_names',{})
 
        #update equilibrium for already loaded systems
        zeff = self.eq_mapping(zeff)

  
        print_line( '  * Fetching VB (slow) ...' )
        lambda0 = 5230.0
        #NOTE slow, VB signals are in a huge time resolution

        ######################   VB array data #############################
        
        if 'VB array' in systems:            
            tree = 'SPECTROSCOPY'

            if self.shot <= 148154:
                printe('VB Calibration Pre-2012 Not Implemented')
                return

            # Razor viewing dumps for 90 and 315 views
            razor = [True, False, False, False, True, True, True, True, 
                     True, False, True, False, False, True, False, True]
            nchans = 16

            
            #Fetch data from MDS+, it is much faster to fetch uncalibrated data than calibrated \VBXX signals !!!!!
            self.MDSconn.openTree(tree, self.shot)            
            path = '\\TOP.VB.TUBE%.2d:'
            nodes = [path+'PMT_COEFFA%d'%d for d in range(0,5)] 
            nodes += [path+n for n in ['REL_CONST','CTRL_CAL','V_CAL','FS_COEFF_I_N']]
            path = '\\TOP.VB.CHORD%.2d:'
            nodes += [path+n for n in ['PHI_START','PHI_END','R_START','R_END','Z_START','Z_END']]
            downsample = .01#s

            tdi = '['+ ','.join(sum([[n%ch for n in nodes] for ch in range(1,nchans+1)],[]))+']'
            out = self.MDSconn.get(tdi).data().reshape(-1,len(nodes)).T.astype('single') 
            calib =  out[:5][::-1]
            REL_CONST,CTRL_CAL,V_CAL,FS_COEFF_I_N, phi_start,phi_end,R_start, R_end,z_start,z_end = out[5:]
            
            tvec = self.MDSconn.get('dim_of(\VB01)').data()/1000 #timebase ms -> s
            nt = len(tvec)

            if nt > 1:
                #downsample data to 10ms
                dt = (tvec[-1]-tvec[0])/(nt-1)
                n = int(np.ceil(downsample/dt))
                imax = tvec.searchsorted(tend)
        
                try:
                    #load raw voltages - only way how to identify saturated signals!
                    #also it is 2x faster then loading VB

                    CTLMID = [self.MDSconn.get('\CTLMIDVB%.2d'%ch).data()[0] for ch in range(1,nchans+1)]

                    #This value should be between 1 and 10 V. Signals above 10 V will cause the system to trip off, which will cause signal levels to drop to 0.            
                    PHDMID = [self.MDSconn.get('\PHDMIDVB%.2d'%ch).data() for ch in range(1,nchans+1)]
                    PHDMID = np.array([p if len(p)>1  else np.zeros_like(tvec) for p in PHDMID]).T
                    
                                #load_beams = list(np.unique(load_beams))
                    #TDI = ['\\NB::TOP.NB{0}:PINJ_{0}'.format(beam) for beam in load_beams] +['\\NB::TOP:TIMEBASE']
                    
                    #self.MDSconn.openTree('NB', self.shot)                      
                    #nbi = self.MDSconn.get('\\NB::TOP.NB{0}:PINJ_{0}'.format('30L')).data()
                    #t = self.MDSconn.get('\\NB::TOP:TIMEBASE').data()
                    
                    #plot(t/1000,nbi/2500000*2);
                    #plot(tvec_,PHDMID_);show()
          

                    PHDMID = np.mean(PHDMID[:imax//n*n].reshape( -1,n,nchans), 1)
                    tvec   = np.mean(tvec[:imax//n*n].reshape( -1, n), 1)
                    
                    
                    
                    
                    
                    #all values after PHDMID == 10 will be invalid , it is not always working!!
                    valid = PHDMID < 9.9
                    valid = np.cumprod(valid, axis=0,dtype='bool')
            
                    #do calibration VB data W/cm**2/A
                    VB = REL_CONST*FS_COEFF_I_N*np.exp(np.polyval(calib,CTRL_CAL))/np.exp(np.polyval(calib,CTLMID))*(PHDMID/V_CAL)
                    
                except Exception as e:
                    print(e)
                    #for older discharges
                    tdi = '['+ ','.join(['\VB%.2d'%ch for ch in range(1,nchans+1)])+']'
                    VB = self.MDSconn.get(tdi ).data().T

                    if len(tvec) > 1:
                        VB = np.mean(VB[:imax//n*n].reshape( -1,n,nchans), 1)
                        tvec   = np.mean(tvec[:imax//n*n].reshape( -1, n), 1)
                        valid = np.ones_like(VB, dtype='bool')
                    #in the older discharges is the signal negative
                    VB *= np.sign(np.mean(VB))
                
                
                self.MDSconn.closeTree(tree, self.shot)

        
                # Toroidal angle in plane polar coordinates (deg)
                phi_start = np.deg2rad(90.0 - phi_start)
                phi_end   = np.deg2rad(90.0 - phi_end)
                # Radius (mm -> m)
                R_start /= 1000.
                R_end /= 1000.
                # Elevation (mm -> m)
                z_start /= 1000.
                z_end /= 1000.
            
        
                #remove offset 
                if any(tvec < 0):
                    offset = VB[tvec<0].mean(0)
                    baseline_err = np.single(VB[tvec<0].std(0)) 
                    VB -= offset
                else:
                    baseline_err = 0
                imin = tvec.searchsorted(tbeg)
                VB = np.single(VB[imin:])
                tvec = tvec[imin:]
                VB_err = abs(VB)*.1+baseline_err/2 #guess
                VB_err[(VB == 0)|~valid[imin:]] = np.infty
         
                
                zeff['VB array'] = xarray.Dataset(attrs={'system':'VB array','wavelength': 5230.0})

                zeff['VB array']['VB'] = xarray.DataArray(VB,dims=['time','channel'], attrs={'units':'W/cm**2/A','label':'VB' })
                zeff['VB array']['VB_err'] = xarray.DataArray(VB_err,dims=['time','channel'], attrs={'units':'W/cm**2/A'})
                zeff['VB array']['diags']= xarray.DataArray( np.tile(('VB array',), VB.shape),dims=['time','channel'])
                
                zeff['VB array']['R_start'] = xarray.DataArray(R_start, dims=['channel'], attrs={'units':'m'})
                zeff['VB array']['R_end'] = xarray.DataArray(R_end, dims=['channel'], attrs={'units':'m'})

                zeff['VB array']['z_start'] = xarray.DataArray(z_start,dims=['channel'], attrs={'units':'m'})
                zeff['VB array']['z_end'] = xarray.DataArray(z_end,dims=['channel'], attrs={'units':'m'})
                
                zeff['VB array']['phi_start'] = xarray.DataArray(phi_start,dims=['channel'], attrs={'units':'m'})
                zeff['VB array']['phi_end'] = xarray.DataArray(phi_end,dims=['channel'], attrs={'units':'m'})
                
                zeff['VB array']['time'] = xarray.DataArray(tvec.astype('single'),dims=['time'], attrs={'units':'s'})
                zeff['VB array']['channel'] = xarray.DataArray(['VB%.2d'%ich for ich in range(1,nchans+1)])
                zeff['VB array']['razor'] = xarray.DataArray(razor,dims=['channel'] )
                zeff['diag_names']['VB array'] = ['VB array']
                
      
        ######################   CER VB data #############################3


            
         

        if any(np.in1d([c+'_'+cer_analysis_type for c in cer_vb_diags], systems)):
            tree = 'IONS'
            #list of MDS+ signals for each channel
            signals = ['VB','VB_ERR', 'TIME','STIME']
            cal_signals = ['WAVELENGTH','LENS_R', 'LENS_Z','LENS_PHI','PLASMA_R','PLASMA_Z','PLASMA_PHI']

            cer_subsys = {'tangential':'CER VB tang'+'_'+cer_analysis_type, 
                          'vertical':'CER VB vert'+'_'+cer_analysis_type}
            TDI, TDI_calib = [],[]
            
            diags_ = []
            
            self.MDSconn.openTree(tree, self.shot)
            channels = []
            subsys = []
            for key, val in cer_subsys.items():
                if val in systems:
                    subsys.append(key)
            
            for ss in subsys:
                path = 'CER.%s.%s.CHANNEL*'%(cer_analysis_type,ss)
                nodes = self.MDSconn.get('getnci("'+path+'","fullpath")')
                for node in nodes:
                    try:
                        node = node.decode()
                    except:
                        pass
                    node = node.strip()
                    channels.append(ss+'.'+node.split('.')[-1])
                    TDI += ['['+','.join([node+':'+sig for sig in signals])+']']
                    node_calib = node.replace(cer_analysis_type.upper(),'CALIBRATION')                    
                    TDI_calib += [[node_calib+':'+sig for sig in cal_signals]]
                    diags_.append(ss)

            self.MDSconn.closeTree(tree, self.shot)
            
       
            #fast fetch             
            out = mds_load(self.MDSconn, TDI, tree, self.shot)
            
            VB_ = [o[0] if len(o) else np.array([]) for o in out]
            VB_err_ = [o[1] if len(o) else np.array([]) for o in out]
            tvec_ = [o[2] if len(o) else np.array([]) for o in out]
            stime = [o[3] if len(o) else np.array([]) for o in out]
   
            #get a time in the center of the signal integration 
            tvec_ = [(t+s/2.)/1000 for t,s in zip(tvec_, stime)] 
            #get any valid position (for example the one with smallest Z )
            self.MDSconn.openTree(tree, self.shot)

            for diag in subsys:
                #identify channels with valid VB data 
                valid_ind = []
                for ich,ch in enumerate(channels):
                    if diags_[ich] != diag:
                        continue
                    nt = len(tvec_[ich])
                    if nt == 0 or len(VB_[ich]) == 0 or not np.any(np.isfinite(VB_[ich])) or np.all(VB_[ich]==0): 
                        continue
                    ind = slice(*tvec_[ich].searchsorted([tbeg,tend]))
                    if ind.start == ind.stop:
                        continue
                    valid_ind.append(ich)

                if len(valid_ind) == 0:
                    #print('No data in '+diag)
                    continue
                
                #load only channesl for which VB data exists
                TDI_calib_ = sum([TDI_calib[i] for i in valid_ind],[])
                
                out = self.MDSconn.get('['+','.join(TDI_calib_)+']').data() 
                lam,R1,Z1,Phi1 = out.reshape(-1,28)[:,:4].T
                R2,Z2,Phi2  = out.reshape(-1,28)[:,4:].reshape(-1,3,8).swapaxes(0,1)

                #merge channels together
                tvec = np.unique(np.hstack([tvec_[i][VB_[i]!=0] for i in valid_ind]))
                beam_num = np.argmin(np.vstack(Z2),1)
                chind = np.arange(len(beam_num))
                R2 = R2[chind,beam_num]
                Z2 = Z2[chind,beam_num]
                phi2 = np.deg2rad(Phi2[chind,beam_num])
                phi1 = np.deg2rad(Phi1)
         
                VB = np.zeros((len(tvec), len(valid_ind)), dtype='single')
                VB_err = np.zeros((len(tvec), len(valid_ind)), dtype='single')-np.inf  #negative err -> it will be ignored and masked
                for i,ii in enumerate(valid_ind):
                    valid = (VB_[ii] != 0 )&np.isfinite(VB_err_[ii])  #zero values are corrupted fits? 
                    t_ind = np.in1d(tvec, tvec_[ii][valid], assume_unique=True, invert=False)
                    VB[t_ind,i] = VB_[ii][valid]
                    VB_err[t_ind,i] = np.hypot(VB_err_[ii][valid],.1*VB_[ii][valid])#add 10% calibration  error

                
                #estimate postion of the LOS wall intersection 
                pos1 = np.array([R1*np.cos(phi1), R1*np.sin(phi1), Z1]) #position of lens
                pos2 = np.array([R2*np.cos(phi2), R2*np.sin(phi2), Z2]) #beam crossection
                pos3 = pos1+2.*(pos2-pos1)     #position outside of the wall
                R_end = np.hypot(pos3[0],pos3[1])
                z_end = pos3[2] 
                phi_end = np.arctan2(pos3[1],pos3[0])


                VB_err[(VB < 0)|~np.isfinite(VB)|np.isnan(VB_err)] = np.infty
                VB[~np.isfinite(VB)] = 0
       
                # conversion from from ph/m2/sr/s/A to W/cm2/A for the VB measurement
                # ph/s * hc/lambda -> W
                # sr-1 * 4pi -> sr^0
                # m-2 * 1e-4 -> cm-2
                # lam -> lam0 ignore lam dependencec in exp term 

                from scipy.constants import h,c
                
                convert = h * c/(lam*1.e-10) * (4. * np.pi) * 1e-4 * (lam/lambda0)**2
                ind = slice(*tvec.searchsorted([tbeg,tend]))
                ds = zeff[cer_subsys[diag]] = xarray.Dataset(attrs={'system':'CER VB','wavelength': lambda0})
                ds['VB'] = xarray.DataArray(VB[ind]*convert,dims=['time','channel'], attrs={'units':'W/cm**2/A','label':'VB' })
                ds['VB_err'] = xarray.DataArray(VB_err[ind]*convert,dims=['time','channel']) 
                ds['R_start'] = xarray.DataArray(R1, dims=['channel'], attrs={'units':'m'})
                ds['R_end'] = xarray.DataArray(R_end,dims=['channel'], attrs={'units':'m'})
                ds['z_start'] = xarray.DataArray(Z1 ,dims=['channel'], attrs={'units':'m'})
                ds['z_end'] = xarray.DataArray(z_end,dims=['channel'], attrs={'units':'m'})
                ds['phi_start'] = xarray.DataArray(phi1,dims=['channel'], attrs={'units':'m'})
                ds['phi_end'] = xarray.DataArray(phi_end,dims=['channel'], attrs={'units':'m'})
                ds['time'] = xarray.DataArray(tvec[ind].astype('single'),dims=['time'], attrs={'units':'s'})
                ds['channel'] = xarray.DataArray([channels[i][0]+channels[i][-2:] for i in valid_ind] ,dims=['channel'])
                names = ['VB '+diags_[i][0].upper()+'_%d'%Phi1[j] for j,i in enumerate(valid_ind)]
                ds['diags'] = xarray.DataArray(np.tile(names, (len(tvec[ind]),1)),dims=['time', 'channel'])
                    
                zeff['diag_names'][cer_subsys[diag]] = np.unique(names).tolist()

            self.MDSconn.closeTree(tree, self.shot)

        ##########################################  EQ mapping  ###############################################

        zeff['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}

        #calculate LOS coordinates in rho 
        for sys in systems:
            if not sys in zeff or sys in ['vertical','tangential']: continue
  
            #begining and end of the LOS 
            phi_start = zeff[sys]['phi_start'].values 
            pos1 = np.array([zeff[sys]['R_start'].values*np.cos(phi_start),
                                zeff[sys]['R_start'].values*np.sin(phi_start),
                                zeff[sys]['z_start'].values])
            
            phi_end =  zeff[sys]['phi_end'].values
            pos2 = np.array([zeff[sys]['R_end'].values*np.cos(phi_end),
                                zeff[sys]['R_end'].values*np.sin(phi_end),
                                zeff[sys]['z_end'].values])
    
            # Vector in polar (x,y,z) from optic to wall
            ##LOS in 3D space
            nl = 501
            t = np.linspace(0,1,nl)
            xyz = (pos2 - pos1)[...,None]*t +pos1[...,None]
            R, Z = np.hypot(xyz[0],  xyz[1]), xyz[2] 
            L = np.linalg.norm(xyz-pos1[...,None],axis=0)
            
                 
            zeff[sys]['R'] = xarray.DataArray(np.single(R), dims=['channel', 'path'], attrs={'units':'m'})
            zeff[sys]['Z'] = xarray.DataArray(np.single(Z), dims=['channel', 'path'], attrs={'units':'m'})
            zeff[sys]['L'] = xarray.DataArray(np.single(L), dims=['channel', 'path'], attrs={'units':'m'})
            zeff[sys]['path'] = xarray.DataArray(t,  dims=['path'])
            
            
            #equilibrium mapping 
            rho = self.eqm.rz2rho(R.flatten(),Z.flatten(),zeff[sys]['time'].values,self.rho_coord).reshape((-1,)+R.shape)

            #find index of measurement tangential to flux surface
            tg_ind = np.argmin(rho.mean(0),axis=1) if 'vert' in sys else np.argmin(R,axis=1)

            R0 = np.interp(zeff[sys]['time'].values,self.eqm.t_eq, self.eqm.ssq['Rmag'])
            ch = np.arange(R.shape[0])
            rho_tg = rho[:,ch,tg_ind]
            #HFS measuremnt will have a negative sign
            rho_tg[R[ch,tg_ind] < R0[:,None]]*= -1
   
            zeff[sys]['rho']  = xarray.DataArray(rho,dims=['time','channel', 'path'], attrs={'units':'-'})
            zeff[sys]['rho_tg'] = xarray.DataArray(rho_tg,dims=['time','channel'], attrs={'units':'-'})

        ##########################################  VB calculation ###############################################
        print('\t done in %.1fs'%(time()-TT))
        TS = self.load_ts(tbeg,tend,('tangential','core'), options=options['CER system'])
        if bool(options['VB array']['Corrections']['rescale by DCN'].get()):
            try:
                TS = self.dcn_correction(TS, tbeg, tend)
            except Exception as e:
                printe('DCN correction failed:'+str(e))
        TT = time()
        print_line( '  * Calculating VB  ...   ' )

        n_e,n_er,T_e,T_er,rho, tvec = [],[],[],[],[],[]
        for sys in TS['systems']:
            if sys not in TS: continue
            ind =  np.isfinite(TS[sys]['Te_err'].values)& np.isfinite(TS[sys]['rho'].values)
            ind &= np.isfinite(TS[sys]['ne_err'].values)&(TS[sys]['ne'].values > 0)&(TS[sys]['ne'].values < 1.5e20)
        
            
            n_e.append(TS[sys]['ne'].values[ind].flatten())
            n_er.append(TS[sys]['ne_err'].values[ind].flatten())
            T_e.append(TS[sys]['Te'].values[ind].flatten())
            rho.append(TS[sys]['rho'].values[ind].flatten())
            T = np.tile(TS[sys]['time'].values, (len(TS[sys]['channel']),1)).T
            tvec.append(T[ind].flatten())
            #add onaxis values reflected on opposite side
            imin = np.argmin(TS[sys]['rho'].values,1)
            I = np.arange(TS[sys]['rho'].shape[0])
            rho.append(-TS[sys]['rho'].values[I, imin][ind[I, imin]])
            n_e.append(TS[sys]['ne'].values[I, imin][ind[I, imin]])
            T_e.append(TS[sys]['Te'].values[I, imin][ind[I, imin]])
            n_er.append(TS[sys]['ne_err'].values[I, imin][ind[I, imin]])
            tvec.append(T[I, imin][ind[I, imin]])
            #set n_e outside last valid measurement to zero 
            if sys == 'core':
                imax = np.argmax(TS[sys]['rho'].values,1)
                I = np.arange(TS[sys]['rho'].shape[0])
                rho.append(TS[sys]['rho'].values[I, imax][ind[I, imax]]+.01)
                tvec.append(T[I, imax][ind[I, imax]])       
                n_e.append(rho[-1]*0)
                T_e.append(rho[-1]*0+1)
                n_er.append(TS[sys]['ne_err'].values[I, imax][ind[I, imax]])
            
  
        n_e  = np.hstack(n_e).astype('double')        # Electron density (m**-3)
        n_er = np.hstack(n_er).astype('double')        # Electron density (m**-3)
        T_e  = np.hstack(T_e).astype('double')        # Electron temperature (eV)
        rho  = np.hstack(rho).astype('double') 
        tvec = np.hstack(tvec)

        # Free-free gaunt factor
        gff = 1.35 * T_e**0.15
        #gff = 5.542-(3.108-log(T_e/1000.))*(0.6905-0.1323/nominal_values(ZeffCD))  #add small Zeff correction in gaunt factor

        # hc for the quantity [hc/(Te*lambda)] for T_e in (eV) and lambda in (A)
        hc = 1.24e4

        vb_coeff = 1.89e-28 * ((n_e*1e-6)**2)  * gff / np.sqrt(T_e)
        wl_resp = np.exp(-hc / (T_e  * lambda0))/lambda0**2
        vb_coeff *= wl_resp
        
        assert all(np.isfinite(vb_coeff)), 'finite vb_coeff'

        Linterp = LinearNDInterpolator(np.vstack((tvec,rho)).T, vb_coeff,rescale=True, fill_value=0) 

         #calculate weights for each rho/time position
        for sys in zeff['systems']:
            if not sys in zeff or sys in ['vertical','tangential']: continue
            RHO =  zeff[sys]['rho'].values
            T = np.tile(zeff[sys]['time'].values,RHO.shape[:0:-1]+(1,)).T
            vb_coeff_interp = Linterp(np.vstack((T.flatten(), RHO.flatten())).T).reshape(RHO.shape)
            
            dL = np.gradient(zeff[sys]['L'].values)[1]*100 #(m->cm) should be constant for each LOS 
            VB_Zeff1 = np.sum(vb_coeff_interp*dL,2) #VB for Zeff == 1, use for normalization

                  
            #remove radiative mantle from VB array
            VB = np.copy(zeff[sys]['VB'].values)
            VB_err = np.copy(zeff[sys]['VB_err'].values)
            

            if sys == 'VB array' and options is not None and bool(options[sys]['Corrections']['radiative mantle'].get()):
                #calculate edge Zeff and  mantle such, that Zeff from two most edge channels will be equal
                try:

                    R_tg = np.amin(zeff[sys]['R'].values,1)[:,None]
                    #position of the separatrix 
                    R_out, R_in = self.eqm.rhoTheta2rz( 1, [0,np.pi])[0][:,:,0].T
                    R_out = np.interp(zeff['VB array']['time'].values, self.eqm.t_eq, R_out)
                    R_in  = np.interp(zeff['VB array']['time'].values, self.eqm.t_eq, R_in)

                    r_mantle = 0.05 #[VBm] guess
                    dl_mantle = 2*(np.sqrt(np.maximum((R_out+r_mantle)**2-R_tg**2,0)) - np.sqrt(np.maximum(R_out**2-R_tg**2,0))) #outer mantle
                    dl_mantle += 2*np.sqrt(np.maximum(R_in**2-R_tg**2,0)) #inner mantle
                    dl_mantle = np.maximum(r_mantle, dl_mantle) #avoid zero division

                    #use two outermost channels to estimate mantle radiation and zeff                 
                    A = np.dstack((VB_Zeff1[:,:2], dl_mantle[:2].T))
                    Zeff_edge, mantle = np.linalg.solve(A,  VB[:,:2]).T
                    #except:
                        #Zeff_edge = 2*np.ones_like(zeff[sys]['time'].values)
                    #make sure that it has some physical vaues
                    Zeff_edge = np.minimum(Zeff_edge,5)
                    #calculate mantle radiation again from Zeff_edge
                    mantle = (np.maximum(0, VB[:,:2] - Zeff_edge[:,None]*VB_Zeff1[:,:2])/dl_mantle[:2].T).mean(1)

                    #substract mantle radiation
                    VB_mantle = np.minimum(VB, (mantle*dl_mantle).T)
                    VB -= VB_mantle
                    VB_err+= abs(VB_mantle)/3. #pesimistic estimate of the uncertinty in the correction
                except Exception as e:
                    print('VB mantle substration has failed: '+str(e))
              
            #avoid zero division  
            VB_Zeff1 = np.maximum(VB_Zeff1,np.minimum(1e-7,abs(VB_err)/10.))
        
            #import IPython
            #IPython.embed()

            #normalized Zeff values will be plotted, rescaling should not affect anything except of the plotting.
            VB_Zeff1 = VB_Zeff1.astype('single')
            
            zeff[sys]['Zeff'] = zeff[sys]['VB'].copy()
            zeff[sys]['Zeff_err'] = zeff[sys]['VB_err'].copy()
            zeff[sys]['Zeff'].values = VB/VB_Zeff1
            zeff[sys]['Zeff_err'].values = VB_err/VB_Zeff1

            zeff[sys]['Zeff'].attrs = {'units':'-','label':'Z_\mathrm{eff}'}

            #remove too high or too small values
            Zeff_min,Zeff_max = .5, 8
            zeff[sys]['Zeff_err'].values[(zeff[sys]['Zeff'].values > Zeff_max)] = np.inf 
            zeff[sys]['Zeff_err'].values[(zeff[sys]['Zeff'].values < Zeff_min)] = -np.inf  #do not show these values
            
            
            #weight for trapez integration along LOS 
            zeff[sys]['weight'] = xarray.DataArray(np.single(vb_coeff_interp*dL)/VB_Zeff1[:,:,None], dims=['time','channel', 'path'])
 
            
        print('\t done in %.1fs'%(time()-TT))

        if 'tangential' in zeff['systems'] or 'vertical' in zeff['systems']:
            cer_sys = list(set(['tangential', 'vertical'])&set(zeff['systems']))
            NIMP = self.load_nimp(tbeg,tend, cer_sys,options['CER system'])
            
            for sys in cer_sys:
                zeff['diag_names'][sys] = NIMP['diag_names'][sys]
                zeff[sys] = deepcopy(NIMP[sys])
                for ich,ch in enumerate(NIMP[sys]):
                    Linterp.values[:] = np.copy(n_e)[:,None]
                    ne = Linterp(np.vstack((ch['time'].values, ch['rho'].values)).T)
                    Linterp.values[:] = np.copy(n_er)[:,None]
                    ne_err = Linterp(np.vstack((ch['time'].values, ch['rho'].values)).T)

                    lineid = ch['diags'].values[0][::-1].rsplit(' ',1)[0][::-1]
                    Zimp = int(lineid[1:]) if lineid[1].isdigit() else int(lineid[2:])
          
                    Zmain = 1 # NOTE suppose the bulk ions are D
                    ne = np.maximum(ch['nimp'].values*6, ne)
                    Zeff=Zimp*(Zimp - Zmain)*ch['nimp'].values/ne + Zmain
                    Zeff_err=(Zeff-Zmain)*np.hypot(ne_err/(ne+1),ch['nimp_err'].values/(ch['nimp'].values+1))
                    ch = ch.drop(['nimp','nimp_err'])
                    if 'nimp_corr' in ch:
                        ch = ch.drop(['nimp_corr','nimp_corr_err'])
                    zeff[sys][ich] = ch
                    zeff[sys][ich]['Zeff'] = xarray.DataArray(np.single(Zeff),dims=['time'], attrs={'units':'-','label':'Z_\mathrm{eff}'})
                    zeff[sys][ich]['Zeff_err'] = xarray.DataArray(np.single(Zeff_err),  dims=['time'])
    
        #embed()

  

        return zeff
    
    

#replace a GUI call

def main():
    
    sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
    import map_equ_20200306 as map_equ
    import dd_20200507 as dd 

    from map_equ import equ_map
    import tkinter as tk
    myroot = tk.Tk(className=' Profiles')

    shot =   30516
    rho_coord = 'rho_tor'
    
 
    eqm.Open(shot, 'EQI' )

    #load EFIT data from MDS+ 
    T = time()
    eqm._read_pfm()
    eqm.read_ssq()
    eqm._read_scalars()
    eqm._read_profiles()
    print('\t done in %.1f'%(time()-T))
    #import IPython 
    #IPython.embed()
            
    #print 'test'
    #exit()

    loader = data_loader(MDSconn, shot, eqm, rho_coord)

       
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
    'systems':OrderedDict((( 'TS system',(['tangential',I(1)], ['core',I(1)],['divertor',I(0)])),
                            ('ECE system',(['slow',I(1)],['fast',I(0)])))),
    'load_options':{'TS system':{"TS revision":(S('BLESSED'),['BLESSED']+ts_revisions)},
                    'ECE system':OrderedDict((
                                ("Bt correction",{'Bt *=': D(1.0)}),
                                ('TS correction',{'rescale':I(0)})))   }})
        
    settings.setdefault('ne', {\
        'systems':OrderedDict((( 'TS system',(['tangential',I(1)], ['core',I(1)],['divertor',I(0)])),
                                ( 'Reflectometer',(['all bands',I(0)],)),
                                ( 'DCN interf.',(['fit DCN',I(1)],['rescale TS',I(1)])) ) ),
        'load_options':{'TS system':{"TS revision":(S('BLESSED'),['BLESSED']+ts_revisions)},
                        'Reflectometer':{'Position error':{'R shift [cm]':D(0.0)}}}})
        
    settings.setdefault('Zeff', {\
        'systems':OrderedDict(( ( 'VB array',  (['tangential',I(1)],                 )),
                                ( 'CER VB',    (['tangential',I(1)],['vertical',I(1)])),
                                ( 'CER system',(['tangential',I(1)],['vertical',I(1)])))), \
        'load_options':{'VB array':{'Corrections':{'radiative mantle':I(1),'rescale by DCN':I(1)}},\
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
    
    #exit()

    #TODO 160645 DCN correction is broken
    #160646,160657  crosscalibrace nimp nefunguje

    #load_zeff(self,tbeg,tend, options=None)
    data = loader( 'Te', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])
 
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



    
    #for q in ['Te', 'ne', 'Ti','omega','nimp']:
        #data = loader( q, load_options,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])

    print('\n\t\t\t Total time %.1fs'%(time()-TT))

  
if __name__ == "__main__":
    main()
 




