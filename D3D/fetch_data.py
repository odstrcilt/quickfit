from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import MDSplus
import numpy as np
from time import time
from scipy.interpolate import interp1d,RectBivariateSpline,NearestNDInterpolator,LinearNDInterpolator,interpn

from collections import OrderedDict
import matplotlib
import tkinter.messagebox
from copy import deepcopy 
from multiprocessing.pool import ThreadPool, Pool
import xarray
import re,sys,os
np.seterr(all='raise')
from IPython import embed
from scipy.stats import trim_mean
from scipy.integrate import cumtrapz
import matplotlib.pylab as plt
import warnings
import sys
from scipy.signal import medfilt
import tkinter as tk

try: 
    assert 'omfit.py' in sys.argv[0]
    #preferably use OMFITncDataset class from OMFIT, data will be stored as CDF files
    from omfit_classes.omfit_data import OMFITncDataset
    Dataset = OMFITncDataset
    from omfit_classes.omfit_base import OMFITtree
    def Tree(init={}): #emulate behavior of dictionary
        tree = OMFITtree()
        tree.update(init)
        return tree    
except:
    #ignore file argument
    def Dataset(name,*args, **kwargs):
        return xarray.Dataset(*args, **kwargs)
    Tree = dict
   
#warnings.simplefilter(action='ignore', category=FutureWarning)
#Note about output errorbars:
#positive finite - OK
#positive infinite - show points but do not use in the fit
#negative finite - disabled in GUI, but it can be enabled
#negative infinite - Do not shown, do not use

np.seterr(all='raise')

def read_adf11(file,Z, Te, ne):
    #read and interpolate basic ads11 adas files 
    

    with open(file) as f:
        header = f.readline()
        n_ions, n_ne, n_T = header.split()[:3]
        details = ' '.join(header.split()[3:])

        f.readline()
        n_ions, n_ne, n_T = int(n_ions), int(n_ne), int(n_T)
        logT = []
        logNe = []
        while len(logNe) < n_ne:
            line = f.readline()
            logNe = logNe + [float(n) for n in line.split()]
        while len(logT) < n_T:
            line = f.readline()
            logT = logT + [float(t) for t in line.split()]

        logT = np.array(logT)
        logNe = np.array(logNe)

        data = []
        for i_ion in range(n_ions):
            f.readline()
            adf11 = []
            while len(adf11) < n_ne * n_T:
                line = f.readline()
                adf11 = adf11 + [float(L) for L in line.split()]
            adf11 = np.array(adf11).reshape(n_T, n_ne)
            data.append(np.array(adf11))

        data = np.array(data)
        
    RectInt = RectBivariateSpline(logT,logNe, data[Z-1],kx=2,ky=2)
    return 10**RectInt.ev(np.log10(Te),np.log10(ne))


def read_adf12(file,block, ein, dens, tion, zeff):
    with open(file,'r') as f:
        nlines = int(f.readline())

        for iline in range(block):
            cer_line = {}
            params = []
            first_line = '0'
            while(not first_line[0].isalpha()):
                first_line =  f.readline()
            
            cer_line['header'] = first_line
            cer_line['qefref'] = np.float(f.readline()[:63].replace('D', 'e'))
            cer_line['parmref'] = np.float_(f.readline()[:63].replace('D', 'e').split())
            cer_line['nparmsc'] = np.int_(f.readline()[:63].split())
            
            for ipar, npar in enumerate(cer_line['nparmsc']):
                for q in range(2):
                    data = []                    
                    while npar > len(data):
                        line = f.readline()
                        if len(line) > 63: 
                            name = line[63:].strip().lower()
                            cer_line[name] = []
                            if q == 0: params.append(name)
                        
                        values = np.float_(line[:63].replace('D', 'E').split())
                        values = values[values > 0]
                        if not len(values):
                            continue
                        data += values.tolist()
                    cer_line[name] = data       

    #interpolate in logspace
    lqefref = np.log(cer_line['qefref'])
    lnq = np.zeros(np.broadcast(ein, dens, tion, zeff).shape)
    lnq+= lqefref*(1-4)
    lnq+= np.interp(np.log(tion),np.log(cer_line['tiev']) ,np.log(cer_line['qtiev']))
    lnq+= np.interp(np.log(dens),np.log(cer_line['densi']),np.log(cer_line['qdensi']))
    lnq+= np.interp(np.log(ein ),np.log(cer_line['ener']) ,np.log(cer_line['qener']))
    lnq+= np.interp(np.log(zeff),np.log(cer_line['zeff']) ,np.log(cer_line['qzeff']))
    return np.exp(lnq)



    
def read_adf12_aug(data_dir, line, beam_spec='D', therm=False, n_neut=1):
    #data_dir = '/fusion/projects/toolbox/sciortinof/atomlib/atomdat_master/adf12_aug/data/'
    from netCDF4 import Dataset

    imp, Z, transition = line.strip().split(' ')
    # convert from roman to arabic
    d = {'l': 50, 'x': 10, 'v': 5, 'i': 1}
    n = [d[i] for i in Z.lower() if i in d]
    Z = sum([i if i >= n[min(j + 1, len(n) - 1)] else -i for j, i in enumerate(n)])
    n_up, n_low = np.int_(transition.split('-'))

    if therm:
        name = 'qef_' + beam_spec + '_' + imp + str(Z - 1) + '_therm'
    else:
        name = 'qef_' + beam_spec + beam_spec + '_' + imp + str(Z - 1) + '_beam'
    # NOTE: for Ar+ there are files DD_Ar15_arf_beam, DD_Ar15_rld_beam, DD_Ar15_orl_beam, DD_Ar15_uam_beam

    if os.path.isfile(data_dir + name):
        fbeam = Dataset(data_dir + name, mode='r')
    else:
        raise Exception('Data file %s was not found' % (data_dir + name))

    if fbeam.n_upper != n_up or fbeam.n_lower != n_low:
        raise Exception('Transition %s do not match loaded file with %d-%d' % (n_up, n_low))

    # beam components
    if not therm:
        fbeam_E = np.log(fbeam.variables['beam_energy'][:])  # eV, not eV/amu!!
    fbeam_ne = np.log(fbeam.variables['electron_density'][:])
    fbeam_Ti = np.log(fbeam.variables['ion_temperature'][:])
    fbeam_Zeff = np.log(fbeam.variables['z_effective'][:])
    fbeam_qeff = np.log(fbeam.variables['n_%d_emission_coefficient' % n_neut][:])

    def interp_qeff_beam(Zeff, Ti, ne, E):
        # extrapolate by the nearest values, E is in eV/amu!!

        grid = fbeam_Zeff, fbeam_Ti, fbeam_ne, fbeam_E
        #lZeff = np.clip(np.log(Zeff), 0, np.log(6))  # Zeff is just up to 4!!, it will extrapolated
        lZeff = np.clip(np.log(Zeff), 0, np.log(4))  # Zeff is just up to 4!!, it will extrapolated

        lTi = np.clip(np.log(Ti), *grid[1][[0, -1]])
        lne = np.clip(np.log(ne), *grid[2][[0, -1]])
        lE = np.log(E*2.014)
        if lE.max() > np.exp(grid[3][-1] ):
            printe('Energy dependence of CX crossection was extrapolated!!')
        lE = np.clip(lE, *grid[3][[0, -1]])
 
        return np.exp(interpn(grid, fbeam_qeff, (lZeff, lTi, lne, lE), 
                                        fill_value=None, bounds_error=False))

    def interp_qeff_therm(Zeff, Ti, ne):
        # extrapolate by the nearest values

        grid = fbeam_Zeff, fbeam_Ti, fbeam_ne
        #lZeff = np.clip(np.log(Zeff), 1, np.log(6))  # Zeff is just up to 4!!, it will extrapolate
        lZeff = np.clip(np.log(Zeff), 1, np.log(4))  # Zeff is just up to 4!!, it will extrapolate

        lTi = np.clip(np.log(Ti), *grid[1][[0, -1]])
        lne = np.clip(np.log(ne), *grid[2][[0, -1]])
        return np.exp(interpn(grid, fbeam_qeff, (lZeff, lTi, lne), 
                                        fill_value=None, bounds_error=False))

    if therm:
        return interp_qeff_therm
    else:
        return interp_qeff_beam
    
#from matplotlib.pylab import *

#path = os.path.dirname(os.path.realpath(__file__))+'/openadas/' 
#line = 'C VI 8-7'
#interp = read_adf12_aug(path, line, beam_spec='D', therm=False, n_neut=1)

#f,ax = subplots(1,3,sharey=True)
#erel = np.linspace(1e3,100e3)
#ti = 3e3
#ne = 4e13

#sca(ax[0])
#zeff = 1
 
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$Z_\mathrm{eff} = 1$')
#zeff = 2
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$Z_\mathrm{eff} = 2$')
#zeff = 3
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$Z_\mathrm{eff} = 3$')
#ylabel(r'$\sigma_{CX}v$ [Ph m$^3$s$^{-1}$]')

#legend(loc='upper left')
#axvline(81)
#xlabel('E beam [keV]')


#sca(ax[1])
#zeff = 2
#ti = 1e3
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$T_\mathrm{i} = 1\,$keV')
#ti = 3e3
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$T_\mathrm{i} = 3\,$keV')
#ti = 6e3
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$T_\mathrm{i} = 6\,$keV')

#legend(loc='upper left')
#axvline(81)
#xlabel('E beam [keV]')


#sca(ax[2])
#ne = 2e13
#ti = 3e3
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$n_\mathrm{e} = 2\cdot 10^{19}m^{-3}$')
#ne = 4e13
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$n_\mathrm{e} = 4\cdot 10^{19}m^{-3}$')
#ne = 8e13
#qeff = interp(zeff, ti, ne, erel)
#plot(erel/1e3, qeff,label='$n_\mathrm{e} = 8\cdot 10^{19}m^{-3}$')

#legend(loc='upper left')
#axvline(81)
#xlabel('E beam [keV]')
#show()





#f,ax = subplots(1,3,sharey=True)
#erel = np.linspace(70e3,90e3)
#ti = 3e3
#ne = 4e13

#sca(ax[0])
#zeff = np.linspace(1,4,40)[:,None]
#qeff = interp(zeff, ti, ne, erel)
#plot(zeff, 1e6*(np.diff(np.log(qeff),axis=1)/np.diff(erel)).mean(1) )
 
#ylabel(r'$\frac{1}{\sigma_{CX}v}\frac{d\sigma_{CX}v}{dE}|_{E=81\mathrm{keV}}$  [10$^6$ Ph m$^3$s$^{-1}$] ')

 
#xlabel('$Z_\mathrm{eff}$')


#sca(ax[1])
#zeff = 2
#ti = np.logspace(0,4,40)[:,None]
#qeff = interp(zeff, ti, ne, erel)
##embed()
#plot(ti/1e3, 1e6*(np.diff(np.log(qeff),axis=1)/np.diff(erel)).mean(1) )
  
 
#xlabel('$T_i$ [keV]')


#sca(ax[2])
#ne = np.logspace(13,14,40)[:,None]
#ti = 3e3
#qeff = interp(zeff, ti, ne, erel)
#plot(ne*1e6/1e20, 1e6*(np.diff(np.log(qeff),axis=1)/np.diff(erel)).mean(1) )

#xlabel('$n_e$ [10$^20$ m$^{-3}$]')
#show()








#embed()



def read_adf15(PECfile, isel, te,ne):
    #read and interpolate ADAS PEC files
    with open(PECfile) as f:
        header = f.readline()
        n_lines = int(header.split()[0])
        
        for iline in range(n_lines):
            header_line = f.readline().split()
            wav,  n_ne, n_Te = header_line[:3]
            n_ne, n_T = int(n_ne), int(n_Te)

            line_isel = int(header_line[-1])
            if line_isel != isel:
                for it in range(int(n_T)):
                    f.readline()
            else:
                break
        
        
        if line_isel != isel:
            raise Exception('Spectral line was not found')
        
        T,Ne = [],[]
        while len(Ne) < n_ne:
            line = f.readline()
            Ne +=  [float(n) for n in line.split()]
        while len(T) < n_T:
            line = f.readline()
            T +=  [float(t) for t in line.split()]

        logT = np.log(T)
        logNe = np.log(Ne)

        adf15 = []
        while len(adf15) < n_ne * n_T:
            line = f.readline()
            adf15 += [float(L) for L in line.split()]
        logcoeff = np.log(adf15).reshape( n_ne,n_T).T
    
    RectInt = RectBivariateSpline(logT,logNe, logcoeff,kx=2,ky=2)
    _lne, _lTe = np.log(ne), np.log(te)
    
    #avoid extrapolation
    _lne = np.clip(_lne, *logNe[[0,-1]])
    _lTe = np.clip(_lTe, *logT[[0,-1]])
   
    return np.exp(RectInt.ev(_lTe,_lne))


def read_adf21(file, Ebeam,Ne_bulk,Te_bulk):
    
    #read adf21 or adf22 file
    with open(file,'r') as f:
        line = f.readline()
        ref = float(line.split()[1].split('=')[1])
        f.readline()
        line = f.readline()
        nE, nne, Teref = line.split()
        nE, nne = int(nE), int(nne)
        Teref = float(Teref.split('=')[1])
        f.readline()
        
        E = []
        while len(E) < nE:
            line = f.readline()
            E.extend([float(f) for f in line.split()])
        E = np.array(E)

        ne = []
        while len(ne) < nne:
            line = f.readline()
            ne.extend([float(f) for f in line.split()])
        ne = np.array(ne)
        f.readline()
        
        Q2 = []
        while len(Q2) < nne*nE:
            line = f.readline()
            Q2.extend([float(f) for f in line.split()])
        Q2 = np.reshape(Q2, (nne, nE))

        f.readline()
        line = f.readline()
        nTe, Eref, Neref = line.split()
        nTe, Eref, Neref = int(nTe),float(Eref.split('=')[1]),float(Neref.split('=')[1])
        
        f.readline()

        Te = []
        while len(Te) < nTe:
            line = f.readline()
            Te.extend([float(f) for f in line.split()])
        Te = np.array(Te)

        f.readline()

        Q1 = []
        while len(Q1) < nTe:
            line = f.readline()
            Q1.extend([float(f) for f in line.split()])
        Q1 = np.array(Q1)

    #clip data in availible range to avoid extrapolation
    Ebeam = np.clip(Ebeam, *E[[0,-1]])
    Ne_bulk = np.clip(Ne_bulk, *ne[[0,-1]])
    Te_bulk = np.clip(Te_bulk, *Te[[0,-1]])
    
    lref = np.log(ref)
    
    
    #interpolate on the requested values
    #comperature correction
    RectInt1 = interp1d(np.log(Te), np.log(Q1)-lref,assume_sorted=True,kind='quadratic')
    RectInt2 = RectBivariateSpline(np.log(ne), np.log(E), np.log(Q2)-lref,kx=2,ky=2)
    
    adf = RectInt1(np.log(Te_bulk))+RectInt2.ev(np.log(Ne_bulk),np.log(Ebeam))
    return  np.exp(adf+lref)
  
           


def beam_get_fractions(Einj=81.0, model='mickey'):
    #BUG is it valid also for H plasmas???
    # Copied from one of BAGs routines
    # Einj  is in keV
    # Fraction
    j = np.array([1.0, 2.0, 3.0])[:,None] 
    
    if Einj is 0.0:
        return {'cfracs': np.zeros(3), 'pfracs': np.zeros(3), 'nfracs': np.zeros(3)}

 
    ## This is what's in TRANSP and FIDAsim and is "Chuck's" one.
    ## This is Current Fractions.
    if model == 'chuck':
        cgfitf = [-7.83224e-5, 0.0144685, -0.109171]
        cgfith = [-7.42683e-8, 0.00255160, 0.0841037]
        ## Current fractions

        current_fractions = np.zeros((3, np.size(Einj)))
        current_fractions[0] = np.polyval(cgfitf, Einj)  # full energy
        current_fractions[1] = np.polyval(cgfith, Einj)  # halve energy
        current_fractions[2] = 1.0 - current_fractions.sum(0)  # the rest is 1/3 energy

        power_fractions = (current_fractions / j) / np.sum(current_fractions / j, 0)
        density_fractions = (current_fractions / np.sqrt(1.0 / j)) / np.sum(current_fractions / np.sqrt(1.0 / j), 0)

    elif model == 'mickey':
        ## Power Fraction stolen from original compute_impdens
        ## for consistency.
        power_fractions = np.zeros((3,  np.size(Einj)))
        power_fractions[0] = (65.1 + 0.19 * Einj) / 100.0
        power_fractions[1] = (-165.5 + 6.87 * Einj - 0.087 * Einj ** 2 + 0.00037 * Einj ** 3) / 100.0  # typo on fig!
        power_fractions[2] = 1.0 - power_fractions.sum(0)

        current_fractions = power_fractions * j / np.sum(power_fractions * j,0)
        density_fractions = (current_fractions / np.sqrt(1.0 / j)) / np.sum(current_fractions / np.sqrt(1.0 / j),0)
        
        
    elif model == 'NBIgroup':
        #https://diii-d.gat.com/diii-d/Beams_results
        power_fractions = np.zeros((3,  np.size(Einj)))
        power_fractions[0] = 68 + 0.11 * Einj
        power_fractions[1] = -159 + 6.53 * Einj - 0.082 * Einj ** 2 + 0.00034 * Einj ** 3
        power_fractions[2] =  192 - 6.64 * Einj + 0.082 * Einj ** 2 - 0.00034 * Einj ** 3
        power_fractions /= 100
        current_fractions = power_fractions * j / np.sum(power_fractions * j)
        density_fractions = (current_fractions / np.sqrt(1.0 / j)) / np.sum(current_fractions / np.sqrt(1.0 / j),0)
        
        
    else:
        print("Must choose Chuck or Mickey!!!")
        # This is implemented in Brians version, just not here, because it requires d3d_beam which requires neutralizer
        # current_fractions, SmixIn = d3d_beam(Einj,2.0,ZBEAM=1.0)
        # power_fractions = (current_fractions/j)/np.sum(current_fractions/j)
        # density_fractions = (current_fractions/np.sqrt(1./j))/np.sum(current_fractions/np.sqrt(1./j))

    return {'cfracs': current_fractions, 'pfracs': power_fractions, 'nfracs': density_fractions}



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



def int2roman(number):
    ROMAN=[(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]

    result = ""
    for arabic, roman in ROMAN:
        factor, number = divmod(number, arabic)
        result += roman * factor
    return result

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

def min_fine(x,y): #find extrem with subpixel precision
    i = np.argmin(y)
    if i==0 or i == len(y)-1: return y[i],x[i]    #not working at the edge!
    A = x[i-1:i+2]**2, x[i-1:i+2], np.ones(3)
    a,b,c = np.linalg.solve(np.array(A).T,y[i-1:i+2])

    ymin =  c-b**2/(4*a)
    xmin = -b/(2*a)
    return  ymin,xmin


      
def inside_convex_hull(boundary,x):
    """x.shape = (...,2), boundary.shape = (:,2), boundary must have the clock wise direction
    it is !!not!! working for nonconvex boundaries"""
    def cross(o, a, b): 
        return (a[:,0] - o[:,0]) * (b[...,1,None] - o[:,1]) - (a[:,1] - o[:,1]) * (b[...,0,None] - o[:,0])
    
    inlayers = np.all(cross(boundary[:-1,:], boundary[1:,:],x) > 0,axis=-1)

    return inlayers

def split_mds_data(data, bytelens, bit_size):
    n = 0
    if np.size(bytelens) == np.size(data) == 0:
        return []

    #split signals by the channel lenghths
    try:
        if (np.sum(bytelens)-len(data)*bit_size)%len(bytelens):
            raise Exception('Something went wrong!')

        bit_offset = (np.sum(bytelens)-len(data)*bit_size)//len(bytelens)
        data_lens = (np.int_(bytelens)-int(bit_offset))//bit_size
        
        #split data
        out = []
        for l in data_lens:
            out.append(data[n:n+l])
            n+=l
    except:
        print('split_mds_data  failed')
        embed()

    if len(data)!=n:
        raise Exception('Something went wrong!')
    return out

 
        
def detect_elms(tvec, signal,threshold=8,min_elm_dist=5e-4, min_elm_len=5e-4):
    #assume signal with a positive peaks during elms

    from scipy.signal import  order_filter
    #embed()
    signal = signal.reshape(-1,10).mean(1)
    tvec = tvec.reshape(-1,10).mean(1)
    #remove background

    filtered = signal-order_filter(signal, np.ones(51), 5)
    n = 25
    threshold_sig = np.interp(tvec,tvec[n*5::5], order_filter(filtered[::5], np.ones(n*2+1), n)[n:]*threshold)
    #embed()
    #filtered /= np.interp(tvec,tvec[::5], order_filter(filtered[::5], np.ones(51), 25))
    #n = 125
    #plt.plot( threshold_sig)
    #plt.plot(filtered)
    #plt.show()

    filtered[tvec < .5] = 0  #assume no elms first 500ms

    #normalize
    #norm = np.nanmedian(np.abs(filtered))
    ##if norm == 0:
        ##printe('Invalid ELMs signal')
        ##return [[]]*4

    #filtered/= norm
    ##find elms
    ind = filtered > threshold_sig
    from scipy.ndimage.morphology import binary_opening, binary_closing ,binary_erosion, binary_dilation
    #ind = binary_closing(ind)
    #plt.plot(   binary_dilation(binary_erosion(ind,[1,1,1]),[1,1,1]) )

    #plt.plot( binary_opening(ind ,[1,1,1,1,1]))
    ##plt.plot( ind )
    #plt.show()

    #remove tiny short elms
    ind = binary_opening(ind ,np.ones(7))

    ind[[0,-1]] = False
    #import matplotlib.pylab as plt
    #plt.axhline(threshold)
    #plt.plot(tvec, filtered)
    #plt.plot(tvec, signal/norm)

    #plt.show()

    #detect start and end of each elm
    elm_start = np.where(np.diff(np.int_(ind))>0)[0]
    elm_end   = np.where(np.diff(np.int_(ind))<0)[0]
    #n_elms = min(len(elm_start),len(elm_end))
    #elm_start,elm_end = elm_start[:n_elms],elm_end[:n_elms]

    assert not np.any( elm_end-elm_start  < 0) ,'something wrong'

    #due to noise before and elm
    short = tvec[elm_start[1:]]-tvec[elm_end[:-1]] < min_elm_dist
    elm_start = elm_start[np.r_[True,~short ]]
    elm_end   = elm_end[  np.r_[~short,True ]]

    #remove too small elms
    short = tvec[elm_end]- tvec[elm_start] < min_elm_len
    elm_start = elm_start[~short]
    elm_end   = elm_end[~short]

    #import matplotlib.pylab as plt
    #plt.plot(tvec,filtered)


    #filtered[filtered<threshold_sig] = np.nan
    #plt.plot(tvec, filtered,'r')
    #[plt.axvline(tvec[i],ls=':') for i in elm_start]
    #[plt.axvline(tvec[i],ls='--') for i in elm_end]
    #embed()

    val = np.ones_like(elm_start)
    elm_val = np.c_[val, -val,val*0 ].flatten()

    t_elm_val = tvec[np.c_[ elm_start, elm_start, elm_end].flatten()]

    #elm free regions will be set to 2
    elm_val[:-1][np.diff(t_elm_val) > .2] = 2

    #plt.plot(t_elm_val, elm_val*1e21)
    #plt.plot(tvec, filtered)

    #plt.plot(tvec, threshold_sig)
    #plt.show()

    #embed()


    #np.savez('/home/tomas/Dropbox (MIT)/LBO_experiment/SXR_data/elms_175901',tvec=t_elm_val,val=elm_val)

    return t_elm_val,elm_val, tvec[elm_start], tvec[elm_end]
 

def default_settings(MDSconn, shot):
    #Load revisions of Thompson scattering
    ts_revisions = []
    imps = []
    if MDSconn is not None:
        try:
            #load all avalible TS revision
            MDSconn.openTree('ELECTRONS', shot)
            ts_revisions = MDSconn.get('getnci("...REVISIONS...", "node")').data()
            if not isinstance(ts_revisions[0],str): 
                ts_revisions = [r.decode() for r in ts_revisions]
                
            ts_revisions = [r.strip() for r in ts_revisions if 'REVIS' in r]
            MDSconn.closeTree('ELECTRONS', shot)
        except:
            pass
        
   
        try:
            TDI_lineid = []
            MDSconn.openTree('IONS', shot)
            for system in ['tangential','vertical']:
                path = 'CER.CALIBRATION.%s.CHANNEL*'%(system)
                #load only existing channels
                lengths = MDSconn.get('getnci("'+path+':BEAMGEOMETRY","LENGTH")')
                nodes = MDSconn.get('getnci("'+path+'","PATH")').data()
                for node, l in zip(nodes, lengths):
                    if l > 0:
                        if not isinstance(node,str):
                            node = node.decode()
                        TDI_lineid += [node+':LINEID']
 
            #fast fetch of MDS+ data
            _line_id = MDSconn.get('['+','.join(TDI_lineid)+']').data()
                
            MDSconn.closeTree('IONS', shot)
            
            line_id = []
            for l in np.unique(_line_id):
                
                if not isinstance(l,str):
                    l = l.split(b'\x00')[0] #sometimes are names quite wierd
                    l = l.decode()
                l = l.strip()
                line_id.append(l)
 
            for l in np.unique(line_id):
                try:
                    tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', l)
                    imp, Z = tmp.group(1), tmp.group(2)
                    imps.append(imp+str(roman2int(Z) ))
                except:
                    print(l)
                    imps.append('XX')
                #embed()

        except:
            imps = ['C6']
    
 
    #build a large dictionary with all settings
    default_settings = OrderedDict()
    
    cer_ed = ('best','fit','auto','quick','real')

    default_settings['Ti']= {\
        'systems':{'CER system':(['tangential',True], ['vertical',False])},\
        'load_options':{'CER system':{'Analysis':('best', cer_ed ),
                                      'Corrections':{'Zeeman Splitting':True, 'Wall reflections':False}} }}
        
        
 
    default_settings['omega']= {\
        'systems':{'CER system':(['tangential',True], )},
        'load_options':{'CER system':{'Analysis':('best', cer_ed),
                                      'Corrections':{  'Wall reflections':False} },
                        }}

   
    default_settings['Te']= {\
    'systems':OrderedDict((( 'TS system',(['tangential',True], ['core',True],['divertor',False])),
                            ('ECE system',(['slow',False],['fast',False])))),
    'load_options':{'TS system':{"TS revision":('BLESSED',['BLESSED']+ts_revisions)},
                    'ECE system':OrderedDict((
                                ("Bt correction",{'Bt *=': 1.0}),
                                ('TS correction',{'rescale':False})))   }}
        
    default_settings['ne']= {\
        'systems':OrderedDict((( 'TS system',(['tangential',True], ['core',True],['divertor',False])),
                                ( 'Reflectometer',(['all bands',False],)),
                                ( 'CO2 interferometer',(['fit CO2 data',False],['rescale TS',True])) ) ),
        'load_options':{'TS system':{"TS revision":('BLESSED',['BLESSED']+ts_revisions)},
                        'Reflectometer':{'Position error':{'Align with TS':True}, }}}                        
        
            
    nimp = {\
        'systems':{'CER system':(['tangential',True], ['vertical',False], ['SPRED',False])},
        'load_options':{'CER system':OrderedDict((
                                ('Analysis', ('best', ('best','fit','auto','quick','real'))),
                                ('Correction',{'Relative calibration':True, 'nz from CER intensity': False,
                                               'remove first data after blip':False,
                                               'Wall reflections': False}  )))   }}
    #,'Remove first point after blip':False
    #if there are multiple impurities
    for imp in imps:
        default_settings['n'+imp] = deepcopy(nimp)


    default_settings['Zeff']= {\
    'systems':OrderedDict(( ( 'CER system',(['tangential',False],['vertical',False],['SPRED',False])),
                            ( 'SPRED',(['He+B+C+O+N',False],)),
                            ( 'VB array',  (['tangential',True],                 )),
                            ( 'CER VB',    (['tangential',True],['vertical',False])),
                            )), \
    'load_options':{'VB array':{'Corrections':{'radiative mantle':True,'rescale by CO2':True,'remove NBI CX':False}},\
                    'CER VB':{'Analysis':('best', ('best','fit','auto','quick',))},
                    'CER system':OrderedDict((
                            ('Analysis', ('best', ('best','fit','auto','quick',))),
                            ('Correction',    {'Relative calibration':True,  'nz from CER intensity':True}),
                            ('TS position error',{'Z shift [cm]':0.0})))
                    }\
        }
    
    default_settings['Mach']= {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':('best', cer_ed)}}}
    default_settings['Te/Ti']= {\
        'systems':{'CER system':(['tangential',True], ['vertical',False] )},
        'load_options':{'CER system':{'Analysis':('best', cer_ed),
                                      'Corrections':{'Zeeman Splitting':True, 'Wall reflections':False}}}}         
        
        
    if len(imps) > 1:
        default_settings['Zeff']['load_options']['CER system']['Impurity'] = ('C6',imps)
        #print(default_settings['Zeff']['load_options']['CER system']['Impurity'])
        
    return default_settings

class data_loader:
    
    def __init__(self,MDSconn, shot, eqm, rho_coord, raw=Tree()):
        
        self.MDSconn = MDSconn
        self.shot = shot
        self.eqm = eqm
        self.rho_coord = rho_coord
        self.RAW = raw
        
    def eq_mapping(self,diag, dr=0,dz=0):
        #update equilibrium mapping 

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
                else:
                    diag[sys]['rho'].values  =  rho 
            except Exception as e:
                printe(e)
                embed()
                
        diag['EQM'] = Tree({'id':id(self.eqm),'dr':np.mean(dr), 'dz':np.mean(dz),'ed':self.eqm.diag})
 


        return diag
            
            
        
    def __call__(self,  quantity=[], options=None,spline_fits=False, tbeg=0, tend=10 ):
   
        if spline_fits:
            return self.load_zipfit()
        
        
        if quantity == 'elms':
            return self.load_elms(options)
        
            
        if quantity == 'sawteeth':
            return self.load_sawteeth()
        
            
        
        T = time()

        
        options = options[quantity]
        
        imp = 'C6'
        if quantity[0]=='n' and quantity not in ['nimp','ne']:
            imp = quantity[1:]
            quantity = 'nimp'


        systems = []
        if quantity in ['Ti', 'omega', 'nimp','Mach','Te/Ti','Zeff']:
            for sys, stat in options['systems']['CER system']:
                if stat.get(): systems.append(sys)
                
        if  quantity in ['Te', 'ne']:
            for sys, stat in options['systems']['TS system']:
                if stat.get(): systems.append(sys)
        if  quantity in ['Zeff']:
            if options['systems']['VB array'][0][1].get():
                systems.append('VB array')
            for sys, stat in options['systems']['CER VB']:
                if stat.get(): systems.append('CER VB '+sys[:4])
            if options['systems']['SPRED'][0][1].get():
                systems.append('SPRED')
       
        data = []
        ts = None
        
        
        if quantity == 'ne' and 'CO2 interferometer' in options['systems'] and options['systems']['CO2 interferometer'][0][1].get():
            data.append(self.load_co2(tbeg, tend))

        if quantity in ['Te', 'ne']  and len(systems) > 0:
            ts = self.load_ts(tbeg, tend, systems, options['load_options']['TS system'])
            if quantity == 'ne' and 'CO2 interferometer' in options['systems'] and options['systems']['CO2 interferometer'][1][1].get():
                ts = self.co2_correction(ts, tbeg, tend)
            data.append(ts)
        if quantity in ['ne'] and options['systems']['Reflectometer'][0][1].get():
            data.append(self.load_refl(tbeg,tend, options['load_options']['Reflectometer'],TS=ts))
 
    
        if quantity in ['Ti', 'omega','VB'] and len(systems) > 0:
            data.append(self.load_cer(tbeg,tend, systems,options['load_options']['CER system']))
            
        if quantity in ['nimp'] and len(systems) > 0:
            cer = dict(options['load_options']['CER system'])
            cer['Impurity'] = imp
            data.append(self.load_nimp(tbeg,tend, systems, cer))
            

        if quantity in ['Te'] and (options['systems']['ECE system'][0][1].get() or options['systems']['ECE system'][1][1].get()):
            data.append(self.load_ece(tbeg,tend,options,ts))
        
        if quantity in ['Zeff']:
            data.append(self.load_zeff(tbeg,tend,systems, options['load_options']))
        #derived quantities
        if quantity == "Mach":
            cer = self.load_cer(tbeg,tend,['tangential'],options['load_options']['CER system'])             
            from scipy.constants import e,m_u
            Mach = deepcopy(cer)
            for ich,ch in enumerate(cer['tangential']):
                if not 'Ti' in ch or not 'omega' in ch:
                    continue

                omg = ch['omega'].values
                omg_err = ch['omega_err'].values
                ti = np.copy(ch['Ti'].values)
                ti_err = ch['Ti_err'].values
                r = ch['R'].values
                t = ch['time'].values
                ch = ch.drop(['time', 'omega','omega_err','Ti','Ti_err'])

                vtor = omg*r
                vtor_err = omg_err*r
                ti[ti<=0] = 1 #avoid zero division
                vtor[vtor==0] = 1 #avoid zero division
                mach = np.sqrt(2*m_u/e*vtor**2/(2*ti))
                mach_err = mach*np.hypot(vtor_err/vtor,ti_err/ti/2.)
                #deuterium mach number 
                ch['Mach'] = xarray.DataArray(mach, dims=['time'], attrs={'units':'-','label':'M_D'})
                ch['Mach_err'] = xarray.DataArray(mach_err, dims=['time'])
                ch['time'] = xarray.DataArray(t, dims=['time'])

                Mach['tangential'][ich] = ch 
            
            data.append(Mach)
        

        if quantity == "Te/Ti" :
    
            TS = self.load_ts(tbeg,tend,['tangential','core'] )
            CER = self.load_cer(tbeg,tend, systems ,options['load_options']['CER system'] )

            rho_Te,tvec_Te,data_Te,err_Te = [],[],[],[]
            for sys in TS['systems']:
                if sys not in TS: continue 
                t = TS[sys]['time'].values
                te = TS[sys]['Te'].values
                e = TS[sys]['Te_err'].values
                r = TS[sys]['rho'].values
                t = np.tile(t, (r.shape[1],1)).T
                ind = np.isfinite(e)|(te>0)
                tvec_Te.append(t[ind])
                data_Te.append(te[ind])
                err_Te.append(e[ind])
                rho_Te.append(r[ind]) 
 
            rho_Te  = np.hstack(rho_Te)
            tvec_Te = np.hstack(tvec_Te)
            data_Te = np.hstack(data_Te)
            err_Te  = np.hstack(err_Te)

            interp = NearestNDInterpolator(np.vstack((tvec_Te,rho_Te)).T, np.copy(data_Te))
            Te_Ti = deepcopy(CER)
            for sys in CER['systems']:
                if sys not in CER: continue
                for ich,ch in enumerate(CER[sys]):
                    if not 'Ti' in ch: continue
                    interp.values[:] = np.copy(data_Te)#[:,None]
                    t = ch['time'].values
                    Te = interp(np.vstack((t, ch['rho'].values)).T).astype('single')
                    interp.values[:] = np.copy(err_Te)#[:,None]
                    Te_err = interp(np.vstack((t, ch['rho'].values)).T).astype('single')
                    Ti = ch['Ti'].values
                    Ti_err = ch['Ti_err'].values
                    if 'omega' in ch: ch = ch.drop(['omega','omega_err'])
                    Te_Ti[sys][ich] = ch.drop(['Ti','Ti_err','time'])
                    Te_Ti[sys][ich]['Te/Ti'] = xarray.DataArray(Te/(Ti+1),dims=['time'], attrs={'units':'-','label':'T_e/T_i'})
                    Te_Ti[sys][ich]['Te/Ti_err'] = xarray.DataArray(Te/(Ti+1)*np.hypot(Te_err/(Te+1),Ti_err/(Ti+1)),dims=['time'])
                    Te_Ti[sys][ich]['time'] = xarray.DataArray(t,dims=['time'] )

            data.append(Te_Ti)
 
     
        #embed()

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
        #udt, cnt = np.unique(np.round(np.diff(np.hstack(times))*1e3,1), return_counts=True)
        #sind = np.argsort(cnt)[::-1]
        #udt[sind[:4]]
        
        output['tres']=np.median(np.round(np.diff(np.hstack(times))*1e3,1))/1e3
        output['tres']=round(output['tres'],6)
        if output['tres'] == 0:    output['tres'] = 0.01
        output['rho_lbl'] = self.rho_coord
        return output





    def load_zipfit(self):
                 
        if 'ZIPFIT' in self.RAW:
            return self.RAW['ZIPFIT']

            
     
        #               name     scale  node    tree
        loading_dict = {'Ti'   :(1e3,  'ITEMP', 'IONS'),
                        'nimp' :(1e19, 'ZDENS', 'IONS'),
                        'omega':(1e3,  'TROT',  'IONS'),
                        'Te'   :(1e3,  'ETEMP', 'ELECTRONS'),
                        'ne'   :(1e19, 'EDENS', 'ELECTRONS')}


        zipfit = Tree()

        print_line('  * Fetching ZIPFIT ... ')
        T = time()

        for prof, (scale_fact,node, tree) in loading_dict.items():
            try:
                path = '::TOP.PROFILE_FITS.ZIPFIT.'
                self.MDSconn.openTree(tree, self.shot)
                ds = Dataset('zipfit_'+prof+'.nc')
                ds[prof] = xarray.DataArray(self.MDSconn.get('_x=\\'+tree+path+node+'FIT').data()*scale_fact, dims=['time','rho'])
                ds['rho']  = xarray.DataArray(self.MDSconn.get('dim_of(_x,0)').data(), dims=['rho'])
                ds['time'] = xarray.DataArray(self.MDSconn.get('dim_of(_x,1)').data()/1000, dims=['time'])

                try:
                    ds[prof+'_err'] = xarray.DataArray(abs(self.MDSconn.get('error_of(_x)').data())*scale_fact, dims=['time','rho'])
                except:
                    pass

                #remap to a new coordinate
                if self.rho_coord != 'rho_tor':
                    rhot = ds['rho'].values
                    data = ds[prof].values
                    err = ds[prof+'_err'].values
                    rho = self.eqm.rho2rho(rhot, t_in=ds['time'].values,coord_in='rho_tor', coord_out=self.rho_coord)
                    for it,t in enumerate(ds['time'].values):
                        data[it] = np.interp(rhot, rho[it], data[it])
                        err[it]  = np.interp(rhot, rho[it], err[it])
                    ds[prof].values = data
                    ds[prof+'_err'].values = err
                    
                zipfit[prof] = ds
            except Exception as e:
                printe( 'MDS error: '+ str(e))
            finally:
                self.MDSconn.closeTree(tree, self.shot)
        
            
        if 'nimp' in zipfit:
            zipfit['nC6'] = zipfit.pop('nimp')
            zipfit['nC6']['nC6'] = zipfit['nC6']['nimp']
        

        try:
            tvec = np.sort(list((set(zipfit['omega']['time'].values)&set(zipfit['Ti']['time'].values))))
            ind_ti  = np.in1d( zipfit['Ti']['time'].values,tvec, assume_unique=True) 
            ind_omg = np.in1d( zipfit['omega']['time'].values,tvec, assume_unique=True) 

            from scipy.constants import e,m_u
            rho = np.mean(zipfit['Ti']['rho'].values,0)
            Rho,Tvec = np.meshgrid(rho,tvec)
            R = self.eqm.rhoTheta2rz(Rho,0, t_in=tvec,coord_in=self.rho_coord)[0][:,0]
            
            vtor = zipfit['omega']['omega'][ind_omg].values*R
            vtor_err = zipfit['omega']['omega_err'][ind_omg].values*R
            ti = zipfit['Ti']['Ti'][ind_ti].values
            ti_err = zipfit['Ti']['Ti_err'][ind_ti].values
            ti[ti<=0] = 1 #avoid zero division
            zipfit['Mach'] = Dataset('zipfit_mach.nc')
            zipfit['Mach']['Mach'] = xarray.DataArray(np.sqrt(2*m_u/e*vtor**2/(2*ti)), dims=['time','rho'])
            zipfit['Mach']['Mach_err'] = xarray.DataArray(zipfit['Mach']['Mach'].values*np.hypot(vtor_err/vtor,ti_err/ti/2), dims=['time','rho'])
            zipfit['Mach']['rho']  = xarray.DataArray(rho, dims=['rho'])
            zipfit['Mach']['time'] = xarray.DataArray(np.array(tvec), dims=['time'])
        except:
            pass
             

        try:
            tvec = zipfit['Ti']['time'].values
            ind_t = (tvec >= zipfit['Te']['time'].values[0])&(tvec<= zipfit['Te']['time'].values[-1])
            #rho is the same for  both zipfits
            Te     = interp1d(zipfit['Te']['time'].values,zipfit['Te']['Te'].values,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            Te_err = interp1d(zipfit['Te']['time'].values,zipfit['Te']['Te_err'].values ,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            Ti     = np.maximum(zipfit['Ti']['Ti'].values[ind_t],.1) #prevent zero division
            Te     = np.maximum(Te,1e-2) #prevent zero division

            Ti_err = zipfit['Ti']['Ti_err'].values[ind_t]
            
            zipfit['Te/Ti'] = Dataset('zipfit_Te_Ti.nc')
            zipfit['Te/Ti']['Te/Ti'] = xarray.DataArray(Te/Ti, dims=['time','rho'])
            zipfit['Te/Ti']['Te/Ti_err'] = xarray.DataArray(Te/Ti*np.hypot(Ti_err/Ti,Te_err/Te), dims=['time','rho'])
            zipfit['Te/Ti']['rho']  = xarray.DataArray(zipfit['Ti']['rho'].values, dims=['rho'])
            zipfit['Te/Ti']['time'] = xarray.DataArray(tvec[ind_t], dims=['time'])
            
        except:
            pass
        
        try:
            tvec = zipfit['nimp']['time'].values
            ind_t = (tvec >= zipfit['ne']['time'].values[0])&(tvec<= zipfit['ne']['time'].values[-1])
            #rho is the same for  both zipfits
            ne     = interp1d(zipfit['ne']['time'].values,zipfit['ne']['ne'].values,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            ne_err = interp1d(zipfit['ne']['time'].values,zipfit['ne']['ne_err'].values ,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            nimp     =  zipfit['nimp']['nimp'].values[ind_t] 
            nimp_err = zipfit['nimp']['nimp_err'].values[ind_t]
            
            ne     = np.maximum(ne,1) #prevent zero division
            nimp   = np.maximum(nimp,1) #prevent zero division

            # NOTE suppose the impruity ion in ZIPFITprofiles is always carbon and bulk ions are D
            Zimp = 6 
            Zmain = 1
            #Rtan is ~0.6m
            
            zipfit['Zeff'] = Dataset('zipfit_Zeff.nc')
            zipfit['Zeff']['Zeff'] = xarray.DataArray(Zimp*(Zimp - Zmain)*nimp/ne + Zmain, dims=['time','rho'])
            zipfit['Zeff']['Zeff_err'] = xarray.DataArray((zipfit['Zeff']['Zeff'].values - Zmain)*np.hypot(ne_err/ne,nimp_err/nimp), dims=['time','rho'])
            zipfit['Zeff']['rho']  = xarray.DataArray(zipfit['nimp']['rho'].values, dims=['rho']) #rho toroidal
            zipfit['Zeff']['time'] = xarray.DataArray(tvec[ind_t], dims=['time'])
            
        except:
            pass
        
        self.RAW['ZIPFIT'] = zipfit

        print('\t done in %.1fs'%(time()-T))

        return zipfit
    
    
    
    def get_cer_types(self,analysis_type,impurity=False):
         
        path = '.IMPDENS.CER%s:TIME' if impurity else '.CER.CER%s:DATE_LOADED'
        tree = 'IONS'
        analysis_types= 'fit','auto','quick','neur',None
 
        if not hasattr(self,'cer_analysis_best'):
            self.cer_analysis_best = {}
        #use cached data if possible to speed up loading 
        if analysis_type == 'best' and impurity in self.cer_analysis_best:
            analysis_type = self.cer_analysis_best[impurity]
            
        elif analysis_type == 'best':  #find the best  
            try:
                self.MDSconn.openTree(tree, self.shot)

                for analysis_type in analysis_types :
                    try:
                        nodes = self.MDSconn.get(path%analysis_type).data()
                        #stop if node exists
                        break
                    except Exception as e:
                        pass

            except Exception as e:
                printe( 'get_cer_types - MDS error: '+ str(e))
            finally:
                self.MDSconn.closeTree(tree, self.shot)
            
            if analysis_type is None:
                return 'cerreal'
            #print('analysis_type', analysis_type)
            
            self.cer_analysis_best[impurity] = analysis_type
 
        return 'cer'+analysis_type
    
 
    def nbi_info(self,  load_beams, nbi=Tree(), fast=False):
        #TODO assumes a constant voltage!! do not load NBI gas, it it slow 
        
        _load_beams = list(set(load_beams)-set(nbi.keys()))
        
        if len(_load_beams) == 0:
            return nbi

        self.MDSconn.openTree('NB',  self.shot)  

        
        paths = ['\\NB::TOP.NB{0}:'.format(b[:2]+b[-1]) for b in _load_beams] 

        TDI = [p+'pinj_scalar' for p in paths] 
        pinj_scal = self.MDSconn.get('['+','.join(TDI)+']')
        fired = pinj_scal > 1e3
        

        #create NBI info dictionary
        for b,f in zip(_load_beams,fired):
            nbi.setdefault(b,{'fired':f})
        

        if not any(fired):
            return nbi
        
        
        s = 'F' if fast else ''

        pinj_scal = np.array(pinj_scal)[fired]
        _load_beams = np.array(_load_beams)[fired]
        paths = np.array(paths)[fired]
        
        TDI  = [p+'PINJ'+s+'_'+p[-4:-1] for p in paths]
        TDI += ['dim_of('+TDI[0]+')']
        #embed()
        try:
            pow_data = list(self.MDSconn.get('['+','.join(TDI)+']').data())
        except:
            try:
                #older discharges do not have PINJ in MDS+
                TDI  = [p+'PTDATA_CAL' for p in paths]
                TDI += ['dim_of('+TDI[0]+')']
                pow_data = list(self.MDSconn.get('['+','.join(TDI)+']').data())       
                #oldest discharges do not have PTDATA_CAL in MDS+
            except:
                TDI  = [p+'BEAMSTAT' for p in paths]
                TDI += ['dim_of('+TDI[0]+')']
                pow_data = list(self.MDSconn.get('['+','.join(TDI)+']').data())
                #calibrate the data
                for i, p in enumerate(pinj_scal):
                    pow_data[i] *=  p/pow_data[i][pow_data[i] > pow_data[i].max()/5].mean()
                
 
        tvec = pow_data.pop()/1e3
        #embed()


        gas = np.array(['D2']*len(_load_beams))
        
        #slowest step!!
        TDI = [p+'gas' for p in paths]
        gas = self.MDSconn.get('['+','.join(TDI)+']').data()
        if not isinstance(gas[0],str): 
            gas = [r.decode() for r in gas]
        gas = [r.strip() for r in gas]
   
     

        if self.shot  > 169568:  #load time dependent voltage
            TDI = [p+'VBEAM'+s for p in paths]
        else:   #load scalar values
            TDI = [p+'NBVAC_SCALAR' for p in paths]
            
        try:
            volt_data = self.MDSconn.get('['+','.join(TDI)+']').data()
        except:
            #shortly after discharge these are not avalible 
            volt_data = [80e3+np.zeros(len(tvec)) for  p in paths]

        self.MDSconn.closeTree('NB', self.shot)
        

        b21sign= 1 if self.shot < 124700 else -1 #BUG?? 210 always assumed to be counter current
        Rtang = {'30L':114.6, '30R':76.2, '21L':76.2*b21sign,
                '21R': 114.6*b21sign, '33L':114.6, '33R':76.2}
        #rmin=[1.15,.77,1.15,.77,1.15,.77,1.15,.77] in   ./4dlib/BEAMS/atten.pro
        #fill NBI info dictionary
        for i,b in enumerate(_load_beams):
            beam = nbi.setdefault(b,Tree())
            beam['fired'] = pinj_scal[i] > 1e3
            beam['volts'] = volt_data[i]
            if np.size(volt_data[i]) > 1:
                #ignore time varying voltage
                beam['volts'] = beam['volts'][pow_data[i] > 1e3].mean() 
            beam['power'] = pinj_scal[i]
            beam['pow_frac'] = beam_get_fractions(beam['volts']/1e3, 'chuck')['pfracs']
            beam['Rtang'] = Rtang[b[:2]+b[-1]]*0.01
            beam['power_timetrace'] = pow_data[i]
            beam['power_trange'] = tvec[[0,-1]]
            beam['mass'] = {'D2':2.014, 'H2':1.007, 'He': 4.0026 }[gas[i]]
            beam['beam_fire_time'] = np.nan
      
        return nbi
        
    def load_nimp_spred(self,imp, beam_order = [], cx_line=True, beam_blip_avg=True):
        #print('-----------------')
        lines = {'He2':'HeII_304','B5':'BV_262', 'Li3':'LiIII_114', #'Li3': 'LIII_135'
                 'C6':'CVI_182','O8':'OVIII_102','N7':'NVII_134', 'Ne10':'NEX_187',
                 'Ni': 'NIXXV_117', 'Cu':'CUXXVII_154', 'Fe': 'FEXXIII_133', 'Mo':  'MOXXXII_177'}
        
        line_ids = {'He2':'He II 2-1','Li3':'Li III 3-1',
                    'B5':'B V 3-2','C6':'C VI 3-2','O8':'OVIII 3-2','N7':'NVII 3-2','Ne10':'Ne X 4-3'}


        if imp not in lines:
            raise Exception('Loading of impurity %s from SPRED is not supported yet'%imp)
        
        line = lines[imp]
 
        tree = 'SPRED'
        self.MDSconn.openTree('SPECTROSCOPY', self.shot)

        spred_data = self.MDSconn.get('_x=\SPECTROSCOPY::'+line).data()
        if all(spred_data==0):
            raise Exception('MDS+ data for line %s are unavailible'%line)
 
        spred_tvec = self.MDSconn.get('dim_of(_x)').data()
 
        spred_data*= 1e4 #convert from ph/cm**2/s/sr to ph/m**2/s/sr
        spred_data = spred_data[spred_tvec > 0]
        spred_tvec = spred_tvec[spred_tvec > 0]
        
        # SPRED records the timestamp at the *end* of the integration period. Opposite to CER.
        spred_stime = np.diff(spred_tvec)
        spred_stime = np.r_[spred_stime[0], spred_stime]
        spred_tvec -= spred_stime #make it consistent with CER
        
        if not cx_line:
            return spred_tvec/1e3, spred_data
            
        spred_tvec__, spred_data__,  spred_stime__ = spred_tvec, spred_data, spred_stime
 
        #SPRED cross beam 30L and 30R close to the magnetic axis 
        load_beams = '30L','30R'
        
        NBI = self.RAW['nimp'].setdefault('NBI',{})
        self.nbi_info(load_beams, NBI)
        beam_geom = np.zeros((2,len(beam_order)), dtype='single')
        
        
        view_R = [1.77, 1.80]
        view_Z = [0.0530, 0.0530]
        view_phi= [13.51, 20.15]
        geomfac = [3.129, 2.848]  # m**-1

        nbi_on_frac = np.zeros((len(spred_tvec),len(load_beams)))
        nbi_pow = np.zeros((len(spred_tvec),len(load_beams)))

        for ib,b in enumerate(load_beams):
            if not NBI[b]['fired']:
                continue
            beam_geom[0,beam_order.index(b+'T ')] = geomfac[ib]
            beam_tmin,beam_tmax = NBI[b]['power_trange']*1e3#ms!
            beam_on  = NBI[b]['power_timetrace'] > 1e5
            beam_time = np.linspace(beam_tmin,beam_tmax, beam_on.size)

            inds = beam_time.searchsorted(np.r_[spred_tvec, 0])
            
            for i,t  in enumerate(spred_tvec):
                if inds[i]<inds[i+1]:
                    nbi_on_frac[i,ib] = np.mean(beam_on[inds[i]:inds[i+1]])
                    nbi_pow[i,ib] = np.mean(NBI[b]['power_timetrace'][inds[i]:inds[i+1]])

        min_beam_frac = 0.8
        if beam_blip_avg:
            #it is better to include this missalighned times, than deal with background substraction
            min_beam_frac = 0.3

        #nbi_all = np.all(nbi_on_frac > 0.8,1)
        nbi_mixed = np.any(nbi_on_frac  > min_beam_frac,1)
        #nbi_single =  nbi_mixed&~nbi_all
        nbi_count = np.sum(nbi_on_frac > min_beam_frac,1)
        #timeslices with 0.1 < nbi_on_frac < 0.8 will be ignored 
        #nbi_off = np.all(nbi_on_frac < 0.1,1)
        #nbi_off &= spred_tvec < spred_tvec[nbi_mixed].max()+100
        

 
        #TODO make a smarter substraction!
        #dont use points too far from reference? extrapolate a ratio of active and passive? 

             
        #estimate passive background
        #from scipy.signal import medfilt 
        #_bg = spred_data[nbi_off]
        #_bg = medfilt(_bg, 11 )
        #bg = np.interp(spred_tvec, spred_tvec[nbi_off], _bg)
        #_active = spred_data[nbi_mixed]/nbi_on_frac.sum(1)[nbi_mixed]
        #_active = medfilt(_active, 11 )
        #active = np.interp(spred_tvec, spred_tvec[nbi_mixed], _active)+1 
        ##assume that bg is proportional to active - better interpolation in the regions where are data not interleaved
        #bg = np.interp(spred_tvec, spred_tvec[nbi_off], (bg/active)[nbi_off])*active
        #bg = np.single(bg)
        #tvec = spred_tvec[nbi_mixed]
        #br = (spred_data-bg)[nbi_mixed]
        #TODO probably remove point entirelly when a proper bckg substraction is impossible 
        

        nt = len(spred_tvec)
        spred_ind = np.arange(nt)
        nbi_count_diff = np.ediff1d(nbi_count, to_begin=1)
        nbi_ind_change = np.append(spred_ind[np.abs(nbi_count_diff) > 0], nt-1) 

 
        if beam_blip_avg:
            max_step = 20.#ms
            spred_data_tmp = []
            spred_tvec_tmp = []
            spred_stime_tmp = []
            nbi_ind_change_tmp = []
            nbi_on_frac_tmp = []

            for i in range(len(nbi_ind_change)-1):
                #estimate the number of intervals needed to split region without changing beams in less than max_step long intervals
                nsteps = max(1,int(np.round((spred_tvec[nbi_ind_change[i+1]]-spred_tvec[nbi_ind_change[i]])/max_step)))
              
                step = (nbi_ind_change[i+1]-nbi_ind_change[i])//nsteps
                nbi_ind_change_tmp.append(len(spred_tvec_tmp))
                #get averaged power and other quantities for each interval
                for j in range(nsteps):
                    ind = slice(nbi_ind_change[i]+j*step, nbi_ind_change[i]+(j+1)*step)
                    spred_data_tmp.append(spred_data[ind].mean())
                    spred_tvec_tmp.append(spred_tvec[ind.start])
                    spred_stime_tmp.append(spred_stime[ind].sum())
                    nbi_on_frac_tmp.append(nbi_on_frac[ind].mean(0))
            
            nbi_ind_change_tmp.append(len(spred_tvec_tmp)-1)

            spred_data = np.array(spred_data_tmp)
            spred_tvec = np.array(spred_tvec_tmp)
            spred_stime = np.array(spred_stime_tmp)
            nbi_ind_change = np.array(nbi_ind_change_tmp)
            nbi_on_frac = np.array(nbi_on_frac_tmp)
            
            nbi_mixed = np.any(nbi_on_frac > min_beam_frac,1)
            nbi_count = np.sum(nbi_on_frac > min_beam_frac,1)

        #plt.clf()
        #plt.step(spred_tvec, nbi_pow, where='post')
        #for ib,b in enumerate(load_beams):
            #beam_tmin,beam_tmax = NBI[b]['power_trange']*1e3#ms!
            #beam_on  = NBI[b]['power_timetrace'] > 1e5
            #beam_time = np.linspace(beam_tmin,beam_tmax, NBI[b]['power_timetrace'].size)
            #plt.plot(beam_time, NBI[b]['power_timetrace'])
        
        #for x in nbi_ind_change:
            #plt.axvline(spred_tvec[x])
        
        #plt.step(spred_tvec_tmp, nbi_pow_tmp, where='post',ls='--')
        
        #for x in nbi_ind_change_tmp:
            #plt.axvline(spred_tvec_tmp[x],ls='--')
        #plt.show()
        
            
            
        #embed() 
    
        #nbi_ind_change
        #nbi_on_frac
        #spred_data 
        #spred_tvec 
        #stime 


        #nbi_region_ind = np.cumsum(nbi_ind_change > 0)
        
        #plt.plot(nbi_count)
        #plt.plot(nbi_count_diff)
        #plt.plot(nbi_region_ind/20)

        #for i in nbi_ind_end: plt.axvline(x=i)
        #for i in nbi_ind_start: plt.axvline(x=i,ls='--')

        #plt.show()
        #embed()
        
        tmax = 50. #ms maximum distance for background substraction
        tbg = 9.9 #ms length of interval use for background substraction, 9.9 is used instead of 10ms de to numerical errors
        #spred_data_ = []
        spred_err_ = []

        bg_  = []
        #spred_tvec_  = []
        nbi_frac_ = []
        nch = len(nbi_ind_change)
        TTSUB_ = []
        TTSUB_ST_ = []
        valid_ind = []
        dt_shift = []
        #median absolute deviation
        MAD = lambda x: 1.48*np.median(np.abs(x-np.median(x)))
        

        nbi_working = spred_tvec <= spred_tvec[nbi_mixed].max()
        for i,(t,s) in enumerate(zip(spred_tvec[nbi_working], spred_stime[nbi_working])):
            if nbi_count[i] == 0 or spred_data[i] == 0:
                continue
            
            ir = nbi_ind_change.searchsorted(i,  side='right')-1
            irl = 0
            #find nearest suitable region for background substraction on the left
            for j in range(1,10):
                if ir >= j and nbi_count[nbi_ind_change[ir-j]] < nbi_count[i]:
                    irl = ir-j
                    break
            
            #find nearest suitable region for background substraction on the right
            irr = len(nbi_ind_change)-1
            for j in range(1,10):
                if ir < nt-j and nbi_count[nbi_ind_change[ir+j]] < nbi_count[i]:
                    irr = ir+j
                    break
            
            bg_inds = []
            nbg = int(np.ceil(tbg/s))

            #don't use one timeslice before the change
            #choose the better one, left or right, if equal use both
            if nbi_ind_change[irr] - i >= i - (nbi_ind_change[irl+1]-1):
                #left is closer or equal
                if t - spred_tvec[nbi_ind_change[irl+1]-1] < tmax:
                    bg_inds.append(slice(max(nbi_ind_change[irl], nbi_ind_change[irl+1]-nbg-1),
                                   max(nbi_ind_change[irl+1]-1, nbi_ind_change[irl]+1)))
            if nbi_ind_change[irr] - i <= i - (nbi_ind_change[irl+1]-1):
                #right is closer or equal
                if spred_tvec[nbi_ind_change[irr]]-t < tmax:
                    bg_inds.append(slice(nbi_ind_change[irr], max(nbi_ind_change[irr]+1, min(nbi_ind_change[irr+1]-1, nbi_ind_change[irr]+nbg))))

     
            #suitable timeslice for background substraction was found 
            for j,bg_ind in enumerate(bg_inds):
                if bg_ind.start < bg_ind.stop:
                    
                    nbi_frac = nbi_on_frac[i]-nbi_on_frac[bg_ind].mean(0)
                    #skip some crapy beam blips 
                    if np.all(nbi_frac < 0.5):
                        continue
                    if np.any(nbi_frac < 0): #something wierd, but rare
                         continue
                     
                    #nbi_frac
                    
                    valid_ind.append(i)
                    #dt_shift.append(j*1e-4) #shift the timebase a bit if data are from second substraction location

                    bg_.append(spred_data[bg_ind].mean())
                    #sometimes the SPRED timing is not accurate, use robust MAD insted STD
                    spred_err_.append( MAD(spred_data[bg_ind]))
                    nbi_frac_.append(nbi_frac)
                    TTSUB_.append(spred_tvec[bg_ind.start])
                    TTSUB_ST_.append(spred_tvec[bg_ind.stop]-spred_tvec[bg_ind.start])
    
       
        valid_ind = np.array(valid_ind)
        spred_stime = spred_stime[valid_ind]#+np.array(dt_shift)
        nbi_frac = np.array(nbi_frac_)
        
      
        spred_data_, spred_tvec_ = spred_data, spred_tvec
        spred_tvec = spred_tvec[valid_ind]
        spred_data = spred_data[valid_ind]-np.array(bg_)
        min_err = 1.0e13
        spred_err = np.hypot(spred_err_, spred_data*0.05)  # at least 5% error
        spred_err = np.maximum(spred_err, min_err)
        
        #plt.plot(spred_tvec, spred_data)
        #plt.show()
        #spred_err[spred_data == 0] = np.infty  #SPRED was not working

        #stime = stime[nbi_mixed]

        #br_err = np.maximum(0.10 * br, min_err)
        #br_err[spred_data[nbi_mixed]==0] = np.infty
        #try:
        R = np.dot(view_R, nbi_frac.T)/nbi_frac.sum(1)
        Z = np.dot(view_Z, nbi_frac.T)/nbi_frac.sum(1)
        phi = np.dot(view_phi, nbi_frac.T)/nbi_frac.sum(1)
        #except:
            #embed()
        TTSUB = np.array(TTSUB_, dtype='single')
        TTSUB_ST = np.array(TTSUB_ST_, dtype='single')

        #TTSUB = np.zeros_like(tvec,dtype='single')
        #TTSUB_ST = np.zeros_like(tvec,dtype='single')

                
        #print(time()-T)

        #plt.plot( spred_data_,'.')

        #plt.plot( valid_ind,bg_,'x')
        #plt.plot( (nbi_pow[:,0]*geomfac[0]+nbi_pow[:,1]*geomfac[1])*1e19/30e6 )
        #plt.plot( (nbi_pow[:,0]*geomfac[0] )*1e19/30e6 )
        #plt.plot( ( nbi_pow[:,1]*geomfac[1])*1e19/30e6 )

        ##plt.plot(nbi_count)
        #plt.plot(spred_data)
        #plt.plot(valid_ind, spred_data[valid_ind],'x')

        #plt.show()
        
 
        #import matplotlib.pylab as plt

        #plt.title(imp+' '+str(self.shot))
        #if NBI['30L']['fired']:
            #plt.plot(beam_time,  NBI['30L']['power_timetrace']*1e19/30e6*geomfac[0],'r',label='30L')
            #plt.step(spred_tvec,nbi_pow[:,0]*geomfac[0]*1e19/30e6,'r' ,where='post')
        #if NBI['30R']['fired']:
            #plt.plot(beam_time,  NBI['30R']['power_timetrace']*1e19/30e6*geomfac[1],'g',label='30R')
            #plt.step(spred_tvec,nbi_pow[:,-1]*geomfac[1]*1e19/30e6,'g', where='post')

        #if NBI['30R']['fired'] and NBI['30L']['fired']:
            #plt.plot(beam_time,  (NBI['30L']['power_timetrace']*geomfac[0]+NBI['30R']['power_timetrace']*geomfac[1])*1e19/30e6,'b-',label='30L+30R')
            #plt.step(spred_tvec,(nbi_pow[:,0]*geomfac[0]+nbi_pow[:,1]*geomfac[1])*1e19/30e6,'b-', where='post')


        #bg = np.ones_like(spred_tvec_)*np.nan
        #bg[valid_ind] = bg_
        #plt.step(spred_tvec_,spred_data_,'k-', where='post',label='SPRED',lw=2)
        #plt.step(spred_tvec_,bg,'y.-', where='post',label='background')
        #plt.plot(spred_tvec+spred_stime/2,spred_data+bg_,'x' )
        #plt.plot(spred_tvec+spred_stime/2 , bg_,'o' )
        #plt.step(spred_tvec__,spred_data__,'y-', where='post',label='SPRED',lw=2)
 
        #for ib,b in enumerate(load_beams):
            #if not NBI[b]['fired']:  continue
            #beam_tmin,beam_tmax = NBI[b]['power_trange']*1e3#ms!
            #beam_on  = NBI[b]['power_timetrace'] > 1e5
            #plt.plot(beam_time, NBI[b]['power_timetrace']/1e6*1e17*10)
        #plt.legend(loc='best')
        #plt.show()
        ##embed()

        #plt.plot(spred_tvec+stime[0]/2,spred_data,'.')
        ##plt.plot(spred_tvec,spred_data)
        #plt.plot(spred_tvec+stime[0]/2,spred_data+np.array(bg_),'x')

        #plt.plot(spred_tvec+stime[0]/2, bg_,'s')
        
        #plt.show()
        #embed()

        ##plt.step(spred_tvec,nbi_mixed*1e19,'--', where='post')
        ##plt.step(spred_tvec,nbi_all*1e19, where='post')
        ##plt.step(spred_tvec,nbi_off*1e19,':', where='post')
 
        #plt.show()
        
        #embed()
        #bg2 = np.zeros_like(spred_tvec)
        #nc2 = np.ones_like(spred_tvec)

        #n_step = 50
        #C = 1
        #ncycle = 10 if any(nbi_all) else 1
        #for j in range(ncycle):
            #for i in range(len(spred_tvec)//(n_step//2)):
                #tind = slice(i*n_step//2-n_step//2,i*n_step//2+n_step//2)
                #if np.all(nbi_all[tind]):
                    ##not possible to estimate background or no background estimate needed
                    #continue
                #n = len(nbi_off[tind])
                #A = np.vstack((np.ones(n),(nbi_pow[tind,0]+nbi_pow[tind,1]*C)/2e6)).T
                #x,r,rr,s = np.linalg.lstsq(A,spred_data[tind],rcond=1e-3)
                #tind = slice(i*n_step//2-n_step//4,i*n_step//2+n_step//4+1)
                #bg2[tind] = x[0]
                #nc2[tind] = x[1]
            
            #x,r,rr,s = np.linalg.lstsq(nbi_pow[nc2>0]/2e6 ,(spred_data-bg2)[nc2>0]/nc2[nc2>0])
            #C = x[1]/x[0]
            #print(j,x)
            
            #plt.step(spred_tvec[nc2>0],np.dot(nbi_pow[nc2>0]/2e6, x),':', where='post')
            #plt.step(spred_tvec[nc2>0],(spred_data-bg2)[nc2>0]/nc2[nc2>0],':', where='post')
            #plt.ylim(0,5)
            #plt.show()
        #embed()
        
        
        #plt.step(spred_tvec,spred_data,':', where='post')
        ##plt.plot(beam_time,  NBI['30L']['power_timetrace']*1e19/30e6*geomfac[0])
        ##plt.plot(beam_time,  NBI['30R']['power_timetrace']*1e19/30e6*geomfac[1])
        ##plt.plot(beam_time,  (NBI['30L']['power_timetrace']*geomfac[0]+NBI['30R']['power_timetrace']*geomfac[1])*1e19/30e6)
        #plt.step(spred_tvec,bg, where='post')
        #plt.step(spred_tvec,bg2, '--',where='post')
        #plt.step(spred_tvec,nbi_pow[:,0]*geomfac[0]*1e19/30e6 , where='post')
        #plt.step(spred_tvec,nbi_pow[:,1]*geomfac[1]*1e19/30e6 , where='post')
        #plt.step(spred_tvec,(nbi_pow[:,0]*geomfac[0]+nbi_pow[:,1]*geomfac[1])*1e19/30e6, where='post')


        #plt.show()
        
        
        
        
        #stime = stime[nbi_mixed]

        #min_err = 1.0e13
        #br_err = np.maximum(0.10 * br, min_err)
        #br_err[spred_data[nbi_mixed]==0] = np.infty
        #R = np.dot(view_R, nbi_on_frac[nbi_mixed].T)/nbi_on_frac.sum(1)[nbi_mixed]
        #Z = np.dot(view_Z, nbi_on_frac[nbi_mixed].T)/nbi_on_frac.sum(1)[nbi_mixed]
        #phi = np.dot(view_phi, nbi_on_frac[nbi_mixed].T)/nbi_on_frac.sum(1)[nbi_mixed]

       
        #TTSUB = np.zeros_like(tvec,dtype='single')
        #TTSUB_ST = np.zeros_like(tvec,dtype='single')
        
        #print(spred_tvec.shape, spred_stime.shape, spred_data.shape, spred_err.shape, R.shape, Z.shape, TTSUB.shape, TTSUB_ST.shape, beam_geom.shape, )
   

        return spred_tvec,spred_stime, spred_data, spred_err,np.single(R), np.single(Z),np.single(phi),TTSUB,TTSUB_ST, line_ids[imp], beam_geom



        
    def load_nimp_intens(self, nimp,load_systems, analysis_type,imp):

        #calculate impurity density directly from the measured intensity
        
      
        tree = 'IONS'   
        load_spred = load_real = False
        if 'SPRED_'+imp in load_systems:
            load_spred = True
            
        if analysis_type == 'cerreal':
            if imp != 'C6':
                raise Exception('CERREAL is availible only for carbon')
            load_real=True
            load_systems = ['tangential']
            
            
        load_systems = [sys for sys in load_systems if 'SPRED' not in sys]
 
        if len(load_systems) == 0 and not load_spred:
            return nimp
 
        ##############################  LOAD DATA ######################################## 

        print_line( '  * Fetching CER INTENSITY %s from %s ...'%(imp, analysis_type.upper(), ) )
        TT = time()



        loaded_chan = []
        TDI_data = []
        TDI_tvec = []
        #TDI_details = {'lineid':[], 'beam_geo':[], 'phi':[], 'R':[],'Z':[]}
        
        TDI_lineid = ''
        TDI_beam_geo = ''
        TDI_phi = ''
        #TDI_R = []
        #TDI_Z = []

        bytelens = []
        
        
        
        self.MDSconn.openTree(tree, self.shot)
        
        #prepare list of loaded channels
        for system in load_systems:
            nimp[system] = []
            nimp['diag_names'][system] = []
            
            if load_real:
                continue
                #nodes = ['CHANNEL%.2d'%ch for ch in range(5,25)]
                #lengths = np.ones(len(nodes))

            #else:
            path = 'CER.%s.%s.CHANNEL*'%(analysis_type,system)
            nodes = self.MDSconn.get('getnci("'+path+'","fullpath")').data()
            lengths = self.MDSconn.get('getnci("'+path+':R","LENGTH")').data()
            #TDI_details['R'] = []
            #TDI_details['Z'] = []

                
            for node,length in zip(nodes, lengths):
                if length == 0: continue
                try:
                    node = node.decode()
                except:
                    pass

                node = node.strip()
                loaded_chan.append(system[0].upper()+node[-2:])
                
                signals = [ 'STIME','R','Z','INTENSITY','INTENSITYERR','TTSUB','TTSUB_STIME']
                bytelens += [[length]*(len(signals)+1)]

                TDI_data += [[node+':'+sig for sig in signals]]
                TDI_tvec += [ node+':'+'TIME']#time can have a different data type than the rest
                
                calib_node = '\\IONS::TOP.CER.CALIBRATION.%s.%s'%(system,node.split('.')[-1])
 
                TDI_lineid +=  calib_node+':LINEID'+','
                TDI_beam_geo += calib_node+':BEAMGEOMETRY'+','
                TDI_phi += calib_node+':LENS_PHI'+','

                #if load_real:
                    
                    
                    #ch = int(node[-2:])
                    #if ch in np.r_[1:8, 17:23]: #30 beam
                        #beams = '0,1'
                    #elif ch in np.r_[8:17, 23:25 ]: #330 beam
                        #beams = '2,3'
                    #else:
                        #raise Exception('Not supported')
                    
                    #take just an average position, ignoring which beams was on
                    #TDI_details['R'] += [f'MEAN(({calib_node}:PLASMA_R)[[{beams}]])']
                    #TDI_details['Z'] += [f'MEAN(({calib_node}:PLASMA_Z)[[{beams}]])']


        tvec, stime, R, Z, INT, INT_ERR,TTSUB,TTSUB_ST,phi,line_id = [],[],[],[],[],[],[],[],[],[]
        beam_geom = np.zeros((0,8))
  
        

        #fast fetch of MDS+ data
        order='\\IONS::TOP.CER.CALIBRATION.BEAM_ORDER'
        try:
            beam_order = list(self.MDSconn.get(order).data())
        except: #old discharges
            beam_order = ['30LT ', '30RT ', '330LT', '330RT'] 

        try:
            beam_order = [b.decode() for b in beam_order]
        except:
            pass


        if len(loaded_chan) > 0 or load_real:
            
            #fetch actual data with impurity radiation
            if not load_real:
    
                TDI = '['+TDI_lineid[:-1]+']','['+TDI_beam_geo[:-1]+']', '['+TDI_phi[:-1]+']'
                
                line_id,beam_geom,phi = mds_load(self.MDSconn,TDI, tree, self.shot)
                
                
                if len(line_id) == 0: #missing in old discharges, assume carbon
                    line_id = ['C VI 8-7']*len(loaded_chan)
                
                if not isinstance(line_id[0],str):
                    line_id = [l.split(b'\x00')[0].decode() for l in line_id]
        
                if imp == 'XX':
                    printe('!!XX impurity is assumed to be Ca XVIII 15-14 !!')
                    #BUG hardcoded
                    imp = 'Ca18'
                    for i in range(len(line_id)):
                        if line_id[i].startswith('unknown'):
                            line_id[i] = 'Ca XVIII 15-14'
                    
                imp_name, charge = re.sub("\d+", '', imp), re.sub('\D', '', imp)
                r_charge = int2roman(int(charge))
                selected_imp = np.array([l.startswith(imp_name) and r_charge in l for l in line_id])
                
                #embed()
                
                selected_imp &= np.any(beam_geom > 0,1) #rarely some channel has all zeros!
                if not any(selected_imp):
                    if sum([len(nimp[sys]) for sys in nimp['systems'] if sys in nimp]):
                        #some data were loaded before, nothing in the actually loaded system
                        return nimp
                    raise Exception('No '+imp+' data in '+analysis_type.upper(),'edition. ', 'Availible are :'+','.join(np.unique(line_id)))
                
                #keep data only from the selected impurity 
                phi = phi[selected_imp]
                line_id = np.array(line_id)[selected_imp]                       
                beam_geom = beam_geom[selected_imp]
                loaded_chan = np.array(loaded_chan)[selected_imp]
                loaded_chan = loaded_chan.tolist()

                #embed()

                TDI_data = ','.join(np.transpose(TDI_data)[:,selected_imp].flatten())
                TDI_tvec = ','.join(np.asarray(TDI_tvec)[selected_imp])

                bytelens = np.transpose(bytelens)[:,selected_imp].flatten()
                TDI = ['['+TDI_tvec+']','['+TDI_data+']']
                tvec,data = mds_load(self.MDSconn,TDI, tree, self.shot)
                tvec = tvec.astype('single') #sometimes it can be double
                
                #split in signals
                if data.ndim == 1:
                    data = np.array(split_mds_data(np.hstack((tvec,data)), bytelens, 4),dtype=object)
            
                splitted_data = np.split( data , len(signals)+1)
                #embed()
                #splitted_data = [s for s in splitted_data]
                
                tvec, stime, R, Z, INT, INT_ERR,TTSUB,TTSUB_ST = splitted_data
             
            else: #real time CER
                channels = range(5,25)
                TDI = ['[dim_of(PTDATA("crsampt{0}",{1})), PTDATA("crsampt{0}",{1})]'.format(ch, self.shot) for ch in channels]
                mds_data = mds_load(self.MDSconn,TDI, tree, self.shot)
                #valid = []
                beam_geom = []
                
                #hardcode values, not availible after discharge
                R_plasma = np.array([2043,2127,2194,2186,2194,2217,2232,2247,2262,\
                        2277,2292,2306,1701,1838,1904,1974,2088,2161,2178,2209])   
                Z_plasma = np.array([-2,5,10,0,0,0,0,0,0,1,1,1,0,-5,-9,-12,1,8,0,0])
                
                for ich, ch_data in enumerate(mds_data):
                    if len(ch_data) == 0 or all(ch_data[1] == 0):
                        continue
                    #valid.append(ich)
                    ch = channels[ich]
                    
                    T,I = ch_data.reshape(2,-1)
                    #where the data becommed availible in realtime and they are nonzero
                    ind = (np.ediff1d(I,to_begin=0)!=0)&(I > 0)
                    t = T[ind]
                    tvec.append(t)
                    #estimate integration time
                    dt = np.diff(t)
                    dt = np.median(dt[dt < dt.min()*2])
                    #print(dt)
                    stime.append(t*0+dt)
                    #embed()
                    INT.append(I[ind])
                    INT_ERR.append(np.maximum(INT[-1],0)*0.05)
                    TTSUB.append(t*0)
                    TTSUB_ST.append(t*0)
 
                    R.append(t*0 + R_plasma[ich]/1e3)
                    Z.append(t*0 + Z_plasma[ich]/1e3)
                    loaded_chan.append('T%.2d'%ch)
                    
                    beam_fact = np.zeros(len(beam_order))       
                    if ch in np.r_[1:8, 17:23]: #30 beam
                        #beam_fact = [1,1,0,0]
                        beams = [0,1]
                    elif ch in np.r_[8:17, 23:25 ]: #330 beam
                        #beam_fact = [0,0,1,1]
                        beams = [2,3]
                    else:
                        raise Exception('Not supported channel ', ch)
                    
                    #beam_fact = np.zeros(4)
                    #beam_fact
                    beam_fact[beams] = 1
                    beam_geom.append(beam_fact)
                    
                
                #print(stime)
                    
                nch = len(tvec)
                phi = np.zeros(nch)    
                line_id = ['C VI 8-7']*nch

                #line_id = [line_id[i] for i in valid]
                print('BUG!!!!!!!!')
                #beam_geom = np.ones((nch, 8))
                beam_geom = np.array(beam_geom)
                #loaded_chan = [loaded_chan[i] for i in valid]
                #print(len(loaded_chan), len(lineid))
        


            #correction of the geometry factors for vertical system
            
            #NOTE geometry factor for beam V330R was ignored in the past, use geom fact from 183168      
            V = ['V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V17', 'V18', 'V19','V20', 'V21', 'V22', 'V23']
            G = [0.941, 0.814, 0.827, 1.076, 1.559, 2.147, 0.836, 1.11 , 0.866, 0.809, 0.798, 0.877, 1.276]
            for ch,g in zip(V,G):
                if ch in loaded_chan:
                    ich = loaded_chan.index(ch)
                    if beam_geom[ich,3] == 0: #not defined geometry factor
                        beam_geom[ich,3] = g 
                    
  
            
        
        if load_spred:
            #fetch SPRED as other CER channel 
            try:
                spred = self.load_nimp_spred(imp, beam_order)
            except Exception as e:
                raise
                printe(e)
                spred = None


            nimp['diag_names']['SPRED_'+imp] = []
            nimp['SPRED_'+imp] = []
            if spred is not None and np.any(np.isfinite(spred[3])):

                #append to already fetched CER data
                tvec=list(tvec)+[spred[0]]
                stime=list(stime)+[spred[1]]
                INT=list(INT)+[spred[2]]
                INT_ERR=list(INT_ERR)+[spred[3]]
                R=list(R)+[spred[4]]
                Z=list(Z)+[spred[5]]
                phi=list(phi)+[spred[6]]
                TTSUB=list(TTSUB)+[spred[7]]
                TTSUB_ST=list(TTSUB_ST)+[spred[8]]
                line_id=list(line_id)+[spred[9]]
                #embed()
                beam_geom=np.append(beam_geom,spred[10][:,:len(beam_order)],0)
                loaded_chan=list(loaded_chan)+['SPRED_'+imp]
            else:
                printe('Loading of SPRED '+imp+' was unsucessful')
        
        #embed()
        if len(loaded_chan) == 0 and (len(load_systems) > 0 or load_spred):
            if sum([len(nimp[sys]) for sys in nimp['systems'] if sys in nimp]):
                    #some data were loaded before, nothing in the actually loaded system
                    return nimp
            raise Exception('Error: no data! try a different edition?')
        
        
        #convert to to seconds
        tvec  =  [t/1e3 for t in tvec]
        stime =  [t/1e3 for t in stime]
        TTSUB =  [t/1e3 for t in TTSUB]
        TTSUB_ST = [t/1e3 for t in TTSUB_ST]

        T_all = np.hstack(tvec)
        stime_all = np.hstack(stime)
        TT_all = np.hstack(TTSUB)
        
        R_all = np.hstack(R)
        Z_all = np.hstack(Z)
        
        #map on rho coordinate
        rho_all = self.eqm.rz2rho(R_all[:,None],Z_all[:,None],T_all+stime_all/2,self.rho_coord)[:,0]
         
        ########################  Get NBI info ################################
        #which beams needs to be loaded

        beam_order = np.array([b.strip()[:-1] for b in beam_order])
        beam_ind = np.any(beam_geom>0,0)
        load_beams = beam_order[beam_ind]
        
        NBI = self.RAW['nimp']['NBI']

        self.nbi_info(load_beams,NBI)
        fired = [NBI[b]['fired'] for b in load_beams]
        load_beams = load_beams[fired]
        beam_geom = beam_geom[:,beam_ind][:,fired]


        beam_tmin, beam_tmax = NBI[load_beams[0]]['power_trange']
        PINJ = np.array([NBI[b]['power_timetrace'] for b in load_beams])

        beam_time = np.linspace(beam_tmin, beam_tmax,PINJ.shape[1])
        
        if load_real:
            for i,b in enumerate(load_beams):
                PINJ[i] = NBI[b]['power'] 
        
        #embed()
        
        #plt.plot(beam_time,PINJ.T)
        #for t in np.unique(T_all):
            #plt.axvline(t)
        #plt.show()
        
        
        
        #NOTE it will fail if two timepoinst has the same T_all+stime_all/2, but different stime_all!!
        nbi_cum_pow = cumtrapz(np.double(PINJ),beam_time,initial=0)
        nbi_cum_pow_int = interp1d(beam_time, nbi_cum_pow,assume_sorted=True,bounds_error=False,fill_value=0)
        utime, utime_ind = np.unique(T_all+stime_all/2, return_index=True)
        ustime = stime_all[utime_ind]
        nbi_pow = np.single(nbi_cum_pow_int(utime+ustime/2)-nbi_cum_pow_int(utime-ustime/2))/ustime


        #when the NBI was turned on , downsample to reduce noise 
        _PINJ = PINJ[:,:(len(beam_time)//10)*10].reshape(len(PINJ),-1,10).mean(2)
        _beam_time = beam_time[:(len(beam_time)//10)*10].reshape(-1,10).mean(1)

        nbi_cum_on = cumtrapz(np.maximum(np.diff(_PINJ,axis=1),0) ,_beam_time[1:],initial=0)
        nbi_cum_on_int = interp1d(_beam_time[1:], nbi_cum_on,assume_sorted=True,bounds_error=False,fill_value=0)
        nbi_on = (nbi_cum_on_int(utime)-nbi_cum_on_int(utime-ustime))/ustime > 2e5
    

        #when background substraction was used
        valid_sub = TT_all > 0
        if any(valid_sub):
            TT_s_all = np.hstack(TTSUB_ST)
            utime_sub, utime_sub_ind = np.unique(TT_all[valid_sub], return_index=True)
            ustime_sub = TT_s_all[valid_sub][utime_sub_ind]
            nbi_pow_sub = (nbi_cum_pow_int(utime_sub+ustime_sub)-nbi_cum_pow_int(utime_sub))/(ustime_sub+1e-6)
        else:
            nbi_pow_sub = None
        
                
        ########### create xarray dataset with the results ################
        n = 0
        beam_intervals = {}
        for ich,ch in enumerate(loaded_chan):
            diag = {'V':'vertical','T':'tangential','S':'SPRED_'+imp}[ch[0]]
            nt = len(tvec[ich])
    
            # List of chords with intensity calibration errors for FY15, FY16 shots after
            # CER upgraded with new fibers and cameras.
            disableChanVert = 'V03', 'V04', 'V05', 'V06', 'V23', 'V24'
            if 162163 <= self.shot <= 167627 and ch in disableChanVert:
                INT_ERR[ich][:] = np.infty
            #apply correctin for some channels
            if ch == 'T07' and 158695 <= self.shot < 169546:
                INT[ich] *= 1.05
            if ch == 'T23' and  158695 <= self.shot < 169546:
                INT[ich] *= 1.05
            if ch == 'T23' and  165322 <= self.shot < 169546:
                INT[ich] *= 1.05
                
                
  
 
            ##############  find beam ID of each timeslice
          
            observed_beams = beam_geom[ich] > 0
            #special case of vertical system and 210 beam which should cross it? 
            if sum(observed_beams) > 2:
                bind = np.argsort(beam_geom[ich, observed_beams])[::-1]
                observed_beams[np.where(observed_beams)[0][bind[2:]]] = False
 
            beams = load_beams[observed_beams]
            #mean background substracted NBI power over CER integration time 
            it = utime.searchsorted(tvec[ich]+stime[ich]/2)
            try:
                beam_pow = nbi_pow[observed_beams][:,it]
            except:
                embed()
            #when background substraction was applied
            if nbi_pow_sub is not None and any(TTSUB[ich] > 0):
                beam_pow[:,TTSUB[ich] > 0] -= nbi_pow_sub[observed_beams][:,utime_sub.searchsorted(TTSUB[ich][TTSUB[ich] > 0])]

            #check if there is any timeslice when the beam was turned on
            beams_used = np.any(beam_pow > 1e5,1)
            beams = beams[beams_used]
            beam_pow = beam_pow[beams_used]
            observed_beams = np.in1d(load_beams, beams)
            beam_on = np.any(nbi_on[observed_beams],0)[it]


            beamid = np.zeros(nt, dtype='U4')
            #how each beam contributes to this channel
            beam_frac = np.ones((len(beams),nt),dtype='single')
            if len(beams) == 0:
                #embed()
                printe(ch+' missing beam power data')
                continue
            elif len(beams) == 1: #only one beam observed
                beamid[:] = beams
            else: #L and R beam
                beam_frac = beam_pow*beam_geom[ich, observed_beams][:,None]
                beam_frac /= beam_frac.sum(0)+1e-6 #prevent zero division for channel V6
                #use only first two beams with highest geom factors
                bind = np.argsort(beam_geom[ich, observed_beams])[::-1]
                
                beamid[(beam_frac[bind[0]] > 0.3) &(beam_frac[1] > 0.3)] = beams[bind[0]][:-1]+'B'
                beamid[(beam_frac[bind[1]] < 0.3)] = beams[bind[0]]
                beamid[(beam_frac[bind[0]] < 0.3)] = beams[bind[1]]
            
            #print(ich, len(line_id))
            tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', line_id[ich])
        

            element, charge, transition = tmp.group(1), tmp.group(2), tmp.group(3) 
            charge = roman2int(charge)     
            
            edge,s = False,''
            if (diag == 'vertical' and  int(phi[ich]) == 331) or (diag == 'tangential' and  int(phi[ich]) == 346): #split vertical 33L in the core and edge
                edge, s = True, 'e'

                        
            names = np.array([diag[0].upper()+'_'+ID.lstrip('0')+s+' '+element+str(charge) for ID in beamid])
            unames,idx,inv_idx = np.unique(names,return_inverse=True,return_index=True)
      
            for name in unames:
                if not name in nimp['diag_names'][diag]:
                    nimp['diag_names'][diag].append(name)

            tvec_ = (tvec[ich]+stime[ich]/2)
            
            #split channels by beams
            for ID in np.unique(inv_idx):
                beam_ind = inv_idx == ID
                if ch == 'V06' and np.any(beams == '330L'):
                     #channel V06 has measurements with zero beam power but finite intensity
                    beam_ind &= beam_pow[beams == '330L'][0] > 1e5
  
                #rarely, R is zero - BUG in CERFIT?? - remove such cases
                beam_ind &= R[ich] > 0
                
                beams_on = beam_pow[:,beam_ind].sum(0) > 1e5
                if not np.all(beams_on):
                    #embed()
                    printe('channel %s has %d slices with zero power'%(ch, (~beams_on).sum()))
                    beam_ind[beam_ind] &= beams_on
                    
                    
                if np.sum(beam_ind) < 2: #sometimes there is just one slice - remove
                    continue
                
                if names[idx[ID]].split()[0] in ['V_330Le','V_330Be']:
                    INT_ERR[ich][beam_ind] *= -1 #show but datapoints will be disable by defauls 

                #make sure that the timebase is sorted, in some rare cases with CERFIT it is not unique (different identification of beam phases??)
                #add random jitter to avoid this issue, important for SPRED 
                tvec_[beam_ind] += 1e-5*np.random.randn(len(tvec_[beam_ind]))
                utvec, uind = np.unique(tvec_[beam_ind], return_index=True)
       
                    
                beam_ind = np.arange(len(tvec_))[beam_ind][uind]

                #save data when the beam was turned on
                bname = beamid[idx[ID]]
                beam_intervals.setdefault(bname,[])
                beam_intervals[bname].append((tvec[ich][beam_ind],stime[ich][beam_ind]))  
                G = beam_geom[ich, observed_beams]
                
                ds = Dataset('CER_'+ch+'_'+bname+'.nc', attrs={'channel':ch+'_'+bname, 'system': diag,'edge':edge,'name':names[idx[ID]],
                                        'beam_geom':G,'Z':charge})

                #fill by zeros for now, 
                ds['nimp'] = xarray.DataArray(0*utvec, dims=['time'], 
                                        attrs={'units':'m^{-3}','label':'n_{%s}^{%d+}'%(element,charge),'Z':charge, 'impurity':element})

                ds['nimp_err']  = xarray.DataArray(0*utvec-np.inf,dims=['time'])
 
                ds['int'] = xarray.DataArray(INT[ich][beam_ind], dims=['time'], 
                                        attrs={'units':'ph / sr m^{3}','line':line_id[ich].strip()})
                ds['int_err']  = xarray.DataArray(INT_ERR[ich][beam_ind],dims=['time'])
                ds['R'] = xarray.DataArray(R[ich][beam_ind],dims=['time'], attrs={'units':'m'})
                ds['Z'] = xarray.DataArray(Z[ich][beam_ind],dims=['time'], attrs={'units':'m'})
                ds['rho'] = xarray.DataArray(rho_all[n:n+nt][beam_ind],dims=['time'], attrs={'units':'-'})

                ds['diags']= xarray.DataArray(names[beam_ind],dims=['time'])
                ds['stime'] = xarray.DataArray(stime[ich][beam_ind],dims=['time'], attrs={'units':'s'})
                ds['beam_pow'] = xarray.DataArray(beam_pow[:,beam_ind],dims=['beams','time'], attrs={'units':'W'})
                #NBI was just turned on
                ds['beam_swiched_on'] = xarray.DataArray(beam_on[beam_ind],dims=['time'] )
                ds['beam_frac'] = xarray.DataArray(beam_frac[:,beam_ind],dims=['beams','time'], attrs={'units':'W'})
                ds['beams'] = xarray.DataArray(beams,dims=['beams'])
                #it is much faster to add "time" the last in the xarray.dataset than as the first one!!
                ds['time'] = xarray.DataArray(utvec,dims=['time'], attrs={'units':'s'})

                nimp[diag].append(ds)
              
            n += nt
            
            
            
            
      
        #how long was each beam turned on
        for n,b in beam_intervals.items():
            NBI.setdefault(n,{})
            beam_times, stime = np.hstack(b)
            ubeam_times,ind = np.unique(beam_times,return_index=True)
            NBI[n]['beam_fire_time'] = np.sum(stime[ind])
    
            
        #if SOL_reflections:
        #print('BUG!!!!!! temporary fix SOl refrection')
        #cer = self.load_cer(0,10, load_systems)
        #if 'tangential' in nimp:
            #for ch in nimp['tangential']:
                #for cch in cer['tangential']:
                    #if ch.attrs['channel'][:3] == cch.attrs['channel']:
                        #ch['int'].values = np.interp(ch['time'].values, cch['time'].values, cch['int'].values )
                        #ch['int_err'].values = np.interp(ch['time'].values, cch['time'].values, cch['int_err'].values )



        nimp['EQM'] = Tree({'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag})
        nimp['loaded_beams'] = np.unique(np.hstack((load_beams,nimp.get('loaded_beams',[]))))

        print('\t done in %.1fs'%(time()-TT))

    
        
        return nimp

    def calc_nimp_intens(self,tbeg,tend, nimp,systems,imp, nC_guess=None, options=None):
        #extract data from all channels
        nimp_data = {'time':[],  'int':[], 'int_err':[], 'R':[]}
        beam_data = {'beam_pow':{},'beam_geom': {}}
        data_index = []
        
        if imp == 'XX':
            printe('bug XX is Ca18')
            imp =  'Ca18'
        
        
        if imp not in ['Li3','B5','C6','He2','Ne10','N7','O8','F9','Ca18','Ar18','Ar16']:
            raise Exception('CX cross-sections are not availible for '+imp)
        


        n = 0
        for diag in systems:
            for ch in nimp[diag]:
                for k in nimp_data.keys():
                    nimp_data[k].append(ch[k].values)
                beams = list(ch['beams'].values)
                nt = len(nimp_data['time'][-1])
                for b in nimp['loaded_beams']:
                    for k in beam_data.keys():
                        beam_data[k].setdefault(b,[])
                        if b in beams:
                            ib = beams.index(b)
                            if k == 'beam_geom':
                                data = np.tile(ch.attrs['beam_geom'][ib],nt)
                            else:
                                data = ch[k].values[ib]
                            beam_data[k][b].append(data)
                        else:
                            beam_data[k][b].append(np.zeros(nt))
                data_index.append(slice(n,n+nt))
                n += nt
        
        if n == 0:
            raise Exception('No CER intensity data')
            
        #merge the data
        for k in nimp_data.keys():
            nimp_data[k] = np.hstack(nimp_data[k])
        for k in beam_data.keys():
            beam_data[k] = np.vstack([np.hstack(beam_data[k][b]) for b in nimp['loaded_beams']])
        
                
        
        NBI = self.RAW['nimp']['NBI']
        nbeam = len(nimp['loaded_beams'])  

        nbi_dict = {'beams':nimp['loaded_beams']}
        for k in ['volts','pow_frac','Rtang','mass']:
            nbi_dict[k] = []
            for b in nimp['loaded_beams']:
                nbi_dict[k].append(NBI[b][k])
            nbi_dict[k] = np.array(nbi_dict[k])
        

        ########################   Get kinetic profiles along the midradius  ##########################
        #split measurements in 100 clusters
        from scipy.cluster.vq import kmeans2
        valid = np.isfinite(nimp_data['int_err'])
        int_err =  nimp_data['int_err'][valid]
        nimp_time = nimp_data['time'][valid][int_err > 0]
        nclust = min(100, len(np.unique(nimp_time))//2)
        #embed()

        _centroid, _label = kmeans2(nimp_time, nclust,100, minit='points')
        _centroid = np.unique(_centroid)
 
        #create a new labels including also removed points 
        ic = 0
        centroid = []
        label = np.zeros(len(nimp_data['time']),dtype='int')
        for i,c in enumerate(_centroid):
            tmin = (c+_centroid[i-1])/2 if i > 0 else -np.inf
            tmax = (c+_centroid[i+1])/2 if i < _centroid.size-1 else np.inf
            ind = (nimp_data['time'] > tmin)&(nimp_data['time'] <= tmax)
            if any(ind):#skip empty clusters
                label[ind] = ic
                centroid.append(c)
                ic += 1
            
 
        ########################   From TS scattering  ##########################

        TS = self.load_ts(tbeg,tend,('tangential','core'))
        try:
            TS = self.co2_correction(TS, tbeg, tend)
        except Exception as e:
            printe('CO2 correction failed:'+str(e))

        #slice data from TS
        beam_profiles = {'ne':{},'ne_err':{}, 'te':{}, 'rho':{}}
        for sys in TS['systems']:
            if sys not in TS: continue
            n_e = TS[sys]['ne'].values
            n_e_err = TS[sys]['ne_err'].values
            T_e = TS[sys]['Te'].values
            T_e_err = TS[sys]['Te_err'].values

            TSrho = TS[sys]['rho'].values
            TStvec = TS[sys]['time'].values
            #remove invalid points 
            TS_valid = np.isfinite(T_e_err)
            TS_valid &= np.isfinite(n_e_err)
            TS_valid &= n_e > 0
            TS_valid &= n_e < 1.5e20
            TS_valid &= T_e > 0
            
            #initialise dict of lists for each timeslice
            for k,d in beam_profiles.items():   d[sys] = []
 
            for it, t in enumerate(centroid):
                tind = label == it
            
                T = nimp_data['time'][tind]
                itmin,itmax = TStvec.searchsorted([T.min(),T.max()])
         
                tslice = slice(max(0,itmin-1),itmax+1) #use at least one measurement before and one after
                ch_valid = np.any(TS_valid[tslice],0)
                
                rho_slice = np.average(TSrho[tslice, ch_valid],0, TS_valid[tslice, ch_valid])
                #if len(rho_slice) < 3:
                    #print('Too low number of valid radial points from TS %s system'%sys)
                    #continue

                    #embed()
                    
                beam_profiles['rho'][sys].append(rho_slice)  
                ne_slice = np.exp(np.average(np.log(n_e[tslice, ch_valid]+1.),0, TS_valid[tslice, ch_valid]))
                
                beam_profiles['ne'][sys].append(ne_slice)
                ne_err_slice = 1/np.average(1/n_e_err[tslice, ch_valid],0, TS_valid[tslice, ch_valid])
                beam_profiles['ne_err'][sys].append(np.maximum(ne_err_slice, .05*ne_slice)) #minimum 5% error
                Te_slice = np.exp(np.average(np.log(T_e[tslice, ch_valid]+1.),0, TS_valid[tslice, ch_valid]))
                beam_profiles['te'][sys].append(Te_slice)
        #BUG is it sorted somewhere??
        #merge data from all TS systems 
        for k, d in beam_profiles.items():
            merged_sys = []
            for i,t in enumerate(centroid):
                #try:
                data = np.hstack([d[sys][i] for sys in TS['systems'] if sys in d])
                #except:
                    #embed()
                if len(data) == 0:
                    data = merged_sys[-1] #use last existing value 
                
                merged_sys.append(data) 
            beam_profiles[k] = merged_sys


        #create radial midplane grid based on location of TS measurements
        Rmin = [nimp_data['R'][label == i].min() for i,t in enumerate(centroid)]
        
        #map midplane radius to rho
        R_midplane = np.r_[nimp_data['R'].min()-.1:2.0:0.01, 2.0:2.4:0.005]
        rho_midplane = self.eqm.rz2rho(R_midplane,R_midplane*0,centroid,self.rho_coord)
        ind_axis = np.argmin(rho_midplane,axis=1)
        Raxis = R_midplane[ind_axis]
        #embed()
        beam_profiles_ = deepcopy(beam_profiles)
        beam_profiles['Rmid'] = []
        for it, t in enumerate(centroid):   
            rho = beam_profiles['rho'][it]
            #print(rho)
            ind_axis = np.argmin(rho_midplane[it])
            #map from rho to Rmid for both LFS and HFS channel
            R_lfs = np.interp(rho,rho_midplane[it][ind_axis:],R_midplane[ind_axis:])
            R_hfs = np.interp(rho,np.r_[rho_midplane[it][ind_axis::-1],1.5],np.r_[R_midplane[ind_axis::-1],1])
            R = np.hstack([R_hfs,R_lfs])
            sind = np.argsort(R)
            sind = sind[R[sind].searchsorted(Rmin[it])-1:] #just splighly outside of outermost measurement
            #create grid in the reversed order from the LFS to HFS
            sind = sind[::-1]
            beam_profiles['Rmid'].append(R[sind])
            #sort all profiles by Rmid
            for k,d in beam_profiles.items():
                if k != 'Rmid':
                    d[it] = d[it][sind%len(rho)]
                    
                    
            #plot(np.hstack([R_hfs,R_lfs]), np.hstack([rho,rho]),'o')
            #show()
            
            

        #embed()

        ########################   From CER     ##########################
        #fetch ion temperature and rotation C:\Users\odstrcil\projects\quickfit\D3D\fetch_beams.py
        cer_systems = [sys for sys in nimp['systems'] if 'SPRED' not in sys]
 
        cer = self.load_cer(tbeg,tend, cer_systems,options=options)
        
        #slice and interpolate omega and Ti on the same coordinates as TS
        beam_profiles.update({'Ti':[], 'omega':[], 'fC':[]})
        cer_data = {'omega':[], 'Ti':[], 'fC': []}
        
        #extract all data from XARRAYs
        for sys in cer['systems']:
            for ch in cer[sys]:
                rho = ch['rho'].values
                tvec = ch['time'].values
                cer_data['Ti'].append((rho, tvec,ch['Ti'].values, ch['Ti_err'].values))
                if 'omega' in ch and sys == 'tangential':
                    cer_data['omega'].append((rho, tvec,ch['omega'].values,ch['omega_err'].values ))

        #initial guess of carbon density
        if nC_guess is not None:  #if not availible assued Zeff = 2
            for sys in nC_guess['systems']:
                for ch in nC_guess[sys]:
                    if ch.attrs['Z'] != 6: continue #only carbon data
                    nz = None
                    #try to load the impurity data in this order                    
                    for c in ['_corr','']:
                        for s in ['impcon','int']:
                            if 'nimp_'+s+c in ch:
                                nz = ch['nimp_'+s+c].values
                                nz_err = ch['nimp_'+s+'_err'+c].values
                                break 
                        if nz is not None:
                            break
  
                        
                    if nz is not None:
                        #HFS data have usually lower quality
                        lfs = ch['R'].values > np.interp(ch['time'].values, centroid, Raxis)
                        cer_data['fC'].append((ch['rho'].values[lfs], ch['time'].values[lfs],nz[lfs],nz_err[lfs]))

        #slice all data in the clusters
        for it, _t in enumerate(centroid):
            lind = label == it
            T = nimp_data['time'][lind]
            Trange = T.min()-1e-3,T.max()+1e-3
            rho = beam_profiles['rho'][it]

            for name, data in cer_data.items():
                mean_rho, mean_data = [],[]
                if len(data):
                    tt = np.hstack([d[1] for d in data])
                    tind = (tt >= Trange[0])&(tt <= Trange[1])
                    for r,t,d,e in data: #rho, time, data for each channel
                        tind, _tind = tind[len(t):],tind[:len(t)]
                        valid = np.isfinite(e)&(e>0)
                        if any(_tind&valid):
                            mean_rho.append( r[_tind&valid].mean())
                            if np.sum(_tind&valid) == 1:
                                mean_data.append( d[_tind&valid].mean() ) 
                            else:
                                mean_data.append(np.average(d[_tind&valid],0,1/np.double(e[_tind&valid])**2)) 
                                
                    
                if len(mean_data) > 1:
                    sind = np.argsort(mean_rho)
                    prof_rho, prof_data = np.array(mean_rho)[sind], np.array(mean_data)[sind]
                    if name == 'fC':  #use carbon concetration - better properties for extrapolation!              
                        ne = np.interp(prof_rho, rho[::-1], beam_profiles['ne'][it][::-1])
                        prof_data = np.clip(prof_data/ne,0.001,1/6.-0.01)
                        #if np.any(prof_data > 0.1):
                            #embed()
                    _data = np.interp(rho,prof_rho, prof_data)  #TODO add radial averaging?? 
                    
                    if name == 'Ti':
                        edge = rho > min(0.95, np.max(prof_rho))
                        _data[edge] = beam_profiles['te'][it][edge] #replace edge ion temperature by electron temperature
                else:
                    if name == 'Ti':
                        _data = beam_profiles['te'][it] #use Te if Ti is not availible
                    elif name == 'omega':
                        _data = 0* rho #assume zero if unknown
                    elif name == 'fC':
                        if len(beam_profiles[name]): #use previous profile, if availible
                            _data = beam_profiles[name][-1].mean()+rho*0
                            #print(beam_profiles[name][-1].mean()+rho*0)
                        else: #else guess Zeff = 2
                            _data = rho*0+(2.-1.)/30. 

                beam_profiles[name].append(_data)
 
        beam_prof_merged = {k:np.hstack(p) for k,p in beam_profiles.items()}

        #####################  Calculate relative beam velocity ###############
        print_line( '  * Calculating '+imp+' density ...')
        TT = time()
        
        if imp != 'C6':
            print('\n\t\tWarning: %s impurity is assumed to be a trace'%imp) #not affecting beam attenuation or CX cross-section

        ab = nbi_dict['mass'] #bean neutral mass in AMU
        n_beam_spec = 3
        from scipy.constants import e,m_p
        # full eV/amu, divide later for half and third
        ab_ = ab[:,None] * np.arange(1,n_beam_spec+1)  # mass in amu assuming Deuterium
        qb = np.ones(n_beam_spec)  # charge in e
        # beam energies for all species on radial grid
        # energy is volts * charge/ mass (?)
        main_ab = ab.min()  #if H beam is used, assume H plasma
 
        # relative beam velocities on radial grid, cosine of angle between beam and
        # toroidal is R_tang/R_meas and toroidal velocity is omega*R_meas
 

        energy = np.outer(nbi_dict['volts'], e * qb)  # J
        vinj = np.sqrt(2 * energy / (ab_ * m_p)).T  # m/s
        beam_profiles.update({'vrel':[],'erel':[], 'dllencm':[]})
        for it,t in enumerate(centroid):
            # Calculate dl along the chord
            Rtang2 = np.square(nbi_dict['Rtang'])
            Rgrid = beam_profiles['Rmid'][it]

            Rmax = Rgrid[0]
            
            # Perform the calculation for each beam
            dllencm = (np.sqrt(Rgrid[:-1,None]**2-Rtang2)-np.sqrt(Rgrid[1:,None]**2-Rtang2)) * 1.0e2 
            # lenth in cm!

            omega = beam_profiles['omega'][it] # rad/s
            vtor = np.outer(omega, nbi_dict['Rtang'])  # m/s
            
            #make sure that there are no crazy values of rotation
            vtor = np.clip(vtor, -2e5, 2e5)

            # see CC notebook VII, pages 50-52, this is just the magnitude of the
            # velocity vector V_beam-V_plasma, note that the cosine of the angle
            # between V_beam and V_plasma, in the midplane, is R_tang/R_measurement

            vrel = vinj[None,:,:]-vtor[:,None,:]  # m/s
            erel = (0.5 * ab * m_p / e) * vrel ** 2/ab  # eV/amu,
            beam_profiles['dllencm'].append(dllencm)
            beam_profiles['erel'].append(erel)
            beam_profiles['vrel'].append(vrel)



        
        #####################  Calculate Beam attenuation ###############33
        # calculate concetration of impurities and main ions as self.frac

        # Change dens to cm-3, temp in eV, energy in eV/amu
        te = beam_prof_merged['te'] #eV
        dens = beam_prof_merged['ne']/1.e6 # cm^-3
        erel = np.vstack(beam_profiles['erel']).T #eV/amu
        vrel = np.vstack(beam_profiles['vrel']).T*100 #cm/s
        
        beam_prof_merged['erel'] = erel

        path = os.path.dirname(os.path.realpath(__file__))+'/openadas/' 

        files_bms = [path+'bms93#h_h1.dat',path+'bms93#h_c6.dat' ]
        files_bmp = [path+'bmp97#h_2_h1.dat',path+'bmp97#h_2_c6.dat' ]
        

        fC_beam = np.copy(beam_prof_merged['fC'])
        fD_beam = 1-fC_beam*6.  #deuterium concentration
        
        #normalise to calculate ion particle fraction        
        Zion = np.array([1,6])
        ion_frac = np.array([fD_beam,fC_beam])
        ion_frac /= ion_frac.sum(0)
        
        #calculate zeff
        zeff = np.dot(Zion**2, ion_frac)/np.dot(Zion, ion_frac)
        beam_prof_merged['zeff'] = zeff

        
        #effective density, based on adas/fortran/adas3xx/adas304/c4spln.for
        eff_dens = dens*zeff/Zion[:,None]


        # beam stopping rate
        bms = [read_adf21(f, erel, dens, te) for f in files_bms] # cm^3/s

        #n=2 excitation rate
        bmp = [read_adf21(f, erel, dens, te) for f in files_bmp] # cm^3/s

        #The stopping coefficients depend on the particular mixture of impurity ions present in the plasma
        #ion weights based on adas/fortran/adas3xx/adaslib/cxbms.for
        weights = ion_frac*Zion[:,None]
        weights/= weights.sum(0)
    
        bms_mix = np.sum(np.array(bms)* weights[:,None,None],0)
        bmp_mix = np.sum(np.array(bmp)* weights[:,None,None],0)
        
        sigma_eff = bms_mix / vrel  #cross-section cm^2

        

        

        #integrate beam attenuation
        beam_att,beam_att_err,n2frac = [],[],[]
        n = 0
        for it,t in enumerate(centroid):
            nR = beam_profiles['Rmid'][it].size
            
            #split the interpolated atomic data in timeslices
            n2frac.append(bmp_mix[:,:,n:n+nR])
            
            dlencm = beam_profiles['dllencm'][it] #cm
            dens = beam_profiles['ne'][it] / 1.0e6  # cm^-3
            dens_err = beam_profiles['ne_err'][it] / 1.0e6  # cm^-3
      
            datt = sigma_eff[:,:,n:n+nR] * dens #differential attenuation
            datt_err = sigma_eff[:,:,n:n+nR] * dens_err #differential attenuation

            #cumulative integrates data 
            datt_b = (datt[:,:,1:]+datt[:,:,:-1])/2
            datt_err_b = (datt_err[:,:,1:]+datt_err[:,:,:-1])/2
          
            lnbeam_att = np.cumsum(datt_b*dlencm.T[:,None],axis=-1)
            lnbeam_att = np.minimum(lnbeam_att, 10) #avoid under flow in exp
            lnbeam_att = np.dstack((np.zeros((nbeam, n_beam_spec)), lnbeam_att))
            lnbeam_att_err = np.cumsum(datt_err_b*dlencm.T[:,None],axis=-1)
            lnbeam_att_err = np.dstack((np.zeros((nbeam, n_beam_spec)), lnbeam_att_err))
            
            beam_att.append(np.exp(-lnbeam_att))
        
            # assume correlated error - ne is systematically higher/lower within the uncertainty
            beam_att_err.append(beam_att[-1] * lnbeam_att_err)  # keep in mind, that uncertaintes of all beams and species are correlated
            # assume 5% error in beam power
            beam_att_err[-1] = np.hypot(beam_att_err[-1], 0.05 * beam_att[-1])
            n += nR
            

        n2frac = np.dstack(n2frac)


        #####################  Calculate Halo  ##################
        #Based on R. McDermott PPCF 2018 paper
        te = beam_prof_merged['te'] #eV
        ti = beam_prof_merged['Ti'] #eV        
        ne = beam_prof_merged['ne'] / 1.0e6  # cm^-3
        v = vrel  # cm/s
        
        #root_dir = os.getenv('ADASCENT', '')
        #adf15d = root_dir + '/adf15/'
        #adf11d = root_dir + '/adf11/'
                

        #ionisation rate of deuterium
        SCDfile = path+'/scd96_h.dat'
        Se = read_adf11(SCDfile,  1, te, ne )#cm**3 s**-1)
        
        valid = (Se > 1e-10) & (beam_prof_merged['rho'] < 1)
        ionis_rate = (ne*Se)[valid] #1/s
        import scipy.constants as consts
        
        #simple neurals transport model for tangential LOS!!
        vth = np.sqrt((2*ti*consts.e)/(main_ab * consts.m_p))#m/s
        halo_std = vth[valid]/ionis_rate*100 #width of neutral gaussian distribution
        #correction for a finite transport of neutrals, assume that the extend in horisontal direction
        #will not change line integral (it will smear the measurements location)
        #but extend in vertical direction will reduce line integral
        #vertical extend of neutral distribution of height of sqrt(beam gaussian**2+height of neutral distribution**2)
        
          
        #30L beam shape B. Grierson  RSI 89 10D116 2018
        R_wall = 2.35#m
        nbi_width  = 10.36+(R_wall-beam_prof_merged['Rmid'][valid])*100*np.tan(0.0123)
        nbi_height = 24.68+(R_wall-beam_prof_merged['Rmid'][valid])*100*np.tan(0.0358)
        corr = np.hypot(nbi_height, halo_std/np.sqrt(2)) / nbi_height
 
        #CX crossection for D beam ions in D plasma
        #data from file qcx#h0_ory#h1.dat
        #E = [10., 50., 100., 200., 300., 500., 1000.]#energies(keV/amu)
        ##sigma = [7.95e-16 ,9.61e-17 ,1.54e-17 ,1.29e-18, 2.35e-19 ,2.20e-20, 6.45e-22 ]#total xsects.(cm2)
        #sigma = [7.70e-16, 6.13e-17, 9.25e-18, 8.44e-19, 1.63e-19, 1.63e-20, 5.03e-22] #n=1 crossection
        #sigmaDD = np.exp(np.interp(np.log(erel/1e3), np.log(E),np.log(sigma)))#cm2 
        
 
        #sigmaDD = np.exp(np.interp(np.log(erel/1e3), np.log(en_grid_fidasim[1:]), np.log(coeff[1:])))
        
        #cross-section from FIDASIM fitted by a polynomial
        p = [0.0051, -0.0552, 0.1017, -0.0383, -0.3721,-34.0339]
        sigmaDD = np.exp(np.polyval(p, np.log(erel/1e3)))

        #n=1 halo fraction, normalised to the total number of injected neutrals
        f0halo1 = np.zeros_like(sigmaDD)# magnitude consistent with Rachael's paper
        f0halo1[:,:,valid] = (sigmaDD*v)[:,:,valid]/(ionis_rate*corr)

    
        #Layman alpha emission
        PECfile = path+'/pec96#h_pju#h0.dat'
        PEC = read_adf15(PECfile, 1, te,ne)#ph cm**3 s**-1)
        A21 = 4.6986e+08#s^-1 einsten coefficient from https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        #n=2 halo fraction, magnitude consistent with Rachael's paper, normalised to the total number of injected neutrals
        f0halo2 = f0halo1*(PEC*ne)/A21
        #  Rachael used FIDASIM to calculate a spatial profile of these ions
        
        #embed()
  
        ######################### Calculate CX cross-sections  ############################# 
        zeff = beam_prof_merged['zeff']
        ti = beam_prof_merged['Ti'] # ev
        ne = beam_prof_merged['ne'] / 1.0e6  # cm^-3
        erel = beam_prof_merged['erel']   #eV/amu
 
        line_ids = []
        for sys in systems:
            for ch in nimp[sys]:
                if ch['int'].attrs['line'] not in line_ids:
                    line_ids.append(ch['int'].attrs['line'])  #NOTE assume that all data are from the same line
      
        
        for line_id in line_ids:
            #BUG it is not very efficient way, everything will be calculated for every channel and at the end
            #I will pick up just the channels with the right line_id
                
          
            if line_id  in ['B V 7-6','C VI 8-7','He II 4-3','Ne X 11-10','N VII 9-8']:
                #CX with beam ions
                adf_interp = read_adf12_aug(path,line_id, n_neut=1)
                qeff = adf_interp(zeff, ti, ne, erel)
                adf_interp = read_adf12_aug(path,line_id, n_neut=2)
                qeff2 = adf_interp(zeff, ti, ne, erel)
                
                #CX with beam halo
                adf_interp = read_adf12_aug(path,line_id, n_neut=1, therm=True)
                qeff_th = adf_interp(zeff, ti, ne).T
                adf_interp = read_adf12_aug(path,line_id, n_neut=2, therm=True)
                qeff2_th = adf_interp(zeff, ti, ne).T
                

                
            elif line_id in ['B V 3-2','C VI 3-2','OVIII 3-2','He II 2-1' ,'NVII 3-2','Ne X 4-3']:
                # Define CX rate coefficient fitted parameters used to construct rate coefficient
                # From /u/whyte/idl/spred/spred_cx.pro on GA workstations

                if imp == 'He2':
                    coeffs_energy = [1.1370244, 0.0016991233, -0.00015731925, 6.4706743e-7, 0.0]
                if imp == 'C6': #with lmixing
                    coeffs_energy = [0.98580991, 0.046291358, -0.0010789930, 8.9991561e-6, -2.7316781e-8]
                if imp == 'N7': #with lmixing, linear combination of O and C
                    coeffs_energy_c = [0.98580991, 0.046291358, -0.0010789930, 8.9991561e-6, -2.7316781e-8]
                    coeffs_energy_o = [1.1975958, 0.025612778, -0.00048135883, 2.1715730e-6, 0.0]
                    coeffs_energy = np.add(coeffs_energy_c, coeffs_energy_o) / 2.0
                if imp == 'O8':
                    coeffs_energy = [1.1975958, 0.025612778, -0.00048135883, 2.1715730e-6, 0.0]
                if imp == 'B5':
                    coeffs_energy = [ 1.15113500e+00,  3.00654905e-02, -6.68824080e-04,  4.35114750e-06,-9.42988278e-09]
                if imp == 'Ne10': #hope that it is not far from 08
                    coeffs_energy = [1.1975958, 0.025612778, -0.00048135883, 2.1715730e-6, 0.0]
                # CX rate coefficient for given energy (ph cm**3/s)
                qeff = 10 ** np.polyval(coeffs_energy[::-1],erel/1e3)* 1.0e-8
                qeff2 = qeff_th = qeff2_th = 0
            
            elif imp in ['Ca18','Ar18','Ar16','F9',  'B5', 'Li3']:

                atom_files = { 'Ca18': ('qef07#h_arf#ar18.dat', 'qef07#h_arf#ar18_n2.dat'),
                               'Ar18': ('qef07#h_arf#ar18.dat', 'qef07#h_arf#ar18_n2.dat'),
                               'Ar16': ('qef07#h_arf#ar16.dat','qef07#h_arf#ar16_n2.dat'),
                                 'F9': ('qef07#h_arf#f9.dat','qef07#h_en2_arf#f9.dat'),
                                 'B5': ('qef93#h_b5.dat','qef97#h_en2_kvi#b5.dat'),
                                'Li3': ('qef97#li_kvi#li3.dat',None)}
 
                blocks = {'Ar18':{'15-14':[5,2]}, 'Ca18':{'15-14':[5,2]},'Ar16':{'14-13':[5,2]},
                          'B5':{'3-2':[1,1]}, 'F9':{'10-9':[2,2]}, 'Li3':{'2-1':[1,0], '3-1':[9,0]}}
                
                
                tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', line_id)
                transition = tmp.group(3)             
                
                #unavailible
                qeff_th = qeff2_th = qeff2 = 0
                file1, file2 = atom_files[imp]
                block1, block2 = blocks[imp][transition]
                qeff  = read_adf12(path+file1,block1, erel, ne, ti, zeff)
                if file2 is not None:
                    qeff2 = read_adf12(path+file2,block2, erel, ne, ti, zeff)
 
                
    
            else:
                raise Exception('CX data for line %s was not found'%line_id)

            
                
                
            ## cm**3/s to m**3/s and  per ster. like CER
            qeff  /= 1e6 * 4.0 * np.pi
            qeff2 /= 1e6 * 4.0 * np.pi


            ## cm**3/s to m**3/s and  per ster. like CER
            qeff_th  /= 1e6 * 4.0 * np.pi
            qeff2_th /= 1e6 * 4.0 * np.pi
        
            ######################### Calculate Impurity density  #############################
            
            isp = np.arange(3)+1  # mass multipliers for beam species
            eV_to_J = consts.e
            mp = consts.m_p



            einj = nbi_dict['volts']  # V
            # Velocity of beam species
            pwrfrc = nbi_dict['pow_frac'].T[0]
            vinj = np.sqrt(2.0 * einj * eV_to_J /(ab_.T * mp))  # m/s
            nb0 = pwrfrc / (einj * eV_to_J * vinj / isp[:, None])  # 1/m/W must be multipled by power
            qeff = qeff * (1 - n2frac) + n2frac * qeff2 #  qeff from n=1 and n=2

            #BUG inaccurate any geometry correction for spatial distribution of halo
            #works only in the limit of high density when the collisions prevent spreading the halo
            
            # Get beam attenuation data and the qeff values
            ne = beam_prof_merged['ne'] / 1.0e6  # cm^-3
            nD = (1-beam_prof_merged['fC']*6)*ne  # cm^-3
            qeff+= (nD* qeff_th)*f0halo1 #small, negligible
            qeff+= (nD*qeff2_th)*f0halo2 #comparable with qeff2



            #embed()
            #printe('BUGGGG')
            #qeff[:] = qeff.mean((0,2))[None,:,None]        
            
            #keys = ['ne',  'te',  'Ti', 'omega', 'fC',] 
            #import matplotlib.pylab as plt
            #C = plt.cm.jet(np.linspace(0,1,100))

            #f,ax = plt.subplots(2,4,sharex=True)
            #ax = ax.flatten()
            #for j in range(100):
                #for i,k in enumerate(keys):
                    ##print()
                    #ax[i].set_title(k)
                    #ax[i].plot(beam_profiles['rho'][j], beam_profiles[k][j],c=C[j])
            
            #ax[5].set_title('atten')
            #for j in range(100):
                ##print(beam_att[it].shape)
                #ax[5].plot(beam_profiles['rho'][j],  beam_att[j][0,0 ].T,c=C[j])
            #ax[6].set_title('qeff')
            #n = 0

            #for it,t in enumerate(centroid):
                #nt = len(beam_profiles['rho'][it])
                #tind = slice(n,n+nt)
                #ax[6].plot(beam_profiles['rho'][it], qeff[0,0,tind].T,c=C[it] )
                #n+=nt


            #f,ax = plt.subplots(2,4,sharex=True)
            #ax = ax.flatten()
            #ax[5].set_title('atten')
            #ax[6].set_title('qeff')
            #n = 0
            #R = np.linspace(1.7,2.3,20)

            #for it,t in enumerate(centroid):
                #Rmid = beam_profiles['Rmid'][it]
                #nr = len(Rmid)
                #C = plt.cm.jet(np.linspace(0,1,20))

                #for i,k in enumerate(keys):
                    #ax[i].set_title(k)
                    #ax[i].scatter(np.ones_like(R)*t,np.interp(R,Rmid[::-1],beam_profiles[k][it][::-1]),c=C)
            
                #ax[5].scatter(np.ones_like(R)*t,np.interp(R,Rmid[::-1],beam_att[it][0,0][::-1]),c=C)
                #tind = slice(n,n+nr)
                #ax[6].scatter(np.ones_like(R)*t, np.interp(R,Rmid[::-1],qeff[0,0,tind][::-1]*1e15),c=C)
                #n+=nr

            #plt.show()
            
            

            #rho_ne,tvec_ne,data_ne,err_ne = [],[],[],[]
            #for sys in TS['systems']:
                #if sys not in TS: continue 
                #t = TS[sys]['time'].values
                #te = TS[sys]['ne'].values
                #e = TS[sys]['ne_err'].values
                #r = TS[sys]['rho'].values
                #t = np.tile(t, (r.shape[1],1)).T
                #ind = np.isfinite(e)|(te>0)
                #tvec_ne.append(t[ind])
                #data_ne.append(te[ind])
                #err_ne.append(e[ind])
                #rho_ne.append(r[ind]) 

            #rho_ne  = np.hstack(rho_ne)
            #tvec_ne = np.hstack(tvec_ne)
            #data_ne = np.hstack(data_ne)
            #err_ne  = np.hstack(err_ne)

                
            #impurity densities foe all channels
            nz = np.zeros_like(nimp_data['int'])
            nz_err = np.ones_like(nimp_data['int'])*np.inf
            n = 0
            beam_fact = beam_data['beam_geom']*beam_data['beam_pow']
        
            for it,t in enumerate(centroid):
                
                Rmid = beam_profiles['Rmid'][it]
                nt = len(Rmid)
                tind = slice(n,n+nt)
                ind = label == it
                n += nt
        

                R_clip = np.minimum(nimp_data['R'][ind],  Rmid[0])  #extrapolate by a constant on the outboard side
                # sum over beam species crossection before interpolation
                try:
                    denom_interp = interp1d(Rmid, np.sum(nb0.T[:,:,None] * beam_att[it] * qeff[:,:,tind], 1))  # nR x nbeam
                except:
                    embed()

                #try:
                # uncertainties in beam_att_err between species are 100% correlated, we can sum them
                denom_err_interp = interp1d(Rmid, np.sum(nb0.T[:,:,None] * beam_att_err[it] * qeff[:,:,tind], 1))  # nR x nbeam
                #except:
                    #embed()
                denom = np.sum(beam_fact[:,ind]*denom_interp(R_clip),0)
                denom_err = np.sum(beam_fact[:,ind]*denom_err_interp(R_clip),0)
                
                #sometimes is power observed by vertical core system is zero, but intensity is nonzero
                invalid = (denom == 0)|np.isnan(nimp_data['int_err'][ind])|(nimp_data['int'][ind]==0)
                if np.any(invalid):
                    ind[ind] &= ~invalid
                    denom=denom[~invalid] 
                    denom_err=denom_err[~invalid]
                    
                nz[ind] = nimp_data['int'][ind]/denom
                nz_err[ind] = nz[ind] * np.hypot(nimp_data['int_err'][ind] / (1+nimp_data['int'][ind]), denom_err / denom)
                nz_err[ind] *= np.sign(nimp_data['int_err'][ind])  #suspicious channels have err < 0 
        
            #interp = NearestNDInterpolator(np.vstack((tvec_ne,rho_ne)).T, np.copy(data_ne))
            #embed()

            #fill the xarray database with calculated impurity densities
            n = 0 
            for diag in systems:
                for ch in nimp[diag]:
                    if line_id != ch['int'].attrs['line']:
                        n+=1
                        continue
                    #interp.values[:] = np.copy(data_ne) 
                    #ne = interp(np.vstack((ch['time'].values, ch['rho'].values)).T)
                    #interp.values[:] = np.copy(err_ne) 
                    #ne_err = interp(np.vstack((ch['time'].values, ch['rho'].values)).T)
                    ch['nimp_int'] = ch['nimp'].copy()
                    ch['nimp_int'].values = nz[data_index[n]]#/ne
                    ch['nimp_int_err'] = ch['nimp'].copy()
                    ch['nimp_int_err'].values = nz_err[data_index[n]]
                    #ch['nimp_int_err'].values = nz[data_index[n]]/ne*np.hypot(nz_err[data_index[n]]/nz[data_index[n]],ne_err/ne)
                    n += 1
                    
         

        #T = np.linspace(3.4,4.6,1000)
        ##print(T)
        #B = '210'
        #print('B210=array([')
        #for name in ['T%.2d'%i for i in range(100)]:
            #I = {B+'L' : np.nan, B+'R' : np.nan}
            #for b in [B+'L',B+'R']:
                #for ch in nimp['tangential']:
                    #if ch.attrs['channel'] == name+'_'+b:
                        #I[b] = np.interp(T, ch['time'].values, ch['nimp_int'].values)
                        #I[b+'R'] = ch['R'].values.mean()
 
            #if np.all(np.isfinite(I[B+'L']/I[B+'R'])):
                #print('[',name, (I[B+'LR']+I[B+'RR'])/2,',', np.mean(I[B+'L']/I[B+'R']),'],')
        #print('])')
        
        ##embed()
        
        #B = '30'
        #print('B30=array([')
        #for name in ['T%.2d'%i for i in range(100)]:
            #I = {B+'L' : np.nan, B+'R' : np.nan}
            #for b in [B+'L',B+'R']:
                #for ch in nimp['tangential']:
                    #if ch.attrs['channel'] == name+'_'+b:
                        #I[b] = np.interp(T, ch['time'].values, ch['nimp_int'].values)
                        #I[b+'R'] = ch['R'].values.mean()
                        ##plt.plot()
 
            #if np.all(np.isfinite(I[B+'L']/I[B+'R'])):
                ##plt.figure()
                ##plt.plot(T,I[B+'L'],T,I[B+'R'])
                #print('[',name, (I[B+'LR']+I[B+'RR'])/2,',', np.mean(I[B+'L']/I[B+'R']),'],')
        #print('])')
        
        
        #B = '330'
        #print('B330=array([')
        #for name in ['T%.2d'%i for i in range(100)]:
            #I = {B+'L' : np.nan, B+'R' : np.nan}
            #for b in [B+'L',B+'R']:
                #for ch in nimp['tangential']:
                    #if ch.attrs['channel'] == name+'_'+b:
                        #I[b] = np.interp(T, ch['time'].values, ch['nimp_int'].values)
                        #I[b+'R'] = ch['R'].values.mean()
                        ##plt.plot()
 
            #if np.all(np.isfinite(I[B+'L']/I[B+'R'])):
                ##plt.figure()
                ##plt.plot(T,I[B+'L'],T,I[B+'R'])
                #print('[',name, (I[B+'LR']+I[B+'RR'])/2,',', np.mean(I[B+'L']/I[B+'R']),'],')
        #print('])')
        

        print('\t done in %.1fs'%(time()-TT))
        
        return nimp
      
 
    def load_nimp_impcon(self, nimp,load_systems, analysis_type,imp):
        #load IMPCON data and split them by CER systems
        #in nimp are already preload channels
        #currently there is no SPRED density from impcon

        load_systems = [sys for sys in load_systems if 'SPRED' not in sys]
            
     
        if len(load_systems) == 0:
            return nimp
        
        
        ##############################  LOAD DATA ######################################## 

        print_line( '  * Fetching IMPCON data from %s ...'%analysis_type.upper())
        T = time()

        
        imp_path = '\%s::TOP.IMPDENS.%s.'%('IONS',analysis_type) 
        nodes = 'IMPDENS', 'ERR_IMPDENS', 'INDECIES', 'TIME'
        TDI = [imp_path+node for node in nodes]
        
        
        #array_order in the order as it is stored in INDECIES
        TDI += ['\IONS::TOP.CER.CALIBRATION:ARRAY_ORDER']
        
        #fast fetch
        nz,nz_err, ch_ind, tvec,array_order = mds_load(self.MDSconn, TDI, 'IONS', self.shot)
        if len(nz) == 0:
            raise Exception('No IMPCON data')

        nz_err[(nz<=0)|(nz > 1e20)] = np.inf
        ch_ind = np.r_[ch_ind,len(tvec)]
        ch_nt = np.diff(ch_ind)
        try:
            array_order = [a.decode() for a in array_order]
        except:
            pass
        array_order = [a[0]+('0'+a[4:].strip())[-2:] for a in array_order]
        
        for sys in load_systems:
            for ch in nimp[sys]:
                ch_name = ch.attrs['channel'].split('_')[0] 
                ich = array_order.index(ch_name)
                ind = slice(ch_ind[ich],ch_ind[ich+1])
                
                if ch_ind[ich]==ch_ind[ich+1]:
                    continue
                
                
                #merge impcon and CER timebases
                tch = np.round((ch['time'].values-ch['stime'].values/2)*1e3,2)
                timp = np.round(tvec[ind],2)#round to componsate for small rounding numerical errors
                #sometimes there are two measuremenst with the same time (175602, AL, T25)

                jt = 0
                t2 = timp[jt]
                nz_, nzerr_ = np.zeros_like(tch), -np.ones_like(tch)*np.inf
                for i, t1 in enumerate(tch):
                    while(t2 < t1):
                        jt += 1
                        if jt < timp.size:
                            t2 = timp[jt]
                        else:
                            break
                    if t2 == t1:
                        nz_[i], nzerr_[i] = nz[ind][jt],nz_err[ind][jt] 
                        jt += 1
                        if jt < timp.size:
                            t2 = timp[jt]
                    #else - point is missing in IMPCON
                
 
                #channel was not included in IMPCON analysis
                if not any(nz_ > 0):
                    continue
                
     
                disableChanVert = 'V03', 'V04', 'V05', 'V06', 'V23', 'V24'
                if 162163 <= self.shot <= 167627 and ch_name in disableChanVert:
                    nzerr_[:] = np.infty
                #corrections of some past calibration errors
                if imp == 'C6' and ch_name == 'T07' and self.shot >= 158695:
                    nz_ *= 1.05
                if imp == 'C6' and ch_name == 'T23' and  158695 <= self.shot < 169546:
                    nz_ *= 1.05
                if imp == 'C6' and ch_name == 'T23' and  165322 <= self.shot < 169546:
                    nz_ *= 1.05
                if imp == 'C6' and ch_name in ['T09','T11','T13','T41','T43','T45'] and 168000<self.shot<172799:
                    nz_ /= 1.12
            
                ch['nimp_impcon'] = ch['nimp'].copy()
                ch['nimp_impcon'].values = nz_  
                ch['nimp_impcon_err'] = ch['nimp_err'].copy()
                ch['nimp_impcon_err'].values = nzerr_
         


        print('\t done in %.1fs'%(time()-T))
        return nimp

       
 
    def load_nimp(self,tbeg,tend,systems, options):        

        selected,analysis_types = options['Analysis']
   
        rcalib = options['Correction']['Relative calibration'].get()
        #calculate impurity density from intensity directly
        try:
            SOL_reflections = options['Correction']['Wall reflections'].get()
        except:
            SOL_reflections = False
            
        analysis_type = self.get_cer_types(selected.get(),impurity=True)
        if analysis_type is None:
            #try at least of CER data exists
            analysis_type = self.get_cer_types(selected.get(),impurity=False)

        
        #show the selected best analysis in the GUI??
        selected.set(analysis_type[3:])

        suffix = ''
        imp = 'C6'
   
        if 'Impurity' in options and options['Impurity'] is not None:
            if isinstance(options['Impurity'], tuple):
                imp = options['Impurity'][0].get()
            else:
                imp = options['Impurity']
            suffix += '_'+imp
        
        if 'SPRED' in systems:
            systems.remove('SPRED')
            systems.append('SPRED_'+imp)

        
        nz_from_intens = False
        if 'nz from CER intensity' in options['Correction']:
            nz_from_intens = options['Correction']['nz from CER intensity'].get() 
            
        rm_after_blip=False
        if 'remove first data after blip' in options['Correction']:
            rm_after_blip = options['Correction']['remove first data after blip'].get() 
 

        #load from catch if possible
        self.RAW.setdefault('nimp',Tree())

        nimp = self.RAW['nimp'].setdefault(analysis_type+suffix ,Tree() )
        self.RAW['nimp'].setdefault('NBI',Tree())

        
        #which cer systems should be loaded
        load_systems_nz = []
        load_systems_intens = []

        nimp_name = 'nimp_int' if nz_from_intens else 'nimp_impcon'
        for sys in systems:
            if sys not in nimp or len(nimp[sys])==0:
                load_systems_nz.append(sys)
                load_systems_intens.append(sys)
            elif not any([nimp_name in ch for ch in nimp[sys]]): #nz data not loaded
                load_systems_nz.append(sys)
  
        nimp['systems'] = systems
        nimp.setdefault('rel_calib_'+nimp_name, rcalib)
        nimp.setdefault('diag_names',Tree())
        

        def return_nimp(nimp):
            #function returning the requested impurity density
            nz_from_intens = options['Correction']['nz from CER intensity'].get() 
            
            nimp_name = 'nimp_int' if nz_from_intens else 'nimp_impcon'

            #if SOL_reflections:
                #print('BUG!!!!!! temporary fix')
                #cer = self.load_cer(0,10, systems)
                #for ch in nimp['tangential']:
                    #for cch in cer['tangential']:
                        #if ch.attrs['channel'][:3] == cch.attrs['channel']:
                            #ch['int'].values = np.interp(ch['time'].values, cch['time'].values, cch['int'].values )
                            #ch['int_err'].values = np.interp(ch['time'].values, cch['time'].values, cch['int_err'].values )

            
            
            
            #rho coordinate of the horizontal line, used later for separatrix aligment 
            #if 'horiz_cut' not in nimp or 'EQM' not in nimp or nimp['EQM']['id'] != id(self.eqm) or nimp['EQM']['ed'] != self.eqm.diag:
            R = np.linspace(1.4,2.5,100)
            rho_horiz = self.eqm.rz2rho(R, np.zeros_like(R), coord_out='rho_tor')
            nimp['horiz_cut'] = {'time':self.eqm.t_eq, 'rho': np.single(rho_horiz), 'R':R}
            
            #build a new dictionary only with the requested and time sliced channels            
            nimp_out = {'systems':systems,'diag_names':Tree()}
            for sys in systems:
                #SPRED data will be always from intensity
                nimp_out[sys] = []
                #list only diag_names which are actually loaded
                nimp_out['diag_names'][sys] = []
                for ch in nimp[sys]:
                    suffix = ''
                    if rcalib and nimp_name+'_corr' in ch: #if the correction has not failed
                        suffix = '_corr'
                    #set requested density timetraces in each channel
                    if nimp_name+suffix in ch:
                        try:
                            ch = ch.sel(time=slice(tbeg,tend))
                        except:
                            print(ch)
                        if len(ch['time']) == 0:
                            continue
                        ch['nimp'] = ch[nimp_name+suffix].copy()
                        nimp_err = np.copy(ch[nimp_name+'_err'+suffix].values)
                        if rm_after_blip:#show points but their will be disabled
                            nimp_err[ch['beam_swiched_on'].values&(nimp_err > 0)] *= -1
                        ch['nimp_err'].values = nimp_err
                        
                        if ch.attrs['name'] not in nimp_out['diag_names'][sys]:
                            nimp_out['diag_names'][sys].append(ch.attrs['name'])
                        nimp_out[sys].append(ch)
  

            return nimp_out

        #skip loading when already loaded,
        same_eq = 'EQM' in nimp and nimp['EQM']['id'] == id(self.eqm) and nimp['EQM']['ed'] == self.eqm.diag

        if len(load_systems_nz) == 0  and  (not nz_from_intens or same_eq) and (not rcalib or nimp['rel_calib_'+nimp_name]):
            #return corrected data if requested
            nimp = self.eq_mapping(nimp)
            return return_nimp(nimp)
        
        
        #update equilibrium of catched channels
        nimp = self.eq_mapping(nimp)
 
        #first load radiation data from MDS+
        if len(load_systems_intens):
            nimp = self.load_nimp_intens(nimp,load_systems_intens, analysis_type,imp)
 
        #load either from IMPCON or calculate from CX intensity
        if nz_from_intens:
            
            #try to load C6 for Zeff estimate needed for CX crossections
            try:
                tmp_imp = options.get('Impurity','C6') 

                options['Correction']['nz from CER intensity'].set(0) 
                options['Correction']['Relative calibration'].set(1)
                options['Impurity'] = 'C6'
                systems_impcon = [sys for sys in systems if not 'SPRED' in sys]
                
                #special case when only SPRED data are loaded
                if len(systems) == 0:
                    systems_impcon = ['tangential']
                    selected.set('auto')
   
                #other cases,  IMPCON density is used as initial guess
                nimp0 = None

                try:
                    nimp0 = self.load_nimp(tbeg,tend,systems_impcon,options) 
                except Exception as e:
                    printe('Error in loading of nC from IMPCON: '+str(e))
                    try:
                        if selected.get() != 'auto':
                            #try to load from AUTO edition
                            selected.set('auto')
                            nimp0 = self.load_nimp(tbeg,tend,systems_impcon,options) 
                    except:
                        printe('Error in loading of nC from IMPCON AUTO edition: '+str(e))
                    #finally:
                        #print('selected',analysis_type, analysis_type[3:] )
                #set back changes made for IMPCON density fetch
                selected.set(analysis_type[3:])
                nimp['systems'] = systems

                nimp = self.calc_nimp_intens(tbeg,tend,nimp,systems,imp, nimp0, options)

            finally:
                options['Correction']['nz from CER intensity'].set(nz_from_intens) 
                options['Correction']['Relative calibration'].set(rcalib)
                options['Impurity'] = tmp_imp
        else:
            #load just the impurity density from MDS+
            try:
                nimp = self.load_nimp_impcon(nimp,load_systems_nz, analysis_type,imp)
                #SPRED must be calculated from intensity 
                if 'SPRED_'+imp in load_systems_nz:
                    try:
                        nimp = self.calc_nimp_intens(tbeg,tend,nimp,['SPRED_'+imp],imp, nimp, options)
                        #it needs to be saved under the same name as data from CER
                        for ch in nimp['SPRED_'+imp]:
                            ch[nimp_name] = ch['nimp_int'] 
                            ch[nimp_name+'_err'] = ch['nimp_int_err'] 
 
                    except Exception as e:
                        printe('SPRED density calculation failed'+str(e))
                    
            except Exception as e:

                printe('Error in loading of impurity density from IMPCON: '+str(e))
                print('Impurity density will be calculated from CER intensity')
                options['Correction']['nz from CER intensity'].set(1) 

                nimp0 = None
                nimp_name = 'nimp_int' 

                #calculate impurity density
                nimp = self.calc_nimp_intens(tbeg,tend,nimp,systems,imp, nimp0, options)
                
 
          
        ##update uncorrected data
        diag_names = sum([nimp['diag_names'][diag] for diag in nimp['systems'] if diag in nimp['diag_names']],[])
        impurities = np.hstack([[ch['nimp'].attrs['impurity'] for ch in nimp[s] if ch.attrs['system'] == s] for s in nimp['systems']])
        unique_impurities = np.unique(impurities)
        T = time()
        all_channels = [ch for s in nimp['systems'] for ch in nimp[s]]
   
        #reduce discrepancy between different CER systems
        if rcalib and len(diag_names) > 1 and len(unique_impurities)==1 and 'tangential' in nimp['systems']:
            print( '\t* Relative calibration of beams  ...')
            NBI = self.RAW['nimp']['NBI']
            
            #treat tangential and vertical system and edge corrections independently
            groups = []
            for ch in all_channels:
                sys = ch.attrs['system'][0].upper()
                edge = 'e' if ch.attrs['edge'] else ''
                groups+= [sys+b+edge for b in ch['beams'].values]
            #embed()
            groups = list(np.unique(groups))
            if 'V210R' in groups:
                groups.remove('V210R') #this is not a primary beams!

            #determine "calibrated density source"
            voltages, times = {},{}
            for name,beam in NBI.items():
                if 'fired' in beam and beam['fired'] or 'beam_fire_time' in beam:
                    print('\t\tinfo beam:%s\tV=%dkV\tsingle beam time = %.2fs'%(name,beam.get('volts',0)/1e3,beam.get('beam_fire_time',np.nan)))

            #if beam 30L is on ~80kV use it for cross-calibration
            if 'T30L' in groups and NBI['30L']['fired'] and ( 77 < NBI['30L']['volts']/1e3 < 83) and NBI['30L']['beam_fire_time'] > .5:
                print('\t* Using beam 30L for cross calibration')
                calib_beam = 'T_30L'

            elif 'T30R' in groups and NBI['30R']['fired'] and (74 < NBI['30R']['volts']/1e3 < 83) and NBI['30R']['beam_fire_time'] > .7:
                print('\t\tUsing beam 30R for cross calibration')
                calib_beam = 'T_30R'
   
            else:
                if 'T30L' in groups:
                    calib_beam = 'T_30L'
                else: #anything else
                    calib_beam = ([b for b in groups if b[-1] != 'e' and b[0]!='S']+['T30L'])[0]
                    calib_beam = calib_beam[0]+'_'+calib_beam[1:]
                printe('\t\tNo reliable beam for cross calibration, guessing.. using '+calib_beam)
            #calib_beam = 'T_30B'
            #cross-calibrate other profiles
            calib = {'t':[],'r':[],'n':[],'nerr':[],'f':[]}
            other = {'t':[],'r':[],'n':[],'nerr':[],'f':[]}
            
            #iterate over all channels
            for ch in all_channels:
                 #ch._variables['time']._data.array._data
                t = ch['time'].values  #normalize a typical time range with respect to typical rho range)
                beam_sys = ch['diags'].values[0].split()[0]
                #print(beam_sys)
                
                sys = ch.attrs['system'][0].upper()
                edge = 'e' if ch.attrs['edge'] else ''

                if not nimp_name in ch: continue
                nz  = ch[nimp_name].values/1e18 #to avoid numerical overflow with singles
                nz_err = ch[nimp_name+'_err'].values/1e18 #to avoid numerical overflow with singles
                rho = ch['rho'].values
                R = ch['R'].values

    
                ind =  (nz > 0)&(nz_err > 0)&(nz_err < 1e2)&(rho < .9)&(R > 1.7) #do not include data in the pedestal and hfs channels
                if any(ind):
                    beam_frac = np.zeros((len(groups), len(t)))
                    beam_frac_= ch['beam_frac'].values
                    for ib,b in enumerate(ch['beams'].values):
                        if sys+b+edge in groups:
                            beam_frac[groups.index(sys+b+edge)] = beam_frac_[ib]
                            
                    data = {'t': t, 'r': R, 'n': nz, 'nerr': nz_err, 'f': beam_frac}
                    data_set = calib if calib_beam == beam_sys else other
                    for key,val in data.items():
                        data_set[key].append(val[...,ind])
                        
            if len(calib['t']) == 0 or len(other['t']) == 0:
                if len(calib['t']) == 0:
                    printe('unsuccessful... no calib beam ')
                else:
                    printe('unsuccessful... no other beams ')

                options['Correction']['Relative calibration'].set(0) 
                rcalib = False
                nimp = return_nimp(nimp)
                return nimp
            #embed()
            #print(groups)
            ind_calib = groups.index(calib_beam.replace('_',''))
            calib = {n:np.hstack(d).T for n,d in calib.items()}
            other = {n:np.hstack(d).T for n,d in other.items()}

            #get indexes of nearest points
            interpn = NearestNDInterpolator(np.c_[calib['t'],calib['r']],np.arange(len(calib['t'])))
            near_ind = interpn(np.c_[other['t'],other['r']])
            dist = np.hypot(calib['t'][near_ind]-other['t'],calib['r'][near_ind]**2-other['r']**2)
            nearest_ind = dist < .1 #use only a really close points 
            #weight = np.exp(-dist**2/.1**2)
            #if some system was nearly ignored, allow a bigger distance
            ignored_sys = (other['f'][nearest_ind]>.4).sum(0) < (other['f']>.4).sum(0)*0.2
            ignored_ind = np.any(other['f'][:,ignored_sys] > .4,1)
            nearest_ind[ignored_ind] |= dist[ignored_ind] < .2
            
 
            interpl = LinearNDInterpolator(np.c_[calib['t'],calib['r']],np.copy(calib['n']))
            n_calib = interpl(np.c_[other['t'],other['r']][nearest_ind])
            interpl.values[:] = calib['nerr'][:,None]
            nerr_calib = interpl(np.c_[other['t'],other['r']][nearest_ind])

            #embed()
            ##n_calib2 = calib['n'][near_ind[nearest_ind]]

            #plt.plot(n_calib)
            #plt.plot(calib['n'][near_ind[nearest_ind]],':')
            #plt.plot(other['n'][nearest_ind],'--')
            #plt.show()
            

            #extrapolate by a nearest value
            nerr_calib[np.isnan(n_calib)] = calib['nerr'][near_ind[nearest_ind]][np.isnan(n_calib)]
            n_calib[np.isnan(n_calib)] = calib['n'][near_ind[nearest_ind]][np.isnan(n_calib)]

            #use least squares to find the cross-calibration factors for each beam
            N = other['n'][nearest_ind]/n_calib
            E = N*np.hypot(nerr_calib/n_calib, other['nerr'][nearest_ind]/other['n'][nearest_ind])
             
            F = other['f'][nearest_ind] #beam fraction
            N -= F[:,ind_calib] #remove contribution from calibration beam, if there is any (175473)
            F = np.delete(F,ind_calib, axis = 1) #delete calibration beam

            #find optimal values of cross-calibration
            #initial guess using linear least squares
            C0 = np.linalg.lstsq(F/E[:,None] , N/E,rcond=None )[0]
            #use robust least squares
            from scipy.optimize import least_squares
            fun = lambda x: np.dot(F/E[:,None],x)- N/E
            out = least_squares(fun,C0,loss='cauchy')
            C = out.x
            
            #embed()

            #in some cases when beam 30L+30R+30B are used and 30R is poorly constrained, it can fail (184773)
            #plt.errorbar(np.arange(len(N)), N, E);plt.show()
                
            #plt.plot(n_calib)
            #plt.plot( other['n'][nearest_ind],'--') 
            ##plt.plot( calib['n'][near_ind[nearest_ind]],':')
            #plt.plot( np.dot(F,C)* n_calib,':')
            
            #plt.show()
            
            #ind = F[:,0]>.3
            #plt.errorbar(np.where(ind)[0], N[ind],E[ind])
            #plt.plot(np.where(ind)[0], np.dot(F,C)[ind])
            #plt.show()
            
            
            
            
            C = np.insert(C,ind_calib,1)  #calibration for calib_beam is forced to be one
            C = {b:c for b,c in zip(groups, C)}
            
            #embed()
 
            #embed()
            #ind = np.argmax(other['f'],1)
            #for i in range(len(other['f'].T)):
                    #plt.plot(other['t'][ind==i], other['r'][ind==i],'o')
            #plt.plot(calib['t'], calib['r'],'x')
            #plt.plot(np.vstack((calib['t'][near_ind[nearest_ind]],other['t'][nearest_ind])),
                #np.vstack((calib['r'][near_ind[nearest_ind]],other['r'][nearest_ind])))
            #plt.show()

            for b,c in C.items():
                if c > 0: print('\t\t correction '+b+': %.2f'%(1/c))
                
                
            #if '30L' in NBI and NBI['30L']['fired'] and ( 77 < NBI['30L']['volts']/1e3 < 83) and NBI['30L']['beam_fire_time'] > .5:
                ##print('saing')
                #with open('beams_corrections_%s_30_noG.txt'%('int' if nz_from_intens else 'impcon'), "a") as file:                
                    #corr_str = str(self.shot)+';'
                    #for b in ['T30L', 'T30R','T330Le','T330Re', 'T210L','T210R', 'V330L', 'V330R','V330Le', 'V330Re','S30L','S30R']:
                        #if b in C and C[b] > 0:   
                            
                            #corr_str+= '%.3f;'%(1/C[b])
                            #b = b.strip('e') 
                            #corr_str+= '%.3f;'%(NBI[b[1:]].get('volts',0)/1e3)
                            #corr_str+= '%.3f;'%(NBI[b[1:]].get('beam_fire_time',np.nan))

                        #else:
                            #corr_str += '\t;\t;\t;'
                        
                    #file.write(corr_str+'\n')
                    #print('==============================='+corr_str)
                ##except:
                
                
   
            #apply correction, store corrected data 
            for ch in all_channels:
                if not nimp_name in ch: continue
                sys = ch.attrs['system'][0].upper()
                edge = 'e' if ch.attrs['edge'] else ''
                ch_beams = np.array([sys+b+edge for b in ch['beams'].values])

                corr = np.dot(ch['beam_frac'].values.T, [C.get(b,1) for b in ch_beams])
       
                valid = corr > 0
                ch[nimp_name+'_corr'] = ch[nimp_name].copy()#copy including the attributes
                ch[nimp_name+'_corr'].values[valid] /= corr[valid]
                ch[nimp_name+'_err_corr'] = ch[nimp_name+'_err'].copy()
                ch[nimp_name+'_err_corr'].values[valid] /= corr[valid]
            
            
        elif rcalib and len(unique_impurities)>1:
            printe('Calibration is not implemented for two impurities in NIMP data  '+str( unique_impurities))
            rcalib = False
       

        nimp['rel_calib_'+nimp_name]  = nimp.get('rel_calib_'+nimp_name, False) or rcalib

        #return corrected data from the right source in ch['nimp'] variable
        
        if rcalib and len(diag_names) > 1 and 'tangential' in nimp['systems']:
            print('\t done in %.1fs'%(time()-T))


        nimp = return_nimp(nimp)
        #embed()

        return nimp
 

    def load_zeff(self,_tbeg,_tend, systems, options=None):
        #load visible bremsstrahlung data
        TT = time()
   
        tbeg,tend = self.eqm.t_eq[[0,-1]]
        #use cached data
        zeff = self.RAW.setdefault('VB',Tree())
        
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
                if cer_analysis_type is not None and sys in cer_vb_diags:
                    systems[i] += '_'+cer_analysis_type
        
        #when chords affected by CX needs to be removed
        VB_array = 'VB array'
        if 'VB array' in systems and options['VB array']['Corrections']['remove NBI CX'].get():
            VB_array += ' w/o CX'
            systems[systems.index('VB array')] = VB_array

 
        self.RAW['VB']['systems'] = systems
        zeff['systems'] = systems
        systems = list(set(systems)-set(zeff.keys()))  #get only systems which are not loaded yet
        #update mapping of the catched data
        zeff = self.eq_mapping(zeff)            
        zeff.setdefault('diag_names',Tree())
 
        #update equilibrium for already loaded systems
        zeff = self.eq_mapping(zeff)

  
        lambda0 = 5230.0
        #NOTE slow, VB signals are stored in a large time resolution

        ######################   VB array data #############################
        
        if VB_array in systems:  
            print_line( '  * Fetching VB (slow) ...' )

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
                    
                 
                    self.MDSconn.closeTree(tree, self.shot)


                    nbi_pow = 0
                    #remove timeslices affected by CX from NBI by some impurities like Ar, Ne, Al,Ca,...
                    if 'VB array w/o CX' == VB_array:
                        self.MDSconn.openTree('NB', self.shot)                      
                        nbi = self.MDSconn.get('\\NB::TOP.NB{0}:PINJ_{0}'.format('30L')).data()
                        nbi = nbi+self.MDSconn.get('\\NB::TOP.NB{0}:PINJ_{0}'.format('30R')).data()
                        t = self.MDSconn.get('\\NB::TOP:TIMEBASE').data()
                        nbi_pow = np.interp(tvec, t/1000,nbi/1e6)
                        #downsample on resolution used for Zeff analysis
                        nbi_pow   = np.mean(nbi_pow[:imax//n*n].reshape( -1, n), 1)


                    PHDMID = np.mean(PHDMID[:imax//n*n].reshape( -1,n,nchans), 1)
                    tvec   = np.mean(tvec[:imax//n*n].reshape( -1, n), 1)
                    
                    #all values after PHDMID == 10 will be invalid , it is not always working!!
                    valid = PHDMID < 9.9
                    valid = np.cumprod(valid, axis=0,dtype='bool')
                    #only for LOSs viewing beams 30L and 30R i.e. 5 to 16
                    if np.any(nbi_pow > 0.1):
                        valid[:,5:] &= nbi_pow[:,None] < 0.1
            
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
                VB_err[VB == 0] = np.infty
                VB_err[~valid[imin:]] *= -1 #posibly invalid, can be enabled in the GUI

                
                zeff[VB_array] = Dataset('ZeffVB.nc',attrs={'system':VB_array,'wavelength': 5230.0})

                zeff[VB_array]['VB'] = xarray.DataArray(VB,dims=['time','channel'], attrs={'units':'W/cm**2/A','label':'VB' })
                zeff[VB_array]['VB_err'] = xarray.DataArray(VB_err,dims=['time','channel'], attrs={'units':'W/cm**2/A'})
                zeff[VB_array]['diags']= xarray.DataArray( np.tile((VB_array,), VB.shape),dims=['time','channel'])
                
                zeff[VB_array]['R_start'] = xarray.DataArray(R_start, dims=['channel'], attrs={'units':'m'})
                zeff[VB_array]['R_end'] = xarray.DataArray(R_end, dims=['channel'], attrs={'units':'m'})

                zeff[VB_array]['z_start'] = xarray.DataArray(z_start,dims=['channel'], attrs={'units':'m'})
                zeff[VB_array]['z_end'] = xarray.DataArray(z_end,dims=['channel'], attrs={'units':'m'})
                
                zeff[VB_array]['phi_start'] = xarray.DataArray(phi_start,dims=['channel'], attrs={'units':'m'})
                zeff[VB_array]['phi_end'] = xarray.DataArray(phi_end,dims=['channel'], attrs={'units':'m'})

                zeff[VB_array]['razor'] = xarray.DataArray(razor,dims=['channel'] )
                
                zeff[VB_array]['channel'] = xarray.DataArray(['VB%.2d'%ich for ich in range(1,nchans+1)])
                zeff[VB_array]['time'] = xarray.DataArray(tvec.astype('single'),dims=['time'], attrs={'units':'s'})

                zeff['diag_names'][VB_array] = [VB_array]
                
      
        ######################   CER VB data #############################3


        if cer_analysis_type is not None and any(np.in1d([c+'_'+cer_analysis_type for c in cer_vb_diags], systems)):
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

                try:
                    lengths = self.MDSconn.get('getnci("'+path+':VB","LENGTH")').data()
                    nodes = self.MDSconn.get('getnci("'+path+'","fullpath")')
                except:
                    nodes = lengths = []
                for lenght,node in zip(lengths, nodes):
                    if lenght  == 0:#no data
                        continue
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

            if len(TDI) > 0:
        
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
                    ds = zeff[cer_subsys[diag]] = Dataset('ZeffCER.nc', attrs={'system':'CER VB','wavelength': lambda0})
                    ds['VB'] = xarray.DataArray(VB[ind]*convert,dims=['time','channel'], attrs={'units':'W/cm**2/A','label':'VB' })
                    ds['VB_err'] = xarray.DataArray(VB_err[ind]*convert,dims=['time','channel']) 
                    ds['R_start'] = xarray.DataArray(R1, dims=['channel'], attrs={'units':'m'})
                    ds['R_end'] = xarray.DataArray(R_end,dims=['channel'], attrs={'units':'m'})
                    ds['z_start'] = xarray.DataArray(Z1 ,dims=['channel'], attrs={'units':'m'})
                    ds['z_end'] = xarray.DataArray(z_end,dims=['channel'], attrs={'units':'m'})
                    ds['phi_start'] = xarray.DataArray(phi1,dims=['channel'], attrs={'units':'m'})
                    ds['phi_end'] = xarray.DataArray(phi_end,dims=['channel'], attrs={'units':'m'})
                    names = ['VB '+diags_[i][0].upper()+'_%d'%Phi1[j] for j,i in enumerate(valid_ind)]
                    ds['diags'] = xarray.DataArray(np.tile(names, (len(tvec[ind]),1)),dims=['time', 'channel'])
                    ds['time'] = xarray.DataArray(tvec[ind].astype('single'),dims=['time'], attrs={'units':'s'})
                    ds['channel'] = xarray.DataArray([channels[i][0]+channels[i][-2:] for i in valid_ind] ,dims=['channel'])

                    zeff['diag_names'][cer_subsys[diag]] = np.unique(names).tolist()
                self.MDSconn.closeTree(tree, self.shot)
            else:
                printe('No CER VB data are in MDS+')
                

        ##########################################  EQ mapping  ###############################################

        zeff['EQM'] = Tree({'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag})

        #calculate LOS coordinates in rho 
        for sys in systems:
            if not sys in zeff or sys in ['vertical','tangential','SPRED']: continue
  
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
        if bool(options['VB array']['Corrections']['rescale by CO2'].get()):
            try:
                TS = self.co2_correction(TS, tbeg, tend)
            except Exception as e:
                printe('CO2 correction failed:'+str(e))
                
    
            
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
        
        # M A Van Zeeland et al 2010 Plasma Phys. Control. Fusion 52 045006
        #use of this formula increases calculated Zeff in dirty discharges and at low Te - increases the observed discrepancies even more!
        #gff = 5.542-(3.108-log(T_e/1000.))*(0.6905-0.1323/Zeff)  #add small Zeff correction in gaunt factor
 
        # hc for the quantity [hc/(Te*lambda)] for T_e in (eV) and lambda in (A)
        hc = 1.24e4

        vb_coeff = 1.89e-28 * ((n_e*1e-6)**2)  * gff / np.sqrt(T_e)
        wl_resp = np.exp(-hc / (T_e  * lambda0))/lambda0**2 #this ter is close to 1/lambda0**2
        vb_coeff *= wl_resp
        
        assert all(np.isfinite(vb_coeff)), 'finite vb_coeff'

        Linterp = LinearNDInterpolator(np.vstack((tvec,rho)).T, vb_coeff,rescale=True, fill_value=0) 

         #calculate weights for each rho/time position
        for sys in zeff['systems']:
            if not sys in zeff or sys in ['vertical','tangential','SPRED']: continue
 
            
            RHO =  zeff[sys]['rho'].values
            T = np.tile(zeff[sys]['time'].values,RHO.shape[:0:-1]+(1,)).T
            vb_coeff_interp = Linterp(np.vstack((T.flatten(), RHO.flatten())).T).reshape(RHO.shape)
            
            dL = np.gradient(zeff[sys]['L'].values)[1]*100 #(m->cm) should be constant for each LOS 
            VB_Zeff1 = np.sum(vb_coeff_interp*dL,2) #VB for Zeff == 1, use for normalization

                  
            #remove radiative mantle from VB array
            VB = np.copy(zeff[sys]['VB'].values)
            VB_err = np.copy(zeff[sys]['VB_err'].values)
            

            if sys == VB_array and options is not None and bool(options['VB array']['Corrections']['radiative mantle'].get()):
                #calculate edge Zeff and  mantle such, that Zeff from two most edge channels will be equal
                try:

                    R_tg = np.amin(zeff[sys]['R'].values,1)[:,None]
                    #position of the separatrix 
                    R_out, R_in = self.eqm.rhoTheta2rz( 1, [0,np.pi])[0][:,:,0].T
                    R_out = np.interp(zeff[VB_array]['time'].values, self.eqm.t_eq, R_out)
                    R_in  = np.interp(zeff[VB_array]['time'].values, self.eqm.t_eq, R_in)

                    r_mantle = 0.05 #[VBm] guess
                    dl_mantle = 2*(np.sqrt(np.maximum((R_out+r_mantle)**2-R_tg**2,0)) - np.sqrt(np.maximum(R_out**2-R_tg**2,0))) #outer mantle
                    dl_mantle += 2*np.sqrt(np.maximum(R_in**2-R_tg**2,0)) #inner mantle
                    dl_mantle = np.maximum(r_mantle, dl_mantle) #avoid zero division

                    #use two outermost channels to estimate mantle radiation and zeff                 
                    A = np.dstack((VB_Zeff1[:,:2], dl_mantle[:2].T))
                    Zeff_edge, mantle = np.linalg.solve(A,  VB[:,:2]).T
              
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

        cer_sys = list(set(['tangential', 'vertical'])&set(zeff['systems']))
 
        if len(cer_sys) > 0:
            #BUG
            #options['CER system']['Correction']['remove first data after blip'] = tk.IntVar(value=1)
            NIMP = self.load_nimp(tbeg,tend, cer_sys,options['CER system'])
            
            for sys in cer_sys:
                zeff['diag_names'][sys] = NIMP['diag_names'][sys]
                zeff[sys] = deepcopy(NIMP[sys])
                for ich,ch in enumerate(NIMP[sys]):
                    valid = np.isfinite(ch['nimp_err'].values)

                    Linterp.values[:] = np.copy(n_e)[:,None]
                    ne = Linterp(np.vstack((ch['time'].values[valid], ch['rho'].values[valid])).T)
                    Linterp.values[:] = np.copy(n_er)[:,None]
                    ne_err = Linterp(np.vstack((ch['time'].values[valid], ch['rho'].values[valid])).T)

                    lineid = ch['diags'].values[0][::-1].rsplit(' ',1)[0][::-1]
                    Zimp = int(lineid[1:]) if lineid[1].isdigit() else int(lineid[2:])
          
                    Zmain = 1 # NOTE suppose the bulk ions are D
                    nz = ch['nimp'].values[valid]
                    nz_err = ch['nimp_err'].values[valid]
                    
                    ne = np.maximum(nz*Zimp, ne)

                    Zeff = np.zeros(len(valid), dtype=np.single)
                    Zeff_err = np.ones(len(valid), dtype=np.single)*-np.inf

                    Zeff[valid]=Zimp*(Zimp - Zmain)*nz/ne + Zmain
                    Zeff_err[valid] = (Zeff[valid]-Zmain)*np.hypot(ne_err/(ne+1),nz_err/(nz+1))*np.sign(nz_err)
         
                    ch = ch.drop(['nimp','nimp_err'])
                    if 'nimp_corr' in ch:
                        ch = ch.drop(['nimp_corr','nimp_corr_err'])
                    zeff[sys][ich] = ch
                    zeff[sys][ich]['Zeff'] = xarray.DataArray(np.single(Zeff),dims=['time'], attrs={'units':'-','label':'Z_\mathrm{eff}'})
           
                    zeff[sys][ich]['Zeff_err'] = xarray.DataArray(np.single(Zeff_err),  dims=['time'])
        
    
        if 'SPRED' in zeff['systems']:
 

            I = lambda x: tk.IntVar(value=x)
            S = lambda x: tk.StringVar(value=x)        
                    
                    
              
                            
            cer_type = options['CER system']['Analysis'][0]
            cer_intens = options['CER system']['Correction']['nz from CER intensity']
            options = {'Analysis': (cer_type, (S('best'),'fit','auto','quick')),'Impurity':'C6',
                       'Correction': {'Relative calibration':I(1),'nz from CER intensity': cer_intens}}
            
            
            zeff['SPRED'] = []
 
            #first load carbon from CER and SPRED and cross-calibrate
            main_imp = self.load_nimp(tbeg,tend, ['SPRED','tangential'],options)
 
            zeff['diag_names']['SPRED'] = []
            if len(main_imp['SPRED_C6']) > 0:
                zeff['diag_names']['SPRED'].append('SPRED')
            else:
                printe('No SPRED data')
                
            SPRED_CX_ions = ['He2','Li3','B5','C6','N7','O8','Ne10']
            #SPRED_CX_ions = ['C6']

            concentrations = {imp:[] for imp in SPRED_CX_ions}
            charge = {'C6':6,'e': -1}
            correction = []
            densities = {'time_cx':[], 'rho_cx': [],'n_e': [], 'n_e_err': []}
            densities.update({'n_'+i+'_cx': [] for i in SPRED_CX_ions})
            densities.update({'n_'+i+'_cx_err': [] for i in SPRED_CX_ions})


            for ib,nC in enumerate(main_imp['SPRED_C6']):
                pass
                    
                channel = nC.attrs['channel']
                #correction estimated from cross-calibration by CER
                corr = 1
                
                if cer_intens.get():
                    if 'nimp_int_corr' in nC:
                        corr = nC['nimp_int_corr'].values/(nC['nimp_int'].values+1e-5)
                else:
                    if 'nimp_impcon_corr' in nC:
                        corr = nC['nimp_impcon_corr'].values/(nC['nimp_impcon'].values+1e-5)
                        
                
                correction.append(corr)
                
                R = nC['R'].values
                Z = nC['Z'].values
                rho = nC['rho'].values
                tvec = nC['time'].values
                t_ind = (tvec >= _tbeg)&(tvec <= _tend)
                
                #zipfit = self.load_zipfit()
               
                #Linterp = RectBivariateSpline(zipfit['ne']['time'], zipfit['ne']['rho'], zipfit['ne']['ne_err'],kx=1,ky=1)


                Linterp.values[:] = np.copy(n_er)[:,None]
                ne_err = Linterp(np.vstack((tvec, rho)).T)/1e19
                #ne_err = Linterp.ev(tvec, rho)/1e19
                #Linterp = RectBivariateSpline(zipfit['ne']['time'], zipfit['ne']['rho'], zipfit['ne']['ne'],kx=1,ky=1)

                ##ne  = Linterp.ev(tvec, rho)/1e19

                Linterp.values[:] = np.copy(n_e)[:,None]
                ne = Linterp(np.vstack((tvec, rho)).T)/1e19
                ne = np.maximum(ne, 1e-3) #prevents zero division
 
                #main contribution from carbon
                imp_conc = nC['nimp'].values/1e19/ne
               
                Zeff = 1+(nC.attrs['Z']-1)*nC.attrs['Z']*imp_conc
                Zeff2_err = ((nC['nimp_err'].values/1e19*(nC.attrs['Z']-1)*nC.attrs['Z'])/ne)**2
                
                concentrations['C6'].append(imp_conc[t_ind])
                options = {'Analysis': (S('SPRED'), (cer_type,'fit','auto','quick')),
                        'Correction': {'Relative calibration':I(0),'nz from CER intensity': I(1)}}
                
                densities['time_cx'].append(tvec[t_ind])
                densities['rho_cx'].append(rho[t_ind])
                densities['n_C6_cx'].append(nC['nimp'].values[t_ind])
                densities['n_C6_cx_err'].append(nC['nimp_err'].values[t_ind])
                densities['n_e'].append(1e19*ne[t_ind])
                densities['n_e_err'].append(1e19*ne_err[t_ind])

                                   
                #load other light impurities from SPRED without cross-calibration
                for imp in SPRED_CX_ions:
                    
                    if imp == 'C6': continue
     

                    options['Impurity'] = imp
                    other_imp = self.load_nimp(tbeg,tend, ['SPRED'],options)['SPRED_'+imp][ib]
                    #rarely the number of timespices from other elemnt can be different than for carbon
                    nimp     = np.interp(tvec, other_imp['time'].values, other_imp['nimp'].values)
                    nimp_err = np.interp(tvec, other_imp['time'].values, other_imp['nimp_err'].values)

                    imp_conc = nimp/1e19*corr/ne
                    imp_conc_err = nimp_err/1e19*corr/ne
                    
                    #just guess, this calibration might be valid from waterleak till ~186000
                    if 186000 > self.shot > 182000:
                        if imp == 'B5': 
                            #calibrate from discharge 184840
                            imp_conc *= 1.294
                        if imp == 'Ne10': 
                            #calibrate from discharge 184846
                            imp_conc *= 2.12/2.85
                        if imp == 'N7': 
                            #calibrate from discharge 184522
                            imp_conc *= 7
                            imp_conc_err*= 7
        
                            #BUG  N7 adds a lots of noise in the, it needs to be repaired
                            if self.shot not in [184847]: continue
      
                    ZZm1 = (other_imp.attrs['Z']-1)*other_imp.attrs['Z']
                    Zeff += ZZm1*imp_conc
                    Zeff2_err += (ZZm1*np.minimum(imp_conc_err, imp_conc))**2

                    
                    concentrations[imp].append(imp_conc[t_ind])
                    charge[imp] = other_imp.attrs['Z']
                    
                    densities['n_'+imp+'_cx'].append(imp_conc[t_ind]*1e19*ne[t_ind])
                    densities['n_'+imp+'_cx_err'].append(imp_conc_err[t_ind]*1e19*ne[t_ind])
                

                
                
                #add uncertainty in ne
                Zeff_err = Zeff*np.sqrt(Zeff2_err/Zeff**2+(ne_err/ne)**2)

                #create dataset
                ds = Dataset('Zeff_SPRED.nc', attrs={'system':'SPRED', 'channel':channel, 'scaled to match C6 CER data by':corr })

                ds['R'] = xarray.DataArray(R, dims=['time'], attrs={'units':'m'})
                ds['Z'] = xarray.DataArray(Z ,dims=['time'], attrs={'units':'m'})
                ds['rho'] = xarray.DataArray(rho,dims=['time'], attrs={'units':'m'})
                ds['diags'] = xarray.DataArray(['SPRED']*len(tvec),dims=['time'])
                ds['Zeff'] = xarray.DataArray( np.single(Zeff) ,dims=['time'],
                                            attrs={'units':'-','label':'Z_\mathrm{eff}'})

                ds['Zeff_err'] = xarray.DataArray( np.single(Zeff_err), dims=['time']) 
                ds['time'] = xarray.DataArray(tvec ,dims=['time'], attrs={'units':'s'})
                zeff['SPRED'].append(ds)
            
            #embed()
            sind = np.argsort(np.hstack(densities['time_cx']))
            time_cx = np.hstack(densities['time_cx'])[sind] 
            
            # join the SPRED measurement from the same time but different background substraction side
            lind, rind = [], []
            i = 0
            while i < len(time_cx):
                lind.append(i)
                #if they are at nearky the same time (small random jitter was introduced in the timebase)
                if i +1 < len(time_cx) and  time_cx[i] > time_cx[i+1] - 1e-4:
                    i+= 1
                i+= 1
                rind.append(i-1)
       
            for k in list(densities.keys()):
                if len(densities[k]):
                    densities[k] = np.hstack(densities[k])[sind].astype('single')
                    densities[k] = (densities[k][lind]+densities[k][rind])/2
                else:
                    densities.pop(k)
     
                
            for highz_imp in ['Cu', 'Mo', 'Fe', 'Ni']:    
                t,n = self.load_nimp_spred(highz_imp, cx_line=False)
                n-=n[t<.5].mean()
                ind = (t > 1.5)&(t < 5)
                t, n = t[ind], n[ind]
                ne =  Linterp(np.vstack((t, t*0)).T)
                #ne  = Linterp.ev(t, t*0) 

                densities[highz_imp] =  np.single(n/ne)
                densities['time'] = t

            
            densities['charge'] = charge
            np.savez_compressed('SPRED_data_%d'%self.shot, **densities)


            if len(zeff['SPRED']) == 0:
                return zeff

            print('\t'+'-'*40)
            
            corr = np.hstack(correction).mean()
            print('Impurity concentrations from SPRED are rescaled by %.3f to match CER carbon density'%corr.mean())
            print('\t Mean concentration on axis between %.3f-%.3fs'%(_tbeg,_tend))
            imp_names = [k for k, c in concentrations.items() if len(c)]
       
            imp_conce = [max(0,np.hstack(concentrations[k]).mean()) for k in imp_names]
            
            #print deuterium concentration
            charge = np.array([charge[k] for k in imp_names])
            cD = 1-sum(imp_conce*charge)
            sind = np.argsort(imp_conce)[::-1]
            print('\t\tion\t conc.\t dZeff' )

            print('\t\tD1\t%.2f%%\t %.2f'%(cD*100, cD))
            for i in sind:
                print('\t\t'+imp_names[i],'\t%.2f%%\t %.2f'%(imp_conce[i]*100, imp_conce[i]*charge[i]*(charge[i]-1)))
            print('\t'+'-'*40)
        
        return zeff
    
    
    def use_zeeman_NN(self, Ti_obs_vals,modB_vals, theta_vals,loc='/fusion/projects/results/cer/haskeysr/Zeeman_NN/',):
        """
        NN, X_scaler, Y_scaler: neural network, X_scaler and Y_scaler objects from the neural network fitting
        Ti_obs_vals: Observed Ti values [eV]
        modB_vals: Magnetic field strength [T], same length and Ti_obs_vals
        theta_vals: Viewing angle [rad], same length and Ti_obs_vals
        written by Shaun Haskey
        
        """
        
        nn_path = loc+'/Zeeman_Corr_NN_12C6.pickle'
        if not os.path.isfile(loc+'/Zeeman_Corr_NN_12C6.pickle'):
            nn_path = os.path.dirname(os.path.realpath(__file__))+'/Zeeman_Corr_NN_12C6.pickle'
            

        with open(nn_path , 'rb') as filehandle:
            import pickle
            NN_dat = pickle._Unpickler(filehandle)
            NN_dat.encoding = 'latin1'
            try:
                NN_dat = NN_dat.load()
            except:
                #printe('Zeeman correction was unsucessful!! Error in loading of NN pickle')
                raise
            
        NN = NN_dat['NN']
        X_scaler = NN_dat['X_scaler']
        Y_scaler = NN_dat['Y_scaler']
        
        X = np.zeros((len(Ti_obs_vals), 4), dtype=float)
        X_scaler.n_features_in_ = 4
        X[:, 0] = +modB_vals  
        X[:, 1] = np.log10(Ti_obs_vals) 
        X[:, 2] = np.sin(theta_vals) 
        X[:, 3] = np.cos(theta_vals) 

        X_norm = X_scaler.transform(X, copy=True)
        Y_NN_norm = NN.predict(X_norm)
        dT = Y_scaler.inverse_transform(Y_NN_norm, copy=True)
        Ti_real = dT + Ti_obs_vals
        
        return Ti_real

  


    def load_cer(self,tbeg,tend, systems, options=None):
        #load Ti and omega at once
        TT = time()

        tree = 'IONS'
        if options is None:
            analysis_type = 'best'
            analysis_types= 'fit','auto','quick'
        else:
            selected,analysis_types = options['Analysis']
            analysis_type = selected.get()
            
        analysis_type = self.get_cer_types(analysis_type)
        if analysis_type is None:
            raise Exception('No CER analysis data')
        
        #switch on/off correction in already loaded systems
        try:
            zeem_split = options['Corrections']['Zeeman Splitting'].get()
        except:
            zeem_split = getattr(self,'zeem_split', False)
        
        try:
            sol_corr = options['Corrections']['Wall reflections'].get()
        except:
            sol_corr = False
            

            
        self.RAW.setdefault('CER',Tree())
        cer = self.RAW['CER'].setdefault(analysis_type,Tree())

        #load from catch if possible
        cer.setdefault('diag_names',Tree())
        cer['systems'] = systems
        
        #if sol_corr or zeem_split is different from previously loaded, reload data
        if getattr(self,'cer_sol_corr', -1) == sol_corr and getattr(self,'zeem_split', -1) == zeem_split:
            load_systems = list(set(systems)-set(cer.keys()))
        else:
            load_systems = systems
            
            
        #update equilibrium for already loaded systems
        
        
        #embed()
        cer = self.eq_mapping(cer)
        if len(load_systems) == 0:
            return cer
   
        print_line( '  * Fetching '+analysis_type.upper()+' data ...' )
        
        cer_data = {
            'Ti': {'label':'Ti','unit':'eV','sig':['TEMP','TEMP_ERR']},
            'omega': {'label':r'\omega_\varphi','unit':'rad/s','sig':['ROT','ROT_ERR']},
            'int': {'label':r'\omega_\varphi','unit':'rad/s','sig':['INTENSITY','INTENSITYERR']}
            }
        #NOTE visible bramstrahlung is not used
       
        #list of MDS+ signals for each channel
        all_nodes = []        
        TDI = []
        TDI_lens_geom = []
        TDI_lineid = []
        diags_ = []
        data_nbit = []
        missing_rot = []
    
        if analysis_type == 'cerreal':
            signals = 'time','amp', 'ti','rot'
            system = 'tangential'
            zeem_split=False
            if 'Corrections' in options and 'Zeeman Splitting' in options['Corrections']:
                options['Corrections']['Zeeman Splitting'].set(False)
            #geom_names = ['LENS_R','LENS_Z','LENS_PHI']
            #availible only for channels 5 - 24 
            cer[system] = []
            cer['diag_names'][system] = []
            for ch in range(5,25):
                #node_calib = "\\IONS::TOP.CER.CALIBRATION."+system+".CHANNEL%.2d"%ch
                #for par in geom_names:
                    #TDI_lens_geom.append(node_calib+':'+par)
 
                #TDI_lineid.append(node_calib+':'+'LINEID')
                #if ch in np.r_[1:8, 17:23]: #30 beam
                    #beams = '0,1'
                #elif ch in np.r_[8:17, 23:25 ]: #330 beam
                    #beams = '2,3'
                #else:
                    #raise Exception('Not supported')
                
                #take just an average position, ignoring which beams was on
                #for par in ['PLASMA_R', 'PLASMA_Z']:
                #TDI_lens_geom.append(f'MEAN(({node_calib}:PLASMA_R)[[{beams}]])')
                #TDI_lens_geom.append(f'MEAN(({node_calib}:PLASMA_Z)[[{beams}]])')

                TDI.append('dim_of(PTDATA("crs%st%d",%d))'%(signals[1],ch,self.shot))
                TDI.extend(['PTDATA("crs%st%d%s",%d)'%(q,ch,'a' if q=='rot' else '', self.shot) for q in signals[1:]])
                all_nodes.append(system+'.T%.2d'%ch)
                diags_.append(system)

                 
            #geom_names += ['PLASMA_R', 'PLASMA_Z']

        else:
            signals = cer_data['Ti']['sig']+cer_data['omega']['sig']+\
                    cer_data['int']['sig'] + ['R','Z','VIEW_PHI','STIME','TIME']
                
            #get list of all avalible CER signals
            try:
                assert analysis_type != 'cerreal'
                
                
                
                self.MDSconn.openTree(tree, self.shot)
                #check if corrected rotation data are availible
                path = 'CER.%s.%s.CHANNEL*'%(analysis_type,'tangential')
                try:
                    lenghts = self.MDSconn.get('getnci("'+path+':ROTC","LENGTH")').data()
                    if any(lenghts > 0):
                        signals[signals.index('ROT')] += 'C'
                except MDSplus.MdsException:
                    pass
                #check if corrected temperature data are availible
                try:
                    lenghts = self.MDSconn.get('getnci("'+path+':TEMPC","LENGTH")').data()
                    if zeem_split and sum(lenghts) > 0:
                        signals[signals.index('TEMP')] += 'C'
                except MDSplus.MdsException:
                    pass
            
                #prepare list of loaded signals
                for system in load_systems:
                    signals_ = deepcopy(signals)
                    cer[system] = []
                    cer['diag_names'][system] = []
                    path = 'CER.%s.%s.CHANNEL*'%(analysis_type,system)
                    
                
                    nodes = self.MDSconn.get('getnci("'+path+'","fullpath")')

                    #lengths_int = self.MDSconn.get('getnci("'+path+':'+signals_[4]+'","LENGTH")').data()                
                    lengths_Rot = self.MDSconn.get('getnci("'+path+':'+signals_[2]+'","LENGTH")').data()
                    #lengths_Ti  = self.MDSconn.get('getnci("'+path+':'+signals_[0]+'","LENGTH")').data()
                    lengths = self.MDSconn.get('getnci("'+path+':STIME","LENGTH")').data()
        

                    for node,length,length_r in zip(nodes,lengths,lengths_Rot):
    
                        if length == 0: continue
                        try:
                            node = node.decode()
                        except:
                            pass
                    
                        diags_.append(system)
    
                        node = node.strip()
                        all_nodes.append(node)
                        data_nbit.extend([length]*len(signals_))
                        
                        node_calib = node.replace(analysis_type.upper(),'CALIBRATION')
                        #geom_names = 
                        for par in ['LENS_R','LENS_Z','LENS_PHI']:
                            TDI_lens_geom.append(node_calib.strip()+':'+par)
                        TDI_lineid.append(node_calib.strip()+':'+'LINEID')

                        for sig in signals_:
                            #sometimes is rotation not availible even if Ti is
                            if length_r == 0 and sig in ['ROT','ROTC','ROT_ERR']:
                                missing_rot.append(node)
                                sig = 'STIME' #replace be something else
    
                            TDI.append(node+':'+sig)

            except Exception as e:
                printe( 'MDS error: '+ str(e))
            finally:
                self.MDSconn.closeTree(tree, self.shot)
        
        ##No data in MDS+
        if len(all_nodes) == 0:
            if any([s in cer for s in cer['systems']]):
                #something was at least in the catch
                return cer
            
            tkinter.messagebox.showerror('No CER data to load',
                'Check if the CER data exists or change analysis method')
            return None
        
        #get a list of joined strings with TDI for fast fetching 
        TDI = np.reshape(TDI,(-1,len(signals)))
        TDI_list = []
         
        if analysis_type != 'cerreal':
            TDI_list+= ['['+','.join(TDI_lineid)+']']
            TDI_list+= ['['+','.join(TDI_lens_geom)+']']
            TDI = TDI.T
              
        TDI_list += ['['+','.join(tdi)+']' for tdi in TDI]
  
        #fetch data
        output = mds_load(self.MDSconn, TDI_list , tree, self.shot)
  

        if len(output) == 0:
            print('CER data fetching has failed!!')
            raise Exception('CER data fetching has failed!!')
        
        if analysis_type != 'cerreal':
            lineid = output.pop(0)
            #lineid = output.pop(0)
            geom_data = output.pop(0)
            LENS_R,LENS_Z,LENS_PHI = geom_data.reshape(-1,3).T
            
            LENS_PHI = np.int_(LENS_PHI)
            #embed()

            if len(set([d.size for d in output])) > 1 :
                raise Exception('CER data for mismatch length of timebases, try to turn off Zeeman correction')
            #get geometry info first
            
            #geom_data


            #geom_data = geom_data.reshape(-1,len(geom_names)).T 
            #geom_data = {name:val for name,val in zip(geom_names, geom_data)}
            #lineid = output.pop()
            
            #split data in list if profiles of list of signals
            mds_data = np.single(output).flatten()
            data_nbit = np.reshape(data_nbit,(-1,len(signals))).T.flatten()
            Ti,Ti_err,rot,rot_err,int_,int_err, R,Z,PHI,stime,tvec = mds_data.reshape(len(signals), -1)
            
            #index for signal splitting
            split_ind = split_mds_data(np.arange(len(tvec)), data_nbit.reshape(len(signals), -1)[0], 4)
            split_ind = [slice(s[0], s[-1]+1) for s in split_ind]
            
        else: #CERREAL
            #R,Z for channels 5-24, averaged over beam L and R
            R_plasma = np.array([2043,2127,2194,2186,2194,2217,2232,2247,2262,\
                    2277,2292,2306,1701,1838,1904,1974,2088,2161,2178,2209])     
            Z_plasma = np.array([-2,5,10,0,0,0,0,0,0,1,1,1,0,-5,-9,-12,1,8,0,0])
                
            data, R, Z, split_ind, valid_ch  = [],[],[],[],[] 
            n = 0
            
            for ich, ch_data in enumerate(output):
                #embed()
                if len(ch_data) == 0 or all(ch_data[1] == 0):
                    continue
        
                #where the data becommed availible in realtime and they are nonzero
                ind = (np.ediff1d(ch_data[1],to_begin=0)!=0)&(ch_data[1] > 0)
                
                data.append(ch_data[:,ind])
                nt = sum(ind)
                 
                R.append(np.zeros(nt)+R_plasma[ich]/1e3)
                Z.append(np.zeros(nt)+Z_plasma[ich]/1e3)
                
                split_ind.append(slice(n, n+nt))
                n+= nt
                valid_ch.append(ich)
                
                
            
            if n == 0:
                #tkinter.messagebox.showerror('No CER REAL data', 'Try different edition')
                raise Exception('No CERREAL data!!')
            
            tvec, int_, Ti, rot = np.hstack(data)
            R = np.hstack(R)
            Z = np.hstack(Z)
            
            #guess the missing quantities
            stime = 5*np.ones_like(tvec) #guess 5ms integration time
            PHI = np.zeros_like(tvec)
            
            #guess 5% error
            int_err = 0.05*int_
            Ti_err = np.hypot(0.05*Ti, 100)
            rot_err = np.hypot(0.05*rot, 0.01*rot.mean())
            #lineid = ['CI']
            lineid = ['C VI 8-7']*len(valid_ch)

            #geom_names = ['LENS_R','LENS_Z','LENS_PHI']
            #geom_data = {}
            LENS_R = [np.nan]*len(valid_ch)
            LENS_Z = [np.nan]*len(valid_ch)
            
                    #if ch in np.r_[1:8, 17:23]: #30 beam
                        #beams = '0,1'
                    #elif ch in np.r_[8:17, 23:25 ]: #330 beam
                        #beams = '2,3'
                    #else:
                    
            loc = np.array(['core']*3+['edge']*9+['core']*6+['edge']*2)
            #print(len(loc), len(np.arange(5,25)))
            LENS_PHI = loc[valid_ch]

            #= {k:v[valid_ch] for k,v in geom_data.items()}
                        
            all_nodes = np.array(all_nodes)[valid_ch]
            
            #embed()


        
        #get a time in the center of the signal integration 
        tvec = (tvec+stime/2)/1e3
        
        #map to radial coordinate 
        rho = self.eqm.rz2rho(R[:,None],Z[:,None],tvec,self.rho_coord)[:,0]
        
        
        #############  SOL reflection corrections #####################
        if sol_corr:
            #estimate of reflected light properties
            T_avg = np.unique(tvec)
            int_old, rot_old, Ti_old = int_, rot, Ti
            rot_err[~np.isfinite(rot)] = np.inf
            rot[~np.isfinite(rot)] = 0

            int_err_old,rot_err_old, Ti_err_old = int_err,rot_err, Ti_err
            int_avg = np.vstack([np.interp(T_avg, tvec[ind],int_[ind]) for ind in split_ind])
            rot_avg = np.vstack([np.interp(T_avg, tvec[ind],rot[ind] ) for ind in split_ind])
            Ti_avg  = np.vstack([np.interp(T_avg, tvec[ind],Ti[ind]  ) for ind in split_ind])
            weight = int_avg
            int_avg = np.average(int_avg,0,weight)
            rot_avg = np.average(rot_avg,0,weight)
            Ti_avg  = np.average( Ti_avg,0,weight) #TODO add somehow also rotation broadening?
    
            #correction for reflected light in intesity and temperature
            i = (rho  > 0.9)&(int_ > 0)&(Ti > 0) #just pedestal

            c = np.linspace(.02,0.05,200)[:,None]
            from scipy.constants import m_p, e,k

            A = int_ - np.interp(tvec, T_avg,int_avg )*c
            W = A/np.clip(int_, A ,100*A) #avoid zero divison if int_==0 and clip W in range 0.01 to 1
            try:
                T = (Ti-np.interp(tvec, T_avg,Ti_avg)*(1-W)-W*(1-W)*(rot-np.interp(tvec, T_avg,rot_avg))**2*(m_p*12)/(2*e))/W  #BUG should I use the corrected rotation??
            except:
                print(' = (Ti-np.interp(tvec, T_avg,Ti_avg)*(1-W)-W*(1-W)*(rot-np.interp(tvec, T_avg,rot_avg))**2*(m_p*12)/(2*e))/W  #BUG should I use the corrected rotation??')
                embed()
            c_cost = np.linalg.norm(T[:,i]*W[:,i]/Ti[i],axis=1)**2+np.linalg.norm(A[:,i]/int_[i],axis=1)**2      #bias towards zero
            #panise more cases where is negaive T or intensity
            neg_T = T[:,i] < 0
            c_cost += 1*np.linalg.norm(T[:,i]*W[:,i]/Ti[i]*neg_T,axis=1)**2
            neg_A = A[:,i] < 0
            c_cost += 1*np.linalg.norm(A[:,i]/int_[i]*neg_A,axis=1)**2      #bias towards zero

            ic = np.argmin(c_cost)
            #ic = np.argmin(np.abs(c-.025))
            W = W[ic]
            int_err,rot_err, Ti_err = int_err/W,rot_err/W,np.clip(Ti_err/W, Ti_err, Ti)
    
            #correction for rotation - how large fraction is blue and red shifted
            r = np.linspace(.2,.5,100)[:,None]
            Rc = (rot-np.interp(tvec, T_avg,rot_avg)*(1-W)*r)/W
            r_cost = np.linalg.norm(Rc[:,i]*W[i],axis=1)**2       #bias towards zero
            ir = np.argmin(r_cost)
            int_,rot,Ti = np.clip(A[ic], int_*0.01, A[ic]*1.5), np.copy(Rc[ir]), np.clip(T[ic],1,Ti*1.5) 
            
            print('\nWall reflection coeff: %.3f  rotation coeff: %.2f'%(c[ic], r[ir]))
 
        ###############  Zeman correction #######################
        if signals[0] != 'TEMPC' and zeem_split:
            #Zeeman splitting correction
                
            #get magnetic field at measurement locations
            Br,Bz,Bt = self.eqm.rz2brzt(R[:,None],Z[:,None],tvec)
            Bvec = np.squeeze((Br,Bz,Bt))
            
            # Now we have the magnetic field strength
            modB = np.linalg.norm(Bvec,axis=0)
            B_hat = -Bvec / modB #make it consistent with OMFITprofiles
            
            #embed()
            #calculate unit vector in LOS direction
            dPhi = np.deg2rad(np.hstack([PHI[ind]-p for p,ind in zip(LENS_PHI,split_ind)]))
            Rlen = np.hstack([R[ind]*0+r for r,ind in zip(LENS_R,split_ind)])
            
            #(X_spot-X_lens, Yspot-Ylens, Z_spot-Zlens)
            dR = R-np.cos(dPhi)*Rlen
            dT = 0-np.sin(dPhi)*Rlen
            dZ = np.hstack([Z[ind]-z for z,ind in zip(LENS_Z,split_ind)])
            
            los_vec = np.array((dR, dZ, dT))
            los_vec /= np.linalg.norm(los_vec,axis=0)
            
            #get angle between B and LOS            
            angle_B_chord = np.arccos(np.sum(los_vec * B_hat, 0))
        
            #evaluate correction neural network
            Ti_corr = np.copy(Ti) #temperature to be corrected
            valid = np.isfinite(Ti_corr) & (Ti_corr > 0)
            try:
                Ti_corr[valid] = self.use_zeeman_NN(Ti_corr[valid], modB[valid], angle_B_chord[valid])
            except:
                try:
                    options['Corrections']['Zeeman Splitting'].set(0)
                except:
                    pass
            Ti_corr = Ti
        #else:
            #minor bug, it will not be possible to reload uncorrected data
            #Ti_corr = np.copy(Ti) 
            #if options is not None and 'Corrections' in options and 'Zeeman Splitting' in options['Corrections']:
                #options['Corrections']['Zeeman Splitting'].set(False)
        
        
        
        if not isinstance( lineid[0], str):
            lineid = [l.split(b'\x00')[0].decode('utf-8') for l in lineid]
 
        lineid = [l.strip() for l in lineid]
            
        ulineid = np.unique(lineid)
        multiple_imps = len(ulineid) > 0
 
        
        for ich,(ch,tind) in enumerate(zip(all_nodes,split_ind)):
            diag = ch.split('.')[-2].lower()
            
            name = diags_[ich][0].upper()+'_'+str(LENS_PHI[ich])
            
            #add impurity name, if there are more impurities in the profile
            unreliable = np.ones(tind.stop-tind.start)
            if len(lineid) > 0:
                line = lineid[ich]
            else: #old discharge swere missing this information
                line = 'C VI 8-7'

            
            tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)',line)
            element, charge = tmp.group(1), roman2int(tmp.group(2))
            
            
            if multiple_imps:
                name += ' '+ element+str(charge) 
                if 'C VI 8-7' in ulineid and line != 'C VI 8-7':
                    unreliable[:] = -1
      
            if not name in cer['diag_names'][diag]:
                cer['diag_names'][diag].append(name)
                
            ds = Dataset(name[0]+ch[-2:]+'.nc', attrs={'channel':name[0]+ch[-2:],'imp':element+str(charge)})
            ds['R'] = xarray.DataArray(R[tind], dims=['time'], attrs={'units':'m'})
            ds['Z'] = xarray.DataArray(Z[tind], dims=['time'], attrs={'units':'m'})
            ds['rho'] = xarray.DataArray(rho[tind], dims=['time'], attrs={'units':'-'})
            ds['diags']= xarray.DataArray(np.array((name,)*(tind.stop-tind.start)),dims=['time'])

   
            if len(rot[tind]) > 0 and ch not in missing_rot:
                corrupted = ~np.isfinite(rot[tind]) | (R[tind]  == 0)
                rot[tind][corrupted] = 0
                corrupted |= (rot[tind]<-1e10)|(rot_err[tind]<=0)
                rot_err[tind][corrupted] = np.infty

                rot[tind][~corrupted]    *= 1e3/R[tind][~corrupted]    
                rot_err[tind][~corrupted] *= 1e3/R[tind][~corrupted] 
                
                if not all(corrupted):
                    ds['omega'] = xarray.DataArray(rot[tind],dims=['time'], attrs={'units':'rad/s','label':r'\omega_\varphi'})
                    ds['omega_err'] = xarray.DataArray(rot_err[tind]*unreliable,dims=['time'], attrs={'units':'rad/s'})
                    
                    
                    
                
                
            if len(Ti[tind]) > 0:
                Ti_err[tind][np.isnan(Ti_err[tind])] = np.infty
                #900eV is probably initial guess, sometimes it does not move from this value
                corrupted = (Ti[tind]==900)|(Ti[tind]>=15e3)|(Ti[tind] <= 0)|(R[tind]  == 0)|(Ti_err[tind]<=0)|~np.isfinite(Ti[tind])
                Ti_err[tind][corrupted] = np.infty
                Ti[tind][~np.isfinite(Ti[tind])] = 0
                
                if not all(corrupted):
                    ds['Ti'] = xarray.DataArray(Ti[tind],dims=['time'], attrs={'units':'eV','label':'T_i', 'zeeman_split':zeem_split})
                    ds['Ti_err'] = xarray.DataArray(Ti_err[tind]*unreliable,dims=['time'], attrs={'units':'eV'})
            

            if len(int_[tind]) > 0:
                int_err[tind][np.isnan(int_err[tind])] = np.infty

                corrupted =  (int_[tind] <= 0)|(R[tind]  == 0)|(int_err[tind]<=0)|~np.isfinite(int_[tind])
             
                int_err[tind][corrupted] = np.infty
                int_[tind][~np.isfinite(int_[tind])] = 0
                
                if not all(corrupted):
                    ds['int_err'] = xarray.DataArray(int_err[tind],dims=['time'], attrs={'units':'ph/s*sr'})
                    ds['int'] =  xarray.DataArray(int_[tind],dims=['time'], attrs={'units':'ph/s*sr'})
            
  

            if 'Ti' in ds or 'omega' in ds:
                #BUG time can be sometimes not unique 183504 CERFIT
                ds['time'] = xarray.DataArray(tvec[tind], dims=['time'], attrs={'units':'s'})
                cer[diag].append(ds)
                #embed()
        
        
        self.cer_sol_corr = sol_corr
        self.zeem_split = zeem_split





        #plt.plot(c, c_cost1)
        #plt.plot(c, c_cost2)
        #plt.plot(c, c_cost2+c_cost1,'--')

        #plt.show()
        
        #plt.plot(T_-T[ic]*W[ic])
        #plt.plot(T_)
        #plt.show()
        
        
        #ic = np.argmin(c_cost2)
        #plt.errorbar(range( len(T_all)),np.clip(T[ic],1,T_*1.5), Te_/W[ic])
        #plt.errorbar(range( len(T_all)),T_,Te_)
        #plt.show()
        
        
        #plt.errorbar(range( len(T_all)),np.clip(T,1,T_*1.5), Te_/W)
        #plt.errorbar(range( len(T_all)),T_,Te_)
        #plt.show()
        
        


        
        #embed(); exit()
        

 
        
        #c = 0.035
        #r = 0.3  #correction for rotation - how large fraction is blue and red shifted
        #N = np.hstack(int_ ) - np.interp(T_all, T_avg,int_avg )*c
        #W = np.clip(N/np.hstack(int_ ),0.001,1)
        #R = (np.hstack(rot)-np.interp(T_all, T_avg,rot_avg )*(1-W)*r)/W
        #T = (np.hstack(Ti)-np.interp(T_all, T_avg,Ti_avg )*(1-W)-W*(1-W)*(np.hstack(rot)-np.interp(T_all, T_avg,rot_avg))**2*(m_p*12)/(2*e) )/W 
        
        
        
         #plt.plot((rot_old-Rc[ir]*W)/1e3 )
        #plt.plot(rot_old/1e3) 
        
        #plt.plot(  (rot_old-np.interp(tvec, T_avg,rot_avg)*(1-W)*r[ir]) )
        #plt.plot( np.interp(tvec, T_avg,rot_avg)*(1-W)*.3)
        #plt.plot(rot_old) 
 
        #plt.show()
        
        
        
        

        #f,ax=plt.subplots(3,1,sharex=True)
        
        
        #ii = np.argsort([R[ind].mean() for ind in split_ind])
        #ind = np.arange(len(R))
        #ind = np.hstack([ind[split_ind[i]] for i in ii])
        #ax[2].plot((Ti_old-T[ic]*W)[ind]/1e3)
        #ax[2].plot(Ti_old[ind]/1e3) 
        ##ax[2].plot(T[ic][ind]/1e3,':',lw=.5)

        
        
        #ax[1].plot((rot_old-Rc[ir]*W)[ind]/1e3 )
        #ax[1].plot(rot_old[ind]/1e3)        
        ##ax[1].plot(Rc[ir][ind]/1e3,':',lw=.5)

        #ax[0].plot(((int_old-int_))[ind]/1e16  )
        #ax[0].semilogy(int_old[ind]/1e16)
        ##ax[0].plot(int_[ind]/1e16,':',lw=.5)

        ##embed()
        #ind = np.arange(len(R))
        #ind = [ind[split_ind[i]] for i in ii]
        #for a in ax:
            #for j,i in zip(ii,ind):
                ##embed();exit()
                #a.axvline(i[0], c='k',ls='--')
                #a.text(i.mean() ,1 ,'T'+all_nodes[j][-2:])
                
                
        #ax[0].set_ylabel('Ampl. [ph/(s*sr)]')
        #ax[1].set_ylabel('Rot. [km/s]')
        #ax[2].set_ylabel('T$_i$ [keV]')
        #ax[2].set_xlabel('index')
 

        #plt.show()
        
        
        #embed()
        #ind = (T_all > 2)&(T_all < 3)
        #f,ax=plt.subplots(3,1,sharex=True)
        #ax[2].plot(rho[ind],(np.hstack(Ti)-T*W)[ind]/1e3,'.')
        #ax[2].plot(rho[ind],np.hstack(Ti)[ind]/1e3,'.')        
        
        #ax[1].plot(rho[ind],(np.hstack(rot)-R*W)[ind]/1e3,'.' )
        #ax[1].plot(rho[ind],np.hstack(rot)[ind]/1e3,'.')        
            
        #ax[0].plot(rho[ind],((np.hstack(int_)-N))[ind] ,'.' )
        #ax[0].semilogy(rho[ind],np.hstack(int_)[ind],'.')
        
        #ax[0].set_ylabel('Ampl. [ph/(s*sr)]')
        #ax[1].set_ylabel('Rot. [km/s]')
        #ax[2].set_ylabel('T$_i$ [keV]')
        #ax[2].set_xlabel('rho')
        #ax[2].set_xlim(.9,1.1)
        ##ax[0].set_ylim(0,None)
        #ax[2].set_ylim(0,None)

        #plt.show()
        #RHO = [rho[ind].mean() for ind in split_ind]

        #plt.figure()
        #C = np.zeros((1, len(split_ind)))
        #for it, ind in enumerate(split_ind):
            ##for i in range(3):
            #i = 0
            #C[i, it] = np.corrcoef(Ti_old[split_ind[-1-i]], np.interp(tvec[split_ind[-1-i]], tvec[ind], Ti_old[ind]))[0,1]**2
                
        #plt.plot(RHO, C.T,'o')      
        ##plt.xlim(1.7,2.3)
        #plt.ylim(0,1)
        #plt.show()
        
        
        #embed()

        
        #corrcoef
        
        

        #f,ax=plt.subplots(1,3,sharex=True)
        
        #ax[2].plot(rho, np.hstack(Ti ),'.')
        #ax[2].plot(rho, T,'.')
   

        #ax[0].plot(rho, np.hstack(int_ ),'.')
        #ax[0].plot(rho, N,'.')        
        

        #ax[1].plot(rho, np.hstack(rot),'.')
        #ax[1].plot(rho, R,'.')
        #ax[2].axhline(0)
        #ax[1].axhline(0)
        #ax[0].axhline(0)

        
        #plt.show()
        
                
        

        #plt.plot( np.interp(T_all, T_avg,int_avg )*c)
        #plt.plot( np.hstack(int_ )- np.interp(T_all, T_avg,int_avg )*c)

         
        #plt.plot( np.hstack(rot ))
        #plt.plot( R)


        #plt.show()
        
        

        
         
        #plt.plot( np.hstack(int_ ))
        #plt.plot( N)


        #plt.show()
        
        

        
        #embed()

             
        #exit()
        
        cer['EQM'] = Tree({'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag})
        print('\t done in %.1fs'%(time()-TT))
        #print(cer)
        return cer
    
    
    
    
    
    
    def load_ts(self, tbeg,tend,systems, options=None):
    
        T = time()

        revision = 'BLESSED'
        zshift = 0 
        if options is not None:
            if 'TS revision' in options:
                selected,revisions = options['TS revision']
                revision = selected.get()
            if 'TS position error' in options:
                zshift = options['TS position error']['Z shift [cm]'].get()/100. #[m] 
  
        #use cached data
        self.RAW.setdefault('TS',Tree())
        ts = self.RAW['TS'].setdefault(revision,{'systems':systems})

        ts['systems'] = list(systems)
        systems = list(set(systems)-set(ts.keys()))
        
        
        #update mapping of the catched data
        ts = self.eq_mapping(ts, dz =zshift )            
        ts.setdefault('diag_names',Tree())

        if len(systems) == 0:
            #assume that equilibrium could be changed
            return ts

        print_line( '  * Fetching TS data ...')

        #extract setting from GUI 
        if revision!= 'BLESSED':
            revision = 'REVISIONS.'+revision
    

            
        signals = 'DENSITY', 'DENSITY_E', 'TEMP', 'TEMP_E','TIME','R','Z','lforder'
    
        tree = 'ELECTRONS'
        TDI = []        
        #prepare list of loaded signals
        for system in systems:
            if system in ts: continue
            tdi = f'\\{tree}::TOP.TS.{revision}.{system}:'
            ts['diag_names'][system]=['TS:'+system]
            for sig in signals:
                TDI.append(tdi+sig)
        
        out = mds_load(self.MDSconn, TDI, tree, self.shot)
        ne,ne_err,Te,Te_err,tvec,R,Z,laser = np.asarray(out).reshape(-1, len(signals)).T
        
        #get shot number with calibration data
        TDIcalib = f'\\{tree}::TOP.TS.{revision}.header:calib_nums'
        calibration_set = mds_load(self.MDSconn, [TDIcalib], tree, self.shot)[0][0]
        
        try:
            TDIcalib = [f'\\tscal::TOP.{sys}.hwmapints' for sys in systems]
            hw_ints = mds_load(self.MDSconn, TDIcalib, 'tscal', calibration_set)
            hw_lens = [np.atleast_2d(hwi)[:,2] for hwi in hw_ints]
        except:
            hw_lens = [np.zeros_like(z,dtype=int) for z in Z]
  
        for isys, sys in enumerate(systems):
            if len(tvec) <= isys or len(tvec[isys]) == 0: 
                ts['systems'].remove(sys)
                continue
            tvec[isys]/= 1e3        
            
            #these points will be ignored and not plotted (negative errobars )
            Te_err[isys][(Te_err[isys]<=0) | (Te[isys] <=0)]  = -np.infty
            ne_err[isys][(ne_err[isys]<=0) | (ne[isys] <=0)]  = -np.infty
       
            #guess corrupted channels
            ne_mean = np.average(ne[isys], weights = 1/ne_err[isys]+1e-30,axis=1)
            
            ne_mean_filter = medfilt(np.r_[ne_mean, ne_mean[::-1]], 3)[:len(ne_mean)]
            #embed()
            corrupted = (abs((ne_mean - ne_mean_filter)/ (ne_mean_filter+1)) > .2)[:,None]&(ne_err[isys] > 0)&(Te_err[isys] > 0)
            
            #remove them, but it can be returned by user
            ne_err[isys][corrupted] *= -1
            Te_err[isys][corrupted] *= -1
            
            #TODO sometimes the Te is near zero with very small uncertainty, how to remove these points? 
            
                
            channel = np.arange(Te_err[isys].shape[0])
            
            rho = self.eqm.rz2rho(R[isys],Z[isys]+zshift,tvec[isys],self.rho_coord)
         
            ts[sys] = Dataset('TS'+sys+'.nc',attrs={'system':sys})
            ts[sys]['ne'] = xarray.DataArray(ne[isys].T,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
            ts[sys]['ne_err'] = xarray.DataArray(ne_err[isys].T,dims=['time','channel'], attrs={'units':'m^{-3}'})
            ts[sys]['Te'] = xarray.DataArray(Te[isys].T,dims=['time','channel'], attrs={'units':'eV','label':'T_e'})
            ts[sys]['Te_err'] = xarray.DataArray(Te_err[isys].T,dims=['time','channel'], attrs={'units':'eV'})
            ts[sys]['diags']= xarray.DataArray( np.tile(('TS:'+sys,), ne[isys].T.shape),dims=['time','channel'])            
            ts[sys]['R'] = xarray.DataArray(R[isys], dims=['channel'], attrs={'units':'m'})
            ts[sys]['Z'] = xarray.DataArray(Z[isys],dims=['channel'], attrs={'units':'m'})
            ts[sys]['laser'] = xarray.DataArray(laser[isys],dims=['time'], attrs={'units':'-'})
            ts[sys]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
            ts[sys]['lens'] = xarray.DataArray(hw_lens[isys],dims=['channel'] )
            ts[sys]['time'] = xarray.DataArray(tvec[isys],dims=['time'], attrs={'units':'s'})
            ts[sys]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})

 
        print('\t done in %.1fs'%(time()-T))
        ts['EQM'] = Tree({'id':id(self.eqm),'dr':0, 'dz':zshift, 'ed':self.eqm.diag})

        return ts 
        
        
        
    def load_refl(self, tbeg,tend, options, TS=None):
        T = time()
        #TODO add REFLECT_F. calculate radial position by myself
        TS_align = options['Position error']['Align with TS'].get() #[m]  
        
        if 'REFL' in self.RAW and not TS_align:
            #assume that equilibrium could be changed
            return self.eq_mapping(self.RAW['REFL'])   
         
        print_line( '  * Fetching reflectometer data ...')

        load_bands = 'VO','V','QO','Q'
        tree = 'ELECTRONS'
        prefix ='\\'+tree+'::TOP.REFLECT.'
        Z0 = 0.0254
    
            
        TDI,bands = [],[]
        self.MDSconn.openTree(tree, self.shot)
 
        for band in load_bands:
            #check if data exists
            try:
                ne_size = self.MDSconn.get('getnci("REFLECT.'+band+'BAND.PROFILES:DENSITY'+'","LENGTH")').data()
            except:
                continue
            if ne_size  == 0:
                continue
            
            TDI.append('dim_of('+prefix+band+'BAND.PROFILES:DENSITY'+',0)')
            TDI.append(prefix+band+'BAND.PROFILES:DENSITY')
            TDI.append(prefix+band+'BAND.PROFILES:R')
            bands.append(band)
            #TDI.append(prefix+band+'BAND.PROFILES:DENSITY_ERR')#dont exist
            #TDI.append(prefix+band+'BAND.PROFILES:R_ERR')#dont exist
            #TDI.append(prefix+band+'PROCESSED:FREQUENCY')# empty
            
        self.MDSconn.closeTree(tree, self.shot)
        
        refl = self.RAW['REFL'] = Tree()
        refl['systems'] = bands


        #fetch data from MDS+
        out = mds_load(self.MDSconn, TDI, tree, self.shot)
     
     
        if np.size(out) == 0:
            printe( '\tNo Reflectometer data')
            return


        out = np.reshape(out, (len(bands), -1))
        
        #fetch TS data for alighnment
        if TS_align:
            if TS is None:
                TS = self.load_ts(tbeg,tend,['core'])
            R_midplane = np.linspace(1.8, 2.4,200)
            horiz_rho = self.eqm.rz2rho(R_midplane, Z0+0*R_midplane,
                                        TS['core']['time'].values,coord_out=self.rho_coord)

      
        refl['diag_names'] = Tree()

        for band, (tvec,ne, R) in zip(bands, out):
            if np.size(tvec) == 0:  continue
            tvec/= 1e3 #s
            R, ne = np.single(R.T), np.single(ne.T)
            z = np.zeros_like(R)+Z0
            phi = np.zeros_like(R)+255
            rho = self.eqm.rz2rho(R,z,tvec,self.rho_coord)
            R_shift = np.zeros_like(tvec)
            R_shift2 = np.zeros_like(tvec)

            if TS_align:

                rho_out = 0.7  #use only data outside rho_out

                TS_time  = TS['core']['time'].values
                TS_rho   = TS['core']['rho'].values
                TS_ne    = TS['core']['ne'].values 
                TS_neerr = TS['core']['ne_err'].values
                valid_ts_t = np.where(np.any(np.isfinite(TS_neerr) &(TS_rho > rho_out),1))[0]

                
    
                
                for it,t in enumerate(tvec):
                    #nearest valid TS point
                    its = valid_ts_t[np.argmin(np.abs(TS_time[valid_ts_t]-t))]
                    valid_ts = np.isfinite(TS_neerr[its]) &(TS_rho[its] > rho_out)
                    valid_rfl = (rho[it] > rho_out)&(ne[it] > 0)
    
                    R_ts = np.interp(TS_rho[its, valid_ts],horiz_rho[its],R_midplane) #midplane R coordinate for TS
                    R_rfl = R[it, valid_rfl]
                    ne_TS = TS_ne[its,valid_ts]/1e19
                    nee_TS = TS_neerr[its,valid_ts]/1e19 
                    ne_RFL = ne[it,valid_rfl]/1e19
                    shift = np.linspace(-0.1,0.1,50)
                    conv = np.zeros_like(shift)
                    for ish, s in enumerate(shift):
                        ne_RFL_shift = np.interp(R_ts, R_rfl+s, ne_RFL)
                        conv[ish] = np.sum((ne_RFL_shift-ne_TS)**2/nee_TS**2)
                    _,R_shift[it] = min_fine(shift, conv)
                rho = self.eqm.rz2rho(R+R_shift[:,None],z,tvec,self.rho_coord)

                


            channel = np.arange(ne.shape[1])
            refl[band] = Tree()

            refl['diag_names'][band] = ['REFL:'+band+'BAND']
            refl[band] = Dataset('REFL_'+band+'.nc',  attrs={'band':band})

            refl[band]['ne'] = xarray.DataArray(ne,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
             #just guess! of 10% errors!!
            refl[band]['ne_err'] = xarray.DataArray(ne*0.1+ne.mean()*0.01 ,dims=['time','channel'], attrs={'units':'m^{-3}'})
            refl[band]['diags']= xarray.DataArray(np.tile(('REFL:'+band+'BAND',), R.shape),dims=['time','channel'])
            refl[band]['R'] = xarray.DataArray(R,dims=['time','channel'], attrs={'units':'m'})
            refl[band]['R_shift'] = xarray.DataArray(R_shift,dims=['time'], attrs={'units':'m'})
            refl[band]['Z'] = xarray.DataArray(z,dims=['time','channel'], attrs={'units':'m'})
            refl[band]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
            refl[band]['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
            refl[band]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})

            
        refl['EQM'] = Tree({'id':id(self.eqm),'dr':np.mean(R_shift), 'dz':0,'ed':self.eqm.diag})

        print('\t done in %.1fs'%(time()-T))

        return refl
                
        
        
    def load_ece(self, tbeg,tend,options,TS=None):
    
        T = time()
        
        fast =  bool(options['systems']['ECE system'][1][1].get())
        bt_correction = float(options['load_options']['ECE system']["Bt correction"]['Bt *='].get())
        ts_correction = bool(options['load_options']['ECE system']["TS correction"]['rescale'].get())

        rate = 'fast' if fast else 'slow'
        suffix = 'F' if fast else ''

        #use cached data
        self.RAW.setdefault('ECE',Tree())
        self.RAW['ECE'].setdefault(rate,Tree())

        
        self.RAW['ECE'][rate].setdefault('ECE',Dataset('ECE.nc', attrs={'system':rate } ))

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
            
            if len(tvec) == 1:
                printe('No ECE data!')
                return
            
            tvec/=1e3
       
            nchs = sum([len(d)>0 for d in data_])
                        
            data_ = np.vstack(data_[:nchs]).T
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
        #embed()
        
        
        #OMFIT['test']['RESULTS']['AEQDSK']['BCENTR']
        
        
        #apply correction from GUI
        Btot *= bt_correction
        from scipy.constants import m_e, e, c,epsilon_0
        horiz_rho = self.eqm.rz2rho(r_in, z_in*np.ones_like(r_in),t_eq,coord_out=self.rho_coord)
        #Accounting for relativistic mass downshift
        #self.MDSconn.openTree('EFIT01',self.shot)

        #sig = '\\EFIT01::TOP.RESULTS.AEQDSK:BCENTR'
        
        #BCENTR = self.MDSconn.get(sig ).data()
        #tvec = self.MDSconn.get('dim_of('+sig+')').data()
        #bt = self.MDSconn.get('PTDATA("bt", 183503)').data()
        #bttvec = self.MDSconn.get('dim_of(PTDATA("bt", 183503))').data()

        
        try:
            zipfit = self.load_zipfit()

            Te_tvec = zipfit['Te']['time'].values
            Te_tvec[[0,-1]] = -10,100
    
            Te_ = interp1d(Te_tvec, zipfit['Te']['Te'].values,axis=0,copy=False, assume_sorted=True)(t_eq)
            Te = np.zeros_like(horiz_rho)

            for it,t in enumerate(t_eq):
                Te[it] = np.interp(horiz_rho[it], zipfit['Te']['rho'].values, np.abs(Te_[it]))

            v=np.sqrt(2*Te*(e/m_e))
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
    
            ne_ = zipfit['ne']['ne'].values.copy()
            ne_err =  zipfit['ne']['ne'].values*0.05 #zipfit errorbars are often wrong, while the fit is OK 
            ne_ += ne_err # upper boundary, to be sure that affected measurements will be removed
            ne_tvec = zipfit['ne']['time'].values
            ne_rho  = zipfit['ne']['rho'].values
            ne = np.zeros((len(ne_tvec), len(r_in)))
            for it,t in enumerate(ne_tvec ):
                iteq = np.argmin(abs(t_eq-t))
                ne[it] = np.interp(horiz_rho[iteq], ne_rho ,  ne_[it])
            
            f_CE = interp1d(t_eq, wce/(2*np.pi*1e9),fill_value="extrapolate",
                            axis=0,copy=False, assume_sorted=True)(ne_tvec)  #f_ce at R
            f_PE = np.sqrt(np.maximum(ne,0)*e**2/m_e/epsilon_0)/(2*np.pi*1e9)#f_pe at R
            f_RHC=(0.5*f_CE)+np.sqrt((0.5*f_CE)**2 + f_PE**2)
            f_cut = np.maximum.accumulate(f_RHC[:,::-1], axis=1)[:,::-1]  #propagate maximum from LFS to HFS 
            
            #embed()
            f_cut_loc = np.zeros((len(ne_tvec), nchs))
            for it,t in enumerate(ne_tvec):
                iteq = np.argmin(abs(t_eq-t))
                f_cut_loc[it] = np.interp(R[iteq],  r_in, f_cut[it])
                
                
            #ind = (ne_tvec > 2 )&(ne_tvec < 5)
            #plt.axhline(110,label='f ECH')
            #plt.plot(r_in, f_RHC[ind].mean(0), label='f RHC cutoff')
            #plt.plot(r_in, f_CE[ind].mean(0), label='f CE')
            #plt.plot(r_in, f_CE[ind].mean(0)*2, label='2x f CE')
            #plt.plot(r_in, f_PE[ind].mean(0), label='f PE')
            #plt.xlabel('R [m]')
            #plt.ylabel('f [GHz]')
            #plt.legend()
            #plt.show()
                
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


        return Tree({'ECE':ece,'diag_names':{'ECE':['ECE']}, 'systems':['ECE']
                                ,'EQM':Tree({'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag})})
    
    
    def load_co2(self, tbeg,tend, calc_weights=True):
        
        T = time()

         
        CO2 = self.RAW.setdefault('CO2',Tree())

        #update mapping of the catched data
        if 'systems' in CO2 and (not calc_weights or 'weight' in CO2[CO2['systems'][0]]):
            CO2 = self.eq_mapping(CO2) 
            #return catched data 
            return CO2
        
        
        CO2['systems'] = ['CO2']
        CO2.setdefault('diag_names',Tree())
        CO2['diag_names']['CO2'] = ['CO2 interferometer']
        CO2['CO2'] = []

        print_line( '  * Fetching CO2 interferometer data ...')

        #Set the z top and bottom for a purely vertical LOS
        Z_top = 1.24
        Z_bottom = -1.375

        #Set the R left and right for a purely horizontal LOS
        R_lfs = 2.36
        R_hfs = 1.01      

        #Format: (name, stat_error_threshold, LOS_pt1, LOS_pt2)
        channels = [('V1',2.4, [1.48,Z_top], [1.48,Z_bottom]),
                    ('V2',2.4, [1.94,Z_top], [1.94,Z_bottom]),
                    ('V3',3.1, [2.10,Z_top], [2.10,Z_bottom]),
                    ('R0',3.1, [R_lfs,0   ], [R_hfs,0      ])]

        sys_name = 'CO2 interferometer'
        TDI_DENS, TDI_STAT = [],[]
        tree = 'ELECTRONS'
        los_names = ['V1','V2','V3','R0']
        for name in los_names:
            TDI_DENS.append('_x=\\BCI::DEN%s'%name)
            TDI_STAT.append('_y=\\BCI::STAT%s'%name)
        TDI_dens_t= '_t=dim_of(_x); [_t[0], _t[size(_t)-1]]'
        TDI_stat_t= '_u=dim_of(_y); [_u[0], _u[size(_u)-1]]'
        TDI = ['['+','.join(TDI_DENS)+']',TDI_dens_t,
               '['+','.join(TDI_STAT)+']',TDI_stat_t]
       
        #fetch data,  slow
        ne_,co2_time,stat,stat_time = mds_load(self.MDSconn, TDI, tree, self.shot)
        
        co2_time   = np.linspace(co2_time[0]/1e3, co2_time[1]/1e3, ne_.shape[1])
        stat_time  = np.linspace(stat_time[0]/1e3, stat_time[1]/1e3, stat.shape[1])
 
        fringe_jump = 1.05e20 #/m^2
        nt = len(ne_.T)//4*4
        ne_lowres = np.median(ne_[:,:nt].reshape(4,-1,4),-1)*1e6/2
        tvec_lowres = np.median(co2_time[:nt].reshape(-1,4),-1)
        ne_correction = ne_lowres-np.unwrap(ne_lowres*(2*np.pi/fringe_jump))*fringe_jump/(2*np.pi)
        ne_correction = interp1d(tvec_lowres, ne_correction,axis=1,kind='nearest',copy=False, fill_value='extrapolate', assume_sorted=True )(co2_time)
        
        
        #plt.plot(ne_[3]-ne_correction[3]/1e6*2-3*fringe_jump-ne_[3,:100].mean(),'--')
        ##plt.plot(ne_[3])
        #plt.show()
        
        

        
        
        n_path = 501
        downsample = 2
        n_ch = len(channels)

        co2_time = np.mean(co2_time[:len(co2_time)//downsample*downsample].reshape(-1,downsample),-1)
        nt = len(co2_time)
        valid = np.zeros((nt, n_ch),dtype=bool)
        ne = np.zeros((nt, n_ch), dtype='single')
        ne_err = np.zeros((nt, n_ch), dtype='single')
        R = np.zeros((n_ch,n_path), dtype='single')
        Z = np.zeros((n_ch,n_path), dtype='single')
        L = np.zeros((n_ch,n_path), dtype='single')
        L_cross = np.zeros((nt,n_ch), dtype='single')
        weight= np.zeros((nt, n_ch, n_path), dtype='single')
        Rlcfs,Zlcfs = self.eqm.rho2rz(0.995)

        t = np.linspace(0,1,n_path, dtype='single')    

        for ilos, (name, stat_error_thresh, LOS_pt1, LOS_pt2) in enumerate(channels):
            
            ne_[ilos] /= 2. #Because of the double pass through the plasma
            ne_[ilos] *= 1e6  #/m^2
            
            #apply fringe jump correction
            ne_[ilos] -= ne_correction[ilos]
            #smooth out jumps around the corrections
            invalid = np.r_[False,np.abs(np.diff(ne_[ilos])) > fringe_jump/5]
            ind = np.arange(len(ne_.T))
            ne_[ilos,invalid] = np.interp(ind[invalid],ind[~invalid], ne_[ilos,~invalid])
    
            
            #Check the status channel - provides a flag for when the signal is not useable due to things like fringeskips
            #signal_invalid = stat_time[stat[ilos]>stat_error_thresh]
            last_valid_ind = -1
            
            #if more 3 frindge jumps, consider as unreliable
            invalid = np.where(np.abs(ne_correction[ilos]/fringe_jump) > 3)[0]
            if len(invalid) > 0:
                last_valid_ind = invalid[0]
            
            #if any(signal_invalid):
                #min_valid_time = signal_invalid[0]
                #last_valid_ind = np.argmin(np.abs(co2_time - min_valid_time))
                ##Chop the data after the last valid index,
                ##need to deal with the case where last_valid_ind is zero...?


            #downsample
            last_valid_ind//= downsample
            ne[:,ilos]   = np.mean(ne_[ilos][:len(ne_[ilos])//downsample*downsample].reshape( -1, downsample), -1)
            valid[:last_valid_ind, ilos] = True
          
            
            #Calculate the error for the signal based on the drift before t=0
            ne_err[:,ilos] = np.median(np.abs(ne[:,ilos][co2_time<0]))+ne[:,ilos]*.05  #guess 5% error
            ne_err[last_valid_ind:, ilos] *= -1  #data will be disabled, but can be enabled in the GUI
            ne_err[:,ilos]
            R[ilos] = (LOS_pt2[0]-LOS_pt1[0])*t+LOS_pt1[0]
            Z[ilos] = (LOS_pt2[1]-LOS_pt1[1])*t+LOS_pt1[1]
            L[ilos] = np.hypot(R[ilos]-LOS_pt1[0],Z[ilos]-LOS_pt1[1])
      
            if calc_weights:
                dL = np.gradient(L[ilos])
                LOS_pt1 = np.array((LOS_pt1[0],LOS_pt1[1]))
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
                L_cross[:,ilos] = np.interp(co2_time, self.eqm.t_eq[L_cross_>0] ,L_cross_[L_cross_>0])
                L_cross[:,ilos] = np.maximum(L_cross[:,ilos], .1) #just to avoid zero division
                L_cross[:,ilos]*= 0.9 # correction just for better plotting of the data 
                #convert from m^-2 -> m^-3
                weight[:,ilos,:] = dL/L_cross[:,ilos][:,None]
                ne[:,ilos] /= L_cross[:,ilos]
                ne_err[:,ilos] /= L_cross[:,ilos]
               
   
        #remove offset
        ne  -= ne[(co2_time > -2)&(co2_time < -.5)].mean(0) 
        #correpted measurements
        ne_err[ne < 0]  = np.infty
 

        CO2['CO2'] = Dataset('interfer.nc')
        CO2['CO2']['channel'] = xarray.DataArray( los_names ,dims=['channel'])
        CO2['CO2']['path'] = xarray.DataArray( t ,dims=['path'])
        CO2['CO2']['time'] = xarray.DataArray( co2_time ,dims=['time'], attrs={'units':'s'})
        CO2['CO2']['valid'] = xarray.DataArray( valid ,dims=['time','channel'], attrs={'units':'s'})
        CO2['CO2']['ne'] = xarray.DataArray(ne, dims=['time', 'channel'], attrs={'units':'m^{-3}','label':'n_e'})
        CO2['CO2']['ne_err'] = xarray.DataArray(ne_err,dims=['time', 'channel'], attrs={'units':'m^{-3}'})
        CO2['CO2']['diags']= xarray.DataArray( np.tile(('CO2 interferometer',), (nt, n_ch)),dims=['time', 'channel'])
        
              #Format: (name, stat_error_threshold, LOS_pt1, LOS_pt2)
        channels = [('V1',2.4, [1.48,Z_top], [1.48,Z_bottom]),
                    ('V2',2.4, [1.94,Z_top], [1.94,Z_bottom]),
                    ('V3',3.1, [2.10,Z_top], [2.10,Z_bottom]),
                    ('R0',3.1, [R_lfs,0   ], [R_hfs,0      ])]
        
        
        CO2['CO2']['R'] = xarray.DataArray(R[None],dims=['none','channel','path'],attrs={'units':'m',} )
        CO2['CO2']['Z'] = xarray.DataArray(Z[None],dims=['none','channel','path'],attrs={'units':'m'} )
        if not 'rho' in CO2['CO2']:
            CO2['CO2']['rho'] = xarray.DataArray(np.zeros((nt, n_ch, n_path), dtype='single'),dims=['time', 'channel','path'])
        CO2['CO2']['L'] = xarray.DataArray(L,dims=['channel','path'],attrs={'units':'m'})
        CO2['CO2']['L_cross']  =  xarray.DataArray(1)


     
        if 'EQM' in CO2: CO2.pop('EQM')
        
        CO2 = self.eq_mapping(CO2) 
        rho_tg = CO2['CO2']['rho'].values.min(2)
        if calc_weights:
            weight[CO2['CO2']['rho'].values >= 1.05] = 0 #make is consistent with C02 correction
            CO2['CO2']['weight']  = xarray.DataArray(weight, dims=['time', 'channel','path'])
            CO2['CO2']['L_cross'] = xarray.DataArray(L_cross,dims=['time','channel'],attrs={'units':'m'})
            #rho_tg = np.average(CO2['CO2']['rho'].values,2, (CO2['CO2']['rho'].values < 0.9)*weight)
            #embed()

        CO2['CO2']['rho_tg'] = xarray.DataArray(np.single(rho_tg),dims=['time', 'channel'])


        CO2['EQM'] = Tree({'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag})

        print('\t done in %.1fs'%(time()-T))

        return CO2


    
    def co2_correction(self,TS, tbeg,tend):
        
        T = time()
        if not 'core' in TS or not 'tangential' in TS:
            print('CO2 correction could not be done, either core or tangential TS data are missing')
            return TS
        
        #if already corrected return corrected values
        if 'ne_corr' in TS['tangential'] and 'ne_corr' in TS['core']:
            TS = deepcopy(TS)
            TS['tangential']['ne'] = TS['tangential']['ne_corr']
            TS['core']['ne'] = TS['core']['ne_corr']
            return TS 
  
        #BUG ignoring user selected wrong channels of TS in GUI 
        
        CO2 = self.load_co2(tbeg,tend, calc_weights = False)['CO2']
        T = time()

        print_line( '  * Calculate TS correction using CO2 ...')
        
        
        TS['tangential']['ne_corr'] = TS['tangential']['ne'].copy()
        TS['core']['ne_corr'] = TS['core']['ne'].copy()


        tbeg = max(tbeg, CO2['time'][0].values)
        tend = min(tend, CO2['time'][-1].values)
   

        tvec_compare =  TS['core']['time'].values
        laser_index  =  TS['core']['laser'].values
        t_ind = (tvec_compare > tbeg) & (tvec_compare < tend)
        ind = np.argsort(tvec_compare[t_ind])
        tvec_compare = tvec_compare[t_ind][ind]
        laser_index  = laser_index[t_ind][ind]
 
       
        LOS_rho = interp1d(CO2['time'].values, CO2['rho'].values,copy=False, axis=0,
                           assume_sorted=True)(np.clip(tvec_compare,*CO2['time'].values[[0,-1]]))
        LOS_L = CO2['L'].values

        
        #1)  correction of the tang system with respect to core

        tang_tvec = TS['tangential']['time'].values
        ind = slice(*tang_tvec.searchsorted((tbeg, tend)))
        tang_tvec = tang_tvec[ind]
        tang_lasers  = TS['tangential']['laser'].values[ind]
        tang_lens  = TS['tangential']['lens'].values
        tang_lasers_list = np.unique(tang_lasers)
        tang_ne  = TS['tangential']['ne'].values[ind]
        tang_err  = TS['tangential']['ne_err'].values[ind]
        tang_rho = TS['tangential']['rho'].values[ind]
        
        
        core_lens  = TS['core']['lens'].values
        core_rho= interp1d(TS['core']['time'],TS['core']['rho'],axis=0,copy=False, assume_sorted=True)(tang_tvec)
        core_ne = interp1d(TS['core']['time'],TS['core']['ne'],axis=0,copy=False, assume_sorted=True)(tang_tvec)
        core_err= interp1d(TS['core']['time'],TS['core']['ne_err'] ,axis=0,copy=False, assume_sorted=True,kind='nearest')(tang_tvec)

        #remove corrupted channels with more than 20% wrong points (usualy in SOL, not important)
        corrupted_tg = np.sum(~np.isfinite(tang_err),0)>tang_err.shape[0]/5.
        corrupted_core = np.sum(~np.isfinite(core_err),0)>core_err.shape[0]/5.
        
 
        rho = np.hstack((tang_rho[:,~corrupted_tg],core_rho[:,~corrupted_core]))
        rho_sort_ind  = np.argsort(rho,axis=1)
        #tang_lasers = tang_lasers[~corrupted_tg]
        core_ne = core_ne[:,~corrupted_core]

        
        time_sort_ind = np.tile(np.arange(rho.shape[0]), (rho.shape[1], 1)).T
        
        rho = rho[time_sort_ind, rho_sort_ind]
        def cost_fun(corr):
            tang_ne_ = np.double(tang_ne[:,~corrupted_tg])
            for l,c in zip(tang_lasers_list, corr):
                tang_ne_[tang_lasers==l] *= np.exp(c)
            core_ne_ = np.double(core_ne)
            #core_ne_[:,core_lens[~corrupted_core] == 0]*= np.exp(corr[-1])
            ne  = np.hstack((tang_ne_,core_ne_ ))/1e19
            ne  = ne[time_sort_ind, rho_sort_ind]
            val = np.nansum((np.diff(ne)/(ne[:,1:]+ne[:,:-1]+.1))**2)
            return val
            
        from scipy.optimize import minimize 
   
        p0 = np.zeros(len(tang_lasers_list))
        opt = minimize(cost_fun,  p0, tol = .0001 )
        corr = np.exp(opt.x)
 
     
        #correct core system
        for l,c in zip(tang_lasers_list , corr):
            TS['tangential']['ne_corr'].values[TS['tangential']['laser'].values == l] *= c

        #correct core r+0 system
        #TS['core']['ne_corr'].values[:,core_lens == 0] *= corr[-1]


        #interpolate data on the time of the core system
        N,R = [],[]
        for diag in ['core','tangential']:
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
        #interpolate density along LOS for each time, assume zero density outside of last measuremenst!
        LOS_ne = [np.interp(lr,r,n,right=0) for lr, r, n in zip(LOS_rho, R,N)]
        #do line integration
        LOS_ne_int = np.trapz(LOS_ne,LOS_L,axis=-1)
        core_lasers = np.unique(laser_index)


        co2_los_names  = CO2['channel'].values
        laser_correction     = np.ones((len(core_lasers),len(co2_los_names)))*np.nan
        laser_correction_err = np.ones((len(core_lasers),len(co2_los_names)))*np.nan
        los_dict =  {n:i for i,n in enumerate(co2_los_names)}
        valid = CO2['valid'].values
        time_co2 = CO2['time'].values
        
        ne = CO2['ne'].values*CO2['L_cross'].values  #m^-2
        fringe_jump = 1.05e20 #/m^2

        import matplotlib.pylab as plt
        for il, l in enumerate(core_lasers):
            ind = laser_index == l
  
            for ilos,los in enumerate( co2_los_names):
                if not np.any(valid[:,ilos]): continue
                t_ind = (tvec_compare > time_co2[valid[:,ilos]][0]) & (tvec_compare < time_co2[valid[:,ilos]][-1])
                if not np.any(ind&t_ind): continue
                co2_interp = np.interp(tvec_compare[ind&t_ind],time_co2[:],ne[:, ilos])

                #most likely overlooked fringe jump 
                valid_ = (abs(co2_interp-LOS_ne_int[ind&t_ind ,ilos]) < fringe_jump/2)&(co2_interp>0)
                
                
                ratio = LOS_ne_int[ind&t_ind ,ilos]/co2_interp
                #plt.plot(co2_interp[valid_])
                #plt.plot(LOS_ne_int[ind&t_ind ,ilos][valid_])
                #plt.show()
                
                laser_correction[il,ilos] = np.median(ratio[valid_])
                laser_correction_err[il,ilos] = ratio[valid_].std()/np.sqrt(np.sum(valid_))
                
 
                
        
        if not np.all(np.any(np.isfinite(laser_correction),1)):
            printe('CO2 rescaling was unsucessful')

            return TS
      
        mean_laser_correction = np.nanmedian(laser_correction,1)
              

        plot = False
        if plot:
            try:
                import matplotlib.pylab as plt

                #show comparims between TS and CO2
                interf_fig = plt.figure(figsize=(14,10))

                ax1 = interf_fig.add_subplot(211)
                for ilos, name in enumerate(co2_los_names):
                    p, = ax1.plot(time_co2,ne[:,ilos], label=name,lw=.5)
                    ax1.plot(time_co2[valid[:,ilos]],ne[valid[:,ilos],ilos],c=p.get_color())
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
                ax2.set_ylabel(r'$\langle n_e \rangle_\mathrm{TS}/\langle n_e \rangle_\mathrm{CO2} $')
                ax2.set_xlim(-0.5,len(co2_los_names)-0.5)

                for il, l in enumerate(core_lasers):
                    ax2.plot([],[],'o',c=colors[il],label='laser %d'%l)
                    for ilos,name in enumerate(co2_los_names):
                        ax2.errorbar(ilos+(il-5)/10.,laser_correction[il,ilos],
                                        laser_correction_err[il,ilos],fmt='o',c=colors[il])
                    ax2.plot(np.arange(len(co2_los_names))+(il-5)/10., 
                                laser_correction[il]/mean_laser_correction[il],'x',c=colors[il])

                leg = ax2.legend(loc='best',numpoints=1)
                for l in leg.legendHandles:  l.set_linewidth(4)
            
                ax2.set_xticks(np.arange(len(co2_los_names)), co2_los_names.tolist())#, rotation='vertical')
                locs = ax2.set_xticks(list(range(len(co2_los_names))))
                labels = ax2.set_xticklabels( co2_los_names )
                ax1.set_prop_cycle(None)    
                ax2.axvline(tbeg)
                ax2.axvline(tend)

                interf_fig.savefig('co2_correction_%d.pdf'%self.shot)

                interf_fig.clf()
                plt.close()
                del interf_fig
            except:
                pass
 

         #correction of laser intensity variation and absolute value
        for sys in ['core', 'tangential']:
            laser = TS[sys]['laser'].values
            data = TS[sys]['ne_corr'].values
            #total correction of all data
            data/= np.mean(mean_laser_correction)
            for l_ind, c in zip(core_lasers,mean_laser_correction):
                #laser to laser variation from of the core system
                data[laser == l_ind] /= c/np.mean(mean_laser_correction)
                
                    
        print('\t done in %.1fs'%(time()-T))
        print('\t\tCO2 corrections:\n\t\t', 'lasers: ',(np.round(mean_laser_correction,3)),
              '\n\t\t tang vs. core:', np.round(corr[:-1]/np.mean(mean_laser_correction),3))
              #', core r+0 vs. core r+1:', np.round(corr[-1]/np.mean(mean_laser_correction),3))
 
 
        #return corrected values
        TS = deepcopy(TS)
        TS['tangential']['ne'] = TS['tangential']['ne_corr']
        TS['core']['ne'] = TS['core']['ne_corr']
        return TS 
    
    
    def load_elms(self,option):
        
        node = option['elm_signal'].get()
        tree = 'SPECTROSCOPY'
        
        self.RAW.setdefault('ELMS',Tree())
        
        if node in self.RAW['ELMS']:
            return self.RAW['ELMS'][node]

        
        print_line( '  * Fetching ELM data... ')
        T = time()
        try:
            try:
                self.MDSconn.openTree(tree, self.shot)
                elms_sig = self.MDSconn.get('_x=\\'+tree+'::'+node).data()
                #elms_tvec = self.MDSconn.get('dim_of(_x)').data()/1e3
                tmin,tmax = self.MDSconn.get('_t = dim_of(_x); [_t[0], _t[size(_t)-1]]').data()/1e3
                elms_tvec =  np.linspace(tmin,tmax, len(elms_sig))                
            except:
                self.MDSconn.openTree(tree, self.shot)
                elms_sig = self.MDSconn.get('_x=\\'+tree+'::'+node+'da').data()
                tmin,tmax = self.MDSconn.get('_t = dim_of(_x); [_t[0], _t[size(_t)-1]]').data()/1e3
                elms_tvec =  np.linspace(tmin,tmax, len(elms_sig))
                
        except Exception as e:
            printe( 'MDS error: '+ str(e))
            elms_tvec, elms_sig = [],[]
        finally:
            try:
                self.MDSconn.closeTree(tree, self.shot)
            except:
                pass
        elm_time, elm_val, elm_beg, elm_end = [],[],[],[]
        if len(elms_sig):
            try:
                elm_time, elm_val, elm_beg, elm_end = detect_elms(elms_tvec, elms_sig)
            except Exception as e:
                print('elm detection failed', e)
                
        self.RAW['ELMS'][node] =  Tree({'tvec': elm_time, 'data':elm_val, 
                     'elm_beg':elm_beg,'elm_end':elm_end,'signal':node})
        print('\t done in %.1fs'%(time()-T))

        return  self.RAW['ELMS'][node]
    
    
    def load_sawteeth(self):
        #BUG how to detect them?
        import os
        try:
            sawteeth = np.loadtxt(os.path.expanduser('~/tomography/tmp/sawtooths_%d.txt'%self.shot))
        except:
            sawteeth = []
        
        return {'tvec':sawteeth}
    
    
    
    
    
    

 
 
 
 
#replace a GUI call

def main():
        
        
        ###x    
    import tkinter as tk

    #myroot = tk.Tk(className=' Profiles')

    #def get_C_data(shot):
        #from map_equ import equ_map
        #mdsserver = 'localhost'
        #import MDSplus

        #MDSconn = MDSplus.Connection(mdsserver)
        #print(shot)
        ##print_line( '  * Fetching EFIT01 data ...')
        #print_line( '  * Fetching EFIT01 data ...')
        #eqm = equ_map(MDSconn)
        #eqm.Open(shot, 'EFIT01', exp='D3D')

        ##load EFIT data from MDS+ 
        #T = time()
        #eqm._read_pfm()
        #eqm.read_ssq()
        #eqm._read_scalars()
        #eqm._read_profiles()
        #print('\t done in %.1f'%(time()-T))

        #rho_coord = 'rho_tor'

        #loader = data_loader(MDSconn, shot, eqm, rho_coord,raw={})

        
        #settings = OrderedDict()
        #I = lambda x: tk.IntVar(value=x)
        #S = lambda x: tk.StringVar(value=x)
        #D = lambda x: tk.DoubleVar(value=x)
    
        

        #settings.setdefault('nimp', {\
            #'systems':{'CER system':(['tangential',I(1)], ['vertical',I(1)],['SPRED',I(1)] )},
            #'load_options':{'CER system':OrderedDict((
                                    #('Analysis', (S('best'), (S('best'),'fit','auto','quick'))),
                                    #('Correction',{'Relative calibration':I(1),'nz from CER intensity':I(0),
                                                #'remove first data after blip':I(1)}  )))   }})
    
        ##embed()
        #loader( 'nimp', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])
        #del loader
        
        
    #b,shots, t1, t2, v1,v2 = np.loadtxt('beams_voltages_all_30.txt').T

    #for s in shots[::-1]:
        ##if s > 186496: continue
        #try:
            #get_C_data(int(s)) 
        #except:
            #try:
                #get_C_data(int(s)) 
            #except:
                ##raise
                #print('********************  shot %d failed *******************'%s)

    
    #exit()
    
    
 
 
    
    mdsserver = 'atlas.gat.com'
    import MDSplus
    try:
        MDSconn = MDSplus.Connection(mdsserver)
    except:
        mdsserver = 'localhost'

        MDSconn = MDSplus.Connection(mdsserver)
    TT = time()

    from map_equ import equ_map
    import tkinter as tk
    myroot = tk.Tk(className=' Profiles')

    shot =   175694
    rho_coord = 'rho_tor'
    shot =   175860
    #shot =   175849
    shot =   175691
    #shot =   175007
    #shot =   179119
    #shot =   179841
    shot =   119315
    shot =   180295
    shot =   175671
    shot =   170453
    #shot =   175860
    shot =   180616
    shot =   174489
    shot =   150427
    #shot =   179605
    ##shot =   175860
    shot =   175886
    shot =   159194#ne+c data
    shot =   163303 
    shot =   170777
    shot =   175694  #BUG blbe zeff1!!
    shot =   171534  #nefunguje omega
    shot =   174823   #TODO BUG blbne background substraction!!!!
    shot =   183185   #nefunguje
    #shot =   175861   
    shot =   178868 
    shot =   156908  #BUG blbe zeff1!!
    shot =  183212
 
    shot = 156908# BUG !! poskozeny prvni po blipu 
    shot = 183505# BUG !! poskozeny prvni po blipu 
    shot = 175602# BUG !! poskozeny prvni po blipu 

    shot = 184777# BUG !! poskozeny prvni po blipu 
    shot = 183167 #TODO blbe nC!!!
    shot = 179119 #many missing timeslices in IMPCON
    shot =  164988
    shot =    176778
    shot =  173054
    shot =  176815
    shot =  169513
    
    shot = 182725 #intensity nc funguje mizerne
    #shot = 185259
    #shot =   168873

    #shot = 164637
    #shot = 183505
    #shot = 184847
    #shot = 184844
    
    #shot = 185157  #BUg uplne blbe relativni kalibrace
    #shot = 184777
    shot = 184840
    #shot = 

    default_settings(MDSconn, shot  )
    #shot = 182725
    #shot =   180907
    #shot = 122596
    #shot =  90256
    #shot =  186473
    #shot = 185215  #nc is often turned off, why??
    #shot =  185307
    #shot = 182725
    #shot = 184847
    #shot = 163303
    #shot = 183151 #blbe loadeni Ti!
    #shot = 163543  #blbbe intenzity nC!
    #shot = 182643  #BUG SOL korekce moc nefunnguje
#30 179801 1.97 1.92
#30 179803 1.97 1.92
#30 179804 1.97 1.92
#30 179805 1.97 1.91
#30 179806 1.97 1.92
#30 179807 1.94 1.91
#30 179808 1.97 1.92
#30 179809 1.97 1.92
#30 179213 1.01 1.01
#30 178643 1.58 1.44
#30 178646 1.85 1.74
#30 178647 1.85 1.76
#30 178648 1.1 1.03
#30 178649 1.73 1.64
#30 178650 1.85 1.73
#30 178652 1.85 1.73
#30 178653 1.85 1.73
#30 178654 1.85 1.75
#30 178655 1.85 1.72
#30 178566 1.87 1.76
#30 178567 1.9 1.72
#30 178568 1.9 1.72
#30 178569 1.85 1.75
#30 178570 1.9 1.69
#210 179390 1.1 1.1
#210 179389 1.09 1.16
#330 178820 1.38 1.38

    #shots = [184847,184773,184777,184778,184822,184825,184826,184829,184831,184833,
             #184834,184837,184839,184840,184841,184843,184844,184845,184846,]
    
    #for shot in shots:
    #shot = 184846
    #shot = 184773
    #shot = 176278 #BUG error carbon density 
    #shot = 184831

    #175694  - better match between onaxis ver and tang denisty after rescaling
    #TODO nacitat 210 data zvlast
    #edge and core v330L system (178800)
    #TODO v legende se ukazuji i systemy co nemaji zadana data
    #kalibrovat core and edge poloidal zvlast
    #TODO H plasmas???

    print(shot)
    print_line( '  * Fetching EFIT02 data ...')
    eqm = equ_map(MDSconn)
    eqm.Open(shot, 'EFIT02', exp='D3D')

    #load EFIT data from MDS+ 
    T = time()
    eqm._read_pfm()
    eqm.read_ssq()
    eqm._read_scalars()
    eqm._read_profiles()
    print('\t done in %.1fs'%(time()-T))
    #import IPython 
    #IPython.embed()
            
    #print 'test'
    #exit()

    loader = data_loader(MDSconn, shot, eqm, rho_coord, raw = {})

    
    settings = OrderedDict()
    I = lambda x: tk.IntVar(value=x)
    S = lambda x: tk.StringVar(value=x)
    D = lambda x: tk.DoubleVar(value=x)

    
    ts_revisions = []

    settings.setdefault('Ti', {\
        'systems':{'CER system':(['tangential',I(1)], ['vertical',I(0)])},\
        'load_options':{'CER system':{'Analysis':(S('real'), ('best','fit','auto','quick','real')) ,
                                    'Corrections':{'Zeeman Splitting':I(1), 'Wall reflections':I(1)}} }})
        
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
        'systems':{'CER system':(['tangential',I(1)], ['vertical',I(0)],['SPRED',I(0)] )},
        'load_options':{'CER system':OrderedDict((
                                ('Analysis', (S('real'), (S('best'),'fit','auto','quick'))),
                                ('Correction',{'Relative calibration':I(1),'nz from CER intensity':I(0),
                                            'remove first data after blip':I(0)}  )))   }})

    settings.setdefault('Te', {\
    'systems':OrderedDict((( 'TS system',(['tangential',I(1)], ['core',I(1)],['divertor',I(0)])),
                            ('ECE system',(['slow',I(1)],['fast',I(0)])))),
    'load_options':{'TS system':{"TS revision":(S('BLESSED'),['BLESSED']+ts_revisions)},
                    'ECE system':OrderedDict((
                                ("Bt correction",{'Bt *=': D(1.0)}),
                                ('TS correction',{'rescale':I(0)})))   }})
        
    settings.setdefault('ne', {\
        'systems':OrderedDict((( 'TS system',(['tangential',I(1)], ['core',I(1)],['divertor',I(0)])),
                                ( 'Reflectometer',(['all bands',I(1)],  )),
                                ( 'CO2 interferometer',(['fit CO2',I(0)],['rescale TS',I(1)])) ) ),
        'load_options':{'TS system':{"TS revision":(S('BLESSED'),['BLESSED']+ts_revisions)},
                        'Reflectometer':{'Position error':{'Align with TS':I(1) }, }                        
                        }})
        
    settings.setdefault('Zeff', {\
        'systems':OrderedDict(( ( 'VB array',  (['tangential',I(0)],                 )),
                                ( 'CER VB',    (['tangential',I(0)],['vertical',I(0)])),
                                ( 'CER system',(['tangential',I(0)],['vertical',I(0)])),
                                ( 'SPRED',(['He+B+C+O+N',I(1)],)),                           
                                )), \
        'load_options':{'VB array':{'Corrections':{'radiative mantle':I(1),'rescale by CO2':I(1), 'remove NBI CX': I(1)}},\
                        'TS':{'Position error':{'Z shift [cm]':D(0.0)}},
                        'CER VB':{'Analysis':(S('auto'), (S('best'),'fit','auto','quick'))},
                        'CER system':OrderedDict((
                                ('Analysis', (S('best'), (S('best'),'fit','auto','quick'))),
                                ('Correction',    {'Relative calibration':I(1), 'nz from CER intensity':I(0)}), #BUG
                                ('TS position error',{'Z shift [cm]':D(0.0)})))
                        }\
            })
        

    settings['nC6'] = settings['nimp']
    settings['nAr16'] = settings['nimp']
    settings['nXX'] = settings['nimp']
    settings['nHe2'] = settings['nimp']
    settings['nN7'] = settings['nimp']
    settings['nCa18'] = settings['nimp']
    settings['nLi3'] = settings['nimp']

    settings['elm_signal'] = S('fs01up')
    settings['elm_signal'] = S('fs04')

    #print settings['Zeff'] 
    
    #exit()

    #TODO 160645 CO2 correction is broken
    #160646,160657  crosscalibrace nimp nefunguje
    #T = time()

    #load_zeff(self,tbeg,tend, options=None)
    #data = loader( 'Ti', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])
    loader.load_elms(settings)
    #data = loader( 'Zeff', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])

    data = loader( 'Ti', settings,tbeg=1.8, tend=5)
    #print(data)
#settings['nimp']= {\
    #'systems':{'CER system':(['tangential',I(1)], ['vertical',I(0)],['SPRED',I(0)] )},
    #'load_options':{'CER system':OrderedDict((
                            #('Analysis', (S('auto'), (S('auto'),'fit','auto','quick'))),
                            #('Correction',{'Relative calibration':I(1),'nz from CER intensity':I(1),
                                            #'remove first data after blip':I(0)}  )))   }}
#data = loader( 'nC6', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])
#print('--------------------')

#settings['nimp']={\
    #'systems':{'CER system':(['tangential',I(1)], ['vertical',I(1)])},
    #'load_options':{'CER system':OrderedDict((
                            #('Analysis', (S('best'), (S('best'),'fit','auto','quick'))),
                            #('Correction',{'Relative calibration':I(1),'nz from CER intensity':I(1)}  )))   }}


    print('\t done in %.1f'%(time()-T))
    #print(data)
    #plt.show()
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
 




