


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



mdsserver = 'atlas.gat.com'
import MDSplus
try:
    MDSconn = MDSplus.Connection(mdsserver)
except:
    mdsserver = 'localhost'

    MDSconn = MDSplus.Connection(mdsserver)
#TT = time()





def beam_get_fractions(Einj,  model='mickey'):
    #BUG is it valid also for H plasmas???
    # Copied from one of BAGs routines
    # Einj  is in keV
    # Fraction
    j = np.array([1.0, 2.0, 3.0])[:,None] 
 
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
 
        power_fractions = current_fractions / j 
        current_fractions /= current_fractions.sum(0) 

    elif model == 'mickey':
        ## Power Fraction stolen from original compute_impdens
        ## for consistency.
        power_fractions = np.zeros((3,  np.size(Einj)))
        power_fractions[0] = (65.1 + 0.19 * Einj) / 100.0
        power_fractions[1] = (-165.5 + 6.87 * Einj - 0.087 * Einj ** 2 + 0.00037 * Einj ** 3) / 100.0  # typo on fig!
        power_fractions[2] = 1.0 - power_fractions.sum(0)

        current_fractions = power_fractions * j 
        current_fractions /= current_fractions.sum(0) 

    elif model == 'NBIgroup':
        #https://diii-d.gat.com/diii-d/Beams_results
        power_fractions = np.zeros((3,  np.size(Einj)))
        power_fractions[0] = 68 + 0.11 * Einj
        power_fractions[1] = -159 + 6.53 * Einj - 0.082 * Einj ** 2 + 0.00034 * Einj ** 3
        power_fractions[2] =  192 - 6.64 * Einj + 0.082 * Einj ** 2 - 0.00034 * Einj ** 3
        power_fractions /= 100
        current_fractions = power_fractions * j 
        current_fractions /= current_fractions.sum(0) 

        
        
    else:
        print("Must choose Chuck or Mickey!!!")
        # This is implemented in Brians version, just not here, because it requires d3d_beam which requires neutralizer
        # current_fractions, SmixIn = d3d_beam(Einj,2.0,ZBEAM=1.0)
   
    density_fractions  = current_fractions * np.sqrt( j)
    density_fractions /= density_fractions.sum(0) 
        
   
    return {'cfracs': current_fractions, 'pfracs': power_fractions, 'nfracs': density_fractions}

def nbi_info(MDSconn,shot,  load_beams, nbi={}):
    #TODO assumes a constant voltage!! do not load NBI gas, it it slow 

    _load_beams = list(set(load_beams)-set(nbi.keys()))
    if len(_load_beams) == 0:
        return nbi
    
    try:
        MDSconn.openTree('NB',  shot)   
    except MDSplus.mdsExceptions.TreeFOPENR:
        print('tree failed', shot)
        return {b:{'fired':False} for b in load_beams}
    

    
    paths = ['\\NB::TOP.NB{0}:'.format(b[:2]+b[-1]) for b in _load_beams] 

    TDI = [p+'pinj_scalar' for p in paths] 
    pinj_scal = MDSconn.get('['+','.join(TDI)+']')
    fired = pinj_scal > 1e3
    

    #create NBI info dictionary
    for b,f in zip(_load_beams,fired):
        nbi.setdefault(b,{'fired':f})
        
    if not any(fired):
        return nbi

    pinj_scal = np.array(pinj_scal)[fired]
    _load_beams = np.array(_load_beams)[fired]
    paths = np.array(paths)[fired]
    
    TDI  = [p+'PINJ_'+p[-4:-1] for p in paths]
    TDI += ['dim_of('+TDI[0]+')']
    try:
        pow_data = list(MDSconn.get('['+','.join(TDI)+']').data())
    except MDSplus.TreeNODATA:
        print('loading failed', shot)
        return {b:{'fired':False} for b in load_beams}
    
    tvec = pow_data.pop()/1e3
    

    gas = np.array(['D2']*len(_load_beams))
    
    ##slowest step!!
    #TDI = [p+'gas' for p in paths]
    #gas = MDSconn.get('['+','.join(TDI)+']').data()
    #if not isinstance(gas[0],str): 
        #gas = [r.decode() for r in gas]
            
    

    if shot  > 169568:  #load time dependent voltage
        TDI = [p+'VBEAM' for p in paths]
    else:   #load scalar values
        TDI = [p+'NBVAC_SCALAR' for p in paths]
    #volt_data = self.MDSconn.get('['+','.join(TDI)+']').data()
    volt_data = list(MDSconn.get('['+','.join(TDI)+']').data())

    MDSconn.closeTree('NB', shot)
    

    b21sign= 1 if shot < 124700 else -1 #BUG?? 210 always assumed to be counter current
    Rtang = {'30L':114.6, '30R':76.2, '21L':76.2*b21sign,
            '21R': 114.6*b21sign, '33L':114.6, '33R':76.2}
    #rmin=[1.15,.77,1.15,.77,1.15,.77,1.15,.77] in   ./4dlib/BEAMS/atten.pro
    #fill NBI info dictionary
    for i,b in enumerate(_load_beams):
        beam = nbi.setdefault(b,{})
        beam['fired'] = pinj_scal[i] > 1e3
        beam['volts'] = volt_data[i]
        if np.size(volt_data[i]) > 1:
            #ignore time varying voltage
            beam['volts'] = beam['volts'][pow_data[i] > 1e3].mean() 
        beam['power'] = pinj_scal[i]
        beam['pow_frac'] = beam_get_fractions(beam['volts']/1e3, 'chuck')['pfracs']
        beam['Rtang'] = Rtang[b[:2]+b[-1]]*0.01
        beam['power_timetrace'] = pow_data[i]
        beam['power_time'] = tvec
        beam['mass'] = {'D2':2.014, 'H2':1.007, 'He': 4.0026 }[gas[i]]
        beam['beam_fire_time'] = np.nan
    
    return nbi

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

def check_cer(shot):
    imps = []
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
        for l in _line_id:
            try:
                line_id.append(l.decode())
            except:
                pass
            
    
        ulines = np.unique(line_id)
        for l in ulines:
            try:
                tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', l)
                imp, Z = tmp.group(1), tmp.group(2)
                imps.append(imp+str(roman2int(Z) ))
            except:
                #embed()
                print(shot, l)
                imps.append('XX')

        return imps
    except:
        return []
#signal = 'cerqtit25' 
##for shot in range(  182200, 172800 ,-1):
#for shot in range( 175860, 185500):

    ##print(shot)
    #try:

        #tag  = MDSconn.get('findsig("'+signal+'",_fstree)').value
        #fstree = MDSconn.get('_fstree').value 
        #MDSconn.openTree(fstree,shot)
        #zdata = MDSconn.get('_s = '+tag).data()
        #if len(zdata > 0):
            #print(shot, len(zdata))
    #except:
        #print('error', shot)
        #continue


#exit()
load_beams = '30L','30R','210L','210R','330L','330R'
#load_beams =  '330L','330R'
my_data = np.genfromtxt('beams_corrections_int.txt', delimiter=';')
shots = my_data[:,0]



#for shot in range( 160257, 185500):
for shot in shots:
    #shot = 135847
    MDSconn.openTree('NB',  shot)   
    perveance = np.zeros(len(load_beams))
    for i,b in enumerate(load_beams):
        #b = 
        perveance[i] =  MDSconn.get('\\NB::TOP.NB{0}:REAL32'.format(b[:2]+b[-1]) )[4]
    #print()
    with open('perveance.txt', "a") as file:
        file.write(str(shot)+'\t'+'\t'.join([str(p) for p in perveance])+ '\n')

exit()

for shot in range( 160257 , 185500):

    #shot = 185800
    try:
        #print(shot)
        #embed()
        
        if len(check_cer(shot)) == 0:
            continue
        try:
            nbi = nbi_info(MDSconn,shot,  load_beams, {})
        except MDSplus.mdsExceptions.TdiTIMEOUT:
            try:
                nbi = nbi_info(MDSconn,shot,  load_beams, {})
            except MDSplus.mdsExceptions.TdiTIMEOUT:
                print('TdiTIMEOUT '+str(shot))
                continue
            except MDSplus.mdsExceptions.MDSplusERROR:
                print('MDSplusERROR')
                continue
        except MDSplus.mdsExceptions.MDSplusERROR:
            print('MDSplusERROR')
            mdsserver = 'localhost'
            MDSconn = MDSplus.Connection(mdsserver)
            continue
                
        for b in [load_beams[0][:-1]]:
            if nbi[b+'L']['fired'] and nbi[b+'R']['fired']:
                #print(b)
                #if np.abs(nbi[b+'L']['volts']-nbi[b+'R']['volts'])/1e3 < 4:
                
                t = nbi[b+'L']['power_time']
                l = nbi[b+'L']['power_timetrace']/nbi[b+'L']['power'] > 0.5
                r = nbi[b+'R']['power_timetrace']/nbi[b+'R']['power'] > 0.5
                
                lr = l & r
                l &= ~lr
                r &= ~lr
                if not any(l) or not any(r):
                    continue
                #try:
                tmin = max(t[l].min(), t[r].min())
                tmax = min(t[l].max(), t[r].max())
                
                ind = (t > tmin)&(t < tmax)
                dt = np.mean(np.diff(t))
                tl = round(dt*np.sum(l[ind]),2)
                tr = round(dt*np.sum(r[ind]),2)
                if tl  > .5 and tr > .3:
                    #embed()
                    print(b, shot, tl,tr,nbi[b+'L']['volts'], nbi[b+'R']['volts'])
                    with open('beams_voltages_all_30.txt', "a") as file:
                        file.write(b+' ' +str(shot)+ ' ' + str(tl)+ ' ' +str(tr)+ ' ' +str(nbi[b+'L']['volts'])+ ' ' + str(nbi[b+'R']['volts'])+'\n')
                #except:
                        #continue
                
    except:
        print('Error ', shot)
        #raise
    #embed()
