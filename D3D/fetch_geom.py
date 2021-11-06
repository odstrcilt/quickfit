





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


from matplotlib.pylab import *
mdsserver = 'atlas.gat.com'
import MDSplus
try:
    MDSconn = MDSplus.Connection(mdsserver)
except:
    mdsserver = 'localhost'

    MDSconn = MDSplus.Connection(mdsserver)
#TT = time()



def mds_load(MDSconn,TDI, tree, shot):
 
    MDSconn.openTree(tree, shot)
    data = []
    for tdi in TDI:
        try:
            data.append(np.atleast_1d(MDSconn.get(tdi).data()))
        except:
            print('Loading failed: ')
            data.append(np.array([]))
    try:
        MDSconn.closeTree(tree, shot)
    except:
        pass
        
    return data

tree = 'IONS'

MDSconn.openTree(tree, 175860)

order='\\IONS::TOP.CER.CALIBRATION.BEAM_ORDER'
beam_order = list(MDSconn.get(order).data())
beam_order = [b.decode() for b in beam_order]
#all_ch = np.unique(np.hstack(loaded_chans))
all_ch = ['T%.2d'%i for i in range(1,57)]
 
load_systems = ['tangential']#,  'vertical'
beam_geom = []
R = []
shots = []
loaded_chans = []
for shot in range(156600, 185000, 10):
#for shot in range(177000, 185000, 100):

    print(shot)
    #shot = 180520
    try:
        TDI_G = ''
        TDI_R = ''
        loaded_chan = []
        MDSconn.openTree(tree, shot)
        
        analysis_type = 'cerauto'


        #prepare list of loaded channels
        for system in load_systems:

            #embed()
            path = 'CER.%s.%s.CHANNEL*'%(analysis_type,system)
            #nodes = self.MDSconn.get('getnci("'+path+'","fullpath")').data()

        
            path = 'CER.CALIBRATION.%s.CHANNEL*'%(system)

            lengths = MDSconn.get('getnci("'+path+':BEAMGEOMETRY","LENGTH")').data()

            nodes = MDSconn.get('getnci("'+path+'","fullpath")').data()
                            
            for node,length in zip(nodes, lengths):
                #print(length)

                if length == 0: continue
            #for node in nodes:
                try:
                    node = node.decode()
                except:
                    pass

                node = node.strip()
                loaded_chan.append(system[0].upper()+node[-2:])
        
        
                TDI_G += node+':BEAMGEOMETRY'+','
                TDI_R += node+':PLASMA_R'+','

        #fast fetch of MDS+ data

 
        if len(loaded_chan) > 0:

            _beam_geom = MDSconn.get('['+TDI_G[:-1]+']').data()
            _R = MDSconn.get('['+TDI_R[:-1]+']').data()
            
            
            if len(_R) > 0 and  _R.shape[1] == 8: 
                #print( '--')
        
                gg = np.zeros((len(all_ch),8))
                gg[np.in1d(all_ch, loaded_chan)] = _beam_geom
                rr = np.zeros((len(all_ch),8))
                rr[np.in1d(all_ch, loaded_chan)] = _R
 
                beam_geom += [gg]
                R += [rr]
                shots.append(shot)
                print('done')
  
        try:
            MDSconn.closeTree(tree, shot)
        except:
            pass
    except:
        pass



maxg = np.dstack(beam_geom).max(0).T
maxg[maxg == 0] = nan

used_beams = [0,1,2,3],[6,7]
for ib,b in zip(used_beams,  ['30+330','210']):
    g = np.dstack(beam_geom)[:,ib]
    g = g.reshape(g.shape[0]*len(ib), -1)
    inds = any(g > 0,0)
    indch = all(g[:,inds] > 0,1)
  
    u,ii,i = unique(g[:,inds][indch], return_inverse=True, return_index=True, axis=1)
    plot(array(shots)[ inds], argsort(ii)[i]+mean(ib)/100,'.-', label=b)
legend()
#[axvline(s,c='k') for s in s_unique]
#[text(s,.5,s) for s in s_unique]

ylabel('calibration index')
xlabel('shot')

show()


    
plot(shots, maxg)
show()


#[[s,b.shape] for b,s in zip(beam_geom, s_unique)]


    


#for g in g_unique:
    #plot(s_unique,   [g[g[:,0]>0,0].mean() for g in g_unique])
#show()




        
        
f,ax = plt.subplots(3,2)
used_beams = [0,1,2,3,6,7]
for ib,a,b in zip(used_beams,ax.flatten(), np.array(beam_order)[used_beams]):
    
    g_unique, r_unique, s_unique = [ ],[],[]
    for r,g,s in zip(R,beam_geom,shots):
        if len(g_unique)==0 or np.abs((g[:,ib]!= g_unique[-1])).sum()> 0:
            if len(g_unique)==0 or  np.abs((abs(g[:,ib]- g_unique[-1]) > .1)[(g_unique[-1] > 0)&(g[:,ib] > 0)]).sum()> 5:
                g_unique.append(g[:,ib])
                r_unique.append(r[:,ib])
                s_unique.append(s)
            else:
                valid = g[:,ib] != 0
                g_unique[-1][valid] = g[valid,ib]
                r_unique[-1][valid] = r[valid,ib]
 
    a.set_title(b) 
    _, sind = np.unique(np.dstack(g_unique), return_index=True,axis=-1)
    sind = sort(sind)
    
    for iu in sind[::-1]:
        valid = (g_unique[iu] > 0 )&( r_unique[iu]<2.99)
        r = r_unique[iu][valid]
        srind = np.argsort(r)
        a.plot(r[srind], g_unique[iu][valid][srind],'-o', label=s_unique[iu])
    a.legend()

        
f,ax = plt.subplots(3)
used_beams = [0,1],[2,3],[6,7]
for ib,a,b in zip(used_beams,ax.flatten(), [30,330,210]):
    a.set_title(b) 
 
    g_unique, r_unique, s_unique = [ ],[],[]
    for r,g,s in zip(R,beam_geom,shots):
        if len(g_unique)==0 or np.abs((g[:,ib]!= g_unique[-1])).sum()> 0:
            if len(g_unique)==0 or  np.abs((abs(g[:,ib]- g_unique[-1]) > .1)[(g_unique[-1] > 0)&(g[:,ib] > 0)]).sum()> 5:
                g_unique.append(g[:,ib])
                r_unique.append(r[:,ib])
                s_unique.append(s)
            else:
                valid = np.all(g[:,ib] != 0,1)
                g_unique[-1][valid] = g[valid][:,ib]
                r_unique[-1][valid] = r[valid][:,ib]
    
    _, sind = np.unique(np.dstack(g_unique), return_index=True,axis=-1)
    sind = sort(sind)
    for iu in sind[::-1]:
        valid = np.all((g_unique[iu] > 0)&( r_unique[iu]<2.99),1)
        r = r_unique[iu][:,0][valid]
        sind = np.argsort(r)
        a.plot(r[sind], g_unique[iu][:,0][valid][sind]/g_unique[iu][:,1][valid][sind],'-o', label=s_unique[iu])
    
    a.legend()
    #a.set_ylabel('G/median(G)')
    a.set_ylabel('$(G_L/G_R)$')

show()


        
f,ax = plt.subplots(3)
used_beams = [0,1],[2,3],[6,7]
for ib,a,b in zip(used_beams,ax.flatten(), [30,330,210]):
    a.set_title(b) 
 
    g_unique, r_unique, s_unique = [ ],[],[]
    for r,g,s in zip(R,beam_geom,shots):
        if len(g_unique)==0 or np.abs((g[:,ib]!= g_unique[-1])).sum()> 0:
            if len(g_unique)==0 or  np.abs((abs(g[:,ib]- g_unique[-1]) > .1)[(g_unique[-1] > 0)&(g[:,ib] > 0)]).sum()> 5:
                g_unique.append(g[:,ib])
                r_unique.append(r[:,ib])
                s_unique.append(s)
            else:
                valid = np.all(g[:,ib] != 0,1)
                g_unique[-1][valid] = g[valid][:,ib]
                r_unique[-1][valid] = r[valid][:,ib]
    
    _, sind = np.unique(np.dstack(g_unique), return_index=True,axis=-1)
    sind = sort(sind)
    for iu in sind[::-1]:
        valid = np.all((g_unique[iu] > 0)&( r_unique[iu]<2.99),1)
        r = r_unique[iu][:,0][valid]
        sind = np.argsort(r)
        ratio = g_unique[iu][:,0][valid][sind]/g_unique[iu][:,1][valid][sind]
        a.plot(r[sind], ratio/np.median(ratio[r[sind] > 1.6]),'-o', label=s_unique[iu])
    
    a.legend()
    a.set_ylabel('$(G_L/G_R)/\mathrm{median}(G_L/G_R)$')
        
show()


 


                
embed()

        
#spocitat pomery analyticky
  #vykreslit pomery, verticalni, updatovat nulove       
        
        
 


