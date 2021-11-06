


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
#T33OMFIT['cer']['CER']['CERFIT']['TANGENTIAL']['CHANNEL33']['AMP']
import matplotlib.pylab as plt


from numpy import *
from matplotlib.pylab import *







my_data = genfromtxt('beams_corrections_int.txt', delimiter=';')
shot_int = my_data[:,0]
shot_int,ind = unique(shot_int,return_index=True, )
scale_int = my_data[ind,1:-1:3]
volt_int = my_data[ind,2:-1:3]
time = my_data[ind,3:-1:3]
ind=isfinite(scale_int[:,1])
scale_int = scale_int[ind]
volt_int = volt_int[ind]
time = time[ind]
shot_int = shot_int[ind]

load_beams = '30L','30R','210L','210R','330L','330R'
my_data = genfromtxt('perveance.txt' )
shot3 = my_data[:,0]
shot3,ind = unique(shot3,return_index=True, )
perveance = my_data[ind,1:]

shots = sort(list(set(shot_int)&set(shot3)))

ind1 = in1d(shot_int,shots)
ind3 = in1d(shot3,shots)

perveance = perveance[:,:2]-median(perveance[:,:2],0)[None]

ind_int =  (scale_int[ind1,1]>0)&isfinite(volt_int[ind1,1])&isfinite(volt_int[ind1,0])&(scale_int[ind1,1]<10)
ind1_  = np.zeros(len(scale_int[ind1]))
ind1_[shot_int[ind1] < 162200] = 1
ind2  = np.zeros(len(scale_int[ind1]))
ind2[(shot_int[ind1] > 162200)&(shot_int[ind1] < 168300)] = 1
ind3_  = np.zeros(len(scale_int[ind1]))
ind3_[(shot_int[ind1] > 168300)&(shot_int[ind1] < 177777)] = 1
ind4  = np.zeros(len(scale_int[ind1]))
ind4[(shot_int[ind1] > 177777)&(shot_int[ind1] < 181200)] = 1
ind5  = np.zeros(len(scale_int[ind1]))
ind5[(shot_int[ind1] > 181200) ] = 1

dV_int = volt_int[ind1,1]-volt_int[ind1,0]
dV_int-=median(dV_int[ind_int])


A_int = np.vstack((ind1_,ind2,ind3_,ind4,ind5, dV_int, dV_int**2, perveance[ind3].T)).T[ind_int]
c_int = linalg.lstsq(A_int,log(scale_int[ind1][ind_int,1]),rcond=None)[0]
scale_ = exp(-dot(A_int[:,4:],c_int[4:]))*scale_int[ind_int,1]
 
 
 
scale_corr = exp(-dot(A_int[:,:-2],c_int[:-2]))


plot(shots,exp(dot(A_int[:,:-2],c_int[:-2])),'.')
plot(shots,exp(dot(A_int ,c_int )),'.')
plot(shots,scale_int[ind1][ind_int,1],'.')


axvline(162000,c='k')
axvline(168615,c='k')
axvline(178000,c='k')
axvline(181111,c='k')

show()




f,ax=subplots(2,sharex=True)
ax[0].plot(shots,perveance[:,:2]-median(perveance[:,:2],0)[None],'o' )
ax[0].set_ylabel('perveance-median(perveance)')
ax[0].legend(loc='best')
ax[0].set_ylim(-1,1)
ax[1].plot(shot1, 1/scale[:,1],'.')
ax[1].set_ylabel('n30R/n30L ')
show()


#plot(shots,perveance,'o' )



my_data = genfromtxt('beams_corrections_impcon.txt', delimiter=';')
shot1 = my_data[:,0]
shot1,ind = unique(shot1,return_index=True, )
scale = my_data[ind,1:-1:3]
volt = my_data[ind,2:-1:3]
time = my_data[ind,3:-1:3]



my_data = genfromtxt('beams_correction_t01_30.txt')

shot2 = my_data[:,0]
shot2,ind = unique(shot2,return_index=True, )
scale2 = my_data[ind,1]



load_beams = '30L','30R','210L','210R','330L','330R'
my_data = genfromtxt('perveance.txt' )
shot3 = my_data[:,0]
shot3,ind = unique(shot3,return_index=True, )
perveance = my_data[ind,1:]


shots = sort(list(set(shot1)&set(shot2)&set(shot3)))

ind1 = in1d(shot1,shots)
ind2 = in1d(shot2,shots)
ind3 = in1d(shot3,shots)

f,ax=subplots(2,sharex=True)
ax[0].plot(shot2[ind2], 1/scale2[ind2] ,'.',label='T01 scale')
ax[0].plot(shot1[ind1], 1/scale[ind1][:,1],'.',label='profile scale')
ax[0].set_ylabel('n30R/n30L')
ax[0].legend(loc='best')
 
ax[1].plot(shot2[ind2], 1/scale2[ind2] -1/scale[ind1][:,1],'.')
ax[1].set_ylabel('difference (T01 scale-profile scale)')
show()


embed()









#T33
mdsserver = 'atlas.gat.com'
import MDSplus
try:
    MDSconn = MDSplus.Connection(mdsserver)
except:
    mdsserver = 'localhost'

    MDSconn = MDSplus.Connection(mdsserver)
my_data = np.genfromtxt('beams_corrections_impcon.txt', delimiter=';')
shot_int = np.int_(my_data[:,0])
for shot in shot_int[::-1]:
    try:
        #shot = 183231

        MDSconn.openTree('IONS', shot)
        #embed()

        beam = MDSconn.get(r'\ions::TOP.CER.CERAUTO.TANGENTIAL.CHANNEL01:BEAMID').data()
        btime = MDSconn.get(r'dim_of(\ions::TOP.CER.CERAUTO.TANGENTIAL.CHANNEL01:AMP)').data()
        nz = MDSconn.get(r'\ions::TOP.IMPDENS.CERAUTO:NZT1').data()
        ntime = MDSconn.get(r'dim_of(\ions::TOP.IMPDENS.CERAUTO:NZT1)').data()
        time = np.sort(list(set(ntime)&set(btime)))
        beam = beam[np.in1d(btime, time)]
        nz = nz[np.in1d(ntime, time)]
        
        ind_L = beam == b'030L' 
        ind_R = beam == b'030R' 
        
        
        from scipy.optimize import minimize
        def fun(s,y1,y2,x1,x2):
            x = np.r_[x1,x2]
            ind = np.argsort(x)
            y = np.r_[y1/np.sqrt(s),y2*np.sqrt(s)][ind]
            yy = np.diff(np.log(y))/np.diff(x)
            return np.nanmean((yy)**2)

        res = minimize(fun,[1],args=(nz[ind_L],nz[ind_R], time[ind_L],time[ind_R]))
        s=res.x[0]


        #plt.plot(time[ind_L],nz[ind_L])
        #plt.plot(time[ind_R],nz[ind_R]*s)
        #plt.show()
 
        MDSconn.closeTree('IONS', shot)
        
        print(shot, s)
        with open('beams_correction_t01_30.txt', "a") as file:
            file.write( str(shot)+ ' ' +  str(s)+'\n')
                #except:
        #if len(Adshot, len(AMP))
        
    except:
        continue
    #break
 
