


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

for shot in range(150000, 160000 ):
    
    try:
        MDSconn.openTree('IONS', shot)

        Z = MDSconn.get(r'\ions::TOP.IMPDENS.CERFIT.ZIMP').data()
        MDSconn.closeTree('IONS', shot)
        if not np.all(Z == 6):
            print(shot, np.unique(Z))
        #nbi = nbi_info(MDSconn,shot,  load_beams, {})
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

    except Exception as e:
        #print(shot, e)
        continue
    
