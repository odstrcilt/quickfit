


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

#T33
mdsserver = 'atlas.gat.com'
import MDSplus
try:
    MDSconn = MDSplus.Connection(mdsserver)
except:
    mdsserver = 'localhost'

    MDSconn = MDSplus.Connection(mdsserver)

for shot in range(188000, 150000,-1 ):
    #shot = 178868
    try:
        MDSconn.openTree('IONS', shot)
        #embed()

        AMP = MDSconn.get(r'\ions::TOP.CER.CERFIT.TANGENTIAL.CHANNEL33:AMP').data()
        MDSconn.closeTree('IONS', shot)
        if len(AMP) > 10:
            print(shot, len(AMP))
        
    except:
        continue
    #break
 
