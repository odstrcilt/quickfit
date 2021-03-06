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

#TOTAL CX CROSS SECTION DD FROM FIDASIM
en_grid_fidasim = np.linspace(0,1,1602)[1:]*0.5*1000.
coeff=[1.95542E-15, 1.95542E-15, 1.70492E-15, 1.55148E-15, 1.43887E-15, 1.34921E-15, 1.27444E-15, 1.21019E-15, 
 1.15381E-15, 1.10356E-15, 1.05825E-15, 1.01698E-15, 9.79119E-16, 9.44150E-16, 9.11676E-16, 8.81376E-16 ,
 8.52987E-16, 8.26289E-16, 8.01101E-16, 7.77266E-16, 7.54651E-16, 7.33141E-16, 7.12636E-16, 6.93050E-16 ,
 6.74304E-16, 6.56332E-16, 6.39073E-16, 6.22473E-16, 6.06485E-16, 5.91065E-16, 5.76174E-16, 5.61777E-16 ,
 5.47844E-16, 5.34345E-16, 5.21255E-16, 5.08549E-16, 4.96207E-16, 4.84209E-16, 4.72538E-16, 4.61176E-16 ,
 4.50109E-16, 4.39324E-16, 4.28808E-16, 4.18549E-16, 4.08538E-16, 3.98764E-16, 3.89220E-16, 3.79896E-16 ,
 3.70785E-16, 3.61881E-16, 3.53176E-16, 3.44666E-16, 3.36344E-16, 3.28206E-16, 3.20247E-16, 3.12463E-16 ,
 3.04849E-16, 2.97402E-16, 2.90118E-16, 2.82993E-16, 2.76025E-16, 2.69210E-16, 2.62545E-16, 2.56028E-16 ,
 2.49656E-16, 2.43427E-16, 2.37337E-16, 2.31384E-16, 2.25567E-16, 2.19883E-16, 2.14329E-16, 2.08903E-16 ,
 2.03603E-16, 1.98428E-16, 1.93374E-16, 1.88440E-16, 1.83623E-16, 1.78922E-16, 1.74334E-16, 1.69858E-16 ,
 1.65491E-16, 1.61231E-16, 1.57077E-16, 1.53026E-16, 1.49076E-16, 1.45226E-16, 1.41472E-16, 1.37814E-16 ,
 1.34250E-16, 1.30777E-16, 1.27393E-16, 1.24097E-16, 1.20886E-16, 1.17759E-16, 1.14714E-16, 1.11749E-16 ,
 1.08862E-16, 1.06052E-16, 1.03316E-16, 1.00653E-16, 9.80611E-17, 9.55385E-17, 9.30835E-17, 9.06946E-17 ,
 8.83700E-17, 8.61081E-17, 8.39074E-17, 8.17663E-17, 7.96832E-17, 7.76567E-17, 7.56852E-17, 7.37674E-17 ,
 7.19018E-17, 7.00870E-17, 6.83217E-17, 6.66046E-17, 6.49342E-17, 6.33095E-17, 6.17291E-17, 6.01918E-17 ,
 5.86964E-17, 5.72418E-17, 5.58269E-17, 5.44505E-17, 5.31115E-17, 5.18090E-17, 5.05419E-17, 4.93092E-17 ,
 4.81100E-17, 4.69432E-17, 4.58081E-17, 4.47036E-17, 4.36289E-17, 4.25832E-17, 4.15657E-17, 4.05755E-17 ,
 3.96119E-17, 3.86741E-17, 3.77614E-17, 3.68730E-17, 3.60083E-17, 3.51666E-17, 3.43472E-17, 3.35495E-17 ,
 3.27729E-17, 3.20168E-17, 3.12806E-17, 3.05637E-17, 2.98655E-17, 2.91856E-17, 2.85235E-17, 2.78785E-17 ,
 2.72503E-17, 2.66383E-17, 2.60422E-17, 2.54613E-17, 2.48954E-17, 2.43440E-17, 2.38066E-17, 2.32830E-17 ,
 2.27726E-17, 2.22752E-17, 2.17903E-17, 2.13176E-17, 2.08568E-17, 2.04076E-17, 1.99695E-17, 1.95424E-17 ,
 1.91259E-17, 1.87197E-17, 1.83235E-17, 1.79370E-17, 1.75601E-17, 1.71923E-17, 1.68336E-17, 1.64835E-17 ,
 1.61419E-17, 1.58086E-17, 1.54833E-17, 1.51659E-17, 1.48560E-17, 1.45535E-17, 1.42582E-17, 1.39700E-17 ,
 1.36885E-17, 1.34137E-17, 1.31453E-17, 1.28832E-17, 1.26273E-17, 1.23772E-17, 1.21330E-17, 1.18945E-17 ,
 1.16614E-17, 1.14337E-17, 1.12112E-17, 1.09938E-17, 1.07814E-17, 1.05737E-17, 1.03708E-17, 1.01724E-17 ,
 9.97852E-18, 9.78896E-18, 9.60363E-18, 9.42242E-18, 9.24524E-18, 9.07197E-18, 8.90253E-18, 8.73680E-18 ,
 8.57471E-18, 8.41616E-18, 8.26106E-18, 8.10933E-18, 7.96088E-18, 7.81562E-18, 7.67349E-18, 7.53441E-18 ,
 7.39830E-18, 7.26508E-18, 7.13469E-18, 7.00706E-18, 6.88212E-18, 6.75980E-18, 6.64005E-18, 6.52281E-18 ,
 6.40800E-18, 6.29558E-18, 6.18548E-18, 6.07765E-18, 5.97204E-18, 5.86860E-18, 5.76727E-18, 5.66801E-18 ,
 5.57076E-18, 5.47548E-18, 5.38212E-18, 5.29065E-18, 5.20100E-18, 5.11316E-18, 5.02706E-18, 4.94267E-18 ,
 4.85996E-18, 4.77888E-18, 4.69940E-18, 4.62148E-18, 4.54508E-18, 4.47018E-18, 4.39673E-18, 4.32471E-18 ,
 4.25408E-18, 4.18481E-18, 4.11688E-18, 4.05024E-18, 3.98489E-18, 3.92077E-18, 3.85788E-18, 3.79617E-18 ,
 3.73564E-18, 3.67624E-18, 3.61796E-18, 3.56077E-18, 3.50464E-18, 3.44956E-18, 3.39551E-18, 3.34245E-18 ,
 3.29038E-18, 3.23926E-18, 3.18908E-18, 3.13981E-18, 3.09145E-18, 3.04397E-18, 2.99734E-18, 2.95156E-18 ,
 2.90661E-18, 2.86247E-18, 2.81911E-18, 2.77653E-18, 2.73472E-18, 2.69364E-18, 2.65329E-18, 2.61366E-18 ,
 2.57473E-18, 2.53648E-18, 2.49890E-18, 2.46198E-18, 2.42570E-18, 2.39005E-18, 2.35503E-18, 2.32060E-18 ,
 2.28678E-18, 2.25353E-18, 2.22086E-18, 2.18874E-18, 2.15717E-18, 2.12614E-18, 2.09564E-18, 2.06565E-18 ,
 2.03617E-18, 2.00718E-18, 1.97869E-18, 1.95067E-18, 1.92311E-18, 1.89602E-18, 1.86938E-18, 1.84318E-18 ,
 1.81741E-18, 1.79207E-18, 1.76714E-18, 1.74263E-18, 1.71851E-18, 1.69479E-18, 1.67145E-18, 1.64849E-18 ,
 1.62591E-18, 1.60369E-18, 1.58183E-18, 1.56032E-18, 1.53915E-18, 1.51832E-18, 1.49783E-18, 1.47766E-18 ,
 1.45781E-18, 1.43828E-18, 1.41905E-18, 1.40013E-18, 1.38150E-18, 1.36317E-18, 1.34512E-18, 1.32736E-18 ,
 1.30987E-18, 1.29265E-18, 1.27570E-18, 1.25901E-18, 1.24258E-18, 1.22640E-18, 1.21047E-18, 1.19479E-18 ,
 1.17934E-18, 1.16413E-18, 1.14915E-18, 1.13439E-18, 1.11986E-18, 1.10555E-18, 1.09146E-18, 1.07757E-18 ,
 1.06390E-18, 1.05042E-18, 1.03715E-18, 1.02408E-18, 1.01120E-18, 9.98511E-19, 9.86010E-19, 9.73692E-19 ,
 9.61556E-19, 9.49598E-19, 9.37816E-19, 9.26205E-19, 9.14763E-19, 9.03488E-19, 8.92376E-19, 8.81425E-19 ,
 8.70632E-19, 8.59995E-19, 8.49510E-19, 8.39176E-19, 8.28990E-19, 8.18950E-19, 8.09052E-19, 7.99295E-19 ,
 7.89677E-19, 7.80194E-19, 7.70846E-19, 7.61629E-19, 7.52543E-19, 7.43583E-19, 7.34749E-19, 7.26038E-19 ,
 7.17449E-19, 7.08979E-19, 7.00627E-19, 6.92391E-19, 6.84268E-19, 6.76258E-19, 6.68357E-19, 6.60565E-19 ,
 6.52880E-19, 6.45300E-19, 6.37824E-19, 6.30449E-19, 6.23175E-19, 6.15999E-19, 6.08920E-19, 6.01937E-19 ,
 5.95048E-19, 5.88252E-19, 5.81546E-19, 5.74931E-19, 5.68404E-19, 5.61964E-19, 5.55610E-19, 5.49340E-19 ,
 5.43154E-19, 5.37049E-19, 5.31025E-19, 5.25080E-19, 5.19213E-19, 5.13424E-19, 5.07710E-19, 5.02071E-19 ,
 4.96505E-19, 4.91012E-19, 4.85590E-19, 4.80239E-19, 4.74957E-19, 4.69743E-19, 4.64596E-19, 4.59516E-19 ,
 4.54500E-19, 4.49549E-19, 4.44662E-19, 4.39836E-19, 4.35072E-19, 4.30369E-19, 4.25725E-19, 4.21141E-19 ,
 4.16614E-19, 4.12144E-19, 4.07731E-19, 4.03373E-19, 3.99070E-19, 3.94820E-19, 3.90624E-19, 3.86480E-19 ,
 3.82387E-19, 3.78346E-19, 3.74355E-19, 3.70413E-19, 3.66519E-19, 3.62674E-19, 3.58876E-19, 3.55125E-19 ,
 3.51420E-19, 3.47760E-19, 3.44145E-19, 3.40573E-19, 3.37046E-19, 3.33561E-19, 3.30119E-19, 3.26718E-19 ,
 3.23358E-19, 3.20039E-19, 3.16760E-19, 3.13520E-19, 3.10319E-19, 3.07156E-19, 3.04031E-19, 3.00944E-19 ,
 2.97893E-19, 2.94879E-19, 2.91900E-19, 2.88956E-19, 2.86048E-19, 2.83174E-19, 2.80333E-19, 2.77526E-19 ,
 2.74752E-19, 2.72010E-19, 2.69301E-19, 2.66623E-19, 2.63977E-19, 2.61361E-19, 2.58775E-19, 2.56220E-19 ,
 2.53694E-19, 2.51197E-19, 2.48729E-19, 2.46290E-19, 2.43879E-19, 2.41495E-19, 2.39139E-19, 2.36809E-19 ,
 2.34506E-19, 2.32230E-19, 2.29979E-19, 2.27754E-19, 2.25554E-19, 2.23379E-19, 2.21228E-19, 2.19102E-19 ,
 2.17000E-19, 2.14922E-19, 2.12866E-19, 2.10834E-19, 2.08825E-19, 2.06838E-19, 2.04873E-19, 2.02930E-19 ,
 2.01008E-19, 1.99108E-19, 1.97229E-19, 1.95371E-19, 1.93533E-19, 1.91716E-19, 1.89918E-19, 1.88141E-19 ,
 1.86382E-19, 1.84643E-19, 1.82923E-19, 1.81222E-19, 1.79540E-19, 1.77875E-19, 1.76229E-19, 1.74601E-19 ,
 1.72990E-19, 1.71396E-19, 1.69820E-19, 1.68261E-19, 1.66719E-19, 1.65193E-19, 1.63683E-19, 1.62190E-19 ,
 1.60712E-19, 1.59251E-19, 1.57805E-19, 1.56374E-19, 1.54958E-19, 1.53558E-19, 1.52172E-19, 1.50801E-19 ,
 1.49445E-19, 1.48103E-19, 1.46774E-19, 1.45460E-19, 1.44160E-19, 1.42873E-19, 1.41600E-19, 1.40340E-19 ,
 1.39093E-19, 1.37859E-19, 1.36638E-19, 1.35429E-19, 1.34234E-19, 1.33050E-19, 1.31879E-19, 1.30719E-19 ,
 1.29572E-19, 1.28437E-19, 1.27313E-19, 1.26200E-19, 1.25099E-19, 1.24010E-19, 1.22931E-19, 1.21864E-19 ,
 1.20807E-19, 1.19761E-19, 1.18725E-19, 1.17700E-19, 1.16686E-19, 1.15682E-19, 1.14687E-19, 1.13703E-19 ,
 1.12729E-19, 1.11764E-19, 1.10809E-19, 1.09864E-19, 1.08928E-19, 1.08002E-19, 1.07084E-19, 1.06176E-19 ,
 1.05277E-19, 1.04387E-19, 1.03505E-19, 1.02633E-19, 1.01769E-19, 1.00913E-19, 1.00066E-19, 9.92270E-20 ,
 9.83965E-20, 9.75740E-20, 9.67596E-20, 9.59532E-20, 9.51546E-20, 9.43638E-20, 9.35807E-20, 9.28052E-20 ,
 9.20373E-20, 9.12768E-20, 9.05236E-20, 8.97777E-20, 8.90390E-20, 8.83075E-20, 8.75830E-20, 8.68654E-20 ,
 8.61547E-20, 8.54508E-20, 8.47537E-20, 8.40632E-20, 8.33793E-20, 8.27019E-20, 8.20310E-20, 8.13664E-20 ,
 8.07082E-20, 8.00561E-20, 7.94103E-20, 7.87705E-20, 7.81368E-20, 7.75091E-20, 7.68872E-20, 7.62712E-20 ,
 7.56610E-20, 7.50565E-20, 7.44577E-20, 7.38644E-20, 7.32767E-20, 7.26944E-20, 7.21176E-20, 7.15461E-20 ,
 7.09800E-20, 7.04190E-20, 6.98633E-20, 6.93127E-20, 6.87672E-20, 6.82267E-20, 6.76912E-20, 6.71606E-20 ,
 6.66349E-20, 6.61140E-20, 6.55979E-20, 6.50865E-20, 6.45797E-20, 6.40776E-20, 6.35800E-20, 6.30870E-20 ,
 6.25985E-20, 6.21143E-20, 6.16346E-20, 6.11592E-20, 6.06881E-20, 6.02212E-20, 5.97586E-20, 5.93001E-20 ,
 5.88457E-20, 5.83954E-20, 5.79492E-20, 5.75069E-20, 5.70686E-20, 5.66342E-20, 5.62037E-20, 5.57770E-20 ,
 5.53541E-20, 5.49349E-20, 5.45195E-20, 5.41077E-20, 5.36996E-20, 5.32951E-20, 5.28942E-20, 5.24968E-20 ,
 5.21029E-20, 5.17124E-20, 5.13254E-20, 5.09418E-20, 5.05615E-20, 5.01846E-20, 4.98109E-20, 4.94405E-20 ,
 4.90733E-20, 4.87093E-20, 4.83485E-20, 4.79908E-20, 4.76363E-20, 4.72847E-20, 4.69363E-20, 4.65908E-20 ,
 4.62483E-20, 4.59087E-20, 4.55721E-20, 4.52384E-20, 4.49075E-20, 4.45795E-20, 4.42542E-20, 4.39318E-20 ,
 4.36121E-20, 4.32951E-20, 4.29808E-20, 4.26692E-20, 4.23602E-20, 4.20539E-20, 4.17502E-20, 4.14490E-20 ,
 4.11503E-20, 4.08542E-20, 4.05606E-20, 4.02695E-20, 3.99808E-20, 3.96945E-20, 3.94106E-20, 3.91291E-20 ,
 3.88500E-20, 3.85732E-20, 3.82987E-20, 3.80265E-20, 3.77565E-20, 3.74888E-20, 3.72234E-20, 3.69601E-20 ,
 3.66990E-20, 3.64401E-20, 3.61833E-20, 3.59286E-20, 3.56760E-20, 3.54255E-20, 3.51771E-20, 3.49307E-20 ,
 3.46863E-20, 3.44440E-20, 3.42036E-20, 3.39652E-20, 3.37287E-20, 3.34942E-20, 3.32615E-20, 3.30308E-20 ,
 3.28019E-20, 3.25749E-20, 3.23498E-20, 3.21264E-20, 3.19049E-20, 3.16851E-20, 3.14672E-20, 3.12509E-20 ,
 3.10365E-20, 3.08237E-20, 3.06127E-20, 3.04033E-20, 3.01957E-20, 2.99896E-20, 2.97853E-20, 2.95826E-20 ,
 2.93814E-20, 2.91819E-20, 2.89840E-20, 2.87876E-20, 2.85928E-20, 2.83996E-20, 2.82079E-20, 2.80177E-20 ,
 2.78290E-20, 2.76418E-20, 2.74560E-20, 2.72718E-20, 2.70889E-20, 2.69075E-20, 2.67276E-20, 2.65490E-20 ,
 2.63719E-20, 2.61961E-20, 2.60217E-20, 2.58487E-20, 2.56770E-20, 2.55066E-20, 2.53376E-20, 2.51699E-20 ,
 2.50034E-20, 2.48383E-20, 2.46745E-20, 2.45119E-20, 2.43506E-20, 2.41905E-20, 2.40316E-20, 2.38740E-20 ,
 2.37176E-20, 2.35624E-20, 2.34084E-20, 2.32555E-20, 2.31038E-20, 2.29533E-20, 2.28040E-20, 2.26557E-20 ,
 2.25086E-20, 2.23627E-20, 2.22178E-20, 2.20740E-20, 2.19313E-20, 2.17897E-20, 2.16492E-20, 2.15098E-20 ,
 2.13713E-20, 2.12340E-20, 2.10976E-20, 2.09623E-20, 2.08281E-20, 2.06948E-20, 2.05625E-20, 2.04312E-20 ,
 2.03009E-20, 2.01716E-20, 2.00432E-20, 1.99158E-20, 1.97893E-20, 1.96638E-20, 1.95392E-20, 1.94156E-20 ,
 1.92928E-20, 1.91710E-20, 1.90501E-20, 1.89300E-20, 1.88109E-20, 1.86926E-20, 1.85752E-20, 1.84587E-20 ,
 1.83430E-20, 1.82282E-20, 1.81142E-20, 1.80010E-20, 1.78887E-20, 1.77772E-20, 1.76665E-20, 1.75567E-20 ,
 1.74476E-20, 1.73393E-20, 1.72318E-20, 1.71251E-20, 1.70192E-20, 1.69140E-20, 1.68096E-20, 1.67059E-20 ,
 1.66030E-20, 1.65009E-20, 1.63995E-20, 1.62988E-20, 1.61988E-20, 1.60996E-20, 1.60010E-20, 1.59032E-20 ,
 1.58061E-20, 1.57096E-20, 1.56139E-20, 1.55188E-20, 1.54245E-20, 1.53308E-20, 1.52377E-20, 1.51454E-20 ,
 1.50536E-20, 1.49626E-20, 1.48721E-20, 1.47824E-20, 1.46932E-20, 1.46047E-20, 1.45168E-20, 1.44296E-20 ,
 1.43429E-20, 1.42569E-20, 1.41714E-20, 1.40866E-20, 1.40024E-20, 1.39187E-20, 1.38356E-20, 1.37532E-20 ,
 1.36712E-20, 1.35899E-20, 1.35091E-20, 1.34289E-20, 1.33493E-20, 1.32702E-20, 1.31917E-20, 1.31137E-20 ,
 1.30362E-20, 1.29593E-20, 1.28829E-20, 1.28071E-20, 1.27317E-20, 1.26569E-20, 1.25826E-20, 1.25088E-20 ,
 1.24355E-20, 1.23628E-20, 1.22905E-20, 1.22187E-20, 1.21474E-20, 1.20766E-20, 1.20063E-20, 1.19365E-20 ,
 1.18671E-20, 1.17982E-20, 1.17298E-20, 1.16619E-20, 1.15944E-20, 1.15273E-20, 1.14608E-20, 1.13946E-20 ,
 1.13290E-20, 1.12637E-20, 1.11989E-20, 1.11346E-20, 1.10706E-20, 1.10072E-20, 1.09441E-20, 1.08815E-20 ,
 1.08192E-20, 1.07574E-20, 1.06960E-20, 1.06351E-20, 1.05745E-20, 1.05143E-20, 1.04546E-20, 1.03952E-20 ,
 1.03362E-20, 1.02777E-20, 1.02195E-20, 1.01617E-20, 1.01043E-20, 1.00472E-20, 9.99056E-21, 9.93427E-21 ,
 9.87836E-21, 9.82282E-21, 9.76765E-21, 9.71283E-21, 9.65838E-21, 9.60429E-21, 9.55055E-21, 9.49717E-21 ,
 9.44414E-21, 9.39145E-21, 9.33911E-21, 9.28711E-21, 9.23545E-21, 9.18413E-21, 9.13314E-21, 9.08248E-21 ,
 9.03216E-21, 8.98216E-21, 8.93249E-21, 8.88314E-21, 8.83411E-21, 8.78539E-21, 8.73700E-21, 8.68891E-21 ,
 8.64114E-21, 8.59367E-21, 8.54651E-21, 8.49965E-21, 8.45310E-21, 8.40684E-21, 8.36089E-21, 8.31522E-21 ,
 8.26985E-21, 8.22477E-21, 8.17998E-21, 8.13547E-21, 8.09125E-21, 8.04732E-21, 8.00366E-21, 7.96028E-21 ,
 7.91717E-21, 7.87434E-21, 7.83178E-21, 7.78949E-21, 7.74747E-21, 7.70572E-21, 7.66423E-21, 7.62300E-21 ,
 7.58203E-21, 7.54132E-21, 7.50087E-21, 7.46067E-21, 7.42073E-21, 7.38104E-21, 7.34159E-21, 7.30239E-21 ,
 7.26344E-21, 7.22474E-21, 7.18627E-21, 7.14805E-21, 7.11006E-21, 7.07231E-21, 7.03480E-21, 6.99752E-21 ,
 6.96047E-21, 6.92365E-21, 6.88706E-21, 6.85070E-21, 6.81457E-21, 6.77865E-21, 6.74296E-21, 6.70749E-21 ,
 6.67224E-21, 6.63721E-21, 6.60239E-21, 6.56779E-21, 6.53340E-21, 6.49922E-21, 6.46526E-21, 6.43150E-21 ,
 6.39794E-21, 6.36460E-21, 6.33145E-21, 6.29851E-21, 6.26577E-21, 6.23323E-21, 6.20089E-21, 6.16875E-21 ,
 6.13680E-21, 6.10505E-21, 6.07349E-21, 6.04212E-21, 6.01094E-21, 5.97995E-21, 5.94915E-21, 5.91853E-21 ,
 5.88810E-21, 5.85786E-21, 5.82779E-21, 5.79791E-21, 5.76821E-21, 5.73868E-21, 5.70933E-21, 5.68016E-21 ,
 5.65117E-21, 5.62235E-21, 5.59370E-21, 5.56522E-21, 5.53691E-21, 5.50877E-21, 5.48080E-21, 5.45300E-21 ,
 5.42536E-21, 5.39789E-21, 5.37058E-21, 5.34343E-21, 5.31645E-21, 5.28962E-21, 5.26295E-21, 5.23644E-21 ,
 5.21009E-21, 5.18390E-21, 5.15785E-21, 5.13197E-21, 5.10623E-21, 5.08065E-21, 5.05521E-21, 5.02993E-21 ,
 5.00480E-21, 4.97981E-21, 4.95497E-21, 4.93027E-21, 4.90572E-21, 4.88131E-21, 4.85705E-21, 4.83293E-21 ,
 4.80895E-21, 4.78510E-21, 4.76140E-21, 4.73784E-21, 4.71441E-21, 4.69112E-21, 4.66796E-21, 4.64494E-21 ,
 4.62205E-21, 4.59929E-21, 4.57667E-21, 4.55417E-21, 4.53181E-21, 4.50957E-21, 4.48747E-21, 4.46549E-21 ,
 4.44363E-21, 4.42191E-21, 4.40030E-21, 4.37883E-21, 4.35747E-21, 4.33624E-21, 4.31513E-21, 4.29414E-21 ,
 4.27327E-21, 4.25251E-21, 4.23188E-21, 4.21137E-21, 4.19097E-21, 4.17069E-21, 4.15052E-21, 4.13047E-21 ,
 4.11053E-21, 4.09070E-21, 4.07099E-21, 4.05139E-21, 4.03190E-21, 4.01252E-21, 3.99325E-21, 3.97409E-21 ,
 3.95503E-21, 3.93609E-21, 3.91725E-21, 3.89851E-21, 3.87988E-21, 3.86136E-21, 3.84294E-21, 3.82462E-21 ,
 3.80641E-21, 3.78830E-21, 3.77029E-21, 3.75238E-21, 3.73457E-21, 3.71686E-21, 3.69924E-21, 3.68173E-21 ,
 3.66431E-21, 3.64699E-21, 3.62977E-21, 3.61264E-21, 3.59561E-21, 3.57867E-21, 3.56183E-21, 3.54508E-21 ,
 3.52842E-21, 3.51185E-21, 3.49538E-21, 3.47899E-21, 3.46270E-21, 3.44649E-21, 3.43038E-21, 3.41435E-21 ,
 3.39842E-21, 3.38256E-21, 3.36680E-21, 3.35112E-21, 3.33553E-21, 3.32002E-21, 3.30460E-21, 3.28926E-21 ,
 3.27401E-21, 3.25884E-21, 3.24375E-21, 3.22875E-21, 3.21382E-21, 3.19898E-21, 3.18422E-21, 3.16953E-21 ,
 3.15493E-21, 3.14041E-21, 3.12596E-21, 3.11159E-21, 3.09731E-21, 3.08309E-21, 3.06896E-21, 3.05490E-21 ,
 3.04091E-21, 3.02701E-21, 3.01317E-21, 2.99941E-21, 2.98573E-21, 2.97212E-21, 2.95858E-21, 2.94511E-21 ,
 2.93172E-21, 2.91839E-21, 2.90514E-21, 2.89196E-21, 2.87885E-21, 2.86581E-21, 2.85284E-21, 2.83994E-21 ,
 2.82711E-21, 2.81434E-21, 2.80164E-21, 2.78901E-21, 2.77645E-21, 2.76395E-21, 2.75152E-21, 2.73916E-21 ,
 2.72686E-21, 2.71463E-21, 2.70246E-21, 2.69035E-21, 2.67831E-21, 2.66633E-21, 2.65442E-21, 2.64256E-21 ,
 2.63077E-21, 2.61904E-21, 2.60738E-21, 2.59577E-21, 2.58423E-21, 2.57274E-21, 2.56132E-21, 2.54995E-21 ,
 2.53865E-21, 2.52740E-21, 2.51621E-21, 2.50508E-21, 2.49401E-21, 2.48299E-21, 2.47204E-21, 2.46114E-21 ,
 2.45029E-21, 2.43951E-21, 2.42877E-21, 2.41810E-21, 2.40748E-21, 2.39691E-21, 2.38640E-21, 2.37594E-21 ,
 2.36554E-21, 2.35519E-21, 2.34489E-21, 2.33465E-21, 2.32446E-21, 2.31432E-21, 2.30423E-21, 2.29420E-21 ,
 2.28421E-21, 2.27428E-21, 2.26440E-21, 2.25457E-21, 2.24479E-21, 2.23506E-21, 2.22538E-21, 2.21574E-21 ,
 2.20616E-21, 2.19663E-21, 2.18714E-21, 2.17770E-21, 2.16832E-21, 2.15897E-21, 2.14968E-21, 2.14043E-21 ,
 2.13123E-21, 2.12208E-21, 2.11297E-21, 2.10391E-21, 2.09489E-21, 2.08592E-21, 2.07700E-21, 2.06811E-21 ,
 2.05928E-21, 2.05049E-21, 2.04174E-21, 2.03304E-21, 2.02438E-21, 2.01576E-21, 2.00719E-21, 1.99866E-21 ,
 1.99017E-21, 1.98173E-21, 1.97332E-21, 1.96496E-21, 1.95664E-21, 1.94836E-21, 1.94013E-21, 1.93193E-21 ,
 1.92378E-21, 1.91566E-21, 1.90759E-21, 1.89955E-21, 1.89156E-21, 1.88361E-21, 1.87569E-21, 1.86781E-21 ,
 1.85998E-21, 1.85218E-21, 1.84442E-21, 1.83670E-21, 1.82901E-21, 1.82137E-21, 1.81376E-21, 1.80619E-21 ,
 1.79865E-21, 1.79116E-21, 1.78370E-21, 1.77627E-21, 1.76889E-21, 1.76154E-21, 1.75422E-21, 1.74694E-21 ,
 1.73970E-21, 1.73249E-21, 1.72532E-21, 1.71818E-21, 1.71108E-21, 1.70401E-21, 1.69697E-21, 1.68997E-21 ,
 1.68301E-21, 1.67607E-21, 1.66917E-21, 1.66231E-21, 1.65548E-21, 1.64868E-21, 1.64191E-21, 1.63518E-21 ,
 1.62848E-21, 1.62181E-21, 1.61517E-21, 1.60857E-21, 1.60199E-21, 1.59545E-21, 1.58894E-21, 1.58247E-21 ,
 1.57602E-21, 1.56960E-21, 1.56322E-21, 1.55686E-21, 1.55054E-21, 1.54424E-21, 1.53798E-21, 1.53174E-21 ,
 1.52554E-21, 1.51936E-21, 1.51322E-21, 1.50710E-21, 1.50101E-21, 1.49496E-21, 1.48893E-21, 1.48292E-21 ,
 1.47695E-21, 1.47101E-21, 1.46509E-21, 1.45920E-21, 1.45334E-21, 1.44751E-21, 1.44171E-21, 1.43593E-21 ,
 1.43018E-21, 1.42446E-21, 1.41876E-21, 1.41309E-21, 1.40745E-21, 1.40183E-21, 1.39624E-21, 1.39068E-21 ,
 1.38514E-21, 1.37963E-21, 1.37414E-21, 1.36868E-21, 1.36325E-21, 1.35784E-21, 1.35246E-21, 1.34710E-21 ,
 1.34177E-21, 1.33646E-21, 1.33117E-21, 1.32591E-21, 1.32068E-21, 1.31547E-21, 1.31028E-21, 1.30512E-21 ,
 1.29998E-21, 1.29487E-21, 1.28978E-21, 1.28471E-21, 1.27967E-21, 1.27465E-21, 1.26965E-21, 1.26467E-21 ,
 1.25972E-21, 1.25479E-21, 1.24989E-21, 1.24501E-21, 1.24014E-21, 1.23531E-21, 1.23049E-21, 1.22570E-21 ,
 1.22092E-21, 1.21617E-21, 1.21145E-21, 1.20674E-21, 1.20205E-21, 1.19739E-21, 1.19275E-21, 1.18813E-21 ,
 1.18353E-21, 1.17895E-21, 1.17439E-21, 1.16985E-21, 1.16534E-21, 1.16084E-21, 1.15636E-21, 1.15191E-21 ,
 1.14747E-21, 1.14306E-21, 1.13866E-21, 1.13429E-21, 1.12993E-21, 1.12560E-21, 1.12128E-21, 1.11698E-21 ,
 1.11271E-21, 1.10845E-21, 1.10421E-21, 1.09999E-21, 1.09579E-21, 1.09161E-21, 1.08744E-21, 1.08330E-21 ,
 1.07917E-21, 1.07507E-21, 1.07098E-21, 1.06691E-21, 1.06285E-21, 1.05882E-21, 1.05480E-21, 1.05081E-21 ,
 1.04683E-21, 1.04286E-21, 1.03892E-21, 1.03499E-21, 1.03108E-21, 1.02719E-21, 1.02331E-21, 1.01946E-21 ,
 1.01562E-21, 1.01179E-21, 1.00798E-21, 1.00419E-21, 1.00042E-21, 9.96665E-22, 9.92926E-22, 9.89202E-22 ,
 9.85496E-22, 9.81805E-22, 9.78131E-22, 9.74473E-22, 9.70831E-22, 9.67206E-22, 9.63596E-22, 9.60002E-22 ,
 9.56424E-22, 9.52861E-22, 9.49314E-22, 9.45783E-22, 9.42267E-22, 9.38767E-22, 9.35282E-22, 9.31812E-22 ,
 9.28357E-22, 9.24918E-22, 9.21493E-22, 9.18084E-22, 9.14689E-22, 9.11309E-22, 9.07944E-22, 9.04593E-22 ,
 9.01257E-22, 8.97936E-22, 8.94628E-22, 8.91336E-22, 8.88057E-22, 8.84793E-22, 8.81543E-22, 8.78307E-22 ,
 8.75085E-22, 8.71877E-22, 8.68682E-22, 8.65502E-22, 8.62335E-22, 8.59182E-22, 8.56042E-22, 8.52916E-22 ,
 8.49804E-22, 8.46705E-22, 8.43619E-22, 8.40546E-22, 8.37487E-22, 8.34440E-22, 8.31407E-22, 8.28387E-22 ,
 8.25380E-22, 8.22385E-22, 8.19403E-22, 8.16435E-22, 8.13478E-22, 8.10535E-22, 8.07604E-22, 8.04685E-22 ,
 8.01779E-22, 7.98885E-22, 7.96004E-22, 7.93134E-22, 7.90277E-22, 7.87433E-22, 7.84600E-22, 7.81779E-22 ,
 7.78970E-22, 7.76173E-22, 7.73388E-22, 7.70615E-22, 7.67853E-22, 7.65103E-22, 7.62365E-22, 7.59638E-22 ,
 7.56922E-22, 7.54218E-22, 7.51526E-22, 7.48845E-22, 7.46175E-22, 7.43516E-22, 7.40869E-22, 7.38232E-22 ,
 7.35607E-22, 7.32993E-22, 7.30389E-22, 7.27797E-22, 7.25215E-22, 7.22644E-22, 7.20084E-22, 7.17535E-22 ,
 7.14996E-22, 7.12468E-22, 7.09950E-22, 7.07443E-22, 7.04946E-22, 7.02460E-22, 6.99984E-22, 6.97518E-22 ,
 6.95062E-22, 6.92617E-22, 6.90182E-22, 6.87757E-22, 6.85342E-22, 6.82937E-22, 6.80542E-22, 6.78157E-22 ,
 6.75781E-22, 6.73416E-22, 6.71060E-22, 6.68714E-22, 6.66378E-22, 6.64051E-22, 6.61734E-22, 6.59426E-22 ,
 6.57128E-22, 6.54839E-22, 6.52560E-22, 6.50290E-22, 6.48029E-22, 6.45778E-22, 6.43536E-22, 6.41303E-22 ,
 6.39079E-22, 6.36864E-22, 6.34658E-22, 6.32462E-22, 6.30274E-22, 6.28095E-22, 6.25925E-22, 6.23764E-22 ,
 6.21612E-22, 6.19468E-22, 6.17333E-22, 6.15207E-22, 6.13089E-22, 6.10980E-22, 6.08880E-22, 6.06788E-22 ,
 6.04705E-22, 6.02630E-22, 6.00563E-22, 5.98505E-22, 5.96455E-22, 5.94413E-22, 5.92380E-22, 5.90355E-22 ,
 5.88338E-22, 5.86329E-22, 5.84328E-22, 5.82335E-22, 5.80351E-22, 5.78374E-22, 5.76405E-22, 5.74444E-22 ,
 5.72491E-22, 5.70546E-22, 5.68608E-22, 5.66678E-22, 5.64757E-22, 5.62842E-22, 5.60936E-22, 5.59037E-22 ,
 5.57145E-22, 5.55261E-22, 5.53385E-22, 5.51516E-22, 5.49654E-22, 5.47800E-22, 5.45954E-22, 5.44114E-22 ,
 5.42282E-22, 5.40458E-22, 5.38640E-22, 5.36830E-22, 5.35027E-22, 5.33231E-22, 5.31442E-22, 5.29660E-22 ,
 5.27885E-22, 5.26117E-22, 5.24357E-22, 5.22603E-22, 5.20856E-22, 5.19116E-22, 5.17383E-22, 5.15656E-22 ,
 5.13937E-22]  
E = [10., 50., 100., 200., 300., 500., 1000.]#energies(keV/amu)
#sigma = [7.95e-16 ,9.61e-17 ,1.54e-17 ,1.29e-18, 2.35e-19 ,2.20e-20, 6.45e-22 ]#total xsects.(cm2)
sigma = [7.95e-16 ,9.61e-17 ,1.54e-17 ,1.29e-18, 2.35e-19 ,2.20e-20, 6.45e-22 ]#total xsects.(cm2)

#sigma = [7.70e-16, 6.13e-17, 9.25e-18, 8.44e-19, 1.63e-19, 1.63e-20, 5.03e-22] #n=1 crossection

#plt.plot(en_grid_fidasim, coeff,label='FIDASIM')
#plt.plot(np.array(E), sigma,label='qcx#h0_ory#h1.dat')

#E2 = [40.,50.,60.,70.,80.,90.,100.,125.,150.,200.,300.,400.,500.,600.,700.,800.,900.,1000.]
#sigma2=[2.08E-16,1.06E-16,5.86E-17,3.46E-17,2.16E-17,1.39E-17,9.33E-18,3.86E-18,1.81E-18,5.14E-19,7.79E-20,1.91E-20,6.20E-21,2.42E-21,1.08E-21,5.34E-22,2.85E-22,1.62E-22]

#plt.title('total CX crossection')

#plt.plot(np.array(E2), sigma2,label='qcx#h0_gyt#h1.dat')
#plt.legend(loc='best')
#plt.xlabel('Energy [keV/amu]')
#plt.ylabel('CX cross-section [cm$^{2}$]')

#plt.show()
#Note about output errorbars:
#positive finite - OK
#positive infinite - show points but do not use in the fit
#negative finite - disabled in GUI, but it can be enabled
#negative infinite - Do not shown, do not use
#def beam_get_fractions(81.0):
    ## Einj  is in keV
    ## Fraction
    #j = np.array([1.0, 2.0, 3.0])

    ##mickey definition 
    ### Power Fraction stolen from orignial compute_impdens for consistency.
    #power_fractions = np.zeros((3,np.size(Einj)))
    #power_fractions[0] = (65.1 + 0.19 * Einj) / 100.0
    #power_fractions[1] = (-165.5 + 6.87 * Einj - 0.087 * Einj ** 2 + 0.00037 * Einj ** 3) / 100.0  # typo on fig!
    #power_fractions[2] = 1.0 - power_fractions.sum(0)

    #return power_fractions


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


import matplotlib.pylab as plt

Einj = np.linspace(45,85,1000)
plt.plot(Einj,beam_get_fractions(Einj, model='chuck')['pfracs'].T,'--' )
plt.gca().set_prop_cycle(None)

plt.plot(Einj,beam_get_fractions(Einj, model='NBIgroup')['pfracs'].T, ':')
plt.gca().set_prop_cycle(None)
#plt.show()

plt.plot(Einj,beam_get_fractions(Einj, model='mickey')['pfracs'].T, '-' )
plt.xlabel('Beam energy [keV]')
plt.ylabel('Power fraction')
plt.gca().set_prop_cycle(None)


plt.plot([],[],'--' ,label='Chuck')
plt.gca().set_prop_cycle(None)

plt.plot([],[], ':',label='NBIgroup' )
plt.gca().set_prop_cycle(None)

plt.plot([],[], '-' ,label='Mickey')

plt.legend(loc='best')

#plt.plot(Einj,beam_get_fractions(Einj).T ,'--')

plt.xlim(45,85)
plt.ylim(0,1)

plt.show()

 
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
        # extrapolate by the nearest values

        grid = fbeam_Zeff, fbeam_Ti, fbeam_ne, fbeam_E
        lZeff = np.clip(np.log(Zeff), 0, np.log(6))  # Zeff is just up to 4!!, it will extrapolated
        lTi = np.clip(np.log(Ti), *grid[1][[0, -1]])
        lne = np.clip(np.log(ne), *grid[2][[0, -1]])
        lE = np.clip(np.log(E), *grid[3][[0, -1]])
        return np.exp(interpn(grid, fbeam_qeff, (lZeff, lTi, lne, lE), 
                                        fill_value=None, bounds_error=False))

    def interp_qeff_therm(Zeff, Ti, ne):
        # extrapolate by the nearest values

        grid = fbeam_Zeff, fbeam_Ti, fbeam_ne
        lZeff = np.clip(np.log(Zeff), 1, 6)  # Zeff is just up to 4!!, it will extrapolate
        lTi = np.clip(np.log(Ti), *grid[1][[0, -1]])
        lne = np.clip(np.log(ne), *grid[2][[0, -1]])
        return np.exp(interpn(grid, fbeam_qeff, (lZeff, lTi, lne), 
                                        fill_value=None, bounds_error=False))

    if therm:
        return interp_qeff_therm
    else:
        return interp_qeff_beam


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
  
        
        
        
0.75, 1.1, 0.967  #mickey, impcon-line 
0.816, 1.153,1.023  #AUG,no Z dependence,  error in n2frac
0.8, 1.13,0.998     #AUG,no Z dependence,  error in beam stop coeff
0.814, 1.162,1.02  #AUG no Z ependence
0.854, 1.22,1.067  #AUG, Z depedent
0.818, 1.152, 0.975 #AUG, halo, Zdependent


A = np.array(
    [[0.814, 1.151, 0.9627], #ref 
[0.862, 1.248, 1.0754], #no halo 
[0.7797, 1.097, 0.936],  #no Zeff dep 
[0.797, 1.11, 0.933], #errro beam atten
[0.84, 1.189, 0.994], #error n=2frac
[0.818, 1.152, 0.975],# beam model
[0.829, 1.192, 0.925],#old CER
[0.755, 1.114, 0.97]])#old CER, old qeff, no Zeff, no halo  #old CER
A =  (A[1:]-A[0])/A[0]
#A = A.mean(1)
A[-1]*=-1
plt.rcdefaults()
fig, ax = plt.subplots()
# Example data
issues = ('n=2 Halo', 'Zeff dependence', 'Beam stop. mix', 'n=2 frac mix', 'Beam fractions model','orig. ADAS rates', 'total')
y_pos = np.arange(len(issues))
#performance = 3 + 10 * np.random.rand(len(people))
#error = np.random.rand(len(people))

ax.barh(y_pos , A.mean(1)*100,   align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(issues)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel(r'% change')
#ax.set_title('How fast do you want to go today?')
ax.set_xlim(-10,10)
ax.axvline(0,c='k')

plt.show()



#embed()
    
    
    
ab = 2.014
dens = 4e13
te = 1600
ti = 1600
erel = 81e3/ab
erel_spec = erel/np.array([1,2,3])
fC = np.linspace(0,1/6.,1000)
fD = 1-fC*6.  #deuterium concentration
Rmid = 1.8
nbi_model = 'NBIgroup'

frac = beam_get_fractions(erel*ab/1e3,nbi_model)['nfracs']
path = os.path.dirname(os.path.realpath(__file__))+'/openadas/' 

files_bms = [path+'bms93#h_h1.dat',path+'bms93#h_c6.dat' ]
files_bmp = [path+'bmp97#h_2_h1.dat',path+'bmp97#h_2_c6.dat' ]


#normalise to calculate ion particle fraction        
Zion = np.array([1,6])
ion_frac = np.array([fD,fC])
ion_frac /= ion_frac.sum(0)

#calculate zeff
zeff = np.dot(Zion**2, ion_frac)/np.dot(Zion, ion_frac)

#effective density, based on adas/fortran/adas3xx/adas304/c4spln.for
eff_dens = dens*zeff/Zion[:,None]

# beam stopping rate
bms = [read_adf21(f, erel_spec[:,None], edens, te) for f,edens in zip(files_bms, eff_dens)] # cm^3/s
#n=2 excitation rate
bmp = [read_adf21(f, erel_spec[:,None], edens, te) for f,edens in zip(files_bmp, eff_dens)] # cm^3/s

#The stopping coefficients depend on the particular mixture of impurity ions present in the plasma
#ion weights based on adas/fortran/adas3xx/adaslib/cxbms.for
weights = ion_frac*Zion[:,None]
weights/= weights.sum(0)
embed()

bms_mix = np.sum(np.array(bms)* weights[:,None],0)
bmp_mix = np.sum(np.array(bmp)* weights[:,None],0)

#print(bms[0].shape  )
# beam stopping rate
bms = [read_adf21(f, erel_spec[:,None], dens, te) for f,edens in zip(files_bms, eff_dens)] # cm^3/s
#n=2 excitation rate
bmp = [read_adf21(f, erel_spec[:,None], dens, te) for f,edens in zip(files_bmp, eff_dens)] # cm^3/s

#The stopping coefficients depend on the particular mixture of impurity ions present in the plasma
#ion weights based on adas/fortran/adas3xx/adaslib/cxbms.for
#print(bms[0]  )
#embed()
bms_mix2 = bms[0]*fD+bms[1]*fC*6
bmp_mix2 = bmp[0]*fD+bmp[1]*fC*6
import matplotlib.pylab as plt
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(fC*6, bms_mix[0],label='Using effective density')
#plt.gca().set_prop_cycle(None)
plt.plot(fC*6, bms_mix2[0],label='Linear combination')
plt.axvline(1/30.*6,c='k',ls='--')
plt.title('+2.5%')
plt.xlim(0,1)
plt.ylim(0,None)
plt.legend(loc='best')
plt.xlabel('Carbon concentration')
plt.ylabel('Beam stopping coefficient')
plt.subplot(122)
plt.title('+60%')
plt.plot(fC*6, bmp_mix[0])
plt.plot(fC*6, bmp_mix2[0])
plt.axvline(1/30.*6,c='k',ls='--')
plt.xlim(0,1)
plt.ylim(0,None)
plt.xlabel('Carbon concentration')
plt.ylabel('n = 2 fraction of beam neutrals')
plt.tight_layout()
plt.show()
#plt.close()
line = 'C VI 8-7 '
imp, Z, transition = line.strip().split(' ')

intrp1 = read_adf12_aug(path, line, beam_spec='D', therm=False, n_neut=1)
intrp2 = read_adf12_aug(path, line, beam_spec='D', therm=False, n_neut=2)
intrp1_th = read_adf12_aug(path, line, beam_spec='D', therm=True, n_neut=1)
intrp2_th = read_adf12_aug(path, line, beam_spec='D', therm=True, n_neut=2)

      
ne = dens

qeff1 = intrp1(zeff,te, dens, erel_spec[:,None]*ab)/(1e6 * 4.0 * np.pi)
qeff2 = intrp2(zeff,te, dens, erel_spec[:,None]*ab)/(1e6 * 4.0 * np.pi)
qeff1_th = intrp1_th(zeff,te, dens)/(1e6 * 4.0 * np.pi)
qeff2_th = intrp2_th(zeff,te, dens)/(1e6 * 4.0 * np.pi)

from scipy.constants import e,m_p

#erel = (0.5 * ab * m_p / e) * vrel ** 2/ab  # eV/amu,

v = np.sqrt(erel_spec[:,None]*ab/(0.5 * ab * m_p / e))*100#cm/s

#v = vrel  # cm/s
 
#ionisation rate of deuterium
SCDfile = path+'/scd96_h.dat'
Se = read_adf11(SCDfile,  1, te, ne )#cm**3 s**-1)

#valid = (Se > 1e-10) & (beam_prof_merged['rho'] < 1)
ionis_rate = (ne*Se)#[valid] #1/s
import scipy.constants as consts

#simple neurals transport model for tangential LOS!!
vth = np.sqrt((2*ti*consts.e)/(ab * consts.m_p))#m/s
halo_std = vth/ionis_rate*100 #width of neutral gaussian distribution
#30L beam shape B. Grierson  RSI 89 10D116 2018
R_wall = 2.35#m
nbi_width  = 10.36+(R_wall-Rmid)*100*np.tan(0.0123)
nbi_height = 24.68+(R_wall-Rmid)*100*np.tan(0.0358)
corr = np.hypot(nbi_height, halo_std) / nbi_height
 
#CX crossection for D beam ions in D plasma
#data from file qcx#3h0_ory#h1.dat
#E = [10., 50., 100., 200., 300., 500., 1000.]#energies(keV/amu)
#sigma = [7.95e-16 ,9.61e-17 ,1.54e-17 ,1.29e-18, 2.35e-19 ,2.20e-20, 6.45e-22 ]#total xsects.(cm2)
#sigma = [7.70e-16, 6.13e-17, 9.25e-18, 8.44e-19, 1.63e-19, 1.63e-20, 5.03e-22] #n=1 crossection


 
sigmaDD = np.exp(np.interp(np.log(erel_spec[:,None]/1e3), np.log(en_grid_fidasim),np.log(coeff)))#cm2
nD = (1-fC*6)*ne

#n=1 halo fraction, normalised to the total number of injected neutrals
f0halo1 = (sigmaDD*v)/(ionis_rate*corr)*nD
f0halo1_ = (sigmaDD*v)/(ionis_rate)*nD


#Layman alpha emission
PECfile = path+'/pec96#h_pju#h0.dat'
PEC = read_adf15(PECfile, 1, te,ne)#ph cm**3 s**-1)
A21 = 4.6986e+08#s^-1 einsten coefficient from https://physics.nist.gov/PhysRefData/ASD/lines_form.html

#n=2 halo fraction, magnitude consistent with Rachael's paper, normalised to the total number of injected neutrals
f0halo2 = f0halo1*(PEC*ne)/A21
f0halo2_ = f0halo1_*(PEC*ne)/A21
#plt.plot(f0halo2)
#plt.show()

#  Rachale used FIDASIM to calculate a spatial profile of these ions
#embed()
#nD_ = (1-(1.5-1)/30*6)*ne
qeff1_=read_adf12(path+'/qef93#h_c6.dat',5, erel_spec[:,None], ne+1e-6, ti, zeff*0+1)  # wrong Zeff dependence
qeff2_=read_adf12(path+'/qef97#h_en2_kvi#c6.dat',5, erel_spec[:,None], ne+1e-6, ti, zeff*0+1)

fC_= 1./30
fD_= 1-6*1./30
bmp_mix2 = bmp[0]*fD_+bmp[1]*fC_*6
qeff1_ /= 1e6 * 4.0 * np.pi
qeff2_ /= 1e6 * 4.0 * np.pi

#embed()

#f,axes=plt.subplots(3, sharey=True)
#for i,ax in enumerate(axes):
    #ax.plot(fC*6,qeff1[i],'r')
    #ax.plot(fC*6,qeff2[i]*bmp_mix,'r--')
    #ax.plot(fC*6,qeff2_th*f0halo2[i],'r:')
    #ax.plot(fC*6,qeff2_th*f0halo2_[i],'r:')

    #ax.plot(fC*6,qeff1_[i],'b')
    #ax.plot(fC*6,qeff2_[i]*bmp_mix2,'b--')



#f,axes=plt.subplots(3, sharey=True)
#for i,ax in enumerate(axes):
    #ax.plot(fC*6,qeff1[i],'r')
    #ax.plot(fC*6,qeff2[i]*bmp_mix,'r--')
    #ax.plot(fC*6,qeff2_th*f0halo2[i],'r:')
    #ax.plot(fC*6,qeff2_th*f0halo2_[i],'r:')

    #ax.plot(fC*6,qeff1_[i],'b')
    #ax.plot(fC*6,qeff2_[i]*bmp_mix2,'b--')

##embed()
plt.figure()
ax= plt.subplot(111)
ax.plot(fC*6,np.dot(qeff1.T,frac),'r-.')
ax.plot(fC*6,np.dot((qeff2*bmp_mix).T,frac),'r--')
ax.plot(fC*6,np.dot((qeff2_th*f0halo2).T,frac),'r:')
#ax.plot(fC*6,np.dot((qeff2_th*f0halo2_).T,frac),'r:')
ax.plot(fC*6,np.dot((qeff2_th*f0halo2).T,frac)+np.dot((qeff2*bmp_mix).T,frac)+np.dot(qeff1.T,frac),'r-')

ax.plot(fC*6,np.dot(qeff1_.T,frac),'b-.')
ax.plot(fC*6,np.dot((qeff2_*bmp_mix2).T,frac),'b--')
ax.plot(fC*6,np.dot(qeff1_.T,frac) + np.dot((qeff2_*bmp_mix2).T,frac),'b-')
ax.set_ylim(1e-17, 1e-15)
ax.set_xlabel(r'C concentration')
ax.set_xlim(0,1)
#plt.show()
#exit()











#embed()
   
ab = 2.014
dens = 4e13
#te = 1600
#ti = 1600
erel = np.linspace(0,1e5,1000)[1:]/ab
erel_spec = erel/np.array([1,2,3])[:,None]
fC =1/30
fD = 1-fC*6.  #deuterium concentration


frac = beam_get_fractions(erel*ab/1e3,nbi_model)['nfracs']
#path = os.path.dirname(os.path.realpath(__file__))+'/openadas/' 

#files_bms = [path+'bms93#h_h1.dat',path+'bms93#h_c6.dat' ]
#files_bmp = [path+'bmp97#h_2_h1.dat',path+'bmp97#h_2_c6.dat' ]


#normalise to calculate ion particle fraction        
Zion = np.array([1,6])
ion_frac = np.array([fD,fC])
ion_frac /= ion_frac.sum(0)

##calculate zeff
zeff = np.dot(Zion**2, ion_frac)/np.dot(Zion, ion_frac)

##effective density, based on adas/fortran/adas3xx/adas304/c4spln.for
eff_dens = dens*zeff/Zion[:,None]

## beam stopping rate
#bms = [read_adf21(f, erel, edens, te) for f,edens in zip(files_bms, eff_dens)] # cm^3/s
##n=2 excitation rate
bmp = [read_adf21(f, erel_spec, edens, te) for f,edens in zip(files_bmp, eff_dens)] # cm^3/s

##The stopping coefficients depend on the particular mixture of impurity ions present in the plasma
##ion weights based on adas/fortran/adas3xx/adaslib/cxbms.for
weights = ion_frac*Zion 
weights/= weights.sum(0)
#embed()
#bms_mix = np.sum(np.array(bms)* weights,0)
bmp_mix = np.sum(np.array(bmp)* weights[:,None,None],0)


## beam stopping rate
#bms = [read_adf21(f, erel, dens, te) for f,edens in zip(files_bms, eff_dens)] # cm^3/s
##n=2 excitation rate
bmp = [read_adf21(f, erel_spec, dens, te) for f,edens in zip(files_bmp, eff_dens)] # cm^3/s

#The stopping coefficients depend on the particular mixture of impurity ions present in the plasma
#ion weights based on adas/fortran/adas3xx/adaslib/cxbms.for
#print(bms[0]  )

#bms_mix2 = bms[0]*fD+bms[1]*fC*6
bmp_mix2 = bmp[0]*fD+bmp[1]*fC*6

#embed()

#import matplotlib.pylab as plt
#plt.figure(figsize=(8,4))
#plt.subplot(121)
#plt.plot(fC*6, bms_mix,label='Using effective density')
#plt.plot(fC*6, bms_mix2,label='Linear combination')
#plt.axvline(1/30.*6,c='k',ls='--')
#plt.title('+2.5%')
#plt.xlim(0,1)
#plt.ylim(0,None)
#plt.legend(loc='best')
#plt.xlabel('Carbon concentration')
#plt.ylabel('Beam stopping coefficient')
#plt.subplot(122)
#plt.title('+60%')
#plt.plot(fC*6, bmp_mix)
#plt.plot(fC*6, bmp_mix2)
#plt.axvline(1/30.*6,c='k',ls='--')
#plt.xlim(0,1)
#plt.ylim(0,None)
#plt.xlabel('Carbon concentration')
#plt.ylabel('n = 2 fraction of beam neutrals')
#plt.tight_layout()
##plt.show()
#plt.close()
line = 'C VI 8-7 '

intrp1 = read_adf12_aug(path, line, beam_spec='D', therm=False, n_neut=1)
intrp2 = read_adf12_aug(path, line, beam_spec='D', therm=False, n_neut=2)
intrp1_th = read_adf12_aug(path, line, beam_spec='D', therm=True, n_neut=1)
intrp2_th = read_adf12_aug(path, line, beam_spec='D', therm=True, n_neut=2)

      
ne = dens

qeff1 = intrp1(zeff,te, dens, erel_spec*ab)/(1e6 * 4.0 * np.pi)
qeff2 = intrp2(zeff,te, dens, erel_spec*ab)/(1e6 * 4.0 * np.pi)
qeff1_th = intrp1_th(zeff,te, dens)/(1e6 * 4.0 * np.pi)
qeff2_th = intrp2_th(zeff,te, dens)/(1e6 * 4.0 * np.pi)

from scipy.constants import e,m_p

#erel = (0.5 * ab * m_p / e) * vrel ** 2/ab  # eV/amu,

v = np.sqrt(erel_spec*ab/(0.5 * ab * m_p / e))*100#cm/s

#v = vrel  # cm/s
 
#ionisation rate of deuterium
SCDfile = path+'/scd96_h.dat'
Se = read_adf11(SCDfile,  1, te, ne )#cm**3 s**-1)

#valid = (Se > 1e-10) & (beam_prof_merged['rho'] < 1)
ionis_rate = (ne*Se)#[valid] #1/s
import scipy.constants as consts

#simple neurals transport model for tangential LOS!!
vth = np.sqrt((2*ti*consts.e)/(ab * consts.m_p))#m/s
halo_std = vth/ionis_rate*100 #width of neutral gaussian distribution
#30L beam shape B. Grierson  RSI 89 10D116 2018
R_wall = 2.35#m
nbi_width  = 10.36+(R_wall-Rmid)*100*np.tan(0.0123)
nbi_height = 24.68+(R_wall-Rmid)*100*np.tan(0.0358)
corr = np.hypot(nbi_height, halo_std) / nbi_height
 
#CX crossection for D beam ions in D plasma
#data from file qcx#3h0_ory#h1.dat
E = [10., 50., 100., 200., 300., 500., 1000.]#energies(keV/amu)
#sigma = [7.95e-16 ,9.61e-17 ,1.54e-17 ,1.29e-18, 2.35e-19 ,2.20e-20, 6.45e-22 ]#total xsects.(cm2)
sigma = [7.70e-16, 6.13e-17, 9.25e-18, 8.44e-19, 1.63e-19, 1.63e-20, 5.03e-22] #n=1 crossection
sigmaDD = np.exp(np.interp(np.log(erel_spec/1e3), np.log(en_grid_fidasim),np.log(coeff)))#cm2
nD = (1-fC*6)*ne

#n=1 halo fraction, normalised to the total number of injected neutrals
f0halo1 = (sigmaDD*v)/(ionis_rate*corr)*nD
f0halo1_ = (sigmaDD*v)/(ionis_rate)*nD


#Layman alpha emission
PECfile = path+'/pec96#h_pju#h0.dat'
PEC = read_adf15(PECfile, 1, te,ne)#ph cm**3 s**-1)
A21 = 4.6986e+08#s^-1 einsten coefficient from https://physics.nist.gov/PhysRefData/ASD/lines_form.html

#n=2 halo fraction, magnitude consistent with Rachael's paper, normalised to the total number of injected neutrals
f0halo2 = f0halo1*(PEC*ne)/A21
f0halo2_ = f0halo1_*(PEC*ne)/A21


plt.plot(erel*2/1e3, f0halo1_.T*frac.T)
plt.plot(erel*2/1e3, np.sum(f0halo1_*frac,0),'--')

plt.gca().set_prop_cycle(None)

#plt.plot(erel*2/1e3, f0halo2_.T*frac.T)
#plt.gca().set_prop_cycle(None)
plt.plot([],[],label=1)
plt.plot([],[],label=2)
plt.plot([],[],label=3)
plt.legend()
plt.xlim(45,90)
plt.ylim(0,2.5)
plt.xlabel('Beam Energy [keV]')
plt.ylabel('$n_{0,halo}/n_{0,beam}$')
plt.show()

embed()
#plt.plot(f0halo2)
#plt.show()

#  Rachale used FIDASIM to calculate a spatial profile of these ions
#nD = (1-(1.5-1)/30*6)*ne
#print(nD)
#exit()
qeff1_=read_adf12(path+'/qef93#h_c6.dat',5, erel_spec, zeff*0+ne+1e-6, ti, zeff*0+1)  # wrong Zeff dependence
qeff2_=read_adf12(path+'/qef97#h_en2_kvi#c6.dat',5, erel_spec, zeff*0+ne+1e-6, ti, zeff*0+1)

fC_= 1./30
fD_= 1-6*1./30
bmp_mix2 = bmp[0]*fD_+bmp[1]*fC_*6
qeff1_ /= 1e6 * 4.0 * np.pi
qeff2_ /= 1e6 * 4.0 * np.pi

#embed()

#f,axes=plt.subplots(3, sharey=True)
#for i,ax in enumerate(axes):
    #ax.plot(fC*6,qeff1[i],'r')
    #ax.plot(fC*6,qeff2[i]*bmp_mix,'r--')
    #ax.plot(fC*6,qeff2_th*f0halo2[i],'r:')
    #ax.plot(fC*6,qeff2_th*f0halo2_[i],'r:')

    #ax.plot(fC*6,qeff1_[i],'b')
    #ax.plot(fC*6,qeff2_[i]*bmp_mix2,'b--')



#f,axes=plt.subplots(3, sharey=True)
#for i,ax in enumerate(axes):
    #ax.plot(fC*6,qeff1[i],'r')
    #ax.plot(fC*6,qeff2[i]*bmp_mix,'r--')
    #ax.plot(fC*6,qeff2_th*f0halo2[i],'r:')
    #ax.plot(fC*6,qeff2_th*f0halo2_[i],'r:')

    #ax.plot(fC*6,qeff1_[i],'b')
    #ax.plot(fC*6,qeff2_[i]*bmp_mix2,'b--')

#embed()
plt.figure()
ax = plt.subplot(111)
ax.plot(erel*2/1e3,np.sum(qeff1*frac,0),'r-.')
ax.plot(erel*2/1e3,np.sum(qeff2*bmp_mix*frac,0),'r--')
ax.plot(erel*2/1e3,np.sum(qeff2_th*f0halo2*frac,0),'r:')
#ax.plot(erel*2/1e3,np.sum(qeff2_th*f0halo2_*frac,0),'r:')
ax.plot(erel*2/1e3,np.sum(qeff2_th*f0halo2*frac,0)+np.sum(qeff2*bmp_mix*frac,0)+np.sum(qeff1*frac,0),'r-')


#Nph = nc*nb*

#176096 81 56 0.50  1.13
#176102 81 57 0.49  1.10
#175552 81 59 0.58  1.11
#175473 81 59 0.54  1.06
#176043 81 64 0.63  1.03
#175534 81 68 0.63  0.95
#175520 81 74 0.73  0.87
#176817 81 78 0.79  0.85
#175738 81 81 0.84  0.85
E,Q = [56,57,59,59,64,68,74,78], [1.13,1.10, 1.11,1.06,1.03,0.95,0.87,  0.85  ]
ax.plot(E,8e-16/np.array(Q)*0.45,'kx')

ax.plot(erel*2/1e3,np.sum(qeff1_*frac,0),'b-.')
ax.plot(erel*2/1e3,np.sum(qeff2_*bmp_mix2*frac,0),'b--')
ax.plot(erel*2/1e3,np.sum(qeff1_*frac,0)+np.sum(qeff2_*bmp_mix2*frac,0),'b-')
ax.set_ylim(0e-17, 6e-16)
ax.set_xlim(40,90)
ax.axvline(81,c='k')
ax.set_ylabel(r'CER emission rate [$m^3s^{-1}$]')
ax.set_xlabel(r'Beam energy [keV]')

#plt.show()

















#dens = np.logspace(18, 20,1000)/1e6
dens = 5e19/1e6

te = np.logspace(0, 4)
#ti = te
#te = 500
ti = 500
erel = 81e3/ab
erel_spec = erel/np.array([1,2,3])
fC = 1/30.
fD = 1-fC*6.  #deuterium concentration



frac = beam_get_fractions(erel*ab/1e3,nbi_model)['nfracs']
path = os.path.dirname(os.path.realpath(__file__))+'/openadas/' 

files_bms = [path+'bms93#h_h1.dat',path+'bms93#h_c6.dat' ]
files_bmp = [path+'bmp97#h_2_h1.dat',path+'bmp97#h_2_c6.dat' ]


#normalise to calculate ion particle fraction        
Zion = np.array([1,6])
ion_frac = np.array([fD,fC])
ion_frac /= ion_frac.sum(0)

#calculate zeff
zeff = np.dot(Zion**2, ion_frac)/np.dot(Zion, ion_frac)

#effective density, based on adas/fortran/adas3xx/adas304/c4spln.for
eff_dens = dens*zeff/Zion[:,None]

# beam stopping rate
bms = [read_adf21(f, erel_spec[:,None], edens, te) for f,edens in zip(files_bms, eff_dens)] # cm^3/s
#n=2 excitation rate
bmp = [read_adf21(f, erel_spec[:,None], edens, te) for f,edens in zip(files_bmp, eff_dens)] # cm^3/s

#The stopping coefficients depend on the particular mixture of impurity ions present in the plasma
#ion weights based on adas/fortran/adas3xx/adaslib/cxbms.for
weights = ion_frac*Zion
weights/= weights.sum(0)
bms_mix = np.sum(np.array(bms).T* weights,-1).T
bmp_mix = np.sum(np.array(bmp).T* weights,-1).T

#print(bms[0].shape  )
# beam stopping rate
bms = [read_adf21(f, erel_spec[:,None], dens, te) for f,edens in zip(files_bms, eff_dens)] # cm^3/s
#n=2 excitation rate
bmp = [read_adf21(f, erel_spec[:,None], dens, te) for f,edens in zip(files_bmp, eff_dens)] # cm^3/s
#embed()

#The stopping coefficients depend on the particular mixture of impurity ions present in the plasma
#ion weights based on adas/fortran/adas3xx/adaslib/cxbms.for
#print(bms[0]  )
#embed()
bms_mix2 = bms[0]*fD+bms[1]*fC*6
bmp_mix2 = bmp[0]*fD+bmp[1]*fC*6
import matplotlib.pylab as plt
#plt.figure(figsize=(8,4))
#plt.subplot(121)
#plt.plot(fC*6, bms_mix.T,label='Using effective density')
#plt.plot(fC*6, bms_mix2.T,label='Linear combination')
#plt.axvline(1/30.*6,c='k',ls='--')
#plt.title('+2.5%')
#plt.xlim(0,1)
#plt.ylim(0,None)
#plt.legend(loc='best')
#plt.xlabel('Carbon concentration')
#plt.ylabel('Beam stopping coefficient')
#plt.subplot(122)
#plt.title('+60%')
#plt.plot(fC*6, bmp_mix.T)
#plt.plot(fC*6, bmp_mix2.T)
#plt.axvline(1/30.*6,c='k',ls='--')
#plt.xlim(0,1)
#plt.ylim(0,None)
#plt.xlabel('Carbon concentration')
#plt.ylabel('n = 2 fraction of beam neutrals')
#plt.tight_layout()
#plt.show()
#plt.close()
line = 'C VI 8-7 '
imp, Z, transition = line.strip().split(' ')

intrp1 = read_adf12_aug(path, line, beam_spec='D', therm=False, n_neut=1)
intrp2 = read_adf12_aug(path, line, beam_spec='D', therm=False, n_neut=2)
intrp1_th = read_adf12_aug(path, line, beam_spec='D', therm=True, n_neut=1)
intrp2_th = read_adf12_aug(path, line, beam_spec='D', therm=True, n_neut=2)

      
ne = dens

qeff1 = intrp1(zeff,te, dens, erel_spec[:,None]*ab)/(1e6 * 4.0 * np.pi)
qeff2 = intrp2(zeff,te, dens, erel_spec[:,None]*ab)/(1e6 * 4.0 * np.pi)
qeff1_th = intrp1_th(zeff,te, dens)/(1e6 * 4.0 * np.pi)
qeff2_th = intrp2_th(zeff,te, dens)/(1e6 * 4.0 * np.pi)

from scipy.constants import e,m_p

#erel = (0.5 * ab * m_p / e) * vrel ** 2/ab  # eV/amu,

v = np.sqrt(erel_spec[:,None]*ab/(0.5 * ab * m_p / e))*100#cm/s

#v = vrel  # cm/s
 
#ionisation rate of deuterium
SCDfile = path+'/scd96_h.dat'
Se = read_adf11(SCDfile,  1, te, ne )#cm**3 s**-1)

#valid = (Se > 1e-10) & (beam_prof_merged['rho'] < 1)
ionis_rate = (ne*Se)#[valid] #1/s
import scipy.constants as consts
#embed()
#plt.figure()
#plt.plot(te, ionis_rate.T)
#plt.show()

#simple neurals transport model for tangential LOS!!
vth = np.sqrt((2*ti*consts.e)/(ab * consts.m_p))#m/s
halo_std = vth/ionis_rate*100 #width of neutral gaussian distribution
#30L beam shape B. Grierson  RSI 89 10D116 2018
R_wall = 2.35#m
nbi_width  = 10.36+(R_wall-Rmid)*100*np.tan(0.0123)
nbi_height = 24.68+(R_wall-Rmid)*100*np.tan(0.0358)
corr = np.hypot(nbi_height, halo_std) / nbi_height
 
#CX crossection for D beam ions in D plasma
#data from file qcx#3h0_ory#h1.dat
E = [10., 50., 100., 200., 300., 500., 1000.]#energies(keV/amu)
#sigma = [7.95e-16 ,9.61e-17 ,1.54e-17 ,1.29e-18, 2.35e-19 ,2.20e-20, 6.45e-22 ]#total xsects.(cm2)
sigma = [7.70e-16, 6.13e-17, 9.25e-18, 8.44e-19, 1.63e-19, 1.63e-20, 5.03e-22] #n=1 crossection
sigmaDD = np.exp(np.interp(np.log(erel_spec[:,None]/1e3), np.log(en_grid_fidasim),np.log(coeff)))#cm2
nD = (1-fC*6)*ne
#embed()

#n=1 halo fraction, normalised to the total number of injected neutrals
f0halo1 = (sigmaDD*v)/(ionis_rate*corr)*nD
f0halo1_ = (sigmaDD*v)/(ionis_rate)*nD


#Layman alpha emission
PECfile = path+'/pec96#h_pju#h0.dat'
PEC = read_adf15(PECfile, 1, te,ne)#ph cm**3 s**-1)
A21 = 4.6986e+08#s^-1 einsten coefficient from https://physics.nist.gov/PhysRefData/ASD/lines_form.html

#n=2 halo fraction, magnitude consistent with Rachael's paper, normalised to the total number of injected neutrals
f0halo2 = f0halo1*(PEC*ne)/A21
f0halo2_ = f0halo1_*(PEC*ne)/A21
#plt.plot(f0halo2)
#plt.show()

#  Rachale used FIDASIM to calculate a spatial profile of these ions
#embed()
#nD_ = (1-(1.5-1)/30*6)*ne
qeff1_=read_adf12(path+'/qef93#h_c6.dat',5, erel_spec[:,None], ne+1e-6, ti, zeff*0+1)  # wrong Zeff dependence
qeff2_=read_adf12(path+'/qef97#h_en2_kvi#c6.dat',5, erel_spec[:,None], ne+1e-6, ti, zeff*0+1)

fC_= 1./30
fD_= 1-6*1./30
bmp_mix2 = bmp[0]*fD_+bmp[1]*fC_*6
qeff1_ /= 1e6 * 4.0 * np.pi
qeff2_ /= 1e6 * 4.0 * np.pi
plt.figure()
#embed()
plt.plot(ne/1e13, 100*np.dot(frac.T,f0halo2).T/ np.dot(frac.T,f0halo1).T,label='thermal diffusion')
plt.plot(ne/1e13,  100*np.dot(frac.T,f0halo2).T/np.dot(frac.T,f0halo1_).T,label='no neutrals transport')
plt.xlabel('$n_e [10^{19}\mathrm{m}^{-3}]$')
plt.ylabel('$n_\mathrm{0,n=1,halo}/n_\mathrm{beam}$')
plt.ylabel('$n_\mathrm{0,n=2,halo}/n_\mathrm{0,n=1,halo}$')

#plt.xlabel('$T_e [keV]$')
plt.xlim(0,10)
plt.ylim(0,.8)
plt.legend(loc='lower right')
plt.show()
#embed()

#f,axes=plt.subplots(3, sharey=True)
#for i,ax in enumerate(axes):
    #ax.plot(fC*6,qeff1[i],'r')
    #ax.plot(fC*6,qeff2[i]*bmp_mix,'r--')
    #ax.plot(fC*6,qeff2_th*f0halo2[i],'r:')
    #ax.plot(fC*6,qeff2_th*f0halo2_[i],'r:')

    #ax.plot(fC*6,qeff1_[i],'b')
    #ax.plot(fC*6,qeff2_[i]*bmp_mix2,'b--')



#f,axes=plt.subplots(3, sharey=True)
#for i,ax in enumerate(axes):
    #ax.plot(fC*6,qeff1[i],'r')
    #ax.plot(fC*6,qeff2[i]*bmp_mix,'r--')
    #ax.plot(fC*6,qeff2_th*f0halo2[i],'r:')
    #ax.plot(fC*6,qeff2_th*f0halo2_[i],'r:')

    #ax.plot(fC*6,qeff1_[i],'b')
    #ax.plot(fC*6,qeff2_[i]*bmp_mix2,'b--')

##embed()
plt.figure()
ax= plt.subplot(111)
#embed()
x = te
x = dens*1e6/1e19
ax.plot(x,np.dot(qeff1.T,frac),'r-.')
ax.plot(x,np.dot((qeff2*bmp_mix).T,frac),'r--')
ax.plot(x,np.dot((qeff2_th*f0halo2).T,frac),'r:')
#ax.plot(x,np.dot((qeff2_th*f0halo2_).T,frac),'r:')
ax.plot(x,np.dot((qeff2_th*f0halo2).T,frac)+np.dot((qeff2*bmp_mix).T,frac)+np.dot(qeff1.T,frac),'r-')

ax.plot(x,np.dot(qeff1_.T,frac),'b-.')
ax.plot(x,np.dot((qeff2_*bmp_mix2).T,frac),'b--')
ax.semilogx(x,np.dot(qeff1_.T,frac) + np.dot((qeff2_*bmp_mix2).T,frac),'b-')
ax.set_ylim(1e-17, 1e-15)
ax.set_xlabel(r'$n_e\ [10^{19}\cdot m^{-3}]$')

plt.show()



exit()





















#3/30*6


#embed()
     
#exit()
    
           

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

    #ind =  cross(boundary[:-1,:], boundary[1:,:],x)>0

    
    inlayers = np.all(cross(boundary[:-1,:], boundary[1:,:],x) > 0,axis=-1)

    return inlayers

def split_mds_data(data, bytelens, bit_size):
    n = 0
    if np.size(bytelens) == np.size(data) == 0:
        return []

    #split signals by the channel lenghths
    try:
        #embed()
        #embed()
        if (np.sum(bytelens)-len(data)*bit_size)%len(bytelens):
            raise Exception('Something went wrong!')

        

        bit_offset = (np.sum(bytelens)-len(data)*bit_size)//len(bytelens)
        data_lens = (np.int_(bytelens)-int(bit_offset))//bit_size
        #sind = np.cumsum(data_lens)[:-1]
        
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
    #return np.split(out[i], sind) binary_dilation
    #for i in range(3):
        #out[i] = np.split(out[i], sind) 
    #t_cer,stime,beam_cer,lineid = out
 
        
#def mds_load_par(MDSconn, TDI, tree, shot,  numMdsTasks=8):
   
    ##use multiprocessing to speed up loading 
    #TDI = np.array_split(TDI, numMdsTasks)

    #server = MDSconn.hostspec
    #args = [(MDSconn,tree, shot, tdi) for tdi in TDI]
    #pool = ThreadPool()
    #out = []
 
    #for o in pool.map(mds_load,args):
        #out.extend(o)
    ##out = np.sum(pool.map(mds_load,args))
    #pool.close()
    #pool.join()
    
    #return out 


 
        
 
 
def detect_elms(tvec, signal,threshold=10,min_elm_dist=5e-4, min_elm_len=5e-4):
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
    ind = binary_opening(ind ,[1,1,1,1,1])
     
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

    val = np.ones_like(elm_start)
    elm_val = np.c_[val, -val,val*0 ].flatten()
    t_elm_val = tvec[np.c_[ elm_start-1, elm_start, elm_end].flatten()]
    
    #plt.plot(t_elm_val, elm_val*100)

    #plt.axhline(threshold_sig)
    #plt.show()
    
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
            line_id = MDSconn.get('['+','.join(TDI_lineid)+']').data()
                
            MDSconn.closeTree('IONS', shot)
            try:
                line_id = [l.decode() for l in line_id]
            except:
                pass
            ulines = np.unique(line_id)
            for l in ulines:
                #embed()
                tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', l)
                imp, Z = tmp.group(1), tmp.group(2)
                imps.append(imp+str(roman2int(Z) ))

        except:
            imps = ['C6']
            ##pass
        #print(  time()-t)
        #t = time()

        #try:

            #MDSconn.openTree('IONS', shot)
            #ZIMP,nZ = np.unique(np.int_(MDSconn.get('\\IONS::CERQZIMP').data()), return_counts=True)
            #if len(nZ) > 1:
                #ZIMP = ZIMP[np.argsort(-nZ)]
            #MDSconn.closeTree('IONS', shot)

        #except:
            #pass
        #print(  time()-t)
        
        #embed()

        
    #build a large dictionary with all settings
    default_settings = OrderedDict()

    default_settings['Ti']= {\
        'systems':{'CER system':(['tangential',True], ['vertical',False])},\
        'load_options':{'CER system':{'Analysis':('best', ('best','fit','auto','quick'))} }}
        
    default_settings['omega']= {\
        'systems':{'CER system':(['tangential',True], )},
        'load_options':{'CER system':{'Analysis':('best', ('best','fit','auto','quick'))}}}

   
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
                                ( 'CO2 interf.',(['fit CO2 data',False],['rescale TS',False])) ) ),
        'load_options':{'TS system':{"TS revision":('BLESSED',['BLESSED']+ts_revisions)},
                        'Reflectometer':{'Position error':{'Align with TS':True}, }}}                        
        
         
    nimp = {\
        'systems':{'CER system':(['tangential',True], ['vertical',True])},
        'load_options':{'CER system':OrderedDict((
                                ('Analysis', ('best', ('best','fit','auto','quick'))),
                                ('Correction',{'Relative calibration':True, 'nz from CER intensity':False}  )))   }}
    #,'Remove first point after blip':False
    #if there are multiple impurities
    for imp in imps:
        default_settings['n'+imp] = deepcopy(nimp)



    default_settings['Zeff']= {\
    'systems':OrderedDict(( ( 'CER system',(['tangential',False],['vertical',False])),
                            ( 'VB array',  (['tangential',True],                 )),
                            ( 'CER VB',    (['tangential',True],['vertical',False])),
                            )), \
    'load_options':{'VB array':{'Corrections':{'radiative mantle':True,'rescale by CO2':False,'remove NBI CX':False}},\
                    'CER VB':{'Analysis':('best', ('best','fit','auto','quick'))},
                    'CER system':OrderedDict((
                            ('Analysis', ('best', ('best','fit','auto','quick'))),
                            ('Correction',    {'Relative calibration':True,  'nz from CER intensity':False}),
                            ('TS position error',{'Z shift [cm]':0.0})))
                    }\
        }
    
    default_settings['Mach']= {\
        'systems':{'CER system':[]},
        'load_options':{'CER system':{'Analysis':('best', ('best','fit','auto','quick'))}}}
    default_settings['Te/Ti']= {\
        'systems':{'CER system':(['tangential',True], ['vertical',False] )},
        'load_options':{'CER system':{'Analysis':('best', ('best','fit','auto','quick'))}}}         
        
        
    if len(imps) > 1:
        default_settings['Zeff']['load_options']['CER system']['Impurity'] = ('C6',imps)
 
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
            
            if isinstance(diag[sys], list):
                for ch,ind in zip(diag[sys],I):
                    ch['rho'].values  = rho[ind,0]
            else:
                diag[sys]['rho'].values  =  rho 
        
        diag['EQM'] = {'id':id(self.eqm),'dr':np.mean(dr), 'dz':np.mean(dz),'ed':self.eqm.diag}
 


        return diag
            
            
        
    def __call__(self,  quantity=[], options=None,zipfit=False, tbeg=0, tend=10 ):
   
        if zipfit:
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
              
     
        data = []
        ts = None
        

        if quantity == 'ne' and options['systems']['CO2 interf.'][0][1].get():
            data.append(self.load_co2(tbeg, tend))

        if quantity in ['Te', 'ne']  and len(systems) > 0:
            ts = self.load_ts(tbeg, tend, systems, options['load_options']['TS system'])
            if quantity == 'ne' and options['systems']['CO2 interf.'][1][1].get():
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

                vtor = omg*r
                vtor_err = omg_err*r
                ti[ti<=0] = 1 #avoid zero division
                vtor[vtor==0] = 1 #avoid zero division
                mach = np.sqrt(2*m_u/e*vtor**2/(2*ti))
                mach_err = mach*np.hypot(vtor_err/vtor,ti_err/ti/2.)
                #deuterium mach number 
                ch['Mach'] = xarray.DataArray(mach, dims=['time'], attrs={'units':'-','label':'M_D'})
                ch['Mach_err'] = xarray.DataArray(mach_err, dims=['time'])
                Mach['tangential'][ich] = ch.drop(['omega','omega_err','Ti','Ti_err'])
            
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
                    Te = interp(np.vstack((ch['time'].values, ch['rho'].values)).T)
                    interp.values[:] = np.copy(err_Te)#[:,None]
                    Te_err = interp(np.vstack((ch['time'].values, ch['rho'].values)).T)
                    Ti = ch['Ti'].values
                    Ti_err = ch['Ti_err'].values
                    if 'omega' in ch: ch = ch.drop(['omega','omega_err'])
                    Te_Ti[sys][ich] = ch.drop(['Ti','Ti_err'])
                    Te_Ti[sys][ich]['Te/Ti'] = xarray.DataArray(Te/(Ti+1),dims=['time'], attrs={'units':'-','label':'T_e/T_i'})
                    Te_Ti[sys][ich]['Te/Ti_err'] = xarray.DataArray(Te/(Ti+1)*np.hypot(Te_err/(Te+1),Ti_err/(Ti+1)),
                                                               dims=['time'])

            data.append(Te_Ti)
 
     
     
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
            output['data'][i]= output['data'][i].sel(time=slice(tbeg,tend))

        if len(output['diag_names']) == 0 or len(output['data'])==0:
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
        #print('timestep = %.2fms'%(output['tres']*1e3))
        #embed()
        t = time()
        diag,count = np.unique(np.hstack([o['diags'].values.flatten() for o in output['data']]), return_counts=True)
        for d,c in zip(diag, count):
            print(d,c)
        #print(time()-t)
        print(len(output['diag_names']), len(diag))
        output['diag_names'] = diag

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


        zipfit = {}

        print_line('  * Fetching ZIPFIT ... ')
        T = time()

        for prof, (scale_fact,node, tree) in loading_dict.items():
            try:
                data = {}
                
                path = '::TOP.PROFILE_FITS.ZIPFIT.'
                self.MDSconn.openTree(tree, self.shot)
                ds = xarray.Dataset( )
                ds['data'] = xarray.DataArray(self.MDSconn.get('_x=\\'+tree+path+node+'FIT').data()*scale_fact, dims=['time','rho'])
                ds['err'] = xarray.DataArray(abs(self.MDSconn.get('error_of(_x)').data())*scale_fact, dims=['time','rho'])
                ds['rho']  = xarray.DataArray(self.MDSconn.get('dim_of(_x,0)').data(), dims=['rho'])
                ds['tvec'] = xarray.DataArray(self.MDSconn.get('dim_of(_x,1)').data()/1000, dims=['tvec'])

                zipfit[prof] = ds
            except Exception as e:
                printe( 'MDS error: '+ str(e))
            finally:
                self.MDSconn.closeTree(tree, self.shot)
        
            
        print('\t done in %.1fs'%(time()-T))
        



        try:
            tvec = np.sort(list((set(zipfit['omega']['tvec'].values)&set(zipfit['Ti']['tvec'].values))))
            ind_ti  = np.in1d( zipfit['Ti']['tvec'].values,tvec) 
            ind_omg = np.in1d( zipfit['omega']['tvec'].values,tvec) 

            from scipy.constants import e,m_u
            rho = zipfit['Ti']['rho'].values
            Rho,Tvec = np.meshgrid(rho,tvec)
            R = self.eqm.rhoTheta2rz(Rho,0, t_in=tvec,coord_in='rho_tor')[0][:,0]
            
            vtor = zipfit['omega']['data'][ind_omg].values*R
            vtor_err = zipfit['omega']['err'][ind_omg].values*R
            ti = zipfit['Ti']['data'][ind_ti].values
            ti_err = zipfit['Ti']['err'][ind_ti].values
            ti[ti<=0] = 1 #avoid zero division
            zipfit['Mach'] = xarray.Dataset( )
            zipfit['Mach']['data'] = xarray.DataArray(np.sqrt(2*m_u/e*vtor**2/(2*ti)), dims=['time','rho'])
            zipfit['Mach']['err'] = xarray.DataArray(zipfit['Mach']['data'].values*np.hypot(vtor_err/vtor,ti_err/ti/2), dims=['time','rho'])
            zipfit['Mach']['rho']  = xarray.DataArray(rho, dims=['rho'])
            zipfit['Mach']['tvec'] = xarray.DataArray(np.array(tvec), dims=['tvec'])
        except:
            pass
             

        try:
            tvec = zipfit['Ti']['tvec'].values
            ind_t = (tvec >= zipfit['Te']['tvec'].values[0])&(tvec<= zipfit['Te']['tvec'].values[-1])
            #rho is the same for  both zipfits
            Te     = interp1d(zipfit['Te']['tvec'].values,zipfit['Te']['data'].values,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            Te_err = interp1d(zipfit['Te']['tvec'].values,zipfit['Te']['err'].values ,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            Ti     = np.maximum(zipfit['Ti']['data'].values[ind_t],.1) #prevent zero division
            Te     = np.maximum(Te,1e-2) #prevent zero division

            Ti_err = zipfit['Ti']['err'].values[ind_t]
            
            zipfit['Te/Ti'] = xarray.Dataset( )
            zipfit['Te/Ti']['data'] = xarray.DataArray(Te/Ti, dims=['time','rho'])
            zipfit['Te/Ti']['err'] = xarray.DataArray(Te/Ti*np.hypot(Ti_err/Ti,Te_err/Te), dims=['time','rho'])
            zipfit['Te/Ti']['rho']  = xarray.DataArray(zipfit['Ti']['rho'].values, dims=['rho'])
            zipfit['Te/Ti']['tvec'] = xarray.DataArray(tvec[ind_t], dims=['time'])
            
        except:
            pass
        
        try:
            tvec = zipfit['nimp']['tvec'].values
            ind_t = (tvec >= zipfit['ne']['tvec'].values[0])&(tvec<= zipfit['ne']['tvec'].values[-1])
            #rho is the same for  both zipfits
            ne     = interp1d(zipfit['ne']['tvec'].values,zipfit['ne']['data'].values,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            ne_err = interp1d(zipfit['ne']['tvec'].values,zipfit['ne']['err'].values ,axis=0,
                                assume_sorted=True,copy=False)(tvec[ind_t])
            nimp     =  zipfit['nimp']['data'].values[ind_t] 
            nimp_err = zipfit['nimp']['err'].values[ind_t]
            
            ne     = np.maximum(ne,1) #prevent zero division
            nimp   = np.maximum(nimp,1) #prevent zero division

            # NOTE suppose the impruity ion in ZIPFITprofiles is always carbon and bulk ions are D
            Zimp = 6 
            Zmain = 1
            
            zipfit['Zeff'] = xarray.Dataset( )
            zipfit['Zeff']['data'] = xarray.DataArray(Zimp*(Zimp - Zmain)*nimp/ne + Zmain, dims=['time','rho'])
            zipfit['Zeff']['err'] = xarray.DataArray((zipfit['Zeff']['data'].values - Zmain)*np.hypot(ne_err/ne,nimp_err/nimp), dims=['time','rho'])
            zipfit['Zeff']['rho']  = xarray.DataArray(zipfit['nimp']['rho'].values, dims=['rho']) #rho toroidal
            zipfit['Zeff']['tvec'] = xarray.DataArray(tvec[ind_t], dims=['time'])
            
        except:
            pass
        
        self.RAW['ZIPFIT'] = zipfit


        return zipfit
    
    
    
    def get_cer_types(self,analysis_type,impurity=False):
         
        path = '.IMPDENS.CER%s:TIME' if impurity else '.CER.CER%s:DATE_LOADED'
        tree = 'IONS'
        analysis_types= 'fit','auto','quick','neural'
        
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
                printe( 'MDS error: '+ str(e))
            finally:
                self.MDSconn.closeTree(tree, self.shot)
            self.cer_analysis_best[impurity] = analysis_type
 
        return 'cer'+analysis_type
    
 
    def nbi_info(self,  load_beams, nbi):
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

        pinj_scal = np.array(pinj_scal)[fired]
        _load_beams = np.array(_load_beams)[fired]
        paths = np.array(paths)[fired]
        
        TDI  = ['\\NB::TOP:TIMEBASE']
        TDI += [p+'PINJ_'+p[-4:-1] for p in paths]
        
        
        pow_data = self.MDSconn.get('['+','.join(TDI)+']').data()
        tvec, pow_data = pow_data[0]/1e3, pow_data[1:]

        if self.shot  > 169568:  #load time dependent voltage
            TDI = [p+'VBEAM' for p in paths]
        else:   #load scalar values
            TDI = [p+'NBVAC_SCALAR' for p in paths]
        volt_data = self.MDSconn.get('['+','.join(TDI)+']').data()

        self.MDSconn.closeTree('NB', self.shot)
        
        
        b21sign= 1 if self.shot < 124700 else -1 #BUG?? 210 always assumed to be counter current
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
            beam['pow_frac'] = beam_get_fractions(beam['volts']/1e3)
            beam['Rtang'] = Rtang[b[:2]+b[-1]]*0.01
            beam['power_timetrace'] = pow_data[i]
            beam['power_time'] = tvec
      
        return nbi
        
        
    def load_nimp_intens(self, nimp,load_systems, analysis_type,imp):

        #calculate impurity density directly from the measured intensity
        
        if len(load_systems) == 0:
            return nimp
     
        tree = 'IONS'        

 
        ##############################  LOAD DATA ######################################## 

        print_line( '  * Fetching CER INTENSITY from %s ...'%analysis_type.upper())
        TT = time()


        self.MDSconn.openTree(tree, self.shot)

        loaded_chan = []
        TDI_data = ''
        TDI_lineid = ''
        TDI_beam_geo = ''
        TDI_phi = ''

        bytelens = []

        #prepare list of loaded channels
        for system in load_systems:
            nimp[system] = []
            nimp['diag_names'][system] = []
            path = 'CER.%s.%s.CHANNEL*'%(analysis_type,system)
            #embed()
            nodes = self.MDSconn.get('getnci("'+path+'","fullpath")').data()
            lengths = self.MDSconn.get('getnci("'+path+':TIME","LENGTH")').data()
            
            for node,length in zip(nodes, lengths):
                if length == 0: continue
                try:
                    node = node.decode()
                except:
                    pass

                node = node.strip()
                loaded_chan.append(system[0].upper()+node[-2:])
                
                signals = ['TIME','STIME','R','Z','INTENSITY','INTENSITYERR','TTSUB','TTSUB_STIME']
                bytelens += [length]*len(signals)

                TDI_data += ','.join([node+':'+sig for sig in signals])+','
                
                calib_node = '\\IONS::TOP.CER.CALIBRATION.%s.%s'%(system,node.split('.')[-1])
 
                TDI_lineid += calib_node+':LINEID'+','
                TDI_beam_geo += calib_node+':BEAMGEOMETRY'+','
                TDI_phi += calib_node+':LENS_PHI'+','

        
        if len(loaded_chan) == 0:
            raise Exception('Error: no data! try a different edition?')

        
        #fast fetch of MDS+ data
        order='\\IONS::TOP.CER.CALIBRATION.BEAM_ORDER'
        TDI = ['['+TDI_data[:-1]+']','['+TDI_lineid[:-1]+']',
               '['+TDI_beam_geo[:-1]+']', '['+TDI_phi[:-1]+']',order]
        data,line_id,beam_geom,phi,beam_order = mds_load(self.MDSconn,TDI, tree, self.shot)
        data = np.single(data)
        try:
            line_id = np.array([l.decode() for l in line_id])
            beam_order = [b.decode() for b in beam_order]
        except:
            pass
        
        #select only  LOS observing selected impurity
        #lines_dict = {'He2':'He II 4-3','B5':'B V 7-6','C6':'C VI 8-7','N7':'N VII 9-8','Ne10':'Ne X 11-10'}
        #Zimp = int(re.sub("[^0-9]", "", imp))
        imp_name = re.sub("\d+", "", imp)
        selected_imp = np.array([l[:2].strip() == imp_name for l in line_id])
        #embed()

        selected_imp &= np.any(beam_geom > 0,1) #rarely some channel has all zeros!
        if not any(selected_imp):
            if sum([len(nimp[sys]) for sys in nimp['systems']]):
                #some data were loaded before, nothing in the actually loaded system
                return nimp
            raise Exception('No '+imp+' data in '+analysis_type.upper(),'edition. ', 'Availible are :'+','.join(np.unique(line_id)))
      
        #split in signals
        splitted_data = split_mds_data(data, bytelens, 8)        
        splitted_data = np.reshape(splitted_data,(-1, len(signals)))[selected_imp].T
        beam_geom = beam_geom[selected_imp]
        phi = phi[selected_imp]

        loaded_chan = np.array(loaded_chan)[selected_imp]
        line_id = line_id[selected_imp]
        tvec, stime, R, Z, INT, INT_ERR,TTSUB,TTSUB_ST = splitted_data
        tvec  =  [t/1e3 for t in tvec]
        TTSUB =  [t/1e3 for t in TTSUB]
        stime =  [t/1e3 for t in stime]

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
        
        NBI = self.RAW['nimp'].setdefault('NBI',{})
        self.nbi_info(load_beams,NBI)
        fired = [NBI[b]['fired'] for b in load_beams]
        load_beams = load_beams[fired]
        beam_geom = beam_geom[:,beam_ind][:,fired]


        beam_time = NBI[load_beams[0]]['power_time']
        PINJ = np.array([NBI[b]['power_timetrace'] for b in load_beams])

        nbi_cum_pow = cumtrapz(np.double(PINJ),beam_time,initial=0)
        nbi_cum_pow_int = interp1d(beam_time, nbi_cum_pow,assume_sorted=True,bounds_error=False,fill_value=0)
        #I hope that it will not mess anything
        utime, utime_ind = np.unique(T_all, return_index=True)
        ustime = stime_all[utime_ind]
        nbi_pow = np.single(nbi_cum_pow_int(utime+ustime)-nbi_cum_pow_int(utime))/ustime

        #when background substraction was used
        if any(TT_all) > 0:
            valid = TT_all > 0
            TT_s_all = np.hstack(TTSUB_ST)/1e3
            utime_sub, utime_sub_ind = np.unique(TT_all[valid], return_index=True)
            ustime_sub = TT_s_all[valid][utime_sub_ind]
            nbi_pow_sub = (nbi_cum_pow_int(utime_sub+ustime_sub)-nbi_cum_pow_int(utime_sub))/(ustime_sub+1e-6)
        else:
            nbi_pow_sub = None
        
        

        
        ########### create xarray dataset with the results ################
        n = 0
        beam_intervals = {}
        for ich,ch in enumerate(loaded_chan):
            diag = 'vertical' if ch[0] == 'V' else 'tangential'
            nt = len(tvec[ich])
    
            # List of chords with intensity calibration errors for FY15, FY16 shots after
            # CER upgraded with new fibers and cameras.
            disableChanVert = 'V03', 'V04', 'V05', 'V06', 'V23', 'V24'
            if 162163 <= self.shot <= 167627 and ch in disableChanVert:
                INT_ERR[ich][:] = np.infty
            #apply correctin for some channels
            if ch == 'T07' and self.shot >= 158695:
                INT[ich] *= 1.05
            if ch == 'T23' and  158695 <= self.shot < 169546:
                INT[ich] *= 1.05
            if ch == 'T23' and  165322 <= self.shot < 169546:
                INT[ich] *= 1.05
            if ch in ['T09','T11','T13','T41','T43','T45'] and 168000<self.shot<172799:
                INT[ich] /= 1.12

            ##############  find beam ID of each timeslice

            observed_beams = beam_geom[ich] > 0
            beams = load_beams[observed_beams]
            
            #mean background substracted NBI power over CER integration time 
            beam_pow = nbi_pow[observed_beams][:,utime.searchsorted(tvec[ich])]
            #when background substraction was applied
            if nbi_pow_sub is not None and any(TTSUB[ich] > 0):
                beam_pow[:,TTSUB[ich] > 0] -= nbi_pow_sub[observed_beams][:,utime_sub.searchsorted(TTSUB[ich][TTSUB[ich] > 0])]

            beamid = np.zeros(nt, dtype='U4')
            #how each beam contributes to this channel
            beam_frac = np.ones((len(beams),nt),dtype='single')
            if len(beams) == 0:
                raise Exception('Something wrong with loaded beam times?')
            elif len(beams) == 1:
                beamid[:] = beams
            elif len(beams) == 2: #L and R beam
                beamid[:] = beams[0][:-1]+'B'
                beamid[beam_pow[0] > 1e5] = beams[0]
                beamid[beam_pow[1] > 1e5] = beams[1]
                
                beam_frac = beam_pow*beam_geom[ich, observed_beams][:,None]
                beam_frac /= beam_frac.sum(0)+1e-6 #prevent zero division for channel V6
                beamid[beam_frac.max(0) < 0.9] = beams[0][:-1]+'B'
            else:
                raise Exception('Observed 3 beams at once??')


            tmp = re.search('([A-Z][a-z]*) *([A-Z]*) *([0-9]*[a-z]*-[0-9]*[a-z]*)', line_id[ich])
            imp, charge, transition = tmp.group(1), tmp.group(2), tmp.group(3) 
            charge = roman2int(charge)     
            
            edge = False
            #if diag == 'tangential' and int(phi[ich]) == 318: #split beam tangential 30 system to the core and midradius subsystems
                #s = 'c'
            if diag == 'vertical' and  int(phi[ich]) == 331: #split vertical 33L in the core and edge
                edge = True

            if diag == 'tangential' and  int(phi[ich]) == 346:
                edge = True  
                
            s = 'e' if edge else ''
            
            #print(  ch, phi[ich],R[ich].mean() )

            names = np.array([diag[0].upper()+'_'+ID.lstrip('0')+s+' '+imp+str(charge) for ID in beamid])
            unames,idx,inv_idx = np.unique(names,return_inverse=True,return_index=True)
            for name in unames:
                if not name in nimp['diag_names'][diag]:
                    nimp['diag_names'][diag].append(name)

            #split channels by beams
            for ID in np.unique(inv_idx):
                beam_ind = (inv_idx == ID)&(beam_pow.sum(0) > 1e5) #channel V06 has measurements with zero beam power but finite intensity
                if not any(beam_ind):
                    continue
                #save data when the beam was turned on
                bname = beamid[idx[ID]]
                beam_intervals.setdefault(bname,[])
                beam_intervals[bname].append((tvec[ich][beam_ind],stime[ich][beam_ind]))

                ds = xarray.Dataset(attrs={'channel':ch+'_'+bname, 'system': diag,'edge':edge,
                                           'beam_geom':beam_geom[ich, observed_beams],'Z':charge})
                #fill be zeros for now, 
                ds['nimp'] = xarray.DataArray(0*tvec[ich][beam_ind], dims=['time'], 
                                        attrs={'units':'m^{-3}','label':'n_{%s}^{%d+}'%(imp,charge),'Z':charge, 'impurity':imp})
                ds['nimp_err']  = xarray.DataArray(0*tvec[ich][beam_ind]-np.inf,dims=['time'])
                ds['int'] = xarray.DataArray(INT[ich][beam_ind], dims=['time'], 
                                        attrs={'units':'ph / sr m^{3}','line':line_id[ich]})
                ds['int_err']  = xarray.DataArray(INT_ERR[ich][beam_ind],dims=['time'])
                ds['R'] = xarray.DataArray(R[ich][beam_ind],dims=['time'], attrs={'units':'m'})
                ds['Z'] = xarray.DataArray(Z[ich][beam_ind],dims=['time'], attrs={'units':'m'})
                ds['rho'] = xarray.DataArray(rho_all[n:n+nt][beam_ind],dims=['time'], attrs={'units':'-'})
                ds['diags']= xarray.DataArray(names[beam_ind],dims=['time'])
                ds['time'] = xarray.DataArray((tvec[ich]+stime[ich]/2)[beam_ind],dims=['time'], attrs={'units':'s'})
                ds['stime'] = xarray.DataArray(stime[ich][beam_ind],dims=['time'], attrs={'units':'s'})
                ds['beam_pow'] = xarray.DataArray(beam_pow[:,beam_ind],dims=['beams','time'], attrs={'units':'W'})
                ds['beam_frac'] = xarray.DataArray(beam_frac[:,beam_ind],dims=['beams','time'], attrs={'units':'W'})
                ds['beams'] = xarray.DataArray(beams,dims=['beams'])
                nimp[diag].append(ds)
            n += nt

        #how long was each beam turned on
        for n,b in beam_intervals.items():
            NBI.setdefault(n,{})
            beam_times, stime = np.hstack(b)
            ubeam_times,ind = np.unique(beam_times,return_index=True)
            NBI[n]['beam_fire_time'] = np.sum(stime[ind])
            
        nimp['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}
        nimp['loaded_beams'] = np.unique(np.hstack((load_beams,nimp.get('loaded_beams',[]))))
        
        print('\t done in %.1fs'%(time()-TT))
        
        return nimp

    def calc_nimp_intens(self,tbeg,tend, nimp,systems,imp, nC_guess=None):
        #extract data from all channels
        nimp_data = {'time':[],  'int':[], 'int_err':[], 'R':[]}
        beam_data = {'beam_pow':{},'beam_geom': {}}
        data_index = []
        
        
        
        if imp not in ['B5','C6','He2','Ne10','N7']:
            raise Exception('CX cross-sections are not availible for '+imp)
        if imp != 'C6':
            print('Warning: %s impurity is assumed to be a trace'%imp) #not affecting beam attenuation or CX cross-section
       
        
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
     

        #merge the data
        for k in nimp_data.keys():
            nimp_data[k] = np.hstack(nimp_data[k])
        for k in beam_data.keys():
            beam_data[k] = np.vstack([np.hstack(beam_data[k][b]) for b in nimp['loaded_beams']])
        
      
        line_id = ch['int'].attrs['line']  #NOTE assume that all data are from the same line
                
        
        NBI = self.RAW['nimp']['NBI']
        nbeam = len(nimp['loaded_beams'])  

        nbi_dict = {}
        for k in ['volts','pow_frac','Rtang']:
            nbi_dict[k] = []
            for b in nimp['loaded_beams']:
                nbi_dict[k].append(NBI[b][k])
            nbi_dict[k] = np.array(nbi_dict[k])
        

        ########################   Get kinetic profiles along the midradius  ##########################
        #split measurements in 100 clusters
        from scipy.cluster.vq import kmeans2
        centroid, label = kmeans2(nimp_data['time'], min(100,n//2),100, minit='points')
        #remove empty clusters
        ulabel = np.unique(label)
        uind = np.cumsum(np.in1d(range(len(centroid)),ulabel))-1
        centroid = centroid[ulabel]
        #sort clusters in time order
        sind = np.argsort(centroid)
        inv_sind = np.argsort(sind)
        label = inv_sind[uind[label]]
        centroid = centroid[sind]

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
                
                beam_profiles['rho'][sys].append(rho_slice)  
                ne_slice = np.exp(np.average(np.log(n_e[tslice, ch_valid]+1.),0, TS_valid[tslice, ch_valid]))
                
                beam_profiles['ne'][sys].append(ne_slice)
                ne_err_slice = 1/np.average(1/n_e_err[tslice, ch_valid],0, TS_valid[tslice, ch_valid])
                beam_profiles['ne_err'][sys].append(np.maximum(ne_err_slice, .05*ne_slice)) #minimum 5% error
                Te_slice = np.exp(np.average(np.log(T_e[tslice, ch_valid]+1.),0, TS_valid[tslice, ch_valid]))
                beam_profiles['te'][sys].append(Te_slice)
        
        #merge data from all systems 
        for k, d in beam_profiles.items():
            merged_sys = []
            for i,t in enumerate(centroid):
                data = np.hstack([d[sys][i] for sys in TS['systems']])
                merged_sys.append(data) 
            beam_profiles[k] = merged_sys
        

        #create radial midplane grid based on location of TS measurements
        Rmin = [nimp_data['R'][label == i].min() for i,t in enumerate(centroid)]
        
        #map midplane radius to rho
        R_midplane = np.r_[nimp_data['R'].min()-.1:2.0:0.01, 2.0:2.4:0.005]
        rho_midplane = self.eqm.rz2rho(R_midplane,R_midplane*0,centroid,self.rho_coord)
        ind_axis = np.argmin(rho_midplane,axis=1)
        Raxis = R_midplane[ind_axis]
 
        beam_profiles['Rmid'] = []
        for it, t in enumerate(centroid):   
            rho = beam_profiles['rho'][it]
            ind_axis = np.argmin(rho_midplane[it])
            #map from rho to Rmid for both LFS and HFS channel
            R_lfs = np.interp(rho,rho_midplane[it][ind_axis:],R_midplane[ind_axis:])
            R_hfs = np.interp(rho,rho_midplane[it][ind_axis::-1],R_midplane[ind_axis::-1])
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
  

        ########################   From CER     ##########################
        #fetch ion temperature and rotation 
        cer = self.load_cer(tbeg,tend, nimp['systems'])
        
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
                            #print('nimp_'+s+c,'nimp_'+s+c in ch )
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
            tind = label == it
            T = nimp_data['time'][tind]
            Trange = T.min()-1e-3,T.max()+1e-3
            rho = beam_profiles['rho'][it]

            for name, data in cer_data.items():
                mean_rho, mean_data = [],[]
                if len(data):
                    tt = np.hstack([d[1] for d in data])
                    tind = (tt >= Trange[0])&(tt <= Trange[1])
                    for r,t,d,e in data: #rho, time, data for each channel
                        tind, _tind = tind[len(t):],tind[:len(t)]
                        valid = np.isfinite(e)
                        if any(_tind&valid):
                            mean_rho.append( r[_tind&valid].mean())
                            mean_data.append(np.average(d[_tind&valid],0,1/e[_tind&valid]**2)) 
                         
                if len(mean_data) > 1:
                    sind = np.argsort(mean_rho)
                    prof_rho, prof_data = np.array(mean_rho)[sind], np.array(mean_data)[sind]
                    if name == 'fC':  #use carbon concetration - better properties for extrapolation!
                        #import matplotlib.pylab as plt
              
                        ne = np.interp(prof_rho, rho[::-1], beam_profiles['ne'][it][::-1])
                        prof_data = np.clip(prof_data/ne,0.01,1/6.-0.01)

                        #plt.plot(prof_rho, prof_data,'o-')
                        ##plt.plot(rho, beam_profiles['ne'][it])
                        #plt.plot(rho, np.interp(rho,prof_rho, prof_data))
                        #plt.show()

                    _data = np.interp(rho,prof_rho, prof_data)  #TODO add radial averaging?? 
                    if name == 'Ti':
                        #printmean_rho(_t, np.max(rho))
                        edge = rho > min(0.95, np.max(prof_rho))
                        _data[edge] = beam_profiles['te'][it][edge] #replace edge ion temperature by electron temperature
                else:
                    if name == 'Ti':
                        _data = beam_profiles['te'][it] #use Te if Ti is not availible
                    elif name == 'omega':
                        _data = 0* rho #assume zero if unknown
                    elif name == 'fC':
                        _data = rho*0+(2.-1.)/30. #guess Zeff = 2

                beam_profiles[name].append(_data)
 
        beam_prof_merged = {k:np.hstack(p) for k,p in beam_profiles.items()}
        
        #beam_prof_merged['fcarbon'] = np.clip(beam_prof_merged['nC']/beam_prof_merged['ne'],0.01,1/6.-0.01)
        #####################  Calculate relative beam velocity ###############
        print_line( '  * Calculating '+imp+' density ...')
        TT = time()

        ab = 2.014
        n_beam_spec = 3
        from scipy.constants import e,m_p
        # full eV/amu, divide later for half and third
        ab_ = ab * np.arange(1,n_beam_spec+1)  # mass in amu assuming Deuterium
        qb = np.ones(n_beam_spec)  # charge in e
        # beam energies for all species on radial grid
        # energy is volts * charge/ mass (?)

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
        bms = [read_adf21(f, erel, edens, te) for f,edens in zip(files_bms, eff_dens)] # cm^3/s
        #n=2 excitation rate
        bmp = [read_adf21(f, erel, edens, te) for f,edens in zip(files_bmp, eff_dens)] # cm^3/s

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
            lnbeam_att = np.minimum(lnbeam_att, 10)
            lnbeam_att = np.dstack((np.zeros((nbeam, n_beam_spec)), lnbeam_att))
            lnbeam_att_err = np.cumsum(datt_err_b*dlencm.T[:,None],axis=-1)
            lnbeam_att_err = np.dstack((np.zeros((nbeam, n_beam_spec)), lnbeam_att_err))
            
            beam_att.append(np.exp(-lnbeam_att))
        
            #print('BUG!!!')
            # assume correlated error - ne is systematically higher/lower within the uncertainty
            beam_att_err.append(beam_att[-1] * lnbeam_att_err)  # keep in mind, that uncertaintes of all beams and species are correlated
            # assume 5% error in beam power
            beam_att_err[-1] = np.hypot(beam_att_err[-1], 0.05 * beam_att[-1])
            n += nR

     

        #####################  Calculate Halo  ##################
        #Based on R. McDermott PPCF 2018 paper
        te = beam_prof_merged['te'] #eV
        ti = beam_prof_merged['Ti'] #eV        
        ne = beam_prof_merged['ne'] / 1.0e6  # cm^-3
        v = vrel  # cm/s
        nD = (1-beam_prof_merged['fC']*6)*ne  # cm^-3
        
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
        vth = np.sqrt((2*ti*consts.e)/(ab * consts.m_p))#m/s
        halo_std = vth[valid]/ionis_rate*100 #width of neutral gaussian distribution
        beam_height = 40  #cm BUG just a guess
        #correction for a finite transport of neutrals, assume that the extend in horisontal direction
        #will not change line integral (it will smear the measurements location)
        #but extend in vertical direction will reduce line integral
        #vertical extend of neutral distribution of height of sqrt(beam gaussian**2+height of neutral distribution**2)
        corr = np.hypot( beam_height,  halo_std  )/beam_height

        #CX crossection for D beam ions in D plasma
        #data from file qcx#3h0_ory#h1.dat
        E = [10., 50., 100., 200., 300., 500., 1000.]#energies(keV/amu)
        #sigma = [7.95e-16 ,9.61e-17 ,1.54e-17 ,1.29e-18, 2.35e-19 ,2.20e-20, 6.45e-22 ]#total xsects.(cm2)
        sigma = [7.70e-16, 6.13e-17, 9.25e-18, 8.44e-19, 1.63e-19, 1.63e-20, 5.03e-22] #n=1 crossection
        sigmaDD = np.exp(np.interp(np.log(erel/1e3), np.log(E),np.log(sigma)))#cm2

        #n=1 halo fraction, normalised to the total number of injected neutrals
        f0halo1 = np.zeros_like(sigmaDD)# magnitude consistent with Rachael's paper
        f0halo1[:,:,valid] = (sigmaDD*v*nD)[:,:,valid]/(ionis_rate*corr)

    
        #Layman alpha emission
        PECfile = path+'/pec96#h_pju#h0.dat'
        PEC = read_adf15(PECfile, 1, te,ne)#ph cm**3 s**-1)
        A21 = 4.6986e+08#s^-1 einsten coefficient from https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        #n=2 halo fraction, magnitude consistent with Rachael's paper, normalised to the total number of injected neutrals
        f0halo2 = f0halo1*(PEC*ne)/A21
        #  Rachale used FIDASIM to calculate a spatial profile of these ions
         
  
        ######################### Calculate CX cross-sections  ############################# 
        zeff = beam_prof_merged['zeff']
        ti = beam_prof_merged['Ti'] # cm^-3
        ne = beam_prof_merged['ne'] / 1.0e6  # cm^-3
        erel_ = beam_prof_merged['erel']* ab  #eV

        
        #CX with beam ions
        adf_interp = read_adf12_aug(path,line_id, n_neut=1)
        qeff = adf_interp(zeff, ti, ne, erel_)
        adf_interp = read_adf12_aug(path,line_id, n_neut=2)
        qeff2 = adf_interp(zeff, ti, ne, erel_)
        ## cm**3/s to m**3/s and  per ster. like CER
        qeff  /= 1e6 * 4.0 * np.pi
        qeff2 /= 1e6 * 4.0 * np.pi

        #CX with beam halo
        adf_interp = read_adf12_aug(path,line_id, n_neut=1, therm=True)
        qeff_th = adf_interp(zeff, ti, ne).T
        adf_interp = read_adf12_aug(path,line_id, n_neut=2, therm=True)
        qeff2_th = adf_interp(zeff, ti, ne).T
        ## cm**3/s to m**3/s and  per ster. like CER
        qeff_th  /= 1e6 * 4.0 * np.pi
        qeff2_th /= 1e6 * 4.0 * np.pi
        
        
        qeff1_=read_adf12(path+'/qef93#h_c6.dat',5, erel, nD, ti, zeff*0+1)  # wrong Zeff dependence
        qeff2_=read_adf12(path+'/qef97#h_en2_kvi#c6.dat',5, erel, nD, ti, zeff*0+1)
        qeff1_ /= 1e6 * 4.0 * np.pi
        qeff2_ /= 1e6 * 4.0 * np.pi
        embed()
        
    
        ######################### Calculate Impurity density  #############################
        
        isp = np.arange(3)+1  # mass multipliers for beam species
        eV_to_J = consts.e
        mp = consts.m_p


 
        einj = nbi_dict['volts']  # V
        # Velocity of beam species
        pwrfrc = nbi_dict['pow_frac'].T[0]
        n2frac = np.dstack(n2frac)

        vinj = np.sqrt(2.0 * einj * eV_to_J) / np.sqrt(ab * isp * mp)[:, None]  # m/s
        nb0 = pwrfrc / (einj * eV_to_J * vinj / isp[:, None])  # 1/m/W must be multipled by power
        qeff = qeff * (1 - n2frac) + n2frac * qeff2 #  qeff from n=1 and n=2

        #BUG inaccurate any geometry correction for spatial distribution of halo
        #works only in the limit of high density when the collisions prevent spreading the halo
        
        # Get beam attenuation data and the qeff values
        ne = beam_prof_merged['ne'] / 1.0e6  # cm^-3
        qeff+= qeff_th*f0halo1 #small, negligible
        qeff+= qeff2_th*f0halo2 #comparable with qeff2

        #impurity densities foe all channels
        nz = np.zeros_like(nimp_data['int'])
        nz_err = np.ones_like(nimp_data['int'])*np.inf
        
        
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
            denom_interp = interp1d(Rmid, np.sum(nb0.T[:,:,None] * beam_att[it] * qeff[:,:,tind], 1))  # nR x nbeam

            # uncertainties in beam_att_err between species are 100% correlated, we can sum them
            denom_err_interp = interp1d(Rmid, np.sum(nb0.T[:,:,None] * beam_att_err[it] * qeff[:,:,tind], 1))  # nR x nbeam
            
            denom = np.sum(beam_fact[:,ind]*denom_interp(R_clip),0)
            denom_err = np.sum(beam_fact[:,ind]*denom_err_interp(R_clip),0)
            
            #sometimes is power observed by vertical core system zero, but intesnty is nonzero
            invalid = (denom == 0)|np.isnan(nimp_data['int_err'][ind])|(nimp_data['int'][ind]==0)
            if np.any(invalid):
                ind[ind] &= ~invalid
                denom=denom[~invalid] 
                denom_err=denom_err[~invalid]

            nz[ind] = nimp_data['int'][ind]/denom
            nz_err[ind] = nz[ind] * np.hypot(nimp_data['int_err'][ind] / (1+nimp_data['int'][ind]), denom_err / denom)
             
        
        #fill the xarray database with calculated impurity densities
        n = 0
        for diag in systems:
            for ch in nimp[diag]:
                ch['nimp_int'] = ch['nimp'].copy()
                #print(ch.attrs['channel'], nz[data_index[n]].max())
                
                #if any(nz[data_index[n]] > 1e19):
                    #embed()
                ch['nimp_int'].values = nz[data_index[n]]
                ch['nimp_int_err'] = ch['nimp'].copy()
                ch['nimp_int_err'].values = nz_err[data_index[n]]
                n += 1
      
      
        print('\t done in %.1fs'%(time()-TT))
      
        return nimp
      
 
    def load_nimp_impcon(self, nimp,load_systems, analysis_type,imp):
        #load IMPCON data and split them by CER systems
        #in nimp are already preload channels
        if len(load_systems) == 0:
            return nimp
        
        
        ##############################  LOAD DATA ######################################## 

        print_line( '  * Fetching IMPCON data from %s ...'%analysis_type)
        T = time()

        tree = 'IONS'        

        
        imp_path = '\%s::TOP.IMPDENS.%s.'%(tree,analysis_type) 
        nodes = 'IMPDENS', 'ERR_IMPDENS', 'INDECIES', 'TIME'
        TDI = [imp_path+node for node in nodes]
        
        
        #array_order in the order as it is stored in INDECIES
        TDI += ['\IONS::TOP.CER.CALIBRATION:ARRAY_ORDER']

        #fast fetch
        nz,nz_err, ch_ind, tvec,array_order = mds_load(self.MDSconn, TDI, tree, self.shot)
        if len(nz) == 0:
            raise Exception('No IMPCON data')
        #embed()
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
                #merge impcon and CER timebases
                tch = np.round((ch['time'].values-ch['stime'].values/2)*1e3,2)
                timp = np.round(tvec[ind],2)#round to componsate for small rounding numerical errors
                valid = np.in1d(tch,timp, assume_unique=True)#round to componsate for small rounding numerical errors
                #channel was not included in IMPCON analysis
                if not any(valid):
                    #if len(timp) > 0:
                        ##embed()
                        #print('no valid points',ch.attrs['channel'], len(tch), len(timp) )
                    continue
                
                tind = np.in1d(timp,tch, assume_unique=True)#round to componsate for small rounding numerical errors

                nz_ = nz[ind][tind]
                nzerr_ = nz_err[ind][tind]
 
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
                ch['nimp_impcon'].values[valid]  = nz_
                ch['nimp_impcon_err'] = ch['nimp_err'].copy()
                ch['nimp_impcon_err'].values[valid]  = nzerr_


        print('\t done in %.1fs'%(time()-T))
        return nimp

       
 
    def load_nimp(self,tbeg,tend,systems, options):
        
        #TODO ad option to remove first timepoint after beamblip

        selected,analysis_types = options['Analysis']
   
        rcalib = options['Correction']['Relative calibration'].get() 
        #calculate impurity density from intensity directly

        
        analysis_type = self.get_cer_types(selected.get())
        suffix = ''
        imp = 'C6'
        if 'Impurity' in options and options['Impurity'] is not None:
            imp = options['Impurity']
            suffix += '_'+imp
        
        nz_from_intens = False
        if 'nz from CER intensity' in options['Correction']:
            nz_from_intens = options['Correction']['nz from CER intensity'].get() 
     

        #load from catch if possible
        self.RAW.setdefault('nimp',{})
        nimp = self.RAW['nimp'].setdefault(analysis_type+suffix ,{} )
        
        #which cer systems should be loaded
        load_systems = deepcopy(systems)
        nimp_name = 'nimp_int' if nz_from_intens else 'nimp_impcon'
        for sys in systems:
            if sys in nimp and (len(nimp[sys])==0 or any([nimp_name in ch for ch in nimp[sys]])): #data already loaded
                load_systems.remove(sys)
            

        #rho coordinate of the horizontal line, used later for separatrix aligment 
        if 'horiz_cut' not in nimp or 'EQM' not in nimp or nimp['EQM']['id'] != id(self.eqm) or nimp['EQM']['ed'] != self.eqm.diag:
            R = np.linspace(1.4,2.5,100)
            rho_horiz = self.eqm.rz2rho(R, np.zeros_like(R), coord_out='rho_tor')
            nimp['horiz_cut'] = {'time':self.eqm.t_eq, 'rho': np.single(rho_horiz), 'R':R}
        
        
        nimp['systems'] = systems
        nimp.setdefault('rel_calib', rcalib)
        nimp.setdefault('diag_names',{})

        #skip loading when already loaded,
        same_eq = 'EQM' in nimp and nimp['EQM']['id'] == id(self.eqm) and nimp['EQM']['ed'] == self.eqm.diag
        if len(load_systems) == 0  and (not nz_from_intens or same_eq) and (not rcalib or nimp['rel_calib']):
            #return corrected data if requested
            nimp = self.eq_mapping(nimp)

            for ch in [ch for s in nimp['systems'] for ch in nimp[s]]:
                suffix = ''
                if rcalib and nimp_name+'_corr' in ch: #if the correction has not failed
                    suffix = '_corr'
                #set requested density timetraces in each channel
                if nimp_name+suffix in ch:
                    ch['nimp'] = ch[nimp_name+suffix]
                    ch['nimp_err'] = ch[nimp_name+'_err'+suffix]
                else: #data are not avalible or this channel
                    ch['nimp'].values[:] = 0
                    ch['nimp_err'][:] = -np.inf
            return nimp
        


        #update equilibrium of catched channels
        nimp = self.eq_mapping(nimp)

        #load either from IMPCON or calculate from CX intensity
        if nz_from_intens:
            options['Correction']['nz from CER intensity'].set(0) 
            options['Correction']['Relative calibration'].set(1)
            options['Impurity'] = 'C6'
            
            try:
                nimp0 = self.load_nimp(tbeg,tend,systems,options) 
            except Exception as e:
                printe('Error: '+str(e))
                nimp0 = None 

            load_systems = list(set(load_systems)-set(nimp.keys()))
                
            #first load data from MDS+
            nimp = self.load_nimp_intens(nimp,load_systems, analysis_type,imp)
            #calculate impurity density
            nimp = self.calc_nimp_intens(tbeg,tend,nimp,systems,imp, nimp0)

            options['Correction']['nz from CER intensity'].set(nz_from_intens) 
            options['Correction']['Relative calibration'].set(rcalib)
        else:
            nimp = self.load_nimp_intens(nimp,load_systems, analysis_type,imp)
            #load just the impurity density
            try:
                nimp = self.load_nimp_impcon(nimp,load_systems, analysis_type,imp)
            except Exception as e:
                printe('Error: '+str(e))
                raise
                
 
        

        ##update uncorrected data
        diag_names = sum([nimp['diag_names'][diag] for diag in nimp['systems']],[])
        impurities = np.hstack([[ch['nimp'].attrs['impurity'] for ch in nimp[s] if ch.system == s] for s in nimp['systems']])
        unique_impurities = np.unique(impurities)
        T = time()
        all_channels = [ch for s in nimp['systems'] for ch in nimp[s]]

        #reduce discrepancy between different CER systems
        if rcalib and len(diag_names) > 1 and len(unique_impurities)==1:
            print( '\n\t* Relative calibration of beams  ...')
            NBI = self.RAW['nimp']['NBI']

            #determine "calibrated density source"
            voltages, times = {},{}
            for name,beam in NBI.items():
                if 'fired' in beam and beam['fired']:
                    print('\t\tinfo beam:%s\tV=%dkV\tsingle beam time = %.2fs'%(name,beam['volts']/1e3,beam.get('beam_fire_time',np.nan)))

            #if beam 30L is on ~80kV use it for cross-calibration
            if '30L' in NBI and NBI['30L']['fired'] and ( 77 < NBI['30L']['volts']/1e3 < 83) and NBI['30L']['beam_fire_time'] > .5:
                print('\t* Using beam 30L for cross calibration')
                calib_beam = '30L'

            elif '30R' in NBI and NBI['30R']['fired'] and (77 < NBI['30R']['volts']/1e3 < 83) and NBI['30R']['beam_fire_time'] > .5:
                print('\t\tUsing beam 30R for cross calibration')
                calib_beam = '30R'
                
            elif '30B' in NBI and (( 77 < NBI['30R']['volts']/1e3 < 83) and ( 77 < NBI['30L']['volts']/1e3 < 83)):
                print('\t\tUsing beam 30R+30L for cross calibration - less reliable')
                calib_beam = '30B'
            else:
                printe('\t\tNo reliable beam for cross calibration, using 30L anyhow...')
                calib_beam = '30L'

            #cross-calibrate other profiles
            calib = {'t':[],'r':[],'n':[]}
            other = {'t':[],'r':[],'n':[],'f':[],'w':[]}
            #iterate over all channels
 
            #treat tangential and vertical system and edge corrections independently
            beams = []
            for ch in all_channels:
                sys = ch.attrs['system'][0].upper()
                edge = 'e' if ch.attrs['edge'] else ''
                beams+= [sys+b+edge for b in ch['beams'].values]
            beams = list(np.unique(beams))
            
            for ch in all_channels:
                 #ch._variables['time']._data.array._data
                t = ch['time'].values  #normalize a typical time range with respect to typical rho range)
                beam_sys = ch['diags'].values[0]
                sys = ch.attrs['system'][0].upper()
                edge = 'e' if ch.attrs['edge'] else ''
                if not nimp_name in ch: continue
                nz  = ch[nimp_name].values/1e18 #to avoid numerical overflow with singles
                nz_err = ch[nimp_name+'_err'].values/1e18 #to avoid numerical overflow with singles
                rho = ch['rho'].values
           
                ind =  (nz > 0)&(nz_err > 0)&(nz_err < 1e2)&(rho < .9) #do not include data on the pedestal
         
                if any(ind):
                    beam_frac = np.zeros((len(beams), len(t)))
                    beam_frac_= ch['beam_frac'].values
                    for ib,b in enumerate(ch['beams'].values):
                        beam_frac[beams.index(sys+b+edge)] = beam_frac_[ib]
                    weight = np.ones_like(t[ind])
                    if 'T_'+calib_beam+' ' in beam_sys: #just midradius channels
                        calib['t'].append(t[ind])
                        calib['r'].append(rho[ind])
                        calib['n'].append(nz[ind])
                        weight*=100  #force calibrated channel to be fixed at correction 1  
                    other['t'].append(t[ind])
                    other['r'].append(rho[ind])
                    other['n'].append(nz[ind])
                    other['f'].append(beam_frac[:,ind])
                    other['w'].append(weight)

            if len(calib['t']) == 0:
                printe('unsuccesful..')
                for ch in all_channels:
                    #set requested density timetraces in each channel
                    if nimp_name in ch:
                        ch['nimp'] = ch[nimp_name]
                        ch['nimp_err'] = ch[nimp_name+'_err']
                    else: #data are not avalible or this channel
                        ch['nimp'].values[:] = 0
                        ch['nimp_err'][:] = -np.inf
                return nimp
                        
            calib = {n:np.hstack(d).T for n,d in calib.items()}
            other = {n:np.hstack(d).T for n,d in other.items()}

            #get indexes of nearest points
            interp = NearestNDInterpolator(np.c_[calib['t'],calib['r']],np.arange(len(calib['t'])))
            near_ind = interp(np.c_[other['t'],other['r']])
            dist = np.hypot(calib['t'][near_ind]-other['t'],calib['r'][near_ind]-other['r'])
            nearest_ind = dist < np.median(dist[dist>0])
            
            #use least squares to find the cross-calibration factors for each beam
            N = other['n'][nearest_ind]/calib['n'][near_ind[nearest_ind]]
            F = other['f'][nearest_ind]
            W = other['w'][nearest_ind]
            C = np.linalg.lstsq(F*W[:,None], N*W)[0]
            
            for c,b in zip(C,beams):
                if c > 0: print('\t\t correction '+b+': %.2f'%(1/c))
    
              
            #apply correction, store corrected data 
            for ch in [ch for s in nimp['systems'] for ch in nimp[s]]:
                if not nimp_name in ch: continue
                sys = ch.attrs['system'][0].upper()
                edge = 'e' if ch.attrs['edge'] else ''
                ch_beams = [sys+b+edge for b in ch['beams'].values]
                corr = np.dot(ch['beam_frac'].values.T, C[np.in1d(beams,ch_beams)])
                valid = corr > 0
                ch[nimp_name+'_corr'] = ch[nimp_name].copy()#copy including the attributes
                ch[nimp_name+'_corr'].values[valid] /= corr[valid]
                ch[nimp_name+'_err_corr'] = ch[nimp_name+'_err'].copy()
                ch[nimp_name+'_err_corr'].values[valid] /= np.sqrt(corr[valid]) #sqrt to make sure that the errorbars will remain larger
            
            
        elif rcalib and len(unique_impurities)>1:
            printe('Calibration is not implemented for two impurities in NIMP data')
            rcalib = False
                 

        nimp['rel_calib'] |= rcalib

        #return corrected data from the right source in ch['nimp'] variable
        
        for ch in all_channels:
            suffix = ''
            if rcalib and nimp_name+'_corr' in ch: #if the correction has not failed
                suffix = '_corr'
            #set requested density timetraces in each channel
            if nimp_name+suffix in ch:
                ch['nimp'] = ch[nimp_name+suffix]
                ch['nimp_err'] = ch[nimp_name+'_err'+suffix]
            else: #data are not avalible or this channel
                ch['nimp'].values[:] = 0
                ch['nimp_err'][:] = -np.inf

        if rcalib:
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
        zeff.setdefault('diag_names',{})
 
        #update equilibrium for already loaded systems
        zeff = self.eq_mapping(zeff)

  
        print_line( '  * Fetching VB (slow) ...' )
        lambda0 = 5230.0
        #NOTE slow, VB signals are stored in a large time resolution

        ######################   VB array data #############################
        
        if VB_array in systems:            
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

                
                zeff[VB_array] = xarray.Dataset(attrs={'system':VB_array,'wavelength': 5230.0})

                zeff[VB_array]['VB'] = xarray.DataArray(VB,dims=['time','channel'], attrs={'units':'W/cm**2/A','label':'VB' })
                zeff[VB_array]['VB_err'] = xarray.DataArray(VB_err,dims=['time','channel'], attrs={'units':'W/cm**2/A'})
                zeff[VB_array]['diags']= xarray.DataArray( np.tile((VB_array,), VB.shape),dims=['time','channel'])
                
                zeff[VB_array]['R_start'] = xarray.DataArray(R_start, dims=['channel'], attrs={'units':'m'})
                zeff[VB_array]['R_end'] = xarray.DataArray(R_end, dims=['channel'], attrs={'units':'m'})

                zeff[VB_array]['z_start'] = xarray.DataArray(z_start,dims=['channel'], attrs={'units':'m'})
                zeff[VB_array]['z_end'] = xarray.DataArray(z_end,dims=['channel'], attrs={'units':'m'})
                
                zeff[VB_array]['phi_start'] = xarray.DataArray(phi_start,dims=['channel'], attrs={'units':'m'})
                zeff[VB_array]['phi_end'] = xarray.DataArray(phi_end,dims=['channel'], attrs={'units':'m'})
                
                zeff[VB_array]['time'] = xarray.DataArray(tvec.astype('single'),dims=['time'], attrs={'units':'s'})
                zeff[VB_array]['channel'] = xarray.DataArray(['VB%.2d'%ich for ich in range(1,nchans+1)])
                zeff[VB_array]['razor'] = xarray.DataArray(razor,dims=['channel'] )
                zeff['diag_names'][VB_array] = [VB_array]
                
      
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
                    try:
                        zeff[sys][ich]['Zeff'] = xarray.DataArray(np.single(Zeff),dims=['time'], attrs={'units':'-','label':'Z_\mathrm{eff}'})
                    except:
                        embed()
                    zeff[sys][ich]['Zeff_err'] = xarray.DataArray(np.single(Zeff_err),  dims=['time'])
    
        #embed()

  

        return zeff


    def load_cer(self,tbeg,tend, systems, options=None):
        #load Ti and omega at once
        T = time()

        tree = 'IONS'
        if options is None:
            analysis_type = 'best'
            analysis_types= 'fit','auto','quick'
        else:
            selected,analysis_types = options['Analysis']
            analysis_type = selected.get()
            
        analysis_type = self.get_cer_types(analysis_type)
         
        self.RAW.setdefault('CER',{})
        cer = self.RAW['CER'].setdefault(analysis_type,{})

        #load from catch if possible
        cer.setdefault('diag_names',{})
        cer['systems'] = systems
        load_systems = list(set(systems)-set(cer.keys()))
 
        #update equilibrium for already loaded systems
        cer = self.eq_mapping(cer)
        

        if len(load_systems) == 0:
            return cer
   
        print_line( '  * Fetching '+analysis_type.upper()+' data ...' )
        
        cer_data = {
            'Ti': {'label':'Ti','unit':'eV','sig':['TEMP','TEMP_ERR']},
            'omega': {'label':r'\omega_\varphi','unit':'rad/s','sig':['ROT','ROT_ERR']}}
        #NOTE visible bramstrahlung is not used
       
        #list of MDS+ signals for each channel
        signals = cer_data['Ti']['sig']+cer_data['omega']['sig'] + ['R','Z','STIME','TIME']

        all_nodes = []        
        TDI = []
        TDI_phi = []
        diags_ = []
        data_nbit = []
        missing_rot = []
        corr_temp = False
        corr_rot = False

        try:
            self.MDSconn.openTree(tree, self.shot)
            #check if corrected rotation datta are availible
            try:
                if len(self.MDSconn.get('getnci("...:ROTC", "depth")')) > 0:
                    corr_rot = True
            except MDSplus.MdsException:
                pass
            #check if corrected temperature data are availible
            try:
                if len(self.MDSconn.get('getnci("...:TEMPC", "depth")')) > 0:
                    corr_temp = True
            except MDSplus.MdsException:
                pass
                  
            #prepare list of loaded signals
            for system in load_systems:
                cer[system] = []
                cer['diag_names'][system] = []
                path = 'CER.%s.%s.CHANNEL*'%(analysis_type,system)
                nodes = self.MDSconn.get('getnci("'+path+'","fullpath")')
                lengths = self.MDSconn.get('getnci("'+path+':TIME","LENGTH")').data()
                lengths_Rot = self.MDSconn.get('getnci("'+path+':ROT","LENGTH")').data()
         
                for node,length,length_r in zip(nodes,lengths,lengths_Rot):
                    if length == 0: continue
                    try:
                        node = node.decode()
                    except:
                        pass
                
                    diags_.append(system)
 
                    node = node.strip()
                    all_nodes.append(node)
                    data_nbit.extend([length]*len(signals) )
                    
                    node_calib = node.replace(analysis_type.upper(),'CALIBRATION')
                    TDI_phi.append(node_calib.strip()+':'+'LENS_PHI')

                    for sig in signals:
                        #sometimes if rotation not availible even if Ti is
                        if length_r == 0 and sig in ['ROT','ROT_ERR']:
                            missing_rot.append(node)
                            sig = 'STIME'
                            
                        #use atomic data corrected signals if availible                        
                        elif (sig == 'ROT' and  system == 'tangential' and corr_rot) or (sig == 'TEMP' and corr_temp):
                            sig = sig+ 'C'
                            
                        TDI.append(node+':'+sig)

        except Exception as e:
            raise
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

        
        TDI_list = np.reshape(TDI,(-1,len(signals))).T
        TDI_list = ['['+','.join(tdi)+']' for tdi in TDI_list]
        TDI_list+= ['['+','.join(TDI_phi)+']']
        
        data = mds_load(self.MDSconn, TDI_list , tree, self.shot) 
        phi = data.pop()
        
        if len(data[-1]) == 0:
            print('Data fetching has failed!!')
            embed()
            raise Exception('Data fetching has failed!!')


        #split data in list if profiles of list of signals
        data = np.single(data).flatten()
        data_nbit = np.reshape(data_nbit,(-1,len(signals))).T.flatten()
        data = split_mds_data(data, data_nbit, 8)
        Ti,Ti_err,rot,rot_err,R,Z,stime,tvec = np.reshape(data, (len(signals),-1))
        


        #get a time in the center of the signal integration 
        tvec = [np.single(t+s/2.)/1000 for t,s in zip(tvec, stime)] 
            
        #map to radial coordinate 
        rho = self.eqm.rz2rho(np.hstack(R)[:,None],np.hstack(Z)[:,None],np.hstack(tvec),self.rho_coord)[:,0]
        
        for ich,ch in enumerate(all_nodes):
            nt = len(tvec[ich])
            #if nt == 0: continue
            rho_,rho = rho[:nt], rho[nt:]
            diag = ch.split('.')[-2].lower()
            name = diags_[ich][:4]+'_%d'%phi[ich]            
      
            if not name in cer['diag_names'][diag]:
                cer['diag_names'][diag].append(name)
            ds = xarray.Dataset(attrs={'channel':ch})
            ds['R'] = xarray.DataArray(R[ich], dims=['time'], attrs={'units':'m'})
            ds['Z'] = xarray.DataArray(Z[ich], dims=['time'], attrs={'units':'m'})
            ds['rho'] = xarray.DataArray(rho_, dims=['time'], attrs={'units':'-'})
            ds['diags']= xarray.DataArray(np.array((name,)*nt),dims=['time'])

   
            if len(rot[ich]) > 0 and ch not in missing_rot:
                corrupted = ~np.isfinite(rot[ich])
                rot[ich][corrupted] = 0
                corrupted |= (rot[ich]<-1e10)|(rot_err[ich]<=0)
                rot_err[ich][corrupted] = np.infty

                rot[ich]    *= 1e3/R[ich]   
                rot_err[ich]*= 1e3/R[ich] 

                ds['omega'] = xarray.DataArray(rot[ich],dims=['time'], attrs={'units':'rad/s','label':r'\omega_\varphi'})
                ds['omega_err'] = xarray.DataArray(rot_err[ich],dims=['time'], attrs={'units':'rad/s'})
            
                
            if len(Ti[ich]) > 0:
                Ti_err[ich][(Ti[ich]>=15e3)|(Ti[ich] <= 0)] = np.infty
                Ti_err[ich][~np.isfinite(Ti[ich])|(Ti[ich]<-1e10)|(Ti_err[ich]<=0)] = np.infty
                Ti[ich][~np.isfinite(Ti[ich])] = 0

                ds['Ti'] = xarray.DataArray(Ti[ich],dims=['time'], attrs={'units':'eV','label':'T_i'})
                ds['Ti_err'] = xarray.DataArray(Ti_err[ich],dims=['time'], attrs={'units':'eV'})
 
            ds['time'] = xarray.DataArray(tvec[ich], dims=['time'], attrs={'units':'s'})
            cer[diag].append(ds)
            
        cer['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}
        print('\t done in %.1fs'%(time()-T))

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
        if revision!= 'BLESSED':
            revision = 'REVISIONS.'+revision
    

            
        signals = 'DENSITY', 'DENSITY_E', 'TEMP', 'TEMP_E','TIME','R','Z','lforder'
    
        tree = 'ELECTRONS'
        TDI = []        
        #prepare list of loaded signals
        for system in systems:
            if system in ts: continue
            tdi = '\\%s::TOP.TS.%s.%s:'%(tree,revision,system)
            ts['diag_names'][system]=['TS:'+system]
            for sig in signals:
                TDI.append(tdi+sig)

        out = mds_load(self.MDSconn, TDI, tree, self.shot)
        
        ne,ne_err,Te,Te_err,tvec,R,Z,laser = np.asarray(out).reshape(-1, len(signals)).T

        for isys, sys in enumerate(systems):
            if len(tvec) <= isys or len(tvec[isys]) == 0: 
                ts['systems'].remove(sys)
                continue
            tvec[isys]/= 1e3        
            
            #these points will be ignored and not plotted (negative errobars )
            Te_err[isys][(Te_err[isys]<=0) | (Te[isys] <=0 )]  = -np.infty
            ne_err[isys][(ne_err[isys]<=0) | (ne[isys] <=0 )]  = -np.infty
                
            channel = np.arange(Te_err[isys].shape[0])
            
            rho = self.eqm.rz2rho(R[isys],Z[isys]+zshift,tvec[isys],self.rho_coord)
         
            ts[sys] = xarray.Dataset(attrs={'system':sys})
            ts[sys]['ne'] = xarray.DataArray(ne[isys].T,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
            ts[sys]['ne_err'] = xarray.DataArray(ne_err[isys].T,dims=['time','channel'], attrs={'units':'m^{-3}'})
            ts[sys]['Te'] = xarray.DataArray(Te[isys].T,dims=['time','channel'], attrs={'units':'eV','label':'T_e'})
            ts[sys]['Te_err'] = xarray.DataArray(Te_err[isys].T,dims=['time','channel'], attrs={'units':'eV'})
            ts[sys]['diags']= xarray.DataArray( np.tile(('TS:'+sys,), ne[isys].T.shape),dims=['time','channel'])            
            ts[sys]['R'] = xarray.DataArray(R[isys], dims=['channel'], attrs={'units':'m'})
            ts[sys]['Z'] = xarray.DataArray(Z[isys],dims=['channel'], attrs={'units':'m'})
            ts[sys]['laser'] = xarray.DataArray(laser[isys],dims=['time'], attrs={'units':'-'})
            ts[sys]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})
            ts[sys]['time'] = xarray.DataArray(tvec[isys],dims=['time'], attrs={'units':'s'})
            ts[sys]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})
            
 
        print('\t done in %.1fs'%(time()-T))
        ts['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':zshift, 'ed':self.eqm.diag}

        return ts 
        
        
        
    def load_refl(self, tbeg,tend, options, TS=None):
        T = time()
        #TODO add REFLECT_F. calculate radial position by myself
        TS_align = options['Position error']['Align with TS'].get() #[m]  
        
        if 'REFL' in self.RAW and not TS_align:
            #assume that equilibrium could be changed
            return self.eq_mapping(self.RAW['REFL'])   
         
        print_line( '  * Fetching reflectometer data ...')

        bands = 'VO','V','QO','Q'
        tree = 'ELECTRONS'
        prefix ='\\'+tree+'::TOP.REFLECT.'


        refl = self.RAW['REFL'] = {}
            
        refl['systems'] = bands

        TDI = []
        for band in bands:
            TDI.append('dim_of('+prefix+band+'BAND.PROFILES:DENSITY'+',0)')
            TDI.append(prefix+band+'BAND.PROFILES:DENSITY')
            TDI.append(prefix+band+'BAND.PROFILES:R')
            #TDI.append(prefix+band+'BAND.PROFILES:DENSITY_ERR')#dont exist
            #TDI.append(prefix+band+'BAND.PROFILES:R_ERR')#dont exist
            #TDI.append(prefix+band+'PROCESSED:FREQUENCY')# empty


        #fetch data from MDS+
        out = mds_load(self.MDSconn, TDI, tree, self.shot)
        out = np.reshape(out, (len(bands), -1))

        if np.size(out) == 0:
            printe( '\tNo Reflectometer data')
            return
        
        #fetch TS data for alighnment
        if TS_align:
            if TS is None:
                TS = self.load_ts(tbeg,tend,['core'])
 
        
        
        
        refl['diag_names'] = {}

        for band, (tvec,ne, R) in zip(bands, out):
            if np.size(tvec) == 0:  continue
            tvec/= 1e3 #s
            R, ne = np.single(R.T), np.single(ne.T)
            z = np.zeros_like(R)+0.0254
            phi = np.zeros_like(R)+255
            rho = self.eqm.rz2rho(R,z,tvec,self.rho_coord)
            R_shift = np.zeros_like(tvec)

            if TS_align:

                TS_time = TS['core']['time'].values
                TS_rho = TS['core']['rho'].values
                TS_ne = TS['core']['ne'].values 
                TS_neerr = TS['core']['ne_err'].values
                
                rho_out = 0.7  #use only data outside rho_out
                for it,t in enumerate(tvec):
                    its = np.argmin(np.abs(TS_time-t))
                    valid_ts = np.isfinite(TS_neerr[its]) &(TS_rho[its] > rho_out)
                    valid_rfl = (rho[it] > rho_out)
    
                    R_ts = np.interp(TS_rho[its, valid_ts],rho[it], R[it]) #midplane R coordinate for TS
                    R_rfl = R[it, valid_rfl]
                    ne_TS = TS_ne[its,valid_ts]/ 1e19
                    ne_RFL = ne[it,valid_rfl]/1e19
                    shift = np.linspace(-0.1,0.1,50)
                    conv = np.zeros_like(shift)
                    for ish, s in enumerate(shift):
                        ne_RFL_shift = np.interp(R_ts, R_rfl+s, ne_RFL)
                        conv[ish] = np.sum((ne_RFL_shift-ne_TS)**2)
        
                    _,R_shift[it] = min_fine(shift, conv)
                
                rho = self.eqm.rz2rho(R+R_shift[:,None],z,tvec,self.rho_coord)

                


            channel = np.arange(ne.shape[1])
            refl[band] = {}

            refl['diag_names'][band] = ['REFL:'+band+'BAND']
            refl[band] = xarray.Dataset(attrs={'band':band})
            refl[band]['time'] = xarray.DataArray(tvec,dims=['time'], attrs={'units':'s'})
            refl[band]['channel'] = xarray.DataArray(channel,dims=['channel'], attrs={'units':'-'})

            refl[band]['ne'] = xarray.DataArray(ne,dims=['time','channel'], attrs={'units':'m^{-3}','label':'n_e'})
             #just guess! of 10% errors!!
            refl[band]['ne_err'] = xarray.DataArray(ne*0.1+ne.mean()*0.01 ,dims=['time','channel'], attrs={'units':'m^{-3}'})
            refl[band]['diags']= xarray.DataArray(np.tile(('REFL:'+band+'BAND',), R.shape),dims=['time','channel'])
            refl[band]['R'] = xarray.DataArray(R,dims=['time','channel'], attrs={'units':'m'})
            refl[band]['R_shift'] = xarray.DataArray(R_shift,dims=['time'], attrs={'units':'m'})
            refl[band]['Z'] = xarray.DataArray(z,dims=['time','channel'], attrs={'units':'m'})
            refl[band]['rho'] = xarray.DataArray(rho,dims=['time','channel'], attrs={'units':'-'})

            
        refl['EQM'] = {'id':id(self.eqm),'dr':np.mean(R_shift), 'dz':0,'ed':self.eqm.diag}

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
            
        #self.eqm.read_ssq()
               
        #embed()
            
        
        #plot(t_eq, R,'k',lw=.2)
        #plot(t_eq, self.eqm.ssq['Rmag'][t_ind],lw=2)
        #show()
        
        
        
     
        

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
    
    
    def load_co2(self, tbeg,tend, calc_weights=True):
        
        T = time()

         
        CO2 = self.RAW.setdefault('CO2',{})

        #update mapping of the catched data
        if 'systems' in CO2 and (not calc_weights or 'weight' in CO2[CO2['systems'][0]]):
            CO2 = self.eq_mapping(CO2) 
            #return catched data 
            return CO2
        
        
        CO2['systems'] = ['CO2']
        CO2.setdefault('diag_names',{})
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
            TDI_DENS.append('\\%s::TOP.BCI.DEN%s'%(tree,name))
            TDI_STAT.append('\\%s::TOP.BCI.STAT%s'%(tree,name))
        TDI_dens_t= 'dim_of(\\%s::TOP.BCI.DEN%s)'%(tree,name)
        TDI_stat_t = 'dim_of(\\%s::TOP.BCI.STAT%s)'%(tree,name)
        TDI = ['['+','.join(TDI_DENS)+']',TDI_dens_t,
               '['+','.join(TDI_STAT)+']',TDI_stat_t]
        
        ne_,co2_time,stat,stat_time = mds_load(self.MDSconn, TDI, tree, self.shot)
        co2_time /= 1e3
        stat_time /= 1e3
  
        Rlcfs,Zlcfs = self.eqm.rho2rz(0.995)
        n_path = 501
        downsample = 5 
        n_ch = len(channels)

        t = np.linspace(0,1,n_path, dtype='single')        
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
        
   
        for ilos, (name, stat_error_thresh, LOS_pt1, LOS_pt2) in enumerate(channels):
            
            ne_[ilos] /= 2. #Because of the double pass through the plasma
            ne_[ilos] *= 1e6  #/m^3
            
            #Check the status channel - provides a flag for when the signal is not useable due to things like fringeskips
            signal_invalid = stat_time[stat[ilos]>stat_error_thresh]
            #signal_invalid = co2_time[stat[ilos]>stat_error_thresh]

            last_valid_ind = -1
            if any(signal_invalid):
                min_valid_time = signal_invalid[0]
                last_valid_ind = np.argmin(np.abs(co2_time - min_valid_time))
                #Chop the data after the last valid index,
                #need to deal with the case where last_valid_ind is zero...?
                

            #downsample 10x
            ne[:,ilos]   = np.mean(ne_[ilos][:len(ne_[ilos])//downsample*downsample].reshape( -1, downsample), -1)
            valid[:last_valid_ind, ilos] = True
        
            #Calculate the error for the signal based on the drift before t=0
            ne_err[:,ilos] = np.median(np.abs(ne[:,ilos][co2_time<0]))+ne[:,ilos]*.05  #guess 5% error
            ne_err[last_valid_ind:, ilos] = np.inf
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
                L_cross[:,ilos] = np.interp(co2_time, self.eqm.t_eq ,L_cross_)
                L_cross[:,ilos] = np.maximum(L_cross[:,ilos], .1) #just to avoid zero division
                L_cross[:,ilos]*= .9 # correction just for better plotting of the data 
                #convert from m^-2 -> m^-3
                weight[:,ilos,:] = dL/L_cross[:,ilos][:,None]
                ne[:,ilos] /= L_cross[:,ilos]
                ne_err[:,ilos] /= L_cross[:,ilos]
               
   
        #remove offset
        ne  -= ne[(co2_time > -2)&(co2_time < 0)].mean(0)   
           
        CO2['CO2'] = xarray.Dataset()
        CO2['CO2']['channel'] = xarray.DataArray( los_names ,dims=['channel'])
        CO2['CO2']['path'] = xarray.DataArray( t ,dims=['path'])
        CO2['CO2']['time'] = xarray.DataArray( co2_time ,dims=['time'], attrs={'units':'s'})
        CO2['CO2']['valid'] = xarray.DataArray( valid ,dims=['time','channel'], attrs={'units':'s'})
        CO2['CO2']['ne'] = xarray.DataArray(ne, dims=['time', 'channel'], attrs={'units':'m^{-3}','label':'n_e'})
        CO2['CO2']['ne_err'] = xarray.DataArray(ne_err,dims=['time', 'channel'], attrs={'units':'m^{-3}'})
        CO2['CO2']['diags']= xarray.DataArray( np.tile(('CO2 interferometer',), (nt, n_ch)),dims=['time', 'channel'])
        CO2['CO2']['R'] = xarray.DataArray(R[None],dims=['','channel','path'],attrs={'units':'m'})
        CO2['CO2']['Z'] = xarray.DataArray(Z[None],dims=['','channel','path'],attrs={'units':'m'})
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

        CO2['CO2']['rho_tg'] = xarray.DataArray(rho_tg,dims=['time', 'channel'])


        CO2['EQM'] = {'id':id(self.eqm),'dr':0, 'dz':0,'ed':self.eqm.diag}

        print('\t done in %.1fs'%(time()-T))

        return CO2


    
    def co2_correction(self,TS, tbeg,tend):
        
        T = time()
        if not 'core' in TS or not 'tangential' in TS:
            print('CO2 correction could not be done')
            return TS
  
        #BUG ignoring user selected wrong channels of TS 
        T = time()
        
        CO2 = self.load_co2(tbeg,tend, calc_weights = False)['CO2']
        
        print_line( '  * Calculate TS correction using CO2 ...')

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
        tang_lasers_list = np.unique(tang_lasers)
        tang_ne  = TS['tangential']['ne'].values[ind]
        tang_err  = TS['tangential']['ne_err'].values[ind]
        tang_rho = TS['tangential']['rho'].values[ind]
        
        
        core_rho= interp1d(TS['core']['time'],TS['core']['rho'],axis=0,copy=False, assume_sorted=True)(tang_tvec)
        core_ne = interp1d(TS['core']['time'],TS['core']['ne'],axis=0,copy=False, assume_sorted=True)(tang_tvec)
        core_err= interp1d(TS['core']['time'],TS['core']['ne_err'] ,axis=0,copy=False, assume_sorted=True,kind='nearest')(tang_tvec)

        #remove corrupted channels
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
            ne  = np.hstack((tang_ne_,core_ne ))/1e19
            ne  = ne[time_sort_ind, rho_sort_ind]
            val = np.nansum((np.diff(ne)/(ne[:,1:]+ne[:,:-1]+.1))**2)
            return val
            
        from scipy.optimize import minimize 
   
        p0 = np.zeros_like(tang_lasers_list)
        opt = minimize(cost_fun,  p0, tol = .0001 )
        corr = np.exp(opt.x)
 
        
        TS = deepcopy(TS)
        
        #correct core system
        for l,c in zip(tang_lasers_list , corr):
            TS['tangential']['ne'].values[TS['tangential']['laser'].values == l] *= c
   

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
        #interpolate density along LOS for each time 
        LOS_ne = [np.interp(lr,r,n,right=0) for lr, r, n in zip(LOS_rho, R,N)]
        #do line integration
        LOS_ne_int = np.trapz(LOS_ne,LOS_L,axis=-1)
        core_lasers = np.unique(laser_index)
        
    
        #weight = interp1d(CO2['time'].values, CO2['weight'].values*CO2['L_cross'].values[:,:,None] ,copy=False, axis=0,
                #assume_sorted=True)(np.clip(tvec_compare,*CO2['time'].values[[0,-1]]))
              

        co2_los_names  = CO2['channel'].values
        laser_correction     = np.ones((len(core_lasers),len(co2_los_names)))*np.nan
        laser_correction_err = np.ones((len(core_lasers),len(co2_los_names)))*np.nan
        los_dict =  {n:i for i,n in enumerate(co2_los_names)}
        valid = CO2['valid'].values
        time_co2 = CO2['time'].values
        
        #import matplotlib.pylab as plt
        ne = CO2['ne'].values*CO2['L_cross'].values 
        #plt.plot(tvec_compare,np.sum( LOS_ne* weight,2 ) )
        #plt.plot(tvec_compare,LOS_ne_int,'--' )
        #plt.plot(time_co2, ne,':' );plt.show()


        for il, l in enumerate(core_lasers):
            ind = laser_index == l
  
            for ilos,los in enumerate( co2_los_names):
                if not np.any(valid[:,ilos]): continue
                t_ind = (tvec_compare > time_co2[valid[:,ilos]][0]) & (tvec_compare < time_co2[valid[:,ilos]][-1])
                if not np.any(ind&t_ind): continue
                ratio = LOS_ne_int[ind&t_ind ,ilos]/np.interp(tvec_compare[ind&t_ind],time_co2[valid[:,ilos]],ne[valid[:,ilos], ilos])
                laser_correction[il,ilos] = np.median(ratio)
                laser_correction_err[il,ilos] = ratio.std()/np.sqrt(len(ratio))
                    
        mean_laser_correction = np.nanmean(laser_correction,1)
              

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
            
                #co2_los_names
                
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
            data = TS[sys]['ne'].values
            #total correction of all data
            data/= np.mean(mean_laser_correction)
            for l_ind, corr in zip(core_lasers,mean_laser_correction):
                #laser to laser variation from of the core system
                data[laser == l_ind] /= corr/np.mean(mean_laser_correction)
                
                    
        print('\t done in %.1fs'%(time()-T))
        print('\t\tCO2 corrections:\n\t\t', (np.round(mean_laser_correction,3)), ' tang vs. core:', np.round(corr,3))

        return TS 
    
    
        
    
    def load_elms(self,option):
        
        node = option['elm_signal'].get()
        tree = 'SPECTROSCOPY'
        
        self.RAW.setdefault('ELMS',{})
        
        if node in self.RAW['ELMS']:
            return self.RAW['ELMS'][node]

        
        print_line( '  * Fetching ELM data... ')
        T = time()

        try:
            try:
                self.MDSconn.openTree(tree, self.shot)
                elms_sig = self.MDSconn.get('_x=\\'+tree+'::'+node).data()
                elms_tvec = self.MDSconn.get('dim_of(_x)').data()/1e3
            except:
                self.MDSconn.openTree(tree, self.shot)
                elms_sig = self.MDSconn.get('_x=\\'+tree+'::'+node+'da').data()
                elms_tvec = self.MDSconn.get('dim_of(_x)').data()/1e3
        except Exception as e:
            printe( 'MDS error: '+ str(e))
            elms_tvec, elms_sig = [],[]
        finally:
            try:
                self.MDSconn.closeTree(tree, self.shot)
            except:
                pass
        try:
            elm_time, elm_val, elm_beg, elm_end = detect_elms(elms_tvec, elms_sig)
        except Exception as e:
            print('elm detection failed', e)
            elm_time, elm_val, elm_beg, elm_end = [],[],[],[]
            
        self.RAW['ELMS'][node] =  {'tvec': elm_time, 'data':elm_val, 
                     'elm_beg':elm_beg,'elm_end':elm_end,'signal':node}
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
    shot =   169997
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
    shot =     174489

    #175694  - better match between onaxis ver and tang denisty after rescaling
    #TODO nacitat 210 data zvlast
    #edge and core v330L system (178800)
    #TODO v legende se ukazuji i systemy co nemaji zadana data
    #kalibrovat core and edge poloidal zvlast

    print(shot)
    print_line( '  * Fetching EFIT01 data ...')
    eqm = equ_map(MDSconn)
    eqm.Open(shot, 'EFIT01', exp='D3D')

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
                                ('Correction',{'Relative calibration':I(1),'nz from CER intensity':I(1)}  )))   }})
 
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
                                ( 'CO2 interf.',(['fit CO2',I(0)],['rescale TS',I(0)])) ) ),
        'load_options':{'TS system':{"TS revision":(S('BLESSED'),['BLESSED']+ts_revisions)},
                        'Reflectometer':{'Position error':{'Align with TS':I(1) }, }                        
                        }})
        
    settings.setdefault('Zeff', {\
        'systems':OrderedDict(( ( 'VB array',  (['tangential',I(1)],                 )),
                                ( 'CER VB',    (['tangential',I(1)],['vertical',I(1)])),
                                ( 'CER system',(['tangential',I(1)],['vertical',I(1)])))), \
        'load_options':{'VB array':{'Corrections':{'radiative mantle':I(1),'rescale by CO2':I(1), 'remove NBI CX': I(1)}},\
                        'TS':{'Position error':{'Z shift [cm]':D(0.0)}},
                        'CER VB':{'Analysis':(S('auto'), (S('best'),'fit','auto','quick'))},
                        'CER system':OrderedDict((
                                ('Analysis', (S('best'), (S('best'),'fit','auto','quick'))),
                                ('Correction',    {'Relative calibration':I(0)}),
                                ('TS position error',{'Z shift [cm]':D(0.0)})))
                        }\
            })
        

        
    settings['elm_signal'] = S('fs01up')
    settings['elm_signal'] = S('fs04')

    #print settings['Zeff'] 
    
    #exit()

    #TODO 160645 CO2 correction is broken
    #160646,160657  crosscalibrace nimp nefunguje
    #T = time()

    #load_zeff(self,tbeg,tend, options=None)
    data = loader( 'nimp', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])
    data = loader( 'nimp', settings,tbeg=eqm.t_eq[0], tend=eqm.t_eq[-1])

    settings['nimp']={\
        'systems':{'CER system':(['tangential',I(1)], ['vertical',I(1)])},
        'load_options':{'CER system':OrderedDict((
                                ('Analysis', (S('best'), (S('best'),'fit','auto','quick'))),
                                ('Correction',{'Relative calibration':I(1),'nz from CER intensity':I(1)}  )))   }}


    print('\t done in %.1f'%(time()-T))
    #loader.load_elms(settings)
    #print(data)

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
 




