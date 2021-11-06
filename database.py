#!/bin/bash
filename='C:/Users/odstrcil/Desktop/DIII-D/carbon_database/data1+roto2_unique.txt'
filename='C:/Users/odstrcil/Desktop/DIII-D/carbon_database/data1+roto4_unique.txt'

# echo Start
#while read p; do         
     #IFS=" " read var1 var2  var3<<< $p
     #python3 ./quickfit.py --shot  $var1 --tmin $var2 --tmax $var3
     

#done < $filename

import numpy as np

shots, tmins,tmaxs = np.loadtxt(filename).T


import os

for shot,tmin,tmax in zip(shots, tmins,tmaxs):
    #if shot <=  175902: continue
    os.system(f'python.exe ./quickfit.py --shot  {int(shot)} --tmin {tmin} --tmax {tmax} --preload')
    #os.system(f'python.exe ./quickfit.py --shot  {int(shot)} --tmin {tmin} --tmax {tmax}')

 
