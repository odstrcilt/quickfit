#!/bin/bash
module purge
# module load defaults
cd /fusion/projects/codes/quickfit/
# source /act/etc/profile.d/actbin.sh
# source /act/etc/profile.d/actbin.sh
# module load omfit/unstable
# module load  adas
# module load  python
# module load mdsplus/6.1.84
# module load  fftw/3.3.6-mpich3.2-gcc4.9.2 
# module load ntcc
module load intel/2019
#module load intel
# export MKL_NUM_THREADS=1
# export openblas_get_num_threads=1
# export OMP_NUM_THREADS=1
# export NUM_THREADS=1
# export C_INCLUDE_PATH=$C_INCLUDE_PATH:/fusion/projects/codes/quickfit/SuiteSparse/lib/ 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fusion/projects/codes/quickfit/SuiteSparse/CHOLMOD/Include/
# export C_INCLUDE_PATH=$C_INCLUDE_PATH:/fusion/projects/codes/quickfit/SuiteSparse/include/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fusion/projects/codes/quickfit/SuiteSparse/lib/ 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fusion/projects/codes/quickfit/SuiteSparse/CHOLMOD/Lib/ 
# export C_INCLUDE_PATH=$C_INCLUDE_PATH:/fusion/projects/codes/quickfit/SuiteSparse/CHOLMOD/Lib/ 
# export PYTHONPATH=$PYTHONPATH:/fusion/projects/codes/quickfit/ 

# module load omfit/unstable

echo 'In case of any problems, contact Tomas Odstrcil  odstrcilt@fusion.gat.com'
module load omfit/unstable 

# /fusion/usc/opt/python/2.7.14/bin/python2.7




# module load omfit/unstable
# module load omfit/conda
# alias python='/fusion/usc/opt/python/2.7.11/bin/python2.7'

# 


python3 quickfit.py $@

 #/fusion/usc/opt/python/3.4.3/bin/python3 quickfit.py $argv
#python quickfit.py $argv
# python2.7 quickfit.py $argv
