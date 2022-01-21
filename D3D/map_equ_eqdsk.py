__authors__ = 'Tomas Odstrcil, Giovanni Tardini'
__email__ = "git@ipp.mpg.de"
__version__ = '1.1'
__date__ = '14.11.2016'

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import UnivariateSpline, interp1d, InterpolatedUnivariateSpline, LinearNDInterpolator
import sys,os
from scipy import integrate



def get_gc(nshot=None):
    ##shape of the vessel
    try:

        program_path = os.path.abspath(__file__)
        program_path = program_path[:program_path.rfind('/')]+'/'

        R,Z = loadtxt(program_path+'/d3d_elev_030_parseModified.dat').T
    except:

        if nshot < 168847: #just guess of the shot number
            Z = 0.000,0.964,0.968,1.001,1.019,1.077,1.070,1.096,1.113,1.138,1.147,1.165,1.217,\
            1.217,1.162,1.162,1.163,1.165,1.166,1.166,1.169,1.172,1.176,1.183,1.183,1.185,\
            1.188,1.191,1.196,1.202,1.208,1.214,1.221,1.231,1.238,1.244,1.254,1.278,1.290,\
            1.331,1.347,1.348,1.348,1.348,1.348,1.348,1.348,1.310,1.310,1.292,1.095,1.077,\
            1.077,1.040,0.993,0.709,0.519,0.389,0.400,0.222,0.133,0.044,-0.044,-0.133,-0.222,\
            -0.400,-0.389,-0.973,-1.174,-1.211,-1.250,-1.250,-1.250,-1.329,-1.329,-1.363,\
            -1.363,-1.363,-1.223,-1.223,-0.830,-0.800,-0.415,-0.400,-0.001,0.000

            R= 1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.012,1.001,\
            1.029,1.042,1.046,1.056,1.097,1.108,1.116,1.134,1.148,1.162,1.181,1.182,1.185,1.190,\
            1.195,1.201,1.209,1.215,1.222,1.228,1.234,1.239,1.242,1.248,1.258,1.263,1.280,1.280,\
            1.280,1.310,1.328,1.361,1.380,1.419,1.419,1.372,1.372,1.608,1.647,1.785,2.070,2.128,\
            2.245,2.323,2.377,2.362,2.364,2.365,2.365,2.365,2.365,2.364,2.362,2.377,2.134,1.786,\
            1.768,1.768,1.682,1.372,1.372,1.420,1.420,1.273,1.153,1.016,1.016,1.016,1.016,1.016,\
            1.016,1.016,1.016

        else:#SAS diveror
            R = [1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016,
            1.016,1.012,1.001,1.029,1.042,1.046,1.056,1.097,1.108,1.116,
            1.134,1.148,1.162,1.181,1.182,1.185,1.19,1.195,1.201,1.209,
            1.215,1.222,1.228,1.234,1.239,1.242,1.248,1.258,1.263,1.28,
            1.28,1.28,1.31,1.328,1.361,1.38,1.419,1.419,1.372,1.37167001,
            1.37003005,1.36688006,1.36719,1.37178004,1.37223995,
            1.38662004,1.38707995,1.40382004,1.41127002,1.41857004,
            1.421,1.48662996,1.49730003,1.49761999,1.49744999,
            1.49275005,1.49259996,1.49260998,1.49278998,1.49339998,
            1.49469995,1.49621999,1.47981,1.48081994,1.48149002,
            1.48645997,1.49094999,1.50304997,1.59696996,1.62549996,
            1.63751996,1.64699996,1.785,2.07,2.128,
            2.245,2.323,2.377,2.362,2.364,2.365,2.365,2.365,2.365,2.364,
            2.362,2.377,2.134,1.786,1.768,1.768,1.682,1.372,1.372,1.42,
            1.42,1.273,1.153,1.016,1.016,1.016,1.016,1.016,1.016,1.016,1.016]
            Z = [0.0,0.964,0.968,1.001,1.019,1.077,1.070,1.096,1.113,1.138,1.147,
            1.165,1.217,1.217,1.162,1.162,1.163,1.165,1.166,1.166,1.169,1.172,
            1.176,1.183,1.183,1.185,1.188,1.191,1.196,1.202,1.208,1.214,1.221,
            1.231,1.238,1.244,1.254,1.278,1.29,1.331,1.347,1.348,1.348,1.348,
            1.348,1.348,1.348,1.310,1.310,1.29238,1.28268,1.25644,1.22955,
            1.19576,1.19402,1.164870,1.16421,1.15696,1.1573,1.16132,1.164,
            1.2405,1.23458004,1.23428,1.23174,1.2133,1.21061,1.2048,1.20213,
            1.19641,1.1851,1.16069,1.124259,1.12256002,1.12137997,1.11692,
            1.114390,1.11244,1.09489,1.0853,1.07988,1.077,1.077,1.04,0.993,
            0.709,0.519,0.389,0.400,0.222,0.1330,0.044,-0.044,-0.133,-0.222,
            -0.400,-0.389,-0.973,-1.174,-1.211,-1.25,-1.25,-1.25,-1.329,
            -1.329,-1.363,-1.363,-1.363,-1.223,-1.223,-0.823,-0.80,-0.415,-0.400,-0.001,0.0]



    comp_r = {'vessel':R}
    comp_z = {'vessel':Z}
    return comp_r,comp_z


class equ_map:

    source = 'EQDSK'
    system = None

    def __init__(self, eqdsk, debug=False):

        self.eq_open = False
        self.debug = debug
        self.eqdsk = eqdsk


    def Open(self, shot=None, diag='EQDSK', exp='DIII-D',ed=0):

        """
        Input
        ---------
        shot: int
            shot number
        diag: str
            diagnsotics used for mapping (EQI, EQH, ...)
        exp: str
            experiment (AUGD)
        ed:  int
            edition
        """
        self.system = diag
        try:


            # Time grid of equilibrium shotfile
            self.times = np.sort(list(self.eqdsk.keys()))
            self.t_eq  = np.array(self.times)/1000.
            # R, z of PFM cartesian grid, assume the same for all timepoints!!
            self.Rmesh = self.eqdsk[self.times[0]]['AuxQuantities']['R']
            self.Zmesh = self.eqdsk[self.times[0]]['AuxQuantities']['Z']

            self.shot = int(self.eqdsk[self.times[0]]['CASE'][3][2:])
            self.diag =     self.eqdsk[self.times[0]]['CASE'][5]
            self.comment = ''.join(self.eqdsk[self.times[0]]['CASE'])

            #if len(self.t_eq) < 2: raise Exception('too few timepoints in equlibrium')


            self.eq_open = True

        except Exception as e:
            print("loading of equilibrium %s was not successful:  "%diag+ str(e))

        return self.eq_open


    def __del__(self):
        self.Close()


    def Close(self):

        """
        Deleting some attributes, closing dd.shotfile()
        """

        if hasattr(self, 'Rmesh'):
            del self.Rmesh
        if hasattr(self, 'pfm'):
            del self.pfm
        if hasattr(self, 'ffp'):
            del self.ffp
        if hasattr(self, 'psi0'):
            del self.psi0
        if hasattr(self, 'ssq'):
            del self.ssq
        self.eq_open = False



    def _read_pfm(self):

        """
        Reads PFM matrix, only if attribute 'pfm' is missing.
        equ_map.pfm is deleted at any equ_map.Open, equ_map.Close call.
        """

        if not self.eq_open:
            return
        if hasattr(self, 'pfm'):
            return

        if self.debug: print('Reading PFM matrix')

        self.pfm = [self.eqdsk[t]['PSIRZ'].T*2*np.pi for t in self.times]



    def read_ssq(self):

        """
        Creating dictionary map_equ.equ_map().ssq, containing all
        SSQ parameters labelled with the corresponding SSQNAM.
        Beware: in different shotfiles, for a given j0
        SSQ[j0, time] can represent a different variable
        """
        if not self.eq_open:
            return
        if hasattr(self, 'ssq'):
            return

        self.ssq = {}

        self.ssq['Rmag'] = [self.eqdsk[t]['RMAXIS'] for t in self.times]
        self.ssq['Zmag'] = [self.eqdsk[t]['ZMAXIS'] for t in self.times]
        # separatrix = self.sf.get('\\'+self.system+'::TOP.RESULTS.gEQDSK.BDRY').data()
        Rseparatrix = [self.eqdsk[t]['RBBBS'] for t in self.times]
        Zseparatrix = [self.eqdsk[t]['ZBBBS'] for t in self.times]
        self.ssq['Zunt'] = [zs.min() for zs in Zseparatrix]
        self.ssq['Zoben'] = [zs.max() for zs in Zseparatrix]




    def _read_scalars(self):

        """
        Reads R, z, psi at magnetic axis and separatrix, only if attribute 'r0' is missing.
        equ_map.r0 is deleted at any equ_map.Open, equ_map.Close call.
        """

        if not self.eq_open:
            return
        if hasattr(self, 'psi0'):
            return

        self.read_ssq()
        if self.debug: print('Reading scalars')


        # Pol. flux at mag axis, separatrix
        self.psi0 = np.array([self.eqdsk[t]['SIMAG'] for t in self.times])*2*np.pi
        self.psix = np.array([self.eqdsk[t]['SIBRY'] for t in self.times])*2*np.pi
        self.ip = np.array([self.eqdsk[t]['CURRENT']  for t in self.times])
        self.orientation = np.sign(np.median(self.ip))  #current orientation
        self.Bt = np.array([self.eqdsk[t]['BCENTR']  for t in self.times])
        self.R0 = np.array([self.eqdsk[t]['RCENTR']  for t in self.times])
        self.BR = self.Bt/self.R0


    def _read_profiles(self):

        """
        Reads profiles from EQ shotfile, only if attribute 'ffp' is missing.
        equ_map.ffp is deleted at any equ_map.Open, equ_map.Close call.
        """

        if not self.eq_open:
            return
        if hasattr(self, 'ffp'):
            return

        if self.debug: print('Reading Profiles')
        nt = len(self.t_eq)
        self._read_scalars()


        PSIN = np.array([self.eqdsk[t]['AuxQuantities']['PSI_NORM'] for t in self.times]).T


# Profiles
        self.pf  = (PSIN*(self.psix-self.psi0)+self.psi0)
        self.ffp = np.array([self.eqdsk[t]['FFPRIM'] for t in self.times]).T
        self.ppp = np.array([self.eqdsk[t]['PPRIME'] for t in self.times]).T
        self.q   = np.array([self.eqdsk[t]['QPSI'] for t in self.times]).T
        self.pres= np.array([self.eqdsk[t]['PRES'] for t in self.times]).T

        #self.vol = np.array([self.eqdsk[t]['fluxSurfaces']['geo']['vol'] for t in self.times]).T

        #The toroidal flux PHI can be found by recognizing that the safety factor is the ratio of the differential toroidal and poloidal fluxes
        self.tf = integrate.cumtrapz(np.sign(self.ip)*np.sign(self.Bt)*self.q,self.pf,initial=0,axis=0)

        from scipy.constants import mu_0

        self.fpol  = np.array([self.eqdsk[t]['FPOL'] for t in self.times]).T/mu_0*2*np.pi


    def _get_nearest_index(self, tarr):

        tarr = np.minimum(np.maximum(tarr, self.t_eq[0]), self.t_eq[-1])
        idx = interp1d(self.t_eq, np.arange(len(self.t_eq)), kind='nearest')(tarr)
        idx = np.atleast_1d(np.int_(idx))
        unique_idx = np.unique(idx)

        return unique_idx, idx


    def rho2rho(self, rho_in, t_in=None, \
               coord_in='rho_pol', coord_out='rho_tor', extrapolate=False):

        """Mapping from/to rho_pol, rho_tor, r_V, rho_V, Psi, r_a
        r_V is the STRAHL-like radial coordinate

        Input
        ----------
        t_in : float or 1darray
            time
        rho_in : float, ndarray
            radial coordinates, 1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
        coord_in:  str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a','Psi_N']
            input coordinate label
        coord_out: str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a','Psi_N']
            output coordinate label
        extrapolate: bool
            extrapolate rho_tor, r_V outside the separatrix

        Output
        -------
        rho : 2d+ array (nt, nr, ...)
        converted radial coordinate

        """

        if not self.eq_open:
            return
        if self.debug: print(('Remapping from %s to %s' %(coord_in, coord_out)))

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)
        rho = np.atleast_1d(rho_in)

        nt_in = np.size(tarr)

        if rho.ndim == 1:
            rho = np.tile(rho, (nt_in, 1))
            
# Trivial case
        if coord_out == coord_in: 
            return rho

        self._read_scalars()
        self._read_profiles()
        
        unique_idx, idx =  self._get_nearest_index(tarr)

            
        if coord_in in ['rho_pol', 'Psi','Psi_N']:
            label_in = self.pf
        elif coord_in == 'rho_tor':
            label_in = self.tf
        elif coord_in in ['rho_V','r_V']:
            label_in = self.vol
            R0 = self.ssq['Rmag']
        elif coord_in in ['r_a','RMNMP']:
            R, _ = self.rhoTheta2rz(self.pf.T, [0, np.pi], coord_in='Psi')
            label_in = (R[:, 0] - R[:, 1]).T**2/4
        else:
            raise Exception('unsupported input coordinate')


        if coord_out in ['rho_pol', 'Psi','Psi_N']:
            label_out = self.pf
        elif coord_out == 'rho_tor':
            label_out = self.tf
        elif coord_out in ['rho_V','r_V']:
            label_out = self.vol
            R0 = self.ssq['Rmag']
        elif coord_out in ['r_a','RMNMP']:
            R, _ = self.rhoTheta2rz(self.pf.T[unique_idx], [0, np.pi],
                                t_in=self.t_eq[unique_idx],coord_in='Psi')
            label_out = np.zeros_like(self.pf)
            label_out[:,unique_idx] = (R[:, 0] - R[:, 1]).T**2/4
        else:
            raise Exception('unsupported output coordinate')

        PFL  = self.orientation*self.pf
        PSIX = self.orientation*self.psix
        PSI0 = self.orientation*self.psi0

        rho_output = np.ones_like(rho)#*np.nan

   
        for i in unique_idx:
  
# Calculate a normalized input and output flux 
            sort_wh = np.argsort(PFL[:, i])
            #get rid of the point out of the separatrix
            ind = (label_out[sort_wh, i] != 0) & (label_in[sort_wh, i] != 0)
            ind[0] = True
            sort_wh = sort_wh[ind]        

            sep_out, mag_out = np.interp([PSIX[i], PSI0[i]], PFL[sort_wh,i], label_out[sort_wh,i])
            sep_in , mag_in  = np.interp([PSIX[i], PSI0[i]], PFL[sort_wh,i], label_in[sort_wh,i])

            if (abs(sep_out - mag_out) < 1e-4) or (abs(sep_in - mag_in) < 1e-4) or np.isnan(sep_in*sep_out): #corrupted timepoint
                #print 'corrupted'
                continue

# Normalize between 0 and 1
            rho_out = (label_out[sort_wh,i] - mag_out)/(sep_out - mag_out)
            rho_in  = (label_in [sort_wh,i] - mag_in )/(sep_in  - mag_in )

            rho_out[(rho_out > 1) | (rho_out < 0)] = 0  #remove rounding errors
            rho_in[ (rho_in  > 1) | (rho_in  < 0)] = 0

            rho_out = np.r_[np.sqrt(rho_out), 1]
            rho_in  = np.r_[np.sqrt(rho_in ), 1]
            

            ind = (rho_out==0) | (rho_in==0)
            rho_out, rho_in = rho_out[~ind], rho_in[~ind]
            
# Profiles can be noisy!  smooth spline must be used
            sortind = np.unique(rho_in, return_index=True)[1]
            w = np.ones_like(sortind)*rho_in[sortind]
            w = np.r_[w[1]/2, w[1:], 1e3]
            ratio = rho_out[sortind]/rho_in[sortind]
            rho_in = np.r_[0, rho_in[sortind]]
            ratio = np.r_[ratio[0], ratio]
            

            s = UnivariateSpline(rho_in, ratio, w=w, k=4, s=5e-3,ext=3)  #BUG s = 5e-3 can be sometimes too much, sometimes not enought :( 
            


            jt = idx == i
            #print  np.where(jt)[0]
            #if 826 in np.where(jt)[0]:
                #import IPython 
                #IPython.embed()

            rho_ = np.copy(rho[jt])

            r0_in,r0_out = 1,1
            if coord_in == 'r_V' :
                r0_in  = np.sqrt(sep_in/ (2*np.pi**2*R0[i]))
            if coord_out == 'r_V' :
                embed()
                r0_out = np.sqrt(sep_out/(2*np.pi**2*R0[i]))
            if coord_in == 'RMNMP' :
                r0_in  = np.sqrt(sep_in)
            if coord_out == 'RMNMP' :
                r0_out = np.sqrt(sep_out)                
            if coord_in == 'Psi' :
                rho_  = np.sqrt(np.maximum(0, (rho_ - self.psi0[i])/(self.psix[i] - self.psi0[i])))
            if coord_in == 'Psi_N' :
                rho_  = np.sqrt(np.maximum(0, rho_ ))
                

# Evaluate spline

            rho_output[jt] = s(rho_.flatten()/r0_in).reshape(rho_.shape)*rho_*r0_out/r0_in

            if np.any(np.isnan(rho_output[jt])):  # UnivariateSpline failed
                rho_output[jt] = np.interp(rho_/r0_in, rho_in, ratio)*rho_*r0_out/r0_in
                
            if not extrapolate:
                rho_output[jt] = np.minimum(rho_output[jt],r0_out) # rounding errors

            rho_output[jt] = np.maximum(0,rho_output[jt]) # rounding errors

            if coord_out  == 'Psi':
                rho_output[jt]  = rho_output[jt]**2*(self.psix[i] - self.psi0[i]) + self.psi0[i]
            if coord_out == 'Psi_N' :
                rho_output[jt]  = rho_output[jt]**2 

        
        return rho_output
    def rz2brzt(self, r_in, z_in, t_in=None):

        """calculates Br, Bz, Bt profiles

        Input
        ----------
        r_in : ndarray
            R coordinates
            1D, size(nr_in) or 2D, size (nt, nr_in)
        z_in : ndarray
            Z coordinates
            1D, size(nz_in) or 2D, size (nt, nz_in)
        t_in : float or 1darray
            time

        Output
        -------
        interpBr : ndarray
            profile of Br on the grid
        interpBz : ndarray
            profile of Bz on the grid
        interpBt : ndarray
            profile of Bt on the grid

        """

        if not self.eq_open:
            return

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)
        r_in = np.atleast_2d(r_in)
        z_in = np.atleast_2d(z_in)

        self._read_scalars()
        self._read_profiles()
        self._read_pfm()

# Poloidal current

        nt = np.size(tarr)

        if np.size(r_in, 0)!= nt:
            r_in = np.tile(r_in, (nt, 1))
        if np.size(z_in, 0)!= nt:
            z_in = np.tile(z_in, (nt, 1))


        nr_in = np.size(r_in, 1)
        nz_in = np.size(z_in, 1)

        interpBr = np.zeros((nt, nr_in, nz_in),dtype='single')
        interpBz = np.zeros((nt, nr_in, nz_in),dtype='single')
        interpBt = np.zeros((nt, nr_in, nz_in),dtype='single')

        from scipy.constants import mu_0
        nr, nz = len(self.Rmesh), len(self.Zmesh)
        dr = (self.Rmesh[-1] - self.Rmesh[0])/(nr - 1)
        dz = (self.Zmesh[-1] - self.Zmesh[0])/(nz - 1)

        scaling = np.array([dr, dz])
        offset  = np.array([self.Rmesh[0], self.Zmesh[0]])

        unique_idx, idx =  self._get_nearest_index(tarr)

        for i in unique_idx:

            Phi = self.pfm[i]

            Br = -np.diff(Phi, axis=1)/(self.Rmesh[:, None])
            Bz =  np.diff(Phi, axis=0)/(.5*(self.Rmesh[1:] + self.Rmesh[:-1])[:, None])
            Bt = np.interp(Phi, self.pf[:, i], self.fpol[:, i])*mu_0/self.Rmesh[:, None]

            jt = idx == i
            r = np.tile(r_in[jt], (nz_in, 1, 1)).transpose(1, 2, 0)
            z = np.tile(z_in[jt], (nr_in, 1, 1)).transpose(1, 0, 2)

            coords =  np.array((r,z))

            index_t = ((coords.T - offset)/scaling).T
            index_r = ((coords.T - offset)/scaling - np.array((0, .5))).T
            index_z = ((coords.T - offset)/scaling - np.array((.5, 0))).T

            interpBr[jt] = map_coordinates(Br, index_r, mode='nearest', order=2, prefilter=True)
            interpBz[jt] = map_coordinates(Bz, index_z, mode='nearest', order=2, prefilter=True)
            interpBt[jt] = map_coordinates(Bt, index_t, mode='nearest', order=2, prefilter=True)

        return interpBr/(2*np.pi*dz), interpBz/(2*np.pi*dr), interpBt/(2.*np.pi)


    def rz2rho(self, r_in, z_in, t_in=None, coord_out='rho_pol', extrapolate=True):

        """Equilibrium mapping routine, map from R,Z -> rho (pol,tor,r_V,...)
           Fast for a large number of points

        Input
        ----------
        t_in : float or 1darray
            time
        r_in : ndarray
            R coordinates
            1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
        z_in : ndarray
            Z coordinates
            1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
        coord_out: str
            mapped coordinates - rho_pol,  rho_tor, r_V, rho_V, Psi
        extrapolate: bool
            extrapolate coordinates (like rho_tor) for values larger than 1

        Output
        -------
        rho : 2D+ array (nt,nx,...)
        Magnetics flux coordinates of the points

        """

        if not self.eq_open:
            return

        if self.debug: print(('Remapping from {R, z} to %s' %coord_out))

        if t_in is None:
            t_in = self.t_eq
        tarr = np.atleast_1d(t_in)
        r_in = np.atleast_2d(r_in)
        z_in = np.atleast_2d(z_in)

        dr = (self.Rmesh[-1] - self.Rmesh[0])/(len(self.Rmesh) - 1)
        dz = (self.Zmesh[-1] - self.Zmesh[0])/(len(self.Zmesh) - 1)

        nt_in = np.size(tarr)
        if r_in.shape!= z_in.shape:
            raise Exception( 'Not equal shape of z_in and r_in %s,%s'\
                            %(str(z_in.shape), str(z_in.shape)) )

        if np.size(r_in,0) != nt_in and np.size(r_in,0) != 1:
            r_in = r_in[None]
            z_in = z_in[None]

        if np.size(r_in, 0) == 1:
            r_in = np.broadcast_to(r_in, (nt_in,)+r_in.shape[1:]) 
            z_in = np.broadcast_to(z_in, (nt_in,)+z_in.shape[1:]) 
 
        self._read_pfm()
        Psi = np.empty((nt_in,)+r_in.shape[1:], dtype=np.single)

        scaling = np.array([dr, dz])
        offset  = np.array([self.Rmesh[0], self.Zmesh[0]])

        unique_idx, idx =  self._get_nearest_index(tarr)

        for i in unique_idx:
            jt = idx == i
            coords = np.array((r_in[jt], z_in[jt]))
            index = ((coords.T - offset) / scaling).T

            Psi[jt] =  map_coordinates(self.pfm[i], index,
                                mode='nearest',order=2, prefilter=True)





        rho_out = self.rho2rho(Psi, t_in=t_in, extrapolate=extrapolate, coord_in='Psi',
                              coord_out=coord_out)

        return rho_out


    def rho2rz(self, rho_in, t_in=None, coord_in='rho_pol', all_lines=False):

        """Get R, Z coordinates of a flux surfaces contours

        Input
        ----------

        t_in : float or 1darray
            time
        rho_in : 1darray,float
            rho coordinates of the searched flux surfaces
        coord_in: str
            mapped coordinates - rho_pol or rho_tor
        all_lines: bool:
            True - return all countours , False - return longest contour

        Output
        -------
        rho : array of lists of arrays [npoinst,2]
            list of times containg list of surfaces for different rho 
            and every surface is decribed by 2d array [R,Z]

        """

        if not self.eq_open:
            return

        if t_in is None:
            t_in = self.t_eq

        tarr  = np.atleast_1d(t_in)
        rhoin = np.atleast_1d(rho_in)

        self._read_pfm()
        self._read_scalars()

        rho_in = self.rho2rho(rhoin, t_in=t_in, \
                 coord_in=coord_in, coord_out='Psi', extrapolate=True )
        
        try:
            import matplotlib._cntr as cntr
        except: #slower option        
            import matplotlib._contour as _contour
 
        nr = len(self.Rmesh)
        nz = len(self.Zmesh)

        R, Z = np.meshgrid(self.Rmesh, self.Zmesh)
        Rsurf = np.empty(len(tarr), dtype='object')
        zsurf = np.empty(len(tarr), dtype='object')
   
        unique_idx, idx =  self._get_nearest_index(tarr)
        for i in unique_idx:
            jt = np.where(idx == i)[0]

            Flux = rho_in[jt[0]]

# matplotlib's contour creation
 
            try: 
                c = cntr.Cntr(R, Z, self.pfm[:nr, :nz, i].T)
            except: #slower option  
                gen = _contour.QuadContourGenerator(R, Z, self.pfm[:nr, :nz, i].T,np.bool_(Z*0), False, 0)
 
            Rs_t = []
            zs_t = []

            for jfl, fl in enumerate(Flux):
                try:
                    nlist = c.trace(level0=fl, level1=fl, nchunk=0)
                    nlist = nlist[:len(nlist)/2]
                except: #slower option  
                    nlist = gen.create_contour(fl)

                j_ctrs = len(nlist)
                if j_ctrs == 0:
                    if fl == self.psi0[i]:
                        Rs_t.append(np.atleast_1d(self.ssq['Rmag'][i]))
                        zs_t.append(np.atleast_1d(self.ssq['Zmag'][i]))
                    else:
                        Rs_t.append(np.zeros(1))
                        zs_t.append(np.zeros(1))
                    continue
                elif all_lines: # for open field lines
                    line = np.vstack([np.vstack(((np.nan,)*2, l)) for l in nlist[:j_ctrs]])[1:]
                    
                else:  #longest filed line 
                    line = []
                    for l in nlist[:j_ctrs]:
                        if len(l) > len(line):
                            line = l
       
                R_surf, z_surf = list(zip(*line))
                R_surf = np.array(R_surf, dtype = np.float32)
                z_surf = np.array(z_surf, dtype = np.float32)
                if not all_lines:
                    ind = (z_surf >= self.ssq['Zunt'][i])
                    if len(ind) > 1:
                        R_surf = R_surf[ind]
                        z_surf = z_surf[ind]
                Rs_t.append(R_surf)
                zs_t.append(z_surf)
   
            for j in jt:f
    def rho2rz(self, rho_in, t_in=None, coord_in='rho_pol', all_lines=False):

        """Get R, Z coordinates of a flux surfaces contours

        Input
        ----------

        t_in : float or 1darray
            time
        rho_in : 1darray,float
            rho coordinates of the searched flux surfaces
        coord_in: str
            mapped coordinates - rho_pol or rho_tor
        all_lines: bool:
            True - return all countours , False - return longest contour

        Output
        -------
        rho : array of lists of arrays [npoinst,2]
            list of times containg list of surfaces for different rho 
            and every surface is decribed by 2d array [R,Z]

        """

        if not self.eq_open:
            return

        if t_in is None:
            t_in = self.t_eq

        tarr  = np.atleast_1d(t_in)
        rhoin = np.atleast_1d(rho_in)

        self._read_pfm()
        self._read_scalars()

        rho_in = self.rho2rho(rhoin, t_in=t_in, \
                 coord_in=coord_in, coord_out='Psi', extrapolate=True )
        
        try:
            import matplotlib._cntr as cntr
        except: #slower option        
            import matplotlib._contour as _contour
 
        nr = len(self.Rmesh)
        nz = len(self.Zmesh)

        R, Z = np.meshgrid(self.Rmesh, self.Zmesh)
        Rsurf = np.empty(len(tarr), dtype='object')
        zsurf = np.empty(len(tarr), dtype='object')
   
        unique_idx, idx =  self._get_nearest_index(tarr)
        for i in unique_idx:
            jt = np.where(idx == i)[0]

            Flux = rho_in[jt[0]]

# matplotlib's contour creation
 
            try: 
                c = cntr.Cntr(R, Z, self.pfm[i].T)
            except: #slower option  
                gen = _contour.QuadContourGenerator(R, Z, self.pfm[i].T,np.bool_(Z*0), False, 0)
 
            Rs_t = []
            zs_t = []

            for jfl, fl in enumerate(Flux):
                try:
                    nlist = c.trace(level0=fl, level1=fl, nchunk=0)
                    nlist = nlist[:len(nlist)/2]
                except: #slower option  
                    nlist = gen.create_contour(fl)

                j_ctrs = len(nlist)
                if j_ctrs == 0:
                    if fl == self.psi0[i]:
                        Rs_t.append(np.atleast_1d(self.ssq['Rmag'][i]))
                        zs_t.append(np.atleast_1d(self.ssq['Zmag'][i]))
                    else:
                        Rs_t.append(np.zeros(1))
                        zs_t.append(np.zeros(1))
                    continue
                elif all_lines: # for open field lines
                    line = np.vstack([np.vstack(((np.nan,)*2, l)) for l in nlist[:j_ctrs]])[1:]
                    
                else:  #longest filed line 
                    line = []
                    for l in nlist[:j_ctrs]:
                        if len(l) > len(line):
                            line = l
       
                R_surf, z_surf = list(zip(*line))
                R_surf = np.array(R_surf, dtype = np.float32)
                z_surf = np.array(z_surf, dtype = np.float32)
                if not all_lines:
                    ind = (z_surf >= self.ssq['Zunt'][i])
                    if len(ind) > 1:
                        R_surf = R_surf[ind]
                        z_surf = z_surf[ind]
                Rs_t.append(R_surf)
                zs_t.append(z_surf)
   
            for j in jt:
                Rsurf[j] = Rs_t
                zsurf[j] = zs_t

        return Rsurf, zsurf


    def getQuantity(self, rho,var_name, t_in=None, coord_in='rho_pol'):

        """Fast evaluation of any PFL quantity from in the equilibrium shotfile

        Input
        ----------
        rho : ndarray
            rho coordinates (rho_pol or rho_tor),
            1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
        var_name: str\
            name of the quantity, like

            DIII-D
                FFPRIM
                PPRIME
                PRES
                FPOL
                QPSI
                QPSI_RT
                RHOVN


            AUG
                Qpsi       q_value vs PFL
                Bave       <B>vac
                B2ave      <B^2>vac
                Jpol       poloidal current,
                dJpol      dJpol/dpsi
                Pres       pressure
                dPres      dPres/dpsi
                Vol        plasma Volume
                dVol       dVol/dpsi
                Area       plasma Area
                dArea      dArea/dpsi
                FFP        ff'
                Rinv       <1/R>
                R2inv      <1/R^2>
                FTRA       fraction of the trapped particles

        t_in : 1darray
            T coordinates, float, 1D (time constant)

        coord_in:str
            choose 'rho_tor',  'rho_pol', 'rho_V','Psi', 'r_V','r_a'

        Output
        -------
        Quantity : 2d array
         profile at times tin and positions rho
        """

        if not self.eq_open:
            return

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)
        Psi = self.rho2rho(rho, t_in=t_in, coord_in=coord_in,
                          coord_out='Psi', extrapolate=True)
        nt = np.size(self.t_eq)

        Qpsi = array([self.eqdsk[t][var_name] for t in self.times]).T


        var_out = np.zeros_like(Psi,dtype='single')
        unique_idx, idx =  self._get_nearest_index(tarr)

        for i in unique_idx:
            jt = idx == i
            sort_wh = np.argsort(self.pf[:, i])
            if var_name in ['Qpsi','QPSI']:
                ii = Qpsi[sort_wh, i].nonzero()
                s = InterpolatedUnivariateSpline(self.pf[sort_wh[ii], i], Qpsi[sort_wh[ii], i])
            else:
                s = InterpolatedUnivariateSpline(self.pf[sort_wh, i], Qpsi[sort_wh, i])
            var_out[jt] = s(Psi[jt].flatten()).reshape(Psi[jt].shape)

        return var_out


    def cross_surf(self, rho=1, r_in=1.65, z_in=0, theta_in=0, t_in=None, coord_in='rho_pol'):
        """
        Computes intersections of a line with any flux surface.

        Input:
        ----------
        rho: float or 1d array, size(nx)
            coordinate of the desired flux surface
        t_in: float or 1darray
            time point/array for the evaluation
        r_in: float
            R position of the point
        z_in: float
            z position of the point
        theta_in: float
            angle of the straight line with respect to horizontal-outward
        coord_in:  str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
            input coordinate label

        Output:
        ----------
        Rout: 3darray size(nt, nx, 2)
            R-position of intersections
        zout: 3darray size(nt, nx, 2)
            z-position of intersections
        """

        if not self.eq_open:
            return

        self._read_scalars()

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)
        rho  = np.atleast_1d(rho)
        r_in = np.float32(r_in)
        z_in = np.float32(z_in)

        unique_idx, idx = self._get_nearest_index(tarr)

        line_m = 3. # line length: 6 m
        n_line = int(200*line_m) + 1 # 1 cm, but then there's interpolation!
        t = np.linspace(-line_m, line_m, n_line, dtype=np.float32)

        line_r = t*np.cos(theta_in) + r_in
        line_z = t*np.sin(theta_in) + z_in

        rho_line = self.rz2rho(line_r, line_z, self.t_eq[unique_idx],
                               coord_out=coord_in, extrapolate=True)
        if coord_in == 'Psi':
            rho_line *= self.orientation
            rho      *= self.orientation
        nt_in = len(tarr)
        nrho  = len(rho)
        Rout = np.zeros((nt_in, nrho, 2), dtype=np.float32)
        zout = np.zeros((nt_in, nrho, 2), dtype=np.float32)
        for i, ii in enumerate(unique_idx):
            jt = idx == i
            for jrho, fl in enumerate(rho):
                ind_gt = (rho_line[i] > fl)
                ind_cross = [j for j, x in enumerate(np.diff(ind_gt)) if x]
                pos = 0
                for j in ind_cross:
                    if pos > 1:
                        zout[jt, jrho, :] = 0
                        Rout[jt, jrho, :] = 0
                        break
                    if rho_line[i, j + 1] > rho_line[i, j]:
                        ind = [j, j + 1]
                    else:
                        ind = [j + 1, j]
                    ztmp = np.interp(fl, rho_line[i, ind], line_z[ind])
                    if ztmp >= self.ssq['Zunt'][ii] and ztmp <= self.ssq['Zoben'][ii]:
                        zout[jt, jrho, pos] = ztmp
                        Rout[jt, jrho, pos] = np.interp(fl, rho_line[i, ind], line_r[ind])
                        pos += 1

        return Rout, zout


    def rhoTheta2rz(self, rho, theta_in, t_in=None, coord_in='rho_pol', n_line=201):

        """
        This routine calculates the coordinates R, z of the intersections of
        a ray starting at the magnetic axis with fluxsurfaces
        at given values of some radial coordinate.
        (slower than countours)

        Input:
        ----------
        rho: float or 1D array (nr) or nD array (nt,nr,...)
            coordinate of the desired flux surface inside LCFS!
        t_in: float or 1darray
            time point/array for the evaluation

        theta_in: float or 1D array n_theta
            angle of the straight line with respect to horizontal-outward, in radians!!

        coord_in:  str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
            input coordinate label

        Output:
        ----------
        R,  z: 3d+ array size(nt, n_theta, nr,...)
        """

        if not self.eq_open:
            return

        self._read_scalars()

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)

        nt_in = len(tarr)

        rho  = np.atleast_1d(rho)
        if rho.ndim == 1:
            rho = np.tile(rho, (nt_in, 1))

        theta_in = np.atleast_1d(theta_in)[:, None]
        ntheta = len(theta_in)

        unique_idx, idx = self._get_nearest_index(tarr)

#        n_line = 201 <=> 5 mm, but then there's interpolation!

        line_r = np.empty((len(unique_idx), ntheta, n_line))
        line_z = np.empty((len(unique_idx), ntheta, n_line))

        line_m = 1.4 # line length: 0.9 m
        t = np.linspace(0, 1, n_line)**.5*line_m
        c, s = np.cos(theta_in), np.sin(theta_in)


        tmpc = c*t
        tmps = s*t
        for i, ii in enumerate(unique_idx):
            line_r[i] = tmpc + self.ssq['Rmag'][ii]
            line_z[i] = tmps + self.ssq['Zmag'][ii]


        rho_line = self.rz2rho(line_r, line_z, self.t_eq[unique_idx],
                               coord_out=coord_in , extrapolate=True)

        R = np.empty((nt_in, ntheta) + rho.shape[1:], dtype='single')
        z = np.empty((nt_in, ntheta) + rho.shape[1:], dtype='single')



        if coord_in == 'Psi':
            rho_line[:,:,0] = self.psi0[unique_idx][:,None]
            rho_line *= self.orientation
            rho      *= self.orientation
        else:
            #solve some issues very close to the core
            rho_line[:,:,0] = 0

        for i, ii in enumerate(unique_idx):
            jt = idx == ii
            for k in range(ntheta):

                monotonicity = np.cumprod(np.ediff1d(rho_line[i, k],1)>0)==1  #troubles with IDE
                imax = np.argmax(rho_line[i, k, monotonicity])

                rspl = InterpolatedUnivariateSpline(rho_line[i, k, :imax+1],
                                                    line_r[i, k, :imax+1], k=2)

                rho_ = np.minimum(rho[jt].flatten(),rho_line[i, k, imax])  #avoid extrapolation
                R[jt, k] = rspl(rho_).reshape(rho[jt].shape)

                zspl = InterpolatedUnivariateSpline( \
                       rho_line[i, k, :imax+1], line_z[i, k, :imax+1], k=2)
                z[jt, k] = zspl(rho_).reshape(rho[jt].shape)

        return R, z


    def mag_theta_star(self, t_in, n_rho=400, n_theta=200, rz_grid=False ):

        """
        Computes theta star

        Input:
        ----------
        t_in: float
            time point for the evaluation
        n_rho: int
            number of flux surfaces equaly spaced from 0 to 1 of rho_pol
        n_theta: int
            number of poloidal points
        rz_grid: bool
            evaluate theta star on the grid

        Output:
        ----------
        R, z, theta: 3d arrays size(n_rho, n_theta)


        """
        rho = np.linspace(0, 1, n_rho+1)[1:]
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        magr, magz = self.rhoTheta2rz(rho, theta, t_in=t_in, coord_in='rho_pol')
        magr, magz = magr[0].T, magz[0].T

        r0 = np.interp(t_in, self.t_eq, self.ssq['Rmag'])
        z0 = np.interp(t_in, self.t_eq, self.ssq['Zmag'])

        drdrho, drtheta = np.gradient(magr)
        dzdrho, dztheta = np.gradient(magz)
        dpsidrho, dpsitheta = np.gradient(np.tile(rho**2, (n_theta, 1)).T )

        grad_rho = np.dstack((drdrho, dzdrho, dpsidrho ))
        grad_theta = np.dstack((drtheta, dztheta, dpsitheta))
        normal = np.cross(grad_rho, grad_theta, axis=-1)

        dpsi_dr = -normal[:, :, 0]/(normal[:, :, 2] + 1e-8) #Bz
        dpsi_dz = -normal[:, :, 1]/(normal[:, :, 2] + 1e-8) #Br

#WARNING not defined on the magnetics axis

        dtheta_star = ((magr - r0)**2 + (magz - z0)**2)/(dpsi_dz*(magz - z0) + dpsi_dr*(magr - r0))/magr
        theta = np.arctan2(magz - z0, - magr + r0)

        theta = np.unwrap(theta - theta[:, (0, )], axis=1)

        from scipy.integrate import cumtrapz

# Definition of the theta star by integral
        theta_star = cumtrapz(dtheta_star, theta, axis=1, initial=0)
        correction = (n_theta - 1.)/n_theta

        theta_star/= theta_star[:, (-1, )]/(2*np.pi)/correction  #normalize to 2pi






        if not rz_grid:
            return magr, magz, theta_star

# Interpolate theta star on a regular grid
        cos_th, sin_th = np.cos(theta_star), np.sin(theta_star)
        Linterp = LinearNDInterpolator(np.c_[magr.ravel(),magz.ravel()], cos_th.ravel(),0)

        nx = 100
        ny = 150

        rgrid = np.linspace(magr.min(), magr.max(), nx)
        zgrid = np.linspace(magz.min(), magz.max(), ny)

        R,Z = np.meshgrid(rgrid, zgrid)
        cos_grid = Linterp(np.c_[R.ravel(), Z.ravel()]).reshape(R.shape)
        Linterp.values[:, 0] = sin_th.ravel() #trick save a some  computing time
        sin_grid = Linterp(np.c_[R.ravel(), Z.ravel()]).reshape(R.shape)

        theta_star = np.arctan2(sin_grid, cos_grid)

        return rgrid, zgrid, theta_star

#__name__ = "__main__"
if __name__ == "__main__":
    print('done')

    from time import time

    import matplotlib.pylab as plt



    root = OMFIT['EFITtime']['OUTPUTS']['gEQDSK']
    eqm = equ_map(root,debug=False)

    eqm.Open(167490,diag='EFIT02')

    t = 5.33,#s
    color = 'b','g'
    rho = np.linspace(0,2,101)

    R,Z = eqm.rho2rz( rho, t, all_lines=True)

    for r,z,c in zip(R,Z,color):
        for rc,zc in zip(r,z):
            plt.plot(rc,zc,c)


    R,Z = eqm.rhoTheta2rz(   np.linspace(0,1,51),np.linspace(-np.pi,np.pi,1000),  t)

    plt.plot(R[0],Z[0],'r')


    Rv,Zv = get_gc()
    for key in list(Rv.keys()):
        plt.plot(Rv[key],Zv[key],'k')
    plt.axis('equal')

    plt.show()
    plt.figure()



    eqm._read_pfm()
    eqm._read_profiles()
    eqm._read_scalars()


    rho = np.linspace(0,0.98,100)
    q = eqm.getQuantity(rho, 'QPSI', 2)[0]

    plt.plot(rho,q)
    plt.show()


 
    br, bz, bt = eqm.rz2brzt(r_in=eqm.R0[:,None], z_in=0, t_in=eqm.t_eq)
    plt.figure()
    plt.plot(eqm.t_eq,np.squeeze( bt))
    plt.plot(eqm.t_eq, np.squeeze(eqm.Bt))


    R = np.linspace(1, 2.2, 100)
    Z = np.linspace(0, 0, 100)
    rhop = np.linspace(0, 1, 100)


    #br, bz, bt = eqm.rz2brzt(r_in=R, z_in=Z, t_in=3)
    #plt.figure(num="Br")

    #plt.pcolor(R,Z,np.squeeze(br).T)
    ##plt.show()
    #plt.figure(num="Bz")
    #plt.pcolor(R,Z,np.squeeze(bz).T)
    #plt.figure(num="Bt")
    #plt.pcolor(R,Z,np.squeeze(bt).T)

    plt.show()

    r1, z1 = eqm.rhoTheta2rz([1,], np.linspace(0,2*np.pi,20), t_in=3, coord_in='rho_pol', n_line=1000)

    r, z = eqm.rhoTheta2rz(rhop, 0, t_in=3)

    plt.figure(1)
    plt.plot(rhop, r[0, 0])
    plt.xlabel('rho_pol')
    plt.ylabel('R')
    plt.title('kkrhorz')

    rho = eqm.rz2rho( R,Z, 3)

    plt.figure(2)
    plt.plot(R, rho[0])
    plt.xlabel('R')
    plt.ylabel('rho_pol')
    plt.title('kkrzptfn')

#------

    rhot = eqm.rho2rho(rhop, 3, coord_in='rho_pol', coord_out='r_a')

    plt.figure(3)
    plt.plot(rhop, rhot[0])
    plt.xlabel('rho_pol')
    plt.ylabel('rho_tor')
    plt.title('kkrhopto')

#------

    br, bz, bt = eqm.rz2brzt(r_in=R, z_in=Z[0], t_in=3)

    plt.figure(4)
    plt.subplot(1, 3, 1)
    plt.ylabel('Br')
    plt.plot(R, br[0, :, 0])
    plt.xlabel('R')
    plt.suptitle('kkrzBrzt')

    plt.subplot(1, 3, 2)
    plt.plot(R, bz[0, :, 0])
    plt.ylabel('Bz')
    plt.xlabel('R')

    plt.subplot(1, 3, 3)
    plt.plot(R, bt[0, :, 0])
    plt.ylabel('Bt')
    plt.xlabel('R')

#------

    rho = [0.1, 1]
    r2, z2 = eqm.rho2rz(rho, t_in=3, coord_in='rho_pol')
    n_theta = len(r2[0][0])
    theta = np.linspace(0, 2*np.pi, 1000)
    r1, z1 = eqm.rhoTheta2rz(rho, theta, t_in=3, coord_in='rho_pol', n_line=1000)

    plt.figure(5, figsize=(10, 12))

    plt.axes().set_aspect('equal')
    plt.plot(r2[0][0], z2[0][0], 'go')
    plt.plot(r1[0, :, 0], z1[0, :, 0], 'r+')
    R,Z = np.meshgrid(eqm.Rmesh, eqm.Zmesh)
    plt.plot(R,Z,",k")

    #plt.figure(6, figsize=(10, 12))

    plt.axes().set_aspect('equal')
    plt.plot(r2[0][1], z2[0][1], 'go')
    plt.plot(r1[0, :, 1], z1[0, :, 1], 'r+')


    rgrid, zgrid, theta_star = eqm.mag_theta_star(3 )

    plt.figure()
    plt.pcolor(rgrid, zgrid,theta_star)
    plt.show()



    plt.show()
