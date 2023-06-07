#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import os,time,sys
import matplotlib
matplotlib.use('TkAgg')
import MDSplus
import signal

#if matplotlib.compare_versions(matplotlib.__version__, '1.9.9'):
# http://matplotlib.org/users/dflt_style_changes.html
params = {'legend.fontsize': 'large',
        'axes.labelsize': 'large',
        'axes.titlesize': 'large',
        'xtick.labelsize' :'medium',
        'ytick.labelsize': 'medium',
        'font.size':12,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        'grid.color': 'k',
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'lines.linewidth'   : 1.0,
            'lines.dashed_pattern' : (6, 6),
            'lines.dashdot_pattern' : (3, 5, 1, 5),
            'lines.dotted_pattern' : (1, 3),
            'lines.scale_dashes': False,
            'errorbar.capsize':3,
            'mathtext.fontset': 'cm',
            #'text.antialiased':False,
            'mathtext.rm' : 'serif',
            'savefig.dpi': 300}

#plt.rcParams['savefig.dpi'] = 300

matplotlib.rcParams.update(params)

 #plt.rcParams['text.antialiased'] = False

import numpy as np
import tkinter as tk
import tkinter.ttk,tkinter.messagebox,tkinter.filedialog
from tooltip import createToolTip
from collections import OrderedDict
from copy import deepcopy 


from interactive_plot import FitPlot, transformations
import traceback

#print('Warning": all=raise')
#np.seterr(all='raise')
from IPython import embed


#AUG naming convention
labels = { 
  'ne'   :{'pre':'N','dlab':'NE'  ,'TRunit':'[cm^-3]','TRfac':1e-6}, 
  'Te'   :{'pre':'E','dlab':'TE'  ,'TRunit':'[eV]'   ,'TRfac':1   }, 
  'Ti'   :{'pre':'I','dlab':'TI'  ,'TRunit':'[eV]'   ,'TRfac':1   }, 
  'omega':{'pre':'V','dlab':'OMG' ,'TRunit':'[rad/s]','TRfac':1   }, 
  'Zeff' :{'pre':'Z','dlab':'ZEFF','TRunit':' '      ,'TRfac':1   }, 
  'nimp' :{'pre':'C','dlab':'V_C' ,'TRunit':'[cm^-3]','TRfac':1e-6}, 
  'Mach' :{'pre':'M','dlab':'MACH','TRunit':'[-]'    ,'TRfac':1   }, 
  }


#DIII-D naming convention
labels = { 
  'ne'   :{'ext':'NEL' ,'pre':'OMF','dlab':'NE'  ,'TRunit':'[cm^-3]','TRfac':1e-6}, 
  'Te'   :{'ext':'TEL' ,'pre':'OMF','dlab':'TE'  ,'TRunit':'[eV]'   ,'TRfac':1   }, 
  'Ti'   :{'ext':'TIO' ,'pre':'OMF','dlab':'TI'  ,'TRunit':'[eV]'   ,'TRfac':1   }, 
  'omega':{'ext':'OME' ,'pre':'OMF','dlab':'OMG' ,'TRunit':'[rad/s]','TRfac':1   }, 
  'Zeff' :{'ext':'ZEF' ,'pre':'OMF','dlab':'ZEFF','TRunit':' '      ,'TRfac':1   }, 
  'nimp' :{'ext':'NC'  ,'pre':'OMF','dlab':'V_C' ,'TRunit':'[cm^-3]','TRfac':1e-6}, 
  'Mach' :{'ext':'MACH','pre':'OMF','dlab':'MACH','TRunit':'[-]'    ,'TRfac':1   }, 
  }




icon_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'icons','') 


def printe(message):
    CSI="\x1B["
    reset=CSI+"m"
    red_start = CSI+"31;40m"
    red_end = CSI + "0m" 
    print(red_start,message,red_end)
    

class DataFit():

    fsize=12

    ts_revisions = []
    fit_options = {'dt':''}
    options = {'eta':.5, 'lam':.5}

    kinprof_ind = None
    default_elms_signal = 'fs04'
    sawteeth = []
    elms = []

    
    def __init__(self, main_frame, MDSserver,device='D3D', shot=None,OMFITsave=None,eqdsk=None,
                 raw_data={},settings=OrderedDict(),coordinate='rho_tor', elmstime=False, elmsphase=False):
        
  
        print('Accesing data from %s tokamak'%device)
        
        self.main_frame=main_frame
 
        #MDS connection or server name used to access data
        self.MDSserver = MDSserver
        #save function from OMFITgui
        self.OMFITsave = OMFITsave
        self.coordinate = coordinate
        self.tstep = None
        self.eqdsk = eqdsk
        self.device = device
        #dict or OMFITtree from OMFIT
        self.raw_data = raw_data
        #dict or OMFITtree from OMFIT
        self.default_settings = deepcopy(settings)
        
        self.elmstime = elmstime
        self.elmsphase = elmsphase
        
        if device == 'DIII-D': self.device = 'D3D'
        if self.device == 'D3D':
            from D3D import fetch_data 
            from D3D.map_equ import equ_map
        elif self.device == 'CMOD': 
            from CMOD import fetch_data 
            from CMOD.map_equ import equ_map
        elif self.device == 'AUG': 
            from AUG import fetch_data 
            from AUG.map_equ import equ_map
        elif self.device in ['NSTX','NSTXU']: 
            self.device = 'NSTX'
            from NSTX import fetch_data 
            from NSTX.map_equ import equ_map
                
        else:
            raise Exception('Device "%s" was not implemented yet'%self.device)

        self.equ_map = equ_map

        from tkinter import   font

        # Creating a Font object of "TkDefaultFont"
        self.defaultFont = font.nametofont("TkDefaultFont")
        # Overriding default-font with custom settings
        # i.e changing font-family, size and weight
        self.defaultFont.configure(family="Segoe UI" )
        if self.defaultFont.actual()['family'] != 'Segoe UI':
            self.defaultFont.configure(family="Helvetica" )
            if self.defaultFont.actual()['family'] !=  "Helvetica" :
                self.defaultFont.configure(family="Liberation Sans" )
                if self.defaultFont.actual()['family'] !=  "Liberation Sans" :
                    print('Issue with TK fonts, use ', self.defaultFont.actual()['family'] )

        
        self.data_loader_class = fetch_data.data_loader
        self.default_settings_loader = fetch_data.default_settings
            
         
        self.fit_frame = tk.LabelFrame(self.main_frame, text="Data Fit", padx=0, pady=5,relief='groove')
        self.fit_frame.pack(side=tk.LEFT,fill=tk.BOTH, expand=1 , padx=5)
        
        self.fitPlot = FitPlot(self, self.fit_frame)
        
        self.options_frame = tk.Frame(self.main_frame, padx=0, pady=0, width = 200)
        self.options_frame.pack(side=tk.RIGHT,fill=tk.Y, expand=0)
        
        self.eq_frame = tk.LabelFrame(self.options_frame, text="Load EFIT", padx=2, pady=2,relief='groove')
        self.eq_frame.pack(side=tk.TOP,fill=tk.BOTH)
     
        self.data_frame = tk.LabelFrame(self.options_frame, text="Load data", padx=2, pady=2,relief='groove' )
        self.data_frame.pack(side=tk.TOP,fill=tk.BOTH,expand=1 )
     
        self.fit_opt_frame = tk.LabelFrame(self.options_frame, text="Fit options", padx=2, pady=2,relief='groove' )
        self.fit_opt_frame.pack(side=tk.TOP,fill=tk.BOTH)
     
        self.save_frame = tk.LabelFrame(self.options_frame, text="Save fit", padx=2, pady=2,relief='groove')
        self.save_frame.pack(side=tk.TOP,fill=tk.BOTH )

        #self.MDSconn = None
        self.shot = None

        self.spline_fits = {}

        self.init_data_frame()
         
        self.load_default_options(len(settings) == 0)

        self.init_eq_frame()

        self.fitPlot.init_fit_frame()
        self.init_save_frame()

        self.set_trange(0,7)
        self.fitPlot.changed_fit_slice()
        self.fitPlot.plot_step()
        self.saved_profiles = False
        



        #load given dicharge (if any)
        if shot is not None: 
            self.shot_entry.insert(0,shot)

    def __del__(self):        
        try:
            if isinstance(self.MDSserver,str):
                self.MDSconn.disconnect()
        except:
            pass
    
    def isfloat(self,num):
        #sanity check for input numbers
        if num == '':
            return True
        
        try:
            float(num) 
            return True
        except:
            return False
        
    def connectMDSplus(self):
        
        if isinstance(self.MDSserver,str):
            #embed()
            try:
                self.MDSconn = MDSplus.Connection(self.MDSserver)
            except:
                try:
                    assert self.MDSserver != 'localhost'
                    self.MDSconn = MDSplus.Connection('localhost')
                except:
                    #print 'MDS connection to %s failed'%self.MDSserver
                    tkinter.messagebox.showerror('MDS error','MDS connection to %s  and localhost failed'%self.MDSserver)
                    self.main_frame.config(cursor="")       
                    return False
            
        else:
            self.MDSconn = self.MDSserver
        return True
        
    def check_shot_num(self,num):
        if num == '':
            return True
        
        num = num.strip()
        if not num.isnumeric():
            return False

        try:
            shot = int(num) 
        except:
            return False
 
        if shot == self.shot:
            return True
        
        if self.device in ['D3D','NSTX']:
            ndig = 6
        elif self.device == 'CMOD':
            ndig = 10
        elif self.device == 'AUG':
            ndig = 5
       
        if len(num) == ndig :
            if self.shot is not None:
                #clean cache
                print('Shot changed to %d, cleaning cache'%shot)
                self.raw_data.clear()
            else:
                print('Loading shot %d'%shot)
                
                
            self.shot = shot 
            self.fitPlot.shot = shot
            #load avalible efit editions
            def handler(signum, frame): raise Exception('MDS connection is broken')

            self.main_frame.config(cursor="watch")
            self.main_frame.update()
       
            connected = self.connectMDSplus()
            if not connected:
                return False

            self.eqm = self.equ_map(self.MDSconn)

            efit_editions = []
            
            #get a list of availible EFIT editions
            if self.device == 'D3D':
                #BUG is there a better way how to access all EFITS? 
                try:
                    self.MDSconn.openTree('MHD', self.shot)
                    efit_editions = self.MDSconn.get('getnci(".EFIT**.*","path")').data()
                    self.MDSconn.closeTree('MHD', self.shot)
                    assert  len(efit_editions) > 0, 'error efitedit '+ efit_editions
                except:
                    efit_editions = []
                try:
                    if not isinstance(efit_editions[0],str):
                        efit_editions = [e.decode() for e in efit_editions]
                except:
                    pass
        
                efit_editions = [e.strip().split(':')[0] for e in efit_editions]
                efit_editions = [e[1:] for e in efit_editions if 'EFIT' in e]
                efit_editions+= [ 'EFITRT1','EFITRT2' ]
                

            if self.device == 'NSTX':
                efit_names = []
                for tree in ['EFIT','LRDFIT']:
                    try:
                        self.MDSconn.openTree(tree, self.shot)
                        
                        TDI = rf'_y = getnci("\\{tree}::TOP.*.RESULTS","minpath");'
                        TDI+= rf'_s = getnci("\\{tree}::TOP.*.RESULTS.GEQDSK:GTIME","length") > 0;'
                        TDI+= r'PACK(_y,_s)'
        
                        efit_names += list(self.MDSconn.get(TDI).data())
                        self.MDSconn.closeTree(tree, self.shot)
                    except:
                        continue

                if len(efit_names) and not isinstance(efit_names[0],str):
                    efit_names = [e.decode() for e in efit_names] 
 
                efit_editions = [n.split('.')[1] for n in efit_names]


         
            efits  = []
            if self.device == 'CMOD':#for cmod
                efits = ['ANALYSIS','EFIT20']+['EFIT%.2d'%i for i in range(1,10)] 
            if self.device == 'D3D':#for D3D
                efits = ['EFIT02er','EFITS1','EFITS2','EFITS2er']+['EFIT%.2d'%i for i in range(1,10)] 
  
            
            for efit in efits:
                if efit in efit_editions: continue
                try:
                    self.MDSconn.openTree(efit, self.shot)
                    self.MDSconn.closeTree(efit, self.shot)
                except:
                    continue
                efit_editions+= [efit]                    

            efit_editions = np.unique(efit_editions).tolist()
            
            if self.eqdsk is not None:
                efit_editions = ['EQDSK']+efit_editions
            
            if len(efit_editions) == 0:
                raise Exception('No EFITs were found. MDS+ issue? ')
            self.efit_combo['values'] = efit_editions
            self.efit_combo.current(0)
            
            if self.eqdsk is None:
                if 'EFIT' in self.default_settings:
                    pref_ef = self.default_settings['EFIT']
                else:
                    prefered_efit = 'LRDFIT09', 'ANALYSIS','EFIT20','EFIT01', 'EFIT02',    'EFIT03',  'EFIT04', efit_editions[0]  
                    if  self.device == 'NSTX':
                        prefered_efit =  'LRDFIT09', 'LRDFIT06', 'EFIT02', 'EFIT01'
                    for pref_ef in prefered_efit:
                        if pref_ef in efit_editions:
                            break
      
                self.efit_combo.set(pref_ef)

            self.efit_combo.configure(state=tk.NORMAL)

            self.BRIEF = ''
            if self.device == 'D3D':                    
                try:

                    self.MDSconn.openTree('D3D', shot)
                    self.BRIEF = self.MDSconn.get(r'\D3D::TOP.COMMENTS:BRIEF').data() #slow
                    self.MDSconn.closeTree('D3D', shot)
                    if not isinstance(self.BRIEF,str):
                        self.BRIEF = self.BRIEF.decode()
                    print(self.BRIEF)
                except:
                    print('Shot comment in MDS+ does not exist')
                  
                #load default filterscope for elm identification
                try:
                    self.MDSconn.openTree('PEDESTAL', shot)
                    self.default_elms_signal = self.MDSconn.get(r'\PEDESTAL::TOP.ELM:ELMDANAME').data()
                    self.default_elms_signal = self.default_elms_signal.decode()
                except:
                    pass
                

            #use a default setting for a new discharge
            self.load_default_options()
            self.init_fit_opt_frame()
            if hasattr(self,'splines'):
                del self.splines
            self.show_splines.set(0)

            self.efit_edition_changed()
            self.init_set_prof_load()
            self.main_frame.config(cursor="")   

        elif len(num) > ndig:
            return False

        return True
    
    def efit_edition_changed(self, event=None):
        
        #embed()
        efit = self.efit_combo.get()
        if efit == self.eqm.system:
            return True

        self.eqm.Close()

        if efit == 'EQDSK':
            if self.device == 'D3D': 
                from D3D.map_equ_eqdsk import equ_map as equ_map_eqdsk
            elif self.device == 'CMOD': 
                from CMOD.map_equ_eqdsk import equ_map as equ_map_eqdsk
            elif self.device == 'NSTX': 
                from NSTX.map_equ_eqdsk import equ_map as equ_map_eqdsk
            else:
                raise Exception(self.device + ' does ot support loading of eqdsk, yet')
            self.eqm = equ_map_eqdsk(self.eqdsk)
            self.eqm.Open()
       
        else:
            if self.eqm.source == 'EQDSK':
                #create a new eqm connected to MDS+
                self.eqm = self.equ_map(self.MDSconn)

            try:
                assert self.eqm.Open(self.shot, efit, exp=self.device), 'EFIT loading problems'
            except:
                tkinter.messagebox.showerror('Loading problems','EFIT '+efit+' could not be loaded, try another edition')
                return False
            
        self.efit_description.config(text=getattr(self.eqm, 'comment', '')[:50])

        if self.kinprof_ind  != -1:   
            self.data_load.config(stat=tk.NORMAL)
        

        tbeg,tend = self.eqm.t_eq[[0,-1]]
        for prof in self.kin_profs:
            self.load_options[prof]['trange'] = tbeg,tend,None
        
        self.set_trange(tbeg,tend)
        
        
        #if avalible use existing dataloader with 
        if hasattr(self,'data_loader') and self.shot == self.data_loader.shot:
            self.data_loader.eqm = self.eqm
            #realod all already loaded profiles?
            self.kinprof_ind = None
            for prof in self.kin_profs:
                self.load_options[prof]['options']['data_loaded'] = False
            
        else:
            self.data_loader = self.data_loader_class(self.MDSconn, self.shot, self.eqm, self.options['rho_coord'], self.raw_data)

            
        self.fitPlot.m2g = None

        self.fitPlot.ax_main.cla()
        self.fitPlot.ax_main.figure.canvas.draw()
        
        return True
            

    def init_eq_frame(self):
        
        
        self.eq_up_frame = tk.Frame(self.eq_frame, padx=2, pady=2)
        self.eq_up_frame.pack(side=tk.TOP,fill=tk.BOTH, expand=tk.Y , padx=5)
        
        self.eq_down_frame = tk.Frame(self.eq_frame, padx=2, pady=2)
        self.eq_down_frame.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=tk.Y , padx=5)
        
        self.eq_shot_frame = tk.Frame(self.eq_up_frame, padx=2, pady=2)
        self.eq_shot_frame.pack(side=tk.LEFT , expand=tk.Y , padx=5)
        
        self.eq_reviz_frame = tk.Frame(self.eq_up_frame, padx=2, pady=2)
        self.eq_reviz_frame.pack(side=tk.RIGHT , expand=tk.Y , padx=2)
        
        shot_lbl = tk.Label( self.eq_shot_frame,text='Shot: ')
        shot_lbl.pack(side=tk.LEFT) 
        
        vcmd =  self.eq_frame.register(self.check_shot_num) 
        
        shot_width = 10 if self.device == 'CMOD' else 7
        self.shot_entry = tk.Entry(self.eq_shot_frame ,
                                   validate="key",width=shot_width,validatecommand=(vcmd, '%P'))
        self.shot_entry.pack(side=tk.RIGHT) 
        #self.check_shot_num()
        edition_lbl = tk.Label( self.eq_reviz_frame,text='Edition:')
        edition_lbl.pack(side=tk.LEFT, anchor='e') 
        
        vcmd2 =  self.eq_frame.register(self.efit_edition_changed) 
    
        self.efit_combo = tkinter.ttk.Combobox(self.eq_reviz_frame,width=7,state='normal',
                                    validatecommand=(vcmd2, '%P'),validate="focusout")
        self.efit_combo.pack(side=tk.RIGHT) 
        self.efit_combo.bind("<<ComboboxSelected>>", self.efit_edition_changed)

        descr_frame = tk.LabelFrame(self.eq_down_frame, padx= 0, pady=0, relief='groove')
        descr_frame.pack(side=tk.LEFT,fill=tk.BOTH, expand=tk.Y, pady=0, padx=0)

        self.efit_description = tk.Label(descr_frame,text='', font=("Courier", 7))
        self.efit_description.pack( side=tk.LEFT,fill=tk.BOTH, expand=tk.Y, pady=0, padx=0)
        


    def init_data_frame(self):

        self.dataset_frame = tk.Frame(self.data_frame)
        self.dataset_frame.pack(side=tk.TOP,fill=tk.BOTH, expand=tk.Y )
        
        self.dataload_frame = tk.LabelFrame(self.data_frame, padx=2, pady=2,relief='groove')
        self.dataload_frame.pack(side=tk.BOTTOM,fill=tk.BOTH )
        
        
        self.diag_nb = tkinter.ttk.Notebook( self.dataset_frame, name='nb')  
        self.diag_nb.pack(fill=tk.BOTH ) # fill "master" but pad sides
 
        self.diag_nb.bind("<<NotebookTabChanged>>",  self.change_set_prof_load )
        if self.kinprof_ind is not None:
            self.diag_nb.select(self.kinprof_ind  )
        
        self.show_splines = tk.IntVar(master=self.data_frame,value=0)
        if self.device in ['D3D','NSTX']:
            spline_frame = tk.LabelFrame(self.data_frame, padx=2, pady=2,relief='groove')
            spline_frame.pack(side=tk.BOTTOM,fill=tk.BOTH ) 

            def newselection():
                #update view of zip fit and load data if not availible
                self.main_frame.config(cursor="watch")
                self.main_frame.update()
                try:
                
                    if self.show_splines.get()==1:
                        self.splines = self.data_loader(spline_fits=True)

                    if self.options['data_loaded']:
                        self.fitPlot.spline_min.set_visible(self.show_splines.get())
                        self.fitPlot.spline_mean.set_visible(self.show_splines.get())
                        self.fitPlot.spline_max.set_visible(self.show_splines.get())
                        self.fitPlot.plot_step()
                    
                finally:
                    self.main_frame.config(cursor="")
            
            label = self.device+' splines'
            if self.device == 'D3D':
                label = 'ZIPFIT'
                
            tk.Checkbutton(spline_frame, text='Show '+label,
                        command=newselection,variable=self.show_splines).pack(anchor='w')

   
        vcmd = self.dataload_frame.register(self.isfloat) 
        time_frame = tk.Frame(self.dataload_frame )
        time_frame.pack(side=tk.LEFT) 

        tk.Label(time_frame,text='From:').pack( side=tk.LEFT)
        self.tbeg_entry = tk.Entry(time_frame, validate="key",width=5,validatecommand=(vcmd, '%P'),justify=tk.CENTER)
        self.tbeg_entry.pack(side=tk.LEFT) 
        tk.Label(time_frame,text='to:'  ).pack( side=tk.LEFT)
        self.tend_entry = tk.Entry(time_frame, validate="key",width=5,validatecommand=(vcmd, '%P'),justify=tk.CENTER)
        self.tend_entry.pack(side=tk.LEFT) 
        tk.Label(time_frame,text='step:').pack(side=tk.LEFT)
        self.tstep_entry= tk.Entry(time_frame,validate="key",width=3,
                                   validatecommand=(vcmd,'%P'),justify=tk.CENTER)
        self.tstep_entry.pack(side=tk.LEFT) 
        tk.Label(time_frame,text='[ms]'  ).pack( side=tk.RIGHT)

        self.data_load = tk.Button(self.dataload_frame,text="Load",command=self.init_data,
                                   state=tk.DISABLED, foreground='#ff0000')
        self.data_load.pack( side=tk.RIGHT,   pady=2, padx=2)
        
        if hasattr(self,'eqm') and self.eqm.eq_open:
            self.data_load.config(stat=tk.NORMAL)

    def load_default_options(self, reset_settings=True):
 
     
        #dictionary with default settings
        if reset_settings and self.shot is not None:

            self.default_settings = self.default_settings_loader(self.MDSconn,self.shot)
            self.kin_profs = list(self.default_settings.keys())
  
           # for key, val in  default_settings.items():
           #     self.default_settings.setdefault(key, OrderedDict())

            #    for k,v in val.items():
           #         self.default_settings[key].setdefault(k,v)
          
        else:
            self.kin_profs = ['ne','Te'] #default list 
 
        for kin_prof in self.kin_profs:
            dic = self.default_settings.setdefault(kin_prof,{})
            
            if 'options' in dic and 'fit_options' in dic:
                dic['options']['data_loaded']=False
                dic['options']['fitted']=False
                dic['options']['fit_prepared']=False
                continue
            
            dic['options'] = { 'eta':.5,
                                  'lam':.4,
                                  'rho_max': 1.1,
                                  'rho_min': 0,                                  
                                  'rho_coord': self.coordinate,
                                  'data_loaded':False,
                                  'fitted':False,
                                  'fit_prepared':False,
                                  'zeroed_outer':False,
                                  'nr_new':101,
                                  }

            dic['fit_options'] = {
                                  'transformation':'sqrt',
                                  'robustfit': 0,
                                  'zeroedge':1,
                                  'elmrem':1,
                                  'elmsync':0,
                                  'sawteeth':1,
                                  'null_outer':0,
                                  'elm_signal':self.default_elms_signal,
                                  'outside_rho':'1.0',
                                  'pedestal_rho':'.95',
                                  'dr':'0.03',
                                  'dt':'',
                                  'sawteeth_times':'[]'
                                  }
            
            
            #specific options
            if kin_prof in ['omega','vtor']:
                dic['fit_options']['transformation']='asinh'

            if kin_prof in ['Te','ne']:
                dic['options']['eta']=.5 #due to slow core laser

            if kin_prof in ['Ti','Mach','vtor','omega'] and self.device == 'D3D':
                #measurememt outside LCFS are useless
                dic['fit_options']['null_outer']=1  
                dic['fit_options']['outside_rho']='0.98'
                dic['fit_options']['elmrem']=0

            if kin_prof in ['Mach']:
                dic['options']['lam']=.45  
                dic['fit_options']['null_outer']=0  

            if kin_prof in [ 'ne','omega','vtor' ]:
                #measurememt outside LCFS are useless
                dic['fit_options']['zeroedge']=0
  
            if kin_prof in [ 'Te/Ti']:
                dic['fit_options']['null_outer']=1  
                dic['fit_options']['zeroedge']=0
            if kin_prof in [ 'Ti']:
                dic['fit_options']['transformation']='log'

                
            if kin_prof in [ 'Zeff']:
                #measurememt outside LCFS are useless
                dic['fit_options']['null_outer']=1  
                dic['fit_options']['zeroedge']=0
                dic['fit_options']['transformation']='linear'
                dic['options']['eta']=.5 
                dic['options']['lam']=.7  
        

        #create working dictionary with TK variables keeping actual setting
        def tk_var(x):
            if isinstance(x, bool):
                return tk.BooleanVar(master=self.main_frame,value=x)
            if isinstance(x, int):
                return tk.IntVar(master=self.main_frame,value=x)
            if isinstance(x, float):
                return tk.DoubleVar(master=self.main_frame,value=x)
            if isinstance(x, str):
                return tk.StringVar(master=self.main_frame,value=x)
            
            raise Exception('mission type '+str(type(x))+'   ',str(x))
        
 
        #initial profile shown in GUI after openning
        kin_prof = self.kin_profs[0]
        for var,val in self.default_settings[kin_prof]['options'].items():
            self.options[var] = val
            
        #build dictionary with TK variables keeping the actual values from GUI
        self.load_options = deepcopy(self.default_settings)

        for kin_prof in self.kin_profs:
            options = self.default_settings[kin_prof]
            if not 'systems' in options: #dict is not fully initialized
                continue
            
            for name,config in options['systems'].items():
                load_enabled = [[ss, tk_var(var)] for ss,var in config]
                self.load_options[kin_prof]['systems'][name] = load_enabled
    
            for system,setting in options['load_options'].items():
                for name, options in setting.items():
                    if isinstance(options, dict):                        
                        for var, opt in options.items():
                            self.load_options[kin_prof]['load_options'][system][name][var] = tk_var(opt)
                    else:
                        self.load_options[kin_prof]['load_options'][system][name] = tk_var(options[0]), options[1]
 
        
        self.tbeg, self.tend = 0, 10
        for prof in self.kin_profs:
            self.load_options[prof]['trange'] = self.tbeg,self.tend,None
   
        #share options with fitPlot object
        self.fitPlot.options = self.options
        self.fitPlot.fit_options = self.fit_options  
                
  
    def set_trange(self,tbeg=None,tend=None,tstep='None'):
        self.tbeg = float(self.tbeg_entry.get())/1e3 if tbeg is None else tbeg
        self.tend = float(self.tend_entry.get())/1e3 if tend is None else tend
        if tstep is None:
            self.tstep = None
        elif tstep == 'None' and self.tstep_entry.get()!= '':
            self.tstep = float(self.tstep_entry.get())/1e3
        elif tstep is not None and tstep != 'None':
            self.tstep = tstep
            
     
        self.tbeg_entry.delete(0,tk.END)
        self.tend_entry.delete(0,tk.END)
        self.tstep_entry.delete(0,tk.END)
        self.tbeg_entry.insert(0,'%d'%np.floor(self.tbeg*1e3))
        self.tend_entry.insert(0,'%d'%np.ceil(self.tend*1e3))
        tstep = '' if self.tstep is None else '%g'%round(self.tstep*1e3,1)
        self.tstep_entry.insert(0,tstep)
        
        self.fitPlot.update_axis_range(tbeg, tend)
        self.fitPlot.tstep = self.tstep
 

    def change_set_prof_load(self,event=None):
        #change fitted quantity
    
        #save previous setting 
        if self.kinprof_ind is not None:
            self.load_options[self.kin_prof]['trange'] = self.tbeg, self.tend, self.tstep
            for var in list(self.load_options[self.kin_prof]['fit_options'].keys()):
                self.load_options[self.kin_prof]['fit_options'][var] = self.fit_options[var].get()
            for var in list(self.load_options[self.kin_prof]['options'].keys()):
                self.load_options[self.kin_prof]['options'][var] = self.options[var]
        
                
    
        #load new setting 
        self.kinprof_ind =  self.diag_nb.index(self.diag_nb.select())
        self.kin_prof = self.kin_profs[self.kinprof_ind]
        self.tstep=None
        self.fitPlot.tstep = None
 
        #prepare variables
        for var,val in self.load_options[self.kin_prof]['options'].items():
            self.options[var] = val

        
        for var,val in self.load_options[self.kin_prof]['fit_options'].items():
            self.fit_options[var].set(val)

        
        #if it was already initialized
        if self.options['data_loaded']:
            for var in ('m2g','ind_diag','diags','channel','name_channels','tres','xlab',
                        'ylab','ylab_diff','plot_rho','plot_tvec'):
                setattr(self.fitPlot,var,self.load_options[self.kin_prof][var])
            self.set_trange(*self.load_options[self.kin_prof]['trange'])
        else:
            self.fitPlot.m2g = None
            
        #update the fit figure
        self.fitPlot.change_set_prof_load()


    def init_set_prof_load(self,event=None):

        #remove points from previous profile, clear diag_nb widget
                    
        for widget in self.diag_nb.winfo_children():
            widget.destroy()
        
        if self.eqm.eq_open:
            self.data_load.config(stat=tk.NORMAL)
 
        for prof in self.kin_profs:
            
            panel = tk.Frame(self.diag_nb)
            opts = self.load_options[prof]

            frames = tk.Frame(panel),tk.Frame(panel) 
            frames[0].pack(side=tk.LEFT , fill=tk.BOTH, expand=tk.Y)
            frames[1].pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.Y)

            nsys = len(opts['systems'])            
            for i, (sys_name, systems) in enumerate(opts['systems'].items()):
                iframe = 0 if i == 0 else 1

                frame = frames[iframe]
                diag_frame = tk.LabelFrame(frame, pady=2,padx=2,relief='groove', text=sys_name)
                diag_frame.pack(side=tk.TOP,  fill=tk.BOTH, padx=2)
                inner_frame = tk.Frame(diag_frame)
                inner_frame.pack(side=tk.TOP,  fill=tk.BOTH)
                inner_frame2 = tk.Frame(inner_frame)
                inner_frame2.pack(side=tk.LEFT)
                for sys,var in systems:
                    tk.Checkbutton(inner_frame2, text=sys, variable=var).pack(anchor='w')
                    
                if not sys_name in opts['load_options']: continue
                
                for opt_name,options in opts['load_options'][sys_name].items():
                    inner_frame = tk.LabelFrame(diag_frame, padx=2,relief='groove', text= opt_name)
                    inner_frame.pack(side=tk.TOP,  fill=tk.BOTH, padx=2, )
                    inner_frame2 = tk.Frame(inner_frame)
                    inner_frame2.pack(side=tk.LEFT)
                    if isinstance(options,dict): #Checkbutton or entry
                        for opt,var in options.items():
                            if isinstance(var,tk.DoubleVar):#Checkbutton o
                                inner_frame3 = tk.Frame(inner_frame2)
                                inner_frame3.pack()
                                tk.Label(inner_frame3,text=opt).pack( side=tk.LEFT)
                                tk.Entry(inner_frame3,width=5,textvariable=var,
                                            justify=tk.CENTER).pack(anchor='w',side=tk.RIGHT)
            
                            else:
                                #if var is None:
                                tk.Checkbutton(inner_frame2,text=opt,variable=var).pack(anchor='w',side=tk.TOP)
                   
                    
                    else : #Combobox
                        tkinter.ttk.Combobox(inner_frame2 ,width=8,textvariable=options[0],
                                     values=options[1]).pack(anchor='w', side=tk.LEFT,pady=2)
   
           
            self.diag_nb.add(panel, text= prof) # add tab to Notebook

        
        
    def init_fit_opt_frame(self):
        #inicialize frame with the fitting options
                          
        for widget in self.fit_opt_frame.winfo_children():
            widget.destroy()
        
        fit_opt_frame_up = tk.Frame(self.fit_opt_frame)
        fit_opt_frame_up.pack( side=tk.TOP, fill=tk.X   ,padx=2,pady=2)
        
        fit_opt_frame_mid = tk.Frame(self.fit_opt_frame)
        fit_opt_frame_mid.pack(side=tk.TOP, fill=tk.BOTH,padx=2,pady=2)

        fit_opt_frame_down = tk.Frame(self.fit_opt_frame)
        fit_opt_frame_down.pack(side=tk.TOP, fill=tk.BOTH,expand=tk.Y,padx=2,pady=2)

        fit_opt_frame_bott = tk.Frame(self.fit_opt_frame)
        fit_opt_frame_bott.pack(side=tk.TOP, fill=tk.BOTH,expand=tk.Y,padx=2,pady=2)


        self.kin_prof = self.kin_profs[0]
        for var,val in self.load_options[self.kin_prof]['fit_options'].items():
            if var not in self.fit_options or isinstance(self.fit_options[var], str):
                if isinstance(val, str):
                    self.fit_options[var] = tk.StringVar(master=self.main_frame,value=val)
                elif isinstance(val, int):
                    self.fit_options[var] = tk.IntVar(master=self.fit_opt_frame, value=val)
                else:
                    #embed()
                    raise Exception('Unsupported type')
                
                

        tk.Label(fit_opt_frame_up,text='Fit transformation:').pack(side=tk.LEFT,padx=10)

        self.fit_trans_combo = tkinter.ttk.Combobox(fit_opt_frame_up,width=5,
                    textvariable=self.fit_options['transformation'],values=list(transformations.keys()))
        self.fit_trans_combo.pack(side=tk.LEFT,padx=0)
        #self.fit_trans_combo.current(0)
        
        def newselection(val=None):
            transform_ind = self.fit_options['transformation'].get()
            self.options['fit_prepared'] = False
            #reload data if something has changed
            #self.load_sawteeth()
            #self.load_elms()
            #print('newselection fit option')
            return self.isfloat(val)
  
                    

        self.fit_trans_combo.bind("<<ComboboxSelected>>", newselection)


        vcmd = fit_opt_frame_mid.register(newselection) 

        pedestal_rho = tk.Entry(fit_opt_frame_mid,width=4,validate="key",
                           validatecommand=(vcmd, '%P'),textvariable=self.fit_options['pedestal_rho']
                           ,justify=tk.CENTER)
        
        pedestal_rho_lbl = tk.Label(fit_opt_frame_mid,text='Pedestal rho =',padx=2)
        pedestal_rho_lbl.pack(side=tk.LEFT)
        pedestal_rho.pack(side=tk.LEFT, padx=20)
        createToolTip(pedestal_rho, 'set expected position of the pedestal')

        checkbuttons =  (('Robust fit','robustfit','Makes the fit less sensitive to outliers' ),
                        ('Zero edge','zeroedge','Press edge of profile to zero' ),
                        ('Remove ELMs; elms signal:','elmrem' ,'Remove points affected by elms' ),
                        ('ELMs sync.','elmsync' ,'Incorporate elms in the time smoothing' ),
                        ('Sawteeth [s]:','sawteeth' ,'Introduce discontinuties in the fit at sawtooth times'),
                        ('Remove points for rho >','null_outer','Ignore measurements outside given radius')) 
        frames = {}
        for name, var, descript in checkbuttons:
            frames[var] = tk.Frame(fit_opt_frame_down)
            frames[var].pack(side=tk.TOP ,fill=tk.X, padx=0, pady=0 )
            button = tk.Checkbutton(frames[var], text = name,
                        variable=self.fit_options[var], command=newselection)

            button.pack(anchor='w',side=tk.LEFT,pady=2, padx=2)
            createToolTip(button, descript)

        outside_rho_entry = tk.Entry(frames['null_outer'],width=4,validate="key",
                    validatecommand=(vcmd, '%P') ,justify=tk.CENTER,  
                    textvariable=self.fit_options['outside_rho'])

        outside_rho_entry.pack(side=tk.LEFT, padx=0)
        

        saw_time_entry = tk.Entry(frames['sawteeth'],justify=tk.RIGHT,  
                    textvariable=self.fit_options.get('sawteeth_times',[]))
        createToolTip(saw_time_entry, 'list of sawteeth times or signal name')
        saw_time_entry.pack(side=tk.LEFT, padx=0, fill=tk.X,expand = True)
        saw_time_entry.bind('<FocusOut>', self.load_sawteeth)
 
        elm_signal_entry = tk.Entry(frames['elmrem'],width=6,justify=tk.CENTER,
                                    textvariable=self.fit_options['elm_signal'])
        elm_signal_entry.pack(side=tk.LEFT, padx=0)
        elm_signal_entry.bind('<FocusOut>', self.load_elms)
 
        saw_time_entry.bind('<Return>', self.load_sawteeth)
        elm_signal_entry.bind('<Return>', self.load_elms)

        #inicialize
        newselection()

    


    def init_data(self):
        #load equilibrium and requested data
        if not hasattr(self,'data_loader'):
            print('Load EFIT first')
            return 
        
        self.fit_frame.config(cursor="watch")
        self.fit_frame.update()
        

        try:
            if not hasattr(self.eqm, 'pfm'):                
                sys.stdout.write('  * Fetching %s ...  '%self.eqm.system)
                sys.stdout.flush()
                T = time.time()
                self.eqm._read_pfm()
                self.eqm.read_ssq()
                self.eqm._read_scalars()
                self.eqm._read_profiles()
                print('\t done in %.1fs'%(time.time()-T))
     
                self.set_trange()
 
            
            self.tbeg = float(self.tbeg_entry.get())/1e3 
            self.tend = float(self.tend_entry.get())/1e3 
            try:
                self.tstep = float(self.tstep_entry.get())/1e3 
            except ValueError:
                pass
            
            if self.tbeg>= self.tend:
                tkinter.messagebox.showerror('Wrong time range','No data on the selected timerange')
                return 
            T = time.time()
            
            
            data_d = self.data_loader(self.kin_prof,self.load_options,tbeg=self.tbeg,tend=self.tend)

            if data_d is None:
                printe('No diagnostic data!')
                self.options['data_loaded'] = False

                return
            if self.show_splines.get():
                self.spline_fits = self.data_loader(spline_fits=True)
            assert data_d['data'] is not None, 'No Data!!!'
            assert len(data_d['data']) != 0, 'No Data!!!'
            
            #load the other used signals
            self.load_elms()
            self.load_sawteeth()            
            #used currently only for NSTX to shwo location of resonance surfaces
            #try:
                #self.mhdmodes = self.data_loader('mhd_modes')
            #except Exception as e:
            self.mhdmodes = None
            
        except Exception as e:
            printe( 'Error in loading:' )
            traceback.print_exc()
            
            tkinter.messagebox.showerror('Error in loading data', str(e))

            raise
            return 
            
        finally:
            self.fit_frame.config(cursor="")
  
        #for impurity load nimp instead of the actual name 
        kin_prof = self.kin_prof
        if kin_prof[0] == 'n' and kin_prof != 'ne' and self.device != 'NSTX':
            kin_prof = 'nimp'
        
        self.fitPlot.init_plot_data(kin_prof, data_d,   self.elms,  self.mhdmodes)

        if self.options['data_loaded']:
            for name in ['m2g','xlab','ylab','ylab_diff','ind_diag','diags','channel','name_channels','tres','plot_rho','plot_tvec']:
                self.load_options[self.kin_prof][name] = getattr(self.fitPlot,name)

    def load_elms(self, val=None):
        if hasattr(self,'data_loader'):
            self.fit_frame.config(cursor="watch")
            self.elms = self.data_loader('elms',{'elm_signal':self.fit_options['elm_signal']})
            edge_dis = getattr(self.fitPlot, 'edge_discontinuties',[])
            for ed in edge_dis:
                ed.remove()
                
            #update plot
            self.fitPlot.edge_discontinuties = [self.fitPlot.ax_main.axvline(t, ls='-',lw=.5,c='k', visible=False) for t in self.elms['elm_beg']] 
            self.fitPlot.plot_step()
            self.fit_frame.config(cursor="")
            self.options['fit_prepared'] = False


    
    def load_sawteeth(self, val=None):
        if hasattr(self,'data_loader'):
            self.fit_frame.config(cursor="watch")
            try:
                self.sawteeth = eval(self.fit_options['sawteeth_times'].get())
            except:
                self.sawteeth = self.data_loader('sawteeth',{'saw_signal':self.fit_options['sawteeth_times']})

            core_dis = getattr(self.fitPlot, 'core_discontinuties', [])
            for ed in core_dis:
                ed.remove()
            
            #update plot
            self.fitPlot.core_discontinuties = [self.fitPlot.ax_main.axvline(t, ls='-',lw=.5,c='k', visible=False) for t in self.sawteeth] 
            self.fitPlot.plot_step()
            self.fit_frame.config(cursor="")
            self.options['fit_prepared'] = False



    def init_save_frame(self):
        
        if self.OMFITsave is not None:
            #executed inside of OMFIT
            save = tk.Button(self.save_frame,text="Save to OMFIT",command=self.save_fits_omfit)
            save.pack( side=tk.TOP,   pady=2, padx=2)

        
        else:
            tk.Label(self.save_frame,text='Output path:').pack(side=tk.LEFT)
            
            self.output_path=tk.StringVar(master=self.save_frame,value='~'+os.sep)
            output_entry = tk.Entry(self.save_frame, validate="key",width=15,textvariable=self.output_path)
            output_entry.pack(side=tk.LEFT) 
            path_button = tk.Button(self.save_frame,text="...",command=self.save_dialog)
            path_button.pack( side=tk.LEFT)

            
            save = tk.Button(self.save_frame,text="Save",command=self.save_fits)
            save.pack( side=tk.RIGHT,   pady=2, padx=2)

        
    def save_dialog(self):
        #initialdir
        fout = tkinter.filedialog.askdirectory()
        home = os.path.expanduser('~'+os.sep)
        if fout != ():
            fout=fout.replace(home,'~'+os.sep)
            self.output_path.set(fout)

    
    
    def save_fits_omfit(self):
        
        if  self.OMFITsave is None:
            return
        
        self.fit_frame.config(cursor="watch")

        #create dictionary of fitted profiles and sent them to OMFIT
        data = {}
        rho = np.nan
        for prof in self.kin_profs:
            if not 'm2g' in self.load_options[prof] or not self.load_options[prof]['m2g'].fitted:
                continue

            data[prof] =  self.load_options[prof]['m2g']
            rho = data[prof].g_r[0]
            
            #correction to get proper values of R/Lx
            if 'Lx_correction' not in data:
                try:
                    r = self.eqm.rho2rho(rho, coord_in=self.options['rho_coord'], coord_out='RMNMP')
            
                    ind = rho <= 1
                    data['Lx_correction'] = np.ones_like(rho)
                    data['Lx_correction'][ind] =  (self.eqm.ssq['Rmag'][:,None]*np.gradient(rho)[ind]/np.gradient(r)[1][:,ind]).mean(0)*0.6/1.7
                except:
                    pass
            #raw values for plotting             
            data[prof].plot_rho  = self.load_options[prof]['plot_rho']  
            data[prof].plot_tvec = self.load_options[prof]['plot_tvec']  

        #BUG just temporary 
        if hasattr(self,'elms'):
 
            data['elms_time'] = self.elms['elm_beg'] 
            data['elms_phase'] = self.elms['data'] 
            data['elms_phase_time'] = self.elms['tvec'] 

        
        self.change_set_prof_load()
        settings = OrderedDict()
        for kin_prof in self.kin_profs:
            options = self.load_options[kin_prof]
            settings[kin_prof] = OrderedDict()
            settings[kin_prof]['systems'] = OrderedDict()
            settings[kin_prof]['load_options'] = OrderedDict()
            settings[kin_prof]['options'] =  options['options']
            settings[kin_prof]['fit_options'] = options['fit_options']

            for name,config in options['systems'].items():
                load_enabled = [[ss, var.get()] for ss,var in config]
                settings[kin_prof]['systems'][name] = load_enabled
    
            for system,setting in options['load_options'].items():
                settings[kin_prof]['load_options'][system] = OrderedDict()
                for name, options in setting.items():
                    if isinstance(options, dict):    
                        settings[kin_prof]['load_options'][system][name] = OrderedDict()
                        for var, opt in options.items():
                            settings[kin_prof]['load_options'][system][name][var] = opt.get()
                    elif isinstance(options, str):
                        settings[kin_prof]['load_options'][system][name] = options
                    else:
                        try:
                            settings[kin_prof]['load_options'][system][name] = options[0].get(), options[1]
                        except Exception as e:
                            print(e)
                            printe(('Error in saving',kin_prof, system,name,var,  options  ))
                            
        settings['EFIT'] = self.eqm.diag
        settings['elmsphase'] = self.elmsphase
        settings['elmstime'] = self.elmstime
        
        #save other radial coordinate - either rho_tor ot Psi_N
        if self.options['rho_coord'] != 'Psi_N':
            data['PsiN'] = self.eqm.rho2rho(rho, coord_in=self.options['rho_coord'], 
                                            coord_out='Psi_N',extrapolate=True)
            data['PsiN_time'] = self.eqm.t_eq
        else:
            data['PsiN'] = rho
            
        if self.options['rho_coord'] != 'rho_tor':
            data['rho'] = self.eqm.rho2rho(rho, coord_in=self.options['rho_coord'], 
                                            coord_out='rho_tor',extrapolate=True)
            data['rho_time'] = self.eqm.t_eq

        else:
            data['rho'] = rho
            

 
        if len(data):
            self.OMFITsave.runNoGUI(shot=self.shot, fitted_profiles=data, setting=settings) 
            self.saved_profiles = True
        self.fit_frame.config(cursor="")


    def save_fits(self,save_ufiles=False):
        #save everything as UFILES
        if not self.options['fitted']:
            return
 
                
        path = os.path.expanduser( self.output_path.get().strip())+os.sep
        
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path+'UFILES') and save_ufiles:
            os.makedirs(path+'UFILES')

        
        output = {}
        
        for prof in self.kin_profs:
            data = self.load_options[prof]
            
            if not 'm2g' in data or not data['m2g'].fitted:
                continue
            
            

            m2g = data['m2g']
            t_out = m2g.g_t[:,0]
            x_out = m2g.g_r[0,:]
            d_out = m2g.g
            d_err = (m2g.g_u-m2g.g_d)/2
    
            #correction to get proper values of R/Lx
            if 'Lx_correction' not in output:
                try:
                    r = self.eqm.rho2rho(x_out, coord_in=self.options['rho_coord'], coord_out='RMNMP')
            
                    ind = x_out <= 1
                    output['Lx_correction'] = np.ones_like(x_out)
                    output['Lx_correction'][ind] =  (self.eqm.ssq['Rmag'][:,None]*np.gradient(x_out)[ind]/np.gradient(r)[1][:,ind]).mean(0)*0.6/1.7
                except:
                    pass
                
            output[prof] = {'tvec':t_out, 'rho':x_out, 'rho_lbl': self.options['rho_coord'],
                            'data':np.single(d_out), 
                            'random_samples': np.single(m2g.g_samples),
                            'err': np.single(d_err),
                            'unc_low':np.single(m2g.g_d),
                            'unc_up':np.single(m2g.g_u),
                            'Lx':np.single(m2g.K*m2g.g_r),
                            'Lx_err':np.single(np.abs(m2g.Kerr_u-m2g.Kerr_d)*m2g.g_r/2),
                            'EFIT':self.eqm.diag
                            }
            
            if prof in labels and save_ufiles:
                import ufiles
                
                eta = data['options']['eta']
                lam = data['options']['lam']

                comment = 'Fitted by quickfit routine, time regularization=%2.2f\
                    space regularization= %2.2f'%(self.eta, self.lam)
                    
                    
                # 2D ufile output
                tlbl='Time                Seconds'
                xlbl=self.options['rho_coord']
                

                vdic = labels[prof]
                pre_str = vdic['pre']
                ext_str = vdic['ext']

                dlbl=vdic['dlab'].ljust(20)+vdic['TRunit'].ljust(10)
                tr_fac=vdic['TRfac']
                uf_d={ 'pre':pre_str,'ext':ext_str,'shot':str(self.shot), \
                'grid': {'X':{'lbl':tlbl,'arr':t_out}, \
                        'Y':{'lbl':xlbl,'arr':x_out} }, \
                'data': {'lbl':dlbl,'arr':tr_fac*d_out}}
            
                uf_d['comm'] = prof+'\n' 
                uf_d['comm'] += ' Mapping:\n'
                uf_d['comm'] += '  diag=%s\n' %(self.eqm.diag)
                uf_d['comm'] += comment
                ufiles.WU(uf_d, udir=path+'UFILES',dev='D3D')
                        #break
            

        kin_path = path+'kin_data_%d.npz'%self.shot
        #update existing file
        if os.path.isfile(kin_path):
            try:
                old_version = dict(np.load(kin_path, allow_pickle=True))
                old_version.update(output)
                output = old_version
            except:
                pass
        
        
        output['sawteeth'] = self.sawteeth#  eval(self.fit_options['sawteeth_times'].get())
        
        
        if hasattr(self,'elms'):
            output['elms'] = self.elms
        
 
        np.savez_compressed(path+'kin_data_%d.npz'%self.shot, **output)
        self.saved_profiles = True
        print('saved to: ', path+'kin_data_%d.npz'%self.shot)

          
    def Quit(self):
         
        if  self.options['fitted'] and not self.saved_profiles:
            if self.OMFITsave is not None:
                if tkinter.messagebox.askyesno("Fit is ready", "Fits were not saved. Do you want to save them to OMFIT?"):
                    self.save_fits_omfit()
            else:    
                if tkinter.messagebox.askyesno("Fit is ready", "Fits were not saved to file, do you want to save them?"):
                    self.save_fits( )
        try:
            self.main_frame.destroy()
        except:
            pass


            
    
def main():
    
    #guess on which device is the quickfit running
    #any better way how to distinguis in betwen different instalations? 
    if os.path.exists('/fusion/projects/codes/quickfit'):
        tok_name = 'D3D'
        mds_server = 'atlas.gat.com'
    elif os.path.exists('/afs/ipp-garching.mpg.de/'):
        tok_name = 'AUG'
        mds_server = 'None'
    elif os.path.exists('/p/'):
        tok_name = 'NSTX'
        mds_server = 'skylark.pppl.gov:8501'
    else:
        tok_name = 'D3D'
        mds_server='localhost'
        print('Local installation of quickfit - DIII-D?')
                
    
    import argparse
    parser = argparse.ArgumentParser( usage='Fast profiles fitting GUI')
    
    parser.add_argument('--shot', metavar='S', help='optional shot number', default='')
    parser.add_argument('--tmin',type=float, metavar='S', help='optional tmin', default=None)
    parser.add_argument('--tmax',type=float, metavar='S', help='optional tmax', default=None)
    parser.add_argument('--preload', help='optional',action='store_true')
    parser.add_argument('--device', type=str,help='tokamak name (D3D, CMOD, NSTX or AUG)',default=tok_name)
    parser.add_argument('--mdsplus', type=str,help='MDS+ server',default=mds_server )
    parser.add_argument('--elmsphase', help='Apply ELMs synchronisation - elm fraction',default=False,action='store_true')
    parser.add_argument('--elmstime', help='Apply ELMs synchronisation - elm time',default=False,action='store_true')
    parser.add_argument('--map_coordinate', type=str,help='rho_pol, rho_tor, Psi_N, r_a',default='rho_tor')

    
 
    args = parser.parse_args()
    


    raw = {}

    
    mdsserver =  args.mdsplus

    
    myroot = tk.Tk(className=' Profiles')
    mlp = DataFit(myroot, mdsserver,shot=args.shot,raw_data = raw,
                  elmsphase=args.elmsphase, elmstime=args.elmstime, 
                  device=args.device,coordinate= args.map_coordinate)
    myroot.title('QUICKFIT')
    myroot.minsize(width=300, height=400)
    imgicon = tk.PhotoImage(file=icon_dir+'icon.gif',master=myroot)
    myroot.tk.call('wm', 'iconphoto', myroot._w, imgicon) 
    myroot.protocol("WM_DELETE_WINDOW", mlp.Quit)
    
    if args.tmin is not None or args.tmax is not None:
        mlp.set_trange(tbeg=args.tmin,tend=args.tmax,tstep='None')

    mlp.elmsphase = args.elmsphase
    mlp.elmstime = args.elmstime

    
    if args.preload:
        
        for i,k in enumerate(mlp.kin_profs):
            if k[0] =='n' and k != 'nC6' or k == 'Zeff': continue
            mlp.diag_nb.select(i)    
            mlp.init_data()
        mlp.diag_nb.select(0)    
        myroot.mainloop()
    
    else:
        myroot.mainloop()

    
    
 
  
if __name__ == "__main__":
    main()
#TODO 
#NIMP - option to select the impurity 

#elm synchroonizaci, sawteeth
#ROBUST FIT
#TODO pedestal weighting

 ##vysvetlivky ke vsem tlacitkum 
 #TS z- posun 
#BUG homebutton blbne , odtsrnit??
 #robust fit option, nastavit podle realne ety!!!
 #3D plot se neaktualizuje, a nemeni se jeho zlabel
 #3d plot blbne pro zmene diagnotiku a po zavreni 
 #170791 nimp kouknout se pomoci omfiptorf
 #save data button!
 #175212 nimp je blbe, ze by se tam menili svazky?
 #pri ukladani obrazku ulozit jen hlavni graf!!
 #indikovat tocenim mysi ze to pracuje 
 
#FIT 

#fitovat jen rozdil mezi hyper tan a normalnim? 
#elm sync 
#vylepsit robsut fit 
#lepesi fit pedestalu 
#nejak se neaktualizuje ekvilibrium, kdyz otevru novy vyboj #
#defaultne nestavit nula v rho - 1.2? 
#TODO 167490 - pedestal ne shouler!!! otesovat 
#TODO 167501 blba poloha pedestalu!!! delat autokorekci? asi jen u efit01!!
#TODO udlat moznost zobrazit tu CO2 kalibraci 
#TODO prizpusobit radialni diskretizaci nelinearni ose, 167501, omega na ose, r
 #meenit barvu panelu podle toho jestliui je fitnuty?
#monoticity?
# konce v case dat nulovou 1. derivaci


#kde jsou VB data z okrajovycg CER kamnalu?
