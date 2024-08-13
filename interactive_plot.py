#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time,os,sys
import numpy as np
import matplotlib
from collections import OrderedDict

matplotlib.use('TkAgg')
import matplotlib.backends.backend_tkagg as tkagg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
except:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavigationToolbar2Tk


# Change the default cursor to any valid TK cursor
# To hide it, you'd use the string "none" (or possibly "no" on windows)
try:
    from matplotlib.backend_bases import cursors
    tkagg.cursord[cursors.POINTER] = 'left_ptr'
except:
    pass
from  IPython import embed
##IPython.embed()

import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.widgets import Slider,RectangleSelector
from matplotlib.figure import Figure

from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d


import tkinter as tk
import tkinter.font
from tooltip import createToolTip
from grid_map import map2grid

np.seterr(all='raise')


#  fun, inversed fun, derivative
transformations = OrderedDict()
transformations['linear']   = lambda x:x,lambda x:x, lambda x:1
transformations['log']   = lambda x: np.log(np.maximum(x,0)/.1+1),  lambda x:np.maximum(np.exp(x)-1,1.e-6)*.1,   lambda x:1/(.1+np.maximum(0, x))  #not a real logarithm..
transformations['sqrt']  = lambda x: np.sqrt(np.maximum(0, x)), np.square,lambda x:.5/np.sqrt(np.maximum(1e-5, x))
transformations['asinh'] = np.arcsinh, np.sinh, lambda x:1./np.hypot(x,1)




icon_dir = os.path.dirname(os.path.realpath(__file__))+'/icons/'

def update_fill_between(fill,x,y_low,y_up,min,max ):
    paths, = fill.get_paths()
    nx = len(x)



    #change between time/radius slice, create a new fill
    if len(paths.vertices) != 2*nx+3:
        _fill = fill
        fill = _fill.axes.fill_between(np.arange(nx),
                np.zeros(nx),np.zeros(nx), alpha=_fill.get_alpha(),
                facecolor=_fill.get_facecolor(),
                edgecolor=_fill.get_edgecolor())

        paths, = fill.get_paths()
        #remove the only fill
        _fill.set_visible(False)
        del _fill #not really working



    y_low = np.maximum(y_low, min)
    y_low[y_low==max] = min
    y_up = np.minimum(y_up,max)

    vertices = paths.vertices.T
    vertices[:,1:nx+1] = x,y_up
    vertices[:,nx+1] =  x[-1],y_up[-1]
    vertices[:,nx+2:-1] = x[::-1],y_low[::-1]
    vertices[:,0] = x[0],y_up[0]
    vertices[:,-1] = x[0],y_up[0]

    return fill


def printe(message):
    CSI="\x1B["
    reset=CSI+"m"
    red_start = CSI+"31;40m"
    red_end = CSI + "0m"
    print(red_start,message,red_end)


class FitPlot():

    plt_time = 0
    plt_radius = 0
    fsize3d=16
    fsize=matplotlib.rcParams['font.size']

    #ts_revisions = []
    edge_discontinuties = []
    core_discontinuties = []

    tbeg = 0
    tend =  7
    picked = False
    grid=False
    logy=False
    m2g = None
    shot=None

    def __init__(self, parent, fit_frame):
        self.parent = parent

        self.fit_frame = fit_frame

        self.rstride=1
        self.cstride=3

        self.tstep = None
        self.xlab = ''
        self.ylab = ''
        self.ylab_diff = ''


    def isfloat(self,num):

        if num == '':
            return True
        try:
            float(num)
            return True
        except:
            return False

    def update_axis_range(self,tbeg, tend):
        if tbeg is None or tend is None:
            return
        self.main_slider.valmin = tbeg
        self.main_slider.valmax = tend
 

        self.sl_ax_main.set_xlim(tbeg, tend )
        self.ax_main.set_xlim(self.options['rho_min'], self.options['rho_max'])
        self.tbeg = tbeg
        self.tend = tend

    def change_set_prof_load(self):
        #update fit figure if the fitted quantity was changed

        self.sl_eta.set_val(self.options['eta'])
        self.sl_lam.set_val(self.options['lam'])

        if self.options['data_loaded']:
            self.init_plot()
            self.changed_fit_slice()
        else:
            self.ax_main.cla()
            self.ax_main.grid(self.grid)
            self.fig.canvas.draw_idle()

    def init_plot_data(self,prof,data_d, elms, mhd_modes):

        #merge all diagnostics together
        unit,labels = '',[]
        data_rho, plot_rho, data, data_err,weights, data_tvec, plot_tvec,diags = [],[],[],[],[],[],[],[]
        #channel and point index for later identification
        ind_channels, name_channels, ind_points = [], [], []
        n_ch,n_points = 0,0
        for ch in data_d['data']:
            if prof not in ch: continue
            d = ch[prof].values
            data.append(d)
            err = np.copy(ch[prof+'_err'].values)
            #NOTE negative values are set to be masked
            mask = err <= 0
            #negative infinite ponts will not be shown
            err[np.isfinite(err)&mask]*=-1
            data_err.append(np.ma.array(err,mask=mask))
            data_rho.append(ch['rho'].values)
            data_tvec.append(np.tile(ch['time'].values, data_rho[-1].shape[:0:-1]+(1,)).T)
            plot_tvec.append(np.tile(ch['time'].values, d.shape[1:]+(1,)).T )
            diags.append(ch['diags'].values)
            labels.append(ch[prof].attrs['label'])

            s = d.shape
            if len(s) == 1:
                dch = 1
                if 'channel' in ch.attrs:
                    name_channels.append(ch.attrs['channel'])
                elif len(diags[-1]):
                    name_channels.append(diags[-1][0]+'_%d'%(n_ch+1))
                else:
                    name_channels.append(str(n_ch+1))
            else:
                dch = s[1]
                if 'channel' in ch:
                    name_channels += list(ch['channel'].values)
                else:
                    try: #error  if there are no data
                        name_channels += [diags[-1][0,i]+'_%d'%(i+1) for i in range(dch)]
                    except:
                        pass

            ind_channels.append(np.tile(np.arange(dch,dtype='uint32')+n_ch,(s[0],1)))
            n_ch+=  dch
            ind_points.append(np.tile(n_points+np.arange(d.size,dtype='uint32').reshape(d.shape).T,
                                      data_rho[-1].shape[d.ndim:]+((1,)*d.ndim)).T)
            n_points+= d.size


            if 'weight' in ch:
                #non-local measurements
                weights.append(ch['weight'].values)
                plot_rho.append(ch['rho_tg'].values )
            else:
                weights.append(np.ones_like(d))
                plot_rho.append(ch['rho'].values )



        unit  = ch[prof].attrs['units']

        if n_ch == 0 or n_points == 0:
            print('No data !! Try to extend time range')
            return

        diag_names = data_d['diag_names']

        #embed()
        label = ','.join(np.unique(labels))

        self.elms = elms
        self.mhd_modes = mhd_modes
        self.name_channels = name_channels

        self.options['data_loaded'] = True
        self.options['fitted'] = False
        rho  = np.hstack([r.ravel()  for r  in data_rho])
        y    = np.hstack([d.ravel()  for d  in data])
        yerr = np.ma.hstack([de.ravel() for de in data_err])

        self.channel=np.hstack([ch.ravel() for ch in ind_channels])
        points =np.hstack([p.ravel() for p in ind_points])
        weights=np.hstack([w.ravel() for w in weights])
        tvec  = np.hstack([t.ravel() for t in data_tvec])
        diags = np.hstack([d.ravel() for d in diags])
        self.plot_tvec = np.hstack([t.ravel()  for t  in plot_tvec])
        self.plot_rho  = np.hstack([r.ravel()  for r  in plot_rho])
        self.options['rho_min'] = np.minimum(0, np.maximum(-1.1,self.plot_rho.min()))
        diag_dict = {d:i for i,d in enumerate(diag_names)}
        self.ind_diag = np.array([diag_dict[d] for d in diags])
        #show legend only for diags which actually have some data
        uind = np.unique(self.ind_diag)
        self.diags = [diag_names[i] for i in uind]

        if self.parent.elmsphase:
            #epl phase
            self.plot_tvec = np.interp(self.plot_tvec,self.elms['tvec'],self.elms['data'])
            tvec = np.interp(tvec,self.elms['tvec'],self.elms['data'])

        if self.parent.elmstime:
            #time from nearest ELM
            self.plot_tvec -= self.elms['elm_beg'][self.elms['elm_beg'].searchsorted(self.plot_tvec)-1]
            tvec -= self.elms['elm_beg'][self.elms['elm_beg'].searchsorted(tvec)-1]


        tstep = 'None'
        if self.tstep is None:
            tstep = float(data_d['tres'])

        self.parent.set_trange(np.amin(tvec),np.amax(tvec),tstep)

        self.tres = self.tstep

        #plot timeslice nearest to the original location where are some data
        self.plt_time = tvec[np.argmin(abs(self.plt_time-tvec))]

        self.ylab = r'$%s\ [\mathrm{%s}]$'%(label,unit)
        xlab = self.options['rho_coord'].split('_')
        self.xlab = xlab[0]
        if self.options['rho_coord'][:3] in ['Psi', 'rho']:
            self.xlab = '\\'+self.xlab
        if len(xlab) > 1:
            self.xlab += '_{'+xlab[1]+'}'
        self.xlab = '$'+self.xlab+'$'

        self.ylab_diff =  r'$R/L_{%s}/\rho\ [-]$'%label

        #create object of the fitting routine
        self.m2g = map2grid(rho,tvec,y,yerr,points, weights, self.options['nr_new'],self.tstep)
        self.options['fit_prepared'] = False
        self.options['zeroed_outer'] = False
        self.options['elmrem_ind'] = False

        self.init_plot()
        self.changed_fit_slice()


    def init_plot(self):
        #clear and inicialize the main plot with the fits

        self.ax_main.cla()
        self.ax_main.grid(self.grid)

        self.ax_main.ticklabel_format(style='sci', scilimits=(-2,2), axis='y')

        self.ax_main.set_ylabel(self.ylab,fontsize=self.fsize+2)
        self.ax_main.set_xlabel(self.xlab,fontsize=self.fsize+2)

        #the plots inside
        colors = matplotlib.cm.brg(np.linspace(0,1,len(self.diags)))
        self.plotline,self.caplines,self.barlinecols = [],[],[]
        self.replot_plot  = [self.ax_main.plot([],[],'+',c=c)[0] for c in colors]

        for i,d in enumerate(self.diags):
            plotline,caplines,barlinecols=self.ax_main.errorbar(0,np.nan,0,fmt='.', capsize = 4,
                                                            label=d,c=colors[i], zorder=1)
            self.plotline.append(plotline)
            self.caplines.append(caplines)
            self.barlinecols.append(barlinecols)

        self.fit_plot, = self.ax_main.plot([],[],'k-',linewidth=.5, zorder=2)
        nr = self.options['nr_new']
        self.fit_confidence = self.ax_main.fill_between(np.arange(nr),
            np.zeros(nr),np.zeros(nr), alpha=.2, facecolor='k',
            edgecolor='None', zorder=0)

        self.lcfs_line = self.ax_main.axvline(1, ls='--',c='k',visible=False)
        self.zero_line = self.ax_main.axhline(0, ls='--',c='k',visible=False)

        sawteeth = self.parent.sawteeth
        self.core_discontinuties = [self.ax_main.axvline(t, ls='-',lw=.5,c='k',visible=False) for t in sawteeth]
        self.edge_discontinuties = [self.ax_main.axvline(t, ls='-',lw=.5,c='k',visible=False) for t in self.elms['elm_beg']]

        if self.mhd_modes is not None:
            self.mhd_locations = {mode:self.ax_main.axvline(np.nan, ls='-',lw=.5,c='k',visible=False) for mode in self.mhd_modes['modes']}
            self.mhd_labels = {mode:self.ax_main.text(  np.nan,0,  mode, visible=False) for mode in self.mhd_modes['modes']}



        self.spline_mean, = self.ax_main.plot([],[],'--',c='.5',linewidth=1,visible=False)
        self.spline_min,  = self.ax_main.plot([],[],':', c='.5',linewidth=1,visible=False)
        self.spline_max,  = self.ax_main.plot([],[],':', c='.5',linewidth=1,visible=False)



        leg = self.ax_main.legend(  fancybox=True,loc='upper right')
        leg.get_frame().set_alpha(0.9)
        try:
            leg.set_draggable(True)
        except:
            leg.draggable()

        #make the legend interactive
        self.leg_diag_ind = {}
        for idiag,legline in enumerate( leg.legendHandles):
            legline.set_picker(20)  # 20 pts tolerance
            self.leg_diag_ind[legline] = idiag

        description = self.parent.device +' %d'%self.shot
        self.plot_description = self.ax_main.text(1.01,.05,description,rotation='vertical',
            transform=self.ax_main.transAxes,verticalalignment='bottom',
            size=10,backgroundcolor='none',zorder=100)

        title_template = '%s, time: %.3fs'    # prints running simulation time
        self.time_text = self.ax_main.text(.05, .95, '', transform=self.ax_main.transAxes)
        self.chi2_text = self.ax_main.text(.05, .90, '', transform=self.ax_main.transAxes)


        self.fit_options['dt'].set('%.2g'%(self.tres*1e3) )
        self.click_event = {1:None,2:None,3:None}

        def line_select_callback(eclick, erelease):
            #'eclick and erelease are the press and release events'

            click_event = self.click_event[eclick.button]
            #make sure that this event is "unique", due to some bug in matplolib
            if click_event is None or click_event.xdata!=eclick.xdata:
                self.click_event[eclick.button] = eclick
                x1, y1 = eclick.xdata,eclick.ydata
                x2, y2 = erelease.xdata,erelease.ydata

                #delete/undelet selected points
                undelete = eclick.button == 3
                what = 'channel' if self.ctrl else 'point'
                self.delete_points(eclick,(x1,x2),(y1,y2), what,undelete)

            self.RS_delete.set_visible(False)
            self.RS_undelete.set_visible(False)

 
            try:#old matplolib issue
                if click_event is None or click_event.xdata!=eclick.xdata:
                    self.RS_delete.visible = True
                    self.RS_undelete.visible = True
            except:
                pass

        try:
            rectprops = dict(facecolor='red', edgecolor = 'red',alpha=0.5, fill=True,zorder=99)


            self.RS_delete = RectangleSelector(self.ax_main, line_select_callback,
                                           #drawtype='box',
                                           useblit=True,
                                           button=[1],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           props=rectprops,
                                           spancoords='pixels',
                                           interactive=True)
            rectprops = dict(facecolor='blue', edgecolor = 'blue',alpha=0.5, fill=True,zorder=99)

            self.RS_undelete = RectangleSelector(self.ax_main, line_select_callback,
                                           #drawtype='box',
                                           useblit=True,
                                           button=[ 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           props=rectprops,
                                           spancoords='pixels',
                                           interactive=True)
        except:#old matplolib issue
            rectprops = dict(facecolor='red', edgecolor = 'red',alpha=0.5, fill=True,zorder=99)


            self.RS_delete = RectangleSelector(self.ax_main, line_select_callback,
                                           drawtype='box',
                                           useblit=False,
                                           button=[1],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           rectprops=rectprops,
                                           spancoords='pixels',
                                           interactive=True)
            rectprops = dict(facecolor='blue', edgecolor = 'blue',alpha=0.5, fill=True,zorder=99)

            self.RS_undelete = RectangleSelector(self.ax_main, line_select_callback,
                                           drawtype='box',
                                           useblit=True,
                                           button=[ 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           rectprops=rectprops,
                                           spancoords='pixels',
                                           interactive=True)



 
    def changed_fit_slice(self):
        #switch between time/radial slice or gradient

        if self.plot_type.get() in [1,2]:
            #radial slice, radial gradient
            self.view_step.config(textvariable=self.fit_options['dt'])
            self.view_step_lbl.config(text='Plot step [ms]')
            self.main_slider.label='Time:'
            self.parent.set_trange(self.tbeg, self.tend)
            self.ax_main.set_xlim(self.options['rho_min'],self.options['rho_max'])
            self.ax_main.set_xlabel(self.xlab,fontsize=self.fsize+2)
            if self.plot_type.get() == 1:
                self.ax_main.set_ylabel(self.ylab,fontsize=self.fsize+2)
            if self.plot_type.get() == 2:
                self.ax_main.set_ylabel(self.ylab_diff,fontsize=self.fsize+2)


        if self.plot_type.get() in [0]:
            #time slice
            self.view_step.config(textvariable=self.fit_options.get('dr',0.02))
            self.view_step_lbl.config(text='Radial step [-]')
            self.main_slider.label='Radius:'
            self.main_slider.valmin = self.options['rho_min']
            self.main_slider.valmax = self.options['rho_max']
            self.sl_ax_main.set_xlim(self.options['rho_min'], self.options['rho_max'])
            self.ax_main.set_xlim(self.tbeg, self.tend)
            self.ax_main.set_xlabel('time [s]',fontsize=self.fsize+2)
            self.ax_main.set_ylabel(self.ylab)


        self.lcfs_line.set_visible(self.plot_type.get() in [1,2])
        self.zero_line.set_visible(self.plot_type.get() in [0,2])

        self.updateMainSlider()
        self.PreparePloting()
        self.plot_step()
        self.plot3d(update=True)


    def init_fit_frame(self):


        #frame with the navigation bar for the main plot

        fit_frame_up  = tk.Frame(self.fit_frame)
        fit_frame_down = tk.LabelFrame(self.fit_frame,relief='groove')
        fit_frame_up.pack(  side=tk.TOP , fill=tk.BOTH )
        fit_frame_down.pack(  side=tk.BOTTOM, fill=tk.BOTH, expand=tk.Y)



        self.plot_type = tk.IntVar(master=self.fit_frame)
        self.plot_type.set(1)
        r_buttons = 'Radial slice', 'Time slice', 'Gradient'
        for nbutt, butt in enumerate(r_buttons):
            button = tk.Radiobutton(fit_frame_up, text=butt, variable=self.plot_type,
                        command=self.changed_fit_slice, value=nbutt)
            button.pack(anchor='w', side=tk.LEFT,pady=0, padx=2)



        self.show_splines = tk.IntVar(master=fit_frame_up,value=0)
        if self.parent.device in ['D3D','NSTX']:

            def newselection():
                #update view of zip fit and load data if not availible
                self.fit_frame.config(cursor="watch")
                self.fit_frame.update()
                try:

                    if self.show_splines.get() == 1:
                        self.splines = self.parent.data_loader(spline_fits=True)

                    if self.options['data_loaded']:
                        self.spline_min.set_visible(self.show_splines.get())
                        self.spline_mean.set_visible(self.show_splines.get())
                        self.spline_max.set_visible(self.show_splines.get())
                        self.plot_step()

                finally:
                    self.fit_frame.config(cursor="")

            label = self.parent.device+' splines'
            if self.parent.device == 'D3D':
                label = 'ZIPFIT'

            spline_cbx = tk.Checkbutton(fit_frame_up , text='Show '+label,
                        command=newselection,variable=self.show_splines)

            spline_cbx.pack(anchor='w', side=tk.RIGHT,pady=0, padx=20)


        # canvas frame
        self.fig = Figure(figsize=(9.5,9), dpi=75)
        self.fig.patch.set_facecolor((.93,.93,.93))
        self.ax_main = self.fig.add_subplot(111)
        self.title = self.ax_main.set_title('')


        self.canvasMPL = tkagg.FigureCanvasTkAgg(self.fig,master=fit_frame_down)
        self.toolbar = NavigationToolbar2Tk( self.canvasMPL, fit_frame_down)

        def print_figure(filename, **kwargs):
            #cheat print_figure function to save only the plot without the sliders.

            #make figure bigger and remove sliders
            for obj in [self.tsmooth_txt,self.rsmooth_txt,self.sl_eta.ax, self.sl_lam.ax, self.main_slider.ax, self.title]:
                obj.set_visible(False)
            self.fig.patch.set_facecolor('w')
            fig_size = self.fig.get_size_inches()
            self.fig.subplots_adjust(left=.15 )
            size = 6
            ratio = 6/5
            #print(fig_size)
            self.fig.set_size_inches(size,size/ratio)
            #print(self.fig.get_size_inches())

            print(f'Figure saved to {filename} at size of {size}x{size} inches and {int(kwargs["dpi"])} dpi')
            self.canvas_print_figure(filename, **kwargs)

            #reset to original setup
            self.fig.subplots_adjust(left=.10 )
            self.fig.set_size_inches(fig_size)
            self.fig.patch.set_facecolor((.93,.93,.93))
            for obj in [self.tsmooth_txt,self.rsmooth_txt,self.sl_eta.ax, self.sl_lam.ax, self.main_slider.ax, self.title]:
                obj.set_visible(True)
            self.fig.canvas.draw_idle()


        self.canvas_print_figure = self.toolbar.canvas.print_figure
        self.toolbar.canvas.print_figure = print_figure

        self.canvasMPL.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvasMPL._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        self.lcfs_line = self.ax_main.axvline(1, ls='--',c='k',visible=False)
        self.zero_line = self.ax_main.axhline(0, ls='--',c='k',visible=False)



        hbox1 = tk.Frame(fit_frame_down)
        hbox1.pack(side=tk.BOTTOM,fill=tk.X)
        mouse_help  = tk.Label(hbox1,text='Mouse: ')
        mouse_left  = tk.Label(hbox1,text='Left (+Ctrl): del point (Channel)  ',fg="#900000")
        mouse_mid   = tk.Label(hbox1,text='Mid: re-fit  ',fg="#009000")
        mouse_right = tk.Label(hbox1,text='Right: undelete point  ',fg="#000090")
        mouse_wheel = tk.Label(hbox1,text='Wheel: shift',fg="#905090")

        for w in (mouse_help, mouse_left, mouse_mid, mouse_right,mouse_wheel):
            w.pack(side=tk.LEFT)

        hbox2 = tk.Frame(fit_frame_down)
        hbox2.pack(side=tk.BOTTOM,fill=tk.X)


        helv36 = tkinter.font.Font(family='Helvetica', size=13, weight='bold')
        calc_button = tk.Button(hbox2,text='Fit',bg='red',command=self.calculate,font=helv36,  height= 2, width=5)
        calc_button.pack(side=tk.LEFT)


        self.playfig = tk.PhotoImage(file=icon_dir+'play.gif',master=self.fit_frame)
        self.pausefig = tk.PhotoImage(file=icon_dir+'pause.gif',master=self.fit_frame)
        self.forwardfig = tk.PhotoImage(file=icon_dir+'forward.gif',master=self.fit_frame)
        self.backwardfig = tk.PhotoImage(file=icon_dir+'backward.gif',master=self.fit_frame)

        self.backward_button = tk.Button(hbox2,command=self.Backward,image=self.backwardfig, height= 45, width=40)
        self.backward_button.pack(side=tk.LEFT)
        self.play_button = tk.Button(hbox2,command=self.Play,image=self.playfig, height= 45, width=40)
        self.play_button.pack(side=tk.LEFT)
        self.forward_button = tk.Button(hbox2,command=self.Forward,image=self.forwardfig, height= 45, width=40)
        self.forward_button.pack(side=tk.LEFT)

        self.button_3d = tk.Button(hbox2,command=self.plot3d,text='3D',font=helv36, height= 2)
        self.button_3d.pack(side=tk.LEFT)



        self.stop=True
        self.ctrl=False
        self.shift=False
        def stop_handler(event=None, self=self):
            self.stop  = True


        self.forward_button.bind( "<Button-1>", stop_handler)
        self.backward_button.bind("<Button-1>", stop_handler)

        vcmd = hbox2.register(self.isfloat)

        self.view_step = tk.Entry(hbox2,width=4,validate="key",
                           validatecommand=(vcmd, '%P'),justify=tk.CENTER )
        self.view_step_lbl = tk.Label(hbox2,text='Plot step [ms]')
        self.view_step.pack(side=tk.RIGHT, padx=10)
        self.view_step_lbl.pack(side=tk.RIGHT)


        axcolor = 'lightgoldenrodyellow'

        self.fig.subplots_adjust(left=.10,bottom=.16,right=.95,top=.95
                          ,hspace=.1, wspace = 0)

        from numpy.lib import NumpyVersion
        kargs = {'facecolor':axcolor} if NumpyVersion(matplotlib.__version__) > NumpyVersion('2.0.0') else {'axisbg':axcolor}
        self.sl_ax_main = self.fig.add_axes([.1,.08, .8, .025],**kargs)
        self.main_slider = Slider(self.sl_ax_main,'',self.tbeg,self.tend, valinit=self.tbeg)

        self.sl_ax = self.fig.add_axes([.1,.02, .35, .025], **kargs)
        self.sl_eta  = Slider(self.sl_ax ,'', 0, 1, valinit=self.options['eta'])

        self.sl_ax2 = self.fig.add_axes([.55,.02, .35, .025], **kargs)
        self.sl_lam = Slider(self.sl_ax2,'', 0, 1, valinit=self.options['lam'])

        self.tsmooth_txt = self.fig.text(.1,.058, 'Time smoothing -->:')
        self.rsmooth_txt = self.fig.text(.55,.058, 'Radial smoothing -->:')


        createToolTip(self.forward_button,'Go forward by one step')
        createToolTip(self.backward_button,'Go backward by one step')
        createToolTip(self.play_button,'Go step by step forward, pause by second press')
        createToolTip(self.view_step,'Plotting time/radial step, this option influences only the plotting, not fitting!')

        #createToolTip(calc_button,'Calculate the 2d fit of the data')


        def update_eta(eta):
            self.options['eta'] = eta
            stop_handler()
        def update_lam(lam):
            stop_handler()
            self.options['lam'] = lam


        def update_slider(val):
            try:
                if self.plot_type.get() in [1,2]:
                    self.plt_time = val
                if self.plot_type.get() in [0]:
                    self.plt_radius = val

                self.updateMainSlider()
                self.plot_step()
            except :
                print('!!!!!!!!!!!!!!!!!!!!!main_slider error!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                raise
        self.main_slider.on_changed(update_slider)


        self.cid1 = self.fig.canvas.mpl_connect('button_press_event',self.MouseInteraction)
        self.cid2 = self.fig.canvas.mpl_connect('scroll_event',self.WheelInteraction)
        self.cid3 = self.fig.canvas.mpl_connect('key_press_event',   self.on_key)
        self.cid4 = self.fig.canvas.mpl_connect('key_release_event', self.off_key)
        self.cid5 = self.fig.canvas.mpl_connect('button_press_event', lambda event:self.fig.canvas._tkcanvas.focus_set())
        self.cid6 = self.fig.canvas.mpl_connect('pick_event', self.legend_pick )

        self.sl_eta.on_changed(update_eta)
        self.sl_lam.on_changed(update_lam)



    def calculate(self):


        if self.shot is None:
            print('Set shot number first!')
            tkinter.messagebox.showerror('Missing shot number','Shot number is not defined')


        if self.m2g is None:
            #print("No data to fit, let's try to load them first...")
            self.parent.init_data()

            if self.m2g is None:
                raise Exception('Loading of data failed!')


        #if self.elms['signal'] != self.fit_options['elm_signal'].get():
            #print('Load new ELM signal '+self.fit_options['elm_signal'].get())
            #self.elms = self.parent.data_loader('elms',{'elm_signal':self.fit_options['elm_signal']})
            #self.edge_discontinuties = [self.ax_main.axvline(t, ls='-',lw=.2,c='k',visible=False) for t in self.elms['elm_beg']]




        sys.stdout.write('  * Fitting  ... \t ')
        sys.stdout.flush()
        T = time.time()
        #make the fit of the data
        self.fit_frame.config(cursor="watch")
        self.fit_frame.update()

        self.parent.saved_profiles = False

        sawteeth = self.parent.sawteeth  if self.fit_options['sawteeth'].get() else []


        elms = self.elms['elm_beg'] if self.fit_options['elmsync'].get() else []
        elm_phase = (self.elms['tvec'],self.elms['data']) if self.fit_options['elmsync'].get() else None

        robust_fit = self.fit_options['robustfit'].get()
        zeroedge = self.fit_options['zeroedge'].get()

        pedestal_rho = float(self.fit_options['pedestal_rho'].get())

        #get functions for profile transformation
        transform = transformations[self.fit_options['transformation'].get()]

        even_fun = self.options['rho_coord'] not in ['Psi','Psi_N']

        if self.m2g is None:
            return

        if not self.options['fit_prepared']:
            #print 'not yet prepared! in calculate'
            self.m2g.PrepareCalculation(zero_edge=zeroedge ,
                                    core_discontinuties =sawteeth,
                                    edge_discontinuties= elms,
                                    transformation = transform,
                                    pedestal_rho=pedestal_rho,
                                    robust_fit = robust_fit,
                                    elm_phase=elm_phase,
                                    even_fun = even_fun)
            self.m2g.corrected = False

            #remove points affected by elms
            if self.fit_options['elmrem'].get() and len(self.elms['tvec']) > 2:
                elm_phase = np.interp(self.plot_tvec,self.elms['tvec'],self.elms['data'])
                elm_ind = (elm_phase < 0.1) | ((elm_phase > 0.95)&(elm_phase < 1)) #remove also data shortly before an elm
                self.options['elmrem_ind'] = (self.plot_rho >.8) & elm_ind
                self.m2g.Yerr.mask |= self.options['elmrem_ind']
            elif np.any(self.options['elmrem_ind']):
                try:
                    self.m2g.Yerr.mask[self.options['elmrem_ind']] = False
                except:
                    print('np.shape(self.options[elmrem_inds]),self.m2g.Yerr.mask.shape ',np.shape(self.options['elmrem_ind']),self.m2g.Yerr.mask.shape , self.plot_rho.shape, self.plot_tvec.shape)

                self.options['elmrem_ind'] = False

            #remove points in outer regions
            zeroed_outer = self.options['zeroed_outer']
            if self.fit_options['null_outer'].get():
                if np.any(zeroed_outer):
                    self.m2g.Yerr.mask[zeroed_outer] = False
                rho_lim =  float(self.fit_options['outside_rho'].get())
                zeroed_outer = self.plot_rho > rho_lim
                self.m2g.Yerr.mask |= zeroed_outer
            elif np.any(zeroed_outer):
                self.m2g.Yerr.mask[zeroed_outer] = False
                zeroed_outer = False
            self.options['zeroed_outer'] = zeroed_outer
            self.options['fit_prepared'] = True

        lam = self.sl_lam.val
        eta = self.sl_eta.val
        self.m2g.Calculate(lam,eta, n_noise_vec=self.options.get('n_noise_vec', 50) )






        print('\t done in %.1fs'%(time.time()-T))

        self.options['fitted'] = True
        self.chi2_text.set_text('$\chi^2/doF$: %.2f'%self.m2g.chi2)

        self.plot_step()
        self.plot3d(update=True)

        self.fit_frame.config(cursor="")




    def Pause(self):
        self.stop = True
        self.play_button['image']  =self.playfig
        self.play_button['command']=self.Play

    def Forward(self,mult=1):
        try:
            dt = self.fit_options['dt'].get()
        except:
            dt = ''

        if not self.isfloat(dt) or dt == '':
            return



        if self.ctrl: mult/= 5

        if self.plot_type.get() in [1,2]:
            dt = float(dt)/1e3
            self.plt_time   += dt*mult
        if self.plot_type.get() in [0]:
            dr = float(self.fit_options['dr'].get())
            self.plt_radius += dr*mult

        self.updateMainSlider()
        self.plot_step()

    def Backward(self):
        self.Forward(-1)

    def Play(self):

        if self.m2g is None:
            return

        #animated plot
        self.stop = False
        self.play_button['image']=self.pausefig
        self.play_button['command']=self.Pause
        try:
            dt = float(self.fit_options['dt'].get())/1e3
        except:
            print('Invalid time step value!')
            dt =.01

        try:
            dr = float(self.fit_options['dr'].get())
        except:
            print('Invalid radial step value!')
            dr =.05


        while True:


            if self.plot_type.get() in [0]:
                self.plt_radius += dr

            if self.plot_type.get() in [1,2]:
                self.plt_time   += dt

            if not (self.m2g.t_min <= self.plt_time <= self.m2g.t_max):
                self.stop = True
            if not (self.options['rho_min'] <= self.plt_radius <= self.options['rho_max']):
                self.stop = True

            self.fit_frame.after(1,self.plot_step)
            time.sleep(1e-3)
            try:
                self.canvasMPL.get_tk_widget().update()
            except:
                return

            self.updateMainSlider()

            if self.stop:
                break


        self.Pause()

    def PreparePloting(self):
        #set the limits and title of the plots
        if hasattr(self.parent,'BRIEF'):
            self.title = self.ax_main.set_title(self.parent.BRIEF)

        if self.m2g is None:
            return

        minlim,maxlim = 0,1
        if self.plot_type.get() in [0,1] and self.options['data_loaded']:
            valid = ~self.m2g.Yerr.mask&(self.m2g.Y.data > self.m2g.Yerr.data)
            minlim, maxlim = mquantiles(self.m2g.Y[valid],[.001,.995])
        if self.plot_type.get() in [2] and self.options['data_loaded'] and self.m2g.prepared:
            minlim, maxlim = mquantiles(self.m2g.K[self.m2g.g_r<.8],[.02,.98])
            maxlim*=2
        elif self.plot_type.get() in [2]:
            minlim, maxlim = 0,10 #just guess of the range



        minlim = min(0, minlim)
        if minlim != 0:
            minlim-= (maxlim-minlim)*.1
        maxlim+= (maxlim-minlim)*.2
        self.ax_main.set_ylim(minlim,maxlim)

    def updateMainSlider(self):

        if self.plot_type.get() in [0]:
            self.plt_radius = min(max(self.plt_radius, self.options['rho_min']), self.options['rho_max'])
            val = self.plt_radius
        if self.plot_type.get() in [1,2]:
            self.plt_time = min(max(self.plt_time,self.tbeg), self.tend)
            val = self.plt_time

        
        if self.main_slider.val != val:
            self.main_slider.set_val(val)
        return 
        
        
        self.main_slider.val = val
       
        poly = self.main_slider.poly.get_xy()
        

        poly[2:4,0] = val
        self.main_slider.poly.set_xy(poly)
        self.main_slider.valtext.set_text('%.3f'%val)

    def plot_step(self):
        #single step plotting routine
        if not self.options['data_loaded'] or self.m2g is None:
            return

        t = self.plt_time
        r = self.plt_radius
        try:
            dt = float(self.fit_options['dt'].get())/1e3
        except:
            print('Invalid time step value!')
            dt =.01
        #dt = float(self.fit_options['dt'].get())/1e3
        dr = float(self.fit_options['dr'].get())
        plot_type = self.plot_type.get()
        kinprof = self.parent.kin_prof


        if plot_type in [0]:
            self.time_text.set_text('rho: %.3f'%r)
            self.select = abs(self.plot_rho - r) <= abs(dr)/2
            X = self.plot_tvec


        if plot_type in [1,2]:
            self.time_text.set_text('time: %.4fs'%t)
            self.select = abs(self.plot_tvec - t) <= abs(dt)/2
            X = self.plot_rho


        for idiag, diag in enumerate(self.diags):
            dind = self.select&(self.ind_diag==idiag)&(self.m2g.Yerr.data>0 )

            if any(dind) and plot_type in [0,1]:
                self.replot_plot[idiag].set_visible(True)
                self.plotline[idiag].set_visible(True)
                for c in self.caplines[idiag]:
                    c.set_visible(True)
                self.barlinecols[idiag][0].set_visible(True)
                x = X[dind]
                y = self.m2g.Y[dind]

                yerr = self.m2g.Yerr.data[dind]
                yerr[self.m2g.Yerr.mask[dind]] = np.infty

                ry = self.m2g.retro_f[dind]
                self.replot_plot[idiag].set_data(x,ry)


                # Replot the data first
                self.plotline[idiag].set_data(x,y)

                # Find the ending points of the errorbars
                error_positions = (x,y-yerr), (x,y+yerr)

                # Update the caplines
                for j,pos in enumerate(error_positions):
                    self.caplines[idiag][j].set_data(pos)

                #Update the error bars
                self.barlinecols[idiag][0].set_segments(list(zip(list(zip(x,y-yerr)), list(zip(x,y+yerr)))))
            else:
                self.replot_plot[idiag].set_visible(False)
                self.plotline[idiag].set_visible(False)
                for c in self.caplines[idiag]:
                    c.set_visible(False)
                self.barlinecols[idiag][0].set_visible(False)

            #plot fit of teh data with uncertainty
            if self.options['fitted'] and hasattr(self.m2g,'g_t'):
                if plot_type == 0: #time slice
                    y,x = self.m2g.g_t[:,0], self.m2g.g_r[0]
                    p = r
                    profiles = self.m2g.g.T, self.m2g.g_d.T, self.m2g.g_u.T

                if plot_type == 1:#radial slice
                    profiles = self.m2g.g, self.m2g.g_d, self.m2g.g_u
                    x,y = self.m2g.g_t[:,0], self.m2g.g_r[0]
                    p = t

                if plot_type == 2:#radial slice of the gradient/rho
                    profiles = self.m2g.K, self.m2g.Kerr_d, self.m2g.Kerr_u
                    x,y = self.m2g.g_t[:,0], self.m2g.g_r[0]
                    p = t

                prof = []
                for d in profiles:
                    if self.m2g.g_t.shape[0] == 1:
                        prof.append(d[0])
                    else:
                        prof.append(interp1d(x, d,
                                axis=0, copy=False, assume_sorted=True)(np.clip(p, x[0],x[-1])))

                self.fit_plot.set_data(y,prof[0])
                self.fit_confidence = update_fill_between(self.fit_confidence,y,prof[1], prof[2],-np.infty, np.infty )


        #show discontinuties in time
        for d in self.core_discontinuties:
            d.set_visible(r < .3 and plot_type == 0)
        for d in self.edge_discontinuties:
            d.set_visible(r > .7 and plot_type == 0)

        #MHD modes
        if self.mhd_modes is not None:
            for name, rho_loc in self.mhd_modes['modes'].items():
                loc = np.nan
                if plot_type in [1,2] and kinprof in ['omega','Ti']:
                    it = np.argmin(abs(self.mhd_modes['tvec']-t))
                    loc = rho_loc[it]

                if np.isfinite(loc):
                    self.mhd_locations[name].set_data([loc,loc], [0, 1])
                    self.mhd_labels[name].set_x(loc)
                    self.mhd_locations[name].set_visible(True)
                    self.mhd_labels[name].set_visible(True)
                else:
                    self.mhd_locations[name].set_visible(False)
                    self.mhd_labels[name].set_visible(False)



            #self.mhd_locations = [self.ax_main.axvline(np.nan, ls='-',lw=.5,c='k',visible=False) for mode in self.mhd_modes]
            #self.mhd_labels = [self.ax_main.text(0, np.nan,  mode) for mode in self.mhd_modes]

            #for txt in

            #set_x
            #axvline1.set_data([event.xdata, event.xdata], [0, 1])
            #axvline2.set_data([event.xdata, event.xdata], [0, 1])




        #show also zipfit profiles
        show_splines = self.show_splines.get() == 1

        if show_splines and kinprof in self.splines:

            splines = self.splines
            if not 'time' in splines:
                splines = splines[kinprof]


            y = splines['time'].values
            x = splines['rho'].values
            z = splines[kinprof].values
            ze = z*0
            #embed()

            if kinprof + '_err' in splines:
                ze = splines[kinprof+'_err'].values

            y0 = t

            if plot_type == 0: #temporal profiles
                z,ze,x = z.T,ze.T, x.T
                y,x = x,y
                y0 = r

            if np.ndim(y) == 2:
                y = y.mean(1)

            i = np.argmin(np.abs(y-y0))

            if np.ndim(x) == 2:
                x = x[i]

            z = z[i]
            ze = ze[i]


            if plot_type in [2]:
                #BUG!!!!
                a0 = 0.6
                R0 = 1.7
                z_ = (z[1:]+z[:-1])/2
                x_ = (x[1:]+x[:-1])/2
                z = -(np.diff(z)/np.diff(x*a0)*R0/x_)[z_!=0]/z_[z_!=0]
                ze = 0
                x = x_[z_!=0]

            self.spline_mean.set_data(x,z)
            self.spline_min.set_data(x,z-ze)
            self.spline_max.set_data(x,z+ze)



        self.spline_mean.set_visible(show_splines)
        self.spline_min.set_visible(show_splines)
        self.spline_max.set_visible(show_splines)

        self.fig.canvas.draw_idle()

    def plot3d(self, update=False):

        if not self.options['fitted']:
            return

        if plt.fignum_exists('3D plot'):
            try:
                ax = self.fig_3d.gca()
                ax.collections.remove(self.wframe)
            except:
                return
        elif not update:
            self.fig_3d=plt.figure('3D plot')
            ax=p3.Axes3D(self.fig_3d)
            ax.set_xlabel(self.xlab,fontsize=self.fsize3d)
            ax.set_ylabel('Time [s]',fontsize=self.fsize3d)
        else:
            return

        ax.set_zlabel(self.ylab,fontsize=self.fsize3d)
        self.wframe = ax.plot_wireframe(self.m2g.g_r,self.m2g.g_t,self.m2g.g,linewidth=.3,
                          rstride=self.rstride,cstride=self.cstride)

        self.fig_3d.show()


    def MouseInteraction(self,event):
        if self.picked: #legend_pick was called first
            self.picked = False
            return

        if event.button == 1 and self.ctrl:
            self.delete_channel(event)
        elif event.button == 1:
            self.delete_point(event)
        elif event.button == 2:
            self.calculate()
        elif event.button == 3 and self.ctrl:
            self.undelete_channel(event)
        elif event.button == 3:
            self.undelete_point(event)


    def legend_pick(self,event):

        if not event.mouseevent.dblclick:
            return

        if event.mouseevent.button == 1:
            undelete = False

        elif event.mouseevent.button == 3:
            undelete = True
        else:
            return

        if not event.artist in self.leg_diag_ind:
            return

        i_diag = self.leg_diag_ind[event.artist]

        ind = np.in1d(self.ind_diag,i_diag)

        self.m2g.Yerr.mask[ind]=not undelete

        self.plot_step()
        self.m2g.corrected = False  #force the upgrade
        self.picked=True



    def WheelInteraction(self,event):
        self.Forward(int(event.step))

    def delete_channel(self,event):
        self.delete_point(event,'channel')

    def undelete_channel(self,event):
        self.delete_point(event,'channel',True)

    def undelete_point(self,event):
        self.delete_point(event,'point',True)

    def delete_point(self,event,what='point', undelete=False):
        if not event.dblclick:
            return

        self.delete_points(event,event.xdata, event.ydata,
                           what=what,undelete=undelete)


    def delete_points(self,event,xc,yc,what='point', undelete=False):
        #delete point closest to xc,yc or in the rectangle decribed by xc,yc
        # what - point, channel, diagnostic

        if self.ax_main != event.inaxes or not self.options['data_loaded']:
            return
        if undelete:
            affected = self.select& self.m2g.Yerr.mask
        else:
            affected = self.select&~self.m2g.Yerr.mask

        if not any(affected):
            return



        if self.plot_type.get() == 1:
            x = self.plot_rho[affected]
        elif self.plot_type.get() == 0:
            x = self.plot_tvec[affected]
        else:
            return

        y = self.m2g.Y[affected]
        if np.size(xc) == 1:
            #get range within the plot
            sx = np.ptp(self.ax_main.get_xlim())
            sy = np.ptp(self.ax_main.get_ylim())

            dist = np.hypot((x-xc)/sx,(y-yc)/sy)
            selected = np.argmin(dist)
        else:
            selected = (x>=min(xc))&(x<=max(xc))&(y>=min(yc))&(y<=max(yc))
            if not any(selected):
                return



        i_ind = np.where(affected)[0]
        ind = i_ind[selected]

        action = 'recovered 'if undelete else 'deleted'

        if what == 'channel':
            ch = np.unique(self.channel[ind])
            ind = np.in1d(self.channel,ch)
            #try:
            print('Channels '+','.join([str(self.name_channels[i]) for i in ch])+' were '+action)
            #except:
                #embed()
        elif what == 'diagnostic':
            i_diag = self.ind_diag[ind]
            ind = np.in1d(self.ind_diag,i_diag)
            print('diagnostic %s was '%i_diag+action)

        elif what == 'point':
            pass
        else:
            print('Removing of "%s" is not supported'%(str(what)))

        self.m2g.Yerr.mask[ind]=not undelete

        self.plot_step()
        self.m2g.corrected = False  #force the upgrade





    def on_key(self,event):
        if 'control' == event.key and hasattr(self,'RS_delete'):
            self.ctrl=True
            #old matplolib
            if hasattr(self.RS_delete,'_eventpress'):
                if self.RS_delete._eventpress is not None:
                    self.RS_delete._eventpress.key=None
                if self.RS_undelete._eventpress is not None:
                    self.RS_undelete._eventpress.key=None
            else:#new matplolib
                if self.RS_delete.eventpress is not None:
                    self.RS_delete.eventpress.key=None
                if self.RS_undelete.eventpress is not None:
                    self.RS_undelete.eventpress.key=None

        if 'shift' == event.key:
            self.shift=True

        if 'left' == event.key:
            self.Backward()

        if 'right' == event.key:
            self.Forward()

        if 'g' == event.key:
            self.grid = not self.grid
            self.ax_main.grid(self.grid)
            self.fig.canvas.draw_idle()

        if 'l' == event.key:

            self.logy = not self.logy
            if self.logy:
                if self.ax_main.get_ylim()[0] <= 0:
                    self.ax_main.set_ylim(1,None)
                self.ax_main.set_yscale('log')
            else:
                self.ax_main.set_yscale('linear')
            self.fig.canvas.draw_idle()


        if ' ' == event.key:
            if self.stop:
                self.Play()
            else:
                self.Pause()


    def off_key(self,event):
        if event.key in ('ctrl+control','control') :
            self.ctrl=False
        if 'shift' == event.key:
            self.shift=False

